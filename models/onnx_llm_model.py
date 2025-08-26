import os
import logging
from pathlib import Path
import torch
import numpy as np
from transformers import AutoTokenizer, AutoConfig
from optimum.onnxruntime import ORTModelForCausalLM, ORTModelForSeq2SeqLM
import onnxruntime as ort
import sys
import urllib.request
import zipfile
import shutil
from tqdm import tqdm

# Add the project root to sys.path to enable importing proto
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from proto import model_service_pb2

logger = logging.getLogger(__name__)

# Define where to save the ONNX model
ONNX_MODEL_CACHE_PATH = Path(__file__).parent / "onnx_llm_model"

class TqdmUpTo(tqdm):
    """Provides `update_to(block_num, block_size, total_size)`."""
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)

def _download_and_unzip_model(url: str, model_path: Path):
    """
    Downloads and unzips the model from a given URL.
    """
    zip_path = model_path.parent / "temp_model.zip"
    
    if not zip_path.exists():
        logger.info(f"模型不存在於 {model_path}，且暫存壓縮檔不存在。開始從網路下載...")
        print(f"⬇️ 正在從 {url} 下載模型...")
        try:
            with TqdmUpTo(unit='B', unit_scale=True, unit_divisor=1024, miniters=1,
                          desc=zip_path.name) as t:
                urllib.request.urlretrieve(url, filename=zip_path, reporthook=t.update_to)
            logger.info("下載完成。")
        except Exception as e:
            logger.error(f"下載模型失敗: {e}")
            if zip_path.exists():
                os.remove(zip_path) # Clean up partial download
            raise RuntimeError(f"無法下載模型檔案。請檢查網路連線或下載連結是否正確。錯誤: {e}")
    else:
        logger.info(f"發現已存在的暫存壓縮檔: {zip_path}。將直接進行解壓縮。")

    logger.info("開始解壓縮模型...")
    print(f"📦 正在解壓縮檔案到 {model_path.parent}...")
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(model_path.parent)
        
        # Verify that the target directory was created
        if not model_path.exists():
            # Sometimes zip files have a top-level directory. Let's find it.
            extracted_folders = [d for d in zip_path.parent.iterdir() if d.is_dir() and d.name != model_path.name]
            # This is a simple heuristic, assuming the zip contains one folder
            if len(extracted_folders) == 1 and extracted_folders[0].name.startswith("Qwen"):
                 # Rename the extracted folder to the target folder name
                 os.rename(extracted_folders[0], model_path)
                 logger.info(f"成功將解壓縮的資料夾更名為 {model_path.name}")
            else:
                 raise FileNotFoundError(f"解壓縮後，預期的模型資料夾 {model_path} 不存在。")

        logger.info("解壓縮完成。")

    except Exception as e:
        logger.error(f"解壓縮模型失敗: {e}")
        # Clean up potentially corrupted extracted files
        if model_path.exists():
            shutil.rmtree(model_path) # Use shutil.rmtree for directories
        raise RuntimeError(f"無法解壓縮模型檔案。請檢查壓縮檔是否完整或磁碟空間是否足夠。錯誤: {e}")
    finally:
        # Clean up the downloaded zip file
        if zip_path.exists():
            os.remove(zip_path)
            logger.info(f"已刪除暫存檔: {zip_path}")

class ONNXLLMModel:
    def __init__(self, model_name: str = "Qwen/Qwen1.5-1.8B-Chat"):
        self.model_name = model_name
        self.model_repo_id = "jiruii/Qwen1.5b" # For reference
        self.model_download_url = "https://huggingface.co/jiruii/Qwen1.5b/resolve/main/Qwen_Qwen1.5-1.8B-Chat.zip?download=true"
        
        # The actual local directory name for the model
        self.cache_path = ONNX_MODEL_CACHE_PATH / model_name.replace("/", "_")
        
        self.tokenizer = None
        self.model = None
        self.model_type = None

        self._load_or_convert_model()

    def _load_or_convert_model(self):
        # Step 1: Check if model exists, if not, download and unzip
        if not self.cache_path.exists():
            _download_and_unzip_model(self.model_download_url, self.cache_path)

        print(f"🤖 正在從本地路徑載入 ONNX LLM 模型: {self.cache_path}")
        
        # Step 2: Determine execution providers
        preferred_providers = []
        if os.name == 'nt':
            preferred_providers.append("DmlExecutionProvider")
        if torch.cuda.is_available():
            preferred_providers.append("CUDAExecutionProvider")
        preferred_providers.append("CPUExecutionProvider")

        available_providers = ort.get_available_providers()
        providers = [p for p in preferred_providers if p in available_providers]
        
        if not providers:
            raise RuntimeError("沒有可用的 ONNX Runtime 執行提供者!")
        
        selected_provider = providers[0]
        print(f"🔧 使用 ONNX Runtime 執行提供者: {selected_provider}")

        # Step 3: Load the model from the local directory
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.cache_path)
            config = AutoConfig.from_pretrained(self.cache_path)
            
            ModelClass = ORTModelForSeq2SeqLM if getattr(config, "is_encoder_decoder", False) else ORTModelForCausalLM
            self.model_type = "seq2seq" if getattr(config, "is_encoder_decoder", False) else "causal"

            self.model = ModelClass.from_pretrained(
                self.cache_path, 
                provider=selected_provider
            )
            print("✅ 成功從快取載入 ONNX 模型。")

        except Exception as e:
            logger.error(f"從 {self.cache_path} 載入模型失敗: {e}")
            raise RuntimeError(f"無法從本地路徑 {self.cache_path} 載入模型，即使在嘗試下載後。請檢查檔案是否完整。")

        # Set padding token if not already set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def GenerateText(self, request, context):
        try:
            logger.info(f"收到文本生成請求 (ONNX): {request.prompt[:50]}...")
            
            inputs = self.tokenizer(
                request.prompt, 
                return_tensors="pt", 
                padding=True, 
                truncation=True
            )
            
            generation_config = {
                "max_new_tokens": request.max_tokens if request.max_tokens > 0 else 100,
                "temperature": request.temperature if request.temperature > 0 else 0.7,
                "top_p": request.top_p if request.top_p > 0 else 0.9,
                "do_sample": True,
                "pad_token_id": self.tokenizer.eos_token_id
            }
            
            outputs = self.model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                **generation_config
            )
            
            input_length = inputs["input_ids"].shape[1]
            generated_text = self.tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True)
            
            logger.info(f"生成完成 (ONNX)，長度: {len(generated_text)} 字符")
            
            return model_service_pb2.TextGenerationResponse(
                generated_text=generated_text,
                success=True
            )
            
        except Exception as e:
            error_msg = f"ONNX 文本生成失敗: {str(e)}"
            logger.error(error_msg)
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(error_msg)
            return model_service_pb2.TextGenerationResponse(
                generated_text="",
                success=False
            )

    def ChatCompletion(self, request, context):
        try:
            logger.info("收到對話完成請求 (ONNX)")
            
            chat = []
            for message in request.messages:
                chat.append({"role": message.role, "content": message.content})

            prompt = self.tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
            
            inputs = self.tokenizer(
                prompt, 
                return_tensors="pt", 
                padding=True, 
                truncation=True
            )
            
            generation_config = {
                "max_new_tokens": request.max_tokens if request.max_tokens > 0 else 150,
                "temperature": request.temperature if request.temperature > 0 else 0.7,
                "top_p": 0.9,
                "do_sample": True,
                "pad_token_id": self.tokenizer.eos_token_id,
                "num_return_sequences": 1
            }
            
            outputs = self.model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                **generation_config
            )
            
            input_length = inputs["input_ids"].shape[1]
            response_text = self.tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True).strip()
            
            logger.info(f"對話回應生成完成 (ONNX)")
            
            return model_service_pb2.ChatCompletionResponse(
                response=response_text,
                success=True
            )
            
        except Exception as e:
            error_msg = f"ONNX 對話完成失敗: {str(e)}"
            logger.error(error_msg)
            import grpc
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(error_msg)
            return model_service_pb2.ChatCompletionResponse(
                response="",
                success=False,
                error_message=error_msg
            )