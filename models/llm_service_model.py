import grpc
import logging
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoConfig, pipeline
import torch
from proto import model_service_pb2, model_service_pb2_grpc

logger = logging.getLogger(__name__)

class LLMServicer(model_service_pb2_grpc.MediaServiceServicer):
    def __init__(self, model_name="microsoft/DialoGPT-medium"):
        """
        初始化 LLM 服務
        
        Args:
            model_name: Hugging Face 模型名稱
            常用選項:
            - "microsoft/DialoGPT-medium" (對話模型)
            - "gpt2" (通用文本生成)
            - "facebook/blenderbot-400M-distill" (聊天機器人)
            - "google/flan-t5-base" (指令跟隨模型)
        """
        self.model_name = model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        print(f"🤖 正在載入 LLM 模型: {model_name}")
        print(f"🔧 使用設備: {self.device}")
        
        try:
            # 載入 tokenizer 和模型配置
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            config = AutoConfig.from_pretrained(model_name)

            # 根據模型類型選擇對應的 AutoModel 和 pipeline 任務
            if getattr(config, "is_encoder_decoder", False):
                ModelClass = AutoModelForSeq2SeqLM
                pipeline_task = "text2text-generation"
                self.model_type = "seq2seq"
                print("🔍 檢測到 Seq2Seq 模型架構")
            else:
                ModelClass = AutoModelForCausalLM
                pipeline_task = "text-generation"
                self.model_type = "causal"
                print("🔍 檢測到 Causal LM 模型架構")

            self.model = ModelClass.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32,
                device_map="auto" if self.device.type == "cuda" else None
            )
            
            # 設置 padding token（如果沒有的話）
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # 創建 pipeline
            self.generator = pipeline(
                pipeline_task,
                model=self.model,
                tokenizer=self.tokenizer,
                device=0 if self.device.type == "cuda" else -1,
                torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32
            )
            
            print("✅ LLM 模型載入成功")
            
        except Exception as e:
            print(f"❌ LLM 模型載入失敗: {e}")
            raise

    def GenerateText(self, request, context):
        """處理文本生成請求"""
        try:
            logger.info(f"收到文本生成請求: {request.prompt[:50]}...")
            
            # 生成參數
            generation_config = {
                "max_new_tokens": request.max_tokens if request.max_tokens > 0 else 100,
                "temperature": request.temperature if request.temperature > 0 else 0.7,
                "top_p": request.top_p if request.top_p > 0 else 0.9,
                "do_sample": True,
                "pad_token_id": self.tokenizer.eos_token_id,
                "return_full_text": False
            }
            
            logger.info(f"生成參數: {generation_config}")
            
            # 執行文本生成
            outputs = self.generator(
                request.prompt,
                **generation_config
            )
            
            # 提取生成的文本
            generated_text = outputs[0]["generated_text"]
            
            logger.info(f"生成完成，長度: {len(generated_text)} 字符")
            
            return model_service_pb2.TextGenerationResponse(
                generated_text=generated_text,
                success=True
            )
            
        except Exception as e:
            error_msg = f"文本生成失敗: {str(e)}"
            logger.error(error_msg)
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(error_msg)
            return model_service_pb2.TextGenerationResponse(
                generated_text="",
                success=False
            )

    def ChatCompletion(self, request, context):
        """處理對話完成請求"""
        try:
            logger.info("收到對話完成請求")
            
            # 構建對話歷史為單一字符串
            conversation_parts = []
            
            for message in request.messages:
                if message.role == "system":
                    conversation_parts.append(f"System: {message.content}")
                elif message.role == "user":
                    conversation_parts.append(f"Human: {message.content}")
                elif message.role == "assistant":
                    conversation_parts.append(f"Assistant: {message.content}")
            
            # 添加 Assistant 提示讓模型回應
            conversation_parts.append("Assistant:")
            conversation = "\n".join(conversation_parts)
            
            logger.info(f"對話歷史長度: {len(conversation)} 字符")
            
            # 生成回應
            generation_config = {
                "max_new_tokens": request.max_tokens if request.max_tokens > 0 else 150,
                "temperature": request.temperature if request.temperature > 0 else 0.7,
                "top_p": 0.9,
                "do_sample": True,
                "pad_token_id": self.tokenizer.eos_token_id,
                "return_full_text": False,
                "num_return_sequences": 1
            }
            
            # 使用 pipeline 生成回應
            outputs = self.generator(
                conversation,
                **generation_config
            )
            
            response_text = outputs[0]["generated_text"].strip()
            
            # 清理回應（移除可能的停止序列）
            stop_sequences = ["\nHuman:", "\nSystem:", "Human:", "System:", "\nuser:", "\nUser:"]
            for stop_seq in stop_sequences:
                if stop_seq in response_text:
                    response_text = response_text.split(stop_seq)[0].strip()
            
            logger.info(f"對話回應生成完成")
            
            return model_service_pb2.ChatCompletionResponse(
                response=response_text,
                success=True
            )
            
        except Exception as e:
            error_msg = f"對話完成失敗: {str(e)}"
            logger.error(error_msg)
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(error_msg)
            return model_service_pb2.ChatCompletionResponse(
                response="",
                success=False
            )