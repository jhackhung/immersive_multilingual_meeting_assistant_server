import asyncio
from collections.abc import AsyncIterator
import numpy as np
import logging
import time
import uuid
import os
import shutil
import zipfile
from typing import Dict, Iterable, Tuple
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

# 改用直接的 ONNX Runtime
import onnxruntime as ort
from transformers import AutoProcessor, WhisperTokenizer, WhisperFeatureExtractor

# 從新生成的 proto 文件導入 gRPC 服務定義
from proto import model_service_pb2
from proto import model_service_pb2_grpc

try:
    from huggingface_hub import hf_hub_download, HfApi
    HUGGINGFACE_AVAILABLE = True
except ImportError:
    HUGGINGFACE_AVAILABLE = False
    logging.warning("huggingface_hub 未安裝，無法自動下載模型。請運行 'pip install huggingface_hub'。")

logger = logging.getLogger(__name__)

# 音訊緩衝區的最大大小（例如 30 秒的 16kHz PCM 16位元音訊）
MAX_AUDIO_BUFFER_SIZE_BYTES = 16000 * 2 * 30 
# 每次處理的音訊塊大小 (例如 5 秒)
PROCESS_CHUNK_SIZE_SAMPLES = 16000 * 5 

def get_project_root():
    """獲取項目根目錄"""
    current_file = os.path.abspath(__file__)
    apis_dir = os.path.dirname(current_file)
    project_root = os.path.dirname(apis_dir)
    return project_root

class STTService:
    """
    STT 服務實現 - 使用 ONNX Runtime 和 gRPC 雙向串流。
    支援自動檢測音訊格式和模型自動下載。
    """
    
    def __init__(self, model_path: str = None, use_qualcomm_model: bool = False):
        # 如果沒有指定模型路徑，使用默認的本地路徑
        if model_path is None:
            project_root = get_project_root()
            self.model_path = os.path.join(project_root, "models", "onnx_whisper_models")
        else:
            self.model_path = model_path
            
        self.use_qualcomm_model = use_qualcomm_model
        self.encoder_session = None
        self.decoder_session = None
        self.processor = None
        self.tokenizer = None
        self.feature_extractor = None
        self.is_initialized = False
        
        self.executor = ThreadPoolExecutor(max_workers=2)
        
        logger.info(f"STT 服務初始化 (ONNX Runtime, 雙向串流)")
        logger.info(f"模型路徑: {self.model_path}")
        logger.info(f"使用高通優化模型: {self.use_qualcomm_model}")
    
    def _check_and_download_qualcomm_models(self) -> Dict[str, str]:
        """
        檢查並下載高通優化的 Whisper ONNX 模型
        """
        if not HUGGINGFACE_AVAILABLE:
            logger.error("huggingface_hub 套件未安裝。無法下載模型。")
            return {}

        try:
            encoder_path = os.path.join(self.model_path, "qualcomm-encoder.onnx")
            decoder_path = os.path.join(self.model_path, "qualcomm-decoder.onnx")

            if os.path.exists(encoder_path) and os.path.exists(decoder_path):
                logger.info(f"高通 Whisper ONNX 模型已存在於: {self.model_path}")
                return {
                    "encoder": encoder_path,
                    "decoder": decoder_path,
                }

            logger.info("本地模型不存在，正在從 Hugging Face 下載高通 Whisper 模型...")
            os.makedirs(self.model_path, exist_ok=True)
            repo_id = "qualcomm/Whisper-Large-V3-Turbo"
            encoder_zip = "precompiled/qualcomm-snapdragon-x-elite/Whisper-Large-V3-Turbo_HfWhisperEncoder.onnx.zip"
            decoder_zip = "precompiled/qualcomm-snapdragon-x-elite/Whisper-Large-V3-Turbo_HfWhisperDecoder.onnx.zip"

            # 下載並解壓 Encoder
            logger.info("正在下載 Encoder zip 文件...")
            encoder_zip_path = hf_hub_download(repo_id=repo_id, filename=encoder_zip)
            with zipfile.ZipFile(encoder_zip_path, 'r') as zip_ref:
                zip_ref.extract("model.onnx/model.onnx", path=self.model_path)
            shutil.move(os.path.join(self.model_path, "model.onnx", "model.onnx"), encoder_path)
            shutil.rmtree(os.path.join(self.model_path, "model.onnx"), ignore_errors=True)
            logger.info(f"Encoder 模型已重命名並移動到: {encoder_path}")

            # 下載並解壓 Decoder
            logger.info("正在下載 Decoder zip 文件...")
            decoder_zip_path = hf_hub_download(repo_id=repo_id, filename=decoder_zip)
            with zipfile.ZipFile(decoder_zip_path, 'r') as zip_ref:
                zip_ref.extract("model.onnx/model.onnx", path=self.model_path)
            shutil.move(os.path.join(self.model_path, "model.onnx", "model.onnx"), decoder_path)
            shutil.rmtree(os.path.join(self.model_path, "model.onnx"), ignore_errors=True)
            logger.info(f"Decoder 模型已重命名並移動到: {decoder_path}")

            logger.info(f"高通 Whisper 模型成功下載並提取到: {self.model_path}")
            return {
                "encoder": encoder_path,
                "decoder": decoder_path,
            }

        except Exception as e:
            logger.error(f"下載與提取高通 Whisper 模型時發生錯誤: {e}")
            return {}
    
    def _check_and_download_onnx_community_models(self) -> Dict[str, str]:
        """
        檢查並下載 onnx-community 的 Whisper ONNX 模型
        """
        if not HUGGINGFACE_AVAILABLE:
            logger.error("huggingface_hub 套件未安裝。無法下載模型。")
            return {}

        try:
            encoder_path = os.path.join(self.model_path, "encoder_model.onnx")
            decoder_path = os.path.join(self.model_path, "decoder_model.onnx")
            encoder_data_path = os.path.join(self.model_path, "encoder_model.onnx_data")

            # 檢查所有必要文件是否存在
            if (os.path.exists(encoder_path) and os.path.exists(decoder_path) 
                and os.path.exists(encoder_data_path)):
                logger.info(f"ONNX Community Whisper 模型已存在於: {self.model_path}")
                return {
                    "encoder": encoder_path,
                    "decoder": decoder_path,
                }

            logger.info("本地模型不存在，正在從 Hugging Face 下載 ONNX Community Whisper 模型...")
            os.makedirs(self.model_path, exist_ok=True)
            
            # 使用 onnx-community 的預編譯模型
            repo_id = "onnx-community/whisper-large-v3-turbo"
            
            # 定義需要下載的文件
            files_to_download = [
                ("onnx/encoder_model.onnx", encoder_path),
                ("onnx/encoder_model.onnx_data", encoder_data_path),
                ("onnx/decoder_model.onnx", decoder_path),
            ]

            # 下載所有必要文件
            for remote_filename, local_path in files_to_download:
                try:
                    logger.info(f"正在下載 {remote_filename}...")
                    downloaded_path = hf_hub_download(
                        repo_id=repo_id, 
                        filename=remote_filename,
                        cache_dir=None
                    )
                    # 複製到我們的模型目錄
                    shutil.copy2(downloaded_path, local_path)
                    logger.info(f"文件已複製到: {local_path}")
                except Exception as e:
                    logger.error(f"下載 {remote_filename} 失敗: {e}")
                    # 如果是 .onnx_data 文件下載失敗，可能模型不需要這個文件
                    if not remote_filename.endswith('.onnx_data'):
                        raise e
                    else:
                        logger.warning(f"跳過 {remote_filename}，可能模型不需要此文件")

            logger.info(f"ONNX Community Whisper 模型成功下載並複製到: {self.model_path}")
            return {
                "encoder": encoder_path,
                "decoder": decoder_path,
            }

        except Exception as e:
            logger.error(f"下載 ONNX Community Whisper 模型時發生錯誤: {e}")
            return {}
    
    def _check_model_files(self) -> bool:
        """檢查必要的模型文件是否存在"""
        if self.use_qualcomm_model:
            required_files = [
                "qualcomm-encoder.onnx",
                "qualcomm-decoder.onnx"
            ]
        else:
            required_files = [
                "encoder_model.onnx",
                "decoder_model.onnx",
                "encoder_model.onnx_data"
            ]
        
        for file_name in required_files:
            file_path = os.path.join(self.model_path, file_name)
            if not os.path.exists(file_path):
                logger.warning(f"缺少模型文件: {file_path}")
                return False
                
        logger.info("所有必要的模型文件都存在")
        return True
    
    def _detect_audio_format(self, audio_bytes: bytes, chunk_count: int = 0) -> Tuple[int, int, int]:
        """
        自動檢測音訊格式 (採樣率、聲道數、位元深度)
        返回: (sample_rate, channels, bits_per_sample)
        """
        try:
            # 基於音訊數據大小和內容進行啟發式檢測
            data_length = len(audio_bytes)
            
            # 常見的採樣率候選
            common_sample_rates = [8000, 16000, 22050, 44100, 48000]
            
            # 假設是 16-bit PCM（最常見）
            bits_per_sample = 16
            bytes_per_sample = bits_per_sample // 8
            
            # 檢測聲道數：嘗試單聲道和立體聲
            possible_channels = [1, 2]
            
            # 分析音訊數據的統計特性
            if data_length >= 2:
                # 轉換為 16-bit 整數進行分析
                audio_data = np.frombuffer(audio_bytes, dtype=np.int16)
                
                # 檢測是否為立體聲（左右聲道交替模式）
                if len(audio_data) >= 4:
                    # 檢查奇偶索引的差異程度
                    left_channel = audio_data[::2]
                    right_channel = audio_data[1::2]
                    
                    if len(left_channel) > 0 and len(right_channel) > 0:
                        correlation = np.corrcoef(left_channel[:min(len(left_channel), len(right_channel))], 
                                                right_channel[:min(len(left_channel), len(right_channel))])[0, 1]
                        
                        # 如果左右聲道相關性很低，可能是立體聲
                        if not np.isnan(correlation) and correlation < 0.8:
                            channels = 2
                        else:
                            channels = 1
                    else:
                        channels = 1
                else:
                    channels = 1
                
                # 基於數據長度和時間推測採樣率
                # 假設每個音訊塊代表 100ms 的音訊
                expected_duration_ms = 100  # 假設客戶端每 100ms 發送一次
                expected_samples = data_length // (bytes_per_sample * channels)
                expected_sample_rate = int(expected_samples * 1000 / expected_duration_ms)
                
                # 找到最接近的標準採樣率
                sample_rate = min(common_sample_rates, 
                                key=lambda x: abs(x - expected_sample_rate))
                
                # 合理性檢查
                if sample_rate < 8000:
                    sample_rate = 16000  # 預設值
                elif sample_rate > 48000:
                    sample_rate = 48000  # 上限
                
            else:
                # 如果數據太少，使用預設值
                sample_rate = 16000
                channels = 1
            
            logger.debug(f"自動檢測音訊格式: {sample_rate}Hz, {channels}聲道, {bits_per_sample}bit")
            return sample_rate, channels, bits_per_sample
            
        except Exception as e:
            logger.warning(f"音訊格式檢測失敗，使用預設值: {e}")
            return 16000, 1, 16  # 預設值
    
    def _simple_resample(self, audio_data: np.ndarray, orig_sr: int, target_sr: int = 16000) -> np.ndarray:
        """
        簡單的重採樣方法（使用線性插值）
        """
        if orig_sr == target_sr:
            return audio_data
        
        # 計算重採樣比例
        ratio = target_sr / orig_sr
        
        # 使用 numpy 的線性插值進行重採樣
        original_length = len(audio_data)
        new_length = int(original_length * ratio)
        
        if new_length == 0:
            return np.array([], dtype=audio_data.dtype)
        
        # 創建新的索引
        old_indices = np.linspace(0, original_length - 1, original_length)
        new_indices = np.linspace(0, original_length - 1, new_length)
        
        # 線性插值
        resampled = np.interp(new_indices, old_indices, audio_data)
        
        return resampled.astype(audio_data.dtype)
    
    def _process_audio_chunk(self, audio_bytes: bytes, sample_rate: int, channels: int, bits_per_sample: int = 16) -> np.ndarray:
        """
        處理音訊塊：重採樣和格式轉換
        """
        try:
            # 根據位元深度確定數據類型
            if bits_per_sample == 16:
                audio_data = np.frombuffer(audio_bytes, dtype=np.int16)
                # 轉換為 float32 範圍 [-1, 1]
                audio_data = audio_data.astype(np.float32) / 32768.0
            elif bits_per_sample == 8:
                audio_data = np.frombuffer(audio_bytes, dtype=np.uint8)
                # 轉換為 float32 範圍 [-1, 1]
                audio_data = (audio_data.astype(np.float32) - 128) / 128.0
            elif bits_per_sample == 32:
                audio_data = np.frombuffer(audio_bytes, dtype=np.int32)
                # 轉換為 float32 範圍 [-1, 1]
                audio_data = audio_data.astype(np.float32) / 2147483648.0
            else:
                # 預設處理為 16-bit
                audio_data = np.frombuffer(audio_bytes, dtype=np.int16)
                audio_data = audio_data.astype(np.float32) / 32768.0
            
            # 處理多聲道轉單聲道
            if channels > 1:
                # 重新形狀為 (samples, channels)
                audio_data = audio_data.reshape(-1, channels)
                # 取平均值轉為單聲道
                audio_data = audio_data.mean(axis=1)
            
            # 重採樣到 16kHz（如果需要）
            if sample_rate != 16000:
                audio_data = self._simple_resample(audio_data, sample_rate, 16000)
                logger.debug(f"音訊重採樣: {sample_rate} Hz -> 16000 Hz")
            
            return audio_data
            
        except Exception as e:
            logger.error(f"音訊處理失敗: {e}")
            return np.array([], dtype=np.float32)
    
    def initialize(self) -> bool:
        """初始化 ONNX 模型和處理器"""
        if self.is_initialized:
            logger.info("模型已經初始化，跳過重新初始化。")
            return True
            
        try:
            # 檢查模型文件是否存在，如果不存在則下載
            if not self._check_model_files():
                logger.info("模型文件不存在，開始下載...")
                
                if self.use_qualcomm_model:
                    model_paths = self._check_and_download_qualcomm_models()
                else:
                    model_paths = self._check_and_download_onnx_community_models()
                
                if not model_paths:
                    logger.error("模型下載失敗")
                    return False
                
                # 再次檢查文件是否存在
                if not self._check_model_files():
                    logger.error("模型文件下載後檢查失敗")
                    return False
            
            logger.info(f"載入本地 ONNX Whisper 模型: {self.model_path}...")
            
            # 根據模型類型設置文件路徑
            if self.use_qualcomm_model:
                encoder_path = os.path.join(self.model_path, "qualcomm-encoder.onnx")
                decoder_path = os.path.join(self.model_path, "qualcomm-decoder.onnx")
            else:
                encoder_path = os.path.join(self.model_path, "encoder_model.onnx")
                decoder_path = os.path.join(self.model_path, "decoder_model.onnx")
            
            # 創建 ONNX Runtime 會話
            providers = ['CPUExecutionProvider']  # 可以根據需要添加 GPU 提供者
            
            logger.info("載入 Encoder 模型...")
            self.encoder_session = ort.InferenceSession(encoder_path, providers=providers)
            
            logger.info("載入 Decoder 模型...")
            self.decoder_session = ort.InferenceSession(decoder_path, providers=providers)
            
            # 載入處理器組件
            processor_model_id = "openai/whisper-large-v3-turbo"
            logger.info(f"載入處理器組件: {processor_model_id}")
            
            self.processor = AutoProcessor.from_pretrained(processor_model_id)
            self.tokenizer = WhisperTokenizer.from_pretrained(processor_model_id)
            self.feature_extractor = WhisperFeatureExtractor.from_pretrained(processor_model_id)
            
            self.is_initialized = True
            logger.info("ONNX 模型載入成功。")
            return True
            
        except Exception as e:
            logger.error(f"ONNX 模型載入失敗: {e}")
            logger.error(f"詳細錯誤: ", exc_info=True)
            self.is_initialized = False
            return False
    
    def _transcribe_audio_chunk(self, audio_array: np.ndarray, language: str) -> Dict:
        """
        對音訊 numpy 陣列進行轉錄。
        使用手動 ONNX 推理。
        """
        if not self.is_initialized:
            return {"success": False, "error": "模型未初始化"}

        try:
            # 確保音訊陣列是單聲道
            if audio_array.ndim > 1:
                audio_array = audio_array.mean(axis=1)
            
            # 預處理音訊輸入（注意：已經是 16kHz 和 float32 格式）
            input_features = self.feature_extractor(
                audio_array, 
                sampling_rate=16000, 
                return_tensors="np"
            ).input_features
            
            # Encoder 推理
            encoder_outputs = self.encoder_session.run(
                None, 
                {"input_features": input_features}
            )
            encoder_hidden_states = encoder_outputs[0]
            
            # 準備 Decoder 輸入
            # 設定語言和任務 token
            if language and language != "auto":
                language_token = f"<|{language}|>"
            else:
                language_token = "<|zh|>"  # 預設中文
            
            # 建構初始 decoder 輸入
            decoder_input_ids = self.tokenizer.encode(
                f"<|startoftranscript|>{language_token}<|transcribe|>",
                return_tensors="np"
            )
            # 確保 decoder_input_ids 是 int64 類型
            if decoder_input_ids.dtype != np.int64:
                decoder_input_ids = decoder_input_ids.astype(np.int64)
            
            # 簡化的貪婪解碼
            max_length = 448  # Whisper 的最大長度
            generated_tokens = []
            
            for _ in range(max_length):
                # Decoder 推理
                decoder_outputs = self.decoder_session.run(
                    None,
                    {
                        "input_ids": decoder_input_ids,
                        "encoder_hidden_states": encoder_hidden_states
                    }
                )
                
                # 獲取下一個 token
                logits = decoder_outputs[0]
                next_token_id = np.argmax(logits[0, -1, :])
                
                # 檢查是否為結束 token
                if next_token_id == self.tokenizer.eos_token_id:
                    break
                
                generated_tokens.append(next_token_id)
                
                # 更新 decoder 輸入
                decoder_input_ids = np.concatenate([
                    decoder_input_ids,
                    np.array([[next_token_id]])
                ], axis=1)
            
            # 解碼生成的 tokens
            transcription = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
            
            segments_data = []
            if transcription:
                segment_text = transcription.strip()
                segment_duration = len(audio_array) / 16000.0
                
                # 簡化的詞級信息 - 將整個轉錄作為一個詞
                words_info = []
                if segment_text:
                    words_info.append({
                        "word": segment_text,
                        "start_time_sec": 0.0,
                        "end_time_sec": segment_duration,
                        "confidence": 0.9
                    })

                segments_data.append({
                    "text": segment_text,
                    "start_time_sec": 0.0,
                    "end_time_sec": segment_duration,
                    "words": words_info,
                    "speaker_id": "unknown"
                })

            return {
                "success": True,
                "transcribed_text": transcription.strip(),
                "detected_language": language,
                "language_confidence": 1.0,
                "segments": segments_data
            }
        
        except Exception as e:
            logger.error(f"ONNX 轉錄失敗: {e}")
            return {"success": False, "error": f"ONNX 轉錄失敗: {str(e)}"}

    async def StreamingRecognize(self,
                        request_iterator,
                        context) -> AsyncIterator[model_service_pb2.StreamingRecognizeResponse]:
        """
        gRPC 雙向串流接口，用於實時語音識別。
        自動檢測音訊格式，無需客戶端指定。
        """
        session_id = str(uuid.uuid4())
        logger.info(f"接收到 gRPC 雙向串流請求。Session ID: {session_id}")

        # 檢查模型是否已初始化
        if not self.is_initialized:
            yield model_service_pb2.StreamingRecognizeResponse(
                session_id=session_id,
                is_final=True,
                message="模型未初始化。",
                rtf=0.0,
                chunk_sec=0.0,
                server_time_sec=0.0
            )
            return

        # 使用 list 來儲存處理後的音訊 float32 數據
        audio_buffer = []  # 儲存 float32 的音訊樣本
        current_audio_duration = 0.0 # 緩衝區中的音訊總時長 (秒)
        
        # 音訊格式檢測變數
        detected_sample_rate = None
        detected_channels = None
        detected_bits = None
        chunk_count = 0
        
        # 轉錄配置，從請求中讀取
        language = "auto"
        
        total_audio_processed_sec = 0.0 # 追蹤伺服器已處理的音訊總時長

        async for request in request_iterator:  # 移除 async for
            server_process_start_time = time.time() # 記錄處理開始時間

            # 提取請求中的語言設定
            if request.language:
                language = request.language

            # 獲取音訊數據
            audio_chunk_bytes = request.audio.audio_bytes
            if not audio_chunk_bytes:
                # 空的音訊塊通常是客戶端關閉串流前發送的信號
                if request.is_last:
                    logger.info(f"Session ID: {session_id} - 接收到最後一個空音訊塊。")
                    break
                continue

            # 自動檢測音訊格式（只在前幾個塊中進行）
            if chunk_count < 3:  # 前3個塊用於格式檢測
                sample_rate, channels, bits = self._detect_audio_format(audio_chunk_bytes, chunk_count)
                
                if detected_sample_rate is None:
                    detected_sample_rate = sample_rate
                    detected_channels = channels
                    detected_bits = bits
                    logger.info(f"Session ID: {session_id} - 檢測到音訊格式: {detected_sample_rate}Hz, {detected_channels}聲道, {detected_bits}bit")
                else:
                    # 驗證格式一致性
                    if abs(sample_rate - detected_sample_rate) > 1000:  # 允許一些誤差
                        logger.warning(f"檢測到不一致的採樣率: 之前={detected_sample_rate}, 現在={sample_rate}")
                
                chunk_count += 1

            # 使用檢測到的格式處理音訊
            if detected_sample_rate is None:
                # 如果檢測失敗，使用預設值
                detected_sample_rate = 16000
                detected_channels = 1
                detected_bits = 16

            # 處理音訊塊：重採樣和格式轉換
            processed_audio = self._process_audio_chunk(
                audio_chunk_bytes, 
                detected_sample_rate, 
                detected_channels,
                detected_bits
            )
            
            if len(processed_audio) == 0:
                logger.warning("音訊處理後為空，跳過此塊")
                continue
            
            # 將處理後的音訊添加到緩衝區
            audio_buffer.extend(processed_audio)
            chunk_duration = len(processed_audio) / 16000.0  # 已經重採樣到 16kHz
            current_audio_duration += chunk_duration
            total_audio_processed_sec += chunk_duration

            # 保持緩衝區在合理大小，丟棄舊數據
            max_samples = 16000 * 30  # 30 秒的樣本數
            while len(audio_buffer) > max_samples:
                samples_to_remove = min(len(audio_buffer) - max_samples + 16000 * 5, len(audio_buffer))
                if samples_to_remove > 0:
                    del audio_buffer[:samples_to_remove]
                    removed_duration = samples_to_remove / 16000.0
                    current_audio_duration -= removed_duration
                    current_audio_duration = max(0.0, current_audio_duration)

            # 當緩衝區累積足夠的音訊時，執行轉錄
            if current_audio_duration >= 5.0:  # 5 秒
                # 轉換為 numpy 陣列
                audio_array = np.array(audio_buffer, dtype=np.float32)
                
                logger.info(f"Session ID: {session_id} - 緩衝區累積 {current_audio_duration:.2f} 秒音訊，開始轉錄...")
                
                try:
                    # 在線程池中執行 CPU 密集型的轉錄任務
                    loop = asyncio.get_event_loop()
                    result = await loop.run_in_executor(
                        self.executor,
                        self._transcribe_audio_chunk,
                        audio_array,
                        language
                    )
                except Exception as e:
                    logger.error(f"異步轉錄執行失敗: {e}")
                    result = {"success": False, "error": f"異步轉錄執行失敗: {str(e)}"}
                
                server_process_end_time = time.time()
                server_time_taken = server_process_end_time - server_process_start_time
                rtf = server_time_taken / current_audio_duration if current_audio_duration > 0 else 0.0

                if result["success"]:
                    segments = []
                    for seg_data in result["segments"]:
                        words = []
                        for word_data in seg_data.get("words", []):
                            words.append(model_service_pb2.WordInfo(
                                word=word_data["word"],
                                start_time_sec=word_data["start_time_sec"],
                                end_time_sec=word_data["end_time_sec"],
                                confidence=word_data.get("confidence", 0.0)
                            ))
                        segments.append(model_service_pb2.Segment(
                            text=seg_data["text"],
                            start_time_sec=seg_data["start_time_sec"],
                            end_time_sec=seg_data["end_time_sec"],
                            words=words,
                            speaker_id=seg_data.get("speaker_id", "unknown")
                        ))
                    
                    yield model_service_pb2.StreamingRecognizeResponse(
                        session_id=session_id,
                        transcript_text=result["transcribed_text"],
                        is_final=False, # 這是中間結果
                        segments=segments,
                        message="",
                        rtf=rtf,
                        chunk_sec=current_audio_duration,
                        server_time_sec=server_time_taken
                    )
                else:
                    yield model_service_pb2.StreamingRecognizeResponse(
                        session_id=session_id,
                        is_final=False,
                        message=result["error"],
                        rtf=rtf,
                        chunk_sec=current_audio_duration,
                        server_time_sec=server_time_taken
                    )
                
                # 清空緩衝區，準備下一段音訊
                audio_buffer.clear()
                current_audio_duration = 0.0
            
            # 如果是最後一個請求，即使緩衝區未滿也要處理
            if request.is_last:
                logger.info(f"Session ID: {session_id} - 接收到 is_last 標記。處理剩餘音訊。")
                break

        # 串流結束，處理剩餘的緩衝區音訊 (如果有)
        if audio_buffer:
            logger.info(f"Session ID: {session_id} - 串流結束，處理剩餘 {current_audio_duration:.2f} 秒音訊...")
            audio_array = np.array(audio_buffer, dtype=np.float32)
            
            server_process_start_time = time.time()
            
            try:
                # 在線程池中執行最終轉錄
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(
                    self.executor,
                    self._transcribe_audio_chunk,
                    audio_array,
                    language
                )
            except Exception as e:
                logger.error(f"最終異步轉錄執行失敗: {e}")
                result = {"success": False, "error": f"最終異步轉錄執行失敗: {str(e)}"}
        
            server_process_end_time = time.time()
            server_time_taken = server_process_end_time - server_process_start_time
            rtf = server_time_taken / current_audio_duration if current_audio_duration > 0 else 0.0
            
            if result["success"]:
                segments = []
                for seg_data in result["segments"]:
                    words = []
                    for word_data in seg_data.get("words", []):
                        words.append(model_service_pb2.WordInfo(
                            word=word_data["word"],
                            start_time_sec=word_data["start_time_sec"],
                            end_time_sec=word_data["end_time_sec"],
                            confidence=word_data.get("confidence", 0.0)
                        ))
                    segments.append(model_service_pb2.Segment(
                        text=seg_data["text"],
                        start_time_sec=seg_data["start_time_sec"],
                        end_time_sec=seg_data["end_time_sec"],
                        words=words,
                        speaker_id=seg_data.get("speaker_id", "unknown")
                    ))
                
                yield model_service_pb2.StreamingRecognizeResponse(
                    session_id=session_id,
                    transcript_text=result["transcribed_text"],
                    is_final=True, # 這是最終結果
                    segments=segments,
                    message="",
                    rtf=rtf,
                    chunk_sec=current_audio_duration,
                    server_time_sec=server_time_taken
                )
            else:
                yield model_service_pb2.StreamingRecognizeResponse(
                    session_id=session_id,
                    is_final=True,
                    message=result["error"],
                    rtf=rtf,
                    chunk_sec=current_audio_duration,
                    server_time_sec=server_time_taken
                )
        else:
            # 如果沒有剩餘音訊，也傳送一個空的最終響應
            yield model_service_pb2.StreamingRecognizeResponse(
                session_id=session_id,
                transcript_text="", 
                is_final=True,
                segments=[],
                message="串流結束。",
                rtf=0.0,
                chunk_sec=0.0,
                server_time_sec=0.0
            )
    def __del__(self):
        """清理資源"""
        if hasattr(self, 'executor') and self.executor:
            self.executor.shutdown(wait=True)