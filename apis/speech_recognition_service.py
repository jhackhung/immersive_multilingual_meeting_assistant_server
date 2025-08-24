import grpc
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import numpy as np
import io
import tempfile
import logging
import os
import librosa
import soundfile as sf
from typing import Optional, List, Dict
import wave
import struct

from proto import model_service_pb2
from proto import model_service_pb2_grpc

logger = logging.getLogger(__name__)

class SpeechRecognitionServicer:
    """
    語音識別服務實現 - 獨立版本，不依賴外部工具
    """
    
    def __init__(self, model_size: str = "large-v3-turbo"):
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        self.model_id = "openai/whisper-large-v3-turbo"
        
        self.model = None
        self.processor = None
        self.pipe = None
        
        logger.info(f"語音識別服務初始化 (獨立版本)")
        logger.info(f"設備: {self.device}, 精度: {self.torch_dtype}")
    
    def initialize(self) -> bool:
        """初始化模型"""
        try:
            logger.info("載入 Whisper V3 Turbo 模型...")
            
            self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
                self.model_id, 
                torch_dtype=self.torch_dtype, 
                low_cpu_mem_usage=True, 
                use_safetensors=True
            )
            self.model.to(self.device)
            
            self.processor = AutoProcessor.from_pretrained(self.model_id)
            
            self.pipe = pipeline(
                "automatic-speech-recognition",
                model=self.model,
                tokenizer=self.processor.tokenizer,
                feature_extractor=self.processor.feature_extractor,
                torch_dtype=self.torch_dtype,
                device=self.device,
            )
            
            logger.info("✅ 模型載入成功")
            return True
            
        except Exception as e:
            logger.error(f"❌ 模型載入失敗: {e}")
            return False
    
    def _detect_audio_format(self, audio_data: bytes) -> str:
        """檢測音頻格式"""
        # WAV 文件標識
        if audio_data.startswith(b'RIFF') and b'WAVE' in audio_data[:12]:
            return 'wav'
        # MP3 文件標識
        elif audio_data.startswith(b'ID3') or audio_data[0:2] == b'\xff\xfb':
            return 'mp3'
        # MP4/M4A 文件標識
        elif b'ftyp' in audio_data[:20]:
            return 'mp4'
        # OGG 文件標識
        elif audio_data.startswith(b'OggS'):
            return 'ogg'
        else:
            return 'unknown'
    
    def _convert_to_wav_bytes(self, audio_data: bytes) -> Optional[bytes]:
        """
        將音頻轉換為 WAV 格式的 bytes
        使用純 Python + librosa，不依賴 ffmpeg
        """
        try:
            # 創建臨時輸入文件
            with tempfile.NamedTemporaryFile(suffix='.tmp', delete=False) as temp_input:
                temp_input.write(audio_data)
                temp_input_path = temp_input.name
            
            try:
                # 使用 librosa 載入音頻
                audio, sr = librosa.load(temp_input_path, sr=16000, mono=True)
                
                # 轉換為 int16 格式
                audio_int16 = (audio * 32767).astype(np.int16)
                
                # 創建 WAV bytes
                with io.BytesIO() as wav_buffer:
                    with wave.open(wav_buffer, 'wb') as wav_file:
                        wav_file.setnchannels(1)  # 單聲道
                        wav_file.setsampwidth(2)  # 16-bit
                        wav_file.setframerate(16000)  # 16kHz
                        wav_file.writeframes(audio_int16.tobytes())
                    
                    wav_bytes = wav_buffer.getvalue()
                
                logger.info(f"音頻轉換成功: {len(audio)} 採樣點 → {len(wav_bytes)} bytes")
                return wav_bytes
                
            finally:
                # 清理臨時文件
                if os.path.exists(temp_input_path):
                    os.unlink(temp_input_path)
                
        except Exception as e:
            logger.error(f"音頻轉換失敗: {e}")
            return None
    
    def _audio_bytes_to_array(self, audio_data: bytes) -> Optional[np.ndarray]:
        """
        將音頻 bytes 轉換為 numpy 陣列
        完全使用 Python，不依賴外部工具
        """
        try:
            # 檢測音頻格式
            format_type = self._detect_audio_format(audio_data)
            logger.info(f"檢測到音頻格式: {format_type}")
            
            # 如果不是 WAV，先轉換
            if format_type != 'wav':
                logger.info("轉換音頻格式為 WAV...")
                wav_data = self._convert_to_wav_bytes(audio_data)
                if wav_data is None:
                    return None
                audio_data = wav_data
            
            # 使用 BytesIO 創建文件對象
            audio_io = io.BytesIO(audio_data)
            
            # 用 librosa 從 BytesIO 載入
            audio, sr = librosa.load(audio_io, sr=16000, mono=True)
            
            logger.info(f"音頻載入成功: {len(audio)} 採樣點, {sr} Hz")
            return audio
            
        except Exception as e:
            logger.error(f"音頻處理失敗: {e}")
            return None
    
    def transcribe_audio(self, 
                        audio_data: bytes, 
                        language: str = "zh",
                        return_timestamps: bool = False) -> Dict:
        """轉錄音頻"""
        if self.pipe is None:
            if not self.initialize():
                return {"success": False, "error": "模型未初始化"}
        
        try:
            # 轉換音頻為 numpy 陣列
            audio_array = self._audio_bytes_to_array(audio_data)
            if audio_array is None:
                return {"success": False, "error": "音頻處理失敗"}
            
            logger.info(f"開始轉錄，語言: {language}")
            
            # 設定參數
            generate_kwargs = {"task": "transcribe"}
            
            if language != "auto":
                generate_kwargs["language"] = language
            
            if return_timestamps:
                generate_kwargs["return_timestamps"] = True
            
            # 執行轉錄
            result = self.pipe(audio_array, generate_kwargs=generate_kwargs)
            
            # 處理結果
            response_data = {
                "success": True,
                "transcribed_text": result.get("text", "").strip(),
                "detected_language": language,
                "language_confidence": 1.0,
                "segments": []
            }
            
            # 處理時間戳
            if return_timestamps and "chunks" in result:
                for chunk in result["chunks"]:
                    if "timestamp" in chunk and chunk["timestamp"]:
                        start_time = chunk["timestamp"][0] or 0.0
                        end_time = chunk["timestamp"][1] or 0.0
                        text = chunk.get("text", "").strip()
                        
                        response_data["segments"].append({
                            "text": text,
                            "start_time": start_time,
                            "end_time": end_time
                        })
            
            logger.info("轉錄完成")
            return response_data
                
        except Exception as e:
            logger.error(f"轉錄失敗: {e}")
            return {"success": False, "error": f"轉錄失敗: {str(e)}"}
    
    def SpeechRecognition(self, request, context):
        """gRPC 接口"""
        try:
            logger.info(f"語音識別請求: {len(request.audio_data)} bytes")
            
            result = self.transcribe_audio(
                audio_data=request.audio_data,
                language=request.language or "zh",
                return_timestamps=request.return_timestamps
            )
            
            if not result["success"]:
                context.set_code(grpc.StatusCode.INTERNAL)
                context.set_details(result["error"])
                return model_service_pb2.SpeechRecognitionResponse(success=False)
            
            segments = []
            if request.return_timestamps:
                for seg in result["segments"]:
                    segment = model_service_pb2.TranscriptionSegment(
                        text=seg["text"],
                        start_time=seg["start_time"],
                        end_time=seg["end_time"]
                    )
                    segments.append(segment)
            
            return model_service_pb2.SpeechRecognitionResponse(
                transcribed_text=result["transcribed_text"],
                detected_language=result["detected_language"],
                language_confidence=result["language_confidence"],
                segments=segments,
                success=True
            )
            
        except Exception as e:
            logger.error(f"gRPC 錯誤: {e}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return model_service_pb2.SpeechRecognitionResponse(success=False)
    
    # 添加缺少的方法
    def get_supported_languages(self) -> Dict[str, str]:
        """
        獲取支援的語言列表
        
        Returns:
            Dict[str, str]: 語言代碼到語言名稱的映射
        """
        supported_languages = {
            "auto": "自動檢測",
            "zh": "中文",
            "en": "英語", 
            "ja": "日語",
            "ko": "韓語",
            "es": "西班牙語",
            "fr": "法語",
            "de": "德語",
            "ru": "俄語",
            "pt": "葡萄牙語",
            "it": "義大利語",
            "ar": "阿拉伯語",
            "hi": "印地語",
            "th": "泰語",
            "vi": "越南語"
        }
        return supported_languages.copy()
    
    def get_model_info(self) -> Dict[str, str]:
        """
        獲取當前模型訊息
        
        Returns:
            Dict[str, str]: 模型訊息
        """
        return {
            "model_id": self.model_id,
            "model_size": "large-v3-turbo",
            "device": self.device,
            "torch_dtype": str(self.torch_dtype),
            "model_loaded": self.pipe is not None,
            "supported_languages": list(self.get_supported_languages().keys()),
            "uses_ffmpeg": False,  # 標示不依賴 ffmpeg
            "uses_librosa": True   # 標示使用 librosa
        }
