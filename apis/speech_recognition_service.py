import grpc
import whisper
import numpy as np
import io
import tempfile
import logging
from typing import Optional, List, Dict

from proto import model_service_pb2
from proto import model_service_pb2_grpc

logger = logging.getLogger(__name__)

class SpeechRecognitionServicer:
    """
    語音識別服務實現，使用 OpenAI Whisper 模型進行語音轉文字
    """
    
    def __init__(self, model_size: str = "base"):
        """
        初始化語音識別服務
        
        Args:
            model_size (str): Whisper 模型大小，可選 "tiny", "base", "small", "medium", "large"
        """
        self.model = None
        self.model_size = model_size
        self.supported_languages = {
            "auto": "自動檢測",
            "en": "英語",
            "zh": "中文",
            "ja": "日語",
            "ko": "韓語",
            "es": "西班牙語",
            "fr": "法語",
            "de": "德語",
            "ru": "俄語",
            "pt": "葡萄牙語",
            "it": "義大利語"
        }
        logger.info(f"語音識別服務初始化，使用模型: {model_size}")
    
    def initialize(self) -> bool:
        """
        初始化 Whisper 模型
        
        Returns:
            bool: 初始化是否成功
        """
        try:
            logger.info(f"正在載入 Whisper {self.model_size} 模型...")
            self.model = whisper.load_model(self.model_size)
            logger.info("✅ Whisper 模型載入成功")
            return True
        except Exception as e:
            logger.error(f"❌ Whisper 模型載入失敗: {e}")
            return False
    
    def _preprocess_audio(self, audio_data: bytes) -> Optional[np.ndarray]:
        """
        預處理音訊數據
        
        Args:
            audio_data (bytes): 原始音訊數據
            
        Returns:
            Optional[np.ndarray]: 預處理後的音訊陣列，失敗時返回 None
        """
        try:
            import soundfile as sf
            
            # 將 bytes 轉換為音訊陣列
            audio_buffer = io.BytesIO(audio_data)
            audio_array, sample_rate = sf.read(audio_buffer)
            
            # 如果是立體聲，轉換為單聲道
            if len(audio_array.shape) > 1:
                audio_array = np.mean(audio_array, axis=1)
            
            # 轉換為 float32
            audio_array = audio_array.astype(np.float32)
            
            # Whisper 需要 16kHz 採樣率
            if sample_rate != 16000:
                import librosa
                audio_array = librosa.resample(y=audio_array, orig_sr=sample_rate, target_sr=16000)
                logger.info(f"音訊重新採樣從 {sample_rate}Hz 到 16000Hz")
            
            return audio_array
            
        except Exception as e:
            logger.error(f"音訊預處理失敗: {e}")
            return None
    
    def _save_temp_audio(self, audio_array: np.ndarray) -> Optional[str]:
        """
        將音訊陣列保存為臨時文件
        
        Args:
            audio_array (np.ndarray): 音訊陣列
            
        Returns:
            Optional[str]: 臨時文件路径，失敗時返回 None
        """
        try:
            import soundfile as sf
            
            # 創建臨時文件
            temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            sf.write(temp_file.name, audio_array, 16000)
            temp_file.close()
            
            return temp_file.name
            
        except Exception as e:
            logger.error(f"保存臨時音訊文件失敗: {e}")
            return None
    
    def _cleanup_temp_file(self, file_path: str):
        """
        清理臨時文件
        
        Args:
            file_path (str): 要刪除的文件路径
        """
        try:
            import os
            if os.path.exists(file_path):
                os.unlink(file_path)
        except Exception as e:
            logger.warning(f"清理臨時文件失敗: {e}")
    
    def transcribe_audio(self, 
                        audio_data: bytes, 
                        language: str = "auto",
                        return_timestamps: bool = False) -> Dict:
        """
        轉錄音訊為文字
        
        Args:
            audio_data (bytes): 音訊數據
            language (str): 語言代碼，"auto" 表示自動檢測
            return_timestamps (bool): 是否返回時間戳訊息
            
        Returns:
            Dict: 轉錄結果
        """
        if self.model is None:
            return {
                "success": False,
                "error": "模型未初始化"
            }
        
        try:
            # 預處理音訊
            audio_array = self._preprocess_audio(audio_data)
            if audio_array is None:
                return {
                    "success": False,
                    "error": "音訊預處理失敗"
                }
            
            # 保存為臨時文件
            temp_file_path = self._save_temp_audio(audio_array)
            if temp_file_path is None:
                return {
                    "success": False,
                    "error": "保存臨時文件失敗"
                }
            
            try:
                # 設定轉錄選項
                options = {
                    "verbose": False,
                    "word_timestamps": return_timestamps
                }
                
                # 如果不是自動檢測，設定語言
                if language != "auto" and language in self.supported_languages:
                    options["language"] = language
                
                # 執行轉錄
                logger.info(f"開始轉錄音訊，語言: {language}, 時間戳: {return_timestamps}")
                result = self.model.transcribe(temp_file_path, **options)
                
                # 準備回應數據
                response_data = {
                    "success": True,
                    "transcribed_text": result.get("text", "").strip(),
                    "detected_language": result.get("language", "unknown"),
                    "language_confidence": 0.0,  # Whisper 不直接提供此數據
                    "segments": []
                }
                
                # 如果需要時間戳，提取片段訊息
                if return_timestamps and "segments" in result:
                    for segment in result["segments"]:
                        response_data["segments"].append({
                            "text": segment.get("text", "").strip(),
                            "start_time": segment.get("start", 0.0),
                            "end_time": segment.get("end", 0.0)
                        })
                
                logger.info(f"轉錄完成，檢測語言: {response_data['detected_language']}")
                return response_data
                
            finally:
                # 清理臨時文件
                self._cleanup_temp_file(temp_file_path)
                
        except Exception as e:
            logger.error(f"語音轉錄失敗: {e}")
            return {
                "success": False,
                "error": f"轉錄失敗: {str(e)}"
            }
    
    def SpeechRecognition(self, request, context):
        """
        gRPC 語音識別服務端點
        
        Args:
            request: SpeechRecognitionRequest
            context: gRPC 上下文
            
        Returns:
            SpeechRecognitionResponse
        """
        try:
            logger.info(f"收到語音識別請求，音訊大小: {len(request.audio_data)} bytes")
            
            # 提取請求參數
            language = request.language if request.language else "auto"
            return_timestamps = request.return_timestamps
            model_size = request.model_size if request.model_size else self.model_size
            
            # 如果請求的模型大小與當前不同，重新載入模型
            if model_size != self.model_size:
                logger.info(f"切換模型大小從 {self.model_size} 到 {model_size}")
                try:
                    self.model = whisper.load_model(model_size)
                    self.model_size = model_size
                    logger.info(f"✅ 模型切換成功")
                except Exception as e:
                    logger.warning(f"模型切換失敗，使用原模型: {e}")
            
            # 執行轉錄
            result = self.transcribe_audio(
                audio_data=request.audio_data,
                language=language,
                return_timestamps=return_timestamps
            )
            
            if not result["success"]:
                context.set_code(grpc.StatusCode.INTERNAL)
                context.set_details(result.get("error", "未知錯誤"))
                return model_service_pb2.SpeechRecognitionResponse(
                    success=False
                )
            
            # 構建回應
            segments = []
            if return_timestamps:
                for seg in result["segments"]:
                    segment = model_service_pb2.TranscriptionSegment(
                        text=seg["text"],
                        start_time=seg["start_time"],
                        end_time=seg["end_time"]
                    )
                    segments.append(segment)
            
            response = model_service_pb2.SpeechRecognitionResponse(
                transcribed_text=result["transcribed_text"],
                detected_language=result["detected_language"],
                language_confidence=result["language_confidence"],
                segments=segments,
                success=True
            )
            
            logger.info("語音識別處理完成")
            return response
            
        except Exception as e:
            logger.error(f"語音識別服務錯誤: {e}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"語音識別失敗: {str(e)}")
            return model_service_pb2.SpeechRecognitionResponse(
                success=False
            )
    
    def get_supported_languages(self) -> Dict[str, str]:
        """
        獲取支援的語言列表
        
        Returns:
            Dict[str, str]: 語言代碼到語言名稱的映射
        """
        return self.supported_languages.copy()
    
    def get_model_info(self) -> Dict[str, str]:
        """
        獲取當前模型訊息
        
        Returns:
            Dict[str, str]: 模型訊息
        """
        return {
            "model_size": self.model_size,
            "model_loaded": self.model is not None,
            "supported_languages": list(self.supported_languages.keys())
        }
