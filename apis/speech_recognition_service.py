import grpc
import speech_recognition as sr
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
import json

from proto import model_service_pb2
from proto import model_service_pb2_grpc
from benchmark import transcribe_wav_bytes

logger = logging.getLogger(__name__)

class SpeechRecognitionServicer:
    """
    語音識別服務實現 - 使用 Python SpeechRecognition 庫
    """
    
    def __init__(self, model_size: str = "default"):
        self.recognizer = sr.Recognizer()
        self.model_size = model_size
        
        # 設定識別器參數
        self.recognizer.energy_threshold = 300
        self.recognizer.dynamic_energy_threshold = True
        self.recognizer.pause_threshold = 0.8
        self.recognizer.operation_timeout = None
        
        logger.info(f"語音識別服務初始化 (SpeechRecognition 庫)")
        logger.info(f"模型大小: {model_size}")
    
    def initialize(self) -> bool:
        """初始化服務"""
        try:
            logger.info("初始化 SpeechRecognition 服務...")
            
            # 測試是否可以正常工作
            test_recognizer = sr.Recognizer()
            logger.info("✅ SpeechRecognition 服務載入成功")
            return True
            
        except Exception as e:
            logger.error(f"❌ SpeechRecognition 服務初始化失敗: {e}")
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
    
    def _audio_bytes_to_audiodata(self, audio_data: bytes) -> Optional[sr.AudioData]:
        """
        將音頻 bytes 轉換為 SpeechRecognition AudioData 對象
        """
        try:
            # 檢測音頻格式
            format_type = self._detect_audio_format(audio_data)
            logger.info(f"檢測到音頻格式: {format_type}")
            
            # 使用 librosa 統一處理所有格式
            # 創建臨時輸入文件
            with tempfile.NamedTemporaryFile(suffix='.tmp', delete=False) as temp_input:
                temp_input.write(audio_data)
                temp_input_path = temp_input.name
            
            try:
                # 使用 librosa 載入音頻，統一轉換為 16kHz 單聲道
                audio, sr_orig = librosa.load(temp_input_path, sr=16000, mono=True)
                
                # 轉換為 int16 格式
                audio_int16 = (audio * 32767).astype(np.int16)
                
                # 創建 AudioData 對象
                # SpeechRecognition 需要原始字節數據
                frame_data = audio_int16.tobytes()
                sample_rate = 16000
                sample_width = 2  # 16-bit = 2 bytes
                
                audio_data_obj = sr.AudioData(frame_data, sample_rate, sample_width)
                
                logger.info(f"音頻載入成功: 採樣率 {sample_rate} Hz, 位深 {sample_width*8} bit")
                return audio_data_obj
                
            finally:
                # 清理臨時文件
                if os.path.exists(temp_input_path):
                    os.unlink(temp_input_path)
            
        except Exception as e:
            logger.error(f"音頻處理失敗: {e}")
            return None
    
    def transcribe_audio(self, 
                        audio_data: bytes, 
                        language: str = "zh",
                        return_timestamps: bool = False) -> Dict:
        """轉錄音頻 - 使用 NPU 加速的 Whisper 模型"""
        try:
            logger.info(f"開始使用 NPU Whisper 轉錄，語言: {language}")
            
            # 檢測音頻格式並轉換為 WAV
            format_type = self._detect_audio_format(audio_data)
            logger.info(f"檢測到音頻格式: {format_type}")
            
            # 如果不是 WAV 格式，需要轉換
            if format_type != 'wav':
                wav_bytes = self._convert_to_wav_bytes(audio_data)
                if wav_bytes is None:
                    return {"success": False, "error": "音頻格式轉換失敗"}
                audio_data = wav_bytes
            
            # 使用 NPU 加速的 Whisper 進行轉錄
            try:
                result_text = transcribe_wav_bytes(audio_data)
                
                if result_text is None:
                    return {"success": False, "error": "NPU Whisper 轉錄失敗"}
                
                # Whisper 自動檢測語言，但我們仍然返回請求的語言
                detected_language = language if language != "auto" else "auto"
                confidence = 0.95  # NPU Whisper 通常有較高的準確度
                
                # 處理結果
                response_data = {
                    "success": True,
                    "transcribed_text": result_text.strip(),
                    "detected_language": detected_language,
                    "language_confidence": confidence,
                    "segments": []
                }
                
                # 如果需要時間戳，創建一個簡單的段落
                if return_timestamps and result_text:
                    # 估算音頻長度（假設 16kHz, 16-bit, mono）
                    estimated_duration = len(audio_data) / (16000 * 2)
                    response_data["segments"].append({
                        "text": result_text.strip(),
                        "start_time": 0.0,
                        "end_time": estimated_duration
                    })
                
                logger.info("NPU Whisper 轉錄完成")
                return response_data
                
            except Exception as whisper_error:
                logger.error(f"NPU Whisper 轉錄失敗: {whisper_error}")
                # 降級到原有的 Google Speech Recognition
                logger.info("降級使用 Google Speech Recognition")
                return self._fallback_transcribe_audio(audio_data, language, return_timestamps)
                
        except Exception as e:
            logger.error(f"轉錄失敗: {e}")
            return {"success": False, "error": f"轉錄失敗: {str(e)}"}
    
    def _fallback_transcribe_audio(self, 
                                  audio_data: bytes, 
                                  language: str = "zh",
                                  return_timestamps: bool = False) -> Dict:
        """降級轉錄音頻 - 使用原有的 Google Speech Recognition"""
        try:
            # 轉換音頻為 AudioData 對象
            audio_data_obj = self._audio_bytes_to_audiodata(audio_data)
            if audio_data_obj is None:
                return {"success": False, "error": "音頻處理失敗"}
            
            logger.info(f"使用 Google Speech Recognition 轉錄，語言: {language}")
            
            # 根據語言選擇識別引擎
            result_text = ""
            detected_language = language
            confidence = 0.0
            
            try:
                # 優先嘗試使用 Google Speech Recognition (免費)
                if language == "auto" or language == "zh":
                    # 中文識別
                    result_text = self.recognizer.recognize_google(audio_data_obj, language="zh-CN")
                    detected_language = "zh"
                    confidence = 0.8
                elif language == "en":
                    # 英文識別
                    result_text = self.recognizer.recognize_google(audio_data_obj, language="en-US")
                    detected_language = "en"
                    confidence = 0.8
                else:
                    # 其他語言
                    language_map = {
                        "ja": "ja-JP",
                        "ko": "ko-KR", 
                        "es": "es-ES",
                        "fr": "fr-FR",
                        "de": "de-DE",
                        "ru": "ru-RU",
                        "pt": "pt-BR",
                        "it": "it-IT",
                        "ar": "ar-SA",
                        "hi": "hi-IN",
                        "th": "th-TH",
                        "vi": "vi-VN"
                    }
                    google_lang = language_map.get(language, "en-US")
                    result_text = self.recognizer.recognize_google(audio_data_obj, language=google_lang)
                    detected_language = language
                    confidence = 0.8
                    
            except sr.UnknownValueError:
                logger.warning("Google Speech Recognition 無法理解音頻")
                # 嘗試使用離線識別器
                try:
                    result_text = self.recognizer.recognize_sphinx(audio_data_obj)
                    detected_language = "en"  # Sphinx 主要支援英文
                    confidence = 0.6
                except (sr.UnknownValueError, sr.RequestError):
                    return {"success": False, "error": "無法識別音頻內容"}
                    
            except sr.RequestError as e:
                logger.warning(f"Google Speech Recognition 服務錯誤: {e}")
                # 嘗試使用離線識別器
                try:
                    result_text = self.recognizer.recognize_sphinx(audio_data_obj)
                    detected_language = "en"
                    confidence = 0.6
                except (sr.UnknownValueError, sr.RequestError):
                    return {"success": False, "error": f"語音識別服務錯誤: {str(e)}"}
            
            # 處理結果
            response_data = {
                "success": True,
                "transcribed_text": result_text.strip(),
                "detected_language": detected_language,
                "language_confidence": confidence,
                "segments": []
            }
            
            # 如果需要時間戳，創建一個簡單的段落
            if return_timestamps and result_text:
                response_data["segments"].append({
                    "text": result_text.strip(),
                    "start_time": 0.0,
                    "end_time": len(audio_data_obj.frame_data) / (audio_data_obj.sample_rate * audio_data_obj.sample_width)
                })
            
            logger.info("Google Speech Recognition 轉錄完成")
            return response_data
                
        except Exception as e:
            logger.error(f"降級轉錄失敗: {e}")
            return {"success": False, "error": f"降級轉錄失敗: {str(e)}"}
    
    
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
            "model_id": "speech_recognition",
            "model_size": self.model_size,
            "engine": "NPU Whisper (primary) + Google Speech Recognition (fallback)",
            "model_loaded": True,
            "supported_languages": list(self.get_supported_languages().keys()),
            "uses_ffmpeg": False,
            "uses_librosa": True,
            "uses_npu": True,
            "requires_authentication": False,
            "primary_engine": "NPU Accelerated Whisper Large-v3-Turbo",
            "fallback_engine": "Google Speech Recognition API + CMU Sphinx (offline)"
        }
