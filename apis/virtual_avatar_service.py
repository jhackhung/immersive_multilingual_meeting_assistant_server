"""
Virtual Avatar Service - 虛擬頭像服務

這個服務提供以下功能：
1. InitAvatar(img, sample_audio) - 初始化虛擬頭像
2. AvatarSpeak(text) - 讓頭像說話
3. 整合 TTS、Wav2Lip、虛擬攝像頭和虛擬麥克風
"""

import os
import sys
import io
import time
import threading
import queue
import tempfile
import cv2
import numpy as np
import grpc
import soundfile as sf
import sounddevice as sd
import pyvirtualcam
from typing import Optional, Dict, Any
import logging

# 添加模型路徑
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Fix protobuf imports
import sys
import os
proto_path = os.path.join(os.path.dirname(__file__), '..', 'proto')
if proto_path not in sys.path:
    sys.path.insert(0, proto_path)

from proto import model_service_pb2, model_service_pb2_grpc
from apis.tts_service import TtsServicer
from apis.wav2lip_service import Wav2LipServicer
from models.wav2lip_pytorch_model import Wav2LipPytorch

logger = logging.getLogger(__name__)

class VirtualMicrophoneManager:
    """虛擬麥克風管理器"""
    
    def __init__(self, device_name='CABLE Input', sample_rate=48000, block_size=1024):
        self.device_name = device_name
        self.sample_rate = sample_rate
        self.block_size = block_size
        self.device_id = None
        self.stream = None
        self.audio_queue = queue.Queue()
        self.is_streaming = False
        self.current_audio = None
        self.current_frame = 0
        
    def find_virtual_device(self):
        """查找虛擬音頻設備"""
        devices = sd.query_devices()
        for i, device in enumerate(devices):
            if (self.device_name.lower() in device['name'].lower() and 
                device['max_output_channels'] > 0):
                logger.info(f"找到虛擬麥克風設備: '{device['name']}' (ID: {i})")
                return i
        return None
    
    def initialize(self):
        """初始化虛擬麥克風"""
        self.device_id = self.find_virtual_device()
        if self.device_id is None:
            logger.warning(f"未找到虛擬麥克風設備 '{self.device_name}'")
            return False
        return True
    
    def audio_callback(self, outdata, frames, time, status):
        """音頻回調函數"""
        if status:
            logger.warning(f"虛擬麥克風音頻狀態: {status}")
        
        # 嘗試從隊列獲取新音頻
        try:
            while not self.audio_queue.empty():
                self.current_audio = self.audio_queue.get_nowait()
                self.current_frame = 0
        except queue.Empty:
            pass
        
        # 填充輸出數據
        if self.current_audio is not None and self.current_frame < len(self.current_audio):
            end_frame = min(self.current_frame + frames, len(self.current_audio))
            chunk_size = end_frame - self.current_frame
            
            outdata[:chunk_size, 0] = self.current_audio[self.current_frame:end_frame]
            
            if chunk_size < frames:
                outdata[chunk_size:, 0] = 0  # 填充靜音
            
            self.current_frame = end_frame
        else:
            # 沒有音頻數據，輸出靜音
            outdata[:, 0] = 0
    
    def start_streaming(self):
        """開始音頻流"""
        if self.device_id is None:
            if not self.initialize():
                return False
        
        try:
            self.stream = sd.OutputStream(
                device=self.device_id,
                channels=1,
                samplerate=self.sample_rate,
                callback=self.audio_callback,
                blocksize=self.block_size
            )
            self.stream.start()
            self.is_streaming = True
            logger.info("虛擬麥克風開始流式傳輸")
            return True
        except Exception as e:
            logger.error(f"啟動虛擬麥克風失敗: {e}")
            return False
    
    def stop_streaming(self):
        """停止音頻流"""
        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None
            self.is_streaming = False
            logger.info("虛擬麥克風停止流式傳輸")
    
    def queue_audio(self, audio_data: np.ndarray, source_sample_rate: int = None):
        """將音頻數據添加到播放隊列"""
        if source_sample_rate and source_sample_rate != self.sample_rate:
            # 重採樣音頻
            import librosa
            audio_data = librosa.resample(
                y=audio_data, 
                orig_sr=source_sample_rate, 
                target_sr=self.sample_rate
            )
        
        # 確保是單聲道
        if len(audio_data.shape) > 1:
            audio_data = np.mean(audio_data, axis=1)
        
        self.audio_queue.put(audio_data)
        logger.debug(f"音頻已加入播放隊列，長度: {len(audio_data)/self.sample_rate:.2f}秒")

class VirtualWebcamManager:
    """虛擬攝像頭管理器"""
    
    def __init__(self, width=640, height=480, fps=25):
        self.width = width
        self.height = height
        self.fps = fps
        self.camera = None
        self.is_streaming = False
        self.video_queue = queue.Queue(maxsize=10)  # 限制隊列大小避免記憶體問題
        self.current_frame = None
        self.streaming_thread = None
        self.stop_flag = threading.Event()
        
    def initialize(self):
        """初始化虛擬攝像頭"""
        try:
            self.camera = pyvirtualcam.Camera(
                width=self.width, 
                height=self.height, 
                fps=self.fps
            )
            logger.info(f"虛擬攝像頭初始化成功: {self.camera.device}")
            return True
        except Exception as e:
            logger.error(f"虛擬攝像頭初始化失敗: {e}")
            return False
    
    def start_streaming(self):
        """開始視頻流"""
        if not self.initialize():
            return False
        
        self.stop_flag.clear()
        self.streaming_thread = threading.Thread(target=self._streaming_loop, daemon=True)
        self.streaming_thread.start()
        self.is_streaming = True
        logger.info("虛擬攝像頭開始流式傳輸")
        return True
    
    def stop_streaming(self):
        """停止視頻流"""
        self.stop_flag.set()
        if self.streaming_thread:
            self.streaming_thread.join(timeout=2)
        
        if self.camera:
            self.camera.close()
            self.camera = None
        
        self.is_streaming = False
        logger.info("虛擬攝像頭停止流式傳輸")
    
    def _streaming_loop(self):
        """視頻流循環"""
        frame_duration = 1.0 / self.fps
        
        # 創建默認幀（黑屏或靜態圖像）
        default_frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        cv2.putText(default_frame, "Avatar Ready", (50, self.height//2), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        while not self.stop_flag.is_set():
            start_time = time.time()
            
            # 獲取最新幀
            frame = None
            try:
                # 嘗試獲取最新的幀，清空舊幀
                while not self.video_queue.empty():
                    frame = self.video_queue.get_nowait()
            except queue.Empty:
                pass
            
            # 使用最新幀或默認幀
            if frame is not None:
                self.current_frame = frame
            
            display_frame = self.current_frame if self.current_frame is not None else default_frame
            
            # 調整幀大小
            if display_frame.shape[:2] != (self.height, self.width):
                display_frame = cv2.resize(display_frame, (self.width, self.height))
            
            # 轉換顏色格式 (BGR -> RGB)
            display_frame_rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
            
            # 發送到虛擬攝像頭
            if self.camera:
                self.camera.send(display_frame_rgb)
                self.camera.sleep_until_next_frame()
            
            # 控制幀率
            elapsed = time.time() - start_time
            sleep_time = max(0, frame_duration - elapsed)
            if sleep_time > 0:
                time.sleep(sleep_time)
    
    def queue_video_frames(self, frames: list):
        """將視頻幀添加到播放隊列"""
        for frame in frames:
            try:
                self.video_queue.put_nowait(frame)
            except queue.Full:
                # 如果隊列滿了，移除舊幀
                try:
                    self.video_queue.get_nowait()
                    self.video_queue.put_nowait(frame)
                except queue.Empty:
                    pass
        
        logger.debug(f"已添加 {len(frames)} 幀到視頻隊列")

class VirtualAvatarService:
    """虛擬頭像服務主類"""
    
    def __init__(self):
        self.tts_service = None
        self.wav2lip_service = None
        self.virtual_mic = None
        self.virtual_webcam = None
        
        # 頭像狀態
        self.avatar_initialized = False
        self.avatar_image = None
        self.avatar_sample_audio = None
        self.avatar_image_path = None
        self.avatar_sample_audio_path = None
        
        # 初始化服務
        self._initialize_services()
    
    def _initialize_services(self):
        """初始化所有服務"""
        try:
            logger.info("初始化 TTS 服務...")
            self.tts_service = TtsServicer()
            
            logger.info("初始化 Wav2Lip 服務...")
            self.wav2lip_service = Wav2LipServicer()
            
            logger.info("初始化虛擬麥克風...")
            self.virtual_mic = VirtualMicrophoneManager()
            
            logger.info("初始化虛擬攝像頭...")
            self.virtual_webcam = VirtualWebcamManager()
            
            logger.info("虛擬頭像服務初始化完成")
            
        except Exception as e:
            logger.error(f"服務初始化失敗: {e}")
            raise
    
    def init_avatar(self, image_data: bytes, sample_audio_data: bytes) -> bool:
        """
        初始化虛擬頭像
        
        Args:
            image_data: 頭像圖片的二進制數據
            sample_audio_data: 樣本音頻的二進制數據
            
        Returns:
            bool: 初始化是否成功
        """
        try:
            logger.info("開始初始化虛擬頭像...")
            
            # 保存頭像數據到臨時文件
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as temp_img:
                temp_img.write(image_data)
                self.avatar_image_path = temp_img.name
            
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio:
                temp_audio.write(sample_audio_data)
                self.avatar_sample_audio_path = temp_audio.name
            
            # 驗證圖片
            test_img = cv2.imread(self.avatar_image_path)
            if test_img is None:
                raise ValueError("無法讀取提供的圖片數據")
            
            # 驗證音頻
            try:
                test_audio, sr = sf.read(self.avatar_sample_audio_path)
                if len(test_audio) == 0:
                    raise ValueError("音頻數據為空")
            except Exception as e:
                raise ValueError(f"無法讀取提供的音頻數據: {e}")
            
            # 保存數據
            self.avatar_image = image_data
            self.avatar_sample_audio = sample_audio_data
            
            # 啟動虛擬設備
            if not self.virtual_mic.start_streaming():
                logger.warning("虛擬麥克風啟動失敗，但繼續初始化")
            
            if not self.virtual_webcam.start_streaming():
                logger.warning("虛擬攝像頭啟動失敗，但繼續初始化")
            
            self.avatar_initialized = True
            logger.info("虛擬頭像初始化成功")
            return True
            
        except Exception as e:
            logger.error(f"虛擬頭像初始化失敗: {e}")
            self._cleanup_temp_files()
            return False
    
    def avatar_speak(self, text: str, language: str = "en") -> bool:
        """
        讓虛擬頭像說話
        
        Args:
            text: 要說的文字
            language: 語言代碼
            
        Returns:
            bool: 是否成功
        """
        if not self.avatar_initialized:
            logger.error("頭像未初始化，無法說話")
            return False
        
        try:
            logger.info(f"頭像開始說話: '{text[:50]}...'")
            
            # 1. 使用 TTS 生成音頻
            logger.debug("生成 TTS 音頻...")
            tts_request = model_service_pb2.TtsRequest(
                text_to_speak=text,
                reference_audio=self.avatar_sample_audio,
                language=language
            )
            
            # 創建模擬的 gRPC 上下文
            class MockContext:
                def set_code(self, code): pass
                def set_details(self, details): pass
            
            tts_response = self.tts_service.Tts(tts_request, MockContext())
            
            if not tts_response.generated_audio:
                logger.error("TTS 生成失敗")
                return False
            
            # 2. 使用 Wav2Lip 生成對嘴視頻
            logger.debug("生成 Wav2Lip 視頻...")
            wav2lip_request = model_service_pb2.Wav2LipRequest(
                audio_data=tts_response.generated_audio,
                image_data=self.avatar_image
            )
            
            wav2lip_response = self.wav2lip_service.Wav2Lip(wav2lip_request, MockContext())
            
            if not wav2lip_response.video_data:
                logger.error("Wav2Lip 生成失敗")
                return False
            
            # 3. 解析音頻和視頻
            logger.debug("解析生成的媒體...")
            
            # 解析音頻
            audio_io = io.BytesIO(tts_response.generated_audio)
            audio_data, audio_sr = sf.read(audio_io)
            
            # 解析視頻
            video_frames = self._extract_video_frames(wav2lip_response.video_data)
            
            if not video_frames:
                logger.error("無法從生成的視頻中提取幀")
                return False
            
            # 4. 同步播放音頻和視頻
            logger.debug("開始同步播放...")
            self._play_synced_media(audio_data, audio_sr, video_frames)
            
            logger.info("頭像說話完成")
            return True
            
        except Exception as e:
            logger.error(f"頭像說話失敗: {e}")
            import traceback
            logger.error(f"詳細錯誤: {traceback.format_exc()}")
            return False
    
    def _extract_video_frames(self, video_data: bytes) -> list:
        """從視頻數據中提取幀"""
        try:
            # 保存視頻到臨時文件
            with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as temp_video:
                temp_video.write(video_data)
                temp_video_path = temp_video.name
            
            # 使用 OpenCV 讀取視頻
            cap = cv2.VideoCapture(temp_video_path)
            frames = []
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frames.append(frame)
            
            cap.release()
            
            # 清理臨時文件
            try:
                os.unlink(temp_video_path)
            except:
                pass
            
            logger.debug(f"從視頻中提取了 {len(frames)} 幀")
            return frames
            
        except Exception as e:
            logger.error(f"提取視頻幀失敗: {e}")
            return []
    
    def _play_synced_media(self, audio_data: np.ndarray, audio_sr: int, video_frames: list):
        """同步播放音頻和視頻"""
        try:
            # 將音頻加入播放隊列
            if self.virtual_mic.is_streaming:
                self.virtual_mic.queue_audio(audio_data, audio_sr)
            
            # 將視頻幀加入播放隊列
            if self.virtual_webcam.is_streaming:
                self.virtual_webcam.queue_video_frames(video_frames)
            
            logger.debug("媒體已加入播放隊列")
            
        except Exception as e:
            logger.error(f"媒體播放失敗: {e}")
    
    def _cleanup_temp_files(self):
        """清理臨時文件"""
        for path in [self.avatar_image_path, self.avatar_sample_audio_path]:
            if path and os.path.exists(path):
                try:
                    os.unlink(path)
                except:
                    pass
    
    def cleanup(self):
        """清理資源"""
        logger.info("清理虛擬頭像服務資源...")
        
        if self.virtual_mic:
            self.virtual_mic.stop_streaming()
        
        if self.virtual_webcam:
            self.virtual_webcam.stop_streaming()
        
        self._cleanup_temp_files()
        
        self.avatar_initialized = False
        logger.info("虛擬頭像服務清理完成")

# gRPC 服務實現
class VirtualAvatarServicer(model_service_pb2_grpc.MediaServiceServicer):
    """虛擬頭像 gRPC 服務實現"""
    
    def __init__(self):
        self.avatar_service = VirtualAvatarService()
        logger.info("虛擬頭像 gRPC 服務初始化完成")
    
    def InitAvatar(self, request, context):
        """初始化頭像 RPC"""
        try:
            success = self.avatar_service.init_avatar(
                request.image_data,
                request.sample_audio_data
            )
            
            return model_service_pb2.InitAvatarResponse(
                success=success,
                message="頭像初始化成功" if success else "頭像初始化失敗"
            )
            
        except Exception as e:
            logger.error(f"InitAvatar RPC 失敗: {e}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"頭像初始化失敗: {str(e)}")
            return model_service_pb2.InitAvatarResponse(
                success=False,
                message=f"初始化失敗: {str(e)}"
            )
    
    def AvatarSpeak(self, request, context):
        """頭像說話 RPC"""
        try:
            success = self.avatar_service.avatar_speak(
                request.text,
                request.language or "en"
            )
            
            return model_service_pb2.AvatarSpeakResponse(
                success=success,
                message="頭像說話成功" if success else "頭像說話失敗"
            )
            
        except Exception as e:
            logger.error(f"AvatarSpeak RPC 失敗: {e}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"頭像說話失敗: {str(e)}")
            return model_service_pb2.AvatarSpeakResponse(
                success=False,
                message=f"說話失敗: {str(e)}"
            )

# 單例模式的全局服務實例
_avatar_service_instance = None

def get_avatar_service() -> VirtualAvatarService:
    """獲取虛擬頭像服務實例（單例模式）"""
    global _avatar_service_instance
    if _avatar_service_instance is None:
        _avatar_service_instance = VirtualAvatarService()
    return _avatar_service_instance

def cleanup_avatar_service():
    """清理虛擬頭像服務"""
    global _avatar_service_instance
    if _avatar_service_instance:
        _avatar_service_instance.cleanup()
        _avatar_service_instance = None
