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
import librosa
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
    
    def __init__(self, device_name='CABLE In', sample_rate=48000, block_size=1024):
        self.device_name = device_name
        self.sample_rate = sample_rate
        self.block_size = block_size
        self.device_id = None
        self.stream = None
        self.audio_queue = queue.Queue()
        self.is_streaming = False
        self.current_audio = None
        self.current_frame = 0
        self.audio_lock = threading.Lock()  # Add lock for thread safety
        
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
        
        with self.audio_lock:
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
        # 如果已經在流式傳輸，避免重複初始化
        if self.is_streaming and self.stream and self.stream.active:
            logger.info("虛擬麥克風已經在流式傳輸中，跳過重新啟動")
            return True
            
        # 如果之前的流還在運行，先停止它
        if self.stream:
            try:
                self.stream.stop()
                self.stream.close()
                self.stream = None
                logger.info("停止了之前的虛擬麥克風流")
            except Exception as e:
                logger.warning(f"停止之前的流時出錯: {e}")
        
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
        self.video_queue = queue.Queue(maxsize=50)  # 增加隊列大小以支持更流暢的播放
        self.current_frame = None
        self.default_frame = None  # 添加默認幀屬性
        self.streaming_thread = None
        self.stop_flag = threading.Event()
        
    def set_default_avatar_image(self, image_path: str):
        """設置默認頭像圖片"""
        try:
            # 讀取圖片
            avatar_img = cv2.imread(image_path)
            if avatar_img is None:
                logger.warning(f"無法讀取頭像圖片: {image_path}")
                return False
            
            logger.info(f"讀取頭像圖片: {image_path}, 原始尺寸: {avatar_img.shape}")
            
            # 調整圖片大小到攝像頭尺寸
            self.default_frame = cv2.resize(avatar_img, (self.width, self.height))
            logger.info(f"頭像圖片已調整到目標尺寸: {self.width}x{self.height}")
            
            # 確保數據類型正確
            if self.default_frame.dtype != np.uint8:
                self.default_frame = self.default_frame.astype(np.uint8)
            
            # 如果當前沒有播放視頻，立即更新當前幀
            if self.current_frame is None:
                self.current_frame = self.default_frame.copy()
            
            logger.info(f"成功設置默認頭像圖片: {image_path}")
            return True
            
        except Exception as e:
            logger.error(f"設置默認頭像圖片失敗: {e}")
            import traceback
            logger.error(f"詳細錯誤: {traceback.format_exc()}")
            return False
        
    def initialize(self):
        """初始化虛擬攝像頭"""
        try:
            # 如果已經有攝像頭實例，先關閉它
            if self.camera is not None:
                try:
                    self.camera.close()
                    logger.info("關閉了之前的虛擬攝像頭實例")
                except Exception as e:
                    logger.warning(f"關閉之前的攝像頭時出錯: {e}")
                
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
        # 如果已經在流式傳輸，避免重複初始化
        if self.is_streaming and self.streaming_thread and self.streaming_thread.is_alive():
            logger.info("虛擬攝像頭已經在流式傳輸中，跳過重新啟動")
            return True
            
        # 如果之前的流還在運行，先停止它
        if self.is_streaming:
            self.stop_streaming()
            logger.info("停止了之前的虛擬攝像頭流")
        
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
        
        # 創建基本默認幀（黑屏或靜態圖像）- 只在沒有設置頭像圖片時使用
        basic_default_frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        cv2.putText(basic_default_frame, "Avatar Ready", (50, self.height//2), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        while not self.stop_flag.is_set():
            start_time = time.time()
            
            # 獲取下一幀（一次只取一幀以保持正確的播放速度）
            frame = None
            try:
                # 只取一幀，不要清空整個隊列
                frame = self.video_queue.get_nowait()
                self.current_frame = frame
                logger.debug("從隊列獲取新視頻幀")
            except queue.Empty:
                # 沒有新幀，繼續使用當前幀
                pass
            
            # 確定要顯示的幀：當前幀 -> 頭像默認幀 -> 基本默認幀
            display_frame = None
            if self.current_frame is not None:
                display_frame = self.current_frame.copy()
            elif self.default_frame is not None:
                display_frame = self.default_frame.copy()
            else:
                display_frame = basic_default_frame.copy()
            
            # 標準化幀
            display_frame = self._normalize_frame(display_frame)
            if display_frame is None:
                logger.error("無法標準化顯示幀，使用基本默認幀")
                display_frame = basic_default_frame.copy()
            
            # 轉換顏色格式 (BGR -> RGB)
            try:
                display_frame_rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
            except Exception as e:
                logger.error(f"顏色格式轉換失敗: {e}")
                # 使用基本默認幀作為後備
                display_frame_rgb = cv2.cvtColor(basic_default_frame, cv2.COLOR_BGR2RGB)
            
            # 發送到虛擬攝像頭
            if self.camera:
                try:
                    self.camera.send(display_frame_rgb)
                    self.camera.sleep_until_next_frame()
                except Exception as e:
                    logger.error(f"發送幀到虛擬攝像頭失敗: {e}")
            
            # 控制幀率
            elapsed = time.time() - start_time
            sleep_time = max(0, frame_duration - elapsed)
            if sleep_time > 0:
                time.sleep(sleep_time)
    
    def queue_video_frames(self, frames: list):
        """將視頻幀添加到播放隊列"""
        if not frames:
            logger.warning("嘗試添加空視頻幀列表")
            return
        
        logger.info(f"開始添加 {len(frames)} 幀到視頻隊列")
        
        # 清空現有隊列以避免舊幀堆積
        while not self.video_queue.empty():
            try:
                self.video_queue.get_nowait()
            except queue.Empty:
                break
        
        # 使用線程逐幀添加以控制播放速度
        def add_frames_gradually():
            frame_interval = 1.0 / self.fps  # 每幀間隔時間
            frames_added = 0
            
            for i, frame in enumerate(frames):
                if frame is None:
                    logger.warning(f"跳過第 {i+1} 個空視頻幀")
                    continue
                
                try:
                    # 統一調整幀尺寸以避免解析度不匹配問題
                    normalized_frame = self._normalize_frame(frame)
                    if normalized_frame is None:
                        logger.warning(f"第 {i+1} 幀標準化失敗，跳過")
                        continue
                    
                    # 如果隊列滿了，等待一下
                    while self.video_queue.full():
                        time.sleep(0.01)
                    
                    self.video_queue.put(normalized_frame, timeout=1.0)
                    frames_added += 1
                    
                    # 控制添加速度，略快於播放速度以保持緩衝
                    if i < len(frames) - 1:  # 不在最後一幀等待
                        time.sleep(frame_interval * 0.8)  # 比播放速度快20%
                        
                except queue.Full:
                    logger.warning(f"視頻隊列已滿，跳過第 {i+1} 幀")
                    continue
                except Exception as e:
                    logger.error(f"添加第 {i+1} 幀時出錯: {e}")
                    continue
            
            logger.info(f"視頻幀添加完成: {frames_added}/{len(frames)} 幀")
        
        # 在後台線程中添加幀
        add_thread = threading.Thread(target=add_frames_gradually, daemon=True)
        add_thread.start()
        
        logger.debug(f"開始後台添加 {len(frames)} 幀到視頻隊列")
    
    def _normalize_frame(self, frame: np.ndarray) -> np.ndarray:
        """標準化幀尺寸和格式"""
        try:
            if frame is None:
                logger.warning("嘗試標準化空幀")
                return None
            
            # 檢查並調整尺寸
            if frame.shape[:2] != (self.height, self.width):
                logger.debug(f"調整幀尺寸從 {frame.shape[:2]} 到 ({self.height}, {self.width})")
                frame = cv2.resize(frame, (self.width, self.height))
            
            # 確保數據類型正確
            if frame.dtype != np.uint8:
                frame = frame.astype(np.uint8)
            
            # 確保是3通道BGR圖像
            if len(frame.shape) == 2:
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            elif len(frame.shape) == 3 and frame.shape[2] == 4:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
            elif len(frame.shape) == 3 and frame.shape[2] != 3:
                logger.warning(f"未知的通道數: {frame.shape[2]}")
                # 如果通道數不是3，嘗試轉換為3通道
                if frame.shape[2] == 1:
                    frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                else:
                    # 截取前3個通道
                    frame = frame[:, :, :3]
            
            return frame
            
        except Exception as e:
            logger.error(f"標準化幀失敗: {e}")
            return None

    def reset_to_default_avatar(self):
        """重置到默認頭像圖片"""
        if self.default_frame is not None:
            self.current_frame = self.default_frame.copy()
            logger.debug("已重置到默認頭像圖片")
        else:
            self.current_frame = None
            logger.debug("已清除當前幀，將顯示基本默認幀")

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
        
        # 虛擬設備狀態
        self.microphone_initialized = False
        self.camera_initialized = False
        
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
            # 清理已初始化的服務
            if hasattr(self, 'virtual_mic'):
                try:
                    self.virtual_mic.stop_streaming()
                except:
                    pass
            if hasattr(self, 'virtual_webcam'):
                try:
                    self.virtual_webcam.stop_streaming()
                except:
                    pass
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
            
            # 啟動虛擬設備（避免重複初始化）
            if not self.microphone_initialized:
                self.microphone_initialized = self.virtual_mic.start_streaming()
                if not self.microphone_initialized:
                    logger.warning("虛擬麥克風啟動失敗")
                else:
                    logger.info("虛擬麥克風已成功啟動")
            else:
                logger.info("虛擬麥克風已經在運行，跳過重新初始化")

            # not working on arm devices
            # if not self.camera_initialized:
            #     self.camera_initialized = self.virtual_webcam.start_streaming()
            #     if not self.camera_initialized:
            #         logger.warning("虛擬攝像頭啟動失敗")
            #     else:
            #         logger.info("虛擬攝像頭已成功啟動")
            # else:
            #     logger.info("虛擬攝像頭已經在運行，跳過重新初始化")
            if not self.camera_initialized:
                logger.warning("虛擬攝像頭啟動失敗")

            # 設置默認頭像圖片（只有在攝像頭成功啟動時才設置）
            if self.camera_initialized:
                if not self.virtual_webcam.set_default_avatar_image(self.avatar_image_path):
                    logger.warning("設置默認頭像圖片失敗")

            # 允許在虛擬攝像頭失敗時仍然初始化成功，只要基本數據驗證通過
            # 至少需要有效的圖片和音頻數據
            self.avatar_initialized = True  # 基本初始化成功
            
            if not self.microphone_initialized:
                logger.warning("頭像初始化成功，但虛擬麥克風不可用")
            if not self.camera_initialized:
                logger.warning("頭像初始化成功，但虛擬攝像頭不可用")
            
            return self.avatar_initialized
            
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
            
            # 檢查虛擬設備狀態
            mic_available = hasattr(self, 'microphone_initialized') and self.microphone_initialized
            cam_available = hasattr(self, 'camera_initialized') and self.camera_initialized
            
            if not mic_available and not cam_available:
                logger.warning("虛擬麥克風和攝像頭都不可用，跳過說話功能")
                return False
            
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
            
            # 2. 如果攝像頭可用且 Wav2Lip 服務可用，使用 Wav2Lip 生成對嘴視頻
            video_frames = []
            if cam_available and self.wav2lip_service is not None:
                try:
                    logger.debug("生成 Wav2Lip 視頻...")
                    wav2lip_request = model_service_pb2.Wav2LipRequest(
                        audio_data=tts_response.generated_audio,
                        image_data=self.avatar_image
                    )
                    
                    wav2lip_response = self.wav2lip_service.Wav2Lip(wav2lip_request, MockContext())
                    
                    if not wav2lip_response.video_data:
                        logger.warning("Wav2Lip 生成視頻失敗，將僅播放音頻")
                    else:
                        logger.debug("從 Wav2Lip 響應中提取視頻幀...")
                        video_frames = self._extract_video_frames(wav2lip_response.video_data)
                except Exception as e:
                    logger.warning(f"Wav2Lip 處理失敗: {e}，將僅播放音頻")
                    video_frames = []
            else:
                if not cam_available:
                    logger.info("虛擬攝像頭不可用，跳過 Wav2Lip 視頻生成，僅播放音頻")
                elif self.wav2lip_service is None:
                    logger.info("Wav2Lip 服務不可用，跳過視頻生成，僅播放音頻")
            
            # 3. 解析音頻
            logger.debug("解析生成的音頻...")
            audio_io = io.BytesIO(tts_response.generated_audio)
            audio_data, audio_sr = sf.read(audio_io)
            
            # 4. 播放媒體
            logger.debug("開始播放...")
            if video_frames:
                logger.info(f"準備播放 {len(video_frames)} 幀視頻，音頻長度: {len(audio_data)/audio_sr:.2f}秒")
            else:
                logger.info(f"準備播放音頻，長度: {len(audio_data)/audio_sr:.2f}秒")
            
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
        temp_video_path = None
        try:
            if not video_data:
                logger.error("視頻數據為空")
                return []
                
            # 保存視頻到臨時文件
            with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as temp_video:
                temp_video.write(video_data)
                temp_video_path = temp_video.name
            
            # 驗證文件是否成功創建
            if not os.path.exists(temp_video_path) or os.path.getsize(temp_video_path) == 0:
                logger.error("臨時視頻文件創建失敗或為空")
                return []
            
            # 使用 OpenCV 讀取視頻
            cap = cv2.VideoCapture(temp_video_path)
            if not cap.isOpened():
                logger.error(f"無法打開視頻文件: {temp_video_path}")
                return []
            
            # 獲取視頻信息
            source_fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            logger.info(f"源視頻信息: FPS={source_fps}, 總幀數={total_frames}, 尺寸={video_width}x{video_height}")
            
            frames = []
            frame_count = 0
            max_frames = 1000  # 限制最大幀數防止記憶體問題
            
            # 如果源視頻幀率與目標幀率不同，進行幀率轉換
            target_fps = self.virtual_webcam.fps if self.virtual_webcam else 25
            frame_skip = max(1, int(source_fps / target_fps)) if source_fps > target_fps else 1
            
            logger.info(f"幀率轉換: 源FPS={source_fps} -> 目標FPS={target_fps}, 跳幀間隔={frame_skip}")
            logger.info(f"解析度轉換: 源尺寸={video_width}x{video_height} -> 目標尺寸={self.virtual_webcam.width if self.virtual_webcam else 640}x{self.virtual_webcam.height if self.virtual_webcam else 480}")
            
            while frame_count < max_frames:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # 按跳幀間隔提取幀以匹配目標幀率
                if frame_count % frame_skip == 0:
                    # 立即調整幀尺寸以確保一致性
                    target_width = self.virtual_webcam.width if self.virtual_webcam else 640
                    target_height = self.virtual_webcam.height if self.virtual_webcam else 480
                    
                    if frame.shape[:2] != (target_height, target_width):
                        frame = cv2.resize(frame, (target_width, target_height))
                    
                    # 確保數據類型正確
                    if frame.dtype != np.uint8:
                        frame = frame.astype(np.uint8)
                    
                    frames.append(frame)
                
                frame_count += 1
            
            cap.release()
            
            logger.info(f"從視頻中提取了 {len(frames)} 幀 (原始 {frame_count} 幀)")
            return frames
            
        except Exception as e:
            logger.error(f"提取視頻幀失敗: {e}")
            import traceback
            logger.error(f"詳細錯誤: {traceback.format_exc()}")
            return []
        finally:
            # 清理臨時文件
            if temp_video_path and os.path.exists(temp_video_path):
                try:
                    os.unlink(temp_video_path)
                except Exception as e:
                    logger.warning(f"清理臨時視頻文件失敗: {e}")
    
    def _play_synced_media(self, audio_data: np.ndarray, audio_sr: int, video_frames: list):
        """同步播放音頻和視頻"""
        try:
            logger.info(f"開始播放媒體 - 音頻: {len(audio_data)/audio_sr:.2f}秒, 視頻: {len(video_frames)}幀")
            
            # 檢查虛擬設備可用性
            mic_available = (hasattr(self, 'microphone_initialized') and 
                           self.microphone_initialized and 
                           self.virtual_mic and 
                           self.virtual_mic.is_streaming)
            
            cam_available = (hasattr(self, 'camera_initialized') and 
                           self.camera_initialized and 
                           self.virtual_webcam and 
                           self.virtual_webcam.is_streaming)
            
            # 將音頻加入播放隊列
            if mic_available:
                self.virtual_mic.queue_audio(audio_data, audio_sr)
                logger.debug("音頻已加入播放隊列")
            else:
                logger.warning("虛擬麥克風未啟動，跳過音頻播放")
            
            # 將視頻幀加入播放隊列
            if cam_available and video_frames:
                self.virtual_webcam.queue_video_frames(video_frames)
                logger.debug("視頻幀已開始加入播放隊列")
                
                # 計算視頻播放時間並在後台等待視頻播放完畢後重置到默認頭像
                video_duration = len(video_frames) / self.virtual_webcam.fps
                logger.info(f"預計視頻播放時間: {video_duration:.2f}秒")
                
                def reset_after_video():
                    time.sleep(video_duration + 1.0)  # 額外等待1秒確保視頻播放完畢
                    logger.info("視頻播放完畢，重置到默認頭像")
                    if cam_available:  # 再次檢查攝像頭是否可用
                        self.virtual_webcam.reset_to_default_avatar()
                
                # 在背景線程中執行重置
                reset_thread = threading.Thread(target=reset_after_video, daemon=True)
                reset_thread.start()
            else:
                if not cam_available:
                    logger.warning("虛擬攝像頭未啟動，跳過視頻播放")
                elif not video_frames:
                    logger.info("沒有視頻幀需要播放")
            
            if mic_available or (cam_available and video_frames):
                logger.info("媒體播放啟動成功")
            else:
                logger.warning("沒有可用的虛擬設備進行媒體播放")
            
        except Exception as e:
            logger.error(f"媒體播放失敗: {e}")
            import traceback
            logger.error(f"詳細錯誤: {traceback.format_exc()}")
    
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
        
        if hasattr(self, 'virtual_mic') and self.virtual_mic:
            self.virtual_mic.stop_streaming()
            
        if hasattr(self, 'virtual_webcam') and self.virtual_webcam:
            self.virtual_webcam.stop_streaming()
        
        self._cleanup_temp_files()
        
        # 重置初始化標誌
        self.avatar_initialized = False
        self.microphone_initialized = False
        self.camera_initialized = False
        
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
