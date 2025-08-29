# apis/pyannote.py (最終雲端載入版)

import os
import torch
import numpy as np
from typing import List, Tuple, Optional
from dotenv import load_dotenv
import subprocess
import tempfile
import librosa
import soundfile as sf
import io
import logging

from pyannote.audio.core.pipeline import Pipeline
from pyannote.core import Annotation

logger = logging.getLogger(__name__)

class OfficialRealtimeDiarizer:
    def __init__(self, clustering_threshold: float = 0.7):
        print("正在初始化官方 Pipeline (雲端載入模式)...")
        
        # --- 1. 從 .env 檔案安全地載入 Hugging Face Token ---
        load_dotenv()
        self.hf_token = os.getenv("HUGGINGFACE_TOKEN")
        if not self.hf_token:
            raise ValueError("Hugging Face Token 未設定！請在專案根目錄的 .env 檔案中設定。")

        # --- 2. 設定運算設備 ---
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"將使用設備: {self.device}")

        # --- 3. 從 Hugging Face Hub 載入預訓練管線 ---
        print("正在從 Hugging Face Hub 載入 Speaker Diarization Pipeline...")
        self.pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=self.hf_token
        ).to(self.device)
        
        # 使用正確的方式修改參數
        self.pipeline.instantiate({
            "clustering": {"threshold": clustering_threshold},
        })
        print(f"✅ Pipeline 載入完成，聚類閾值設定為: {clustering_threshold}")

        # --- 4. 狀態管理 ---
        self.sample_rate = 16000
        self.audio_buffer = np.array([], dtype=np.float32)

        # --- 5. 音訊處理設定 ---
        self.ffmpeg_available = self._check_ffmpeg()
        if self.ffmpeg_available:
            print("✅ FFmpeg 可用，支援更多音訊格式")
        else:
            print("⚠️ FFmpeg 不可用，使用 librosa 作為備用方案")

        print("✅ 官方即時講者分辨器已就緒。")

    def _check_ffmpeg(self) -> bool:
        """檢查系統是否安裝了 ffmpeg"""
        try:
            result = subprocess.run(['ffmpeg', '-version'], 
                                  capture_output=True, timeout=5)
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.SubprocessError):
            return False

    def _detect_audio_format(self, audio_data: bytes) -> str:
        """檢測音訊格式"""
        if len(audio_data) < 12:
            return 'unknown'
            
        # WAV 檔案標識
        if audio_data.startswith(b'RIFF') and b'WAVE' in audio_data[:12]:
            return 'wav'
        # MP3 檔案標識
        elif audio_data.startswith(b'ID3') or (len(audio_data) >= 2 and audio_data[0:2] == b'\xff\xfb'):
            return 'mp3'
        # MP4/M4A 檔案標識
        elif b'ftyp' in audio_data[:32]:
            return 'mp4'
        # OGG 檔案標識
        elif audio_data.startswith(b'OggS'):
            return 'ogg'
        # FLAC 檔案標識
        elif audio_data.startswith(b'fLaC'):
            return 'flac'
        else:
            return 'unknown'

    def _convert_with_ffmpeg(self, input_file: str, output_file: str) -> bool:
        """使用 ffmpeg 轉換音訊檔案"""
        if not self.ffmpeg_available:
            return False
        
        cmd = [
            'ffmpeg', '-y', '-i', input_file,
            '-ac', '1',  # 單聲道
            '-ar', str(self.sample_rate),  # 採樣率
            '-acodec', 'pcm_s16le',  # 16-bit PCM
            '-f', 'wav',  # WAV 格式
            output_file
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            if result.returncode == 0:
                logger.info(f"FFmpeg 轉換成功: {input_file} -> {output_file}")
                return True
            else:
                logger.error(f"FFmpeg 轉換失敗: {result.stderr}")
                return False
        except subprocess.TimeoutExpired:
            logger.error("FFmpeg 轉換超時")
            return False
        except Exception as e:
            logger.error(f"FFmpeg 轉換錯誤: {e}")
            return False

    def _convert_with_librosa(self, input_file: str, output_file: str) -> bool:
        """使用 librosa 轉換音訊檔案（備用方案）"""
        try:
            # 使用 librosa 載入音訊
            audio, sr = librosa.load(input_file, sr=self.sample_rate, mono=True)
            
            # 保存為 WAV
            sf.write(output_file, audio, sr)
            logger.info(f"Librosa 轉換成功: {input_file} -> {output_file}")
            return True
        except Exception as e:
            logger.error(f"Librosa 轉換失敗: {e}")
            return False

    def _load_audio_from_bytes(self, audio_data: bytes) -> Optional[np.ndarray]:
        """
        將音訊 bytes 轉換為 numpy 陣列
        支援多種格式，優先使用 ffmpeg
        """
        try:
            # 檢測音訊格式
            format_type = self._detect_audio_format(audio_data)
            logger.info(f"檢測到音訊格式: {format_type}")
            
            # 創建臨時輸入檔案
            with tempfile.NamedTemporaryFile(suffix=f'.{format_type}', delete=False) as temp_input:
                temp_input.write(audio_data)
                temp_input_path = temp_input.name
            
            # 創建臨時輸出檔案
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_output:
                temp_output_path = temp_output.name
            
            try:
                audio = None
                
                # 方法 1: 如果已經是 WAV 格式，嘗試直接載入
                if format_type == 'wav':
                    try:
                        audio, sr = librosa.load(temp_input_path, sr=self.sample_rate, mono=True)
                        logger.info("直接載入 WAV 檔案成功")
                        return audio
                    except Exception as e:
                        logger.warning(f"直接載入 WAV 失敗，嘗試轉換: {e}")
                
                # 方法 2: 優先使用 ffmpeg 轉換
                if self.ffmpeg_available and format_type != 'wav':
                    if self._convert_with_ffmpeg(temp_input_path, temp_output_path):
                        try:
                            audio, sr = librosa.load(temp_output_path, sr=self.sample_rate, mono=True)
                            logger.info("FFmpeg 轉換載入成功")
                            return audio
                        except Exception as e:
                            logger.warning(f"FFmpeg 轉換後載入失敗: {e}")
                
                # 方法 3: 使用 librosa 轉換
                if self._convert_with_librosa(temp_input_path, temp_output_path):
                    try:
                        audio, sr = librosa.load(temp_output_path, sr=self.sample_rate, mono=True)
                        logger.info("Librosa 轉換載入成功")
                        return audio
                    except Exception as e:
                        logger.warning(f"Librosa 轉換後載入失敗: {e}")
                
                # 方法 4: 嘗試直接用 librosa 載入（最後嘗試）
                try:
                    audio, sr = librosa.load(temp_input_path, sr=self.sample_rate, mono=True)
                    logger.info("Librosa 直接載入成功")
                    return audio
                except Exception as e:
                    logger.error(f"所有載入方法都失敗了: {e}")
                    return None
                
            finally:
                # 清理臨時檔案
                for temp_file in [temp_input_path, temp_output_path]:
                    try:
                        if os.path.exists(temp_file):
                            os.unlink(temp_file)
                    except Exception as e:
                        logger.warning(f"清理臨時檔案失敗: {e}")
                
        except Exception as e:
            logger.error(f"音訊處理失敗: {e}")
            return None

    def _load_audio_from_file(self, file_path: str) -> Optional[np.ndarray]:
        """
        從檔案載入音訊
        支援多種格式，自動選擇最佳轉換方法
        """
        if not os.path.exists(file_path):
            logger.error(f"檔案不存在: {file_path}")
            return None
        
        try:
            # 直接嘗試用 librosa 載入
            try:
                audio, sr = librosa.load(file_path, sr=self.sample_rate, mono=True)
                logger.info(f"直接載入檔案成功: {file_path}")
                return audio
            except Exception as e:
                logger.warning(f"直接載入失敗，嘗試轉換: {e}")
            
            # 如果直接載入失敗，讀取檔案內容並用 bytes 方法處理
            with open(file_path, 'rb') as f:
                audio_data = f.read()
            
            return self._load_audio_from_bytes(audio_data)
            
        except Exception as e:
            logger.error(f"載入檔案失敗: {e}")
            return None

    def process(self, audio_chunk: np.ndarray) -> List[Tuple[str, float, float]]:
        """累積傳入的音訊塊"""
        if audio_chunk is None or len(audio_chunk) == 0:
            logger.warning("收到空的音訊塊")
            return []
        
        self.audio_buffer = np.concatenate([self.audio_buffer, audio_chunk])
        logger.debug(f"音訊緩衝區大小: {len(self.audio_buffer)} 採樣點")
        return []

    def process_file(self, file_path: str) -> List[Tuple[str, float, float]]:
        """
        處理整個音訊檔案
        便利方法，一次性載入並處理檔案
        """
        logger.info(f"處理音訊檔案: {file_path}")
        
        # 載入音訊
        audio = self._load_audio_from_file(file_path)
        if audio is None:
            logger.error("音訊載入失敗")
            return []
        
        logger.info(f"音訊載入成功，長度: {len(audio)} 採樣點，時長: {len(audio)/self.sample_rate:.2f} 秒")
        
        # 重置狀態
        self.reset()
        
        # 處理音訊
        self.process(audio)
        
        # 獲取結果
        return self.flush()

    def process_bytes(self, audio_data: bytes) -> List[Tuple[str, float, float]]:
        """
        處理音訊 bytes 數據
        便利方法，用於 gRPC 服務
        """
        logger.info(f"處理音訊數據: {len(audio_data)} bytes")
        
        # 載入音訊
        audio = self._load_audio_from_bytes(audio_data)
        if audio is None:
            logger.error("音訊載入失敗")
            return []
        
        logger.info(f"音訊載入成功，長度: {len(audio)} 採樣點，時長: {len(audio)/self.sample_rate:.2f} 秒")
        
        # 重置狀態
        self.reset()
        
        # 處理音訊
        self.process(audio)
        
        # 獲取結果
        return self.flush()

    def flush(self) -> List[Tuple[str, float, float]]:
        """處理整個音訊緩衝區並回傳最終結果"""
        if len(self.audio_buffer) == 0:
            logger.warning("音訊緩衝區為空")
            return []

        logger.info(f"正在處理累積的音訊緩衝區 ({len(self.audio_buffer)} 採樣點, {len(self.audio_buffer)/self.sample_rate:.2f} 秒)...")
        
        try:
            # 轉換為 PyTorch tensor
            waveform = torch.from_numpy(self.audio_buffer).unsqueeze(0).to(self.device)
            
            # 執行語者辨識
            diarization = self.pipeline({
                "waveform": waveform, 
                "sample_rate": self.sample_rate
            })

            # 提取結果
            final_segments = []
            for segment, _, speaker in diarization.itertracks(yield_label=True):
                final_segments.append((speaker, segment.start, segment.end))
            
            logger.info(f"語者辨識完成，找到 {len(final_segments)} 個片段")
            
            # 重置狀態
            self.reset()
            
            return final_segments
            
        except Exception as e:
            logger.error(f"語者辨識處理失敗: {e}")
            self.reset()
            return []

    def reset(self):
        """重置所有狀態"""
        self.audio_buffer = np.array([], dtype=np.float32)
        logger.debug("✅ 狀態已重置。")

    def get_supported_formats(self) -> List[str]:
        """獲取支援的音訊格式列表"""
        base_formats = ['wav', 'mp3', 'flac', 'ogg']
        
        if self.ffmpeg_available:
            # ffmpeg 支援更多格式
            additional_formats = ['mp4', 'm4a', 'aac', 'wma', 'avi', 'mov', 'mkv']
            return base_formats + additional_formats
        else:
            return base_formats

    def get_system_info(self) -> dict:
        """獲取系統資訊"""
        return {
            "device": str(self.device),
            "sample_rate": self.sample_rate,
            "ffmpeg_available": self.ffmpeg_available,
            "supported_formats": self.get_supported_formats(),
            "buffer_size": len(self.audio_buffer),
            "buffer_duration": len(self.audio_buffer) / self.sample_rate if len(self.audio_buffer) > 0 else 0
        }