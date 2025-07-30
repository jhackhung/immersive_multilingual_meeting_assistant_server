# apis/pyannote.py

# --- 1. 匯入必要的函式庫 ---
import onnxruntime as rt
import numpy as np
import torchaudio
import torchaudio.transforms as T
import io
from sklearn.cluster import AgglomerativeClustering
from scipy.signal import find_peaks
import logging
from pathlib import Path

# --- 2. 設定日誌 ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- 3. 建立服務類別 ---
class PyannoteService:
    def __init__(self, models_path: Path = Path("models")):
        """
        在服務實例化時，載入所有必要的 ONNX 模型。
        :param models_path: 存放 ONNX 模型的資料夾路徑。
        """
        self.SEG_MODEL_PATH = models_path / "segmentation.onnx"
        self.EMB_MODEL_PATH = models_path / "embedding.onnx"
        
        self.seg_session = None
        self.emb_session = None

        try:
            logger.info(f"正在從 {self.SEG_MODEL_PATH} 載入分割模型...")
            self.seg_session = rt.InferenceSession(str(self.SEG_MODEL_PATH))
            
            logger.info(f"正在從 {self.EMB_MODEL_PATH} 載入嵌入模型...")
            self.emb_session = rt.InferenceSession(str(self.EMB_MODEL_PATH))
            
            logger.info("✅ Pyannote ONNX 模型載入完成！")
        except Exception as e:
            logger.error(f"❌ 模型載入失敗: {e}", exc_info=True)
            
    def diarize(self, audio_bytes: bytes):
        """
        執行說話者日誌分析的核心邏輯。
        接收音訊的 bytes，回傳分析結果的字典。
        """
        if not self.seg_session or not self.emb_session:
            logger.error("模型未成功載入，無法執行分析。")
            return None

        # --- a. 音訊預處理 ---
        try:
            audio_buffer = io.BytesIO(audio_bytes)
            waveform, sample_rate = torchaudio.load(audio_buffer)

            if sample_rate != 16000:
                resampler = T.Resample(orig_freq=sample_rate, new_freq=16000)
                waveform = resampler(waveform)
            
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)

            waveform_np = waveform.numpy().astype(np.float32)
        except Exception as e:
            logger.error(f"音訊預處理失敗: {e}", exc_info=True)
            raise ValueError("無法處理音訊檔案")

        # --- b. 執行「分割模型」---
        CHUNK_DURATION_SAMPLES = 16000 * 5
        if waveform_np.shape[1] > CHUNK_DURATION_SAMPLES:
            logger.warning(f"音訊長度 ({waveform_np.shape[1]/16000:.2f}s) 超過模型輸入尺寸，將截斷為前 5 秒。")
            waveform_np = waveform_np[:, :CHUNK_DURATION_SAMPLES]
        elif waveform_np.shape[1] < CHUNK_DURATION_SAMPLES:
            padding = np.zeros((1, CHUNK_DURATION_SAMPLES - waveform_np.shape[1]), dtype=np.float32)
            waveform_np = np.concatenate([waveform_np, padding], axis=1)

        seg_input = {'input_audio': waveform_np}
        seg_output, = self.seg_session.run(None, seg_input)
        
        # --- c. 從分割結果中找出語音片段 (簡化邏輯) ---
        speech_prob = seg_output[0, :, 0]
        peaks, _ = find_peaks(speech_prob, height=0.5, distance=15)
        
        if len(peaks) == 0:
            return {"message": "未偵測到任何語音活動。"}

        speech_segments = []
        FRAME_SHIFT_S = 0.016
        current_segment = {'start': round(peaks[0] * FRAME_SHIFT_S, 2), 'end': 0}
        for i in range(1, len(peaks)):
            if (peaks[i] - peaks[i-1]) * FRAME_SHIFT_S > 0.5:
                current_segment['end'] = round(peaks[i-1] * FRAME_SHIFT_S, 2)
                speech_segments.append(current_segment)
                current_segment = {'start': round(peaks[i] * FRAME_SHIFT_S, 2), 'end': 0}
        current_segment['end'] = round(peaks[-1] * FRAME_SHIFT_S, 2)
        speech_segments.append(current_segment)
        
        # --- d. 對每個語音片段執行「嵌入模型」---
        embeddings = []
        EMB_CHUNK_SAMPLES = 16000 * 2
        for segment in speech_segments:
            start_sample = int(segment['start'] * 16000)
            end_sample = int(segment['end'] * 16000)
            chunk = waveform_np[:, start_sample:end_sample]

            if chunk.shape[1] < EMB_CHUNK_SAMPLES:
                padding = np.zeros((1, EMB_CHUNK_SAMPLES - chunk.shape[1]), dtype=np.float32)
                chunk = np.concatenate([chunk, padding], axis=1)
            else:
                chunk = chunk[:, :EMB_CHUNK_SAMPLES]
                
            emb_input = {'input_audio': chunk}
            embedding, = self.emb_session.run(None, emb_input)
            embeddings.append(embedding.flatten())

        # --- e. 對嵌入向量進行「聚類」---
        if len(embeddings) < 2:
            if speech_segments:
                speech_segments[0]['speaker'] = 'SPEAKER_00'
            return {"diarization": speech_segments}

        clustering = AgglomerativeClustering(n_clusters=None, distance_threshold=0.8).fit(np.array(embeddings))
        labels = clustering.labels_

        # --- f. 整理結果並回傳 ---
        for i, segment in enumerate(speech_segments):
            segment['speaker'] = f'SPEAKER_{labels[i]:02d}'

        return {"diarization": speech_segments}

