# apis/pyannote.py (最終雲端載入版)

import os
import torch
import numpy as np
from typing import List, Tuple
from dotenv import load_dotenv

from pyannote.audio.core.pipeline import Pipeline
from pyannote.core import Annotation

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
        # 首次執行會下載模型，之後會從本地快取載入
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

        print("✅ 官方即時講者分辨器已就緒。")

    def process(self, audio_chunk: np.ndarray) -> List[Tuple[str, float, float]]:
        """累積傳入的音訊塊"""
        self.audio_buffer = np.concatenate([self.audio_buffer, audio_chunk])
        return []

    def flush(self) -> Annotation: # <--- 修改點 1: 改變回傳型別提示
        """處理整個音訊緩衝區並回傳最終結果的 Annotation 物件"""
        if len(self.audio_buffer) == 0:
            return Annotation() # 回傳一個空的 Annotation

        print("正在處理累積的音訊緩衝區...")
        waveform = torch.from_numpy(self.audio_buffer).unsqueeze(0).to(self.device)
        
        diarization = self.pipeline({"waveform": waveform, "sample_rate": self.sample_rate})

        self.reset()
        return diarization # <--- 修改點 2: 直接回傳整個 diarization 物件

    def reset(self):
        """重置所有狀態"""
        self.audio_buffer = np.array([], dtype=np.float32)
        print("✅ 狀態已重置。")