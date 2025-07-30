from pyannote.audio import Pipeline
from typing import Dict, List, Tuple
import torch
import os                 # <--- 1. 匯入 os
from dotenv import load_dotenv # <--- 2. 匯入 load_dotenv

class SpeakerDiarization:
    def __init__(self): # <--- 3. 移除 access_token 參數
        """
        初始化說話者分辨系統。
        會自動從 .env 檔案中讀取 HUGGING_FACE_TOKEN。
        """
        # 4. 載入 .env 檔案中的環境變數
        load_dotenv()

        # 5. 從環境變數中取得 access token
        access_token = os.getenv("HUGGING_FACE_TOKEN")

        # 6. 檢查 token 是否存在
        if not access_token:
            raise ValueError(
                "錯誤：未能在環境變數中找到 HUGGING_FACE_TOKEN。\n"
                "請確認您的專案根目錄下有 .env 檔案，且內容包含 HUGGING_FACE_TOKEN='your_token'。"
            )

        self.pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=access_token  # <--- 7. 使用讀取到的 token
        )
        
        # 如果有GPU就使用GPU
        if torch.cuda.is_available():
            print("偵測到 CUDA，將 pyannote 模型移至 GPU。")
            self.pipeline = self.pipeline.to(torch.device("cuda"))
        else:
            print("未偵測到 CUDA，pyannote 模型將使用 CPU。")

    def process_audio(self, audio_file: str) -> List[Dict[str, any]]:
        """
        處理音訊檔案並進行說話者分辨
        """
        # ( ... 以下程式碼維持不變 ... )
        diarization = self.pipeline(audio_file)
        
        results = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            results.append({
                'start': turn.start,
                'end': turn.end,
                'speaker': speaker
            })
            
        return results

    def get_speaker_timeline(self, audio_file: str) -> Dict[str, List[Tuple[float, float]]]:
        """
        獲取每個說話者的時間軸
        """
        # ( ... 以下程式碼維持不變 ... )
        diarization = self.pipeline(audio_file)
        
        speaker_timeline = {}
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            if speaker not in speaker_timeline:
                speaker_timeline[speaker] = []
            speaker_timeline[speaker].append((turn.start, turn.end))
            
        return speaker_timeline