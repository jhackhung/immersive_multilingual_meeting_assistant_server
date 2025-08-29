import grpc
from proto import model_service_pb2, model_service_pb2_grpc
import os
import numpy as np
import cv2
import librosa
import tempfile
import subprocess
import sys

# 添加模型路徑
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from models.wav2lip_pytorch_model import Wav2LipPytorch

# --- Constants and Configuration ---
IMG_SIZE = 96

# --- gRPC Service Implementation ---

class Wav2LipServicer(model_service_pb2_grpc.MediaServiceServicer):
    def __init__(self, checkpoint_path="models/wav2lip_gan.pth"):
        self.wav2lip_model = Wav2LipPytorch(checkpoint_path)
        print("✅ Wav2Lip PyTorch 服務初始化完成")

    def Wav2Lip(self, request, context):
        temp_files = []
        try:
            # 創建臨時檔案
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio_file, \
                 tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as temp_image_file, \
                 tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as temp_output_video:
                
                temp_audio_path = temp_audio_file.name
                temp_image_path = temp_image_file.name
                output_video_path = temp_output_video.name
                final_output_path = tempfile.mktemp(suffix=".mp4")
                
                temp_files.extend([temp_audio_path, temp_image_path, output_video_path, final_output_path])

            # 寫入請求數據到臨時檔案
            with open(temp_audio_path, "wb") as f:
                f.write(request.audio_data)
            with open(temp_image_path, "wb") as f:
                f.write(request.image_data)

            print("⏳ 開始 Wav2Lip PyTorch 推理...")
            
            # 使用 PyTorch 模型進行推理（不包含音檔）
            result_video_path = self.wav2lip_model.inference(
                image_path=temp_image_path,
                audio_path=temp_audio_path,
                output_path=output_video_path,
                include_audio=False  # 明確設定為不包含音檔
            )
            
            print("📹 生成無聲影片（不附加音檔）...")
            
            # 直接使用生成的無聲影片，不進行音訊合併
            output_path = result_video_path
            print("✅ 無聲影片生成完成")

            # 讀取最終影片資料
            with open(output_path, "rb") as f:
                final_video_data = f.read()
            
            print("✅ Wav2Lip PyTorch 處理完成")
            return model_service_pb2.Wav2LipResponse(video_data=final_video_data)

        except Exception as e:
            import traceback
            error_msg = f"Wav2Lip PyTorch 處理出錯: {e}"
            print(f"❌ {error_msg}")
            print(f"詳細錯誤: {traceback.format_exc()}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(error_msg)
            return model_service_pb2.Wav2LipResponse()

        finally:
            print("🧹 清理臨時檔案...")
            for path in temp_files:
                if os.path.exists(path):
                    try:
                        i=1
                        # os.remove(path)
                    except Exception as e:
                        print(f"⚠️ 無法刪除臨時檔案 {path}: {e}")