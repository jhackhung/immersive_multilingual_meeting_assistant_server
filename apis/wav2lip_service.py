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
            
            # 使用 PyTorch 模型進行推理
            result_video_path = self.wav2lip_model.inference(
                image_path=temp_image_path,
                audio_path=temp_audio_path,
                output_path=output_video_path
            )
            
            print("⏳ 使用 ffmpeg 合併音訊和影片...")
            
            # 檢查 ffmpeg 是否可用
            try:
                subprocess.run(['ffmpeg', '-version'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
                ffmpeg_available = True
            except (subprocess.CalledProcessError, FileNotFoundError):
                ffmpeg_available = False
                print("⚠️ ffmpeg 不可用，返回無音訊影片")
            
            if ffmpeg_available:
                # 使用 ffmpeg 合併音訊
                command = [
                    'ffmpeg', '-y',
                    '-i', result_video_path,
                    '-i', temp_audio_path,
                    '-c:v', 'libx264',
                    '-c:a', 'aac',
                    '-strict', 'experimental',
                    '-shortest',
                    final_output_path
                ]
                
                result = subprocess.run(command, capture_output=True, text=True)
                if result.returncode == 0:
                    output_path = final_output_path
                    print("✅ ffmpeg 音訊合併成功")
                else:
                    print(f"⚠️ ffmpeg 合併失敗: {result.stderr}")
                    output_path = result_video_path
            else:
                output_path = result_video_path

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
                        os.remove(path)
                    except Exception as e:
                        print(f"⚠️ 無法刪除臨時檔案 {path}: {e}")