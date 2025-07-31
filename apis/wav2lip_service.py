import grpc
from proto import model_service_pb2, model_service_pb2_grpc
import os
import subprocess

def validate_file_format(file_path, expected_formats):
    if not any(file_path.endswith(fmt) for fmt in expected_formats):
        raise ValueError(f"檔案格式不正確，期望 {expected_formats}，但得到 {file_path}")

class Wav2LipServicer(model_service_pb2_grpc.MediaServiceServicer):
    def __init__(self, model_path="models/wav2lip.onnx"):
        self.model_path = model_path

    def Wav2Lip(self, request, context):

        audio_path = "temp_audio.wav"
        video_path = "temp_video.mp4"
        output_video_path = "output_lip_sync.mp4"

        # 將 bytes 寫入臨時檔案
        with open(audio_path, "wb") as audio_file:
            audio_file.write(request.audio_data)
        with open(video_path, "wb") as video_file:
            video_file.write(request.image_data)

        try:
            validate_file_format(audio_path, [".wav"])
            validate_file_format(video_path, [".png", ".jpg"])

            # 執行 Wav2Lip 模型
            command = [
                "python", "run_wav2lip.py",
                "--model", self.model_path,
                "--audio", audio_path,
                "--video", video_path,
                "--output", output_video_path
            ]
            subprocess.run(command, check=True)

            # 讀取生成的影片並返回
            with open(output_video_path, "rb") as output_file:
                video_data = output_file.read()

            return model_service_pb2.Wav2LipResponse(video_data=video_data)

        except Exception as e:
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"生成對嘴影片失敗: {e}")
            return model_service_pb2.Wav2LipResponse()

        finally:
            # 清理臨時檔案
            os.remove(audio_path)
            os.remove(video_path)
            if os.path.exists(output_video_path):
                os.remove(output_video_path)

