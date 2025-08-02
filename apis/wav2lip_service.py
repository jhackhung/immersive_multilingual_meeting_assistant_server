import grpc
from proto import model_service_pb2, model_service_pb2_grpc
import os
import numpy as np
import onnxruntime as ort

def validate_file_format(file_path, expected_formats):
    if not any(file_path.endswith(fmt) for fmt in expected_formats):
        raise ValueError(f"檔案格式不正確，期望 {expected_formats}，但得到 {file_path}")

class Wav2LipServicer(model_service_pb2_grpc.MediaServiceServicer):
    def __init__(self, model_path="models/wav2lip.onnx"):
        self.model_path = model_path

        # 初始化 ONNX Runtime
        print("⏳ 正在載入 Wav2Lip ONNX 模型...")
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if ort.get_device() == 'GPU' else ['CPUExecutionProvider']
        try:
            self.onnx_session = ort.InferenceSession(self.model_path, providers=providers)
            print(f"✅ Wav2Lip ONNX 模型載入成功，使用 provider: {self.onnx_session.get_providers()}")
        except Exception as e:
            raise RuntimeError(f"❌ 載入 Wav2Lip ONNX 模型失敗: {e}")

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

            # 使用 ONNX Runtime 推理
            print("⏳ 正在執行 Wav2Lip 模型推理...")
            onnx_inputs = {
                "audio": np.fromfile(audio_path, dtype=np.float32),
                "video": np.fromfile(video_path, dtype=np.float32)
            }
            output = self.onnx_session.run(None, onnx_inputs)

            # 將結果寫入輸出影片
            with open(output_video_path, "wb") as output_file:
                output_file.write(output[0])

            print("✅ Wav2Lip 模型推理完成，影片已生成。")
            return model_service_pb2.Wav2LipResponse(video_data=output[0])

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

