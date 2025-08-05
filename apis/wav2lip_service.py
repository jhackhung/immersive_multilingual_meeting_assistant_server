import grpc
from proto import model_service_pb2, model_service_pb2_grpc
import os
import numpy as np
import onnxruntime as ort
import cv2
import librosa
import tempfile
import subprocess

# --- Constants and Configuration ---
IMG_SIZE = 96 

# --- Helper Functions ---

def get_smoothened_boxes(boxes, T=5):
    """Smooths bounding box detections over a temporal window."""
    for i in range(len(boxes)):
        if i + T > len(boxes):
            window = boxes[len(boxes) - T:]
        else:
            window = boxes[i : i + T]
        boxes[i] = np.mean(window, axis=0)
    return boxes

def face_detect(images, cascade_path):
    """Detects faces in a list of images using a Haar Cascade classifier."""
    cascade = cv2.CascadeClassifier(cascade_path)
    
    results = []
    pady1, pady2, padx1, padx2 = (0, 10, 0, 0)

    for image in images:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = cascade.detectMultiScale(gray, 1.1, 4)

        if len(faces) > 0:
            x, y, w, h = faces[0]
            y1 = max(0, y + pady1)
            y2 = min(image.shape[0], y + h + pady2)
            x1 = max(0, x + padx1)
            x2 = min(image.shape[1], x + w + padx2)
            results.append([x1, y1, x2, y2])
        else:
            results.append([0, 0, image.shape[1], image.shape[0]])

    boxes = np.array(results)
    return get_smoothened_boxes(boxes, T=5)

def crop_and_resize(images, boxes):
    """Crops the face region from images and resizes them."""
    cropped_images = []
    for i, image in enumerate(images):
        x1, y1, x2, y2 = boxes[i]
        cropped = image[y1:y2, x1:x2]
        resized = cv2.resize(cropped, (IMG_SIZE, IMG_SIZE))
        cropped_images.append(resized)
    return np.array(cropped_images)

def get_mel_spectrogram(audio_path):
    """Generates a Mel spectrogram from an audio file."""
    wav = librosa.load(audio_path, sr=16000)[0]
    mel = librosa.feature.melspectrogram(y=wav, sr=16000, n_fft=800, hop_length=200, win_length=800, n_mels=80)
    return mel

# --- gRPC Service Implementation ---

class Wav2LipServicer(model_service_pb2_grpc.MediaServiceServicer):
    def __init__(self, model_path="models/wav2lip.onnx", face_cascade_path="models/haarcascade_frontalface_default.xml"):
        self.model_path = model_path
        self.face_cascade_path = face_cascade_path

        if not os.path.exists(self.face_cascade_path):
            raise FileNotFoundError(f"Haar Cascade face detector not found at: {self.face_cascade_path}")

        print("⏳ Initializing Wav2Lip ONNX model...")
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if ort.get_device() == 'GPU' else ['CPUExecutionProvider']
        try:
            self.onnx_session = ort.InferenceSession(self.model_path, providers=providers)
            print(f"✅ Wav2Lip ONNX model loaded successfully using: {self.onnx_session.get_providers()}")
        except Exception as e:
            raise RuntimeError(f"❌ Failed to load Wav2Lip ONNX model: {e}")

    def Wav2Lip(self, request, context):
        temp_files = []
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio_file, \
                 tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as temp_image_file, \
                 tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as temp_output_video:
                
                temp_audio_path = temp_audio_file.name
                temp_image_path = temp_image_file.name
                output_video_path = temp_output_video.name
                final_output_path = tempfile.mktemp(suffix=".mp4")
                
                temp_files.extend([temp_audio_path, temp_image_path, output_video_path, final_output_path])

            with open(temp_audio_path, "wb") as f:
                f.write(request.audio_data)
            with open(temp_image_path, "wb") as f:
                f.write(request.image_data)

            print("⏳ Pre-processing: Detecting face and preparing image...")
            full_frame = cv2.imread(temp_image_path)
            if full_frame is None:
                raise ValueError("Failed to read the input image.")
            
            face_boxes = face_detect([full_frame], self.face_cascade_path)
            x1, y1, x2, y2 = face_boxes[0]
            face_frame = crop_and_resize([full_frame], face_boxes)[0]
            
            face_frame_gray = cv2.cvtColor(face_frame, cv2.COLOR_BGR2GRAY)
            
            static_face_input = np.stack([face_frame_gray] * 6, axis=0)
            static_face_input = (static_face_input / 127.5) - 1.0
            static_face_input = static_face_input[np.newaxis, ...].astype(np.float32)

            print("⏳ Pre-processing: Generating Mel spectrogram from audio...")
            mel = get_mel_spectrogram(temp_audio_path)
            
            print("⏳ Running Wav2Lip model inference...")
            mel_chunks = []
            i = 0
            while i < mel.shape[1]:
                chunk = mel[:, i:i+16]
                if chunk.shape[1] < 16:
                    chunk = np.pad(chunk, ((0,0), (0, 16 - chunk.shape[1])), mode='constant')
                mel_chunks.append(chunk)
                i += 16

            # *** THE MEMORY FIX: Open video writer before the loop ***
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_video_path, fourcc, 25, (full_frame.shape[1], full_frame.shape[0]))

            for mel_chunk in mel_chunks:
                onnx_inputs = {
                    "audio": mel_chunk[np.newaxis, np.newaxis, ...].astype(np.float32),
                    "face": static_face_input
                }
                output = self.onnx_session.run(None, onnx_inputs)[0]
                
                gen_frame_bgr = np.clip(((output[0] + 1.0) * 127.5), 0, 255).astype(np.uint8)
                gen_frame_bgr = np.transpose(gen_frame_bgr, (1, 2, 0))
                
                # *** THE MEMORY FIX: Create final frame and write immediately ***
                resized_gen = cv2.resize(gen_frame_bgr, (x2-x1, y2-y1))
                final_frame = full_frame.copy()
                final_frame[y1:y2, x1:x2] = resized_gen
                out.write(final_frame)

            # *** THE MEMORY FIX: Release the video writer after the loop ***
            out.release()

            print(f"⏳ Merging audio and video using ffmpeg...")
            command = f'ffmpeg -y -i {output_video_path} -i {temp_audio_path} -c:v copy -c:a aac -strict experimental {final_output_path}'
            subprocess.call(command, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

            with open(final_output_path, "rb") as f:
                final_video_data = f.read()
            
            print("✅ Wav2Lip processing completed successfully.")
            return model_service_pb2.Wav2LipResponse(video_data=final_video_data)

        except Exception as e:
            import traceback
            print(f"❌ An error occurred during Wav2Lip processing: {e}")
            print(f"Traceback: {traceback.format_exc()}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"Failed to generate lip-synced video: {e}")
            return model_service_pb2.Wav2LipResponse()

        finally:
            print("🧹 Cleaning up temporary files...")
            for path in temp_files:
                if os.path.exists(path):
                    os.remove(path)