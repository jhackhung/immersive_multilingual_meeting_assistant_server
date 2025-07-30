import grpc
import torch
import numpy as np
import os
import time
import onnxruntime as ort
import io
import tempfile
from scipy.io.wavfile import write as write_wav

from proto import model_service_pb2
from proto import model_service_pb2_grpc

from TTS.api import TTS

print("🚀 正在初始化 TTS 服務...")
FIXED_MEL_CHUNK_LENGTH = 100
DEFAULT_SPEAKER_WAV_PATH = "./tts_sample/segment.wav"
ONNX_MODEL_PATH = "../model/hifigan.onnx"
SAMPLE_RATE = 22050

if not os.path.exists(DEFAULT_SPEAKER_WAV_PATH):
    raise FileNotFoundError(f"❌ 找不到預設參考音訊檔案: {DEFAULT_SPEAKER_WAV_PATH}。")
if not os.path.exists(ONNX_MODEL_PATH):
    raise FileNotFoundError(f"❌ 找不到優化後的 ONNX 模型: {ONNX_MODEL_PATH}。")

# --- 設定運算設備 ---
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"✅ TTS 服務使用裝置: {device}")

# --- 載入 XTTS-v2 模型 ---
print("⏳ 正在載入 XTTS-v2 模型...")
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)
tts_model = tts.synthesizer.tts_model
print("✅ XTTS-v2 模型載入成功。")

# --- 載入優化後的 ONNX 聲碼器模型 ---
print("⏳ 正在載入優化後的 ONNX 聲碼器...")

if device == 'cuda':
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
else:
    providers = ['CPUExecutionProvider']
    
onnx_session = ort.InferenceSession(ONNX_MODEL_PATH, providers=providers)
print(f"✅ ONNX 聲碼器載入成功，使用 provider: {onnx_session.get_providers()}")

print("✅ TTS 服務初始化完成。")

class TtsServicer(model_service_pb2_grpc.TtsServiceServicer):
    """
    實現 tts.proto 中定義的 TtsService。
    """
    def Tts(self, request, context):
        """
        處理 TTS 請求並回傳生成的音訊。
        'request' 是一個 tts_pb2.TtsRequest 物件。
        必須回傳一個 tts_pb2.TtsResponse 物件。
        """
        start_time = time.time()
        speaker_wav_path = DEFAULT_SPEAKER_WAV_PATH
        temp_file_path = None

        try:
            # --- 步驟 1: 處理參考音訊 ---
            # gRPC 直接傳輸 bytes，如果提供了 reference_audio，我們將其寫入暫存檔
            if request.reference_audio:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
                    tmp_file.write(request.reference_audio)
                    temp_file_path = tmp_file.name
                    speaker_wav_path = temp_file_path
                print("🎤 使用了客戶端提供的參考音訊。")
            else:
                print(f"🎤 使用預設的參考音訊: {DEFAULT_SPEAKER_WAV_PATH}")

            # --- 步驟 2: 生成梅爾頻譜 (核心邏輯與之前相同) ---
            text_to_speak = request.text_to_speak
            print(f"📝 準備生成文字: '{text_to_speak[:30]}...'")
            language = request.language if request.language else "en"
            gpt_cond_latent, speaker_embedding = tts_model.get_conditioning_latents(audio_path=speaker_wav_path)
            gpt_cond_latent = gpt_cond_latent.to(device)
            speaker_embedding = speaker_embedding.to(device)

            text_tokens = torch.IntTensor(tts_model.tokenizer.encode(text_to_speak, lang=language)).unsqueeze(0).to(device)

            with torch.no_grad():
                gpt_codes = tts_model.gpt.generate(
                    cond_latents=gpt_cond_latent, text_inputs=text_tokens,
                    output_attentions=False, temperature=0.75, top_p=0.9,
                    repetition_penalty=5.0, length_penalty=1.5
                )
                expected_output_len = torch.tensor([gpt_codes.shape[-1] * tts_model.gpt.code_stride_len], device=device)
                text_len = torch.tensor([text_tokens.shape[-1]], device=device)
                gpt_latents = tts_model.gpt(
                    text_tokens, text_len, gpt_codes, expected_output_len,
                    cond_latents=gpt_cond_latent, return_attentions=False, return_latent=True
                )
            
            mel_spectrogram_tensor = gpt_latents.detach()

            # --- 步驟 3: 使用 ONNX 聲碼器進行分塊推論 ---
            full_mel_spectrogram_np = mel_spectrogram_tensor.cpu().numpy().astype(np.float32)
            speaker_embedding_np = speaker_embedding.cpu().numpy().astype(np.float32)

            final_audio_waveform = []
            num_chunks = (full_mel_spectrogram_np.shape[1] + FIXED_MEL_CHUNK_LENGTH - 1) // FIXED_MEL_CHUNK_LENGTH
            
            for i in range(num_chunks):
                start_idx = i * FIXED_MEL_CHUNK_LENGTH
                end_idx = min((i + 1) * FIXED_MEL_CHUNK_LENGTH, full_mel_spectrogram_np.shape[1])
                mel_chunk = full_mel_spectrogram_np[:, start_idx:end_idx, :]
                current_chunk_length = mel_chunk.shape[1]

                if current_chunk_length < FIXED_MEL_CHUNK_LENGTH:
                    padding_needed = FIXED_MEL_CHUNK_LENGTH - current_chunk_length
                    mel_chunk_padded = np.pad(mel_chunk, ((0, 0), (0, padding_needed), (0, 0)), mode='constant', constant_values=0)
                else:
                    mel_chunk_padded = mel_chunk

                onnx_inputs = {"mel_spectrogram": mel_chunk_padded, "speaker_embedding": speaker_embedding_np}
                chunk_output = onnx_session.run(None, onnx_inputs)[0]
                final_audio_waveform.append(chunk_output.squeeze())

            # --- 步驟 4: 拼接音訊並轉換為 bytes ---
            if not final_audio_waveform:
                context.set_code(grpc.StatusCode.INTERNAL)
                context.set_details("音訊生成失敗，沒有任何音訊塊產生。")
                return model_service_pb2.TtsResponse()

            final_audio_waveform = np.concatenate(final_audio_waveform)
            
            buffer = io.BytesIO()
            write_wav(buffer, SAMPLE_RATE, final_audio_waveform.astype(np.float32))
            wav_bytes = buffer.getvalue()
            
            end_time = time.time()
            print(f"✅ 請求處理完成，總耗時: {end_time - start_time:.2f} 秒。")
            
            # --- 步驟 5: 回傳 gRPC 回應 ---
            return model_service_pb2.TtsResponse(generated_audio=wav_bytes)

        except Exception as e:
            print(f"❌ 處理請求時發生錯誤: {e}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"內部伺服器錯誤: {str(e)}")
            return model_service_pb2.TtsResponse()
        
        finally:
            # --- 清理暫存檔案 ---
            if temp_file_path and os.path.exists(temp_file_path):
                os.unlink(temp_file_path)
                print(f"🗑️ 已刪除暫存檔案: {temp_file_path}")