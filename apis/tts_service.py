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

class TtsServicer(model_service_pb2_grpc.MediaServiceServicer):
    """
    實現 .proto 中定義的 TtsService。
    這個類別將模型載入和推論邏輯封裝在一起。
    """
    
    def __init__(self, 
                 onnx_model_path="./models/hifigan_decoder.onnx", 
                 default_speaker_wav="./tts_sample/segment.wav"):
        """
        初始化 TtsServicer，載入所有必要的模型和設定。
        這個方法只會在伺服器啟動、建立此類別實例時執行一次。
        
        Args:
            onnx_model_path (str): ONNX 聲碼器模型的路徑。
            default_speaker_wav (str): 預設參考音訊的路徑。
        """

        print("🚀 正在初始化 TTS 服務...")
        self.onnx_model_path = onnx_model_path
        self.default_speaker_wav_path = default_speaker_wav
        self.sample_rate = 22050
        self.fixed_mel_chunk_length = 100

        if not os.path.exists(self.default_speaker_wav_path):
            raise FileNotFoundError(f"❌ 找不到預設參考音訊檔案: {self.default_speaker_wav_path}")
        if not os.path.exists(self.onnx_model_path):
            raise FileNotFoundError(f"❌ 找不到優化後的 ONNX 模型: {self.onnx_model_path}")

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"✅ TTS 服務使用裝置: {self.device}")    

        # --- 載入 XTTS-v2 模型 ---
        print("⏳ 正在載入 XTTS-v2 模型...")
        tts_instance = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(self.device)
        self.tts_model = tts_instance.synthesizer.tts_model
        print("✅ XTTS-v2 模型載入成功。")

        # --- 載入優化後的 ONNX 聲碼器模型 ---
        print("⏳ 正在載入優化後的 ONNX 聲碼器...")
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if self.device == 'cuda' else ['CPUExecutionProvider']
        try:
            self.onnx_session = ort.InferenceSession(self.onnx_model_path, providers=providers)
            print(f"✅ ONNX 聲碼器載入成功，使用 provider: {self.onnx_session.get_providers()}")
        except Exception as e:
            raise RuntimeError(f"❌ 載入 ONNX 模型失敗: {e}")
        
        print("✅ TTS 服務初始化完成。")

    def Tts(self, request, context):
        """
        處理 TTS 請求並回傳生成的音訊。
        'request' 是一個 tts_pb2.TtsRequest 物件。
        必須回傳一個 tts_pb2.TtsResponse 物件。
        """
        start_time = time.time()
        speaker_wav_path = self.default_speaker_wav_path
        temp_file_path = None

        try:
            # --- 處理參考音訊 ---
            if request.reference_audio:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
                    tmp_file.write(request.reference_audio)
                    temp_file_path = tmp_file.name
                    speaker_wav_path = temp_file_path
                print("🎤 使用了客戶端提供的參考音訊。")
            else:
                print(f"🎤 使用預設的參考音訊: {self.default_speaker_wav_path}")
                
            # --- 生成梅爾頻譜 ---
            text_to_speak = request.text_to_speak
            language = request.language or "en"
            print(f"📝 準備生成文字 (語言: {language}): '{text_to_speak[:30]}...'")
            
            # 使用 self.tts_model 和 self.device
            gpt_cond_latent, speaker_embedding = self.tts_model.get_conditioning_latents(audio_path=speaker_wav_path)
            gpt_cond_latent = gpt_cond_latent.to(self.device)
            speaker_embedding = speaker_embedding.to(self.device)

            text_tokens = torch.IntTensor(self.tts_model.tokenizer.encode(text_to_speak, lang=language)).unsqueeze(0).to(self.device)


            with torch.no_grad():
                gpt_codes = self.tts_model.gpt.generate(
                    cond_latents=gpt_cond_latent, text_inputs=text_tokens,
                    output_attentions=False, temperature=0.75, top_p=0.9,
                    repetition_penalty=5.0, length_penalty=1.5
                )
                expected_output_len = torch.tensor([gpt_codes.shape[-1] * self.tts_model.gpt.code_stride_len], device=self.device)
                text_len = torch.tensor([text_tokens.shape[-1]], device=self.device)
                gpt_latents = self.tts_model.gpt(
                    text_tokens, text_len, gpt_codes, expected_output_len,
                    cond_latents=gpt_cond_latent, return_attentions=False, return_latent=True
                )
            
            mel_spectrogram_tensor = gpt_latents.detach()

            # --- 步驟 3: 使用 ONNX 聲碼器進行分塊推論 ---
            full_mel_spectrogram_np = mel_spectrogram_tensor.cpu().numpy().astype(np.float32)
            speaker_embedding_np = speaker_embedding.cpu().numpy().astype(np.float32)

            final_audio_waveform = []
            num_chunks = (full_mel_spectrogram_np.shape[1] + self.fixed_mel_chunk_length - 1) // self.fixed_mel_chunk_length
            
            for i in range(num_chunks):
                start_idx = i * self.fixed_mel_chunk_length
                end_idx = min((i + 1) * self.fixed_mel_chunk_length, full_mel_spectrogram_np.shape[1])
                mel_chunk = full_mel_spectrogram_np[:, start_idx:end_idx, :]
                current_chunk_length = mel_chunk.shape[1]

                if current_chunk_length < self.fixed_mel_chunk_length:
                    padding_needed = self.fixed_mel_chunk_length - current_chunk_length
                    mel_chunk_padded = np.pad(mel_chunk, ((0, 0), (0, padding_needed), (0, 0)), mode='constant', constant_values=0)
                else:
                    mel_chunk_padded = mel_chunk

                onnx_inputs = {"mel_spectrogram": mel_chunk_padded, "speaker_embedding": speaker_embedding_np}
                # 使用 self.onnx_session
                chunk_output = self.onnx_session.run(None, onnx_inputs)[0]
                final_audio_waveform.append(chunk_output.squeeze())

            # --- 拼接音訊並轉換為 bytes ---
            if not final_audio_waveform:
                context.set_code(grpc.StatusCode.INTERNAL)
                context.set_details("音訊生成失敗，沒有任何音訊塊產生。")
                return model_service_pb2.TtsResponse()

            final_audio_waveform = np.concatenate(final_audio_waveform)
            
            buffer = io.BytesIO()
            write_wav(buffer, self.sample_rate, final_audio_waveform.astype(np.float32))
            wav_bytes = buffer.getvalue()
            
            end_time = time.time()
            print(f"✅ 請求處理完成，總耗時: {end_time - start_time:.2f} 秒。")
            
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