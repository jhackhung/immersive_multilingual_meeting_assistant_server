import grpc
import torch
import numpy as np
import os
import time
import onnxruntime as ort
import io
import re
import hashlib
import torchaudio
import threading
import logging
from functools import lru_cache
from scipy.io.wavfile import write as write_wav
from typing import Tuple, Optional, List

from proto import model_service_pb2
from proto import model_service_pb2_grpc
from TTS.api import TTS

# 設置結構化日誌
logger = logging.getLogger(__name__)

class TtsServicer(model_service_pb2_grpc.MediaServiceServicer):
    """
    實現 .proto 中定義的 TtsService - NPU 友好優化版本
    """
    
    def __init__(self, 
                 onnx_model_path="./models/hifigan_decoder.onnx", 
                 default_speaker_wav="./tts_sample/segment.wav"):
        """
        初始化 TtsServicer - 修正版本
        """
        logger.info("🚀 正在初始化 NPU 友好 TTS 服務...")
        self.onnx_model_path = onnx_model_path
        self.default_speaker_wav_path = default_speaker_wav
        self.sample_rate = 22050
        self.fixed_mel_chunk_length = 100
        
        # 輸入驗證
        if not os.path.exists(self.default_speaker_wav_path):
            raise FileNotFoundError(f"❌ 找不到預設參考音訊檔案: {self.default_speaker_wav_path}")
        if not os.path.exists(self.onnx_model_path):
            raise FileNotFoundError(f"❌ 找不到優化後的 ONNX 模型: {self.onnx_model_path}")

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger.info(f"✅ TTS 服務使用裝置: {self.device}")

        # --- 載入 XTTS-v2 模型 ---
        logger.info("⏳ 正在載入 XTTS-v2 模型...")
        tts_instance = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(self.device)
        self.tts_model = tts_instance.synthesizer.tts_model
        
        # 獲取真實的 hop_length（修正點 3）
        self.hop_length = getattr(self.tts_model, "hop_length", 256)
        logger.info(f"✅ XTTS-v2 模型載入成功，hop_length: {self.hop_length}")

        # --- 載入優化後的 ONNX 聲碼器模型 (NPU 友好) ---
        logger.info("⏳ 正在載入優化後的 ONNX 聲碼器...")
        self.onnx_session, self.active_providers = self._build_ort_session(self.onnx_model_path)
        logger.info(f"✅ ONNX 聲碼器載入成功，使用 EP: {self.active_providers}")

        # --- 快取系統初始化（修正點 1）---
        self._blob_cache = {}  # 分離儲存音訊 bytes
        
        # --- 設定 overlap 為 hop 的整數倍（修正點 4）---
        self.overlap_samples = 4 * self.hop_length
        logger.info(f"🎵 Overlap samples: {self.overlap_samples}")
        
        # --- 預載預設參考音訊到記憶體 ---
        self._load_default_audio_to_memory()
        
        # --- Warm-up ---
        self._warmup()
        
        # 1. 分句快取 (最重要的優化)
        self._sentence_cache = {}  # 句子級別快取
        self._phrase_cache = {}    # 短語級別快取
        
        # 2. 常用 LLM 回答模式預生成
        self._pregenerate_llm_patterns()
        
        # 3. 快速處理參數
        self.fast_mode_enabled = True
        self.fast_chunk_size = 100      # 更小的分塊用於短句
        self.fast_overlap = self.hop_length * 2  # 減少 overlap
        
        logger.info("✅ NPU 友好 TTS 服務初始化完成。")
    
    def _build_ort_session(self, model_path: str) -> Tuple[ort.InferenceSession, list]:
        """建立 ONNX Runtime Session - 支援多 EP 回退"""
        # SessionOptions 優化
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        so.enable_mem_pattern = True
        so.enable_cpu_mem_arena = True
        so.intra_op_num_threads = max(1, os.cpu_count() // 2)
        
        # 從環境變數讀取 EP 優先序，若未設定則使用預設順序
        custom_ep_order_str = os.environ.get("ONNX_EXECUTION_PROVIDERS")
        
        if custom_ep_order_str:
            logger.info(f"🔧 使用環境變數自訂的 EP 順序: {custom_ep_order_str}")
            ep_candidates = [ep.strip() for ep in custom_ep_order_str.split(',')]
        else:
            # 預設 EP 優先序
            ep_candidates = [
                "QNNExecutionProvider",         # 1. Qualcomm NPU
                "DirectMLExecutionProvider",    # 2. AMD/Windows NPU & GPU
                "CUDAExecutionProvider",        # 3. NVIDIA GPU
                "OpenVINOExecutionProvider",    # 4. Intel CPU/GPU
                "CPUExecutionProvider",         # 5. CPU Fallback
            ]
            logger.info("🔧 使用預設的 EP 順序 (可透過 ONNX_EXECUTION_PROVIDERS 環境變數覆寫)")

        available_providers = ort.get_available_providers()
        # 過濾出當前環境可用的 EP
        providers = [ep for ep in ep_candidates if ep in available_providers]
        
        if not providers:
            logger.warning("⚠️ 找不到任何建議的 EP，強制使用 CPU。")
            providers = ["CPUExecutionProvider"]
        
        logger.debug(f"🔍 可用的 EP: {available_providers}")
        logger.info(f"🎯 選用的 EP 順序: {providers}")
        
        try:
            sess = ort.InferenceSession(model_path, sess_options=so, providers=providers)
            actual_providers = sess.get_providers()
            return sess, actual_providers
        except Exception as e:
            logger.warning(f"⚠️ 使用優先 EP 失敗，回退到 CPU: {e}")
            sess = ort.InferenceSession(model_path, sess_options=so, providers=["CPUExecutionProvider"])
            return sess, ["CPUExecutionProvider"]

    def _load_default_audio_to_memory(self):
        """預載預設參考音訊到記憶體"""
        with open(self.default_speaker_wav_path, "rb") as f:
            self._default_wav_bytes = f.read()
        self._default_hash = self._hash_bytes(self._default_wav_bytes)
        
        # 存入 blob_cache（修正點 1）
        self._blob_cache[self._default_hash] = self._default_wav_bytes
        
        logger.info(f"📁 預設參考音訊已載入記憶體，hash: {self._default_hash[:8]}...")

    def _warmup(self):
        """Warm-up 推論引擎"""
        logger.info("🔥 正在進行 Warm-up...")
        try:
            # 修正 speaker embedding 維度 - 檢測實際維度
            dummy_mel = np.zeros((1, self.fixed_mel_chunk_length, 1024), dtype=np.float32)
            
            # 先獲取實際的 speaker embedding 維度
            temp_gpt_cond_latent, temp_speaker_embedding = self._get_cached_conditioning(self._default_hash)
            actual_spk_dim = temp_speaker_embedding.shape[-1]
            
            dummy_spk = np.zeros((1, 512, actual_spk_dim), dtype=np.float32)
            
            _ = self.onnx_session.run(None, {
                "mel_spectrogram": dummy_mel,
                "speaker_embedding": dummy_spk,
            })
            logger.info("✅ Warm-up 完成")
        except Exception as e:
            logger.warning(f"⚠️ Warm-up 失敗，但不影響服務: {e}")

    @staticmethod
    def _hash_bytes(b: bytes) -> str:
        """計算 bytes 的 SHA256 hash"""
        return hashlib.sha256(b).hexdigest()

    @lru_cache(maxsize=256)
    def _get_cached_conditioning(self, hash_key: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        快取的條件潛變數與說話者嵌入獲取（修正版本）
        修正點 1 & 2: 只用 hash_key 當 key，回傳 CPU tensor
        """
        try:
            # 從 blob_cache 獲取音訊資料
            wav_bytes = self._blob_cache[hash_key]
            
            # In-memory 解析音訊
            buf = io.BytesIO(wav_bytes)
            wav, sr = torchaudio.load(buf, format="wav")
            
            # 重採樣到目標採樣率
            if sr != self.sample_rate:
                wav = torchaudio.functional.resample(wav, sr, self.sample_rate)
            
            # 轉為單聲道
            if wav.shape[0] > 1:
                wav = wav.mean(dim=0, keepdim=True)
            
            # 暫存到記憶體檔案
            tmp_buf = io.BytesIO()
            torchaudio.save(tmp_buf, wav, self.sample_rate, format="wav")
            tmp_buf.seek(0)
            
            # 使用暫存檔案路徑
            import tempfile
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
                tmp_file.write(tmp_buf.getvalue())
                tmp_path = tmp_file.name
            
            try:
                gpt_cond_latent, speaker_embedding = self.tts_model.get_conditioning_latents(audio_path=tmp_path)
                # 修正點 2: 回傳 CPU tensor 以節省 GPU 記憶體
                return gpt_cond_latent.cpu(), speaker_embedding.cpu()
            finally:
                os.unlink(tmp_path)
                
        except Exception as e:
            logger.warning(f"⚠️ 解析參考音訊失敗: {e}")
            # 修正點 2: 避免無限遞迴
            if hash_key != self._default_hash:
                return self._get_cached_conditioning(self._default_hash)
            else:
                # 預設也失敗就拋出異常
                raise RuntimeError(f"無法處理預設參考音訊: {e}")

    def _overlap_add_chunks(self, chunks: list) -> np.ndarray:
        """使用 Hann 窗進行 overlap-add，消除接縫噪音 - 最終穩健版本"""
        if not chunks:
            return np.array([], dtype=np.float32)
        
        if len(chunks) == 1:
            return chunks[0].astype(np.float32)
        
        if self.overlap_samples <= 0:
            return np.concatenate(chunks, axis=0)

        out = chunks[0].astype(np.float32)
        fade = np.hanning(self.overlap_samples * 2).astype(np.float32)
        fade_in, fade_out = fade[:self.overlap_samples], fade[self.overlap_samples:]

        for i in range(1, len(chunks)):
            current_chunk = chunks[i].astype(np.float32)
            
            # 確保有足夠長度進行 overlap
            if len(out) < self.overlap_samples or len(current_chunk) < self.overlap_samples:
                 # 如果其中一個塊太短，直接拼接
                out = np.concatenate([out, current_chunk])
                continue

            # 交疊區域淡入淡出
            out_tail = out[-self.overlap_samples:] * fade_out
            chunk_head = current_chunk[:self.overlap_samples] * fade_in
            mixed = out_tail + chunk_head
            
            # 拼接
            out = np.concatenate([
                out[:-self.overlap_samples], 
                mixed, 
                current_chunk[self.overlap_samples:]
            ])
        
        return out
    
    def _trim_trailing_silence(self, audio: np.ndarray, threshold_db: float = -45.0, chunk_size: int = 1024) -> np.ndarray:
        """
        從音訊尾部修剪靜音部分。

        Args:
            audio (np.ndarray): 輸入的音訊波形 (float32)。
            threshold_db (float): 靜音的音量閾值 (dB)。低於此值被視為靜音。
            chunk_size (int): 用於分析的塊大小。

        Returns:
            np.ndarray: 修剪掉尾部靜音後的音訊波形。
        """
        if audio.size == 0:
            return audio

        # 將 dB 轉換為線性振幅閾值
        threshold = 10 ** (threshold_db / 20)

        # 從後向前迭代，尋找第一個非靜音的塊
        for i in range(len(audio), 0, -chunk_size):
            start = max(0, i - chunk_size)
            chunk = audio[start:i]
            # 檢查塊的最大振幅是否超過閾值
            if np.max(np.abs(chunk)) > threshold:
                # 找到了第一個非靜音塊，我們可以在這裡停止
                # 為了更精確，我們可以在這個塊內找到最後一個超過閾值的樣本
                non_silent_indices = np.where(np.abs(audio[:i]) > threshold)[0]
                if len(non_silent_indices) > 0:
                    last_sample_index = non_silent_indices[-1]
                    # 增加一點緩衝 (例如 100ms)，避免切得太突然
                    padding = int(self.sample_rate * 0.1) 
                    return audio[:last_sample_index + padding]
                else:
                    # 理論上不應該發生，但作為保護
                    return audio[:i]
        
        # 如果整個音訊都是靜音，則回傳空陣列
        return np.array([], dtype=np.float32)

    def _validate_request(self, request) -> Tuple[bool, str]:
        """驗證請求參數"""
        # 文字長度限制
        if not request.text_to_speak or len(request.text_to_speak.strip()) == 0:
            return False, "文字內容不能為空"
        
        if len(request.text_to_speak) > 1000:  # 1000 字元限制
            return False, "文字長度超過限制 (1000 字元)"
        
        # 語言檢查
        supported_languages = ["en", "zh-cn", "zh", "ja", "ko", "fr", "de", "es", "it", "pt", "ru"]
        language = request.language or "en"
        if language not in supported_languages:
            return False, f"不支援的語言: {language}，支援語言: {supported_languages}"
        
        # 參考音訊大小限制 (10MB)
        if request.reference_audio and len(request.reference_audio) > 10 * 1024 * 1024:
            return False, "參考音訊檔案過大 (限制 10MB)"
        
        return True, ""

    def _pregenerate_llm_patterns(self):
        """預生成常用的 LLM 回答模式"""
        
        # LLM 常用開頭和結尾
        llm_patterns = [
            # 開頭
            "Based on", "According to", "In general", "Typically", "Usually",
            "The answer is", "Simply put", "In summary", "To explain",
            "This means", "For example", "However", "Therefore", "Actually",
            
            # 結尾  
            "in conclusion", "to summarize", "overall", "in short",
            "Does this help", "Let me know if", "Is there anything else",
            
            # 常用連接詞
            "First", "Second", "Third", "Next", "Finally", "Also", "Additionally",
            "Moreover", "Furthermore", "On the other hand", "In contrast",
            
            # 確認和澄清
            "I understand", "That's correct", "Exactly", "Precisely", 
            "Not quite", "Actually", "More specifically", "To clarify"
        ]
        
        def bg_pregenerate():
            """背景預生成"""
            try:
                logger.info("🔄 開始預生成 LLM 回答模式...")
                for i, pattern in enumerate(llm_patterns):
                    try:
                        self._fast_generate_sentence(pattern, "en", cache_only=True)
                        if (i + 1) % 5 == 0:
                            logger.debug(f"預生成進度: {i+1}/{len(llm_patterns)}")
                            time.sleep(0.02)  # 避免阻塞
                    except Exception as e:
                        continue
                
                logger.info(f"✅ 預生成完成: {len(self._sentence_cache)} 個模式")
                
            except Exception as e:
                logger.warning(f"預生成失敗: {e}")
        
        # 背景執行
        threading.Thread(target=bg_pregenerate, daemon=True).start()
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """智能分句 - 針對 LLM 回答優化"""
        
        # 第一步：標準句子分割
        sentences = re.split(r'[.!?]+(?:\s+|$)', text.strip())
        sentences = [s.strip() for s in sentences if s.strip()]
        
        # 第二步：合併過短的句子
        merged_sentences = []
        current_chunk = ""
        
        for sentence in sentences:
            # 如果當前塊 + 新句子 < 80 字符，合併
            if len(current_chunk + " " + sentence) < 80:
                current_chunk += (" " + sentence if current_chunk else sentence)
            else:
                # 保存當前塊，開始新塊
                if current_chunk:
                    merged_sentences.append(current_chunk)
                current_chunk = sentence
        
        # 添加最後一塊
        if current_chunk:
            merged_sentences.append(current_chunk)
        
        return merged_sentences
    
    def _get_sentence_cache_key(self, sentence: str, language: str) -> str:
        """生成句子快取鍵"""
        # 標準化句子（移除標點、多餘空格）
        normalized = sentence.lower().strip()
        normalized = re.sub(r'[^\w\s]', '', normalized)  # 移除標點
        normalized = re.sub(r'\s+', ' ', normalized)     # 標準化空格
        
        return f"sent_{normalized}_{language}"
    
    def _fast_generate_sentence(self, sentence: str, language: str, cache_only: bool = False) -> bytes:
        """快速生成單句音頻"""
        cache_key = self._get_sentence_cache_key(sentence, language)
        
        # 檢查快取
        if cache_key in self._sentence_cache:
            return self._sentence_cache[cache_key]
        
        if cache_only:
            return b""
        
        try:
            logger.debug(f"🚀 快速生成句子: '{sentence[:30]}...'")
            
            # 使用快取的 conditioning
            gpt_cond_latent, speaker_embedding = self._get_cached_conditioning(self._default_hash)
            gpt_cond_latent = gpt_cond_latent.to(self.device, non_blocking=True)
            speaker_embedding = speaker_embedding.to(self.device, non_blocking=True)
            
            # 文字編碼
            text_tokens = torch.IntTensor(
                self.tts_model.tokenizer.encode(sentence, lang=language)
            ).unsqueeze(0).to(self.device)
            
            # 快速生成參數 - 針對短句優化
            with torch.no_grad():
                gpt_codes = self.tts_model.gpt.generate(
                    cond_latents=gpt_cond_latent,
                    text_inputs=text_tokens,
                    output_attentions=False,
                    repetition_penalty=4.0,   # 更高，避免重複
                    temperature=0.7,          # 降低隨機性
                    top_p=0.85,              # 更確定性的選擇
                    # max_length=100          # 限制長度
                )
                
                expected_output_len = torch.tensor([
                    gpt_codes.shape[-1] * self.tts_model.gpt.code_stride_len
                ], device=self.device)
                text_len = torch.tensor([text_tokens.shape[-1]], device=self.device)
                
                gpt_latents = self.tts_model.gpt(
                    text_tokens, text_len, gpt_codes, expected_output_len,
                    cond_latents=gpt_cond_latent,
                    return_attentions=False,
                    return_latent=True
                )
            
            mel_tensor = gpt_latents.detach()
            
            # 快速 ONNX 推理 - 單句優化
            audio_bytes = self._fast_onnx_inference(mel_tensor, speaker_embedding)
            
            # 快取結果
            self._sentence_cache[cache_key] = audio_bytes
            
            # 快取管理（防止記憶體溢出）
            if len(self._sentence_cache) > 1000:
                # 移除最舊的 200 個
                old_keys = list(self._sentence_cache.keys())[:200]
                for key in old_keys:
                    del self._sentence_cache[key]
            
            return audio_bytes
            
        except Exception as e:
            logger.error(f"快速生成句子失敗: {e}")
            return b""
    
    def _fast_onnx_inference(self, mel_tensor: torch.Tensor, speaker_embedding: torch.Tensor) -> bytes:
        """快速 ONNX 推理 - 針對單句優化"""
        try:
            mel_np = mel_tensor.cpu().numpy().astype(np.float32)
            speaker_np = speaker_embedding.cpu().numpy().astype(np.float32)
            
            mel_length = mel_np.shape[1]
            
            # 短句：直接處理，不分塊
            if mel_length <= self.fast_chunk_size:
                # 填充到最小塊大小
                if mel_length < self.fast_chunk_size:
                    padding = self.fast_chunk_size - mel_length
                    mel_padded = np.pad(mel_np, ((0, 0), (0, padding), (0, 0)), mode='constant')
                else:
                    mel_padded = mel_np
                
                # 直接推理
                onnx_inputs = {
                    "mel_spectrogram": mel_padded,
                    "speaker_embedding": speaker_np
                }
                
                audio_output = self.onnx_session.run(None, onnx_inputs)[0]
                final_audio = audio_output.squeeze()
                
                # 裁剪到實際長度
                actual_length = mel_length * self.hop_length
                final_audio = final_audio[:actual_length]
                
            else:
                # 較長句子：快速分塊
                audio_chunks = []
                
                for i in range(0, mel_length, self.fast_chunk_size):
                    end_idx = min(i + self.fast_chunk_size, mel_length)
                    mel_chunk = mel_np[:, i:end_idx, :]
                    
                    # 填充
                    if mel_chunk.shape[1] < self.fast_chunk_size:
                        padding = self.fast_chunk_size - mel_chunk.shape[1]
                        mel_chunk = np.pad(mel_chunk, ((0, 0), (0, padding), (0, 0)), mode='constant')
                    
                    onnx_inputs = {
                        "mel_spectrogram": mel_chunk,
                        "speaker_embedding": speaker_np
                    }
                    
                    chunk_output = self.onnx_session.run(None, onnx_inputs)[0]
                    audio_chunks.append(chunk_output.squeeze())
                
                # 使用減少的 overlap 快速合併
                original_overlap = self.overlap_samples
                self.overlap_samples = self.fast_overlap
                final_audio = self._overlap_add_chunks(audio_chunks)
                self.overlap_samples = original_overlap
            
            # 輕量級後處理
            final_audio = self._trim_trailing_silence(final_audio, threshold_db=-35.0)
            
            # 轉換為 WAV bytes
            buffer = io.BytesIO()
            write_wav(buffer, self.sample_rate, final_audio.astype(np.float32))
            
            return buffer.getvalue()
            
        except Exception as e:
            logger.error(f"快速 ONNX 推理失敗: {e}")
            return b""
    
    def _parallel_sentence_generation(self, sentences: List[str], language: str) -> List[bytes]:
        """並行生成多個句子"""
        from concurrent.futures import ThreadPoolExecutor, as_completed
        
        audio_results = [None] * len(sentences)
        
        # 使用線程池並行處理
        with ThreadPoolExecutor(max_workers=2) as executor:  # 限制 2 個並行，避免 GPU 記憶體不足
            future_to_index = {
                executor.submit(self._fast_generate_sentence, sentence, language): i
                for i, sentence in enumerate(sentences)
            }
            
            for future in as_completed(future_to_index):
                index = future_to_index[future]
                try:
                    audio_results[index] = future.result(timeout=15)
                except Exception as e:
                    logger.warning(f"句子 {index} 生成失敗: {e}")
                    audio_results[index] = b""
        
        return [result for result in audio_results if result]
    
    def _merge_audio_files(self, audio_chunks: List[bytes]) -> bytes:
        """合併多個 WAV 檔案"""
        if not audio_chunks:
            return b""
        
        if len(audio_chunks) == 1:
            return audio_chunks[0]
        
        try:
            import wave
            
            # 讀取所有音頻資料
            audio_data = []
            sample_rate = self.sample_rate
            
            for chunk in audio_chunks:
                if not chunk:
                    continue
                
                # 解析 WAV
                wav_io = io.BytesIO(chunk)
                with wave.open(wav_io, 'rb') as wav_file:
                    frames = wav_file.readframes(wav_file.getnframes())
                    audio_array = np.frombuffer(frames, dtype=np.int16).astype(np.float32)
                    audio_data.append(audio_array)
            
            # 合併音頻資料
            if audio_data:
                merged_audio = np.concatenate(audio_data)
                
                # 轉換為 WAV
                buffer = io.BytesIO()
                write_wav(buffer, sample_rate, merged_audio)
                return buffer.getvalue()
            
        except Exception as e:
            logger.warning(f"音頻合併失敗，使用簡單拼接: {e}")
            # 簡單拼接作為備用方案
            return b"".join(audio_chunks)
        
        return b""
    
    def Tts(self, request, context):
        """
        處理 TTS 請求 - NPU 友好優化版本（修正版）
        """
        start_time = time.time()
        
        try:
            # --- 輸入驗證 ---
            is_valid, error_msg = self._validate_request(request)
            if not is_valid:
                context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
                context.set_details(error_msg)
                return model_service_pb2.TtsResponse()
            
            # --- 快取的條件潜變數獲取（修正版）---
            if request.reference_audio:
                hash_key = self._hash_bytes(request.reference_audio)
                # 存入 blob_cache
                self._blob_cache[hash_key] = request.reference_audio
                logger.debug("🎤 使用了客戶端提供的參考音訊（已快取）")
            else:
                hash_key = self._default_hash
                logger.debug(f"🎤 使用預設參考音訊（已快取）")
            
            # 獲取快取的 conditioning（CPU tensor）
            gpt_cond_latent, speaker_embedding = self._get_cached_conditioning(hash_key)
            
            # 移動到設備（修正點 1: 使用時才移動到 GPU）
            gpt_cond_latent = gpt_cond_latent.to(self.device, non_blocking=True)
            speaker_embedding = speaker_embedding.to(self.device, non_blocking=True)
                
            # --- 生成梅爾頻譜 ---
            text_to_speak = request.text_to_speak.strip()
            language = request.language or "en"
            logger.info(f"📝 準備生成文字 (語言: {language}): '{text_to_speak[:30]}...'")
            
            # 文字編碼與生成
            text_tokens = torch.IntTensor(
                self.tts_model.tokenizer.encode(text_to_speak, lang=language)
            ).unsqueeze(0).to(self.device)

            with torch.no_grad():
                gpt_codes = self.tts_model.gpt.generate(
                    cond_latents=gpt_cond_latent, 
                    text_inputs=text_tokens,
                    output_attentions=False, 
                    repetition_penalty=5.0, 
                    # temperature=0.75, 
                    # top_p=0.9,
                    # length_penalty=1.5
                )
                
                expected_output_len = torch.tensor([
                    gpt_codes.shape[-1] * self.tts_model.gpt.code_stride_len
                ], device=self.device)
                text_len = torch.tensor([text_tokens.shape[-1]], device=self.device)
                
                gpt_latents = self.tts_model.gpt(
                    text_tokens, text_len, gpt_codes, expected_output_len,
                    cond_latents=gpt_cond_latent, 
                    return_attentions=False, 
                    return_latent=True
                )
            
            mel_spectrogram_tensor = gpt_latents.detach()

            # --- NPU 友好的分塊推論（修正版：使用真實 hop_length）---
            full_mel_spectrogram_np = mel_spectrogram_tensor.cpu().numpy().astype(np.float32)
            speaker_embedding_np = speaker_embedding.cpu().numpy().astype(np.float32)

            total_mel_length = full_mel_spectrogram_np.shape[1]
            logger.info(f"📊 總梅爾頻譜長度: {total_mel_length}")

            audio_chunks = []
            num_chunks = (full_mel_spectrogram_np.shape[1] + self.fixed_mel_chunk_length - 1) // self.fixed_mel_chunk_length
            
            logger.debug(f"🔄 準備處理 {num_chunks} 個音訊塊...")
            
            for i in range(num_chunks):
                start_idx = i * self.fixed_mel_chunk_length
                end_idx = min((i + 1) * self.fixed_mel_chunk_length, full_mel_spectrogram_np.shape[1])
                mel_chunk = full_mel_spectrogram_np[:, start_idx:end_idx, :]
                current_chunk_length = mel_chunk.shape[1]

                # 填充到固定長度
                if current_chunk_length < self.fixed_mel_chunk_length:
                    padding_needed = self.fixed_mel_chunk_length - current_chunk_length
                    mel_chunk_padded = np.pad(
                        mel_chunk, 
                        ((0, 0), (0, padding_needed), (0, 0)), 
                        mode='constant', 
                        constant_values=0
                    )
                else:
                    mel_chunk_padded = mel_chunk

                # NPU/ONNX 推論
                onnx_inputs = {
                    "mel_spectrogram": mel_chunk_padded, 
                    "speaker_embedding": speaker_embedding_np
                }
                
                try:
                    chunk_output = self.onnx_session.run(None, onnx_inputs)[0]
                    # actual_audio_length = current_chunk_length * self.hop_length
                    # chunk_audio = chunk_output.squeeze()[:actual_audio_length]
                    audio_chunks.append(chunk_output)
                    
                    # 增加日誌以追蹤每個塊的進度
                    logger.info(f"✅ 第 {i+1}/{num_chunks} 塊推論成功。梅爾長度: {current_chunk_length}, 音訊長度: {len(chunk_output)}")
                except Exception as e:
                    logger.warning(f"⚠️ 第 {i+1} 塊推論失敗: {e}")
                    continue

            # --- Overlap-Add 拼接（消除接縫） ---
            if not audio_chunks:
                context.set_code(grpc.StatusCode.INTERNAL)
                context.set_details("音訊生成失敗，沒有任何音訊塊產生。")
                return model_service_pb2.TtsResponse()

            logger.debug("🎵 使用 overlap-add 拼接音訊塊...")
            final_audio_waveform = self._overlap_add_chunks(audio_chunks)
            
            ### --- 修改 --- ###
            # 修剪尾部靜音
            original_length = len(final_audio_waveform)
            final_audio_waveform = self._trim_trailing_silence(final_audio_waveform)
            trimmed_length = len(final_audio_waveform)
            if original_length > trimmed_length:
                original_duration = original_length / self.sample_rate
                trimmed_duration = trimmed_length / self.sample_rate
                logger.info(f"✂️ 已修剪尾部靜音。音訊長度從 {original_duration:.2f}s 減少到 {trimmed_duration:.2f}s。")
            ### --- 修改結束 --- ###
            
            # --- 轉換為 WAV bytes ---
            buffer = io.BytesIO()
            write_wav(buffer, self.sample_rate, final_audio_waveform.astype(np.float32))
            wav_bytes = buffer.getvalue()
            
            end_time = time.time()
            logger.info(f"✅ 請求處理完成，總耗時: {end_time - start_time:.2f} 秒")
            logger.info(f"📊 使用的 EP: {self.active_providers[0] if self.active_providers else 'Unknown'}")
            logger.info(f"✅ 最終音訊預估時長: {len(wav_bytes) / (self.sample_rate * 4):.2f} 秒")
            # TODO: 修正點 6 - 加入 metrics 追蹤
            # metrics.tts_request_latency_ms.observe((end_time - start_time) * 1000)
            # metrics.onnx_ep_in_use.labels(self.active_providers[0]).inc()
            
            return model_service_pb2.TtsResponse(generated_audio=wav_bytes)

        except Exception as e:
            logger.error(f"❌ 處理請求時發生錯誤: {e}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"內部伺服器錯誤: {str(e)}")
            return model_service_pb2.TtsResponse()
    
    # def Tts(self, request, context):
    #     """
    #     LLM 回答場景優化的 TTS 處理 - 修復版本
    #     """
    #     start_time = time.time()

    #     try:
    #         # 輸入驗證
    #         is_valid, error_msg = self._validate_request(request)
    #         if not is_valid:
    #             context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
    #             context.set_details(error_msg)
    #             return model_service_pb2.TtsResponse()
            
    #         text = request.text_to_speak.strip()
    #         language = request.language or "en"
            
    #         logger.info(f"📝 LLM TTS 請求: '{text[:50]}...' ({len(text)} 字符)")
            
    #         # 快速策略選擇
    #         use_fast_mode = (
    #             len(text) <= 200 and  # 中短文本
    #             not request.reference_audio and  # 不使用自定義聲音
    #             self.fast_mode_enabled  # 快速模式啟用
    #         )
            
    #         if use_fast_mode:
    #             logger.info("🚀 嘗試快速模式")
                
    #             if len(text) <= 50:
    #                 # 短文本：直接快速生成
    #                 logger.info("📝 使用短文本快速模式")
    #                 audio_bytes = self._fast_generate_sentence(text, language)
                    
    #                 if audio_bytes:
    #                     total_time = time.time() - start_time
    #                     logger.info(f"⚡ 短文本完成: {total_time:.2f}s")
    #                     return model_service_pb2.TtsResponse(generated_audio=audio_bytes)
    #                 else:
    #                     logger.warning("⚠️ 短文本快速模式失敗，回退到標準模式")
                        
    #             else:
    #                 # 中等文本：分句處理
    #                 logger.info("📚 使用分句模式")
    #                 sentences = self._split_into_sentences(text)
    #                 logger.info(f"📊 分割為 {len(sentences)} 個句子")
                    
    #                 if len(sentences) == 1:
    #                     # 只有一句，直接處理
    #                     audio_bytes = self._fast_generate_sentence(sentences[0], language)
    #                     if audio_bytes:
    #                         total_time = time.time() - start_time
    #                         logger.info(f"⚡ 單句完成: {total_time:.2f}s")
    #                         return model_service_pb2.TtsResponse(generated_audio=audio_bytes)
    #                 else:
    #                     # 多句處理 - 先嘗試快速，失敗則回退
    #                     try:
    #                         audio_chunks = []
    #                         all_success = True
                            
    #                         for i, sentence in enumerate(sentences):
    #                             chunk = self._fast_generate_sentence(sentence, language)
    #                             if chunk:
    #                                 audio_chunks.append(chunk)
    #                             else:
    #                                 logger.warning(f"⚠️ 句子 {i+1} 快速生成失敗")
    #                                 all_success = False
    #                                 break
                            
    #                         if all_success and audio_chunks:
    #                             audio_bytes = self._merge_audio_files(audio_chunks)
    #                             if audio_bytes:
    #                                 total_time = time.time() - start_time
    #                                 logger.info(f"⚡ 分句並行完成: {total_time:.2f}s")
    #                                 return model_service_pb2.TtsResponse(generated_audio=audio_bytes)
                            
    #                     except Exception as e:
    #                         logger.warning(f"⚠️ 分句並行失敗: {e}")
            
    #         # 回退到標準模式（原有的完整邏輯）
    #         logger.info("🔄 使用標準 TTS 模式")
    #         return self._standard_tts_process(request, context, start_time)
            
    #     except Exception as e:
    #         logger.error(f"❌ TTS 處理錯誤: {e}")
    #         context.set_code(grpc.StatusCode.INTERNAL)
    #         context.set_details(f"TTS 處理失敗: {str(e)}")
    #         return model_service_pb2.TtsResponse()
    
    def _standard_tts_process(self, request, context, start_time):
        """
        標準 TTS 處理邏輯 - 從原有的 Tts 方法重構
        """
        try:
            # --- 快取的條件潜變數獲取 ---
            if request.reference_audio:
                hash_key = self._hash_bytes(request.reference_audio)
                # 存入 blob_cache
                self._blob_cache[hash_key] = request.reference_audio
                logger.debug("🎤 使用了客戶端提供的參考音訊（已快取）")
            else:
                hash_key = self._default_hash
                logger.debug(f"🎤 使用預設參考音訊（已快取）")
            
            # 獲取快取的 conditioning（CPU tensor）
            gpt_cond_latent, speaker_embedding = self._get_cached_conditioning(hash_key)
            
            # 移動到設備
            gpt_cond_latent = gpt_cond_latent.to(self.device, non_blocking=True)
            speaker_embedding = speaker_embedding.to(self.device, non_blocking=True)
                
            # --- 生成梅爾頻譜 ---
            text_to_speak = request.text_to_speak.strip()
            language = request.language or "en"
            logger.info(f"📝 準備生成文字 (語言: {language}): '{text_to_speak[:30]}...'")
            
            # 文字編碼與生成
            text_tokens = torch.IntTensor(
                self.tts_model.tokenizer.encode(text_to_speak, lang=language)
            ).unsqueeze(0).to(self.device)

            with torch.no_grad():
                gpt_codes = self.tts_model.gpt.generate(
                    cond_latents=gpt_cond_latent, 
                    text_inputs=text_tokens,
                    output_attentions=False, 
                    repetition_penalty=5.0, 
                )
                
                expected_output_len = torch.tensor([
                    gpt_codes.shape[-1] * self.tts_model.gpt.code_stride_len
                ], device=self.device)
                text_len = torch.tensor([text_tokens.shape[-1]], device=self.device)
                
                gpt_latents = self.tts_model.gpt(
                    text_tokens, text_len, gpt_codes, expected_output_len,
                    cond_latents=gpt_cond_latent, 
                    return_attentions=False, 
                    return_latent=True
                )
            
            mel_spectrogram_tensor = gpt_latents.detach()

            # --- NPU 友好的分塊推論 - 使用正確的 chunk_size ---
            full_mel_spectrogram_np = mel_spectrogram_tensor.cpu().numpy().astype(np.float32)
            speaker_embedding_np = speaker_embedding.cpu().numpy().astype(np.float32)

            total_mel_length = full_mel_spectrogram_np.shape[1]
            logger.info(f"📊 總梅爾頻譜長度: {total_mel_length}")

            audio_chunks = []
            # 使用固定的 chunk_size = 100 (與 ONNX 模型匹配)
            chunk_size = self.fixed_mel_chunk_length  # 100
            num_chunks = (total_mel_length + chunk_size - 1) // chunk_size
            
            logger.debug(f"🔄 準備處理 {num_chunks} 個音訊塊...")
            
            for i in range(num_chunks):
                start_idx = i * chunk_size
                end_idx = min((i + 1) * chunk_size, total_mel_length)
                mel_chunk = full_mel_spectrogram_np[:, start_idx:end_idx, :]
                current_chunk_length = mel_chunk.shape[1]

                # 填充到固定長度 (100)
                if current_chunk_length < chunk_size:
                    padding_needed = chunk_size - current_chunk_length
                    mel_chunk_padded = np.pad(
                        mel_chunk, 
                        ((0, 0), (0, padding_needed), (0, 0)), 
                        mode='constant', 
                        constant_values=0
                    )
                else:
                    mel_chunk_padded = mel_chunk

                # ONNX 推論
                onnx_inputs = {
                    "mel_spectrogram": mel_chunk_padded, 
                    "speaker_embedding": speaker_embedding_np
                }
                
                try:
                    chunk_output = self.onnx_session.run(None, onnx_inputs)[0]
                    # 根據實際長度裁剪音頻
                    actual_audio_length = current_chunk_length * self.hop_length
                    chunk_audio = chunk_output.squeeze()[:actual_audio_length]
                    audio_chunks.append(chunk_audio)
                    
                    logger.debug(f"✅ 第 {i+1}/{num_chunks} 塊推論成功")
                except Exception as e:
                    logger.warning(f"⚠️ 第 {i+1} 塊推論失敗: {e}")
                    continue

            # --- Overlap-Add 拼接 ---
            if not audio_chunks:
                context.set_code(grpc.StatusCode.INTERNAL)
                context.set_details("音訊生成失敗，沒有任何音訊塊產生。")
                return model_service_pb2.TtsResponse()

            logger.debug("🎵 使用 overlap-add 拼接音訊塊...")
            final_audio_waveform = self._overlap_add_chunks(audio_chunks)
            
            # 修剪尾部靜音
            original_length = len(final_audio_waveform)
            final_audio_waveform = self._trim_trailing_silence(final_audio_waveform)
            trimmed_length = len(final_audio_waveform)
            if original_length > trimmed_length:
                original_duration = original_length / self.sample_rate
                trimmed_duration = trimmed_length / self.sample_rate
                logger.info(f"✂️ 已修剪尾部靜音。音訊長度從 {original_duration:.2f}s 減少到 {trimmed_duration:.2f}s。")
            
            # --- 轉換為 WAV bytes ---
            buffer = io.BytesIO()
            write_wav(buffer, self.sample_rate, final_audio_waveform.astype(np.float32))
            wav_bytes = buffer.getvalue()
            
            end_time = time.time()
            logger.info(f"✅ 標準模式完成，總耗時: {end_time - start_time:.2f} 秒")
            
            return model_service_pb2.TtsResponse(generated_audio=wav_bytes)
            
        except Exception as e:
            logger.error(f"❌ 標準模式處理錯誤: {e}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"標準模式處理失敗: {str(e)}")
            return model_service_pb2.TtsResponse()