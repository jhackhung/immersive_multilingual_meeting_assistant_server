"""
StreamingDiarization: 處理即時傳入的音訊流 (串流處理)
針對高通 NPU 優化的版本

必要安裝的函式庫:
pip install numpy
pip install onnxruntime
pip install scikit-learn
pip install librosa
"""

import os
import numpy as np
import onnxruntime as ort
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import cdist
import librosa
import time
from typing import List, Tuple, Optional
import logging

class StreamingDiarization:
    def __init__(self, 
                 segmentation_model_path: str = "models/pyannote_segmentation_static.onnx", 
                 embedding_model_path: str = "models/pyannote_embedding_static.onnx", 
                 device: str = 'cpu',
                 use_qualcomm_npu: bool = False):
        """
        初始化針對高通 NPU 優化的即時串流語者辨識器。

        Args:
            segmentation_model_path (str): VAD/分割模型 (.onnx) 的路徑
            embedding_model_path (str): 聲紋嵌入模型 (.onnx) 的路徑
            device (str): 'cpu', 'cuda', 或 'qnn' (高通 NPU)
            use_qualcomm_npu (bool): 是否使用高通 NPU 加速
        """
        print("正在初始化高通優化的 StreamingDiarization...")
        
        # 檢查模型檔案是否存在
        if not os.path.exists(segmentation_model_path):
            raise FileNotFoundError(f"找不到分割模型: {segmentation_model_path}")
        if not os.path.exists(embedding_model_path):
            raise FileNotFoundError(f"找不到嵌入模型: {embedding_model_path}")
        
        # 設定執行提供者
        if use_qualcomm_npu and device == 'qnn':
            # 高通 NPU 執行提供者
            providers = ['QNNExecutionProvider', 'CPUExecutionProvider']
            provider_options = [
                {
                    'backend_path': 'QnnHtp.dll',  # Windows
                    # 'backend_path': 'libQnnHtp.so',  # Linux/Android
                    'profiling_level': 'basic'
                },
                {}
            ]
        elif device.lower() == 'cuda':
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            provider_options = None
        else:
            providers = ['CPUExecutionProvider']
            provider_options = None

        # 載入模型
        try:
            if provider_options:
                self.seg_session = ort.InferenceSession(
                    segmentation_model_path, 
                    providers=providers,
                    provider_options=provider_options
                )
                self.emb_session = ort.InferenceSession(
                    embedding_model_path, 
                    providers=providers,
                    provider_options=provider_options
                )
            else:
                self.seg_session = ort.InferenceSession(segmentation_model_path, providers=providers)
                self.emb_session = ort.InferenceSession(embedding_model_path, providers=providers)
                
            print(f"✅ 模型已載入，使用執行提供者: {providers[0]}")
            
        except Exception as e:
            print(f"❌ 模型載入失敗: {e}")
            raise
        
        # 獲取模型輸入/輸出資訊
        self.seg_input_name = self.seg_session.get_inputs()[0].name
        self.seg_input_shape = self.seg_session.get_inputs()[0].shape
        self.seg_output_shape = self.seg_session.get_outputs()[0].shape
        
        self.emb_input_name = self.emb_session.get_inputs()[0].name
        self.emb_input_shape = self.emb_session.get_inputs()[0].shape
        self.emb_output_shape = self.emb_session.get_outputs()[0].shape
        
        print(f"分割模型: 輸入 {self.seg_input_shape}, 輸出 {self.seg_output_shape}")
        print(f"嵌入模型: 輸入 {self.emb_input_shape}, 輸出 {self.emb_output_shape}")

        # 音訊處理參數 (針對移動裝置優化)
        self.sample_rate = 16000
        self.chunk_seconds = 0.5
        self.chunk_samples = int(self.chunk_seconds * self.sample_rate)
        self.buffer_seconds = 5.0
        self.buffer_samples = int(self.buffer_seconds * self.sample_rate)
        
        # 模型輸入長度 (從 ONNX 模型獲取)
        self.seg_required_length = self.seg_input_shape[1] if len(self.seg_input_shape) > 1 else self.chunk_samples
        self.emb_required_length = self.emb_input_shape[1] if len(self.emb_input_shape) > 1 else int(1.5 * self.sample_rate)
        
        # 狀態管理
        self.audio_buffer = np.array([], dtype=np.float32)
        self.speech_in_progress = False
        self.speech_start_sample = 0
        self.total_samples_processed = 0

        # 優化的聚類參數 (適合即時處理)
        self.speaker_embeddings = []  # 每個說話者的嵌入集合
        self.speaker_centroids = []   # 每個說話者的中心點
        self.clustering_threshold = 0.7  # 餘弦相似度閾值
        self.min_segment_duration = 0.3  # 最小片段長度
        self.max_speakers = 10  # 最大說話者數量限制
        
        # 效能優化
        self.enable_voice_activity_detection = True
        self.vad_threshold_high = 0.6  # 語音開始閾值  
        self.vad_threshold_low = 0.3   # 語音結束閾值
        
        print("✅ 高通優化串流辨識器已就緒")

    def process(self, audio_chunk: np.ndarray) -> List[Tuple[str, float, float]]:
        """
        處理一小段傳入的音訊 (針對即時性優化)

        Args:
            audio_chunk (np.ndarray): 音訊數據 (float32, 單聲道)

        Returns:
            list: 新確定的 (speaker_label, start_time, end_time) 元組列表
        """
        if not isinstance(audio_chunk, np.ndarray):
            audio_chunk = np.array(audio_chunk, dtype=np.float32)

        # 確保音訊是單聲道且正確格式
        if len(audio_chunk.shape) > 1:
            audio_chunk = np.mean(audio_chunk, axis=1)
        
        # 更新緩衝區
        self.audio_buffer = np.concatenate([self.audio_buffer, audio_chunk])

        newly_finalized_segments = []
        
        # 當緩衝區有足夠數據時進行處理
        if len(self.audio_buffer) >= self.seg_required_length:
            
            # 1. 語音活動檢測 (VAD)
            if self.enable_voice_activity_detection:
                speech_prob = self._detect_speech_activity()
                
                # 2. 狀態機處理
                segments = self._handle_speech_state_machine(speech_prob, len(audio_chunk))
                newly_finalized_segments.extend(segments)
            
            # 3. 更新狀態
            self.total_samples_processed += len(audio_chunk)
            
            # 4. 維護緩衝區大小 (記憶體管理)
            if len(self.audio_buffer) > self.buffer_samples:
                self.audio_buffer = self.audio_buffer[-self.buffer_samples:]

        return newly_finalized_segments

    def _detect_speech_activity(self) -> float:
        """
        使用 ONNX VAD 模型檢測語音活動
        """
        try:
            # 準備輸入數據
            input_audio = self._prepare_vad_input()
            
            # ONNX 推理
            ort_inputs = {self.seg_input_name: input_audio}
            ort_outputs = self.seg_session.run(None, ort_inputs)
            
            # 解析輸出 (假設輸出格式為 [batch, time, classes] 或 [batch, classes])
            output = ort_outputs[0]
            
            if len(output.shape) == 3:
                # [batch, time, classes] - 取 speech 類別的平均機率
                speech_prob = np.mean(output[0, :, 1])  # 假設 index 1 是 speech
            elif len(output.shape) == 2:
                # [batch, classes] - 直接取 speech 類別
                speech_prob = output[0, 1]
            else:
                # [batch] - 直接使用輸出值
                speech_prob = output[0]
            
            return float(speech_prob)
            
        except Exception as e:
            logging.error(f"VAD 推理錯誤: {e}")
            return 0.0

    def _prepare_vad_input(self) -> np.ndarray:
        """準備 VAD 模型的輸入"""
        # 取最新的音訊片段
        if len(self.audio_buffer) >= self.seg_required_length:
            audio_segment = self.audio_buffer[-self.seg_required_length:]
        else:
            # 不足則補零
            audio_segment = np.zeros(self.seg_required_length, dtype=np.float32)
            audio_segment[:len(self.audio_buffer)] = self.audio_buffer
        
        # 標準化
        if np.std(audio_segment) > 0:
            audio_segment = (audio_segment - np.mean(audio_segment)) / np.std(audio_segment)
        
        # 轉換為模型期望的格式 [batch, samples] 或 [batch, 1, samples]
        if len(self.seg_input_shape) == 3:
            return np.expand_dims(np.expand_dims(audio_segment, axis=0), axis=1).astype(np.float32)
        else:
            return np.expand_dims(audio_segment, axis=0).astype(np.float32)

    def _handle_speech_state_machine(self, speech_prob: float, chunk_size: int) -> List[Tuple[str, float, float]]:
        """處理語音檢測的狀態機"""
        segments = []
        
        # 語音開始檢測
        if speech_prob > self.vad_threshold_high and not self.speech_in_progress:
            self.speech_in_progress = True
            self.speech_start_sample = self.total_samples_processed
            
        # 語音結束檢測  
        elif speech_prob < self.vad_threshold_low and self.speech_in_progress:
            self.speech_in_progress = False
            segment_end_sample = self.total_samples_processed + chunk_size
            
            # 檢查片段長度
            duration = (segment_end_sample - self.speech_start_sample) / self.sample_rate
            if duration >= self.min_segment_duration:
                
                # 提取並處理這段語音
                segment = self._process_speech_segment(segment_end_sample)
                if segment:
                    segments.append(segment)
        
        return segments

    def _process_speech_segment(self, end_sample: int) -> Optional[Tuple[str, float, float]]:
        """處理一個完整的語音片段"""
        try:
            # 從緩衝區提取語音片段
            start_in_buffer = max(0, self.speech_start_sample - 
                                (self.total_samples_processed - len(self.audio_buffer)))
            end_in_buffer = len(self.audio_buffer)
            
            speech_audio = self.audio_buffer[start_in_buffer:end_in_buffer]
            
            if len(speech_audio) < self.sample_rate * 0.1:  # 太短則忽略
                return None
            
            # 提取聲紋嵌入
            embedding = self._extract_speaker_embedding(speech_audio)
            if embedding is None:
                return None
            
            # 分配說話者
            speaker_id = self._assign_speaker(embedding)
            
            # 建立結果
            speaker_label = f"SPEAKER_{speaker_id:02d}"
            start_time = self.speech_start_sample / self.sample_rate
            end_time = end_sample / self.sample_rate
            
            return (speaker_label, start_time, end_time)
            
        except Exception as e:
            logging.error(f"語音片段處理錯誤: {e}")
            return None

    def _extract_speaker_embedding(self, speech_audio: np.ndarray) -> Optional[np.ndarray]:
        """
        使用 ONNX 嵌入模型提取說話者特徵
        """
        try:
            # 準備輸入 (確保長度符合模型要求)
            if len(speech_audio) < self.emb_required_length:
                # 重複填充
                repeat_factor = int(np.ceil(self.emb_required_length / len(speech_audio)))
                speech_audio = np.tile(speech_audio, repeat_factor)[:self.emb_required_length]
            elif len(speech_audio) > self.emb_required_length:
                # 取中間部分
                mid = len(speech_audio) // 2
                half_len = self.emb_required_length // 2
                speech_audio = speech_audio[mid - half_len:mid + half_len]
            
            # 音訊預處理
            if np.std(speech_audio) > 0:
                speech_audio = (speech_audio - np.mean(speech_audio)) / np.std(speech_audio)
            
            # 準備 ONNX 輸入
            if len(self.emb_input_shape) == 3:
                onnx_input = np.expand_dims(np.expand_dims(speech_audio, axis=0), axis=1).astype(np.float32)
            else:
                onnx_input = np.expand_dims(speech_audio, axis=0).astype(np.float32)
            
            # 推理
            ort_inputs = {self.emb_input_name: onnx_input}
            ort_outputs = self.emb_session.run(None, ort_inputs)
            
            # 提取嵌入向量 (通常是最後一個輸出的第一個批次)
            embedding = ort_outputs[0][0]
            
            # L2 正規化
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm
            
            return embedding
            
        except Exception as e:
            logging.error(f"嵌入提取錯誤: {e}")
            return None

    def _assign_speaker(self, embedding: np.ndarray) -> int:
        """
        分配說話者 ID (優化的線上聚類)
        """
        if len(self.speaker_centroids) == 0:
            # 第一個說話者
            self.speaker_embeddings.append([embedding])
            self.speaker_centroids.append(embedding.copy())
            return 0
        
        # 計算與現有說話者的相似度
        similarities = []
        for centroid in self.speaker_centroids:
            similarity = cosine_similarity([embedding], [centroid])[0][0]
            similarities.append(similarity)
        
        max_similarity = max(similarities)
        best_speaker_idx = similarities.index(max_similarity)
        
        # 判斷是否分配給現有說話者或建立新說話者
        if max_similarity > self.clustering_threshold:
            # 分配給現有說話者
            self.speaker_embeddings[best_speaker_idx].append(embedding)
            
            # 更新該說話者的中心點 (移動平均)
            alpha = 0.1  # 學習率
            self.speaker_centroids[best_speaker_idx] = (
                (1 - alpha) * self.speaker_centroids[best_speaker_idx] + 
                alpha * embedding
            )
            
            return best_speaker_idx
        else:
            # 建立新說話者 (如果未達到最大數量限制)
            if len(self.speaker_centroids) < self.max_speakers:
                self.speaker_embeddings.append([embedding])
                self.speaker_centroids.append(embedding.copy())
                return len(self.speaker_centroids) - 1
            else:
                # 達到限制，分配給最相似的現有說話者
                return best_speaker_idx

    def get_statistics(self) -> dict:
        """獲取統計資訊"""
        return {
            "total_speakers": len(self.speaker_centroids),
            "total_processed_time": self.total_samples_processed / self.sample_rate,
            "buffer_size": len(self.audio_buffer),
            "speech_in_progress": self.speech_in_progress
        }

    def reset(self):
        """重置所有狀態"""
        self.audio_buffer = np.array([], dtype=np.float32)
        self.speech_in_progress = False
        self.speech_start_sample = 0
        self.total_samples_processed = 0
        self.speaker_embeddings = []
        self.speaker_centroids = []
        print("✅ StreamingDiarization 已重置")

# 輔助函數：模型驗證
def validate_onnx_models(seg_model_path: str = "models/pyannote_segmentation_static.onnx", 
                        emb_model_path: str = "models/pyannote_embedding_static.onnx"):
    """驗證 ONNX 模型是否正確載入"""
    try:
        print("🔍 驗證模型...")
        
        # 檢查檔案是否存在
        if not os.path.exists(seg_model_path):
            print(f"❌ 找不到分割模型: {seg_model_path}")
            return False
        if not os.path.exists(emb_model_path):
            print(f"❌ 找不到嵌入模型: {emb_model_path}")
            return False
        
        # 載入模型
        seg_session = ort.InferenceSession(seg_model_path)
        emb_session = ort.InferenceSession(emb_model_path)
        
        print(f"✅ 分割模型: {seg_session.get_inputs()[0].shape}")
        print(f"✅ 嵌入模型: {emb_session.get_inputs()[0].shape}")
        
        # 測試推理
        seg_input_shape = seg_session.get_inputs()[0].shape
        emb_input_shape = emb_session.get_inputs()[0].shape
        
        # 建立測試輸入
        if len(seg_input_shape) == 3:
            test_seg_input = np.random.randn(1, 1, seg_input_shape[2]).astype(np.float32)
        else:
            test_seg_input = np.random.randn(1, seg_input_shape[1]).astype(np.float32)
            
        if len(emb_input_shape) == 3:
            test_emb_input = np.random.randn(1, 1, emb_input_shape[2]).astype(np.float32)
        else:
            test_emb_input = np.random.randn(1, emb_input_shape[1]).astype(np.float32)
        
        # 測試推理
        seg_output = seg_session.run(None, {seg_session.get_inputs()[0].name: test_seg_input})
        emb_output = emb_session.run(None, {emb_session.get_inputs()[0].name: test_emb_input})
        
        print(f"✅ 分割模型輸出: {seg_output[0].shape}")
        print(f"✅ 嵌入模型輸出: {emb_output[0].shape}")
        print("🎉 模型驗證成功！")
        
        return True
        
    except Exception as e:
        print(f"❌ 模型驗證失敗: {e}")
        return False
