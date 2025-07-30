"""
StreamingDiarization: 處理即時傳入的音訊流 (串流處理)。

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
from scipy.spatial.distance import cdist
import librosa
import time

--- 即時音訊串流處理 ---

class StreamingDiarization:
    def __init__(self, segmentation_model_path, embedding_model_path, device='cpu'):
        """
        初始化即時串流語者辨識器。

        Args:
            segmentation_model_path (str): 分割模型 (.onnx) 的路徑。
            embedding_model_path (str): 聲紋嵌入模型 (.onnx) 的路徑。
            device (str): 運行的裝置， 'cpu' 或 'cuda'。
        """
        print("正在初始化 StreamingDiarization...")
        providers = ['CPUExecutionProvider']
        if device.lower() == 'cuda':
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']

        self.seg_session = ort.InferenceSession(segmentation_model_path, providers=providers)
        self.emb_session = ort.InferenceSession(embedding_model_path, providers=providers)
        
        self.seg_input_name = self.seg_session.get_inputs()[0].name
        self.emb_input_name = self.emb_session.get_inputs()[0].name

        # --- 串流處理的參數 ---
        self.sample_rate = 16000
        self.chunk_seconds = 0.5  # 每次處理的音訊塊長度(秒)
        self.chunk_samples = int(self.chunk_seconds * self.sample_rate)
        self.buffer_seconds = 5.0   # 維護一個 5 秒的滑動緩衝區
        self.buffer_samples = int(self.buffer_seconds * self.sample_rate)
        
        # --- 狀態管理 ---
        self.audio_buffer = np.array([], dtype=np.float32)
        self.speech_in_progress = False
        self.speech_start_sample = 0
        self.total_samples_processed = 0

        # --- 聚類參數 ---
        self.speaker_clusters = []  # 儲存每個 speaker 的聲紋向量中心
        self.clustering_threshold = 0.5  # 判斷是否為新講者的餘弦距離閾值

        # --- 嵌入提取參數 ---
        self.embedding_chunk_samples = int(1.5 * self.sample_rate)
        
        print("即時串流辨識器已就緒。")

    def process(self, audio_chunk):
        """
        處理一小段傳入的音訊。

        Args:
            audio_chunk (np.ndarray): 一小段音訊數據 (float32)。

        Returns:
            list: 一個列表，包含本次處理中新確定的 (speaker_label, start_time, end_time) 元組。
                  如果沒有新的片段確定，則回傳空列表。
        """
        if not isinstance(audio_chunk, np.ndarray):
            audio_chunk = np.array(audio_chunk, dtype=np.float32)

        # 1. 更新音訊緩衝區
        self.audio_buffer = np.concatenate([self.audio_buffer, audio_chunk])

        # 2. 執行分割模型，判斷當前是否在說話
        # 我們只對緩衝區的最新部分進行推論以節省計算
        if len(self.audio_buffer) >= self.chunk_samples:
            
            # 取緩衝區尾部來進行推斷
            inference_chunk = self.audio_buffer[-self.chunk_samples:]
            onnx_input = np.expand_dims(inference_chunk, axis=0).astype(np.float32)
            
            # 確保輸入長度符合模型要求，不足則補零
            required_input_length = self.seg_session.get_inputs()[0].shape[1]
            if len(inference_chunk) < required_input_length:
                 padding = np.zeros(required_input_length - len(inference_chunk), dtype=np.float32)
                 onnx_input = np.expand_dims(np.concatenate([inference_chunk, padding]), axis=0)

            ort_outs = self.seg_session.run(None, {self.seg_input_name: onnx_input})
            speech_prob = np.mean(ort_outs[0][0, :, 1]) # 索引 1 代表 'speech'

            newly_finalized_segments = []
            
            # 3. 狀態機：根據機率判斷語音的開始與結束
            if speech_prob > 0.5 and not self.speech_in_progress:
                # 語音開始
                self.speech_in_progress = True
                self.speech_start_sample = self.total_samples_processed

            elif speech_prob < 0.4 and self.speech_in_progress:
                # 語音結束
                self.speech_in_progress = False
                segment_end_sample = self.total_samples_processed + len(audio_chunk)
                
                # 從緩衝區提取剛結束的這段完整語音
                start_in_buffer = max(0, self.speech_start_sample - (self.total_samples_processed - len(self.audio_buffer)))
                end_in_buffer = len(self.audio_buffer)
                speech_segment_audio = self.audio_buffer[start_in_buffer:end_in_buffer]
                
                # 4. 提取聲紋並進行線上聚類
                finalized_segment = self._finalize_segment(speech_segment_audio, self.speech_start_sample, segment_end_sample)
                if finalized_segment:
                    newly_finalized_segments.append(finalized_segment)

            # 5. 更新已處理的樣本總數
            self.total_samples_processed += len(audio_chunk)

            # 6. 維護緩衝區大小，移除過舊的數據
            if len(self.audio_buffer) > self.buffer_samples:
                self.audio_buffer = self.audio_buffer[-self.buffer_samples:]

            return newly_finalized_segments
        
        return []

    def _finalize_segment(self, segment_audio, start_sample, end_sample):
        """當一段語音結束時，提取嵌入並指派給一個講者"""
        duration_ms = (end_sample - start_sample) / self.sample_rate * 1000
        if duration_ms < 200: # 忽略過短的片段
            return None

        # a. 提取聲紋嵌入
        # 確保音訊長度足夠，不足則補齊
        if len(segment_audio) < self.embedding_chunk_samples:
            segment_audio = np.tile(segment_audio, int(np.ceil(self.embedding_chunk_samples / len(segment_audio))))[:self.embedding_chunk_samples]
        else: # 過長則取中間
            mid = len(segment_audio) // 2
            segment_audio = segment_audio[mid - self.embedding_chunk_samples//2 : mid + self.embedding_chunk_samples//2]

        onnx_input = np.expand_dims(segment_audio, axis=0).astype(np.float32)
        ort_outs = self.emb_session.run(None, {self.emb_input_name: onnx_input})
        embedding = ort_outs[0][0]

        # b. 線上聚類 (Online Clustering)
        if not self.speaker_clusters:
            # 第一個講者
            self.speaker_clusters.append([embedding])
            speaker_id = 0
        else:
            # 計算與現有講者群中心的距離
            cluster_centroids = [np.mean(cluster, axis=0) for cluster in self.speaker_clusters]
            distances = cdist(np.expand_dims(embedding, axis=0), np.array(cluster_centroids), 'cosine')[0]
            
            min_dist_idx = np.argmin(distances)
            if distances[min_dist_idx] < self.clustering_threshold:
                # 分配給現有講者
                speaker_id = min_dist_idx
                self.speaker_clusters[speaker_id].append(embedding)
            else:
                # 建立新講者
                self.speaker_clusters.append([embedding])
                speaker_id = len(self.speaker_clusters) - 1

        speaker_label = f"SPEAKER_{speaker_id:02d}"
        start_time = start_sample / self.sample_rate
        end_time = end_sample / self.sample_rate

        return (speaker_label, start_time, end_time)
