# 檔案: server.py (最終修正版 v2)

from concurrent import futures
import grpc
import time
import logging

# 匯入 gRPC 模組
from proto import model_service_pb2
from proto import model_service_pb2_grpc

# 匯入所有 API 服務層
from apis.wav2lip_service import Wav2LipServicer
from apis.translator_service import TranslatorService
from apis.tts_service import TtsServicer

# 設定日誌
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- 定義一個較大的訊息長度，例如 100MB ---
MAX_MESSAGE_LENGTH = 100 * 1024 * 1024

class TranslatorServicer(model_service_pb2_grpc.TranslatorServiceServicer):
    """gRPC 翻譯服務實現"""
    
    def __init__(self, translator_api: TranslatorService):
        self.translator_api = translator_api
        logger.info("TranslatorServicer 已初始化")

    def Translate(self, request, context):
        request_data = {
            "text": request.text_to_translate,
            "source_lang": request.source_language,
            "target_lang": request.target_language
        }
        logger.info(f"收到翻譯請求: {request_data}")
        result = self.translator_api.process_translation_request(request_data)
        if result["success"]:
            return model_service_pb2.TranslateResponse(
                translated_text=result["translated_text"]
            )
        else:
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(result["error"])
            return model_service_pb2.TranslateResponse()

class MediaServicer(model_service_pb2_grpc.MediaServiceServicer):
    """統一的媒體服務實現"""
    
    def __init__(self, tts_servicer, wav2lip_servicer, speaker_annote_servicer):
        self.tts_servicer = tts_servicer
        self.wav2lip_servicer = wav2lip_servicer
        self.speaker_annote_servicer = speaker_annote_servicer
        logger.info("MediaServicer 已初始化")
    
    def Tts(self, request, context):
        logger.info("收到 TTS 請求")
        return self.tts_servicer.Tts(request, context)
    
    def Wav2Lip(self, request, context):
        logger.info("收到 Wav2Lip 請求")
        return self.wav2lip_servicer.Wav2Lip(request, context)
    
    def SpeakerAnnote(self, request, context):
        logger.info("收到 SpeakerAnnote 請求")
        return self.speaker_annote_servicer.SpeakerAnnote(request, context)


class SpeakerAnnoteServicer:
    """語者辨識服務的包裝器"""
    
    def __init__(self):
        self.diarization_model = None
        logger.info("SpeakerAnnoteServicer 已初始化")
    
    def initialize(self) -> bool:
        try:
            from apis.identify import OfficialRealtimeDiarizer
            logger.info("正在載入語者辨識模型...")
            self.diarization_model = OfficialRealtimeDiarizer(clustering_threshold=0.7)
            logger.info("語者辨識模型載入成功")
            return True
        except Exception as e:
            logger.error(f"語者辨識模型初始化失敗: {e}")
            return False
            
    def _merge_segments(self, diarization_results, max_silence_for_merge=2.0):
        """一個不依賴特定套件版本的手動合併函式。"""
        if not diarization_results:
            return []

        # 確保片段按開始時間排序
        diarization_results.sort(key=lambda x: x[1])

        merged = []
        current_speaker, current_start, current_end = diarization_results[0]

        for i in range(1, len(diarization_results)):
            next_speaker, next_start, next_end = diarization_results[i]

            if (next_speaker == current_speaker and
                (next_start - current_end) < max_silence_for_merge):
                current_end = next_end
            else:
                merged.append((current_speaker, current_start, current_end))
                current_speaker, current_start, current_end = next_speaker, next_start, next_end

        merged.append((current_speaker, current_start, current_end))
        return merged

    def SpeakerAnnote(self, request, context):
        if self.diarization_model is None:
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details("語者辨識模型未初始化")
            return model_service_pb2.SpeakerAnnoteResponse()
        
        try:
            logger.info(f"收到語者辨識請求，音訊數據大小: {len(request.audio_data)} bytes")
            
            import io
            import soundfile as sf
            import numpy as np
            
            audio_buffer = io.BytesIO(request.audio_data)
            audio_data, sample_rate = sf.read(audio_buffer)
            
            if len(audio_data.shape) > 1:
                audio_data = np.mean(audio_data, axis=1)
            audio_data = audio_data.astype(np.float32)
            
            if sample_rate != 16000:
                import librosa
                audio_data = librosa.resample(y=audio_data, orig_sr=sample_rate, target_sr=16000)
                logger.info("已將音訊重新採樣至 16000Hz")
            
            self.diarization_model.process(audio_data)
            
            # 從模型取得 Annotation 物件
            diarization_annotation = self.diarization_model.flush()
            
            # =================== 修正點：將 Annotation 物件轉換為列表 ===================
            # 這是解決 'Annotation' object has no attribute 'sort' 錯誤的關鍵
            raw_segments = []
            for segment, _, speaker in diarization_annotation.itertracks(yield_label=True):
                raw_segments.append((speaker, segment.start, segment.end))
            
            logger.info(f"講者分辨完成，原始片段數量: {len(raw_segments)}")
            # =======================================================================

            # 使用轉換後的列表來進行合併
            merged_results = self._merge_segments(raw_segments)
            logger.info(f"片段合併完成，合併後片段數量: {len(merged_results)}")

            all_segments = []
            speaker_timelines = {}
            
            for speaker, start_time, end_time in merged_results:
                segment = model_service_pb2.DiarizationSegment(
                    speaker=speaker,
                    start_time=float(start_time),
                    end_time=float(end_time)
                )
                all_segments.append(segment)
                
                if speaker not in speaker_timelines:
                    speaker_timelines[speaker] = []
                speaker_timelines[speaker].append(segment)
            
            timeline_objects = []
            for speaker, segments in speaker_timelines.items():
                timeline = model_service_pb2.SpeakerTimeline(
                    speaker=speaker,
                    segments=segments
                )
                timeline_objects.append(timeline)
            
            response = model_service_pb2.SpeakerAnnoteResponse(
                all_segments=all_segments,
                speaker_timelines=timeline_objects
            )
            
            logger.info("講者分辨結果已準備完成")
            return response
            
        except Exception as e:
            logger.error(f"語者辨識處理失敗: {e}")
            import traceback
            logger.error(f"詳細錯誤: {traceback.format_exc()}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"語者辨識處理失敗: {str(e)}")
            return model_service_pb2.SpeakerAnnoteResponse()

class ServerManager:
    """伺服器管理器，負責初始化和管理所有服務"""
    
    def __init__(self):
        self.translator_api = None
        self.tts_servicer = None
        self.wav2lip_servicer = None
        self.speaker_annote_servicer = None
        self.server = None
        
    def initialize_models(self) -> bool:
        try:
            logger.info("正在初始化翻譯服務...")
            self.translator_api = TranslatorService()
            if not self.translator_api.initialize():
                return False
            
            logger.info("正在初始化 TTS 服務...")
            self.tts_servicer = TtsServicer()
            
            logger.info("正在初始化 Wav2Lip 服務...")
            self.wav2lip_servicer = Wav2LipServicer()
            
            logger.info("正在初始化語者辨識服務...")
            self.speaker_annote_servicer = SpeakerAnnoteServicer()
            if not self.speaker_annote_servicer.initialize():
                logger.warning("語者辨識服務初始化失敗，但繼續啟動其他服務")
            
            return True
            
        except Exception as e:
            logger.error(f"服務初始化失敗: {e}")
            return False
    
    def setup_server(self):
        self.server = grpc.server(
            futures.ThreadPoolExecutor(max_workers=10),
            options=[
                ('grpc.max_send_message_length', MAX_MESSAGE_LENGTH),
                ('grpc.max_receive_message_length', MAX_MESSAGE_LENGTH),
            ]
        )
        
        model_service_pb2_grpc.add_TranslatorServiceServicer_to_server(
            TranslatorServicer(self.translator_api), 
            self.server
        )
        
        media_servicer = MediaServicer(
            tts_servicer=self.tts_servicer,
            wav2lip_servicer=self.wav2lip_servicer,
            speaker_annote_servicer=self.speaker_annote_servicer
        )
        model_service_pb2_grpc.add_MediaServiceServicer_to_server(
            media_servicer, 
            self.server
        )
        
        # 修正綁定埠口問題
        self.server.add_insecure_port('0.0.0.0:50051')
        logger.info("gRPC 伺服器設定完成")
    
    def start_server(self):
        if not self.initialize_models():
            logger.error("服務初始化失敗，伺服器無法啟動")
            return
        
        self.setup_server()
        self.server.start()
        
        logger.info("🚀 gRPC 伺服器已成功啟動，監聽埠 50051...")
        
        try:
            while True:
                time.sleep(86400)
        except KeyboardInterrupt:
            logger.info("收到關閉信號，正在關閉伺服器...")
            self.server.stop(0)
            logger.info("伺服器已關閉")
        

def serve():
    server_manager = ServerManager()
    server_manager.start_server()

if __name__ == '__main__':
    serve()