# 檔案: server.py

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

class ImmersiveMultilingualMeetingAssistantServer(model_service_pb2_grpc.TranslatorServiceServicer):
    """gRPC 翻譯服務實現"""
    
    def __init__(self, translator_api: TranslatorService):
        self.translator_api = translator_api
        logger.info("ImmersiveMultilingualMeetingAssistantServer 已初始化")

    def Translate(self, request, context):
        """處理翻譯請求"""
        # 將 gRPC 請求轉換為 API 格式
        request_data = {
            "text": request.text_to_translate,
            "source_lang": request.source_language,
            "target_lang": request.target_language
        }
        
        logger.info(f"收到翻譯請求: {request_data}")
        
        # 使用 API 處理請求
        result = self.translator_api.process_translation_request(request_data)
        
        if result["success"]:
            return model_service_pb2.TranslateResponse(
                translated_text=result["translated_text"]
            )
        else:
            # 設定錯誤狀態
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(result["error"])
            return model_service_pb2.TranslateResponse()


class MediaServicer(model_service_pb2_grpc.MediaServiceServicer):
    """統一的媒體服務實現，整合 TTS、Wav2Lip 和 SpeakerAnnote"""
    
    def __init__(self, tts_servicer, wav2lip_servicer, speaker_annote_servicer):
        self.tts_servicer = tts_servicer
        self.wav2lip_servicer = wav2lip_servicer
        self.speaker_annote_servicer = speaker_annote_servicer
        logger.info("MediaServicer 已初始化")
    
    def Tts(self, request, context):
        """處理 TTS 請求"""
        logger.info("收到 TTS 請求")
        return self.tts_servicer.Tts(request, context)
    
    def Wav2Lip(self, request, context):
        """處理 Wav2Lip 請求"""
        logger.info("收到 Wav2Lip 請求")
        return self.wav2lip_servicer.Wav2Lip(request, context)
    
    def SpeakerAnnote(self, request, context):
        """處理語者辨識請求"""
        logger.info("收到 SpeakerAnnote 請求")
        return self.speaker_annote_servicer.SpeakerAnnote(request, context)


class SpeakerAnnoteServicer:
    """語者辨識服務的包裝器"""
    
    def __init__(self):
        self.diarization_model = None
        logger.info("SpeakerAnnoteServicer 已初始化")
    
    def initialize(self) -> bool:
        """初始化語者辨識模型"""
        try:
            from apis.pyannote import StreamingDiarization
            logger.info("正在載入語者辨識模型...")
            
            self.diarization_model = StreamingDiarization(
                device='cpu',  # 可以根據需要改為 'cuda'
                use_qualcomm_npu=False
            )
            
            logger.info("語者辨識模型載入成功")
            return True
            
        except Exception as e:
            logger.error(f"語者辨識模型初始化失敗: {e}")
            return False
    
    def SpeakerAnnote(self, request, context):
        """處理語者辨識請求"""
        if self.diarization_model is None:
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details("語者辨識模型未初始化")
            return model_service_pb2.SpeakerAnnoteResponse()
        
        try:
            # 這裡需要根據您的 pyannote 實現來處理音訊數據
            # 假設 request.audio_data 是音訊的 bytes 數據
            
            # 暫時返回空的回應，您需要根據實際的 pyannote API 來實現
            logger.info("處理語者辨識請求...")
            
            # TODO: 實際的語者辨識處理邏輯
            # results = self.diarization_model.process(audio_data)
            
            return model_service_pb2.SpeakerAnnoteResponse(
                # 根據您的 proto 定義填入實際數據
            )
            
        except Exception as e:
            logger.error(f"語者辨識處理失敗: {e}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
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
        """初始化所有模型"""
        try:
            # 初始化翻譯服務
            logger.info("正在初始化翻譯服務...")
            self.translator_api = TranslatorService()
            if not self.translator_api.initialize():
                logger.error("翻譯服務初始化失敗")
                return False
            
            # 初始化 TTS 服務
            logger.info("正在初始化 TTS 服務...")
            self.tts_servicer = TtsServicer()
            
            # 初始化 Wav2Lip 服務
            logger.info("正在初始化 Wav2Lip 服務...")
            self.wav2lip_servicer = Wav2LipServicer()
            
            # 初始化語者辨識服務
            logger.info("正在初始化語者辨識服務...")
            self.speaker_annote_servicer = SpeakerAnnoteServicer()
            if not self.speaker_annote_servicer.initialize():
                logger.warning("語者辨識服務初始化失敗，但繼續啟動其他服務")
                # 不返回 False，讓其他服務繼續工作
            
            logger.info("所有服務初始化完成")
            return True
            
        except Exception as e:
            logger.error(f"服務初始化失敗: {e}")
            return False
    
    def setup_server(self):
        """設定 gRPC 伺服器"""
        self.server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
        
        # 註冊翻譯服務
        model_service_pb2_grpc.add_TranslatorServiceServicer_to_server(
            TranslatorServicer(self.translator_api), 
            self.server
        )
        
        # 註冊統一的媒體服務
        media_servicer = MediaServicer(
            tts_servicer=self.tts_servicer,
            wav2lip_servicer=self.wav2lip_servicer,
            speaker_annote_servicer=self.speaker_annote_servicer
        )
        model_service_pb2_grpc.add_MediaServiceServicer_to_server(
            media_servicer, 
            self.server
        )
        
        self.server.add_insecure_port('[::]:50051')
        logger.info("gRPC 伺服器設定完成")
        logger.info("已註冊服務:")
        logger.info("  - TranslatorService (翻譯)")
        logger.info("  - MediaService (TTS, Wav2Lip, SpeakerAnnote)")
    
    def start_server(self):
        """啟動伺服器"""
        if not self.initialize_models():
            logger.error("服務初始化失敗，伺服器無法啟動")
            return False
        
        self.setup_server()
        self.server.start()
        
        logger.info("🚀 gRPC 伺服器已成功啟動，監聽埠 50051...")
        logger.info("伺服器提供以下服務:")
        logger.info("  📝 翻譯服務 (TranslatorService)")
        logger.info("  🎤 TTS 文字轉語音")
        logger.info("  🎬 Wav2Lip 對嘴影片生成")
        logger.info("  👥 語者辨識 (SpeakerAnnote)")
        
        try:
            while True:
                time.sleep(86400)
        except KeyboardInterrupt:
            logger.info("收到關閉信號，正在關閉伺服器...")
            self.server.stop(0)
            logger.info("伺服器已關閉")
        
        return True


def serve():
    """主要服務函式"""
    server_manager = ServerManager()
    server_manager.start_server()


if __name__ == '__main__':
    serve()