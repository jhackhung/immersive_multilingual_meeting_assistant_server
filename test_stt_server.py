import sys
import os
import asyncio
import logging
from concurrent import futures
import importlib

# Add the project root to sys.path for module discovery
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import grpc
from proto import model_service_pb2_grpc

# 由於 STTService 類別已不在全域作用域，
# 我們需要手動從 apis.stt_service 載入它
from apis.stt_service import STTService

# 設定日誌
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] - %(message)s"
)
logger = logging.getLogger(__name__)

# gRPC 設定
MAX_MESSAGE_LENGTH = 100 * 1024 * 1024  # 100MB
MAX_METADATA_SIZE = 2 * 1024 * 1024     # 2MB

class STTMediaServicer(model_service_pb2_grpc.MediaServiceServicer):
    """
    專門處理 STT 的 MediaService 實現
    只實現 StreamingRecognize 方法，其他方法返回未實現錯誤
    """
    
    def __init__(self, stt_service):
        self.stt_service = stt_service
        logger.info("STTMediaServicer 已初始化")
    
    async def StreamingRecognize(self, request_iterator, context):
        """
        處理雙向串流語音辨識請求
        """
        logger.info("收到 StreamingRecognize 請求")
        try:
            async for response in self.stt_service.StreamingRecognize(request_iterator, context):
                yield response
        except Exception as e:
            logger.error(f"StreamingRecognize 處理失敗: {e}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"STT 服務處理失敗: {str(e)}")
    
    # 其他不需要的方法返回未實現錯誤
    def Tts(self, request, context):
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details("TTS 服務未在此伺服器中實現")
        return None
    
    def Wav2Lip(self, request, context):
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details("Wav2Lip 服務未在此伺服器中實現")
        return None
    
    def SpeakerAnnote(self, request, context):
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details("SpeakerAnnote 服務未在此伺服器中實現")
        return None
    
    def SpeechRecognition(self, request, context):
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details("SpeechRecognition 服務未在此伺服器中實現")
        return None
    
    def GenerateText(self, request, context):
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details("GenerateText 服務未在此伺服器中實現")
        return None
    
    def ChatCompletion(self, request, context):
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details("ChatCompletion 服務未在此伺服器中實現")
        return None
    
    def AnswerQuestionFromDocuments(self, request, context):
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details("AnswerQuestionFromDocuments 服務未在此伺服器中實現")
        return None


class STTServerManager:
    """
    STT 專用伺服器管理器
    """
    
    def __init__(self, port=50052):
        self.port = port
        self.stt_service = None
        self.server = None
        
    def initialize_stt_service(self) -> bool:
        """
        初始化 STT 服務
        """
        try:
            logger.info("🎙️ 正在初始化 STT 串流語音識別服務...")
            
            # 創建 STT 服務實例
            self.stt_service = STTService()
            
            # 💡 修正 1: 調用實例初始化方法
            self.stt_service.initialize()
            
            # 💡 修正 2: 檢查實例變數，而不是舊的全域變數
            if not self.stt_service._model_cache:
                logger.error("❌ STT 服務初始化失敗：模型載入失敗")
                return False
            
            logger.info("✅ STT 服務初始化成功")
            return True
            
        except Exception as e:
            logger.error(f"❌ STT 服務初始化失敗: {e}")
            import traceback
            logger.error(f"詳細錯誤: {traceback.format_exc()}")
            return False
    
    def setup_server(self):
        """
        設定 gRPC 伺服器
        """
        self.server = grpc.aio.server(
            futures.ThreadPoolExecutor(max_workers=10),
            options=[
                ('grpc.max_send_message_length', MAX_MESSAGE_LENGTH),
                ('grpc.max_receive_message_length', MAX_MESSAGE_LENGTH),
                ('grpc.max_receive_metadata_size', MAX_METADATA_SIZE),
                ('grpc.max_send_metadata_size', MAX_METADATA_SIZE),
                ('grpc.keepalive_time_ms', 30000),
                ('grpc.keepalive_timeout_ms', 5000),
                ('grpc.keepalive_permit_without_calls', True),
                ('grpc.http2.max_pings_without_data', 0),
                ('grpc.http2.min_ping_interval_without_data_ms', 300000),
            ]
        )
        
        stt_media_servicer = STTMediaServicer(self.stt_service)
        model_service_pb2_grpc.add_MediaServiceServicer_to_server(
            stt_media_servicer,
            self.server
        )
        
        listen_addr = f'localhost:{self.port}'
        self.server.add_insecure_port(listen_addr)
        
        logger.info(f"🚀 STT gRPC 伺服器設定完成，監聽: {listen_addr}")
    
    async def start_server(self):
        """
        啟動伺服器
        """
        if not self.initialize_stt_service():
            logger.error("💥 STT 服務初始化失敗，伺服器無法啟動")
            return
        
        self.setup_server()
        
        await self.server.start()
        
        logger.info("🎯 STT 伺服器已成功啟動")
        logger.info(f"📡 監聽埠: {self.port}")
        logger.info("🎙️ 可用服務: StreamingRecognize (STT 串流語音識別)")
        logger.info("💡 使用方式:")
        logger.info(f"   - gRPC 端點: localhost:{self.port}")
        logger.info("   - 服務方法: MediaService.StreamingRecognize")
        logger.info("   - 支援串流雙向通訊")
        
        try:
            await self.server.wait_for_termination()
        except KeyboardInterrupt:
            logger.info("🛑 收到關閉信號，正在關閉伺服器...")
            await self.server.stop(grace=5.0)
            logger.info("✅ STT 伺服器已安全關閉")


async def serve_stt(port=50052):
    """
    啟動 STT 專用伺服器
    
    Args:
        port: 監聽埠號，預設 50052
    """
    server_manager = STTServerManager(port=port)
    await server_manager.start_server()


def main():
    """
    主程式入口
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='STT 串流語音識別伺服器')
    parser.add_argument(
        '--port', 
        type=int, 
        default=50052, 
        help='伺服器監聽埠號 (預設: 50052)'
    )
    parser.add_argument(
        '--log-level',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='日誌等級 (預設: INFO)'
    )
    
    args = parser.parse_args()
    
    logging.getLogger().setLevel(getattr(logging, args.log_level))
    
    logger.info("=" * 60)
    logger.info("🎙️  STT 串流語音識別伺服器")
    logger.info("=" * 60)
    logger.info(f"📡 監聽埠: {args.port}")
    logger.info(f"📊 日誌等級: {args.log_level}")
    logger.info("🔄 正在啟動...")
    
    try:
        asyncio.run(serve_stt(port=args.port))
    except Exception as e:
        logger.error(f"💥 伺服器啟動失敗: {e}")
        import traceback
        logger.error(f"詳細錯誤: {traceback.format_exc()}")


if __name__ == '__main__':
    main()