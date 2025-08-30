#!/usr/bin/env python3
"""
語音識別服務測試腳本
"""

import grpc
import io
import logging
from proto import model_service_pb2
from proto import model_service_pb2_grpc

# 設定日誌
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_speech_recognition():
    """測試語音識別服務"""
    
    # 連接到 gRPC 伺服器
    channel = grpc.insecure_channel('localhost:50051')
    stub = model_service_pb2_grpc.MediaServiceStub(channel)
    
    try:
        # 檢查是否有測試音訊文件
        test_audio_path = "./tts_sample/en_sample.wav"
        
        try:
            with open(test_audio_path, 'rb') as f:
                audio_data = f.read()
            logger.info(f"載入測試音訊文件: {test_audio_path}, 大小: {len(audio_data)} bytes")
        except FileNotFoundError:
            logger.error(f"找不到測試音訊文件: {test_audio_path}")
            logger.info("請確保有音訊文件可供測試")
            return
        
        # 測試基本語音識別（不含時間戳）
        logger.info("=== 測試基本語音識別 ===")
        request = model_service_pb2.SpeechRecognitionRequest(
            audio_data=audio_data,
            language="auto",  # 自動檢測語言
            return_timestamps=False,
            model_size="base"
        )
        
        response = stub.SpeechRecognition(request)
        
        if response.success:
            logger.info("語音識別成功!")
            logger.info(f"轉錄文本: {response.transcribed_text}")
            logger.info(f"檢測語言: {response.detected_language}")
            logger.info(f"語言置信度: {response.language_confidence}")
        else:
            logger.error("語音識別失敗")
        
        # 測試帶時間戳的語音識別
        logger.info("\n=== 測試帶時間戳的語音識別 ===")
        request_with_timestamps = model_service_pb2.SpeechRecognitionRequest(
            audio_data=audio_data,
            language="auto",
            return_timestamps=True,
            model_size="base"
        )
        
        response_with_timestamps = stub.SpeechRecognition(request_with_timestamps)
        
        if response_with_timestamps.success:
            logger.info("帶時間戳語音識別成功!")
            logger.info(f"轉錄文本: {response_with_timestamps.transcribed_text}")
            logger.info(f"檢測語言: {response_with_timestamps.detected_language}")
            
            if response_with_timestamps.segments:
                logger.info("時間戳片段:")
                for i, segment in enumerate(response_with_timestamps.segments):
                    logger.info(f"  片段 {i+1}: [{segment.start_time:.2f}s - {segment.end_time:.2f}s] {segment.text}")
            else:
                logger.info("無時間戳片段數據")
        else:
            logger.error("帶時間戳語音識別失敗")
        
        # 測試指定語言的語音識別
        logger.info("\n=== 測試指定語言(英語)的語音識別 ===")
        request_en = model_service_pb2.SpeechRecognitionRequest(
            audio_data=audio_data,
            language="en",  # 指定英語
            return_timestamps=False,
            model_size="base"
        )
        
        response_en = stub.SpeechRecognition(request_en)
        
        if response_en.success:
            logger.info("英語語音識別成功!")
            logger.info(f"轉錄文本: {response_en.transcribed_text}")
            logger.info(f"檢測語言: {response_en.detected_language}")
        else:
            logger.error("英語語音識別失敗")
    
    except grpc.RpcError as e:
        logger.error(f"gRPC 錯誤: {e.code()} - {e.details()}")
    except Exception as e:
        logger.error(f"測試失敗: {e}")
    finally:
        channel.close()

def test_service_availability():
    """測試服務可用性"""
    
    channel = grpc.insecure_channel('localhost:50051')
    
    try:
        # 測試連接
        grpc.channel_ready_future(channel).result(timeout=10)
        logger.info("成功連接到 gRPC 伺服器")
        
        # 測試服務是否可用
        stub = model_service_pb2_grpc.MediaServiceStub(channel)
        
        # 嘗試發送一個空請求來測試服務
        test_request = model_service_pb2.SpeechRecognitionRequest(
            audio_data=b"",
            language="auto",
            return_timestamps=False,
            model_size="base"
        )
        
        try:
            response = stub.SpeechRecognition(test_request)
            logger.info("語音識別服務可用")
        except grpc.RpcError as e:
            if e.code() == grpc.StatusCode.INTERNAL:
                logger.info("語音識別服務可用（預期的內部錯誤，因為音訊數據為空）")
            else:
                logger.error(f"語音識別服務不可用: {e.details()}")
        
    except grpc.FutureTimeoutError:
        logger.error("連接伺服器超時")
    except Exception as e:
        logger.error(f"連接失敗: {e}")
    finally:
        channel.close()

if __name__ == "__main__":
    logger.info("語音識別服務測試開始")
    
    # 先測試服務可用性
    test_service_availability()
    
    print("\n" + "="*50)
    
    # 然後測試實際功能
    test_speech_recognition()
    
    logger.info("語音識別服務測試完成")
