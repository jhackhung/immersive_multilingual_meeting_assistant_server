#!/usr/bin/env python3
"""
測試新的 SpeechRecognition 服務實現
"""

import sys
import os
sys.path.append(os.path.dirname(__file__))

from apis.speech_recognition_service import SpeechRecognitionServicer
import logging

# 設定日誌
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_speech_recognition_service():
    """測試語音識別服務"""
    logger.info("=== 測試 SpeechRecognition 服務 ===")
    
    # 初始化服務
    service = SpeechRecognitionServicer()
    
    # 測試初始化
    logger.info("測試服務初始化...")
    if service.initialize():
        logger.info("✅ 服務初始化成功")
    else:
        logger.error("❌ 服務初始化失敗")
        return False
    
    # 測試獲取支援的語言
    logger.info("測試獲取支援的語言...")
    languages = service.get_supported_languages()
    logger.info(f"支援的語言: {list(languages.keys())}")
    
    # 測試獲取模型資訊
    logger.info("測試獲取模型資訊...")
    model_info = service.get_model_info()
    logger.info("模型資訊:")
    for key, value in model_info.items():
        logger.info(f"  {key}: {value}")
    
    # 測試使用測試音頻文件（如果存在）
    test_audio_files = [
        "tts_sample/en_sample.wav",
        "tts_sample/oswin.wav",
        "identify_sample/ta.wav"
    ]
    
    for audio_file in test_audio_files:
        if os.path.exists(audio_file):
            logger.info(f"測試音頻文件: {audio_file}")
            try:
                with open(audio_file, 'rb') as f:
                    audio_data = f.read()
                
                result = service.transcribe_audio(
                    audio_data=audio_data,
                    language="auto",
                    return_timestamps=True
                )
                
                if result["success"]:
                    logger.info(f"✅ 轉錄成功: {result['transcribed_text']}")
                    logger.info(f"檢測語言: {result['detected_language']}")
                    logger.info(f"信心度: {result['language_confidence']}")
                    if result['segments']:
                        logger.info(f"時間段: {len(result['segments'])} 個")
                else:
                    logger.warning(f"⚠️ 轉錄失敗: {result['error']}")
                    
            except Exception as e:
                logger.error(f"❌ 測試音頻文件時發生錯誤: {e}")
        else:
            logger.info(f"跳過不存在的測試文件: {audio_file}")
    
    logger.info("=== 測試完成 ===")
    return True

def main():
    """主函數"""
    try:
        test_speech_recognition_service()
    except Exception as e:
        logger.error(f"測試過程中發生錯誤: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
