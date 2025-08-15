#!/usr/bin/env python3
"""
語音識別服務演示腳本
演示如何使用新增的語音識別功能
"""

import os
import tempfile
import logging
from apis.speech_recognition_service import SpeechRecognitionServicer

# 設定日誌
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_test_audio():
    """創建一個簡單的測試音訊文件（使用 TTS）"""
    try:
        from TTS.api import TTS
        import torch
        
        # 初始化 TTS
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        tts = TTS("tts_models/en/ljspeech/tacotron2-DDC").to(device)
        
        # 生成測試音訊
        test_text = "Hello, this is a test of the speech recognition service. The weather is nice today."
        
        # 創建臨時文件
        temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        temp_file.close()
        
        # 生成音訊
        tts.tts_to_file(text=test_text, file_path=temp_file.name)
        
        logger.info(f"測試音訊已生成: {temp_file.name}")
        logger.info(f"原始文本: {test_text}")
        
        return temp_file.name, test_text
        
    except Exception as e:
        logger.error(f"生成測試音訊失敗: {e}")
        return None, None

def test_standalone_speech_recognition():
    """測試獨立的語音識別服務"""
    
    logger.info("=== 語音識別服務獨立測試 ===")
    
    # 初始化語音識別服務
    logger.info("正在初始化語音識別服務...")
    speech_recognizer = SpeechRecognitionServicer(model_size="base")
    
    if not speech_recognizer.initialize():
        logger.error("❌ 語音識別服務初始化失敗")
        return
    
    logger.info("✅ 語音識別服務初始化成功")
    
    # 檢查支援的語言
    supported_languages = speech_recognizer.get_supported_languages()
    logger.info(f"支援的語言: {list(supported_languages.keys())}")
    
    # 檢查模型資訊
    model_info = speech_recognizer.get_model_info()
    logger.info(f"模型資訊: {model_info}")
    
    # 測試現有音訊文件
    test_files = [
        "./tts_sample/en_sample.wav",
        "./tts_sample/segment.wav",
        "./identify_sample/ta.wav"
    ]
    
    for audio_file in test_files:
        if os.path.exists(audio_file):
            logger.info(f"\n--- 測試音訊文件: {audio_file} ---")
            
            try:
                # 讀取音訊文件
                with open(audio_file, 'rb') as f:
                    audio_data = f.read()
                
                # 測試基本轉錄
                logger.info("執行基本語音轉錄...")
                result = speech_recognizer.transcribe_audio(
                    audio_data=audio_data,
                    language="auto",
                    return_timestamps=False
                )
                
                if result["success"]:
                    logger.info(f"✅ 轉錄成功")
                    logger.info(f"   轉錄文本: {result['transcribed_text']}")
                    logger.info(f"   檢測語言: {result['detected_language']}")
                else:
                    logger.error(f"❌ 轉錄失敗: {result.get('error', '未知錯誤')}")
                
                # 測試帶時間戳的轉錄
                logger.info("執行帶時間戳的語音轉錄...")
                result_with_timestamps = speech_recognizer.transcribe_audio(
                    audio_data=audio_data,
                    language="auto",
                    return_timestamps=True
                )
                
                if result_with_timestamps["success"] and result_with_timestamps["segments"]:
                    logger.info(f"✅ 帶時間戳轉錄成功")
                    logger.info(f"   共 {len(result_with_timestamps['segments'])} 個片段:")
                    for i, segment in enumerate(result_with_timestamps["segments"]):
                        logger.info(f"     片段 {i+1}: [{segment['start_time']:.2f}s - {segment['end_time']:.2f}s] {segment['text']}")
                
            except Exception as e:
                logger.error(f"❌ 處理音訊文件 {audio_file} 時出錯: {e}")
        else:
            logger.warning(f"⚠️ 音訊文件不存在: {audio_file}")
    
    # 如果沒有現有音訊文件，嘗試生成測試音訊
    if not any(os.path.exists(f) for f in test_files):
        logger.info("\n--- 生成測試音訊進行演示 ---")
        test_audio_path, original_text = create_test_audio()
        
        if test_audio_path:
            try:
                with open(test_audio_path, 'rb') as f:
                    audio_data = f.read()
                
                logger.info("使用生成的測試音訊進行轉錄...")
                result = speech_recognizer.transcribe_audio(
                    audio_data=audio_data,
                    language="en",
                    return_timestamps=True
                )
                
                if result["success"]:
                    logger.info(f"✅ 轉錄成功")
                    logger.info(f"   原始文本: {original_text}")
                    logger.info(f"   轉錄文本: {result['transcribed_text']}")
                    logger.info(f"   檢測語言: {result['detected_language']}")
                    
                    if result["segments"]:
                        logger.info(f"   時間戳片段:")
                        for i, segment in enumerate(result["segments"]):
                            logger.info(f"     片段 {i+1}: [{segment['start_time']:.2f}s - {segment['end_time']:.2f}s] {segment['text']}")
                
                # 清理臨時文件
                os.unlink(test_audio_path)
                
            except Exception as e:
                logger.error(f"❌ 測試生成的音訊時出錯: {e}")

def demonstrate_features():
    """演示語音識別服務的各種特性"""
    
    logger.info("=== 語音識別服務特性演示 ===")
    
    # 顯示服務資訊
    speech_recognizer = SpeechRecognitionServicer()
    
    logger.info("\n🎤 語音識別服務特性:")
    logger.info("   • 支援多種語言的語音轉文字")
    logger.info("   • 自動語言檢測")
    logger.info("   • 時間戳信息提取")
    logger.info("   • 多種模型大小選擇 (tiny, base, small, medium, large)")
    logger.info("   • 高準確度的 OpenAI Whisper 模型")
    
    supported_languages = speech_recognizer.get_supported_languages()
    logger.info(f"\n🌐 支援的語言 ({len(supported_languages)} 種):")
    for code, name in supported_languages.items():
        logger.info(f"   • {code}: {name}")
    
    logger.info("\n📋 使用場景:")
    logger.info("   • 會議記錄轉錄")
    logger.info("   • 多語言音訊內容分析")
    logger.info("   • 語音助手交互")
    logger.info("   • 音訊內容搜索與索引")
    logger.info("   • 語言學習輔助")
    
    logger.info("\n🔧 API 使用方式:")
    logger.info("   • gRPC 端點: MediaService.SpeechRecognition")
    logger.info("   • 輸入: 音訊 bytes + 語言設定 + 選項")
    logger.info("   • 輸出: 轉錄文本 + 語言檢測 + 時間戳(可選)")

if __name__ == "__main__":
    logger.info("🎤 語音識別服務演示開始")
    
    # 演示服務特性
    demonstrate_features()
    
    print("\n" + "="*60)
    
    # 執行實際測試
    test_standalone_speech_recognition()
    
    logger.info("\n🎤 語音識別服務演示完成")
    logger.info("\n💡 提示: 可以通過 gRPC 客戶端調用 MediaService.SpeechRecognition 使用此服務")
