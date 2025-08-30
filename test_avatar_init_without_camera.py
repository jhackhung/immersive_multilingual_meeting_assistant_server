#!/usr/bin/env python3
"""
測試虛擬頭像在沒有虛擬攝像頭的情況下也能初始化
"""

import os
import sys
import logging

# 添加項目路徑
sys.path.append(os.path.dirname(__file__))

from apis.virtual_avatar_service import VirtualAvatarService

# 設置日誌
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_avatar_init_without_camera():
    """測試在沒有虛擬攝像頭的情況下初始化頭像"""
    logger.info("測試虛擬頭像初始化（可能沒有虛擬攝像頭）...")
    
    try:
        # 創建虛擬頭像服務
        avatar_service = VirtualAvatarService()
        logger.info("虛擬頭像服務創建成功")
        
        # 準備測試數據
        # 使用一個簡單的測試圖片（創建一個小的 JPEG 數據）
        test_image_data = create_test_image()
        
        # 使用一個簡單的測試音頻（創建一個小的 WAV 數據）
        test_audio_data = create_test_audio()
        
        # 嘗試初始化頭像
        logger.info("嘗試初始化頭像...")
        success = avatar_service.init_avatar(test_image_data, test_audio_data)
        
        if success:
            logger.info("✅ 頭像初始化成功！")
            
            # 檢查設備狀態
            logger.info(f"麥克風狀態: {'✅ 可用' if avatar_service.microphone_initialized else '❌ 不可用'}")
            logger.info(f"攝像頭狀態: {'✅ 可用' if avatar_service.camera_initialized else '❌ 不可用'}")
            
            # 如果至少有一個設備可用，嘗試說話
            if avatar_service.microphone_initialized or avatar_service.camera_initialized:
                logger.info("嘗試讓頭像說話...")
                speak_success = avatar_service.avatar_speak("Hello, this is a test!", "en")
                logger.info(f"說話測試: {'✅ 成功' if speak_success else '❌ 失敗'}")
            else:
                logger.warning("沒有可用的虛擬設備，跳過說話測試")
        else:
            logger.error("❌ 頭像初始化失敗")
        
        # 清理
        avatar_service.cleanup()
        logger.info("清理完成")
        
        return success
        
    except Exception as e:
        logger.error(f"測試失敗: {e}")
        import traceback
        logger.error(f"詳細錯誤: {traceback.format_exc()}")
        return False

def create_test_image():
    """創建一個簡單的測試圖片數據"""
    import cv2
    import numpy as np
    import tempfile
    
    # 創建一個簡單的彩色圖片
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    img[:, :] = [100, 150, 200]  # 藍灰色背景
    
    # 添加一些文字
    cv2.putText(img, "Test Avatar", (200, 240), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
    
    # 保存到臨時文件並讀取字節
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as temp_file:
        cv2.imwrite(temp_file.name, img)
        with open(temp_file.name, 'rb') as f:
            data = f.read()
        os.unlink(temp_file.name)
    
    return data

def create_test_audio():
    """創建一個簡單的測試音頻數據"""
    import numpy as np
    import soundfile as sf
    import tempfile
    
    # 創建一個簡單的正弦波音頻（1秒，440Hz）
    sample_rate = 22050
    duration = 1.0
    frequency = 440.0
    
    t = np.linspace(0, duration, int(sample_rate * duration))
    audio = 0.3 * np.sin(2 * np.pi * frequency * t)
    
    # 保存到臨時文件並讀取字節
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
        sf.write(temp_file.name, audio, sample_rate)
        with open(temp_file.name, 'rb') as f:
            data = f.read()
        os.unlink(temp_file.name)
    
    return data

if __name__ == "__main__":
    success = test_avatar_init_without_camera()
    if success:
        print("\n🎉 測試通過：虛擬頭像可以在沒有虛擬攝像頭的情況下初始化！")
    else:
        print("\n❌ 測試失敗")
    
    sys.exit(0 if success else 1)
