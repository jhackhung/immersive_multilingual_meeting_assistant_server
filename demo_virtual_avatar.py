"""
Demo Script for Virtual Avatar Functions

這個腳本演示虛擬頭像的基本功能：
1. InitAvatar(img, sample_audio) - 初始化虛擬頭像
2. AvatarSpeak(text) - 讓頭像說話

注意：需要安裝虛擬攝像頭和虛擬麥克風驅動程式
"""

import os
import sys
import time
import tempfile

# 添加項目根目錄到路徑
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def demo_virtual_avatar():
    """演示虛擬頭像功能"""
    
    print("🎭 虛擬頭像功能演示")
    print("=" * 50)
    
    try:
        # 導入虛擬頭像服務
        from apis.virtual_avatar_service import VirtualAvatarService
        
        # 創建服務實例
        print("⏳ 初始化虛擬頭像服務...")
        avatar_service = VirtualAvatarService()
        print("✅ 服務初始化完成")
        
        # 準備測試數據
        print("\n📁 準備測試數據...")
        
        # 檢查測試檔案
        test_image_path = "wav2lip_sample/tom.jpg"
        test_audio_path = "identify_sample/ta.wav"
        
        if not os.path.exists(test_image_path):
            print(f"❌ 找不到測試圖片: {test_image_path}")
            print("💡 請確保圖片檔案存在，或使用其他圖片檔案")
            return
        
        if not os.path.exists(test_audio_path):
            print(f"❌ 找不到測試音頻: {test_audio_path}")
            print("💡 請確保音頻檔案存在，或使用其他音頻檔案")
            return
        
        # 讀取測試數據
        with open(test_image_path, "rb") as f:
            image_data = f.read()
        print(f"✅ 圖片讀取成功: {len(image_data)} bytes")
        
        with open(test_audio_path, "rb") as f:
            audio_data = f.read()
        print(f"✅ 音頻讀取成功: {len(audio_data)} bytes")
        
        # 1. 初始化頭像
        print("\n🎯 步驟 1: 初始化虛擬頭像...")
        print("⏳ 正在初始化頭像（這可能需要一些時間）...")
        
        success = avatar_service.init_avatar(image_data, audio_data)
        
        if success:
            print("✅ 頭像初始化成功！")
            print("📺 虛擬攝像頭已啟動")
            print("🎵 虛擬麥克風已啟動")
        else:
            print("❌ 頭像初始化失敗")
            return
        
        # 2. 讓頭像說話
        print("\n🗣️ 步驟 2: 測試頭像說話功能...")
        
        test_sentences = [
            ("Hello, I am your virtual avatar!", "en"),
            ("This is a demonstration of avatar speech synthesis.", "en"),
            ("你好，我是虛擬頭像！", "zh-cn")
        ]
        
        for i, (text, language) in enumerate(test_sentences, 1):
            print(f"\n📝 測試 {i}: '{text}' ({language})")
            print("⏳ 正在生成語音和視頻...")
            
            start_time = time.time()
            success = avatar_service.avatar_speak(text, language)
            end_time = time.time()
            
            if success:
                print(f"✅ 頭像說話成功！")
                print(f"⏱️ 處理時間: {end_time - start_time:.2f} 秒")
                print("📺 檢查虛擬攝像頭輸出（例如在 OBS 或視頻會議軟體中）")
                print("🎵 檢查虛擬麥克風輸出（例如在音頻錄製軟體中）")
            else:
                print("❌ 頭像說話失敗")
            
            # 等待一下再進行下一個測試
            if i < len(test_sentences):
                print("⏳ 等待 5 秒再進行下一個測試...")
                time.sleep(5)
        
        print("\n🎉 演示完成！")
        print("\n📋 使用說明:")
        print("1. 在 OBS Studio 中添加 '虛擬攝像頭' 作為視頻源")
        print("2. 在音頻軟體中選擇 'CABLE Input' 作為麥克風")
        print("3. 頭像會在說話時同步嘴型和聲音")
        print("4. 可以在視頻會議軟體中使用虛擬設備")
        
        # 等待用戶輸入再關閉
        input("\n按 Enter 鍵退出並清理資源...")
        
        # 清理資源
        print("🧹 清理資源...")
        avatar_service.cleanup()
        print("✅ 清理完成")
        
    except ImportError as e:
        print(f"❌ 導入模組失敗: {e}")
        print("💡 請確保所有必要的套件都已安裝")
    except Exception as e:
        print(f"❌ 演示過程中發生錯誤: {e}")
        import traceback
        print(f"詳細錯誤: {traceback.format_exc()}")

def check_dependencies():
    """檢查依賴項"""
    print("🔍 檢查依賴項...")
    
    dependencies = [
        ("OpenCV", "cv2"),
        ("NumPy", "numpy"),
        ("SoundFile", "soundfile"),
        ("SoundDevice", "sounddevice"),
        ("PyVirtualCam", "pyvirtualcam"),
        ("LibROSA", "librosa"),
        ("gRPC", "grpc")
    ]
    
    missing = []
    
    for name, module in dependencies:
        try:
            __import__(module)
            print(f"✅ {name}")
        except ImportError:
            print(f"❌ {name} (未安裝)")
            missing.append(name)
    
    if missing:
        print(f"\n⚠️ 缺少依賴項: {', '.join(missing)}")
        print("請使用 pip 或 conda 安裝缺少的套件")
        return False
    else:
        print("✅ 所有依賴項都已安裝")
        return True

def check_virtual_devices():
    """檢查虛擬設備"""
    print("\n🔍 檢查虛擬設備...")
    
    # 檢查虛擬麥克風
    try:
        import sounddevice as sd
        devices = sd.query_devices()
        
        virtual_mic_found = False
        for device in devices:
            if 'cable' in device['name'].lower() and device['max_output_channels'] > 0:
                print(f"✅ 找到虛擬麥克風: {device['name']}")
                virtual_mic_found = True
                break
        
        if not virtual_mic_found:
            print("❌ 未找到虛擬麥克風 (CABLE Input)")
            print("💡 請安裝 VB-Cable 或類似的虛擬音頻設備")
    
    except Exception as e:
        print(f"❌ 檢查音頻設備時出錯: {e}")
    
    # 檢查虛擬攝像頭
    try:
        import pyvirtualcam
        print("✅ PyVirtualCam 可用")
        print("💡 確保已安裝 OBS Virtual Camera 或類似軟體")
    except Exception as e:
        print(f"❌ PyVirtualCam 不可用: {e}")
        print("💡 請安裝 pyvirtualcam 套件")

if __name__ == "__main__":
    print("🎭 虛擬頭像功能演示腳本")
    print("=" * 60)
    
    # 檢查依賴項
    if not check_dependencies():
        print("\n⚠️ 請先安裝缺少的依賴項再運行演示")
        sys.exit(1)
    
    # 檢查虛擬設備
    check_virtual_devices()
    
    print("\n" + "=" * 60)
    
    # 運行演示
    demo_virtual_avatar()
