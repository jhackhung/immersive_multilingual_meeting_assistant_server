"""
Virtual Avatar Test Client - 虛擬頭像測試客戶端

這個腳本演示如何使用虛擬頭像服務：
1. 初始化頭像（提供圖片和樣本音頻）
2. 讓頭像說話（輸入文字，生成語音和對嘴視頻）
"""

import grpc
import sys
import os
import time

# 添加項目根目錄到路徑
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from proto import model_service_pb2
from proto import model_service_pb2_grpc

def test_virtual_avatar():
    """測試虛擬頭像功能"""
    
    # 連接到 gRPC 服務器
    channel = grpc.insecure_channel('localhost:50051')
    stub = model_service_pb2_grpc.MediaServiceStub(channel)
    
    print("🎭 虛擬頭像測試開始...")
    
    try:
        # 1. 準備測試數據
        print("📁 準備測試數據...")
        
        # 使用測試圖片（如果存在）
        test_image_path = "wav2lip_sample/tom.jpg"  # 從workspace結構中看到的圖片
        if not os.path.exists(test_image_path):
            print(f"❌ 找不到測試圖片: {test_image_path}")
            print("請確保圖片檔案存在，或修改路徑")
            return
        
        # 使用測試音頻（如果存在）
        # 從workspace結構中看到的音頻
        test_audio_path = "tts_sample\en_sample.wav"
        if not os.path.exists(test_audio_path):
            print(f"❌ 找不到測試音頻: {test_audio_path}")
            print("請確保音頻檔案存在，或修改路徑")
            return
        
        # 讀取圖片數據
        with open(test_image_path, "rb") as f:
            image_data = f.read()
        print(f"✅ 圖片讀取成功: {len(image_data)} bytes")
        
        # 讀取音頻數據
        with open(test_audio_path, "rb") as f:
            audio_data = f.read()
        print(f"✅ 音頻讀取成功: {len(audio_data)} bytes")
        
        # 2. 初始化虛擬頭像
        print("\n🎯 正在初始化虛擬頭像...")
        
        init_request = model_service_pb2.InitAvatarRequest(
            image_data=image_data,
            sample_audio_data=audio_data
        )
        
        try:
            init_response = stub.InitAvatar(init_request)
            
            if init_response.success:
                print(f"✅ 頭像初始化成功: {init_response.message}")
            else:
                print(f"❌ 頭像初始化失敗: {init_response.message}")
                return
                
        except grpc.RpcError as e:
            print(f"❌ gRPC 錯誤: {e.code()} - {e.details()}")
            return
        
        # 3. 讓頭像說話測試
        print("\n🗣️ 測試頭像說話功能...")
        
        test_texts = [
            ("Hello, I am your virtual avatar!", "en"),
            ("你好，我是你的虛擬頭像！", "zh-cn"),
            ("Nice to meet you!", "en")
        ]
        
        for i, (text, language) in enumerate(test_texts, 1):
            print(f"\n📝 測試 {i}: '{text}' ({language})")
            
            speak_request = model_service_pb2.AvatarSpeakRequest(
                text=text,
                language=language
            )
            
            try:
                print("⏳ 正在生成語音和視頻...")
                start_time = time.time()
                
                speak_response = stub.AvatarSpeak(speak_request)
                
                end_time = time.time()
                duration = end_time - start_time
                
                if speak_response.success:
                    print(f"✅ 頭像說話成功: {speak_response.message}")
                    print(f"⏱️ 處理時間: {duration:.2f} 秒")
                    print("📺 請檢查虛擬攝像頭輸出（如 OBS 或視頻通話軟體）")
                    print("🎵 請檢查虛擬麥克風輸出（如音頻錄製軟體）")
                else:
                    print(f"❌ 頭像說話失敗: {speak_response.message}")
                
                # 等待一下再進行下一個測試
                if i < len(test_texts):
                    print("⏳ 等待 3 秒再進行下一個測試...")
                    time.sleep(3)
                    
            except grpc.RpcError as e:
                print(f"❌ gRPC 錯誤: {e.code()} - {e.details()}")
        
        print("\n🎉 虛擬頭像測試完成！")
        print("\n📋 測試結果總結:")
        print("- 如果看到 '頭像說話成功' 訊息，表示服務正常運行")
        print("- 請檢查虛擬攝像頭是否有視頻輸出")
        print("- 請檢查虛擬麥克風是否有音頻輸出")
        print("- 如果沒有輸出，請檢查虛擬音視頻設備是否正確安裝")
        
    except Exception as e:
        print(f"❌ 測試過程中發生錯誤: {e}")
        import traceback
        print(f"詳細錯誤: {traceback.format_exc()}")
    
    finally:
        channel.close()

def check_virtual_devices():
    """檢查虛擬設備狀態"""
    print("🔍 檢查虛擬設備狀態...")
    
    try:
        import sounddevice as sd
        print("\n🎵 可用音頻設備:")
        devices = sd.query_devices()
        for i, device in enumerate(devices):
            if 'cable' in device['name'].lower() or 'virtual' in device['name'].lower():
                print(f"  📢 {i}: {device['name']} (虛擬設備)")
            elif device['max_output_channels'] > 0:
                print(f"  🔊 {i}: {device['name']}")
    except Exception as e:
        print(f"❌ 無法檢查音頻設備: {e}")
    
    try:
        import pyvirtualcam
        print("\n📹 虛擬攝像頭狀態:")
        # 嘗試檢查虛擬攝像頭
        print("  📷 pyvirtualcam 模組可用")
    except Exception as e:
        print(f"❌ 虛擬攝像頭不可用: {e}")

if __name__ == "__main__":
    print("🎭 虛擬頭像功能測試")
    print("=" * 50)
    
    # 首先檢查虛擬設備
    check_virtual_devices()
    
    print("\n" + "=" * 50)
    
    # 測試虛擬頭像功能
    test_virtual_avatar()
