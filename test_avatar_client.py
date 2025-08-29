"""
Dedicated Virtual Avatar gRPC Client Test

這個腳本專門測試虛擬頭像的 gRPC 功能：
1. 檢查先決條件
2. 測試 InitAvatar RPC
3. 測試 AvatarSpeak RPC
4. 互動式測試
5. 壓力測試
"""

import grpc
import sys
import os
import time
import threading
from typing import List, Tuple

# 添加項目根目錄到路徑
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from proto import model_service_pb2
from proto import model_service_pb2_grpc

# gRPC 配置
MAX_MESSAGE_LENGTH = 100 * 1024 * 1024
MAX_METADATA_SIZE = 2 * 1024 * 1024

class VirtualAvatarClient:
    """虛擬頭像 gRPC 客戶端類"""
    
    def __init__(self, server_address='localhost:50051'):
        self.server_address = server_address
        self.channel = None
        self.stub = None
        self.avatar_initialized = False
        
    def connect(self):
        """連接到 gRPC 服務器"""
        print(f"🔗 連接到 gRPC 服務器: {self.server_address}")
        
        channel_options = [
            ('grpc.max_send_message_length', MAX_MESSAGE_LENGTH),
            ('grpc.max_receive_message_length', MAX_MESSAGE_LENGTH),
            ('grpc.max_receive_metadata_size', MAX_METADATA_SIZE),
            ('grpc.max_send_metadata_size', MAX_METADATA_SIZE),
        ]
        
        self.channel = grpc.insecure_channel(self.server_address, options=channel_options)
        self.stub = model_service_pb2_grpc.MediaServiceStub(self.channel)
        
        # 測試連接
        try:
            # 使用 gRPC 健康檢查或簡單的服務調用來測試連接
            grpc.channel_ready_future(self.channel).result(timeout=10)
            print("✅ 成功連接到服務器")
            return True
        except grpc.FutureTimeoutError:
            print("❌ 連接超時")
            return False
        except Exception as e:
            print(f"❌ 連接失敗: {e}")
            return False
    
    def disconnect(self):
        """斷開連接"""
        if self.channel:
            self.channel.close()
            print("🔌 已斷開連接")
    
    def check_prerequisites(self) -> bool:
        """檢查測試先決條件"""
        print("🔍 檢查虛擬頭像測試先決條件...")
        
        # 檢查測試檔案
        test_files = {
            "圖片檔案": "wav2lip_sample/tom.jpg",
            "音頻檔案": "identify_sample/ta.wav"
        }
        
        files_ok = True
        for file_type, file_path in test_files.items():
            if os.path.exists(file_path):
                file_size = os.path.getsize(file_path)
                print(f"✅ {file_type}: {file_path} ({file_size} bytes)")
            else:
                print(f"❌ {file_type}: {file_path} (不存在)")
                files_ok = False
        
        # 檢查Python依賴
        dependencies = [
            ("gRPC", "grpc"),
            ("Protobuf", "google.protobuf")
        ]
        
        deps_ok = True
        for name, module in dependencies:
            try:
                __import__(module)
                print(f"✅ {name}")
            except ImportError:
                print(f"❌ {name}")
                deps_ok = False
        
        result = files_ok and deps_ok
        print(f"📋 先決條件檢查: {'✅ 通過' if result else '❌ 失敗'}")
        return result
    
    def init_avatar(self, image_path: str, audio_path: str) -> bool:
        """初始化虛擬頭像"""
        print(f"\n🎯 初始化虛擬頭像")
        print(f"   圖片: {image_path}")
        print(f"   音頻: {audio_path}")
        
        try:
            # 讀取檔案
            with open(image_path, "rb") as f:
                image_data = f.read()
            print(f"📷 圖片讀取成功: {len(image_data)} bytes")
            
            with open(audio_path, "rb") as f:
                audio_data = f.read()
            print(f"🎵 音頻讀取成功: {len(audio_data)} bytes")
            
            # 創建請求
            request = model_service_pb2.InitAvatarRequest(
                image_data=image_data,
                sample_audio_data=audio_data
            )
            
            # 發送請求
            print("⏳ 發送初始化請求...")
            start_time = time.time()
            
            response = self.stub.InitAvatar(request)
            
            end_time = time.time()
            duration = end_time - start_time
            
            # 處理回應
            if response.success:
                print(f"✅ 頭像初始化成功: {response.message}")
                print(f"⏱️ 初始化時間: {duration:.2f} 秒")
                self.avatar_initialized = True
                return True
            else:
                print(f"❌ 頭像初始化失敗: {response.message}")
                return False
                
        except FileNotFoundError as e:
            print(f"❌ 檔案不存在: {e}")
            return False
        except grpc.RpcError as e:
            print(f"❌ gRPC 錯誤: {e.code()} - {e.details()}")
            return False
        except Exception as e:
            print(f"❌ 初始化錯誤: {e}")
            return False
    
    def avatar_speak(self, text: str, language: str = "en") -> bool:
        """讓頭像說話"""
        if not self.avatar_initialized:
            print("❌ 頭像未初始化，請先調用 init_avatar()")
            return False
        
        print(f"🗣️ 頭像說話: '{text}' ({language})")
        
        try:
            # 創建請求
            request = model_service_pb2.AvatarSpeakRequest(
                text=text,
                language=language
            )
            
            # 發送請求
            print("⏳ 發送說話請求...")
            start_time = time.time()
            
            response = self.stub.AvatarSpeak(request)
            
            end_time = time.time()
            duration = end_time - start_time
            
            # 處理回應
            if response.success:
                print(f"✅ 頭像說話成功: {response.message}")
                print(f"⏱️ 處理時間: {duration:.2f} 秒")
                print("📺 請檢查虛擬攝像頭輸出")
                print("🎵 請檢查虛擬麥克風輸出")
                return True
            else:
                print(f"❌ 頭像說話失敗: {response.message}")
                return False
                
        except grpc.RpcError as e:
            print(f"❌ gRPC 錯誤: {e.code()} - {e.details()}")
            return False
        except Exception as e:
            print(f"❌ 說話錯誤: {e}")
            return False
    
    def run_basic_test(self) -> bool:
        """執行基本功能測試"""
        print("\n🧪 執行基本功能測試")
        print("=" * 40)
        
        # 檢查先決條件
        if not self.check_prerequisites():
            return False
        
        # 初始化頭像
        if not self.init_avatar("wav2lip_sample/tom.jpg", "identify_sample/ta.wav"):
            return False
        
        # 測試說話功能
        test_phrases = [
            ("Hello, I am your virtual avatar!", "en"),
            ("This is a basic functionality test.", "en"),
            ("你好，我是虛擬頭像！", "zh-cn"),
            ("測試基本功能。", "zh-cn")
        ]
        
        success_count = 0
        for i, (text, lang) in enumerate(test_phrases, 1):
            print(f"\n📝 測試 {i}/{len(test_phrases)}")
            if self.avatar_speak(text, lang):
                success_count += 1
            
            # 等待一下再進行下一個測試
            if i < len(test_phrases):
                time.sleep(2)
        
        success_rate = success_count / len(test_phrases) * 100
        print(f"\n📊 基本測試結果: {success_count}/{len(test_phrases)} 成功 ({success_rate:.1f}%)")
        
        return success_rate >= 75  # 75% 成功率視為通過
    
    def run_multilingual_test(self) -> bool:
        """執行多語言測試"""
        print("\n🌍 執行多語言測試")
        print("=" * 40)
        
        if not self.avatar_initialized:
            print("❌ 頭像未初始化")
            return False
        
        multilingual_tests = [
            ("Hello world!", "en"),
            ("你好世界！", "zh-cn"),
            ("Bonjour le monde!", "fr"),
            ("¡Hola mundo!", "es"),
            ("こんにちは世界！", "ja"),
            ("Hallo Welt!", "de"),
            ("Ciao mondo!", "it"),
            ("Olá mundo!", "pt"),
            ("Привет мир!", "ru"),
            ("안녕하세요 세계!", "ko")
        ]
        
        success_count = 0
        for i, (text, lang) in enumerate(multilingual_tests, 1):
            print(f"\n🌐 多語言測試 {i}/{len(multilingual_tests)} ({lang})")
            if self.avatar_speak(text, lang):
                success_count += 1
            time.sleep(1.5)  # 較短的等待時間
        
        success_rate = success_count / len(multilingual_tests) * 100
        print(f"\n📊 多語言測試結果: {success_count}/{len(multilingual_tests)} 成功 ({success_rate:.1f}%)")
        
        return success_rate >= 70
    
    def run_stress_test(self, num_requests: int = 10) -> bool:
        """執行壓力測試"""
        print(f"\n💪 執行壓力測試 ({num_requests} 個請求)")
        print("=" * 40)
        
        if not self.avatar_initialized:
            print("❌ 頭像未初始化")
            return False
        
        test_texts = [
            ("Stress test number", "en"),
            ("壓力測試編號", "zh-cn"),
            ("Performance test", "en"),
            ("性能測試", "zh-cn")
        ]
        
        success_count = 0
        total_time = 0
        
        for i in range(num_requests):
            text_template, lang = test_texts[i % len(test_texts)]
            text = f"{text_template} {i + 1}"
            
            print(f"⚡ 壓力測試 {i + 1}/{num_requests}: '{text}' ({lang})")
            
            start_time = time.time()
            if self.avatar_speak(text, lang):
                success_count += 1
            end_time = time.time()
            
            total_time += (end_time - start_time)
            
            # 不等待，連續發送請求
        
        success_rate = success_count / num_requests * 100
        avg_time = total_time / num_requests
        
        print(f"\n📊 壓力測試結果:")
        print(f"   成功率: {success_count}/{num_requests} ({success_rate:.1f}%)")
        print(f"   平均處理時間: {avg_time:.2f} 秒")
        print(f"   總時間: {total_time:.2f} 秒")
        
        return success_rate >= 80
    
    def run_interactive_test(self):
        """執行互動式測試"""
        print("\n🎮 執行互動式測試")
        print("=" * 40)
        
        if not self.avatar_initialized:
            print("❌ 頭像未初始化")
            return
        
        print("💬 互動式頭像對話 (輸入 'quit' 退出)")
        print("   支援中文和英文，系統會自動檢測語言")
        
        while True:
            try:
                user_input = input("\n👤 你: ").strip()
                
                if user_input.lower() in ['quit', 'exit', '退出', 'q']:
                    print("👋 結束互動式測試")
                    break
                
                if not user_input:
                    print("⚠️ 請輸入有效文字")
                    continue
                
                # 簡單語言檢測
                language = "zh-cn" if any('\u4e00' <= char <= '\u9fff' for char in user_input) else "en"
                
                print(f"🤖 頭像: (說話中... 語言: {language})")
                self.avatar_speak(user_input, language)
                
            except KeyboardInterrupt:
                print("\n👋 用戶中斷，結束互動式測試")
                break
            except Exception as e:
                print(f"❌ 互動式測試錯誤: {e}")
    
    def run_comprehensive_test(self):
        """執行完整測試套件"""
        print("\n🏆 執行完整測試套件")
        print("=" * 50)
        
        test_results = {}
        
        # 1. 基本功能測試
        print("\n1️⃣ 基本功能測試")
        test_results["基本功能"] = self.run_basic_test()
        
        # 2. 多語言測試
        print("\n2️⃣ 多語言測試")
        test_results["多語言"] = self.run_multilingual_test()
        
        # 3. 壓力測試
        print("\n3️⃣ 壓力測試")
        test_results["壓力測試"] = self.run_stress_test(5)  # 5個請求的輕量壓力測試
        
        # 4. 互動式測試（可選）
        try:
            user_choice = input("\n❓ 是否進行互動式測試？(y/n): ").strip().lower()
            if user_choice in ['y', 'yes', '是', 'Y']:
                self.run_interactive_test()
        except KeyboardInterrupt:
            print("\n⏭️ 跳過互動式測試")
        
        # 總結結果
        print(f"\n🏁 測試套件完成")
        print("=" * 50)
        print("📋 測試結果總結:")
        
        passed_tests = 0
        total_tests = len(test_results)
        
        for test_name, result in test_results.items():
            status = "✅ 通過" if result else "❌ 失敗"
            print(f"   {test_name}: {status}")
            if result:
                passed_tests += 1
        
        overall_success_rate = passed_tests / total_tests * 100
        print(f"\n🎯 總體通過率: {passed_tests}/{total_tests} ({overall_success_rate:.1f}%)")
        
        if overall_success_rate >= 75:
            print("🎉 虛擬頭像服務測試通過！")
        else:
            print("⚠️ 虛擬頭像服務需要檢查和改進")

def main():
    """主函數"""
    print("🎭 虛擬頭像 gRPC 客戶端測試")
    print("=" * 60)
    
    # 創建客戶端
    client = VirtualAvatarClient()
    
    try:
        # 連接服務器
        if not client.connect():
            print("❌ 無法連接到服務器，請確保服務器正在運行")
            return
        
        # 詢問測試模式
        print("\n🔧 選擇測試模式:")
        print("1. 完整測試套件 (推薦)")
        print("2. 僅基本功能測試")
        print("3. 僅多語言測試")
        print("4. 僅壓力測試")
        print("5. 僅互動式測試")
        
        try:
            choice = input("\n請選擇 (1-5): ").strip()
        except KeyboardInterrupt:
            print("\n👋 用戶取消")
            return
        
        # 根據選擇執行測試
        if choice == "1":
            client.run_comprehensive_test()
        elif choice == "2":
            if client.check_prerequisites():
                client.init_avatar("wav2lip_sample/tom.jpg", "identify_sample/ta.wav")
                client.run_basic_test()
        elif choice == "3":
            if client.check_prerequisites():
                client.init_avatar("wav2lip_sample/tom.jpg", "identify_sample/ta.wav")
                client.run_multilingual_test()
        elif choice == "4":
            if client.check_prerequisites():
                client.init_avatar("wav2lip_sample/tom.jpg", "identify_sample/ta.wav")
                num_requests = int(input("請輸入測試請求數量 (預設10): ") or "10")
                client.run_stress_test(num_requests)
        elif choice == "5":
            if client.check_prerequisites():
                client.init_avatar("wav2lip_sample/tom.jpg", "identify_sample/ta.wav")
                client.run_interactive_test()
        else:
            print("❌ 無效選擇")
    
    except Exception as e:
        print(f"❌ 測試過程中發生錯誤: {e}")
        import traceback
        print(f"詳細錯誤: {traceback.format_exc()}")
    
    finally:
        # 斷開連接
        client.disconnect()

if __name__ == "__main__":
    main()
