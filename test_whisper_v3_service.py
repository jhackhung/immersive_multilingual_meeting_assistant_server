import grpc
from proto import model_service_pb2
from proto import model_service_pb2_grpc
import os
import sys

def test_whisper_v3_direct():
    """直接測試 Whisper V3 服務"""
    print("🎙️ 直接測試 Whisper V3 Turbo 服務")
    print("=" * 50)
    
    try:
        # 導入並初始化服務
        from apis.speech_recognition_service import SpeechRecognitionServicer
        
        print("🔧 初始化 Whisper V3 服務...")
        servicer = SpeechRecognitionServicer()
        
        if not servicer.initialize():
            print("❌ 服務初始化失敗")
            return False
        
        # 模型資訊
        model_info = servicer.get_model_info()
        print(f"📋 模型資訊:")
        for key, value in model_info.items():
            print(f"   {key}: {value}")
        
        # 測試音訊檔案
        test_files = [
            "sample.wav",
            "sample.mp3", 
            "sample.mp4",
            "test_audio.wav",
            "chinese_news.wav",
            "./wav2lip_sample/chinese_news.wav"
        ]
        
        # 找到可用的測試檔案
        audio_file = None
        for file in test_files:
            if os.path.exists(file):
                audio_file = file
                break
        
        if audio_file is None:
            print("❌ 找不到測試音訊檔案")
            print("請提供以下任一檔案：")
            for file in test_files:
                print(f"   - {file}")
            return False
        
        print(f"📁 使用測試檔案: {audio_file}")
        
        # 讀取音訊檔案
        with open(audio_file, "rb") as f:
            audio_data = f.read()
        
        print(f"📊 音訊大小: {len(audio_data)} bytes")
        
        # 測試 1: 中文轉錄
        print("\n🇨🇳 測試 1: 中文轉錄")
        print("-" * 30)
        
        result = servicer.transcribe_audio(
            audio_data=audio_data,
            language="zh",
            return_timestamps=False
        )
        
        if result["success"]:
            print(f"✅ 轉錄成功!")
            print(f"📝 結果: {result['transcribed_text']}")
            print(f"🌍 檢測語言: {result['detected_language']}")
            print(f"📊 信心度: {result['language_confidence']:.2f}")
        else:
            print(f"❌ 轉錄失敗: {result.get('error', '未知錯誤')}")
        
        # 測試 2: 帶時間戳的轉錄
        print("\n⏰ 測試 2: 帶時間戳轉錄")
        print("-" * 30)
        
        result_with_timestamps = servicer.transcribe_audio(
            audio_data=audio_data,
            language="zh",
            return_timestamps=True
        )
        
        if result_with_timestamps["success"]:
            print(f"✅ 轉錄成功!")
            print(f"📝 完整文本: {result_with_timestamps['transcribed_text']}")
            
            if result_with_timestamps["segments"]:
                print("\n⏱️ 時間片段:")
                for i, segment in enumerate(result_with_timestamps["segments"], 1):
                    start = segment["start_time"]
                    end = segment["end_time"]
                    text = segment["text"]
                    print(f"  {i:2d}. [{start:6.2f}s - {end:6.2f}s]: {text}")
            else:
                print("❌ 沒有時間片段資訊")
        else:
            print(f"❌ 轉錄失敗: {result_with_timestamps.get('error', '未知錯誤')}")
        
        # 測試 3: 英文轉錄
        print("\n🇺🇸 測試 3: 英文轉錄")
        print("-" * 30)
        
        result_en = servicer.transcribe_audio(
            audio_data=audio_data,
            language="en",
            return_timestamps=False
        )
        
        if result_en["success"]:
            print(f"✅ 轉錄成功!")
            print(f"📝 結果: {result_en['transcribed_text']}")
        else:
            print(f"❌ 轉錄失敗: {result_en.get('error', '未知錯誤')}")
        
        # 測試 4: 自動語言檢測
        print("\n🌐 測試 4: 自動語言檢測")
        print("-" * 30)
        
        result_auto = servicer.transcribe_audio(
            audio_data=audio_data,
            language="auto",
            return_timestamps=False
        )
        
        if result_auto["success"]:
            print(f"✅ 轉錄成功!")
            print(f"📝 結果: {result_auto['transcribed_text']}")
            print(f"🌍 檢測語言: {result_auto['detected_language']}")
        else:
            print(f"❌ 轉錄失敗: {result_auto.get('error', '未知錯誤')}")
        
        print("\n🎉 直接測試完成！")
        return True
        
    except Exception as e:
        print(f"❌ 測試錯誤: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_whisper_v3_grpc():
    """透過 gRPC 測試 Whisper V3 服務"""
    print("🌐 透過 gRPC 測試 Whisper V3 服務")
    print("=" * 50)
    
    try:
        # 連接到 gRPC 伺服器
        print("🔗 連接到 gRPC 伺服器...")
        with grpc.insecure_channel('localhost:50051') as channel:
            grpc.channel_ready_future(channel).result(timeout=10)
            stub = model_service_pb2_grpc.MediaServiceStub(channel)
            
            print("✅ 連接成功！")
            
            # 找測試檔案
            test_files = [
                "sample.wav",
                "chinese_news.wav",
                "./wav2lip_sample/chinese_news.wav"
            ]
            
            audio_file = None
            for file in test_files:
                if os.path.exists(file):
                    audio_file = file
                    break
            
            if audio_file is None:
                print("❌ 找不到測試音訊檔案")
                return False
            
            print(f"📁 使用測試檔案: {audio_file}")
            
            # 讀取音訊檔案
            with open(audio_file, "rb") as f:
                audio_data = f.read()
            
            # 測試基本語音識別
            print("\n🎙️ 測試基本語音識別...")
            request = model_service_pb2.SpeechRecognitionRequest(
                audio_data=audio_data,
                language="zh",
                return_timestamps=False
            )
            
            response = stub.SpeechRecognition(request)
            
            if response.success:
                print(f"✅ 成功!")
                print(f"📝 轉錄結果: {response.transcribed_text}")
                print(f"🌍 檢測語言: {response.detected_language}")
                print(f"📊 信心度: {response.language_confidence:.2f}")
            else:
                print("❌ 失敗")
            
            # 測試帶時間戳的識別
            print("\n⏰ 測試帶時間戳的語音識別...")
            request_with_timestamps = model_service_pb2.SpeechRecognitionRequest(
                audio_data=audio_data,
                language="zh",
                return_timestamps=True
            )
            
            response_with_timestamps = stub.SpeechRecognition(request_with_timestamps)
            
            if response_with_timestamps.success:
                print(f"✅ 成功!")
                print(f"📝 完整文本: {response_with_timestamps.transcribed_text}")
                
                if response_with_timestamps.segments:
                    print("\n⏱️ 時間片段:")
                    for i, segment in enumerate(response_with_timestamps.segments, 1):
                        start = segment.start_time
                        end = segment.end_time
                        text = segment.text
                        print(f"  {i:2d}. [{start:6.2f}s - {end:6.2f}s]: {text}")
                else:
                    print("❌ 沒有時間片段資訊")
            else:
                print("❌ 失敗")
            
            print("\n🎉 gRPC 測試完成！")
            return True
                
    except grpc.RpcError as e:
        print(f"❌ gRPC 錯誤: {e.code()} - {e.details()}")
        return False
    except Exception as e:
        print(f"❌ 錯誤: {e}")
        return False

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "grpc":
        test_whisper_v3_grpc()
    else:
        test_whisper_v3_direct()