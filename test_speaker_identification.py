import os
import sys
import numpy as np
from typing import List, Tuple

def test_speaker_diarization_direct():
    """直接測試語者辨識功能"""
    print("👥 直接測試語者辨識 (PyAnnote)")
    print("=" * 50)
    
    try:
        # 檢查 .env 檔案
        env_file = ".env"
        if not os.path.exists(env_file):
            print("❌ 找不到 .env 檔案")
            print("請創建 .env 檔案並設定 HUGGINGFACE_TOKEN")
            print("範例:")
            print("HUGGINGFACE_TOKEN=hf_xxxxxxxxxxxxxxxxxx")
            return False
        
        # 導入並初始化服務
        print("🔧 初始化語者辨識服務...")
        from apis.identify import OfficialRealtimeDiarizer
        
        # 初始化 diarizer
        diarizer = OfficialRealtimeDiarizer(clustering_threshold=0.7)
        print("✅ 語者辨識服務初始化成功")
        
        # 顯示系統資訊
        info = diarizer.get_system_info()
        print(f"\n📊 系統資訊:")
        for key, value in info.items():
            print(f"   {key}: {value}")
        
        # 尋找測試音訊檔案
        test_files = [
            "./identify_sample/meeting.mp4",
            # "./identify_sample/ta.wav",
            # "./wav2lip_sample/chinese_news.wav",
            # "sample.wav",
            # "test_audio.wav",
            # "chinese_news.wav"
        ]
        
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
            
            # 創建測試音訊
            print("\n🎵 創建測試音訊...")
            create_test_audio()
            audio_file = "test_multi_speaker.wav"
        
        print(f"📁 使用測試檔案: {audio_file}")
        
        # 檢查檔案資訊
        file_size = os.path.getsize(audio_file)
        print(f"📊 檔案大小: {file_size} bytes ({file_size/1024/1024:.2f} MB)")
        
        # 測試 1: 使用新的 process_file 方法
        print(f"\n🔄 測試 1: 使用 process_file() 方法")
        print("-" * 40)
        
        try:
            segments = diarizer.process_file(audio_file)
            
            if segments:
                print(f"✅ 語者辨識成功！發現 {len(segments)} 個語者片段:")
                print("-" * 50)
                
                # 統計語者
                speakers = set()
                total_duration = 0
                
                for speaker, start, end in segments:
                    speakers.add(speaker)
                    duration = end - start
                    total_duration += duration
                    print(f"🎤 {speaker}: {start:6.2f}s - {end:6.2f}s (時長: {duration:5.2f}s)")
                
                print(f"\n📊 統計:")
                print(f"   總語者數: {len(speakers)}")
                print(f"   語者列表: {', '.join(sorted(speakers))}")
                print(f"   總發言時間: {total_duration:.2f}s")
                
                # 計算每個語者的總時長
                speaker_durations = {}
                for speaker, start, end in segments:
                    duration = end - start
                    speaker_durations[speaker] = speaker_durations.get(speaker, 0) + duration
                
                print(f"\n⏱️ 各語者總時長:")
                for speaker in sorted(speaker_durations.keys()):
                    duration = speaker_durations[speaker]
                    percentage = (duration / total_duration) * 100 if total_duration > 0 else 0
                    print(f"   {speaker}: {duration:5.2f}s ({percentage:4.1f}%)")
                
                # 如果是會議檔案，額外分析
                if "meeting" in audio_file.lower():
                    print(f"\n🏢 會議分析:")
                    if len(speakers) > 1:
                        avg_duration = total_duration / len(speakers)
                        print(f"   平均每人發言: {avg_duration:.2f}s")
                        
                        most_active = max(speaker_durations.items(), key=lambda x: x[1])
                        print(f"   最活躍語者: {most_active[0]} ({most_active[1]:.2f}s)")
                        
                        least_active = min(speaker_durations.items(), key=lambda x: x[1])
                        print(f"   最少發言語者: {least_active[0]} ({least_active[1]:.2f}s)")
                    else:
                        print("   單人會議或音訊品質問題")
            else:
                print("❌ 未檢測到語者片段")
                print("💡 可能原因:")
                print("   - 音訊品質差")
                print("   - 只有一個語者")
                print("   - 背景噪音過大")
                print("   - 檔案格式不支援")
                
        except Exception as e:
            print(f"❌ process_file() 方法失敗: {e}")
            import traceback
            traceback.print_exc()
        
        # 測試 2: 使用 process_bytes 方法
        print(f"\n🔄 測試 2: 使用 process_bytes() 方法")
        print("-" * 40)
        
        try:
            # 讀取檔案為 bytes
            with open(audio_file, 'rb') as f:
                audio_data = f.read()
            
            print(f"📊 讀取 {len(audio_data)} bytes 數據")
            
            segments = diarizer.process_bytes(audio_data)
            
            if segments:
                print(f"✅ bytes 處理成功！發現 {len(segments)} 個語者片段:")
                
                speakers = set()
                for speaker, start, end in segments:
                    speakers.add(speaker)
                    duration = end - start
                    print(f"🎤 {speaker}: {start:6.2f}s - {end:6.2f}s (時長: {duration:5.2f}s)")
                
                print(f"\n📊 bytes 處理統計:")
                print(f"   總語者數: {len(speakers)}")
                print(f"   語者列表: {', '.join(sorted(speakers))}")
            else:
                print("❌ bytes 處理未檢測到語者片段")
                
        except Exception as e:
            print(f"❌ process_bytes() 方法失敗: {e}")
            import traceback
            traceback.print_exc()
        
        # 測試 3: 傳統的分段處理（如果前面兩個方法都成功的話）
        if segments:  # 如果上面有成功檢測到語者
            print(f"\n🔄 測試 3: 傳統分段處理模式")
            print("-" * 40)
            
            try:
                # 使用 diarizer 的私有方法載入音訊
                audio = diarizer._load_audio_from_file(audio_file)
                
                if audio is not None:
                    print(f"✅ 音訊載入成功")
                    print(f"   長度: {len(audio)} 採樣點")
                    print(f"   採樣率: 16000 Hz")
                    print(f"   時長: {len(audio)/16000:.2f} 秒")
                    
                    # 重置 diarizer
                    diarizer.reset()
                    
                    # 分段處理（模擬即時處理）
                    chunk_size = 16000 * 2  # 2 秒為一段
                    total_chunks = len(audio) // chunk_size + (1 if len(audio) % chunk_size > 0 else 0)
                    
                    print(f"將音訊分為 {total_chunks} 段處理...")
                    
                    for i in range(0, len(audio), chunk_size):
                        chunk = audio[i:i+chunk_size]
                        print(f"處理第 {i//chunk_size + 1}/{total_chunks} 段 (長度: {len(chunk)} 採樣點)")
                        
                        # 處理音訊段
                        segments_chunk = diarizer.process(chunk)
                        if segments_chunk:
                            print(f"   發現 {len(segments_chunk)} 個語者片段")
                            for speaker, start, end in segments_chunk:
                                print(f"     {speaker}: {start:.2f}s - {end:.2f}s")
                    
                    # 獲取最終結果
                    print("\n🏁 獲取分段處理最終結果...")
                    final_segments = diarizer.flush()
                    
                    if final_segments:
                        print(f"✅ 分段處理完成！發現 {len(final_segments)} 個語者片段:")
                        
                        speakers = set()
                        for speaker, start, end in final_segments:
                            speakers.add(speaker)
                            duration = end - start
                            print(f"🎤 {speaker}: {start:6.2f}s - {end:6.2f}s (時長: {duration:5.2f}s)")
                        
                        print(f"\n📊 分段處理統計:")
                        print(f"   總語者數: {len(speakers)}")
                        print(f"   語者列表: {', '.join(sorted(speakers))}")
                    else:
                        print("❌ 分段處理未檢測到語者片段")
                else:
                    print("❌ 無法載入音訊檔案進行分段處理")
                    
            except Exception as e:
                print(f"❌ 分段處理失敗: {e}")
        
        print("\n🎉 語者辨識測試完成！")
        return True
        
    except ImportError as e:
        print(f"❌ 導入錯誤: {e}")
        print("請安裝必要套件:")
        print("pip install pyannote.audio torch torchaudio")
        return False
    except Exception as e:
        print(f"❌ 測試錯誤: {e}")
        import traceback
        traceback.print_exc()
        return False

def create_test_audio():
    """創建測試用的多語者音訊"""
    print("🎵 創建測試音訊檔案...")
    
    try:
        import numpy as np
        import soundfile as sf
        
        # 創建簡單的測試音訊（模擬多語者）
        sr = 16000
        duration = 10  # 10 秒
        t = np.linspace(0, duration, sr * duration)
        
        # 語者 1: 低頻音調 (0-4秒)
        freq1 = 200
        speaker1 = np.sin(2 * np.pi * freq1 * t[:sr*4]) * 0.5
        
        # 靜音間隔 (4-5秒)
        silence = np.zeros(sr * 1)
        
        # 語者 2: 高頻音調 (5-9秒)
        freq2 = 400
        speaker2 = np.sin(2 * np.pi * freq2 * t[:sr*4]) * 0.5
        
        # 最後靜音 (9-10秒)
        final_silence = np.zeros(sr * 1)
        
        # 合併音訊
        test_audio = np.concatenate([speaker1, silence, speaker2, final_silence])
        
        # 保存為 WAV 檔案
        sf.write("test_multi_speaker.wav", test_audio, sr)
        print("✅ 測試音訊已創建: test_multi_speaker.wav")
        
    except Exception as e:
        print(f"❌ 創建測試音訊失敗: {e}")

def test_with_grpc():
    """透過 gRPC 測試語者辨識"""
    print("🌐 透過 gRPC 測試語者辨識")
    print("=" * 50)
    
    try:
        import grpc
        from proto import model_service_pb2
        from proto import model_service_pb2_grpc
        
        # 連接到 gRPC 伺服器
        print("🔗 連接到 gRPC 伺服器...")
        with grpc.insecure_channel('localhost:50051') as channel:
            grpc.channel_ready_future(channel).result(timeout=10)
            stub = model_service_pb2_grpc.MediaServiceStub(channel)
            
            print("✅ 連接成功！")
            
            # 找測試檔案
            test_files = [
                "./identify_sample/meeting.mp4",
                "./identify_sample/ta.wav",
                "./wav2lip_sample/chinese_news.wav",
                "test_multi_speaker.wav"
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
            
            print(f"📊 音訊大小: {len(audio_data)} bytes")
            
            # 測試語者辨識
            print("\n👥 測試語者辨識...")
            request = model_service_pb2.SpeakerAnnoteRequest(
                audio_data=audio_data
            )
            
            response = stub.SpeakerAnnote(request)
            
            if response.success and response.segments:
                print(f"✅ 成功！檢測到 {len(response.segments)} 個語者片段:")
                
                speakers = set()
                for segment in response.segments:
                    speakers.add(segment.speaker_id)
                    duration = segment.end_time - segment.start_time
                    print(f"🎤 {segment.speaker_id}: {segment.start_time:6.2f}s - {segment.end_time:6.2f}s (時長: {duration:5.2f}s)")
                
                print(f"\n📊 統計:")
                print(f"   總語者數: {len(speakers)}")
                print(f"   語者列表: {', '.join(sorted(speakers))}")
            else:
                print("❌ 語者辨識失敗或未檢測到語者")
            
            print("\n🎉 gRPC 測試完成！")
            return True
                
    except ImportError:
        print("❌ gRPC 測試需要啟動 server")
        print("請先執行: python server.py")
        return False
    except Exception as e:
        print(f"❌ gRPC 測試錯誤: {e}")
        return False

def check_environment():
    """檢查環境設定"""
    print("🔍 檢查環境設定")
    print("=" * 30)
    
    # 檢查 .env 檔案
    env_file = ".env"
    if os.path.exists(env_file):
        print("✅ .env 檔案存在")
        
        # 檢查 token
        from dotenv import load_dotenv
        load_dotenv()
        token = os.getenv("HUGGINGFACE_TOKEN")
        
        if token:
            print(f"✅ HUGGINGFACE_TOKEN 已設定 (長度: {len(token)})")
        else:
            print("❌ HUGGINGFACE_TOKEN 未設定")
            return False
    else:
        print("❌ .env 檔案不存在")
        print("請創建 .env 檔案並設定:")
        print("HUGGINGFACE_TOKEN=hf_xxxxxxxxxxxxxxxxxx")
        return False
    
    # 檢查必要套件
    required_packages = [
        "pyannote.audio",
        "torch", 
        "torchaudio",
        "librosa",
        "soundfile"
    ]
    
    print("\n📦 檢查必要套件:")
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace(".", "_") if "." in package else package)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package}")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n❌ 缺少套件: {', '.join(missing_packages)}")
        print("請安裝:")
        print(f"pip install {' '.join(missing_packages)}")
        return False
    
    print("\n✅ 環境檢查通過！")
    return True

if __name__ == "__main__":
    import sys
    
    # 檢查參數
    if len(sys.argv) > 1:
        if sys.argv[1] == "grpc":
            test_with_grpc()
        elif sys.argv[1] == "env":
            check_environment()
        else:
            test_speaker_diarization_direct()
    else:
        # 預設流程
        print("🚀 語者辨識測試程式 (增強版)")
        print("=" * 40)
        
        # 先檢查環境
        # if check_environment():
        #     print("\n" + "="*50)
        test_speaker_diarization_direct()
        # else:
        #     print("\n❌ 環境檢查失敗，請先修復環境問題")