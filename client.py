# 檔案: client.py

import grpc
import numpy as np
import wave
from proto import model_service_pb2
from proto import model_service_pb2_grpc

# --- 新增：定義與伺服器端匹配的訊息長度限制 ---
MAX_MESSAGE_LENGTH = 100 * 1024 * 1024

def run_tts_test(stub, text, language, output_filename):
    """一個輔助函式，用來執行單次 TTS 並儲存結果"""
    print(f"\n[客戶端] 發送 TTS 請求: '{text}' (語言: {language})")
    try:
        # 準備請求物件，包含文字和語言
        reference_audio_file_path = "./tts_sample/en_sample.wav"
        with open(reference_audio_file_path, "rb") as f:
            reference_audio = f.read()
        # 準備請求物件，包含所有需要的欄位
        
        request = model_service_pb2.TtsRequest(
            text_to_speak=text,
            language=language,
            reference_audio=reference_audio
        )
        
        # 呼叫遠端的 Tts 服務
        response = stub.Tts(request)
        
        # 檢查回應是否有音訊資料
        if response.generated_audio:
            with open(output_filename, "wb") as f:
                f.write(response.generated_audio)
            print(f"✅ [客戶端] 收到音訊回應，已保存至 {output_filename}")
        else:
            print("❌ [客戶端] 伺服器回傳了空的音訊。")
            
    except grpc.RpcError as e:
        print(f"❌ [客戶端] TTS 請求失敗: {e.code()} - {e.details()}")

def run_translation_test(stub, text, src_lang, tgt_lang):
    """一個輔助函式，用來執行單次翻譯並印出結果"""
    print(f"\n[客戶端] 發送翻譯請求: '{text}' ({src_lang} -> {tgt_lang})")
    
    try:
        # 準備請求物件，包含所有需要的欄位
        request = model_service_pb2.TranslateRequest(
            text_to_translate=text,
            source_language=src_lang,
            target_language=tgt_lang
        )
        
        # 呼叫遠端的 Translate 服務
        response = stub.Translate(request)
        
        print(f"✅ [客戶端] 收到翻譯結果: '{response.translated_text}'")
    except grpc.RpcError as e:
        print(f"❌ [客戶端] 翻譯請求失敗: {e.code()} - {e.details()}")

def run_speaker_identification_test(stub, audio_file_path):
    """一個輔助函式，用來執行講者分辨並印出結果"""
    print(f"\n[客戶端] 發送講者分辨請求: '{audio_file_path}'")
    
    try:
        # 讀取音訊檔案
        with open(audio_file_path, "rb") as f:
            audio_data = f.read()
        
        print(f"📁 音訊檔案大小: {len(audio_data)} bytes")
        
        # 準備請求物件 - 根據實際 proto 定義
        request = model_service_pb2.SpeakerAnnoteRequest(
            audio_data=audio_data
        )
        
        # 呼叫遠端的 SpeakerAnnote 服務
        response = stub.SpeakerAnnote(request)
        
        # 根據實際的 proto 定義處理回應
        print("✅ [客戶端] 講者分辨結果:")
        
        # 處理 all_segments (所有分割片段)
        if response.all_segments:
            print(f"📊 總共找到 {len(response.all_segments)} 個語音片段:")
            for i, segment in enumerate(response.all_segments):
                print(f"  🎤 片段 {i+1}: {segment.speaker} ({segment.start_time:.2f}s - {segment.end_time:.2f}s)")
        
        # 處理 speaker_timelines (按講者分組的時間軸)
        if response.speaker_timelines:
            print(f"\n👥 發現 {len(response.speaker_timelines)} 位講者:")
            for timeline in response.speaker_timelines:
                speaker_name = timeline.speaker
                segment_count = len(timeline.segments)
                total_duration = sum(seg.end_time - seg.start_time for seg in timeline.segments)
                
                print(f"  🗣️  {speaker_name}:")
                print(f"      📈 說話片段數: {segment_count}")
                print(f"      ⏱️  總說話時間: {total_duration:.2f} 秒")
                print(f"      📋 詳細片段:")
                
                for j, segment in enumerate(timeline.segments):
                    duration = segment.end_time - segment.start_time
                    print(f"          {j+1}. {segment.start_time:.2f}s - {segment.end_time:.2f}s ({duration:.2f}s)")
        
        # 如果沒有任何結果
        if not response.all_segments and not response.speaker_timelines:
            print("⚠️ 沒有檢測到任何講者或語音片段")
            print("   這可能是因為:")
            print("   - 音訊檔案太短")
            print("   - 音訊品質不佳")
            print("   - 沒有包含語音內容")
            print("   - 伺服器端模型尚未完全實現")
            
    except FileNotFoundError:
        print(f"❌ [客戶端] 找不到音訊檔案: {audio_file_path}")
    except grpc.RpcError as e:
        print(f"❌ [客戶端] 講者分辨請求失敗: {e.code()} - {e.details()}")
    except Exception as e:
        print(f"❌ [客戶端] 處理講者分辨時發生錯誤: {e}")


def run_wav2lip_test(stub, audio_file_path, image_file_path, output_filename="output_wav2lip.mp4"):
    """一個輔助函式，用來執行 Wav2Lip 對嘴影片生成"""
    print(f"\n[客戶端] 發送 Wav2Lip 請求: 音訊='{audio_file_path}', 圖片='{image_file_path}'")
    
    try:
        # 讀取音訊和圖片檔案
        with open(audio_file_path, "rb") as f:
            audio_data = f.read()
        
        with open(image_file_path, "rb") as f:
            image_data = f.read()
        
        # 準備請求物件
        request = model_service_pb2.Wav2LipRequest(
            audio_data=audio_data,
            image_data=image_data
        )
        
        # 呼叫遠端的 Wav2Lip 服務
        response = stub.Wav2Lip(request)
        
        # 檢查回應是否有影片資料
        if response.video_data:
            with open(output_filename, "wb") as f:
                f.write(response.video_data)
            print(f"✅ [客戶端] 收到 Wav2Lip 影片，已保存至 {output_filename}")
        else:
            print("❌ [客戶端] 伺服器回傳了空的影片資料。")
            
    except FileNotFoundError as e:
        print(f"❌ [客戶端] 找不到檔案: {e}")
    except grpc.RpcError as e:
        print(f"❌ [客戶端] Wav2Lip 請求失敗: {e.code()} - {e.details()}")
    except Exception as e:
        print(f"❌ [客戶端] 處理 Wav2Lip 時發生錯誤: {e}")

def main():
    # 連接到 gRPC 伺服器
    print("🔗 正在連接到 gRPC 伺服器...")
    
    # --- 修改處：在這裡加入 options ---
    channel_options = [
        ('grpc.max_send_message_length', MAX_MESSAGE_LENGTH),
        ('grpc.max_receive_message_length', MAX_MESSAGE_LENGTH),
    ]
    
    with grpc.insecure_channel('localhost:50051', options=channel_options) as channel:
        # 建立客戶端 Stub
        translator_stub = model_service_pb2_grpc.TranslatorServiceStub(channel)
        media_stub = model_service_pb2_grpc.MediaServiceStub(channel)

        print("\n" + "="*60)
        print("🚀 開始測試所有服務功能")
        print("="*60)

        # --- 執行翻譯測試 ---
        print("\n📝 測試翻譯服務:")
        print("-" * 30)
        run_translation_test(translator_stub, "Hello world", "英文", "中文")
        run_translation_test(translator_stub, "這是個很棒的系統", "中文", "日文")
        run_translation_test(translator_stub, "Wie geht es Ihnen?", "德文", "英文")
        run_translation_test(translator_stub, "Ceci est un test.", "法文", "西班牙文")
        # 測試一個不支援的語言
        run_translation_test(translator_stub, "Test", "英文", "火星文")

        # --- 執行 TTS 測試 ---
        print("\n🎤 測試 TTS 服務:")
        print("-" * 30)
        run_tts_test(media_stub, "This is a test of the text to speech API.", "en", "output_en.wav")
        run_tts_test(media_stub, "你好，這是一個語音合成的測試。", "zh-cn", "output_zh-cn.wav")

        # --- 執行講者分辨測試 ---
        print("\n👥 測試講者分辨服務:")
        print("-" * 30)
        
        # 方法 1: 使用現有的音訊檔案（如果存在）
        existing_audio_files = [
            "./identify_sample/ta.wav",
        ]
        
        test_file_found = False
        for audio_file in existing_audio_files:
            try:
                with open(audio_file, 'rb'):
                    print(f"📁 使用現有音訊檔案進行測試: {audio_file}")
                    run_speaker_identification_test(media_stub, audio_file)
                    test_file_found = True
                    break
            except FileNotFoundError:
                continue

        # --- 執行 Wav2Lip 測試（可選，需要圖片檔案） ---
        print("\n🎬 測試 Wav2Lip 服務:")
        print("-" * 30)
        
        # 檢查是否有測試用的圖片檔案
        test_image_files = ["test_face.jpg", "test_face.png", "sample_face.jpg"]
        image_file_found = False
        
        for image_file in test_image_files:
            try:
                with open(image_file, 'rb'):
                    audio_file = "output_en.wav"  # 使用之前生成的音訊
                    try:
                        with open(audio_file, 'rb'):
                            print(f"📁 使用音訊檔案: {audio_file}, 圖片檔案: {image_file}")
                            run_wav2lip_test(media_stub, audio_file, image_file)
                            image_file_found = True
                            break
                    except FileNotFoundError:
                        print(f"⚠️ 找不到音訊檔案 {audio_file}，跳過 Wav2Lip 測試")
                        break
            except FileNotFoundError:
                continue
        
        if not image_file_found:
            print("⚠️ 沒有找到測試用的圖片檔案，跳過 Wav2Lip 測試")
            print("   如需測試 Wav2Lip，請準備 test_face.jpg 或 test_face.png")

        print("\n" + "="*60)
        print("✅ 所有測試完成！")
        print("="*60)

if __name__ == '__main__':
    main()