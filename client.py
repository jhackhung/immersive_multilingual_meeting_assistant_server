# 檔案: client.py

import grpc
from proto import model_service_pb2
from proto import model_service_pb2_grpc

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
    print(f"\n[客戶端] 發送請求: '{text}' ({src_lang} -> {tgt_lang})")
    
    try:
        # 準備請求物件，包含所有需要的欄位
        request = model_service_pb2.TranslateRequest(
            text_to_translate=text,
            source_language=src_lang,
            target_language=tgt_lang
        )
        
        # 呼叫遠端的 Translate 服務
        response = stub.Translate(request)
        
        print(f"✅ [客戶端] 收到回應: '{response.translated_text}'")
    except grpc.RpcError as e:
        print(f"❌ [客戶端] 請求失敗: {e.code()} - {e.details()}")

def main():
    # 連接到 gRPC 伺服器
    with grpc.insecure_channel('localhost:50051') as channel:
        # 建立客戶端 Stub
        stub = model_service_pb2_grpc.TranslatorServiceStub(channel)

        # # --- 執行多個翻譯測試 ---
        run_translation_test(stub, "Hello world", "英文", "中文")
        run_translation_test(stub, "這是個很棒的系統", "中文", "日文")
        run_translation_test(stub, "Wie geht es Ihnen?", "德文", "英文")
        run_translation_test(stub, "Ceci est un test.", "法文", "西班牙文")
        # # 測試一個不支援的語言
        run_translation_test(stub, "Test", "英文", "火星文")

        tts_stub = model_service_pb2_grpc.MediaServiceStub(channel)
        run_tts_test(tts_stub, "This is a test of the text to speech API.", "en", "output_en.wav")
        run_tts_test(tts_stub, "你好，這是一個語音合成的測試。", "zh-cn", "output_zh-cn.wav")
if __name__ == '__main__':
    main()