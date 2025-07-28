# 檔案: client.py

import grpc
import model_service_pb2
import model_service_pb2_grpc

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

        # --- 執行多個翻譯測試 ---
        run_translation_test(stub, "Hello world", "英文", "中文")
        run_translation_test(stub, "這是個很棒的系統", "中文", "日文")
        run_translation_test(stub, "Wie geht es Ihnen?", "德文", "英文")
        run_translation_test(stub, "Ceci est un test.", "法文", "西班牙文")
        # 測試一個不支援的語言
        run_translation_test(stub, "Test", "英文", "火星文")
        
if __name__ == '__main__':
    main()