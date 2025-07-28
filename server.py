# 檔案: server.py

from concurrent import futures
import grpc
import time

# 匯入 gRPC 模組
import model_service_pb2
import model_service_pb2_grpc

# 從你的模型檔案中，匯入 MBartTranslator 類別
from models.mbart_translator_model import MBartTranslator

# 伺服器邏輯的實現
class TranslatorServicer(model_service_pb2_grpc.TranslatorServiceServicer):
    
    # 新增 __init__ 方法，用來接收已經載入好的模型實例
    def __init__(self, translator_instance):
        self.translator = translator_instance
        print("TranslatorServicer 已初始化，並接收到模型實例。")

    # 實作 .proto 中定義的 Translate 函式
    def Translate(self, request, context):
        # 1. 從請求中取出所有需要的資料
        text = request.text_to_translate
        source_lang = request.source_language
        target_lang = request.target_language
        
        print(f"[伺服器端] 收到翻譯請求: '{text}' 從 '{source_lang}' 到 '{target_lang}'")
        
        # 2. 使用已載入的模型進行翻譯
        #    這一步會直接呼叫 MBartTranslator 類別中的 translate 方法
        try:
            translated_text = self.translator.translate(text, source_lang, target_lang)
            print(f"[伺服器端] 翻譯完成: '{translated_text}'")
            
            # 3. 準備回應
            return model_service_pb2.TranslateResponse(translated_text=translated_text)

        except Exception as e:
            error_message = f"翻譯時發生錯誤: {e}"
            print(f"[伺服器端] {error_message}")
            # 你可以設定一個錯誤狀態回傳給客戶端
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(error_message)
            return model_service_pb2.TranslateResponse()


def serve():
    # --- 關鍵步驟：伺服器啟動時，預先載入模型 ---
    print("正在初始化翻譯模型...")
    translator = MBartTranslator()
    
    # 執行耗時的模型載入
    model_loaded = translator.load_model()
    
    # 如果模型載入失敗，就不要啟動伺服器
    if not model_loaded:
        print("模型載入失敗，伺服器無法啟動。")
        return
    # ----------------------------------------------------

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))

    # 將我們寫好的服務邏輯 (Servicer) 加入到伺服器中
    # 注意：我們把載入好的 translator 實例傳入 Servicer 的建構子
    model_service_pb2_grpc.add_TranslatorServiceServicer_to_server(
        TranslatorServicer(translator_instance=translator), server
    )

    server.add_insecure_port('[::]:50051')
    print("\n🚀 gRPC 伺服器已成功啟動，模型已載入，監聽埠 50051...")
    server.start()
    
    try:
        while True:
            time.sleep(86400)
    except KeyboardInterrupt:
        server.stop(0)
        print("伺服器已關閉。")

if __name__ == '__main__':
    serve()