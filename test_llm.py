import grpc
from proto import model_service_pb2
from proto import model_service_pb2_grpc
import sys
import os
sys.path.append('.')

def test_llm_direct():
    """直接測試 LLM API 服務"""
    print("直接測試 LLM API 服務")
    print("=" * 50)
    
    try:
        # 導入並初始化服務
        from apis.llm_service import LLMServicer
        
        print("🔧 初始化 LLM 服務...")
        servicer = LLMServicer(model_name="microsoft/DialoGPT-medium")  # 使用小模型測試
        
        # 創建假的上下文
        class FakeContext:
            def __init__(self):
                self.error_code = None
                self.error_details = None
            
            def set_code(self, code): 
                self.error_code = code
                print(f"錯誤代碼: {code}")
            
            def set_details(self, details): 
                self.error_details = details
                print(f"錯誤詳情: {details}")
        
        context = FakeContext()
        
        # 測試 1: 文本生成
        print("\n測試文本生成...")
        text_request = model_service_pb2.TextGenerationRequest(
            prompt="The future of artificial intelligence is",
            max_tokens=50,
            temperature=0.7,
            top_p=0.9
        )
        
        print(f"請求: {text_request.prompt}")
        response = servicer.GenerateText(text_request, context)
        
        if response.success:
            print(f"生成成功:")
            print(f"結果: {response.generated_text}")
        else:
            print("生成失敗")
            if context.error_details:
                print(f"錯誤: {context.error_details}")
        
        # 測試 2: 對話
        print("\n測試對話...")
        chat_request = model_service_pb2.ChatCompletionRequest(
            messages=[
                model_service_pb2.ChatMessage(role="system", content="You are a helpful assistant."),
                model_service_pb2.ChatMessage(role="user", content="Hello! Can you tell me about AI?"),
            ],
            max_tokens=80,
            temperature=0.7
        )
        
        print("對話訊息:")
        for msg in chat_request.messages:
            print(f"  {msg.role}: {msg.content}")
        
        chat_response = servicer.ChatCompletion(chat_request, context)
        
        if chat_response.success:
            print(f"對話成功:")
            print(f"回應: {chat_response.response}")
        else:
            print("對話失敗")
            if context.error_details:
                print(f"錯誤: {context.error_details}")
        
        print("\n直接測試完成！")
        return True
        
    except Exception as e:
        print(f"測試錯誤: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_llm_grpc():
    """透過 gRPC 測試 LLM 服務"""
    print("透過 gRPC 測試 LLM 服務")
    print("=" * 50)
    
    try:
        # 連接到 gRPC 伺服器
        print("連接到 gRPC 伺服器...")
        with grpc.insecure_channel('localhost:50051') as channel:
            stub = model_service_pb2_grpc.MediaServiceStub(channel)
            
            # 測試文本生成
            print("\n發送文本生成請求...")
            request = model_service_pb2.TextGenerationRequest(
                prompt="人工智慧的未來發展趨勢是",
                max_tokens=60,
                temperature=0.8
            )
            
            print(f"請求: {request.prompt}")
            response = stub.GenerateText(request)
            
            if response.success:
                print(f"成功!")
                print(f"生成結果: {response.generated_text}")
            else:
                print("失敗")

            # 測試對話
            print("\n發送對話請求...")
            chat_request = model_service_pb2.ChatCompletionRequest(
                messages=[
                    model_service_pb2.ChatMessage(role="user", content="你好！請簡單介紹一下你自己。"),
                ],
                max_tokens=100,
                temperature=0.7
            )
            
            print("對話訊息:")
            for msg in chat_request.messages:
                print(f"  {msg.role}: {msg.content}")
            
            chat_response = stub.ChatCompletion(chat_request)
            
            if chat_response.success:
                print(f"成功!")
                print(f"回應: {chat_response.response}")
            else:
                print("失敗")

        print("\ngRPC 測試完成！")
        return True
                
    except grpc.RpcError as e:
        print(f"gRPC 錯誤: {e.code()} - {e.details()}")
        return False
    except Exception as e:
        print(f"錯誤: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "grpc":
        test_llm_grpc()
    else:
        test_llm_direct()