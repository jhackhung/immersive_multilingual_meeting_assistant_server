import grpc
import logging
import sys
import os

# 添加模型路徑
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from models.llm_service_model import LLMServicer as LLMModel
from proto import model_service_pb2, model_service_pb2_grpc

logger = logging.getLogger(__name__)

class LLMServicer(model_service_pb2_grpc.MediaServiceServicer):
    """LLM API 服務層"""
    
    def __init__(self, model_name="gpt2"):
        """
        初始化 LLM 服務
        
        Args:
            model_name: 使用的模型名稱
            推薦模型:
            - "gpt2" (快速測試)
            - "microsoft/DialoGPT-medium" (對話)
            - "facebook/blenderbot-400M-distill" (聊天)
        """
        self.model_name = model_name
        print(f"🤖 初始化 LLM API 服務 - 模型: {model_name}")
        
        try:
            # 初始化底層模型
            self.llm_model = LLMModel(model_name=model_name)
            print("✅ LLM API 服務初始化完成")
            
        except Exception as e:
            print(f"❌ LLM API 服務初始化失敗: {e}")
            raise

    def GenerateText(self, request, context):
        """處理文本生成請求"""
        try:
            logger.info(f"LLM API: 收到文本生成請求")
            logger.info(f"提示詞: {request.prompt[:100]}...")
            logger.info(f"參數: max_tokens={request.max_tokens}, temperature={request.temperature}")
            
            # 委託給模型層處理
            response = self.llm_model.GenerateText(request, context)
            
            if response.success:
                logger.info(f"LLM API: 文本生成成功，長度: {len(response.generated_text)}")
            else:
                logger.error("LLM API: 文本生成失敗")
            
            return response
            
        except Exception as e:
            error_msg = f"LLM API 文本生成錯誤: {str(e)}"
            logger.error(error_msg)
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(error_msg)
            return model_service_pb2.TextGenerationResponse(
                generated_text="",
                success=False
            )

    def ChatCompletion(self, request, context):
        """處理對話完成請求"""
        try:
            logger.info(f"LLM API: 收到對話完成請求")
            logger.info(f"訊息數量: {len(request.messages)}")
            
            # 記錄對話內容
            for i, message in enumerate(request.messages):
                logger.info(f"訊息 {i+1}: {message.role} - {message.content[:50]}...")
            
            # 委託給模型層處理
            response = self.llm_model.ChatCompletion(request, context)
            
            if response.success:
                logger.info(f"LLM API: 對話完成成功")
                logger.info(f"回應: {response.response[:100]}...")
            else:
                logger.error("LLM API: 對話完成失敗")
            
            return response
            
        except Exception as e:
            error_msg = f"LLM API 對話完成錯誤: {str(e)}"
            logger.error(error_msg)
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(error_msg)
            return model_service_pb2.ChatCompletionResponse(
                response="",
                success=False
            )