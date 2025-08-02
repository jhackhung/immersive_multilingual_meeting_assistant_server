from typing import Dict, Any, Optional
import logging

class TranslatorService:
    """翻譯 API 層，處理翻譯相關的業務邏輯"""
    
    def __init__(self):
        self.translator = None
        self.logger = logging.getLogger(__name__)
        
    def initialize(self) -> bool:
        """初始化翻譯模型"""
        try:
            # 在這裡才引入模型，保持 server.py 的清潔
            from models.mbart_translator_model import MBartTranslator
            
            self.logger.info("正在載入翻譯模型...")
            self.translator = MBartTranslator()
            
            if not self.translator.load_model():
                self.logger.error("翻譯模型載入失敗")
                return False
                
            self.logger.info("翻譯模型載入成功")
            return True
            
        except Exception as e:
            self.logger.error(f"翻譯模型初始化失敗: {e}")
            return False
        
    def process_translation_request(self, request_data: Dict[str, str]) -> Dict[str, Any]:
        """
        處理翻譯請求
        
        Args:
            request_data: 包含 text, source_lang, target_lang 的字典
            
        Returns:
            包含翻譯結果或錯誤資訊的字典
        """
        if self.translator is None:
            return {
                "success": False,
                "error": "翻譯模型未初始化",
                "translated_text": ""
            }
            
        try:
            # 驗證輸入
            validation_result = self._validate_input(request_data)
            if not validation_result["is_valid"]:
                return {
                    "success": False,
                    "error": validation_result["error"],
                    "translated_text": ""
                }
            
            text = request_data["text"]
            source_lang = request_data["source_lang"]
            target_lang = request_data["target_lang"]
            
            self.logger.info(f"處理翻譯請求: '{text}' 從 '{source_lang}' 到 '{target_lang}'")
            
            # 執行翻譯
            translated_text = self.translator.translate(text, source_lang, target_lang)
            
            self.logger.info(f"翻譯完成: '{translated_text}'")
            
            return {
                "success": True,
                "translated_text": translated_text,
                "error": ""
            }
            
        except Exception as e:
            error_message = f"翻譯處理失敗: {str(e)}"
            self.logger.error(error_message)
            return {
                "success": False,
                "error": error_message,
                "translated_text": ""
            }
    
    def _validate_input(self, request_data: Dict[str, str]) -> Dict[str, Any]:
        """驗證輸入資料"""
        required_fields = ["text", "source_lang", "target_lang"]
        
        for field in required_fields:
            if field not in request_data or not request_data[field].strip():
                return {
                    "is_valid": False,
                    "error": f"缺少必要欄位: {field}"
                }
        
        # 檢查文字長度
        if len(request_data["text"]) > 1000:
            return {
                "is_valid": False,
                "error": "文字長度超過限制 (1000 字元)"
            }
        
        return {"is_valid": True, "error": ""}
    
    def get_supported_languages(self) -> Dict[str, str]:
        """取得支援的語言清單"""
        if self.translator and hasattr(self.translator, 'supported_languages'):
            return self.translator.supported_languages
        return {}
    
    def health_check(self) -> Dict[str, Any]:
        """健康檢查"""
        if self.translator is None:
            return {
                "status": "unhealthy",
                "model_loaded": False,
                "error": "翻譯模型未初始化"
            }
            
        try:
            # 簡單的翻譯測試
            test_result = self.translator.translate("test", "英文", "中文")
            return {
                "status": "healthy",
                "model_loaded": self.translator.model is not None,
                "test_translation": test_result
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "model_loaded": False,
                "error": str(e)
            }