"""
翻譯 API 測試檔案
測試 TranslatorService 的各種功能
"""
import sys
import os
import json
from typing import Dict, Any

# 添加專案根目錄到 Python 路徑
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from apis.translator_service import TranslatorService

class TranslatorAPITester:
    """翻譯 API 測試器"""
    
    def __init__(self):
        self.service = TranslatorService()
        self.test_results = []
        
    def print_test_result(self, test_name: str, result: Dict[str, Any], expected: bool = True):
        """印出測試結果"""
        success = result.get("success", False)
        status = "✅ PASS" if success == expected else "❌ FAIL"
        
        print(f"\n{status} {test_name}")
        print(f"結果: {json.dumps(result, ensure_ascii=False, indent=2)}")
        
        self.test_results.append({
            "test_name": test_name,
            "success": success == expected,
            "result": result
        })
        
    def test_valid_translation(self):
        """測試有效的翻譯請求"""
        print("\n" + "=" * 50)
        print("測試 3: 有效翻譯請求")
        print("=" * 50)
        
        test_cases = [
            {
                "name": "英文翻中文",
                "request": {
                    "text": "Hello, how are you?",
                    "source_lang": "英文",
                    "target_lang": "中文"
                }
            },
            {
                "name": "中文翻英文",
                "request": {
                    "text": "你好，今天天氣很好",
                    "source_lang": "中文",
                    "target_lang": "英文"
                }
            },
            {
                "name": "短句翻譯",
                "request": {
                    "text": "謝謝",
                    "source_lang": "中文",
                    "target_lang": "英文"
                }
            }
        ]
        
        for test_case in test_cases:
            result = self.service.process_translation_request(test_case["request"])
            self.print_test_result(test_case["name"], result, expected=True)
            
    def run_all_tests(self):
        """執行所有測試"""
        if not self.service.initialize():
            print("翻譯服務初始化失敗，無法執行測試。")
            return
        
        self.test_valid_translation()
        
        # 總結測試結果
        total_tests = len(self.test_results)
        passed_tests = sum(1 for r in self.test_results if r["success"])
        
        print("\n" + "=" * 50)
        print(f"測試總結: {passed_tests}/{total_tests} 測試通過")
        print("=" * 50)
        
def main():
    """主函數"""
    tester = TranslatorAPITester()
    tester.run_all_tests()

if __name__ == "__main__":
    main()