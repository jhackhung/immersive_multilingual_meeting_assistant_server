from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
import torch

class MBartTranslator:
    """使用 mBART-50 的多語言翻譯器"""
    
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # mBART-50 支援的語言代碼
        self.supported_languages = {
            "中文": "zh_CN",
            "英文": "en_XX", 
            "日文": "ja_XX",
            "韓文": "ko_KR",
            "法文": "fr_XX",
            "德文": "de_DE",
            "西班牙文": "es_XX",
            "義大利文": "it_IT",
            "葡萄牙文": "pt_XX",
            "俄文": "ru_RU",
            "阿拉伯文": "ar_AR",
            "泰文": "th_TH",
            "越南文": "vi_VN",
            "印度文": "hi_IN",
            "土耳其文": "tr_TR",
            "荷蘭文": "nl_XX",
            "波蘭文": "pl_PL",
            "瑞典文": "sv_SE",
            "芬蘭文": "fi_FI",
            "挪威文": "no_XX",
            "丹麥文": "da_DK",
            "希臘文": "el_GR",
            "希伯來文": "he_IL",
            "羅馬尼亞文": "ro_RO",
            "匈牙利文": "hu_HU",
            "捷克文": "cs_CZ",
            "斯洛伐克文": "sk_SK",
            "克羅埃西亞文": "hr_HR",
            "塞爾維亞文": "sr_Cyrl",
            "保加利亞文": "bg_BG",
            "立陶宛文": "lt_LT",
            "拉脫維亞文": "lv_LV",
            "愛沙尼亞文": "et_EE",
            "斯洛維尼亞文": "sl_SI",
            "烏克蘭文": "uk_UA",
            "白俄羅斯文": "be_BY",
            "馬其頓文": "mk_MK",
            "波士尼亞文": "bs_BA",
            "阿爾巴尼亞文": "sq_AL",
            "印尼文": "id_ID",
            "馬來文": "ms_MY",
            "菲律賓文": "tl_XX",
            "緬甸文": "my_MM",
            "高棉文": "km_KH",
            "老撾文": "lo_LA",
            "尼泊爾文": "ne_NP",
            "僧伽羅文": "si_LK",
            "古吉拉特文": "gu_IN",
            "旁遮普文": "pa_IN",
            "泰米爾文": "ta_IN",
            "泰盧固文": "te_IN",
            "卡納達文": "kn_IN",
            "馬拉雅拉姆文": "ml_IN"
        }
    
    def load_model(self):
        """載入 mBART-50 模型"""
        print("載入 mBART-50 模型...")
        print(f"使用設備: {self.device}")
        
        try:
            self.tokenizer = MBart50TokenizerFast.from_pretrained(
                "facebook/mbart-large-50-many-to-many-mmt"
            )
            self.model = MBartForConditionalGeneration.from_pretrained(
                "facebook/mbart-large-50-many-to-many-mmt"
            ).to(self.device)
            
            print("mBART-50 模型載入成功")
            return True
            
        except Exception as e:
            print(f"模型載入失敗: {e}")
            return False
    
    def translate(self, text, source_lang, target_lang):
        """翻譯文本"""
        if self.model is None or self.tokenizer is None:
            if not self.load_model():
                return "模型載入失敗"
        
        # 轉換語言代碼
        src_code = self.get_language_code(source_lang)
        tgt_code = self.get_language_code(target_lang)
        
        if not src_code or not tgt_code:
            return "不支援的語言代碼"
        
        try:
            # 設定來源語言
            self.tokenizer.src_lang = src_code
            
            # 編碼輸入文本
            encoded = self.tokenizer(text, return_tensors="pt").to(self.device)
            
            # 生成翻譯
            generated_tokens = self.model.generate(
                **encoded,
                forced_bos_token_id=self.tokenizer.lang_code_to_id[tgt_code],
                max_length=200,
                num_beams=5,
                early_stopping=True
            )
            
            # 解碼結果
            translation = self.tokenizer.batch_decode(
                generated_tokens, skip_special_tokens=True
            )[0]
            
            return translation
            
        except Exception as e:
            return f"翻譯錯誤: {e}"
    
    def get_language_code(self, lang_input):
        """獲取語言代碼"""
        # 如果輸入是中文名稱
        if lang_input in self.supported_languages:
            return self.supported_languages[lang_input]
        
        # 如果輸入是代碼（如 zh, en）
        for name, code in self.supported_languages.items():
            if lang_input.lower() == code.split('_')[0]:
                return code
        
        # 直接檢查是否是完整代碼
        if lang_input in self.supported_languages.values():
            return lang_input
        
        return None
    
    def show_supported_languages(self):
        """顯示支援的語言"""
        print("=== mBART-50 支援的語言 ===")
        print("格式: 語言名稱 -> 代碼")
        print("-" * 40)
        
        for name, code in self.supported_languages.items():
            short_code = code.split('_')[0]
            print(f"{name:12} -> {short_code:3} ({code})")

def interactive_translate():
    """互動式翻譯"""
    translator = MBartTranslator()
    
    print("mBART-50 多語言翻譯器")
    print("=" * 50)
    
    translator.show_supported_languages()
    
    print("\n使用說明:")
    print("- 可以使用語言名稱（如：中文、英文）")
    print("- 也可以使用短代碼（如：zh、en）")
    print("- 輸入 'quit' 退出，'help' 查看幫助")
    print("- 輸入 'langs' 查看支援語言")
    
    while True:
        try:
            print("\n" + "-" * 40)
            text = input("輸入要翻譯的文字: ").strip()
            
            if text.lower() in ['quit', 'exit', 'q']:
                print("再見！")
                break
            
            if text.lower() == 'help':
                print("使用方法:")
                print("1. 輸入要翻譯的文字")
                print("2. 輸入來源語言（中文/英文/zh/en）")
                print("3. 輸入目標語言（中文/英文/zh/en）")
                continue
            
            if text.lower() == 'langs':
                translator.show_supported_languages()
                continue
            
            if not text:
                continue
            
            source_lang = input("來源語言: ").strip()
            target_lang = input("目標語言: ").strip()
            
            if not source_lang or not target_lang:
                print("請輸入有效的語言")
                continue
            
            print("翻譯中...")
            result = translator.translate(text, source_lang, target_lang)
            
            print(f"\n原文 ({source_lang}): {text}")
            print(f"翻譯 ({target_lang}): {result}")
            
        except KeyboardInterrupt:
            print("\n\n再見！")
            break
        except Exception as e:
            print(f"發生錯誤: {e}")

def quick_translate_demo():
    """快速翻譯示例"""
    translator = MBartTranslator()
    
    test_cases = [
        ("Hello, how are you?", "英文", "中文"),
        ("你好嗎？", "中文", "英文"),
        ("こんにちは", "日文", "中文"),
        ("Bonjour", "法文", "英文"),
        ("Hola, ¿cómo estás?", "西班牙文", "中文"),
        ("Guten Tag", "德文", "英文"),
        ("안녕하세요", "韓文", "中文"),
        ("Привет", "俄文", "英文"),
        ("مرحبا", "阿拉伯文", "英文"),
        ("สวัสดี", "泰文", "英文"),
    ]
    
    print("🚀 mBART-50 翻譯測試")
    print("=" * 50)
    
    for i, (text, source, target) in enumerate(test_cases, 1):
        print(f"\n{i:2d}. {text} ({source} -> {target})")
        result = translator.translate(text, source, target)
        print(f"    翻譯: {result}")