import grpc
import numpy as np
import wave
from proto import model_service_pb2
from proto import model_service_pb2_grpc

# --- 新增：定義與伺服器端匹配的訊息長度限制 ---
MAX_MESSAGE_LENGTH = 100 * 1024 * 1024
# --- 增加元數據大小限制 ---
MAX_METADATA_SIZE = 2 * 1024 * 1024  # 2MB

def run_rag_qa_test(stub, query):
    """測試 RAG 問答功能"""
    print(f"\n[客戶端] 發送 RAG 問答請求: '{query}'")
    
    try:
        # 準備請求物件
        request = model_service_pb2.AnswerQuestionRequest(query=query)
        
        # 呼叫遠端的 AnswerQuestionFromDocuments 服務
        response = stub.AnswerQuestionFromDocuments(request)
        
        # 檢查回應
        if response.success:
            print(f"[客戶端] RAG 問答成功:")
            print(f"模型回答: {response.answer}")
            if response.sources:
                print(f"參考來源:")
                for source in response.sources:
                    print(f"  - {source}")
        else:
            print("[客戶端] RAG 問答失敗")
            
    except grpc.RpcError as e:
        print(f"[客戶端] RAG 問答請求失敗: {e.code()} - {e.details()}")
    except Exception as e:
        print(f"[客戶端] 處理 RAG 問答時發生錯誤: {e}")

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
            print(f"[客戶端] 收到音訊回應，已保存至 {output_filename}")
        else:
            print("[客戶端] 伺服器回傳了空的音訊。")
            
    except grpc.RpcError as e:
        print(f"[客戶端] TTS 請求失敗: {e.code()} - {e.details()}")

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
        
        print(f"[客戶端] 收到翻譯結果: '{response.translated_text}'")
    except grpc.RpcError as e:
        print(f"[客戶端] 翻譯請求失敗: {e.code()} - {e.details()}")

def run_speaker_identification_test(stub, audio_file_path):
    """一個輔助函式，用來執行講者分辨並印出結果"""
    print(f"\n[客戶端] 發送講者分辨請求: '{audio_file_path}'")
    
    try:
        # 讀取音訊檔案
        with open(audio_file_path, "rb") as f:
            audio_data = f.read()
        
        print(f"音訊檔案大小: {len(audio_data)} bytes")
        
        # 準備請求物件 - 根據實際 proto 定義
        request = model_service_pb2.SpeakerAnnoteRequest(
            audio_data=audio_data
        )
        
        # 呼叫遠端的 SpeakerAnnote 服務
        response = stub.SpeakerAnnote(request)
        
        # 根據實際的 proto 定義處理回應
        print("[客戶端] 講者分辨結果:")
        
        # 處理 all_segments (所有分割片段)
        if response.all_segments:
            print(f"總共找到 {len(response.all_segments)} 個語音片段:")
            for i, segment in enumerate(response.all_segments):
                print(f"  片段 {i+1}: {segment.speaker} ({segment.start_time:.2f}s - {segment.end_time:.2f}s)")
        
        # 處理 speaker_timelines (按講者分組的時間軸)
        if response.speaker_timelines:
            print(f"\n 發現 {len(response.speaker_timelines)} 位講者:")
            for timeline in response.speaker_timelines:
                speaker_name = timeline.speaker
                segment_count = len(timeline.segments)
                total_duration = sum(seg.end_time - seg.start_time for seg in timeline.segments)
                
                print(f"    {speaker_name}:")
                print(f"       說話片段數: {segment_count}")
                print(f"       總說話時間: {total_duration:.2f} 秒")
                print(f"       詳細片段:")
                
                for j, segment in enumerate(timeline.segments):
                    duration = segment.end_time - segment.start_time
                    print(f"          {j+1}. {segment.start_time:.2f}s - {segment.end_time:.2f}s ({duration:.2f}s)")
        
        # 如果沒有任何結果
        if not response.all_segments and not response.speaker_timelines:
            print(" 沒有檢測到任何講者或語音片段")
            print("   這可能是因為:")
            print("   - 音訊檔案太短")
            print("   - 音訊品質不佳")
            print("   - 沒有包含語音內容")
            print("   - 伺服器端模型尚未完全實現")
            
    except FileNotFoundError:
        print(f"[客戶端] 找不到音訊檔案: {audio_file_path}")
    except grpc.RpcError as e:
        print(f"[客戶端] 講者分辨請求失敗: {e.code()} - {e.details()}")
    except Exception as e:
        print(f"[客戶端] 處理講者分辨時發生錯誤: {e}")


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
            print(f"[客戶端] 收到 Wav2Lip 影片，已保存至 {output_filename}")
        else:
            print("[客戶端] 伺服器回傳了空的影片資料。")
            
    except FileNotFoundError as e:
        print(f"[客戶端] 找不到檔案: {e}")
    except grpc.RpcError as e:
        print(f"[客戶端] Wav2Lip 請求失敗: {e.code()} - {e.details()}")
    except Exception as e:
        print(f"[客戶端] 處理 Wav2Lip 時發生錯誤: {e}")

def run_llm_text_generation_test(stub, prompt, max_tokens=100, temperature=0.7):
    """測試 LLM 文本生成功能"""
    print(f"\n[客戶端] 發送 LLM 文本生成請求: '{prompt}'")
    
    try:
        # 準備請求物件
        request = model_service_pb2.TextGenerationRequest(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=0.9
        )
        
        # 呼叫遠端的 GenerateText 服務
        response = stub.GenerateText(request)
        
        # 檢查回應
        if response.success:
            print(f"[客戶端] LLM 文本生成成功:")
            print(f"結果: {response.generated_text}")
        else:
            print("[客戶端] LLM 文本生成失敗")
            
    except grpc.RpcError as e:
        print(f"[客戶端] LLM 文本生成請求失敗: {e.code()} - {e.details()}")
    except Exception as e:
        print(f"[客戶端] 處理 LLM 文本生成時發生錯誤: {e}")

def run_llm_chat_test(stub, messages, max_tokens=120, temperature=0.7):
    """測試 LLM 對話功能"""
    print(f"\n[客戶端] 發送 LLM 對話請求:")
    
    # 顯示對話內容
    for msg in messages:
        role_icon = "👤" if msg["role"] == "user" else "🤖" if msg["role"] == "assistant" else "⚙️"
        print(f"  {role_icon} {msg['role']}: {msg['content']}")
    
    try:
        # 構建 gRPC 消息
        grpc_messages = []
        for msg in messages:
            grpc_messages.append(
                model_service_pb2.ChatMessage(
                    role=msg["role"],
                    content=msg["content"]
                )
            )
        
        # 準備請求物件
        request = model_service_pb2.ChatCompletionRequest(
            messages=grpc_messages,
            max_tokens=max_tokens,
            temperature=temperature
        )
        
        # 呼叫遠端的 ChatCompletion 服務
        response = stub.ChatCompletion(request)
        
        # 檢查回應
        if response.success:
            print(f"[客戶端] LLM 對話成功:")
            print(f"助手回應: {response.response}")
        else:
            print("[客戶端] LLM 對話失敗")
            
    except grpc.RpcError as e:
        print(f"[客戶端] LLM 對話請求失敗: {e.code()} - {e.details()}")
    except Exception as e:
        print(f"[客戶端] 處理 LLM 對話時發生錯誤: {e}")

def run_llm_comprehensive_test(stub):
    """執行 LLM 的完整測試套件"""
    print("\n測試 LLM 服務:")
    print("-" * 30)
    
    # 測試 1: 基本文本生成
    print("\n測試 1: 基本文本生成")
    text_prompts = [
        "The future of artificial intelligence is",
        "人工智慧的應用包括",
        "Technology has changed our lives by",
        "在未來十年，科技發展將會"
    ]
    
    for prompt in text_prompts:
        run_llm_text_generation_test(stub, prompt, max_tokens=256, temperature=0.7)
    
    # 測試 2: 不同溫度參數
    print("\n測試 2: 不同溫度參數對比")
    base_prompt = "The benefits of machine learning include"
    temperatures = [0.3, 0.7, 1.0]
    
    for temp in temperatures:
        print(f"\n溫度 {temp}:")
        run_llm_text_generation_test(stub, base_prompt, max_tokens=256, temperature=temp)
    
    # 測試 3: 基本對話
    print("\n測試 3: 基本對話")
    
    basic_conversations = [
        [{"role": "user", "content": "Hello! How are you?"}],
        [{"role": "user", "content": "你好！請介紹一下你自己。"}],
        [{"role": "user", "content": "What can you help me with?"}],
        [{"role": "user", "content": "Tell me about artificial intelligence."}]
    ]
    
    for i, conversation in enumerate(basic_conversations, 1):
        print(f"\n對話 {i}:")
        run_llm_chat_test(stub, conversation, max_tokens=100, temperature=0.7)
    
    # 測試 4: 系統提示對話
    print("\n測試 4: 角色扮演對話（系統提示）")
    
    role_conversations = [
        [
            {"role": "system", "content": "You are a helpful programming assistant."}, 
            {"role": "user", "content": "Explain what is Python programming language."}
        ],
        [
            {"role": "system", "content": "你是一個友善的中文助手。"},
            {"role": "user", "content": "請解釋什麼是機器學習。"},
        ],
        [
            {"role": "system", "content": "You are a creative writer who loves storytelling."}, 
            {"role": "user", "content": "Write the beginning of a short story about robots."}
        ]
    ]
    
    for i, conversation in enumerate(role_conversations, 1):
        print(f"\n角色對話 {i}:")
        run_llm_chat_test(stub, conversation, max_tokens=120, temperature=0.8)
    
    # 測試 5: 多輪對話
    print("\n測試 5: 多輪對話")
    
    # 模擬一個連續的對話
    conversation_history = []
    user_inputs = [
        "Hi, I want to learn about machine learning.",
        "What are the main types of machine learning?",
        "Can you give me an example of supervised learning?",
        "Thank you for the explanation!"
    ]
    
    for turn, user_input in enumerate(user_inputs, 1):
        print(f"\n對話回合 {turn}:")
        
        # 添加用戶輸入到歷史
        conversation_history.append({"role": "user", "content": user_input})
        
        # 執行對話
        run_llm_chat_test(stub, conversation_history, max_tokens=100, temperature=0.6)
        
        # 注意：這裡我們沒有真的把助手回應加到歷史中
        # 因為我們無法從 run_llm_chat_test 取得回應
        # 在實際應用中，您會想要保存回應並加到歷史中

def run_virtual_avatar_test(stub, image_path, audio_path):
    """測試虛擬頭像功能"""
    print(f"\n測試虛擬頭像服務:")
    print("-" * 30)
    
    try:
        # 1. 測試初始化頭像
        print(f"\n步驟 1: 初始化虛擬頭像")
        print(f"   圖片路徑: {image_path}")
        print(f"   音頻路徑: {audio_path}")
        
        # 讀取圖片和音頻數據
        try:
            with open(image_path, "rb") as f:
                image_data = f.read()
            print(f"圖片讀取成功: {len(image_data)} bytes")
        except FileNotFoundError:
            print(f"找不到圖片檔案: {image_path}")
            return
        
        try:
            with open(audio_path, "rb") as f:
                audio_data = f.read()
            print(f"音頻讀取成功: {len(audio_data)} bytes")
        except FileNotFoundError:
            print(f"找不到音頻檔案: {audio_path}")
            return
        
        # 發送初始化請求
        init_request = model_service_pb2.InitAvatarRequest(
            image_data=image_data,
            sample_audio_data=audio_data
        )
        
        print("正在初始化頭像...")
        init_response = stub.InitAvatar(init_request)
        
        if init_response.success:
            print(f"頭像初始化成功: {init_response.message}")
        else:
            print(f"頭像初始化失敗: {init_response.message}")
            return
        
        # 2. 測試頭像說話功能
        print(f"\n步驟 2: 測試頭像說話功能")
        
        test_sentences = [
            ("Hello, I am your virtual avatar!", "en"),
            ("This is a test of the avatar speech system.", "en"),
            ("你好，我是你的虛擬頭像！", "zh-cn"),
            ("歡迎使用虛擬頭像系統。", "zh-cn"),
            ("Nice to meet you! How can I help you today?", "en")
        ]
        
        for i, (text, language) in enumerate(test_sentences, 1):
            print(f"\n測試 {i}/{len(test_sentences)}: '{text}' ({language})")
            
            speak_request = model_service_pb2.AvatarSpeakRequest(
                text=text,
                language=language
            )
            
            print("正在生成語音和視頻...")
            import time
            start_time = time.time()
            
            try:
                speak_response = stub.AvatarSpeak(speak_request)
                
                end_time = time.time()
                duration = end_time - start_time
                
                if speak_response.success:
                    print(f"頭像說話成功: {speak_response.message}")
                    print(f"⏱處理時間: {duration:.2f} 秒")
                    print("請檢查虛擬攝像頭輸出（如 OBS 或視頻會議軟體）")
                    print("請檢查虛擬麥克風輸出（如音頻錄製軟體）")
                else:
                    print(f"頭像說話失敗: {speak_response.message}")
                
                # 等待一下再進行下一個測試
                if i < len(test_sentences):
                    print("等待 3 秒再進行下一個測試...")
                    time.sleep(3)
                    
            except grpc.RpcError as e:
                print(f"gRPC 錯誤: {e.code()} - {e.details()}")
        
        # 3. 測試不同參數的說話功能
        print(f"\n步驟 3: 測試不同語言的頭像說話")
        
        multilingual_tests = [
            ("Bonjour, je suis votre avatar virtuel!", "fr"),
            ("¡Hola! Soy tu avatar virtual.", "es"),
            ("こんにちは、私はあなたのバーチャルアバターです。", "ja"),
            ("Guten Tag! Ich bin Ihr virtueller Avatar.", "de"),
            ("Ciao! Sono il tuo avatar virtuale.", "it")
        ]
        
        for i, (text, language) in enumerate(multilingual_tests, 1):
            print(f"\n多語言測試 {i}: '{text}' ({language})")
            
            speak_request = model_service_pb2.AvatarSpeakRequest(
                text=text,
                language=language
            )
            
            try:
                start_time = time.time()
                speak_response = stub.AvatarSpeak(speak_request)
                end_time = time.time()
                
                if speak_response.success:
                    print(f"成功: {speak_response.message} ({end_time - start_time:.2f}s)")
                else:
                    print(f"失敗: {speak_response.message}")
                    
                # 短暫等待
                time.sleep(2)
                    
            except grpc.RpcError as e:
                print(f"gRPC 錯誤: {e.code()} - {e.details()}")
        
        print(f"\n虛擬頭像測試完成！")
        print("\n測試結果總結:")
        print("- 如果看到 '頭像說話成功' 訊息，表示服務正常運行")
        print("- 請檢查虛擬攝像頭是否有視頻輸出")
        print("- 請檢查虛擬麥克風是否有音頻輸出")
        print("- 如果沒有輸出，請檢查虛擬音視頻設備是否正確安裝")
        
    except grpc.RpcError as e:
        print(f"虛擬頭像測試失敗: {e.code()} - {e.details()}")
    except Exception as e:
        print(f"測試過程中發生錯誤: {e}")
        import traceback
        print(f"詳細錯誤: {traceback.format_exc()}")

def run_virtual_avatar_interactive_test(stub):
    """執行互動式虛擬頭像測試"""
    print(f"\n🎮 互動式虛擬頭像測試")
    print("-" * 30)
    
    # 預設檔案路徑
    default_image = "wav2lip_sample/tom.jpg"
    default_audio = "identify_sample/ta.wav"
    
    print(f"預設圖片: {default_image}")
    print(f"預設音頻: {default_audio}")
    
    # 檢查檔案是否存在
    import os
    if not os.path.exists(default_image):
        print(f"預設圖片不存在: {default_image}")
        return
    
    if not os.path.exists(default_audio):
        print(f"預設音頻不存在: {default_audio}")
        return
    
    try:
        # 初始化頭像
        print("\n正在初始化虛擬頭像...")
        
        with open(default_image, "rb") as f:
            image_data = f.read()
        with open(default_audio, "rb") as f:
            audio_data = f.read()
        
        init_request = model_service_pb2.InitAvatarRequest(
            image_data=image_data,
            sample_audio_data=audio_data
        )
        
        init_response = stub.InitAvatar(init_request)
        
        if not init_response.success:
            print(f"頭像初始化失敗: {init_response.message}")
            return
        
        print(f"頭像初始化成功: {init_response.message}")
        
        # 互動式對話
        print(f"\n開始互動式對話（輸入 'quit' 退出）:")
        
        while True:
            user_input = input("\n請輸入要讓頭像說的話: ").strip()
            
            if user_input.lower() in ['quit', 'exit', '退出', 'q']:
                print("結束互動式測試")
                break
            
            if not user_input:
                print("請輸入有效文字")
                continue
            
            # 簡單語言檢測（基於字符）
            language = "zh-cn" if any('\u4e00' <= char <= '\u9fff' for char in user_input) else "en"
            
            print(f"檢測到語言: {language}")
            print(f"正在讓頭像說話...")
            
            speak_request = model_service_pb2.AvatarSpeakRequest(
                text=user_input,
                language=language
            )
            
            try:
                import time
                start_time = time.time()
                speak_response = stub.AvatarSpeak(speak_request)
                end_time = time.time()
                
                if speak_response.success:
                    print(f"頭像說話成功 ({end_time - start_time:.2f}s)")
                    print("請檢查虛擬攝像頭和麥克風輸出")
                else:
                    print(f"頭像說話失敗: {speak_response.message}")
                    
            except grpc.RpcError as e:
                print(f"gRPC 錯誤: {e.code()} - {e.details()}")
            except KeyboardInterrupt:
                print("\n用戶中斷，結束測試")
                break
    
    except Exception as e:
        print(f"互動式測試錯誤: {e}")

def check_virtual_avatar_prerequisites():
    """檢查虛擬頭像的先決條件"""
    print(f"\n檢查虛擬頭像先決條件:")
    print("-" * 30)
    
    import os
    
    # 檢查測試檔案
    test_files = {
        "測試圖片": "wav2lip_sample/tom.jpg",
        "測試音頻": "identify_sample/ta.wav"
    }
    
    files_ok = True
    for file_type, file_path in test_files.items():
        if os.path.exists(file_path):
            file_size = os.path.getsize(file_path)
            print(f"{file_type}: {file_path} ({file_size} bytes)")
        else:
            print(f"{file_type}: {file_path} (不存在)")
            files_ok = False
    
    # 檢查虛擬設備
    print(f"\n🎵 檢查虛擬音頻設備:")
    try:
        import sounddevice as sd
        devices = sd.query_devices()
        virtual_audio_found = False
        
        for i, device in enumerate(devices):
            if ('cable' in device['name'].lower() or 
                'virtual' in device['name'].lower() or
                'vb-audio' in device['name'].lower()):
                print(f"虛擬音頻設備: {device['name']}")
                virtual_audio_found = True
        
        if not virtual_audio_found:
            print("未找到虛擬音頻設備")
            print("請安裝 VB-Cable 或類似軟體")
    except Exception as e:
        print(f"檢查音頻設備時出錯: {e}")
    
    print(f"\n📹 檢查虛擬攝像頭支援:")
    try:
        import pyvirtualcam
        print("PyVirtualCam 可用")
        print("請確保已安裝 OBS Virtual Camera 或類似軟體")
    except ImportError:
        print("PyVirtualCam 不可用")
        print("請安裝: pip install pyvirtualcam")
    
    # 檢查其他依賴
    print(f"\n檢查必要套件:")
    required_packages = [
        ("OpenCV", "cv2"),
        ("NumPy", "numpy"),
        ("SoundFile", "soundfile"),
        ("LibROSA", "librosa")
    ]
    
    packages_ok = True
    for name, module in required_packages:
        try:
            __import__(module)
            print(f"{name}")
        except ImportError:
            print(f"{name}")
            packages_ok = False
    
    # 總結
    print(f"\n📋 先決條件檢查結果:")
    if files_ok and packages_ok:
        print("所有先決條件都已滿足，可以進行虛擬頭像測試")
        return True
    else:
        print("存在缺失的先決條件，請先解決上述問題")
        return False

def main():
    # 連接到 gRPC 伺服器
    print("正在連接到 gRPC 伺服器...")
    
    # --- 修改處：在這裡加入 options ---
    channel_options = [
        ('grpc.max_send_message_length', MAX_MESSAGE_LENGTH),
        ('grpc.max_receive_message_length', MAX_MESSAGE_LENGTH),
        ('grpc.max_receive_metadata_size', MAX_METADATA_SIZE),
        ('grpc.max_send_metadata_size', MAX_METADATA_SIZE),
    ]
    
    with grpc.insecure_channel('localhost:50051', options=channel_options) as channel:
        # 建立客戶端 Stub
        translator_stub = model_service_pb2_grpc.TranslatorServiceStub(channel)
        media_stub = model_service_pb2_grpc.MediaServiceStub(channel)

        print("\n" + "="*60)
        print("開始測試所有服務功能")
        print("="*60)

        # --- 檢查虛擬頭像先決條件 ---
        print("\ 先決條件檢查:")
        print("-" * 30)
        avatar_ready = check_virtual_avatar_prerequisites()

        # --- 執行虛擬頭像測試 ---
        if avatar_ready:
            print("\n測試虛擬頭像服務:")
            print("-" * 30)
            
            # 基本虛擬頭像測試
            run_virtual_avatar_test(media_stub, "wav2lip_sample/tom.jpg", "identify_sample/ta.wav")
            
            # 詢問是否進行互動式測試
            try:
                user_choice = input("\n是否進行互動式頭像測試？(y/n): ").strip().lower()
                if user_choice in ['y', 'yes', '是', 'Y']:
                    run_virtual_avatar_interactive_test(media_stub)
                else:
                    print("跳過互動式測試")
            except KeyboardInterrupt:
                print("\n跳過互動式測試")
        else:
            print("跳過虛擬頭像測試（先決條件不滿足）")

        return
    
        # --- 執行 RAG 問答測試 ---
        print("\n測試 RAG 問答服務:")
        print("-" * 30)
        run_rag_qa_test(media_stub, "預算超支多少？")
        run_rag_qa_test(media_stub, "What is the core function of the immersive assistant?")

        # --- 執行 LLM 測試 ---
        run_llm_comprehensive_test(media_stub)

        # --- 執行翻譯測試 ---
        print("\n測試翻譯服務:")
        print("-" * 30)
        run_translation_test(translator_stub, "Hello world", "英文", "中文")
        run_translation_test(translator_stub, "這是個很棒的系統", "中文", "日文")
        run_translation_test(translator_stub, "Wie geht es Ihnen?", "德文", "英文")
        run_translation_test(translator_stub, "Ceci est un test.", "法文", "西班牙文")
        # 測試一個不支援的語言
        run_translation_test(translator_stub, "Test", "英文", "火星文")

        # --- 執行 TTS 測試 ---
        print("\n測試 TTS 服務:")
        print("-" * 30)
        run_tts_test(media_stub, "This is a test of the text to speech API.", "en", "output_en.wav")
        run_tts_test(media_stub, "你好，這是一個語音合成的測試。", "zh-cn", "output_zh-cn.wav")

        # --- 執行講者分辨測試 ---
        print("\n測試講者分辨服務:")
        print("-" * 30)
        
        # 方法 1: 使用現有的音訊檔案（如果存在）
        existing_audio_files = [
            "./identify_sample/ta.wav",
        ]
        
        test_file_found = False
        for audio_file in existing_audio_files:
            try:
                with open(audio_file, 'rb'):
                    print(f"使用現有音訊檔案進行測試: {audio_file}")
                    run_speaker_identification_test(media_stub, audio_file)
                    test_file_found = True
                    break
            except FileNotFoundError:
                continue

        # --- 執行 Wav2Lip 測試 ---
        print("\n測試 Wav2Lip 服務:")
        print("-" * 30)
        
        audio_file_path = "wav2lip_sample/chinese_news.wav"
        image_file_path = "wav2lip_sample/tom.jpg"

        try:
            # 檢查檔案是否存在
            with open(audio_file_path, 'rb'):
                pass
            with open(image_file_path, 'rb'):
                pass
            
            print(f"使用音訊檔案: {audio_file_path}, 圖片檔案: {image_file_path}")
            run_wav2lip_test(media_stub, audio_file_path, image_file_path)

        except FileNotFoundError:
            print(f"找不到測試檔案，跳過 Wav2Lip 測試。")
            print(f"   請確認 '{audio_file_path}' 和 '{image_file_path}' 是否存在。")

        print("\n" + "="*60)
        print("所有測試完成！")
        print("="*60)

if __name__ == '__main__':
    main()
