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
            print(f"✅ [客戶端] RAG 問答成功:")
            print(f"🤖 模型回答: {response.answer}")
            if response.sources:
                print(f"📚 參考來源:")
                for source in response.sources:
                    print(f"  - {source}")
        else:
            print("❌ [客戶端] RAG 問答失敗")
            
    except grpc.RpcError as e:
        print(f"❌ [客戶端] RAG 問答請求失敗: {e.code()} - {e.details()}")
    except Exception as e:
        print(f"❌ [客戶端] 處理 RAG 問答時發生錯誤: {e}")

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
            print(f"✅ [客戶端] LLM 文本生成成功:")
            print(f"📄 結果: {response.generated_text}")
        else:
            print("❌ [客戶端] LLM 文本生成失敗")
            
    except grpc.RpcError as e:
        print(f"❌ [客戶端] LLM 文本生成請求失敗: {e.code()} - {e.details()}")
    except Exception as e:
        print(f"❌ [客戶端] 處理 LLM 文本生成時發生錯誤: {e}")

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
            print(f"✅ [客戶端] LLM 對話成功:")
            print(f"🤖 助手回應: {response.response}")
        else:
            print("❌ [客戶端] LLM 對話失敗")
            
    except grpc.RpcError as e:
        print(f"❌ [客戶端] LLM 對話請求失敗: {e.code()} - {e.details()}")
    except Exception as e:
        print(f"❌ [客戶端] 處理 LLM 對話時發生錯誤: {e}")

def run_llm_comprehensive_test(stub):
    """執行 LLM 的完整測試套件"""
    print("\n🤖 測試 LLM 服務:")
    print("-" * 30)
    
    # 測試 1: 基本文本生成
    print("\n📝 測試 1: 基本文本生成")
    text_prompts = [
        "The future of artificial intelligence is",
        "人工智慧的應用包括",
        "Technology has changed our lives by",
        "在未來十年，科技發展將會"
    ]
    
    for prompt in text_prompts:
        run_llm_text_generation_test(stub, prompt, max_tokens=80, temperature=0.7)
    
    # 測試 2: 不同溫度參數
    print("\n🌡️ 測試 2: 不同溫度參數對比")
    base_prompt = "The benefits of machine learning include"
    temperatures = [0.3, 0.7, 1.0]
    
    for temp in temperatures:
        print(f"\n🔥 溫度 {temp}:")
        run_llm_text_generation_test(stub, base_prompt, max_tokens=60, temperature=temp)
    
    # 測試 3: 基本對話
    print("\n💬 測試 3: 基本對話")
    
    basic_conversations = [
        [{"role": "user", "content": "Hello! How are you?"}],
        [{"role": "user", "content": "你好！請介紹一下你自己。"}],
        [{"role": "user", "content": "What can you help me with?"}],
        [{"role": "user", "content": "Tell me about artificial intelligence."}]
    ]
    
    for i, conversation in enumerate(basic_conversations, 1):
        print(f"\n💭 對話 {i}:")
        run_llm_chat_test(stub, conversation, max_tokens=100, temperature=0.7)
    
    # 測試 4: 系統提示對話
    print("\n🎭 測試 4: 角色扮演對話（系統提示）")
    
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
        print(f"\n🎪 角色對話 {i}:")
        run_llm_chat_test(stub, conversation, max_tokens=120, temperature=0.8)
    
    # 測試 5: 多輪對話
    print("\n🔄 測試 5: 多輪對話")
    
    # 模擬一個連續的對話
    conversation_history = []
    user_inputs = [
        "Hi, I want to learn about machine learning.",
        "What are the main types of machine learning?",
        "Can you give me an example of supervised learning?",
        "Thank you for the explanation!"
    ]
    
    for turn, user_input in enumerate(user_inputs, 1):
        print(f"\n🔄 對話回合 {turn}:")
        
        # 添加用戶輸入到歷史
        conversation_history.append({"role": "user", "content": user_input})
        
        # 執行對話
        run_llm_chat_test(stub, conversation_history, max_tokens=100, temperature=0.6)
        
        # 注意：這裡我們沒有真的把助手回應加到歷史中
        # 因為我們無法從 run_llm_chat_test 取得回應
        # 在實際應用中，您會想要保存回應並加到歷史中

def main():
    # 連接到 gRPC 伺服器
    print("🔗 正在連接到 gRPC 伺服器...")
    
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
        print("🚀 開始測試所有服務功能")
        print("="*60)

        # --- 執行 RAG 問答測試 ---
        print("\n📚 測試 RAG 問答服務:")
        print("-" * 30)
        run_rag_qa_test(media_stub, "預算超支多少？")
        run_rag_qa_test(media_stub, "What is the core function of the immersive assistant?")

        # --- 執行 LLM 測試 ---
        run_llm_comprehensive_test(media_stub)

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

        # --- 執行 Wav2Lip 測試 ---
        print("\n🎬 測試 Wav2Lip 服務:")
        print("-" * 30)
        
        audio_file_path = "wav2lip_sample/chinese_news.wav"
        image_file_path = "wav2lip_sample/tom.jpg"

        try:
            # 檢查檔案是否存在
            with open(audio_file_path, 'rb'):
                pass
            with open(image_file_path, 'rb'):
                pass
            
            print(f"📁 使用音訊檔案: {audio_file_path}, 圖片檔案: {image_file_path}")
            run_wav2lip_test(media_stub, audio_file_path, image_file_path)

        except FileNotFoundError:
            print(f"⚠️ 找不到測試檔案，跳過 Wav2Lip 測試。")
            print(f"   請確認 '{audio_file_path}' 和 '{image_file_path}' 是否存在。")

        print("\n" + "="*60)
        print("✅ 所有測試完成！")
        print("="*60)

if __name__ == '__main__':
    main()
