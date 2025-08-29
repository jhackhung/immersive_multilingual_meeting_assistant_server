import grpc
import os
import time
import argparse
import logging

# 導入從 .proto 產生的模組
from proto import model_service_pb2
from proto import model_service_pb2_grpc

# --- 組態設定 ---

# 設定日誌
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 定義多語言測試用例
# 根據您 TtsServicer 中的 supported_languages 列表
TEST_CASES = {
    "en": "Hello, this is a test of the text-to-speech API in English. I hope you have a wonderful day!",
    "zh-cn": "你好，这是一个测试文本转语音功能的中文API。希望你今天过得愉快!",
    # "ja": "こんにちは、これは日本語のテキスト読み上げAPIのテストです。素晴らしい一日をお過ごしください。",
    # "ko": "안녕하세요, 이것은 한국어 텍스트 음성 변환 API 테스트입니다. 좋은 하루 보내세요!",
    # "fr": "Bonjour, ceci est un test de l'API de synthèse vocale en français. J'espère que vous passez une excellente journée!",
    "de": "Hallo, dies ist ein Test der Text-to-Speech-API auf Deutsch. Ich hoffe, Sie haben einen wunderschönen Tag!",
    # "es": "Hola, esta es una prueba de la API de texto a voz en español. ¡Espero que tengas un día maravilloso!",
    # "it": "Ciao, questo è un test dell'API di sintesi vocale in italiano. Spero che tu abbia una splendida giornata!",
    # "pt": "Olá, este é um teste da API de conversão de texto em fala em português. Espero que você tenha um dia maravilhoso!",
    # "ru": "Привет, это тест API для преобразования текста в речь на русском языке. Надеюсь, у вас будет замечательный день!"
}

def run_tts_request(stub, test_name: str, language: str, text: str, output_path: str, reference_audio_bytes: bytes = None):
    """
    執行單次 TTS 請求並處理結果的核心函式。

    Args:
        stub: gRPC stub 物件。
        test_name: 測試的描述性名稱 (用於日誌)。
        language: 語言代碼。
        text: 要轉換的文字。
        output_path: 儲存 .wav 檔案的路徑。
        reference_audio_bytes: 參考音訊的 bytes，若為 None 則使用伺服器預設聲音。
    """
    logging.info(f"\n--- 開始測試: {test_name} (語言: {language}) ---")
    logging.info(f"    文字: '{text[:40]}...'")
    
    try:
        # 建立請求物件
        request = model_service_pb2.TtsRequest(
            text_to_speak=text,
            language=language
        )
        if reference_audio_bytes:
            request.reference_audio = reference_audio_bytes

        # 發送 gRPC 請求
        start_time = time.time()
        response = stub.Tts(request, timeout=120)  # 設定 120 秒超時
        duration = time.time() - start_time

        if response and response.generated_audio:
            with open(output_path, "wb") as f:
                f.write(response.generated_audio)
            logging.info(f"✅ 測試成功: 音訊已儲存至 {output_path} (耗時: {duration:.2f} 秒)")
        else:
            logging.warning(f"⚠️ 請求成功但未收到音訊資料。")

    except grpc.RpcError as e:
        # 處理 gRPC 錯誤，這對於錯誤案例測試是預期行為
        logging.error(f"❌ 測試期間發生 gRPC 錯誤 (這是預期的錯誤案例嗎？)")
        logging.error(f"    狀態碼: {e.code()}")
        logging.error(f"    詳細訊息: {e.details()}")
    except Exception as e:
        logging.error(f"❌ 發生未知的例外狀況: {e}")

def main():
    """主執行函式"""
    parser = argparse.ArgumentParser(description="TTS gRPC 服務綜合測試客戶端")
    parser.add_argument(
        "--server",
        type=str,
        default="localhost:50051",
        help="gRPC 伺服器地址與端口"
    )
    parser.add_argument(
        "--speaker_wav",
        type=str,
        default=None,
        help="用於聲音克隆的參考 .wav 檔案路徑 (可選)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="tts_test_output",
        help="儲存生成音訊的檔案夾"
    )
    args = parser.parse_args()

    # 建立輸出檔案夾
    os.makedirs(args.output_dir, exist_ok=True)

    # 讀取參考音訊檔案
    reference_audio_bytes = None
    if args.speaker_wav:
        try:
            with open(args.speaker_wav, "rb") as f:
                reference_audio_bytes = f.read()
            logging.info(f"🔊 已載入參考音訊: {args.speaker_wav}")
        except FileNotFoundError:
            logging.error(f"❌ 找不到參考音訊檔案: {args.speaker_wav}，將僅測試預設聲音。")
            
    # 建立 gRPC 連線
    try:
        channel = grpc.insecure_channel(args.server)
        grpc.channel_ready_future(channel).result(timeout=10)
        stub = model_service_pb2_grpc.MediaServiceStub(channel)
        logging.info(f"✅ 成功連接到 gRPC 伺服器: {args.server}")
    except grpc.FutureTimeoutError:
        logging.error(f"❌ 無法連接到 gRPC 伺服器: {args.server}。請檢查伺服器是否已啟動。")
        return

    # --- 場景 1: 使用指定的參考音訊進行多語言測試 ---
    if reference_audio_bytes:
        logging.info("\n" + "="*60)
        logging.info("🚀 開始場景 1: 使用指定的參考音訊進行多語言合成")
        logging.info("="*60)
        for lang, text in TEST_CASES.items():
            output_filename = f"custom_speaker_{lang}.wav"
            output_path = os.path.join(args.output_dir, output_filename)
            run_tts_request(stub, "自定義聲音合成", lang, text, output_path, reference_audio_bytes)
            time.sleep(1)

    # --- 場景 2: 使用伺服器預設聲音進行多語言測試 ---
    # logging.info("\n" + "="*60)
    # logging.info("🚀 開始場景 2: 使用伺服器預設聲音進行多語言合成")
    # logging.info("="*60)
    # for lang, text in TEST_CASES.items():
    #     output_filename = f"default_speaker_{lang}.wav"
    #     output_path = os.path.join(args.output_dir, output_filename)
    #     run_tts_request(stub, "預設聲音合成", lang, text, output_path, None)
    #     time.sleep(1)

    # --- 場景 3: 邊界與錯誤條件測試 ---
    logging.info("\n" + "="*60)
    logging.info("🚀 開始場景 3: 邊界與錯誤條件測試")
    logging.info("="*60)

    # 3.1 錯誤請求: 空文字
    run_tts_request(stub, "錯誤請求 (空文字)", "en", "", os.path.join(args.output_dir, "error_empty_text.wav"))

    # 3.2 錯誤請求: 過長文字
    long_text = "a" * 1001
    run_tts_request(stub, "錯誤請求 (過長文字)", "en", long_text, os.path.join(args.output_dir, "error_long_text.wav"))

    # 3.3 錯誤請求: 不支援的語言
    run_tts_request(stub, "錯誤請求 (不支援的語言)", "xyz", "This language is not supported.", os.path.join(args.output_dir, "error_unsupported_lang.wav"))

    # 3.4 錯誤請求: 過大參考音訊 (11MB)
    logging.info("\n--- 開始測試: 錯誤請求 (過大參考音訊) ---")
    try:
        large_audio_bytes = os.urandom(11 * 1024 * 1024)
        request = model_service_pb2.TtsRequest(
            text_to_speak="Test large audio file.",
            language="en",
            reference_audio=large_audio_bytes
        )
        stub.Tts(request, timeout=10)
        logging.error("❌ 測試失敗: 伺服器未對過大的檔案返回錯誤。")
    except grpc.RpcError as e:
        logging.info(f"✅ 測試成功: 伺服器按預期返回了 gRPC 錯誤。")
        logging.info(f"    狀態碼: {e.code()}")
        logging.info(f"    詳細訊息: {e.details()}")
    except Exception as e:
        logging.error(f"❌ 測試期間發生未知的例外狀況: {e}")

    channel.close()
    logging.info("\n" + "="*60)
    logging.info("✅ 所有測試執行完畢！")
    
if __name__ == '__main__':
    main()
