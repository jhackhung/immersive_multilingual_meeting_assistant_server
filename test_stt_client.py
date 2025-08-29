import grpc
import asyncio
import time
import numpy as np
import soundfile as sf
import logging
from typing import AsyncIterable

# 添加音頻處理庫
try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False
    logging.warning("librosa 未安裝，將使用簡單的重採樣方法")

from proto import model_service_pb2
from proto import model_service_pb2_grpc

# 設定日誌
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# gRPC 服務器地址
SERVER_ADDRESS = 'localhost:50051'

# 音訊參數 (必須與伺服器預期的一致: 16kHz, 單聲道, 16位元 PCM)
SAMPLE_RATE_HZ = 16000
CHUNK_DURATION_MS = 100 # 每 100 毫秒發送一個音訊塊
CHUNK_SIZE_SAMPLES = int(SAMPLE_RATE_HZ * (CHUNK_DURATION_MS / 1000))

# 語音檔案路徑 (請替換為您自己的 .wav 檔案路徑)
AUDIO_FILE_PATH = "identify_sample/ta.wav"

def simple_resample(audio_data: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """
    簡單的重採樣方法（如果沒有 librosa）
    """
    if orig_sr == target_sr:
        return audio_data
    
    # 計算重採樣比例
    ratio = target_sr / orig_sr
    
    # 使用 numpy 的線性插值進行重採樣
    original_length = len(audio_data)
    new_length = int(original_length * ratio)
    
    # 創建新的索引
    old_indices = np.linspace(0, original_length - 1, original_length)
    new_indices = np.linspace(0, original_length - 1, new_length)
    
    # 線性插值
    resampled = np.interp(new_indices, old_indices, audio_data)
    
    return resampled.astype(audio_data.dtype)

def load_and_process_audio(audio_path: str) -> np.ndarray:
    """
    載入並處理音頻文件，確保符合服務器要求
    """
    try:
        if LIBROSA_AVAILABLE:
            # 使用 librosa 載入並重採樣到 16kHz
            logger.info(f"使用 librosa 載入音頻文件: {audio_path}")
            audio_data, _ = librosa.load(audio_path, sr=SAMPLE_RATE_HZ, mono=True)
            # 轉換為 int16 格式
            audio_data = (audio_data * 32767).astype(np.int16)
        else:
            # 使用 soundfile 載入，然後手動重採樣
            logger.info(f"使用 soundfile 載入音頻文件: {audio_path}")
            audio_data, current_sample_rate = sf.read(audio_path, dtype='float32')
            
            # 轉換為單聲道
            if audio_data.ndim > 1:
                logger.info("轉換多聲道音頻為單聲道")
                audio_data = audio_data.mean(axis=1)
            
            # 重採樣
            if current_sample_rate != SAMPLE_RATE_HZ:
                logger.info(f"重採樣音頻: {current_sample_rate} Hz -> {SAMPLE_RATE_HZ} Hz")
                audio_data = simple_resample(audio_data, current_sample_rate, SAMPLE_RATE_HZ)
            
            # 轉換為 int16 格式
            audio_data = (audio_data * 32767).astype(np.int16)
        
        logger.info(f"音頻處理完成: {len(audio_data)} 樣本, 時長: {len(audio_data) / SAMPLE_RATE_HZ:.2f} 秒")
        return audio_data
        
    except Exception as e:
        logger.error(f"載入音頻文件失敗: {e}")
        raise

async def generate_audio_requests(audio_path: str, language: str = "zh") -> AsyncIterable[model_service_pb2.StreamingRecognizeRequest]:
    """
    從音訊檔案生成一系列 StreamingRecognizeRequest 訊息。
    """
    try:
        # 載入並處理音頻
        audio_data = load_and_process_audio(audio_path)
        total_samples = len(audio_data)
        
        logger.info(f"開始串流音訊檔案: {audio_path}, 總時長: {total_samples / SAMPLE_RATE_HZ:.2f} 秒")

        # 發送第一個配置請求
        config_request = model_service_pb2.StreamingRecognizeRequest(
            language=language,
            # return_word_timestamps=return_word_timestamps,
            audio=model_service_pb2.AudioChunk(audio_bytes=b""),  # 空的音頻數據作為配置請求
            is_last=False
        )
        yield config_request
        logger.info(f"發送配置請求: language={language}")

        # 分塊發送音頻數據
        for i in range(0, total_samples, CHUNK_SIZE_SAMPLES):
            chunk_samples = audio_data[i : i + CHUNK_SIZE_SAMPLES]
            audio_chunk_bytes = chunk_samples.tobytes()

            # 建立 StreamingRecognizeRequest 訊息
            request = model_service_pb2.StreamingRecognizeRequest(
                audio=model_service_pb2.AudioChunk(audio_bytes=audio_chunk_bytes),
                language=language,
                is_last=False
            )
            yield request
            
            # 模擬實時發送，等待一段時間
            await asyncio.sleep(CHUNK_DURATION_MS / 1000.0)
            
            # 記錄進度
            progress = (i + len(chunk_samples)) / total_samples * 100
            if i % (CHUNK_SIZE_SAMPLES * 10) == 0:  # 每秒記錄一次
                logger.info(f"發送進度: {progress:.1f}%")

        # 在串流結束時發送最後一個請求，標記串流結束
        final_request = model_service_pb2.StreamingRecognizeRequest(
            audio=model_service_pb2.AudioChunk(audio_bytes=b""),
            language=language,
            is_last=True
        )
        yield final_request
        logger.info("音訊串流發送完成。")

    except FileNotFoundError:
        logger.error(f"錯誤: 找不到音訊檔案在 {audio_path}。請確認路徑正確。")
    except Exception as e:
        logger.error(f"生成音訊請求時發生錯誤: {e}")

async def run_client():
    """
    gRPC 客戶端主程式，用於連接服務器並執行語音識別串流。
    """
    logger.info(f"嘗試連接到 gRPC 服務器: {SERVER_ADDRESS}")
    
    # 設置 gRPC 選項以支持大型消息
    channel_options = [
        ('grpc.max_send_message_length', 100 * 1024 * 1024),
        ('grpc.max_receive_message_length', 100 * 1024 * 1024),
        ('grpc.keepalive_time_ms', 30000),
        ('grpc.keepalive_timeout_ms', 5000),
    ]
    
    async with grpc.aio.insecure_channel(SERVER_ADDRESS, options=channel_options) as channel:
        stub = model_service_pb2_grpc.MediaServiceStub(channel)

        try:
            # 檢查音頻文件是否存在
            import os
            if not os.path.exists(AUDIO_FILE_PATH):
                logger.error(f"音頻文件不存在: {AUDIO_FILE_PATH}")
                return

            # 調用雙向串流 RPC
            response_iterator: AsyncIterable[model_service_pb2.StreamingRecognizeResponse] = stub.StreamingRecognize(
                generate_audio_requests(AUDIO_FILE_PATH, language="en")  # 修改為英文，根據你的音頻文件
            )

            logger.info("開始接收轉錄響應...")
            response_count = 0
            
            async for response in response_iterator:
                response_count += 1
                logger.info(f"\n=== 響應 #{response_count} ===")
                
                # 處理來自服務器的每個響應
                if response.transcript_text:
                    logger.info(f"轉錄文本 ({'最終' if response.is_final else '部分'}): '{response.transcript_text}'")
                    
                    if response.segments:
                        for i, segment in enumerate(response.segments):
                            logger.info(f"  片段 {i+1} [{segment.start_time_sec:.2f}s - {segment.end_time_sec:.2f}s]: '{segment.text}'")
                            logger.info(f"    說話者: {segment.speaker_id}")
                            
                            # 詞級時間戳
                            for j, word_info in enumerate(segment.words):
                                logger.info(f"    詞彙 {j+1} [{word_info.start_time_sec:.2f}s - {word_info.end_time_sec:.2f}s]: '{word_info.word}' (信心: {word_info.confidence:.2f})")
                else:
                    logger.info("轉錄文本: (空)")
                
                # 顯示性能指標
                logger.info(f"會話 ID: {response.session_id}")
                logger.info(f"RTF (實時因子): {response.rtf:.4f}")
                logger.info(f"音頻塊時長: {response.chunk_sec:.2f} 秒")
                logger.info(f"服務器處理時間: {response.server_time_sec:.4f} 秒")
                
                # 檢查是否有錯誤或提示訊息
                if response.message:
                    if response.message == "串流結束。":
                        logger.info(f"伺服器訊息: {response.message}")
                    else:
                        logger.warning(f"伺服器訊息: {response.message}")

                # 如果這是最終響應，顯示詳細資訊並結束迴圈
                if response.is_final:
                    logger.info("\n=== 最終結果總結 ===")
                    logger.info(f"總共收到 {response_count} 個響應")
                    if response.transcript_text:
                        logger.info(f"最終轉錄結果: '{response.transcript_text}'")
                    else:
                        logger.info("最終轉錄結果: (無)")
                    break

        except grpc.RpcError as e:
            logger.error(f"gRPC 服務器錯誤: {e.code()} - {e.details()}")
        except Exception as e:
            logger.error(f"客戶端發生錯誤: {e}")

def generate_test_audio():
    """生成測試音頻文件（如果需要）"""
    try:
        if LIBROSA_AVAILABLE:
            import librosa
            # 生成 5 秒的正弦波測試音頻
            duration = 5.0
            sr = 16000
            t = np.linspace(0, duration, int(sr * duration), False)
            frequency = 440  # A4 音符
            audio = 0.3 * np.sin(2 * np.pi * frequency * t)
            
            # 保存為 WAV 文件
            test_file = "test_audio_16k.wav"
            sf.write(test_file, audio, sr)
            logger.info(f"生成測試音頻文件: {test_file}")
            return test_file
    except Exception as e:
        logger.error(f"生成測試音頻失敗: {e}")
    return None

if __name__ == '__main__':
    # 檢查音頻文件是否存在，如果不存在則生成測試音頻
    import os
    if not os.path.exists(AUDIO_FILE_PATH):
        logger.warning(f"音頻文件 {AUDIO_FILE_PATH} 不存在")
        test_file = generate_test_audio()
        if test_file:
            # 使用生成的測試文件
            AUDIO_FILE_PATH = test_file
        else:
            logger.error("無法生成測試音頻文件，請確保有有效的音頻文件")
            exit(1)
    
    # 執行客戶端程式
    asyncio.run(run_client())