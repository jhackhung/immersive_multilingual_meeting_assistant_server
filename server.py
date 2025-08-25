print("!!! SERVER.PY HAS BEEN MODIFIED SUCCESSFULLY !!!")

from concurrent import futures
import grpc
import time
import logging

# 匯入 gRPC 模組
from proto import model_service_pb2
from proto import model_service_pb2_grpc

# 匯入所有 API 服務層
from apis.wav2lip_service import Wav2LipServicer
from apis.translator_service import TranslatorService
from apis.tts_service import TtsServicer
from apis.llm_service import LLMServicer
from apis.speech_recognition_service import SpeechRecognitionServicer
from apis.rag_service import RAGService

# 設定日誌
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- 定義一個較大的訊息長度，例如 100MB ---
MAX_MESSAGE_LENGTH = 100 * 1024 * 1024
# --- 增加元數據大小限制 ---
MAX_METADATA_SIZE = 2 * 1024 * 1024  # 2MB

class TranslatorServicer(model_service_pb2_grpc.TranslatorServiceServicer):
    """gRPC 翻譯服務實現"""
    
    def __init__(self, translator_api: TranslatorService):
        self.translator_api = translator_api
        logger.info("TranslatorServicer 已初始化")

    def Translate(self, request, context):
        request_data = {
            "text": request.text_to_translate,
            "source_lang": request.source_language,
            "target_lang": request.target_language
        }
        logger.info(f"收到翻譯請求: {request_data}")
        result = self.translator_api.process_translation_request(request_data)
        if result["success"]:
            return model_service_pb2.TranslateResponse(
                translated_text=result["translated_text"]
            )
        else:
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(result["error"])
            return model_service_pb2.TranslateResponse()

class MediaServicer(model_service_pb2_grpc.MediaServiceServicer):
    """統一的媒體服務實現"""
    
    def __init__(self, tts_servicer, wav2lip_servicer, speaker_annote_servicer, llm_servicer, speech_recognition_servicer, rag_service):
        self.tts_servicer = tts_servicer
        self.wav2lip_servicer = wav2lip_servicer
        self.speaker_annote_servicer = speaker_annote_servicer
        self.llm_servicer = llm_servicer
        self.speech_recognition_servicer = speech_recognition_servicer
        self.rag_service = rag_service
        logger.info("MediaServicer 已初始化（包含 RAG, LLM 和語音識別服務）")
    
    def Tts(self, request, context):
        logger.info("收到 TTS 請求")
        return self.tts_servicer.Tts(request, context)
    
    def Wav2Lip(self, request, context):
        logger.info("收到 Wav2Lip 請求")
        return self.wav2lip_servicer.Wav2Lip(request, context)
    
    def SpeakerAnnote(self, request, context):
        logger.info("收到 SpeakerAnnote 請求")
        return self.speaker_annote_servicer.SpeakerAnnote(request, context)
    
    def SpeechRecognition(self, request, context):
        logger.info("收到 SpeechRecognition 請求")
        if self.speech_recognition_servicer:
            return self.speech_recognition_servicer.SpeechRecognition(request, context)
        else:
            context.set_code(grpc.StatusCode.UNIMPLEMENTED)
            context.set_details("語音識別服務未啟用")
            return model_service_pb2.SpeechRecognitionResponse(
                success=False
            )
    
    def GenerateText(self, request, context):
        logger.info("收到 GenerateText 請求")
        if self.llm_servicer:
            return self.llm_servicer.GenerateText(request, context)
        else:
            context.set_code(grpc.StatusCode.UNIMPLEMENTED)
            context.set_details("LLM 服務未啟用")
            return model_service_pb2.TextGenerationResponse(
                generated_text="",
                success=False
            )
    
    def ChatCompletion(self, request, context):
        logger.info("收到 ChatCompletion 請求")
        if self.llm_servicer:
            return self.llm_servicer.ChatCompletion(request, context)
        else:
            context.set_code(grpc.StatusCode.UNIMPLEMENTED)
            context.set_details("LLM 服務未啟用")
            return model_service_pb2.ChatCompletionResponse(
                response="",
                success=False
            )

    def AnswerQuestionFromDocuments(self, request, context):
        logger.info(f"收到 AnswerQuestionFromDocuments 請求: '{request.query}'")
        try:
            if not self.rag_service or not self.llm_servicer:
                context.set_code(grpc.StatusCode.UNIMPLEMENTED)
                context.set_details("RAG 或 LLM 服務未啟用")
                return model_service_pb2.AnswerQuestionResponse(success=False)

            logger.info(f"正在用 RAG 檢索相關文件...")
            results = self.rag_service.query(request.query)

            if not results:
                logger.info("找不到相關文件，無法生成答案。")
                return model_service_pb2.AnswerQuestionResponse(
                    answer="抱歉，我在知識庫中找不到與您問題相關的資訊。",
                    sources=[],
                    success=True
                )

            sources = list(set([doc.metadata.get('source', 'N/A') for doc in results]))
            logger.info(f"找到 {len(sources)} 個相關文件來源: {sources}")

            rag_context = " ".join([doc.page_content for doc in results])
            prompt = f"""
            System: 你是一個樂於助人的助理，會根據提供的上下文來回答問題。你的答案應該要簡潔，並使用與問題相同的語言。請直接回答問題，不要補充無關的資訊。

            Context:
            {rag_context}
            
            Question: {request.query}
            
            Answer:
            """
            logger.info("正在生成最終答案...")
            logger.info(f"發送給 LLM 的完整 Prompt:\n{prompt[:1000]}...") # Log first 1000 chars of prompt

            generation_config = {
                "max_new_tokens": 150,
                "temperature": 0.7,
                "top_p": 0.95,
                "do_sample": True,
                "pad_token_id": self.llm_servicer.llm_model.tokenizer.eos_token_id
            }

            if self.llm_servicer.llm_model.model_type == "causal":
                generation_config["return_full_text"] = False
            
            outputs = self.llm_servicer.llm_model.generator(
                prompt,
                **generation_config
            )
            
            raw_generated_text = outputs[0]["generated_text"]
            logger.info(f"LLM 原始生成文本:\n{raw_generated_text[:1000]}...") # Log raw output
            
            final_answer = raw_generated_text.strip()
            logger.info(f"答案生成成功: {final_answer[:100]}...")

            return model_service_pb2.AnswerQuestionResponse(
                answer=final_answer,
                sources=sources,
                success=True
            )

        except Exception as e:
            logger.error(f"AnswerQuestionFromDocuments 處理失敗: {e}")
            import traceback
            logger.error(f"詳細錯誤: {traceback.format_exc()}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"處理問答時發生內部錯誤: {str(e)}")
            return model_service_pb2.AnswerQuestionResponse(success=False)

class SpeakerAnnoteServicer:
    """語者辨識服務的包裝器"""
    
    def __init__(self):
        self.diarization_model = None
        logger.info("SpeakerAnnoteServicer 已初始化")
    
    def initialize(self) -> bool:
        try:
            from apis.identify import OfficialRealtimeDiarizer
            logger.info("正在載入語者辨識模型...")
            self.diarization_model = OfficialRealtimeDiarizer(clustering_threshold=0.7)
            logger.info("語者辨識模型載入成功")
            return True
        except Exception as e:
            logger.error(f"語者辨識模型初始化失敗: {e}")
            return False
            
    def _merge_segments(self, diarization_results, max_silence_for_merge=2.0):
        if not diarization_results:
            return []
        diarization_results.sort(key=lambda x: x[1])
        merged = []
        current_speaker, current_start, current_end = diarization_results[0]
        for i in range(1, len(diarization_results)):
            next_speaker, next_start, next_end = diarization_results[i]
            if (next_speaker == current_speaker and (next_start - current_end) < max_silence_for_merge):
                current_end = next_end
            else:
                merged.append((current_speaker, current_start, current_end))
                current_speaker, current_start, current_end = next_speaker, next_start, next_end
        merged.append((current_speaker, current_start, current_end))
        return merged

    def SpeakerAnnote(self, request, context):
        if self.diarization_model is None:
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details("語者辨識模型未初始化")
            return model_service_pb2.SpeakerAnnoteResponse()
        
        try:
            logger.info(f"收到語者辨識請求，音訊數據大小: {len(request.audio_data)} bytes")
            
            import io
            import soundfile as sf
            import numpy as np
            
            audio_buffer = io.BytesIO(request.audio_data)
            audio_data, sample_rate = sf.read(audio_buffer)
            
            if len(audio_data.shape) > 1:
                audio_data = np.mean(audio_data, axis=1)
            audio_data = audio_data.astype(np.float32)
            
            if sample_rate != 16000:
                import librosa
                audio_data = librosa.resample(y=audio_data, orig_sr=sample_rate, target_sr=16000)
                logger.info("已將音訊重新採樣至 16000Hz")
            
            self.diarization_model.process(audio_data)
            raw_segments = self.diarization_model.flush()
            logger.info(f"講者分辨完成，原始片段數量: {len(raw_segments)}")

            merged_results = self._merge_segments(raw_segments)
            logger.info(f"片段合併完成，合併後片段數量: {len(merged_results)}")

            all_segments = []
            speaker_timelines_dict = {}
            
            for speaker, start_time, end_time in merged_results:
                segment = model_service_pb2.DiarizationSegment(
                    speaker=speaker,
                    start_time=float(start_time),
                    end_time=float(end_time)
                )
                all_segments.append(segment)
                
                if speaker not in speaker_timelines_dict:
                    speaker_timelines_dict[speaker] = []
                speaker_timelines_dict[speaker].append(segment)
            
            timeline_objects = []
            for speaker, segments in speaker_timelines_dict.items():
                timeline = model_service_pb2.SpeakerTimeline(
                    speaker=speaker,
                    segments=segments
                )
                timeline_objects.append(timeline)
            
            response = model_service_pb2.SpeakerAnnoteResponse(
                all_segments=all_segments,
                speaker_timelines=timeline_objects  # <-- THE FIX
            )
            
            logger.info("講者分辨結果已準備完成")
            return response
            
        except Exception as e:
            logger.error(f"語者辨識處理失敗: {e}")
            import traceback
            logger.error(f"詳細錯誤: {traceback.format_exc()}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"語者辨識處理失敗: {str(e)}")
            return model_service_pb2.SpeakerAnnoteResponse()

class ServerManager:
    """伺服器管理器，負責初始化和管理所有服務"""
    
    def __init__(self):
        self.translator_api = None
        self.tts_servicer = None
        self.wav2lip_servicer = None
        self.speaker_annote_servicer = None
        self.llm_servicer = None
        self.speech_recognition_servicer = None
        self.rag_service = None
        self.server = None
        
    def initialize_models(self) -> bool:
        try:
            logger.info("正在初始化翻譯服務...")
            self.translator_api = TranslatorService()
            if not self.translator_api.initialize():
                return False
            
            logger.info("正在初始化 TTS 服務...")
            self.tts_servicer = TtsServicer()
            
            logger.info("正在初始化 Wav2Lip 服務...")
            self.wav2lip_servicer = Wav2LipServicer()
            
            logger.info("正在初始化語者辨識服務...")
            self.speaker_annote_servicer = SpeakerAnnoteServicer()
            if not self.speaker_annote_servicer.initialize():
                logger.warning("語者辨識服務初始化失敗，但繼續啟動其他服務")
            
            logger.info("正在初始化語音識別服務...")
            try:
                self.speech_recognition_servicer = SpeechRecognitionServicer(model_size="base")
                if self.speech_recognition_servicer.initialize():
                    logger.info("✅ 語音識別服務初始化成功")
                else:
                    logger.warning("❌ 語音識別服務初始化失敗，但繼續啟動其他服務")
                    self.speech_recognition_servicer = None
            except Exception as e:
                logger.warning(f"❌ 語音識別服務初始化失敗: {e}，但繼續啟動其他服務")
                self.speech_recognition_servicer = None
            
            logger.info("正在初始化 LLM 服務 (使用 Qwen)... ")
            try:
                self.llm_servicer = LLMServicer(model_name="Qwen/Qwen1.5-1.8B-Chat")
                logger.info("✅ LLM 服務初始化成功")
            except Exception as e:
                logger.warning(f"❌ LLM 服務初始化失敗: {e}")
                self.llm_servicer = None

            logger.info("正在初始化 RAG 服務...")
            try:
                self.rag_service = RAGService()
                logger.info("✅ RAG 服務初始化成功")
            except Exception as e:
                logger.warning(f"❌ RAG 服務初始化失敗: {e}")
                self.rag_service = None
            
            return True
            
        except Exception as e:
            logger.error(f"服務初始化失敗: {e}")
            return False
    
    def setup_server(self):
        self.server = grpc.server(
            futures.ThreadPoolExecutor(max_workers=10),
            options=[
                ('grpc.max_send_message_length', MAX_MESSAGE_LENGTH),
                ('grpc.max_receive_message_length', MAX_MESSAGE_LENGTH),
                ('grpc.max_receive_metadata_size', MAX_METADATA_SIZE),
                ('grpc.max_send_metadata_size', MAX_METADATA_SIZE),
            ]
        )
        
        model_service_pb2_grpc.add_TranslatorServiceServicer_to_server(
            TranslatorServicer(self.translator_api), 
            self.server
        )
        
        media_servicer = MediaServicer(
            tts_servicer=self.tts_servicer,
            wav2lip_servicer=self.wav2lip_servicer,
            speaker_annote_servicer=self.speaker_annote_servicer,
            llm_servicer=self.llm_servicer,
            speech_recognition_servicer=self.speech_recognition_servicer,
            rag_service=self.rag_service
        )
        model_service_pb2_grpc.add_MediaServiceServicer_to_server(
            media_servicer, 
            self.server
        )
        
        self.server.add_insecure_port('0.0.0.0:50051')
        logger.info("gRPC 伺服器設定完成（包含 RAG, LLM 和語音識別服務）")
    
    def start_server(self):
        if not self.initialize_models():
            logger.error("服務初始化失敗，伺服器無法啟動")
            return
        
        self.setup_server()
        self.server.start()
        
        services = ["🔤 翻譯服務", "🔊 TTS 服務", "🎬 Wav2Lip 服務"]
        if self.speaker_annote_servicer:
            services.append("👥 語者辨識服務")
        if self.speech_recognition_servicer:
            services.append("🎤 語音識別服務")
        if self.llm_servicer:
            services.append("🤖 LLM 服務")
        if self.rag_service:
            services.append("📚 RAG 問答服務")

        logger.info("🚀 gRPC 伺服器已成功啟動，監聽埠 50051...")
        logger.info(f"📋 可用服務: {', '.join(services)}")
        
        try:
            while True:
                time.sleep(86400)
        except KeyboardInterrupt:
            logger.info("收到關閉信號，正在關閉伺服器...")
            self.server.stop(0)
            logger.info("伺服器已關閉")
        

def serve():
    server_manager = ServerManager()
    server_manager.start_server()

if __name__ == '__main__':
    serve()
