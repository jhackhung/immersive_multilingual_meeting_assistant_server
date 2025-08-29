# API Reference

## Overview

This service provides core gRPC APIs covering the complete workflow of multilingual meeting assistance.

## 🌍 1. Translation Service

### Description
Multilingual text translation using mBART-50 model, supporting 50+ language pairs.

### Features
- **50+ Languages**: Support for major world languages
- **Auto Detection**: Automatic source language detection
- **High Accuracy**: mBART-50 based translation quality

### Usage Example
```python
import grpc
import proto.model_service_pb2_grpc as pb2_grpc
import proto.model_service_pb2 as pb2

channel = grpc.insecure_channel('localhost:50051')
stub = pb2_grpc.TranslatorServiceStub(channel)

request = pb2.TranslateRequest(
    text_to_translate="Hello, how are you?",
    source_language="en",
    target_language="zh-CN"
)
response = stub.Translate(request)
print(f"Translation: {response.translated_text}")
```

### Supported Languages
- **Chinese**: zh-CN, zh-TW
- **English**: en
- **Japanese**: ja
- **Korean**: ko
- **French**: fr
- **German**: de
- **Spanish**: es
- **More**: 50+ total languages

---

## 🔊 2. Text-to-Speech Service

### Description
High-quality multilingual speech synthesis based on XTTS-v2 with voice cloning capabilities.

### Features
- **Voice Cloning**: Only 6-10 seconds of audio sample needed
- **Fast Mode**: 5-10 seconds completion for short texts
- **ONNX Optimization**: GPU/NPU accelerated inference
- **Memory Caching**: Intelligent caching strategy

### Usage Example
```python
request = pb2.TtsRequest(
    text_to_speak="Hello, this is a test.",
    language="en"
)
response = media_stub.Tts(request)
# Save response.generated_audio to file
```

---

## 🎙️ 3. Speech Recognition Service

### Description
High-accuracy speech-to-text using OpenAI Whisper model.

### Features
- **11+ Languages**: Multi-language support
- **Auto Detection**: Automatic language detection
- **Noise Suppression**: Audio preprocessing and noise reduction
- **Timestamps**: Word-level timing information

### Usage Example
```python
request = pb2.SpeechRecognitionRequest(
    audio_data=audio_bytes,
    language="auto",
    return_timestamps=True
)
response = media_stub.SpeechRecognition(request)
print(f"Transcription: {response.transcribed_text}")
```

---

## 👥 4. Speaker Identification Service

### Description
Real-time speaker separation and identification using PyAnnote Audio.

### Features
- **Auto Speaker Count**: Automatic detection of speaker numbers
- **Timestamps**: Time-stamped speaker segments
- **Voice Print Analysis**: Speaker voice pattern analysis

### Usage Example
```python
request = pb2.SpeakerAnnoteRequest(audio_data=audio_bytes)
response = media_stub.SpeakerAnnote(request)

# Method 1: Chronological segments
for segment in response.all_segments:
    print(f"{segment.speaker}: {segment.start_time:.2f}s - {segment.end_time:.2f}s")

# Method 2: Grouped by speaker
for timeline in response.speaker_timelines:
    print(f"Speaker {timeline.speaker}:")
    for seg in timeline.segments:
        print(f"  {seg.start_time:.2f}s - {seg.end_time:.2f}s")
```

---

## 🤖 5. LLM Q&A Service

### Description
AI-powered intelligent Q&A based on Qwen 1.5, supporting meeting content understanding.

### Features
- **Meeting Understanding**: Context-aware responses
- **Multi-turn Conversation**: Support for conversation history
- **Content Summarization**: Automatic meeting summary generation

### Usage Example
```python
messages = [
    pb2.ChatMessage(role="system", content="You are a meeting assistant"),
    pb2.ChatMessage(role="user", content="Summarize the key points from the meeting")
]

request = pb2.ChatCompletionRequest(
    messages=messages,
    max_tokens=500,
    temperature=0.7
)
response = media_stub.ChatCompletion(request)
print(f"AI Response: {response.response}")
```

---

## 🎬 6. Virtual Avatar Service

### Description
Lip-sync virtual avatar generation using Wav2Lip technology.

### Features
- **Realistic Lip Sync**: High-accuracy facial animation
- **Multiple Formats**: JPG, PNG image input, MP4 video output
- **GPU Acceleration**: Significantly improved processing speed

### Usage Example
```python
# Method 1: Direct generation
request = pb2.Wav2LipRequest(
    audio_data=audio_bytes,
    image_data=face_image_bytes
)
response = media_stub.Wav2Lip(request)
# Save response.video_data to file

# Method 2: Initialize avatar first
init_request = pb2.InitAvatarRequest(
    image_data=face_image_bytes,
    sample_audio_data=voice_sample_bytes
)
init_response = media_stub.InitAvatar(init_request)

# Then make avatar speak
speak_request = pb2.AvatarSpeakRequest(
    text="Hello, nice to meet you!",
    language="en"
)
speak_response = media_stub.AvatarSpeak(speak_request)
```

---

## 🔍 7. RAG Document Search Service

### Description
Intelligent document retrieval and Q&A system based on ChromaDB.

### Features
- **Semantic Search**: Vector similarity-based search
- **Multi-format Support**: PDF, DOCX, TXT documents
- **Context Enhancement**: RAG architecture for improved answer quality
- **Source Attribution**: References to original documents

### Usage Example
```python
request = pb2.AnswerQuestionRequest(
    query="What were the main decisions made in today's meeting?"
)
response = media_stub.AnswerQuestionFromDocuments(request)
print(f"Answer: {response.answer}")
print(f"Sources: {', '.join(response.sources)}")
```

---

## 🌊 8. Streaming Recognition Service

### Description
Bidirectional streaming speech recognition for real-time applications.

### Features
- **Real-time Processing**: Live audio transcription
- **Speaker Detection**: Integrated speaker identification
- **Low Latency**: Sub-second response time
- **Session Management**: Continuous conversation tracking

### Usage Example
```python
def stream_audio():
    for chunk in audio_chunks:
        yield pb2.StreamingRecognizeRequest(
            audio=pb2.AudioChunk(audio_bytes=chunk),
            language="en"
        )

# Stream processing
responses = media_stub.StreamingRecognize(stream_audio())
for response in responses:
    if response.is_final:
        print(f"Final: {response.transcript_text}")
    else:
        print(f"Interim: {response.transcript_text}")
```

---

## 📝 Complete Workflow Example

### Multilingual Meeting Processing Pipeline
```python
import grpc
import proto.model_service_pb2_grpc as pb2_grpc
import proto.model_service_pb2 as pb2

# Establish connections
channel = grpc.insecure_channel('localhost:50051')
translator_stub = pb2_grpc.TranslatorServiceStub(channel)
media_stub = pb2_grpc.MediaServiceStub(channel)

# 1. Speech Recognition
stt_request = pb2.SpeechRecognitionRequest(
    audio_data=audio_bytes,
    language="auto",
    return_timestamps=True
)
stt_response = media_stub.SpeechRecognition(stt_request)

# 2. Speaker Identification
speaker_request = pb2.SpeakerAnnoteRequest(audio_data=audio_bytes)
speaker_response = media_stub.SpeakerAnnote(speaker_request)

# 3. Translation
translate_request = pb2.TranslateRequest(
    text_to_translate=stt_response.transcribed_text,
    source_language=stt_response.detected_language,
    target_language="zh-CN"
)
translate_response = translator_stub.Translate(translate_request)

# 4. Text-to-Speech
tts_request = pb2.TtsRequest(
    text_to_speak=translate_response.translated_text,
    language="zh-CN"
)
tts_response = media_stub.Tts(tts_request)

# 5. Virtual Avatar
avatar_request = pb2.Wav2LipRequest(
    audio_data=tts_response.generated_audio,
    image_data=face_image_bytes
)
avatar_response = media_stub.Wav2Lip(avatar_request)

# 6. AI Assistant Summary
chat_request = pb2.ChatCompletionRequest(
    messages=[
        pb2.ChatMessage(role="system", content="You are a meeting assistant"),
        pb2.ChatMessage(role="user", content=f"Summarize this meeting: {stt_response.transcribed_text}")
    ]
)
chat_response = media_stub.ChatCompletion(chat_request)
```

## 🔧 Error Handling

All APIs use standard gRPC status codes:

- `OK`: Success
- `INVALID_ARGUMENT`: Parameter error
- `RESOURCE_EXHAUSTED`: Resource insufficient
- `INTERNAL`: Internal error
- `UNAVAILABLE`: Service unavailable

### Error Handling Example
```python
import grpc

try:
    response = stub.SomeMethod(request)
    if hasattr(response, 'success') and not response.success:
        print(f"Service error: {response.error_message}")
    else:
        # Process successful response
        pass
except grpc.RpcError as e:
    print(f"gRPC error: {e.code()} - {e.details()}")
```

## 📈 Performance Guidelines

### Recommended Configuration
- **Concurrent Connections**: 10-50
- **Request Timeout**: 30-60 seconds
- **GPU Memory**: 8GB+
- **Batch Size**: 1-4

### Best Practices
1. Use connection pools for gRPC connections
2. Implement appropriate retry mechanisms
3. Monitor memory usage
4. Enable GPU acceleration (if available)

### Performance Metrics

| Service | Processing Time | GPU Memory | Key Features |
|---------|----------------|------------|--------------|
| **Translation** | < 1 sec | ~1GB | 50+ languages |
| **TTS** | 5-10 sec | ~3GB | Voice cloning |
| **STT** | < 2 sec | ~2GB | Multi-language |
| **Speaker ID** | < 3 sec | ~1.5GB | Real-time separation |
| **LLM Chat** | 2-5 sec | ~2GB | Intelligent Q&A |
| **Avatar** | 10-30 sec | ~4GB | Lip sync |
| **RAG Search** | < 1 sec | ~0.5GB | Semantic search |

## 🛠️ Development Tools

### Testing Services
```bash
# Test specific services
python test_tts_service.py
python test_speaker_identification.py

# Connection test
python client.py

# Health check
grpcurl -plaintext localhost:50051 list
```

### Connection Management
```python
# Recommended: Use connection pooling
import grpc
from grpc_health.v1 import health_pb2_grpc

# Create channel with options
options = [
    ('grpc.keepalive_time_ms', 30000),
    ('grpc.keepalive_timeout_ms', 5000),
    ('grpc.keepalive_permit_without_calls', True),
    ('grpc.http2.max_pings_without_data', 0),
    ('grpc.http2.min_time_between_pings_ms', 10000),
    ('grpc.http2.min_ping_interval_without_data_ms', 300000)
]

channel = grpc.insecure_channel('localhost:50051', options=options)
```

## 📚 Related Documentation

### Proto Files Reference
For detailed protocol buffer definitions and message structures:

- [Proto Reference](proto-reference.md) - Complete protocol buffer definitions
