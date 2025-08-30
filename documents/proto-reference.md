# Protocol Buffers Reference

This document provides comprehensive documentation for the gRPC service definitions in `model_service.proto`.

## Service Overview

The system provides two main gRPC services:

| Service | Purpose | Methods |
|---------|---------|---------|
| **TranslatorService** | Text translation | `Translate` |
| **MediaService** | Audio/video processing and AI | 10 methods including TTS, STT, Avatar generation |

---

## TranslatorService

### Service Definition

```protobuf
service TranslatorService {
  rpc Translate (TranslateRequest) returns (TranslateResponse) {}
}
```

### Translate

**Purpose**: Translate text between supported languages using mBART-50 model.

#### Translate Request

```protobuf
message TranslateRequest {
  string text_to_translate = 1;  // Text to be translated
  string source_language = 2;    // Source language code
  string target_language = 3;    // Target language code
}
```

#### Translate Response

```protobuf
message TranslateResponse {
  string translated_text = 1;    // Translated text result
}
```

#### Usage Example

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

---

## MediaService

### MediaService Definition

```protobuf
service MediaService {
  rpc Wav2Lip(Wav2LipRequest) returns (Wav2LipResponse);
  rpc Tts(TtsRequest) returns (TtsResponse);
  rpc SpeakerAnnote(SpeakerAnnoteRequest) returns (SpeakerAnnoteResponse);
  rpc SpeechRecognition(SpeechRecognitionRequest) returns (SpeechRecognitionResponse);
  rpc GenerateText(TextGenerationRequest) returns (TextGenerationResponse);
  rpc ChatCompletion(ChatCompletionRequest) returns (ChatCompletionResponse);
  rpc AnswerQuestionFromDocuments(AnswerQuestionRequest) returns (AnswerQuestionResponse);
  rpc StreamingRecognize(stream StreamingRecognizeRequest) returns (stream StreamingRecognizeResponse);
  rpc InitAvatar(InitAvatarRequest) returns (InitAvatarResponse);
  rpc AvatarSpeak(AvatarSpeakRequest) returns (AvatarSpeakResponse);
}
```

---

### Wav2Lip

**Purpose**: Generate lip-sync videos using face images and audio.

#### Wav2Lip Request

```protobuf
message Wav2LipRequest {
  bytes audio_data = 1;    // Raw audio file bytes
  bytes image_data = 2;    // Raw image file bytes
}
```

#### Wav2Lip Response

```protobuf
message Wav2LipResponse {
  bytes video_data = 1;    // Generated video file bytes (MP4)
}
```

---

### Tts (Text-to-Speech)

**Purpose**: Convert text to natural speech with optional voice cloning.

#### TTS Request

```protobuf
message TtsRequest {
  string text_to_speak = 1;     // Text to convert to speech
  bytes reference_audio = 2;    // Optional: Reference audio for voice cloning
  string language = 3;          // Language code
}
```

#### Response Message

```protobuf
message TtsResponse {
  bytes generated_audio = 1;    // Generated audio file bytes (WAV)
}
```

#### Features

- **Voice Cloning**: Upload reference audio for custom voice
- **ONNX Optimization**: GPU/NPU acceleration
- **Memory Caching**: Intelligent caching for repeated requests

---

### SpeakerAnnote (Speaker Diarization)

**Purpose**: Identify and separate different speakers in audio.

#### Request Message

```protobuf
message SpeakerAnnoteRequest {
  bytes audio_data = 1;    // Audio file bytes for diarization
}
```

#### Response Message

```protobuf
message SpeakerAnnoteResponse {
  repeated DiarizationSegment all_segments = 1;      // Flat list of segments
  repeated SpeakerTimeline speaker_timelines = 2;    // Grouped by speaker
}

message DiarizationSegment {
  string speaker = 1;      // Speaker label (e.g., "SPEAKER_00")
  double start_time = 2;   // Start time in seconds
  double end_time = 3;     // End time in seconds
}

message SpeakerTimeline {
  string speaker = 1;                        // Speaker identifier
  repeated DiarizationSegment segments = 2;  // All segments for this speaker
}
```

---

### SpeechRecognition (Speech-to-Text)

**Purpose**: Convert audio to text using OpenAI Whisper.

#### Request Message

```protobuf
message SpeechRecognitionRequest {
  bytes audio_data = 1;        // Audio file bytes
  string language = 2;         // Language code ("auto" for detection)
  bool return_timestamps = 3;  // Whether to return timestamp info
  string model_size = 4;       // Model size: "base", "small", "medium", "large"
}
```

#### Response Message

```protobuf
message SpeechRecognitionResponse {
  string transcribed_text = 1;              // Full transcription
  string detected_language = 2;             // Detected language code
  float language_confidence = 3;            // Language detection confidence
  repeated TranscriptionSegment segments = 4; // Timestamped segments
  bool success = 5;                         // Processing success flag
}

message TranscriptionSegment {
  string text = 1;         // Segment text
  double start_time = 2;   // Start time in seconds
  double end_time = 3;     // End time in seconds
}
```

---

### GenerateText (Text Generation)

**Purpose**: Generate text based on prompts using LLM.

#### Request Message

```protobuf
message TextGenerationRequest {
  string prompt = 1;        // Input prompt
  int32 max_tokens = 2;     // Maximum tokens to generate
  float temperature = 3;    // Creativity parameter (0.0-1.0)
  float top_p = 4;         // Top-p sampling parameter
}
```

#### Response Message

```protobuf
message TextGenerationResponse {
  string generated_text = 1; // Generated text
  bool success = 2;          // Success flag
  string error_message = 3;  // Error details if failed
}
```

---

### ChatCompletion (AI Chat Assistant)

**Purpose**: Multi-turn conversation with AI assistant using Qwen 1.5 model.

#### Request Message

```protobuf
message ChatCompletionRequest {
  repeated ChatMessage messages = 1; // Conversation history
  int32 max_tokens = 2;             // Maximum response tokens
  float temperature = 3;            // Creativity parameter (0.0-1.0)
}

message ChatMessage {
  string role = 1;      // "system", "user", "assistant"
  string content = 2;   // Message content
}
```

#### Response Message

```protobuf
message ChatCompletionResponse {
  string response = 1;       // AI assistant response
  bool success = 2;          // Success flag
  string error_message = 3;  // Error details if failed
}
```

---

### AnswerQuestionFromDocuments (RAG Q&A)

**Purpose**: Answer questions based on document knowledge using RAG.

#### Request Message

```protobuf
message AnswerQuestionRequest {
  string query = 1;    // User's question
}
```

#### Response Message

```protobuf
message AnswerQuestionResponse {
  string answer = 1;         // Generated answer
  repeated string sources = 2; // Source document references
  bool success = 3;          // Success flag
  string error_message = 4;  // Error details
}
```

---

### StreamingRecognize (Real-time STT)

**Purpose**: Bidirectional streaming speech recognition for real-time applications.

#### StreamingRecognize Request (Streaming)

```protobuf
message StreamingRecognizeRequest {
  AudioChunk audio = 1;  // Audio data chunk
  string language = 2;   // Language code
  bool is_last = 3;      // Mark end of audio stream
}

message AudioChunk {
  bytes audio_bytes = 1; // Raw audio bytes
}
```

#### StreamingRecognize Response (Streaming)

```protobuf
message StreamingRecognizeResponse {
  string session_id = 1;           // Session identifier
  string transcript_text = 2;      // Transcribed text
  bool is_final = 3;              // Whether result is final
  repeated Segment segments = 4;   // Structured segment info
  string message = 5;             // Status message
  double rtf = 6;                 // Real-time factor
  double chunk_sec = 7;           // Processed audio duration
  double server_time_sec = 8;     // Server processing time
}

message Segment {
  string text = 1;                // Recognized text
  float start_time_sec = 2;       // Start time
  float end_time_sec = 3;         // End time
  repeated WordInfo words = 4;    // Word-level information
  string speaker_id = 5;          // Speaker identifier
}

message WordInfo {
  string word = 1;             // Word text
  float start_time_sec = 2;    // Word start time
  float end_time_sec = 3;      // Word end time
  float confidence = 4;        // Recognition confidence
}
```

---

### Avatar Services

#### InitAvatar

**Purpose**: Initialize virtual avatar with face image and voice sample.

```protobuf
message InitAvatarRequest {
  bytes image_data = 1;        // Avatar face image
  bytes sample_audio_data = 2; // Voice sample for cloning
}

message InitAvatarResponse {
  bool success = 1;            // Initialization success
  string message = 2;          // Response message
}
```

#### AvatarSpeak

**Purpose**: Make initialized avatar speak specified text.

```protobuf
message AvatarSpeakRequest {
  string text = 1;     // Text to speak
  string language = 2; // Language code (optional, default "en")
}

message AvatarSpeakResponse {
  bool success = 1;            // Success flag
  string message = 2;          // Response message
}
```

---

## Error Handling

### Standard gRPC Status Codes

| Code | Description | Common Causes |
|------|-------------|---------------|
| `OK` | Success | Request completed successfully |
| `INVALID_ARGUMENT` | Invalid parameters | Missing required fields, invalid format |
| `RESOURCE_EXHAUSTED` | Resource limits | GPU memory full, rate limit exceeded |
| `INTERNAL` | Server error | Model loading failed, processing error |
| `UNAVAILABLE` | Service unavailable | Server overloaded, maintenance mode |

### Error Response Pattern

Services with `success` and `error_message` fields:

- `GenerateText`
- `ChatCompletion`  
- `AnswerQuestionFromDocuments`
- `SpeechRecognition`
- `InitAvatar`
- `AvatarSpeak`

### Client Error Handling Example

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

---

## Performance Guidelines

### Request Size Limits

| Service | Field | Recommendation |
|---------|-------|----------------|
| `Wav2Lip` | `audio_data` | < 30 seconds audio |
| `Wav2Lip` | `image_data` | 512x512 - 1024x1024 px |
| `Tts` | `text_to_speak` | < 1000 characters |
| `SpeechRecognition` | `audio_data` | < 60 minutes |
| `StreamingRecognize` | `audio_bytes` | 1-2 second chunks |

### Timeout Settings

| Service | Recommended Timeout |
|---------|-------------------|
| `Translate` | 10 seconds |
| `Tts` | 60 seconds |
| `SpeechRecognition` | 30 seconds |
| `Wav2Lip` | 120 seconds |
| `ChatCompletion` | 30 seconds |
| `StreamingRecognize` | No timeout (streaming) |

---

## Development Tools

### Protocol Buffer Compilation

```bash
# Generate Python code
python -m grpc_tools.protoc \
  --proto_path=. \
  --python_out=./proto \
  --grpc_python_out=./proto \
  model_service.proto
```

### Testing Services

```bash
# Test specific services
python test_tts_service.py
python test_speaker_identification.py

# Health check
grpcurl -plaintext localhost:50051 list
```
