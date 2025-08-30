# gRPC Client Testing Guide

This guide provides comprehensive instructions for testing all gRPC services in the Immersive Multilingual Meeting Assistant Server, including translation, TTS, speech recognition, and virtual avatar services.

## Overview

The server provides multiple gRPC services that can be tested using various client implementations:

- **Translation Service**: Text translation between multiple languages
- **TTS Service**: Text-to-speech conversion with multi-language support
- **Speech Recognition Service**: Real-time speech-to-text with speaker identification
- **Virtual Avatar Service**: AI-powered virtual avatar with lip-sync capabilities
- **RAG QA Service**: Question-answering using retrieval-augmented generation

## Available Test Clients

### 1. Main Integration Client (`client.py`)

The primary client that tests all services comprehensively:

```bash
python client.py
```

**Features:**
- Tests all gRPC services in sequence
- Automatic prerequisite checking
- Comprehensive error handling
- Multi-language testing scenarios

### 2. STT Dedicated Client (`test_stt_client.py`)

Specialized client for speech recognition testing:

```bash
python test_stt_client.py
```

**Features:**
- Streaming audio recognition
- Real-time transcription
- Speaker identification testing
- Performance metrics reporting

### 3. TTS Dedicated Client (`test_tts_service.py`)

Focused client for text-to-speech testing:

```bash
python test_tts_service.py
```

**Features:**
- Multi-language TTS testing
- Audio output generation
- Quality assessment
- Performance benchmarking

### 4. Virtual Avatar Client (`test_avatar_client.py`)

Specialized client for virtual avatar functionality:

```bash
python test_avatar_client.py
```

**Features:**
- Avatar initialization testing
- Multi-language speech synthesis
- Interactive conversation mode
- Stress testing capabilities

## Prerequisites

### System Requirements

1. **Python Environment**: Python 3.8+ with required dependencies
2. **gRPC Server**: Server must be running on `localhost:50051`
3. **Test Assets**: Required media files for testing
4. **Virtual Devices** (Optional): For avatar testing

### Required Dependencies

Install from `environment.yml`:
```bash
conda env create -f environment.yml
conda activate server
```

Key packages:
- `grpcio` and `grpcio-tools`
- `soundfile` and `librosa`
- `numpy` and `opencv-python`
- `pyvirtualcam` (for virtual camera)

### Test Assets

Ensure these files exist:
- `wav2lip_sample/tom.jpg` - Avatar reference image
- `identify_sample/ta.wav` - Voice sample for avatar
- Additional audio files for STT testing

## Testing Procedures

### 1. Server Preparation

Start the gRPC server:
```bash
python server.py
```

Verify server is running:
```bash
# Check if port is accessible
netstat -an | grep 50051

# Test gRPC connection
python -c "
import grpc
channel = grpc.insecure_channel('localhost:50051')
grpc.channel_ready_future(channel).result(timeout=5)
print('Server is ready')
"
```

### 2. Translation Service Testing

Using the main client:
```bash
python client.py
```

The client will test various translation scenarios:
- English to Chinese
- Chinese to Japanese
- German to English
- French to Spanish
- Error handling for unsupported languages

Expected output format:
```
[Client] Sending translation request: 'Hello world' (English -> Chinese)
[Client] Received translation result: '你好世界'
```

### 3. TTS Service Testing

Direct TTS testing:
```bash
python test_tts_service.py
```

Test parameters:
- Multiple languages (en, zh-cn, ja, ko, etc.)
- Various text lengths
- Audio quality assessment
- Output file generation

Expected behavior:
- Audio files generated in specified format
- Proper language-specific pronunciation
- Consistent audio quality

### 4. Speech Recognition Testing

STT streaming test:
```bash
python test_stt_client.py
```

Testing features:
- Real-time audio streaming
- Transcription accuracy
- Speaker identification
- Performance metrics (RTF - Real Time Factor)

Expected output:
```
Transcription (Final): 'This is a test of speech recognition'
Segment 1 [0.00s - 2.50s]: 'This is a test'
  Speaker: speaker_001
  Word 1 [0.00s - 0.25s]: 'This' (confidence: 0.95)
```

### 5. Virtual Avatar Testing

Comprehensive avatar testing:
```bash
python test_avatar_client.py
```

Test modes available:
1. **Complete test suite** - All functionality
2. **Basic functionality** - Core features only
3. **Multi-language testing** - Language variations
4. **Stress testing** - Performance limits
5. **Interactive testing** - Manual conversation

Testing flow:
1. Prerequisites verification
2. Avatar initialization
3. Speech synthesis testing
4. Multi-language capabilities
5. Performance evaluation

## Error Handling and Debugging

### Common gRPC Errors

1. **Connection Refused**
```
grpc.RpcError: <_InactiveRpcError of RPC that terminated with:
    status = StatusCode.UNAVAILABLE
```
**Solution**: Ensure server is running on correct port

2. **Method Not Implemented**
```
grpc.RpcError: <_InactiveRpcError of RPC that terminated with:
    status = StatusCode.UNIMPLEMENTED
```
**Solution**: Check if specific service is enabled in server

3. **Message Size Limits**
```
grpc.RpcError: <_InactiveRpcError of RPC that terminated with:
    status = StatusCode.RESOURCE_EXHAUSTED
```
**Solution**: Adjust message size limits in client configuration

### Debug Configuration

Enable detailed logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

gRPC channel options for large messages:
```python
channel_options = [
    ('grpc.max_send_message_length', 100 * 1024 * 1024),
    ('grpc.max_receive_message_length', 100 * 1024 * 1024),
    ('grpc.max_receive_metadata_size', 2 * 1024 * 1024),
    ('grpc.max_send_metadata_size', 2 * 1024 * 1024),
]
```

### Performance Monitoring

Key metrics to monitor:
- **Response Time**: Service call duration
- **Throughput**: Requests per second
- **Error Rate**: Failed request percentage
- **Resource Usage**: CPU and memory consumption

Example performance logging:
```python
start_time = time.time()
response = stub.SomeMethod(request)
duration = time.time() - start_time
logger.info(f"Method completed in {duration:.2f} seconds")
```

## Test Scenarios

### Functional Testing

1. **Basic Service Availability**
   - Verify all services respond
   - Check method implementations
   - Validate response formats

2. **Data Flow Testing**
   - Text processing pipeline
   - Audio streaming capabilities
   - Binary data handling

3. **Multi-language Support**
   - Translation accuracy
   - TTS quality across languages
   - Character encoding handling

### Integration Testing

1. **Service Chaining**
   - STT -> Translation -> TTS pipeline
   - Avatar with multi-language speech
   - End-to-end conversation flow

2. **Concurrent Usage**
   - Multiple client connections
   - Parallel service calls
   - Resource sharing

### Stress Testing

1. **Load Testing**
   - High request volume
   - Large message sizes
   - Extended operation periods

2. **Resource Limits**
   - Memory usage patterns
   - Processing time limits
   - Connection pooling

## Production Considerations

### Security

1. **TLS Configuration**
```python
credentials = grpc.ssl_channel_credentials()
channel = grpc.secure_channel('server:443', credentials)
```

2. **Authentication**
```python
def auth_interceptor(method, request, callback, metadata):
    metadata = list(metadata or [])
    metadata.append(('authorization', f'Bearer {token}'))
    return callback(request, metadata)
```

### Reliability

1. **Retry Logic**
```python
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def call_service(stub, request):
    return stub.SomeMethod(request)
```

2. **Circuit Breaker Pattern**
```python
circuit_breaker = CircuitBreaker(failure_threshold=5, recovery_timeout=30)

@circuit_breaker
def protected_call(stub, request):
    return stub.SomeMethod(request)
```

### Monitoring

1. **Health Checks**
```python
health_stub = health_pb2_grpc.HealthStub(channel)
health_request = health_pb2.HealthCheckRequest(service="")
response = health_stub.Check(health_request)
```

2. **Metrics Collection**
```python
def collect_metrics(method_name, duration, success):
    metrics.histogram('grpc_duration_seconds', 
                     duration, 
                     labels={'method': method_name, 'success': success})
```

## Troubleshooting Guide

### Installation Issues

1. **Protocol Buffer Compilation**
```bash
python -m grpc_tools.protoc \
  --proto_path=. \
  --python_out=./proto \
  --grpc_python_out=./proto \
  model_service.proto
```

2. **Dependency Conflicts**
```bash
conda env export > current_env.yml
conda env create -f environment.yml --force
```

### Runtime Issues

1. **Audio Device Problems**
```bash
# List available audio devices
python -c "import sounddevice; print(sounddevice.query_devices())"
```

2. **Virtual Camera Setup**
```bash
# Install OBS Virtual Camera or v4l2loopback
# Windows: Use OBS Studio Virtual Camera
# Linux: sudo modprobe v4l2loopback
```

### Service-Specific Issues

1. **STT Service**: Check audio format compatibility (16kHz, mono, 16-bit PCM)
2. **TTS Service**: Verify language model availability
3. **Translation Service**: Ensure model files are downloaded
4. **Avatar Service**: Check GPU memory availability

## Best Practices

### Client Implementation

1. **Connection Management**
```python
class GrpcClient:
    def __init__(self, address):
        self.channel = grpc.insecure_channel(address)
        self.stub = model_service_pb2_grpc.SomeServiceStub(self.channel)
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.channel.close()
```

2. **Error Handling**
```python
try:
    response = stub.SomeMethod(request)
    if hasattr(response, 'success') and not response.success:
        logger.error(f"Service error: {response.error_message}")
except grpc.RpcError as e:
    logger.error(f"gRPC error: {e.code()} - {e.details()}")
```

3. **Streaming Best Practices**
```python
def stream_audio(stub, audio_generator):
    try:
        responses = stub.StreamingRecognize(audio_generator)
        for response in responses:
            yield response
    except grpc.RpcError as e:
        logger.error(f"Streaming error: {e}")
        raise
```

This guide provides a comprehensive framework for testing all gRPC services in the system. Follow the procedures systematically to ensure proper functionality and identify any issues early in the development