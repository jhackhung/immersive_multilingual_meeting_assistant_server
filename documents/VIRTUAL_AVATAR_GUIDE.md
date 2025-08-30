# Virtual Avatar Implementation Guide

This document explains the implementation of virtual avatar functions that can create a virtual character using a provided image and sample audio, then feed it to virtual webcam and microphone for real-time avatar communication.

## Overview

The virtual avatar system consists of several integrated components:

1. **Virtual Avatar Service** (`apis/virtual_avatar_service.py`)
2. **Virtual Webcam Manager** - Streams generated videos to virtual camera
3. **Virtual Microphone Manager** - Streams generated audio to virtual microphone
4. **TTS Integration** - Text-to-speech using custom voice
5. **Wav2Lip Integration** - Lip-sync video generation

## Core Functions

### 1. InitAvatar(img, sample_audio)

Initializes a virtual avatar with:
- **img**: Image data (JPEG/PNG bytes) of the avatar face
- **sample_audio**: Audio data (WAV bytes) for voice cloning

```python
# Example usage
with open("avatar_image.jpg", "rb") as f:
    image_data = f.read()

with open("voice_sample.wav", "rb") as f:
    audio_data = f.read()

success = avatar_service.init_avatar(image_data, audio_data)
```

### 2. AvatarSpeak(text)

Makes the avatar speak with lip-sync:
- **text**: Text to be spoken
- **language**: Language code (optional, defaults to "en")

```python
# Example usage
success = avatar_service.avatar_speak("Hello, I am your virtual avatar!", "en")
```

## System Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────────┐
│   Client Input  │───▶│  Virtual Avatar  │───▶│   Virtual Devices   │
│   (Text)        │    │     Service      │    │                     │
└─────────────────┘    └──────────────────┘    └─────────────────────┘
                              │                          │
                              ▼                          ▼
                    ┌─────────────────────┐    ┌──────────────────────┐
                    │  TTS + Wav2Lip      │    │  Virtual Webcam      │
                    │  Processing         │    │  Virtual Microphone  │
                    └─────────────────────┘    └──────────────────────┘
```

## Installation Requirements

### 1. Python Dependencies

All required packages are included in `environment.yml`:

```yaml
- pyvirtualcam==0.13.0  # Virtual camera support
- sounddevice==0.5.2    # Audio device management
- opencv-python         # Video processing
- soundfile==0.13.1     # Audio file handling
```

### 2. Virtual Audio Device

Install VB-Cable or similar virtual audio driver:

**Windows:**
- Download VB-Cable from VB-Audio website
- Install and restart computer
- Device will appear as "CABLE Input" in audio settings

**macOS:**
- Install BlackHole virtual audio driver
- Configure as "BlackHole 2ch"

**Linux:**
- Use PulseAudio virtual sink:
```bash
pacmd load-module module-null-sink sink_name=virtual_mic
```

### 3. Virtual Camera Support

**Windows/macOS:**
- OBS Studio with Virtual Camera plugin (built-in from OBS 26+)

**Linux:**
- v4l2loopback kernel module:
```bash
sudo modprobe v4l2loopback devices=1 video_nr=2 card_label="Virtual Camera"
```

## Usage Examples

### 1. Standalone Demo

Run the demo script to test basic functionality:

```bash
python demo_virtual_avatar.py
```

### 2. gRPC Server Integration

Start the server with virtual avatar support:

```bash
python server.py
```

Use the test client:

```bash
python test_virtual_avatar.py
```

### 3. gRPC API Usage

```python
import grpc
from proto import model_service_pb2, model_service_pb2_grpc

# Connect to server
channel = grpc.insecure_channel('localhost:50051')
stub = model_service_pb2_grpc.MediaServiceStub(channel)

# Initialize avatar
init_request = model_service_pb2.InitAvatarRequest(
    image_data=image_bytes,
    sample_audio_data=audio_bytes
)
response = stub.InitAvatar(init_request)

# Make avatar speak
speak_request = model_service_pb2.AvatarSpeakRequest(
    text="Hello world!",
    language="en"
)
response = stub.AvatarSpeak(speak_request)
```

## Integration with Applications

### 1. Video Conferencing

1. Start the avatar service
2. In your video conferencing app (Zoom, Teams, etc.):
   - Select "OBS Virtual Camera" as camera source
   - Select "CABLE Input" as microphone source
3. Use `AvatarSpeak()` to make the avatar talk during calls

### 2. Live Streaming

1. Configure OBS Studio:
   - Add "Video Capture Device" source
   - Select "OBS Virtual Camera"
   - Add "Audio Input Capture" source
   - Select "CABLE Input"
2. Stream normally while controlling avatar via API

### 3. Voice Assistants

```python
# Example integration with voice assistant
def on_voice_command(text):
    # Process the command
    response = process_command(text)
    
    # Make avatar respond
    avatar_service.avatar_speak(response, "en")
```

## Technical Details

### Processing Pipeline

1. **Text Input** → TTS Service → **Audio Generation**
2. **Avatar Image + Generated Audio** → Wav2Lip → **Lip-sync Video**
3. **Generated Audio** → Virtual Microphone → **Audio Output**
4. **Generated Video** → Virtual Webcam → **Video Output**

### Performance Considerations

- **TTS Generation**: ~2-3 seconds for short sentences
- **Wav2Lip Processing**: ~1-2 seconds per second of audio
- **Memory Usage**: ~2-4GB for full pipeline
- **GPU Acceleration**: Recommended for real-time performance

### Synchronization

The system automatically synchronizes audio and video:
- Audio frames are queued to virtual microphone
- Video frames are queued to virtual webcam
- Frame timing ensures lip-sync accuracy

## Troubleshooting

### Common Issues

1. **Virtual devices not found**
   - Ensure virtual audio/video drivers are installed
   - Restart applications after driver installation

2. **Poor lip-sync quality**
   - Use higher quality input images (clear face, good lighting)
   - Ensure sample audio is clear and representative

3. **High latency**
   - Use GPU acceleration if available
   - Reduce batch sizes in processing pipeline
   - Use faster TTS models for real-time applications

4. **Audio crackling**
   - Increase audio buffer size
   - Check sample rate compatibility
   - Ensure virtual audio device is properly configured

### Debug Mode

Enable debug logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Performance Monitoring

Check system resources:
- GPU memory usage (for CUDA-enabled processing)
- CPU usage (for audio/video encoding)
- Memory usage (for model loading and caching)

## Advanced Configuration

### Custom Voice Models

Replace the default TTS model with custom trained voices:

```python
# In apis/virtual_avatar_service.py
tts_service = TtsServicer(model_path="path/to/custom/model")
```

### Video Quality Settings

Adjust video output parameters:

```python
# In VirtualWebcamManager
virtual_webcam = VirtualWebcamManager(
    width=1920,  # Higher resolution
    height=1080,
    fps=30       # Higher frame rate
)
```

### Audio Quality Settings

Configure audio parameters:

```python
# In VirtualMicrophoneManager
virtual_mic = VirtualMicrophoneManager(
    sample_rate=48000,  # Professional audio quality
    block_size=512      # Lower latency
)
```

## API Reference

### VirtualAvatarService Methods

```python
class VirtualAvatarService:
    def init_avatar(self, image_data: bytes, sample_audio_data: bytes) -> bool:
        """Initialize virtual avatar with image and voice sample"""
        
    def avatar_speak(self, text: str, language: str = "en") -> bool:
        """Make avatar speak the given text"""
        
    def cleanup(self):
        """Clean up resources and stop virtual devices"""
```

### gRPC Messages

```protobuf
message InitAvatarRequest {
    bytes image_data = 1;
    bytes sample_audio_data = 2;
}

message InitAvatarResponse {
    bool success = 1;
    string message = 2;
}

message AvatarSpeakRequest {
    string text = 1;
    string language = 2;
}

message AvatarSpeakResponse {
    bool success = 1;
    string message = 2;
}
```

## Future Enhancements

1. **Real-time Processing**: Reduce latency for live conversations
2. **Expression Control**: Add facial expression and gesture control
3. **Multi-language Support**: Expand language coverage
4. **Voice Emotion**: Add emotional tone control
5. **Custom Models**: Support for user-trained models
6. **WebRTC Integration**: Direct browser support
7. **Mobile Support**: iOS/Android compatibility

## License and Credits

This implementation integrates several open-source components:
- Wav2Lip for lip-synchronization
- XTTS for text-to-speech
- PyVirtualCam for virtual camera support
- SoundDevice for audio management

Please respect the licenses of all integrated components.
