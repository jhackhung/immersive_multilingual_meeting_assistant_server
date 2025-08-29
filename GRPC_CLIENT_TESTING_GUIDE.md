# Virtual Avatar gRPC Client Testing Guide

This guide explains how to test the virtual avatar functionality using the provided gRPC clients.

## Available Test Clients

### 1. Integrated Client (`client.py`)
The main client that tests all services including virtual avatar:
```bash
python client.py
```

**Features:**
- Tests all gRPC services (TTS, Wav2Lip, Translation, etc.)
- Includes virtual avatar testing as part of comprehensive test suite
- Automatically checks prerequisites before avatar testing
- Provides interactive avatar testing option

### 2. Dedicated Avatar Client (`test_avatar_client.py`)
A specialized client focused solely on virtual avatar testing:
```bash
python test_avatar_client.py
```

**Features:**
- Focused virtual avatar testing
- Multiple test modes (basic, multilingual, stress, interactive)
- Comprehensive test suite with detailed reporting
- Interactive conversation mode

### 3. Standalone Demo (`demo_virtual_avatar.py`)
Direct service testing without gRPC:
```bash
python demo_virtual_avatar.py
```

**Features:**
- Tests virtual avatar service directly
- No server required
- Good for debugging service issues

## Test Requirements

### Prerequisites
1. **Test Files:** Ensure these files exist:
   - `wav2lip_sample/tom.jpg` - Avatar image
   - `identify_sample/ta.wav` - Voice sample

2. **Virtual Devices (Optional but Recommended):**
   - **Virtual Audio:** VB-Cable or similar
   - **Virtual Camera:** OBS Virtual Camera or v4l2loopback

3. **Server Running:** For gRPC tests, start the server first:
   ```bash
   python server.py
   ```

### Dependencies
All required packages are in `environment.yml`:
- `grpc` - gRPC communication
- `pyvirtualcam` - Virtual camera support
- `sounddevice` - Audio device management
- `opencv-python` - Video processing
- `soundfile` - Audio file handling

## Test Scenarios

### 1. Quick Functionality Test
```bash
# Start server in one terminal
python server.py

# Run basic test in another terminal
python test_avatar_client.py
# Choose option 2 (Basic functionality test)
```

### 2. Comprehensive Testing
```bash
# Start server
python server.py

# Run full test suite
python test_avatar_client.py
# Choose option 1 (Comprehensive test suite)
```

### 3. Interactive Avatar Chat
```bash
# Start server
python server.py

# Run interactive test
python test_avatar_client.py
# Choose option 5 (Interactive test)
```

### 4. All Services Test (Including Avatar)
```bash
# Start server
python server.py

# Run integrated client
python client.py
# This will test all services including virtual avatar
```

## Expected Test Results

### Successful Test Output:
```
🎭 虛擬頭像 gRPC 客戶端測試
============================================================
🔗 連接到 gRPC 服務器: localhost:50051
✅ 成功連接到服務器

🔍 檢查虛擬頭像測試先決條件...
✅ 圖片檔案: wav2lip_sample/tom.jpg (xxx bytes)
✅ 音頻檔案: identify_sample/ta.wav (xxx bytes)
✅ gRPC
✅ Protobuf
📋 先決條件檢查: ✅ 通過

🎯 初始化虛擬頭像
📷 圖片讀取成功: xxx bytes
🎵 音頻讀取成功: xxx bytes
⏳ 發送初始化請求...
✅ 頭像初始化成功: 頭像初始化成功
⏱️ 初始化時間: x.xx 秒

🗣️ 頭像說話: 'Hello, I am your virtual avatar!' (en)
⏳ 發送說話請求...
✅ 頭像說話成功: 頭像說話成功
⏱️ 處理時間: x.xx 秒
📺 請檢查虛擬攝像頭輸出
🎵 請檢查虛擬麥克風輸出
```

### Test Result Summary:
```
🏁 測試套件完成
==================================================
📋 測試結果總結:
   基本功能: ✅ 通過
   多語言: ✅ 通過
   壓力測試: ✅ 通過

🎯 總體通過率: 3/3 (100.0%)
🎉 虛擬頭像服務測試通過！
```

## Troubleshooting

### Common Issues

1. **Connection Failed**
   ```
   ❌ 連接超時
   ```
   **Solution:** Ensure the server is running (`python server.py`)

2. **Test Files Not Found**
   ```
   ❌ 圖片檔案: wav2lip_sample/tom.jpg (不存在)
   ```
   **Solution:** Ensure test files are in the correct locations

3. **gRPC Errors**
   ```
   ❌ gRPC 錯誤: UNIMPLEMENTED - 虛擬頭像服務未啟用
   ```
   **Solution:** Check if virtual avatar service is properly initialized in server

4. **Import Errors**
   ```
   ModuleNotFoundError: No module named 'xxx'
   ```
   **Solution:** Install missing dependencies or activate correct conda environment

5. **Virtual Device Issues**
   ```
   ⚠️ 未找到虛擬音頻設備
   ```
   **Solution:** Install VB-Cable or similar virtual audio driver

### Debug Mode

Enable detailed logging:
```bash
export PYTHONPATH=$PYTHONPATH:.
python -c "
import logging
logging.basicConfig(level=logging.DEBUG)
import test_avatar_client
test_avatar_client.main()
"
```

### Manual Testing

Test individual components:

1. **Test protobuf imports:**
   ```bash
   python -c "from proto import model_service_pb2; print('Protobuf OK')"
   ```

2. **Test gRPC connection:**
   ```bash
   python -c "
   import grpc
   channel = grpc.insecure_channel('localhost:50051')
   grpc.channel_ready_future(channel).result(timeout=5)
   print('gRPC Connection OK')
   "
   ```

3. **Test service import:**
   ```bash
   python -c "from apis.virtual_avatar_service import VirtualAvatarService; print('Service Import OK')"
   ```

## Performance Expectations

### Typical Performance:
- **Avatar Initialization:** 5-15 seconds (depends on model loading)
- **Speech Generation:** 2-5 seconds per sentence
- **Processing Time:** ~1-2x real-time (for short sentences)

### System Requirements:
- **RAM:** 4-8GB available
- **GPU:** Recommended for faster processing
- **CPU:** Multi-core recommended
- **Storage:** 2-3GB for models

## Integration Examples

### Video Conferencing Setup:
1. Start server and initialize avatar
2. In Zoom/Teams:
   - Camera: Select "OBS Virtual Camera"
   - Microphone: Select "CABLE Input"
3. Use client to make avatar speak during calls

### Live Streaming Setup:
1. Configure OBS:
   - Add Video Capture Device → OBS Virtual Camera
   - Add Audio Input Capture → CABLE Input
2. Start streaming while controlling avatar via client

### Custom Application Integration:
```python
import grpc
from proto import model_service_pb2, model_service_pb2_grpc

# Connect to avatar service
channel = grpc.insecure_channel('localhost:50051')
stub = model_service_pb2_grpc.MediaServiceStub(channel)

# Initialize avatar
with open("avatar.jpg", "rb") as f:
    image_data = f.read()
with open("voice.wav", "rb") as f:
    audio_data = f.read()

init_request = model_service_pb2.InitAvatarRequest(
    image_data=image_data,
    sample_audio_data=audio_data
)
response = stub.InitAvatar(init_request)

# Make avatar speak
speak_request = model_service_pb2.AvatarSpeakRequest(
    text="Hello from my application!",
    language="en"
)
response = stub.AvatarSpeak(speak_request)
```

## Next Steps

1. **Production Deployment:**
   - Use proper authentication (TLS)
   - Add error handling and retries
   - Implement connection pooling

2. **Performance Optimization:**
   - Use GPU acceleration
   - Implement caching
   - Optimize batch sizes

3. **Feature Extensions:**
   - Add emotion control
   - Support multiple avatars
   - Implement real-time streaming
