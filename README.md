# Immersive Multilingual Meeting Assistant Server

A gRPC-based multilingual meeting assistant backend service providing real-time translation, text-to-speech (TTS), speech-to-text (STT), speaker diarization, and virtual avatar generation. Optimized for Snapdragon-powered devices with QNN (Qualcomm Neural Network) acceleration support.

## Application Description

This project is a comprehensive multilingual meeting assistant solution providing the following core functionalities:

###  Core Features

- **Real-time Multilingual Translation**: Using mBART-50 model, supporting 50+ language pairs
- **High-Quality Speech Synthesis**: XTTS-v2 based multilingual TTS with voice cloning capabilities
- **Accurate Speech Recognition**: OpenAI Whisper model supporting 11+ languages
- **Intelligent Speaker Diarization**: pyannote.audio-based speaker identification and analysis
- **Virtual Avatar System**: Image-driven virtual avatar generation with real-time streaming
- **Hardware Acceleration**: Optimized for Snapdragon platforms with QNN, DirectML support

### Use Case
- International conference real-time translation
- Multilingual video conferencing
- Online education and training
- Cross-language streaming and content creation
- Business communication and negotiation

## Team Members
| Name | Email |
|------|-------|
| 洪永結 | j112403537@g.ncu.edu.tw |
| 游晉毅 | 112403522@cc.ncu.edu.tw |
| 楊宗賀 | 112403516@cc.ncu.edu.tw |
| 陳紀睿 | 113403054@cc.ncu.edu.tw |
| 李宥寬 | metasausage@g.ncu.edu.tw |

## Documentation

### Quick References
- **[API Reference](./documents/api-reference.md)** - Complete API documentation and usage examples
- **[Protocol Buffers Reference](./documents/proto-reference.md)** - gRPC service definitions and message formats
- **[gRPC Client Testing Guide](./documents/GRPC_CLIENT_TESTING_GUIDE.md)** - Comprehensive testing procedures for all services

### Implementation Guides
- **[Implementation Summary](./documents/IMPLEMENTATION_SUMMARY.md)** - Overview of system architecture and components
- **[Speech Recognition Implementation](./documents/SPEECH_RECOGNITION_IMPLEMENTATION.md)** - Detailed STT service implementation
- **[Virtual Avatar Guide](./documents/VIRTUAL_AVATAR_GUIDE.md)** - Avatar system setup and usage

## Installation Guide

### Building and Running the C# UI

```bash
# Navigate to the UI directory
cd ui

# Restore NuGet packages
dotnet restore

# Build the application
dotnet build

# Run the application
dotnet run
```

### System Requirements

#### Minimum Requirements
- **Operating System**: Windows 10/11
- **Processor**: Intel i5-8th gen+ or AMD Ryzen 5 3600+ or Snapdragon 8cx+
- **Memory**: 16GB RAM (32GB recommended)
- **Storage**: 50GB available space
- **Python**: 3.10 or newer
- **Network**: Stable internet connection (for model downloads)

#### Recommended Configuration (Snapdragon Platform)
- **Processor**: Snapdragon X elite or newer
- **Memory**: 32GB RAM
- **Storage**: NVMe SSD 100GB+
- **QNN SDK**: Qualcomm AI Engine Direct SDK 2.10+

### Step 1: Environment Setup

#### 1.1 Install Conda

```bash
# Download and install Miniconda
# https://docs.conda.io/en/latest/miniconda.html
```

#### 1.2 Git Installation and Project Cloning

```bash
# Install Git (if not already installed)
# Windows: https://git-scm.com/download/win

# Clone the project
git clone https://github.com/jhackhung/immersive_multilingual_meeting_assistant_server.git
cd immersive_multilingual_meeting_assistant_server
```

### Step 2: Dependencies Installation

#### 2.1 Create Conda Environment

```bash
# Create dedicated environment
conda env create -f environment.yml

# Activate environment
conda activate server

# Verify Python version
python --version  # Should display Python 3.10+
```

### Step 3: Environment Variables Configuration

Create `.env` file:

```bash
# Hugging Face Token (用於 pyannote.audio 等)
HUGGINGFACE_TOKEN=your_huggingface_token_here
```

>  **Get Hugging Face Token**: Go [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) to create token

### Step 4: Generate gRPC Code

```bash
# Generate Protocol Buffers code
python -m grpc_tools.protoc --proto_path=. --python_out=./proto --grpc_python_out=./proto model_service.proto
```

### Step 5. Starting the Service

```bash
# Start main service
python server.py

# Test connection (with another terminal)
python client.py
```

## Testing and Development

### Quick Start Testing

After starting the server, you can test individual services:

```bash
# Test all services comprehensively
python client.py

# Test speech recognition service
python test_stt_client.py

# Test text-to-speech service
python test_tts_service.py

# Test virtual avatar service
python test_avatar_client.py
```

For detailed testing procedures and troubleshooting, see the **[gRPC Client Testing Guide](./documents/GRPC_CLIENT_TESTING_GUIDE.md)**.

### Service Implementation Details

- **Speech Recognition**: Refer to [Speech Recognition Implementation](./documents/SPEECH_RECOGNITION_IMPLEMENTATION.md) for STT service details
- **Virtual Avatar**: See [Virtual Avatar Guide](./documents/VIRTUAL_AVATAR_GUIDE.md) for avatar setup and configuration
- **System Architecture**: Check [Implementation Summary](./documents/IMPLEMENTATION_SUMMARY.md) for overall system design

## API Usage

The server exposes multiple gRPC services. For complete API documentation including request/response formats, error codes, and usage examples, refer to:

- **[API Reference](./documents/api-reference.md)** - Service methods and parameters
- **[Protocol Buffers Reference](./documents/proto-reference.md)** - Message definitions and data structures

## License

This project is licensed under the **MIT License**.

### MIT License

```
Copyright (c) 2025 Jirui的甜蜜小窩

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```