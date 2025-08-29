# Immersive Multilingual Meeting Assistant Server

一個基於 gRPC 的多語言會議助手後端服務，提供即時翻譯、語音合成 (TTS)、說話者分離及虛擬頭像生成等功能。

##  功能特色

- **多語言翻譯**: 使用 mBART-50 模型，支援 50+ 種語言互譯
- **語音合成 (TTS)**: 基於 XTTS-v2 的多語言語音合成，支援聲音複製
- **語音識別 (STT)**: 使用 OpenAI Whisper 模型進行高準確度語音轉文字，支援 11+ 種語言。
- **說話者分離**: 使用 pyannote.audio 進行說話者日誌分析
- **高性能 gRPC 服務**: 支援併發請求處理
- **虛擬頭像服務**: 根據圖片與聲音樣本建立虛擬頭像，並將其即時串流至虛擬攝影機與麥克風，支援視訊會議與直播。
- **GPU/NPU 加速**: 自動偵測並使用 CUDA、DirectML 或 QNN (如果可用) 來加速模型運算。

##  專案架構

```
immersive_multilingual_meeting_assistant_server/
├── server.py                    # 主要 gRPC 伺服器
├── client.py                    # 測試客戶端
├── model_service.proto          # gRPC 服務定義
├── environment.yml              # Conda 環境配置
├── apis/                       # API 服務實作
│   ├── tts_service.py          # TTS 服務
│   ├── speech_recognition_service.py # 語音識別服務
│   ├── pyannote.py             # 說話者分離服務
│   └── tts_sample/             # TTS 範例音檔
├── models/                     # 模型實作
│   ├── mbart_translator_model.py # mBART 翻譯模型
│   └── hifigan.onnx            # HiFi-GAN 聲碼器 (ONNX)
└── proto/                      # 自動生成的 gRPC 檔案
    ├── model_service_pb2.py
    ├── model_service_pb2_grpc.py
    └── model_service_pb2.pyi
```

##  快速開始

### 環境需求

- Python 3.10+

- Conda 或 Miniconda

- CUDA 11.8+ (可選，用於 NVIDIA GPU 加速)

- 16GB+ RAM (建議)

### 1. 環境設置

```bash
# 使用 Conda 建立環境
conda env create -f environment.yml
conda activate server
```

### 2. 環境變數配置

建立 `.env` 檔案：

```bash
# Hugging Face Token (用於 pyannote.audio 等)
HUGGINGFACE_TOKEN=your_huggingface_token_here
```

>  **取得 Hugging Face Token**: 前往 [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) 建立 token

### 3. 生成 gRPC 程式碼

```bash
python -m grpc_tools.protoc --proto_path=. --python_out=./proto --grpc_python_out=./proto model_service.proto
```

### 4. 啟動服務

```bash
# 啟動主要翻譯服務
python server.py

# 測試連線 (另開終端)
python client.py
```

