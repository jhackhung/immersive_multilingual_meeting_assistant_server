# 語音識別服務更新文檔

## 更新概述

已將語音識別服務從基於 Whisper 的實現更新為基於 Python `SpeechRecognition` 庫的實現，無需身份驗證即可使用。

## 主要變更

### 1. 依賴項更改

**移除的依賴：**
- `torch` (PyTorch)
- `transformers` (Hugging Face Transformers)
- Whisper 模型相關依賴

**新增的依賴：**
- `SpeechRecognition` - Python 語音識別庫
- `pyaudio` - 音頻輸入支持（可選）

### 2. 識別引擎

**主要引擎：**
- Google Speech Recognition API (免費，無需認證)
  - 支持多種語言
  - 在線服務，需要網絡連接

**備用引擎：**
- CMU Sphinx (離線，本地處理)
  - 主要支持英語
  - 當 Google API 不可用時自動切換

### 3. 支持的語言

- 中文 (zh)
- 英語 (en)
- 日語 (ja)
- 韓語 (ko)
- 西班牙語 (es)
- 法語 (fr)
- 德語 (de)
- 俄語 (ru)
- 葡萄牙語 (pt)
- 義大利語 (it)
- 阿拉伯語 (ar)
- 印地語 (hi)
- 泰語 (th)
- 越南語 (vi)
- 自動檢測 (auto)

### 4. 功能特性

**保持的功能：**
- gRPC 接口兼容
- 多種音頻格式支持（通過 librosa 轉換）
- 時間戳支持（基本實現）
- 語言檢測

**新增特性：**
- 無需身份驗證
- 自動引擎切換（在線/離線）
- 更輕量級的實現
- 更快的啟動時間

**限制：**
- 在線服務需要網絡連接
- Google API 有使用限制
- 離線引擎功能較為基礎

## 安裝說明

```bash
# 激活 conda 環境
conda activate server

# 安裝必要的包
pip install SpeechRecognition

# 可選：安裝 pyaudio 以支持麥克風輸入
pip install pyaudio
```

## 使用示例

```python
from apis.speech_recognition_service import SpeechRecognitionServicer

# 初始化服務
service = SpeechRecognitionServicer()
service.initialize()

# 轉錄音頻
with open("audio.wav", "rb") as f:
    audio_data = f.read()

result = service.transcribe_audio(
    audio_data=audio_data,
    language="zh",  # 或 "auto" 自動檢測
    return_timestamps=True
)

if result["success"]:
    print(f"轉錄結果: {result['transcribed_text']}")
    print(f"檢測語言: {result['detected_language']}")
else:
    print(f"錯誤: {result['error']}")
```

## API 兼容性

gRPC 接口保持不變，現有客戶端代碼無需修改：

```python
# 客戶端代碼保持相同
request = model_service_pb2.SpeechRecognitionRequest(
    audio_data=audio_bytes,
    language="zh",
    return_timestamps=True
)

response = stub.SpeechRecognition(request)
```

## 性能考慮

**優勢：**
- 更快的服務啟動時間
- 更低的內存使用
- 無需 GPU 支持
- 無需下載大型模型

**權衡：**
- 在線服務依賴網絡
- 可能受到 API 使用限制
- 離線模式功能有限

## 故障排除

### 常見問題

1. **網絡連接問題**
   - 服務會自動切換到離線模式
   - 檢查網絡連接

2. **音頻格式不支持**
   - 使用 librosa 自動轉換
   - 支持 WAV, MP3, MP4, OGG 等格式

3. **語音識別失敗**
   - 檢查音頻質量
   - 嘗試不同語言設置
   - 確保音頻包含清晰的語音

### 日誌級別

設置 `logging.DEBUG` 可以獲得更詳細的調試信息：

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 測試

運行測試腳本驗證功能：

```bash
python test_speech_recognition_new.py
```

## 未來改進

- 支持更多語音識別引擎
- 改進時間戳精度
- 添加信心度評估
- 支持實時流式識別
