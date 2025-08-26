import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import os
import sys

# =================== 直接定義卷積層 ===================
class Conv2d(nn.Module):
    def __init__(self, cin, cout, kernel_size, stride, padding, residual=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conv_block = nn.Sequential(
                            nn.Conv2d(cin, cout, kernel_size, stride, padding),
                            nn.BatchNorm2d(cout)
                            )
        self.act = nn.ReLU()
        self.residual = residual

    def forward(self, x):
        out = self.conv_block(x)
        if self.residual:
            out += x
        return self.act(out)

class nonorm_Conv2d(nn.Module):
    def __init__(self, cin, cout, kernel_size, stride, padding, residual=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conv_block = nn.Sequential(
                            nn.Conv2d(cin, cout, kernel_size, stride, padding),
                            )
        self.act = nn.LeakyReLU(0.01, inplace=True)

    def forward(self, x):
        out = self.conv_block(x)
        return self.act(out)

class Conv2dTranspose(nn.Module):
    def __init__(self, cin, cout, kernel_size, stride, padding, output_padding=0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conv_block = nn.Sequential(
                            nn.ConvTranspose2d(cin, cout, kernel_size, stride, padding, output_padding),
                            nn.BatchNorm2d(cout)
                            )
        self.act = nn.ReLU()

    def forward(self, x):
        out = self.conv_block(x)
        return self.act(out)

# =================== 直接定義 Wav2Lip 模型 ===================
class Wav2Lip(nn.Module):
    def __init__(self):
        super(Wav2Lip, self).__init__()

        self.face_encoder_blocks = nn.ModuleList([
            nn.Sequential(Conv2d(6, 16, kernel_size=7, stride=1, padding=3)), # 96,96

            nn.Sequential(Conv2d(16, 32, kernel_size=3, stride=2, padding=1), # 48,48
            Conv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True)),

            nn.Sequential(Conv2d(32, 64, kernel_size=3, stride=2, padding=1),    # 24,24
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True)),

            nn.Sequential(Conv2d(64, 128, kernel_size=3, stride=2, padding=1),   # 12,12
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True)),

            nn.Sequential(Conv2d(128, 256, kernel_size=3, stride=2, padding=1),       # 6,6
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True)),

            nn.Sequential(Conv2d(256, 512, kernel_size=3, stride=2, padding=1),     # 3,3
            Conv2d(512, 512, kernel_size=3, stride=1, padding=1, residual=True),),
            
            nn.Sequential(Conv2d(512, 512, kernel_size=3, stride=1, padding=0),     # 1, 1
            Conv2d(512, 512, kernel_size=1, stride=1, padding=0)),])

        self.audio_encoder = nn.Sequential(
            Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            Conv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(32, 64, kernel_size=3, stride=(3, 1), padding=1),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(64, 128, kernel_size=3, stride=3, padding=1),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(128, 256, kernel_size=3, stride=(3, 2), padding=1),
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(256, 512, kernel_size=3, stride=1, padding=0),
            Conv2d(512, 512, kernel_size=1, stride=1, padding=0),)

        self.face_decoder_blocks = nn.ModuleList([
            nn.Sequential(Conv2d(512, 512, kernel_size=1, stride=1, padding=0),),

            nn.Sequential(Conv2dTranspose(1024, 512, kernel_size=3, stride=1, padding=0), # 3,3
            Conv2d(512, 512, kernel_size=3, stride=1, padding=1, residual=True),),

            nn.Sequential(Conv2dTranspose(1024, 512, kernel_size=3, stride=2, padding=1, output_padding=1),
            Conv2d(512, 512, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(512, 512, kernel_size=3, stride=1, padding=1, residual=True),), # 6, 6

            nn.Sequential(Conv2dTranspose(768, 384, kernel_size=3, stride=2, padding=1, output_padding=1),
            Conv2d(384, 384, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(384, 384, kernel_size=3, stride=1, padding=1, residual=True),), # 12, 12

            nn.Sequential(Conv2dTranspose(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1),
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),), # 24, 24

            nn.Sequential(Conv2dTranspose(320, 128, kernel_size=3, stride=2, padding=1, output_padding=1), 
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),), # 48, 48

            nn.Sequential(Conv2dTranspose(160, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),),]) # 96,96

        self.output_block = nn.Sequential(Conv2d(80, 32, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(32, 3, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid()) 

    def forward(self, audio_sequences, face_sequences):
        # audio_sequences = (B, T, 1, 80, 16)
        B = audio_sequences.size(0)

        input_dim_size = len(face_sequences.size())
        if input_dim_size > 4:
            audio_sequences = torch.cat([audio_sequences[:, i] for i in range(audio_sequences.size(1))], dim=0)
            face_sequences = torch.cat([face_sequences[:, :, i] for i in range(face_sequences.size(2))], dim=0)

        audio_embedding = self.audio_encoder(audio_sequences) # B, 512, 1, 1

        feats = []
        x = face_sequences
        for f in self.face_encoder_blocks:
            x = f(x)
            feats.append(x)

        x = audio_embedding
        for f in self.face_decoder_blocks:
            x = f(x)
            try:
                x = torch.cat((x, feats[-1]), dim=1)
            except Exception as e:
                print(x.size())
                print(feats[-1].size())
                raise e
            
            feats.pop()

        x = self.output_block(x)

        if input_dim_size > 4:
            x = torch.split(x, B, dim=0) # [(B, C, H, W)]
            outputs = torch.stack(x, dim=2) # (B, C, T, H, W)

        else:
            outputs = x
            
        return outputs

# 將 Wav2Lip 設置為主要模型
Wav2LipModel = Wav2Lip

# 導入其他模組
try:
    import face_detection
    print("✅ 人臉檢測模組導入成功")
except ImportError:
    print("⚠️ 人臉檢測模組導入失敗，將使用備用方案")
    face_detection = None

# 直接使用 librosa 進行音訊處理，不再嘗試導入自定義的 'audio' 模組
import librosa
print("✅ 音訊處理將使用 librosa")

print("✅ Wav2Lip 模組載入成功 (自包含版本)")

class Wav2LipPytorch:
    """使用 PyTorch 版本的 Wav2Lip 模型"""
    
    def __init__(self, checkpoint_path="models/wav2lip_gan.pth"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.face_detector = None
        self.checkpoint_path = checkpoint_path
        self.img_size = 96
        
        # 預設參數
        self.pads = [0, 10, 0, 0]  # top, bottom, left, right padding
        self.resize_factor = 1
        self.face_det_batch_size = 16
        self.wav2lip_batch_size = 128
        
        print(f"🚀 初始化 Wav2Lip PyTorch 模型，設備: {self.device}")
        
    def load_model(self):
        """載入 Wav2Lip 模型"""
        if self.model is not None:
            return
            
        print(f"⏳ 載入 Wav2Lip 模型從: {self.checkpoint_path}")
        
        if not os.path.exists(self.checkpoint_path):
            raise FileNotFoundError(f"❌ 找不到模型檔案: {self.checkpoint_path}")
        
        # 載入模型
        self.model = Wav2LipModel()
        
        # 載入權重
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        
        # 處理不同的 checkpoint 格式
        if "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        elif "model" in checkpoint:
            state_dict = checkpoint["model"]
        else:
            state_dict = checkpoint
            
        # 載入權重
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()
        
        print("✅ Wav2Lip PyTorch 模型載入成功")
        
    def load_face_detector(self):
        """載入人臉檢測器"""
        if self.face_detector is not None:
            return
            
        print("⏳ 載入人臉檢測器...")
        try:
            if face_detection is not None:
                self.face_detector = face_detection.FaceAlignment(
                    face_detection.LandmarksType._2D, 
                    flip_input=False, 
                    device=self.device
                )
                print("✅ 人臉檢測器載入成功")
            else:
                print("⚠️ 人臉檢測模組不可用，將使用 OpenCV 作為備用方案")
                import cv2
                cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
                self.face_detector = cv2.CascadeClassifier(cascade_path)
                print("✅ OpenCV 人臉檢測器載入成功")
        except Exception as e:
            print(f"❌ 人臉檢測器載入失敗: {e}")
            print("將使用 OpenCV 作為備用方案")
            import cv2
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            self.face_detector = cv2.CascadeClassifier(cascade_path)
            
    def detect_faces(self, images):
        """檢測人臉"""
        if self.face_detector is None:
            self.load_face_detector()
            
        # 檢查是否使用 face_detection 模組還是 OpenCV
        if hasattr(self.face_detector, 'get_detections_for_batch'):
            # 使用 face_detection 模組
            return self._detect_faces_with_face_detection(images)
        else:
            # 使用 OpenCV
            return self._detect_faces_with_opencv(images)
    
    def _detect_faces_with_face_detection(self, images):
        """使用 face_detection 模組檢測人臉"""
        batch_size = self.face_det_batch_size
        
        while True:
            predictions = []
            try:
                for i in range(0, len(images), batch_size):
                    batch = np.array(images[i:i + batch_size])
                    predictions.extend(self.face_detector.get_detections_for_batch(batch))
                break
            except RuntimeError as e:
                if batch_size == 1:
                    raise RuntimeError('圖片太大無法進行人臉檢測。請使用 --resize_factor 參數')
                batch_size //= 2
                print(f'從 OOM 錯誤恢復；新批次大小: {batch_size}')
                continue
                
        # 處理檢測結果
        results = []
        pady1, pady2, padx1, padx2 = self.pads
        
        for rect, image in zip(predictions, images):
            if rect is None:
                # 如果沒有檢測到人臉，使用整個圖片
                print("⚠️ 未檢測到人臉，使用整個圖片")
                results.append([0, 0, image.shape[1], image.shape[0]])
                continue
                
            # 確保座標是整數
            y1 = max(0, int(rect[1] - pady1))
            y2 = min(image.shape[0], int(rect[3] + pady2))
            x1 = max(0, int(rect[0] - padx1))
            x2 = min(image.shape[1], int(rect[2] + padx2))
            
            results.append([x1, y1, x2, y2])
            
        return self.get_smoothened_boxes(results)
    
    def _detect_faces_with_opencv(self, images):
        """使用 OpenCV 檢測人臉"""
        results = []
        pady1, pady2, padx1, padx2 = self.pads
        
        for image in images:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            faces = self.face_detector.detectMultiScale(gray, 1.1, 4)
            
            if len(faces) > 0:
                x, y, w, h = faces[0]
                # 確保座標是整數
                y1 = max(0, int(y + pady1))
                y2 = min(image.shape[0], int(y + h + pady2))
                x1 = max(0, int(x + padx1))
                x2 = min(image.shape[1], int(x + w + padx2))
                results.append([x1, y1, x2, y2])
            else:
                print("⚠️ 未檢測到人臉，使用整個圖片")
                results.append([0, 0, image.shape[1], image.shape[0]])
                
        return self.get_smoothened_boxes(results)
        
    def get_smoothened_boxes(self, boxes, T=5):
        """平滑化邊界框"""
        for i in range(len(boxes)):
            if i + T > len(boxes):
                window = boxes[len(boxes) - T:]
            else:
                window = boxes[i : i + T]
            boxes[i] = np.mean(window, axis=0)
        # 確保所有座標都是整數
        for i in range(len(boxes)):
            boxes[i] = [int(x) for x in boxes[i]]
        return boxes
        
    def datagen(self, frames, mels, face_coords):
        """生成模型輸入數據"""
        img_batch, mel_batch, frame_batch, coords_batch = [], [], [], []
        
        for i, (frame, mel, coord) in enumerate(zip(frames, mels, face_coords)):
            # 裁剪人臉區域 - 確保座標是整數
            x1, y1, x2, y2 = [int(c) for c in coord]
            face = frame[y1:y2, x1:x2]
            face = cv2.resize(face, (self.img_size, self.img_size))
            
            img_batch.append(face)
            mel_batch.append(mel)
            frame_batch.append(frame)
            coords_batch.append(coord)
            
            if len(img_batch) >= self.wav2lip_batch_size:
                img_batch = np.asarray(img_batch)
                mel_batch = np.asarray(mel_batch)
                
                # 轉換為模型輸入格式
                img_masked = img_batch.copy()
                img_masked[:, self.img_size//2:] = 0
                
                img_batch = np.concatenate((img_masked, img_batch), axis=3) / 255.
                mel_batch = np.reshape(mel_batch, [len(mel_batch), mel_batch.shape[1], mel_batch.shape[2], 1])
                
                yield img_batch, mel_batch, frame_batch, coords_batch
                img_batch, mel_batch, frame_batch, coords_batch = [], [], [], []
                
        if len(img_batch) > 0:
            img_batch = np.asarray(img_batch)
            mel_batch = np.asarray(mel_batch)
            
            img_masked = img_batch.copy()
            img_masked[:, self.img_size//2:] = 0
            
            img_batch = np.concatenate((img_masked, img_batch), axis=3) / 255.
            mel_batch = np.reshape(mel_batch, [len(mel_batch), mel_batch.shape[1], mel_batch.shape[2], 1])
            
            yield img_batch, mel_batch, frame_batch, coords_batch
            
    def inference(self, image_path, audio_path, output_path):
        """執行 Wav2Lip 推理"""
        if self.model is None:
            self.load_model()
            
        print("📷 讀取圖片...")
        
        # 讀取圖片
        if image_path.lower().endswith(('.png', '.jpg', '.jpeg')):
            frame = cv2.imread(image_path)
            if frame is None:
                raise ValueError(f"無法讀取圖片: {image_path}")
            frames = [frame]
        else:
            raise ValueError("目前只支援靜態圖片")
            
        print("🎵 處理音訊...")
        
        # 處理音訊 - 直接使用 librosa
        try:
            wav, sr = librosa.load(audio_path, sr=16000)
            mel = librosa.feature.melspectrogram(y=wav, sr=16000, n_mels=80, fmax=8000)
            mel = np.log(mel + 1e-6)
            print(f"✅ 使用 librosa 音訊處理成功，mel 形狀: {mel.shape}")
            
            if np.isnan(mel.reshape(-1)).sum() > 0:
                raise ValueError('音訊檔案中包含 NaN 值')
                
        except Exception as e:
            print(f"❌ 音訊處理失敗: {e}")
            raise # Re-raise the exception if librosa also fails
            
        mel_chunks = []
        mel_idx_multiplier = 80./25. 
        i = 0
        while True:
            start_idx = int(i * mel_idx_multiplier)
            if start_idx + 16 < len(mel[0]):
                mel_chunks.append(mel[:, start_idx : start_idx + 16])
            else:
                break
            i += 1
            
        print(f"📊 音訊長度: {len(mel_chunks)} 幀")
        
        # 擴展幀以匹配音訊長度
        full_frames = frames * len(mel_chunks)
        
        print("🎯 檢測人臉...")
        
        # 檢測人臉
        try:
            face_coords = self.detect_faces(full_frames)
        except Exception as e:
            print(f"⚠️ 人臉檢測失敗: {e}")
            print("使用整個圖片作為人臉區域")
            # 如果人臉檢測失敗，使用整個圖片
            h, w = frames[0].shape[:2]
            face_coords = [[0, 0, w, h]] * len(full_frames)
        
        print("🤖 執行 Wav2Lip 推理...")
        
        # 創建影片寫入器
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, 25, (frames[0].shape[1], frames[0].shape[0]))
        
        # 執行推理
        for img_batch, mel_batch, frame_batch, coords_batch in self.datagen(full_frames, mel_chunks, face_coords):
            with torch.no_grad():
                img_batch = torch.FloatTensor(np.transpose(img_batch, (0, 3, 1, 2))).to(self.device)
                mel_batch = torch.FloatTensor(np.transpose(mel_batch, (0, 3, 1, 2))).to(self.device)
                
                pred = self.model(mel_batch, img_batch)
                pred = pred.cpu().numpy().transpose(0, 2, 3, 1) * 255.
                
                for p, f, coord in zip(pred, frame_batch, coords_batch):
                    # 確保座標是整數
                    x1, y1, x2, y2 = [int(c) for c in coord]
                    p = cv2.resize(p.astype(np.uint8), (x2 - x1, y2 - y1))
                    
                    f[y1:y2, x1:x2] = p
                    out.write(f)
                    
        out.release()
        print(f"✅ 影片生成完成: {output_path}")
        
        return output_path
