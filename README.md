# 🏛️ Monumathic - Monument Detection Pipeline

A complete deep learning pipeline for Indian monument image classification using PyTorch and ResNet18.

## 📋 Overview

Monumathic provides an end-to-end solution for monument detection:

| Script | Purpose |
|--------|---------|
| `1_fetch_images.py` | Download monument images from APIs |
| `2_augment_data.py` | Augment images for robust training |
| `3_train_model.py` | Train ResNet18 classifier |
| `4_circle_image.py` | Visualize detections |
| `5_predict.py` | Production inference |

## 🚀 Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run the complete pipeline
python 1_fetch_images.py --create-dirs
python 2_augment_data.py --mode basic
python 3_train_model.py --epochs 50
python 5_predict.py --interactive
```

## 📁 Directory Structure

```
Monumathic/
├── 1_fetch_images.py      # Image downloader
├── 2_augment_data.py      # Data augmentation
├── 3_train_model.py       # Model training
├── 4_circle_image.py      # Detection visualization
├── 5_predict.py           # Inference engine
├── requirements.txt       # Dependencies
├── README.md              # This file
│
├── dataset/               # Raw images (created by script 1)
│   ├── Taj_Mahal/
│   ├── Red_Fort_Delhi/
│   └── ...
│
├── dataset_augmented/     # Augmented images (created by script 2)
│   └── ...
│
├── models/                # Trained models (created by script 3)
│   ├── fast_monument_cnn.pth
│   ├── class_names.json
│   └── model_metadata.json
│
├── circled_output/        # Visualized detections (created by script 4)
│   └── ...
│
└── training_logs/         # Training history (created by script 3)
    └── ...
```

## 📖 Detailed Usage

### 1️⃣ Fetch Images

Downloads monument images from Unsplash, Pixabay, and Pexels APIs.

```bash
# Create directory structure only
python 1_fetch_images.py --create-dirs

# Fetch images for all monuments
python 1_fetch_images.py

# Fetch images for specific monument
python 1_fetch_images.py --monument "Taj Mahal"

# Show manual collection instructions
python 1_fetch_images.py --manual
```

**Environment Variables (Optional):**
```bash
export UNSPLASH_API_KEY="your_key"
export PIXABAY_API_KEY="your_key"
export PEXELS_API_KEY="your_key"
```

**Supported Monuments:**
- Taj Mahal, Red Fort Delhi, Qutub Minar, India Gate
- Gateway of India, Hawa Mahal, Mysore Palace, Charminar
- Victoria Memorial, Golden Temple, Lotus Temple
- Ajanta Caves, Ellora Caves, Konark Sun Temple
- And 16 more...

### 2️⃣ Augment Data

Applies various transformations to increase dataset size and variety.

```bash
# Basic augmentation (faster)
python 2_augment_data.py --mode basic

# Full augmentation (more variations)
python 2_augment_data.py --mode full

# Augment specific monument
python 2_augment_data.py --monument Taj_Mahal

# Verify augmented dataset
python 2_augment_data.py --verify
```

**Augmentation Techniques:**
| Technique | Description |
|-----------|-------------|
| Rotation | 90°, 180°, 270° |
| Flip | Horizontal & Vertical |
| Brightness | 0.7x - 1.3x |
| Contrast | 0.7x - 1.3x |
| Saturation | 0.5x - 1.5x |
| Sharpness | 0.5x - 2.0x |
| Blur | Gaussian (radius 1-2) |
| Crop | Center & Random crops |
| Noise | Gaussian noise |
| Grayscale | B&W conversion |
| Sepia | Vintage tone |

### 3️⃣ Train Model

Trains a ResNet18-based classifier with transfer learning.

```bash
# Default training
python 3_train_model.py

# Custom parameters
python 3_train_model.py --epochs 100 --batch-size 64 --lr 0.0001

# Custom dataset path
python 3_train_model.py --dataset path/to/dataset
```

**Training Features:**
- ✅ Transfer learning from ImageNet
- ✅ Learning rate scheduling (ReduceLROnPlateau)
- ✅ Early stopping (patience=10)
- ✅ Data augmentation during training
- ✅ Train/Val/Test split (80/10/10)
- ✅ Per-class accuracy metrics
- ✅ Automatic model checkpointing

**Output:**
- `models/fast_monument_cnn.pth` - Best model weights
- `models/class_names.json` - Class labels
- `models/model_metadata.json` - Training config
- `training_logs/training_history_*.json` - Loss/accuracy curves

### 4️⃣ Circle Image (Visualization)

Draws circles, bounding boxes, and annotations around detected monuments.

```bash
# Visualize single image (full style)
python 4_circle_image.py --image test.jpg --style full

# Different visualization styles
python 4_circle_image.py --image test.jpg --style simple
python 4_circle_image.py --image test.jpg --style circle
python 4_circle_image.py --image test.jpg --style minimal

# Batch visualization
python 4_circle_image.py --folder path/to/images/
```

**Visualization Styles:**
| Style | Description |
|-------|-------------|
| `full` | Box + label + confidence bar + top-5 sidebar |
| `simple` | Bounding box with label and confidence |
| `circle` | Circle highlight with glow effect |
| `minimal` | Bounding box only |

### 5️⃣ Predict

Production-ready inference for monument classification.

```bash
# Predict from file
python 5_predict.py --image path/to/image.jpg

# Predict from URL
python 5_predict.py --url https://example.com/monument.jpg

# Interactive mode
python 5_predict.py --interactive

# Benchmark on test folder
python 5_predict.py --benchmark path/to/test/folder

# Custom model
python 5_predict.py --image test.jpg --model custom.pth --classes classes.json
```

**API Integration:**
```python
from Monumathic import predict_monument, MonumentPredictor

# Simple prediction
result = predict_monument(image_bytes)
print(result['monument'])      # "Taj Mahal"
print(result['confidence'])    # 0.95
print(result['description'])   # "The Taj Mahal is..."

# Advanced usage
predictor = MonumentPredictor("model.pth", "classes.json")
result = predictor.predict(image_path, top_k=5)
print(result.monument_name)
print(result.top_predictions)
```

## 🔧 Configuration

### Image Size
All scripts use `IMG_SIZE = 128` for consistency. Modify in each script if needed.

### Model Architecture
```
ResNet18 (pretrained=True)
└── fc: Sequential
    ├── Dropout(0.5)
    ├── Linear(512, 512)
    ├── ReLU
    ├── Dropout(0.3)
    └── Linear(512, num_classes)
```

### Training Defaults
| Parameter | Default |
|-----------|---------|
| Batch Size | 32 |
| Epochs | 50 |
| Learning Rate | 0.001 |
| Train Split | 80% |
| Val Split | 10% |
| Test Split | 10% |
| Early Stopping | 10 epochs |

## 📊 Expected Results

With a well-prepared dataset (~100+ images per class):

| Metric | Expected |
|--------|----------|
| Training Accuracy | 95-99% |
| Validation Accuracy | 85-95% |
| Test Accuracy | 80-90% |
| Inference Time | 10-50ms/image |

## 🐛 Troubleshooting

### No images downloaded
- Check API keys are set correctly
- Use `--manual` flag for manual collection instructions

### Model not found
- Ensure `3_train_model.py` completed successfully
- Check `models/` directory for `fast_monument_cnn.pth`

### Low accuracy
- Increase dataset size (100+ images per class recommended)
- Use `--mode full` for augmentation
- Increase training epochs
- Check for class imbalance

### CUDA out of memory
- Reduce batch size: `--batch-size 16`
- Reduce image size in script configuration

## 📄 License

MIT License - See main project LICENSE file.

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/improvement`)
3. Commit changes (`git commit -am 'Add improvement'`)
4. Push to branch (`git push origin feature/improvement`)
5. Open Pull Request

---

**Part of [Trend Tripper](https://github.com/your-repo/trend-tripper) - Your AI-Powered Travel Companion**
