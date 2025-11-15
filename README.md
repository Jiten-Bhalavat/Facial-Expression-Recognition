# Facial Expression Recognition (FER)

## 📋 Problem Statement

Human emotions are critical indicators in various fields including mental health, customer service, security, and human-computer interaction. However, manually analyzing facial expressions is time-consuming, subjective, and prone to human error. 

**Challenge:** Develop an automated system that can accurately detect and classify human emotions from facial images and videos in real-time.

**Solution:** This project implements a deep learning-based Facial Expression Recognition system that uses Convolutional Neural Networks (CNNs) to detect faces and classify emotions into 7 categories: angry, disgust, fear, happy, sad, surprise, and neutral.

---

## 🎯 Results

### Image Detection
![Sample Detection](Jiten-Bhalavat_drawn.jpg)

**Input:** Face image  
**Output:** Detected emotions with confidence scores and bounding box

### Key Metrics
- **Emotions Detected:** 7 categories (angry, disgust, fear, happy, sad, surprise, neutral)
- **Face Detection:** OpenCV Haar Cascade / MTCNN
- **Model Accuracy:** Pre-trained on FER2013 dataset
- **Real-time Processing:** ✅ Webcam support
- **Video Analysis:** ✅ Frame-by-frame emotion tracking

---

## 🚀 How to Use

### Prerequisites
- Python 3.8+
- Windows/Linux/Mac
- Webcam (optional, for real-time detection)

### Installation

1. **Clone/Download the project**
```bash
git clone https://github.com/yourusername/fer.git
cd fer
```

2. **Create Virtual Environment**
```bash
python -m venv venv_fer
```

3. **Activate Virtual Environment**

**Windows:**
```bash
.\venv_fer\Scripts\Activate.ps1
```

**Linux/Mac:**
```bash
source venv_fer/bin/activate
```

4. **Install Dependencies**
```bash
pip install -r requirements.txt
pip install tensorflow>=2.0.0
pip install click
pip install -e .
```

### Usage

#### 1. Detect Emotions in Image
```bash
python demo.py image Jiten-Bhalavat.png
```

Output: `Jiten-Bhalavat_drawn.png` with detected emotions

#### 2. Analyze Video
```bash
python demo.py video demo.mp4
```

Shows emotion analysis over time with graphs

#### 3. Real-time Webcam Detection
```bash
python demo.py webcam 0
```

Press 'q' to quit

#### 4. Use MTCNN (More Accurate)
```bash
python demo.py image Jiten-Bhalavat.png --mtcnn
```

### Python API

```python
import cv2
from fer import FER

# Load image
img = cv2.imread("Jiten-Bhalavat.png")

# Create detector
detector = FER()

# Detect emotions
result = detector.detect_emotions(img)
print(result)

# Get top emotion
emotion, score = detector.top_emotion(img)
print(f"Detected: {emotion} ({score:.2%})")
```

**Output Example:**
```python
[{
  'box': [277, 90, 48, 63],
  'emotions': {
    'angry': 0.02,
    'disgust': 0.00,
    'fear': 0.05,
    'happy': 0.85,
    'sad': 0.01,
    'surprise': 0.02,
    'neutral': 0.05
  }
}]
```

---

## 🛠️ Tech Stack

### Core Technologies
- **Python 3.13** - Programming language
- **TensorFlow 2.20** - Deep learning framework
- **Keras 3.12** - Neural network API
- **OpenCV 4.12** - Computer vision library

### Machine Learning
- **Pre-trained CNN Model** - Emotion classification
- **FER2013 Dataset** - Training data
- **Model Format:** HDF5 / TensorFlow Lite (quantized)

### Face Detection
- **Haar Cascade Classifier** - Fast face detection (default)
- **MTCNN (Multi-task CNN)** - Accurate face detection (optional)
- **PyTorch 2.9.1** - Backend for MTCNN

### Data Processing
- **NumPy 2.2.6** - Numerical computations
- **Pandas 2.3.3** - Data manipulation
- **Matplotlib 3.10.7** - Visualization

### Video Processing
- **MoviePy 1.0.3** - Video editing
- **FFmpeg** - Video encoding/decoding
- **ImageIO 2.37.2** - Image/video I/O

### Additional Libraries
- **Click 8.3.0** - CLI interface
- **Pillow 11.3.0** - Image processing
- **tqdm 4.67.1** - Progress bars

### Development Tools
- **Virtual Environment (venv)** - Isolated Python environment
- **pip** - Package management

---

## 📊 Model Architecture

**Emotion Detection Model:**
- Architecture: Convolutional Neural Network (CNN)
- Input: 48x48 grayscale face images
- Output: 7 emotion probabilities
- Training Dataset: FER2013 (35,887 images)
- Frameworks: Keras/TensorFlow

**Face Detection:**
- Primary: Haar Cascade (fast, real-time)
- Alternative: MTCNN (accurate, slower)

---

## 📁 Project Structure

```
fer/
├── demo.py                 # Main CLI application
├── Jiten-Bhalavat.png     # Test image
├── demo.mp4               # Test video
├── src/fer/               # Core library
│   ├── fer.py            # Emotion detector
│   ├── classes.py        # Video processing
│   ├── utils.py          # Utilities
│   └── data/             # Pre-trained models
│       ├── emotion_model.hdf5
│       ├── emotion_model_quantized.tflite
│       └── haarcascade_frontalface_default.xml
├── venv_fer/             # Virtual environment
├── requirements.txt      # Dependencies
└── README.md            # This file
```

---

## 🎓 Features

✅ **Multiple Input Sources**
- Static images (JPG, PNG)
- Video files (MP4, AVI)
- Real-time webcam feed

✅ **7 Emotion Categories**
- Angry
- Disgust
- Fear
- Happy
- Sad
- Surprise
- Neutral

✅ **Flexible Face Detection**
- Fast: Haar Cascade
- Accurate: MTCNN

✅ **Output Options**
- Annotated images
- Emotion graphs
- JSON/CSV export
- Real-time visualization

---

## 📝 License

MIT License - See [LICENSE](LICENSE) file for details.

---


