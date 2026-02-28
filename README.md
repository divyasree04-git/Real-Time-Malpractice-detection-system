
# Exam Cheating Detection using CNN + LSTM

## Features
- Detect cheating behaviors from laptop webcam
- Classes: normal, looking around, using mobile, sharing answers, leaning to copy
- CNN for spatial features + LSTM for temporal behavior
- Real-time inference with OpenCV


# Real-Time Malpractice Detection System

This project detects exam malpractice using AI and computer vision.

## Features

- Multi-student detection using YOLOv8
- Mobile phone detection
- Passing chits detection
- Head direction detection (left/right)
- Seeing others' paper detection
- Real-time webcam monitoring

## Technologies Used

- Python
- PyTorch
- YOLOv8 (Ultralytics)
- OpenCV
- MobileNetV2 CNN

## How to Run

```bash
pip install -r requirements.txt
python src/inference/webcam_detect_advanced.py
