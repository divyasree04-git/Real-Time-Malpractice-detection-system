
# Exam Cheating Detection using CNN + LSTM

## Features
- Detect cheating behaviors from laptop webcam
- Classes: normal, looking around, using mobile, sharing answers, leaning to copy
- CNN for spatial features + LSTM for temporal behavior
- Real-time inference with OpenCV

## How to Run
```bash
pip install -r requirements.txt
python src/training/train.py
python src/inference/webcam_detect.py
```
