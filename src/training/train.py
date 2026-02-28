import cv2
import torch
import numpy as np
from ultralytics import YOLO
from torchvision import models, transforms
from torch import nn

# ============================================
# Device Configuration (GPU / CPU)
# ============================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Running on:", device)

# ============================================
# Load YOLOv8 model (person + mobile detection)
# ============================================

yolo_model = YOLO("yolov8n.pt")

PERSON_CLASS_ID = 0
PHONE_CLASS_ID = 67

# ============================================
# Load Cheating Classification Model
# ============================================

CLASSES = [
    "leaning to copy",
    "looking around",
    "normal",
    "sharing answers",
    "using mobile"
]

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

cnn = models.mobilenet_v2(weights=None)
cnn.classifier = nn.Identity()

classifier = nn.Sequential(
    cnn,
    nn.Linear(1280, len(CLASSES))
)

classifier.load_state_dict(
    torch.load("models/cheating_model.pth", map_location=device)
)

classifier.to(device)
classifier.eval()

# ============================================
# Face detection (Head direction)
# ============================================

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# ============================================
# Webcam
# ============================================

cap = cv2.VideoCapture(0)

while True:

    ret, frame = cap.read()
    if not ret:
        break

    # ========================================
    # YOLO Detection
    # ========================================

    results = yolo_model(frame, verbose=False)

    mobile_detected = False

    for r in results:

        boxes = r.boxes

        for box in boxes:

            cls_id = int(box.cls[0])
            confidence = float(box.conf[0])

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # ====================================
            # Mobile Phone Detection
            # ====================================

            if cls_id == PHONE_CLASS_ID:

                mobile_detected = True

                cv2.rectangle(frame, (x1,y1), (x2,y2), (0,0,255), 3)

                cv2.putText(
                    frame,
                    f"Mobile Phone ({confidence:.2f})",
                    (x1, y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0,0,255),
                    2
                )

            # ====================================
            # Student Detection
            # ====================================

            if cls_id == PERSON_CLASS_ID:

                student_img = frame[y1:y2, x1:x2]

                if student_img.size == 0:
                    continue

                # ================================
                # Cheating Classification
                # ================================

                img = cv2.cvtColor(student_img, cv2.COLOR_BGR2RGB)
                img = transform(img).unsqueeze(0).to(device)

                with torch.no_grad():

                    output = classifier(img)
                    probs = torch.softmax(output, dim=1)

                    conf, pred = torch.max(probs, 1)

                    label = CLASSES[pred.item()]
                    conf = conf.item()

                # ================================
                # Head Direction Detection
                # ================================

                gray = cv2.cvtColor(student_img, cv2.COLOR_BGR2GRAY)

                faces = face_cascade.detectMultiScale(gray, 1.3, 5)

                direction = "Center"

                for (fx, fy, fw, fh) in faces:

                    face_center = fx + fw//2
                    student_center = student_img.shape[1]//2

                    if face_center < student_center - 30:
                        direction = "Looking Left"

                    elif face_center > student_center + 30:
                        direction = "Looking Right"

                # ================================
                # Override if mobile detected
                # ================================

                if mobile_detected:
                    label = "using mobile"

                # ================================
                # Draw Student Box
                # ================================

                if label == "normal":
                    color = (0,255,0)
                else:
                    color = (0,0,255)

                cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)

                cv2.putText(
                    frame,
                    f"{label} ({conf:.2f})",
                    (x1, y1-30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,
                    2
                )

                cv2.putText(
                    frame,
                    direction,
                    (x1, y2+20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255,0,0),
                    2
                )

    # ========================================
    # Show frame
    # ========================================

    cv2.imshow("Advanced Exam Cheating Detection System", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()