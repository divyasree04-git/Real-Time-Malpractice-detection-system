import cv2
import torch
import numpy as np
from ultralytics import YOLO
from torchvision import models, transforms
from torch import nn
from collections import deque, defaultdict

# ============================================
# Device
# ============================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Running on:", device)

# ============================================
# YOLO model
# ============================================

yolo_model = YOLO("yolov8n.pt")

PERSON_CLASS = 0
PHONE_CLASS = 67

# ============================================
# CNN classifier
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
# Face detection
# ============================================

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# ============================================
# Stabilization system
# ============================================

prediction_history = defaultdict(lambda: deque(maxlen=10))

def get_student_id(box):

    x1,y1,x2,y2 = box
    cx = (x1+x2)//2
    cy = (y1+y2)//2

    return (cx//50, cy//50)


def get_stable_label(student_id, label):

    history = prediction_history[student_id]
    history.append(label)

    return max(set(history), key=history.count)

# ============================================
# Helper functions
# ============================================

def is_overlapping(boxA, boxB):

    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    return xA < xB and yA < yB


def get_center(box):

    x1,y1,x2,y2 = box
    return ((x1+x2)//2, (y1+y2)//2)


def distance(boxA, boxB):

    c1 = get_center(boxA)
    c2 = get_center(boxB)

    return np.sqrt((c1[0]-c2[0])**2 + (c1[1]-c2[1])**2)

# ============================================
# Webcam
# ============================================

cap = cv2.VideoCapture(0)

while True:

    ret, frame = cap.read()

    if not ret:
        break

    results = yolo_model(frame, verbose=False)

    phone_boxes = []
    student_boxes = []

    # ========================================
    # Detect phones and persons
    # ========================================

    for r in results:

        for box in r.boxes:

            cls = int(box.cls[0])
            conf = float(box.conf[0])

            if conf < 0.6:
                continue

            x1,y1,x2,y2 = map(int, box.xyxy[0])

            if cls == PHONE_CLASS:

                phone_boxes.append((x1,y1,x2,y2))

                cv2.rectangle(frame,(x1,y1),(x2,y2),(0,0,255),2)

                cv2.putText(frame,"Mobile",
                            (x1,y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (0,0,255),
                            2)

            if cls == PERSON_CLASS:

                student_boxes.append((x1,y1,x2,y2))

    # ========================================
    # Process each student
    # ========================================

    for student_box in student_boxes:

        x1,y1,x2,y2 = student_box

        student_img = frame[y1:y2, x1:x2]

        if student_img.size == 0:
            continue

        label = "normal"

        # ====================================
        # CNN classification
        # ====================================

        img = cv2.cvtColor(student_img, cv2.COLOR_BGR2RGB)
        img = transform(img).unsqueeze(0).to(device)

        with torch.no_grad():

            output = classifier(img)
            probs = torch.softmax(output, dim=1)

            conf, pred = torch.max(probs, 1)

            if conf.item() > 0.75:
                label = CLASSES[pred.item()]

        # ====================================
        # Mobile overlap detection
        # ====================================

        for phone_box in phone_boxes:

            if is_overlapping(student_box, phone_box):
                label = "using mobile"

        # ====================================
        # Head direction detection
        # ====================================

        gray = cv2.cvtColor(student_img, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray,1.3,5)

        direction = "Center"

        for (fx,fy,fw,fh) in faces:

            face_center = fx + fw//2
            student_center = student_img.shape[1]//2

            if face_center < student_center - 30:
                direction = "Looking Left"

            elif face_center > student_center + 30:
                direction = "Looking Right"

        # ====================================
        # Seeing others paper logic
        # ====================================

        for other_box in student_boxes:

            if other_box == student_box:
                continue

            if distance(student_box, other_box) < 150:

                if direction != "Center":
                    label = "seeing others paper"

        # ====================================
        # Passing chit logic
        # ====================================

        for other_box in student_boxes:

            if other_box == student_box:
                continue

            if distance(student_box, other_box) < 100:

                if direction != "Center":
                    label = "passing chit"

        # ====================================
        # Stabilization
        # ====================================

        student_id = get_student_id(student_box)

        label = get_stable_label(student_id, label)

        # ====================================
        # Draw result
        # ====================================

        color = (0,255,0)

        if label != "normal":
            color = (0,0,255)

        cv2.rectangle(frame,(x1,y1),(x2,y2),color,2)

        cv2.putText(frame,
                    label,
                    (x1,y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    color,
                    2)

        cv2.putText(frame,
                    direction,
                    (x1,y2+20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255,0,0),
                    2)

    cv2.imshow("Advanced Exam Cheating Detection System", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()