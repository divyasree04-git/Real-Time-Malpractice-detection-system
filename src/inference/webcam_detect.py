import cv2
import torch
from torchvision import models, transforms
from torch import nn

# Classes
CLASSES = [
    "leaning to copy",
    "looking around",
    "normal",
    "sharing answers",
    "using mobile"
]

# Transform
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

# Load model
cnn = models.mobilenet_v2(weights=None)
cnn.classifier = nn.Identity()

model = nn.Sequential(
    cnn,
    nn.Linear(1280, len(CLASSES))
)

model.load_state_dict(
    torch.load("models/cheating_model.pth", map_location="cpu")
)

model.eval()

# Webcam
cap = cv2.VideoCapture(0)

while True:

    ret, frame = cap.read()

    if not ret:
        break

    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = transform(img).unsqueeze(0)

    with torch.no_grad():

        output = model(img)
        pred = torch.argmax(output, 1)

        label = CLASSES[pred.item()]

    if label == "normal":
        color = (0,255,0)
    else:
        color = (0,0,255)

    cv2.putText(
        frame,
        label,
        (20,40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        color,
        2
    )

    cv2.imshow("Basic Cheating Detection", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()