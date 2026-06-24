from io import BytesIO
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

from helper_lib.model import get_model

CLASS_NAMES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = Path("checkpoints/cnn_cifar10.pt")

transform = transforms.Compose(
    [
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
)

model = get_model("CNN").to(DEVICE)

if MODEL_PATH.exists():
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
else:
    print(
        "WARNING: checkpoints/cnn_cifar10.pt was not found. "
        "API will run with randomly initialized weights until train_cifar10.py is executed."
    )

model.eval()

def predict_image(image_bytes: bytes) -> dict:
    image = Image.open(BytesIO(image_bytes)).convert("RGB")
    input_tensor = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(input_tensor)
        probabilities = torch.softmax(logits, dim=1)
        confidence, predicted_class = torch.max(probabilities, dim=1)

    class_id = int(predicted_class.item())
    class_name = CLASS_NAMES[class_id]

    return {
        "class_id": class_id,
        "class_name": class_name,
        "confidence": float(confidence.item()),
    }
