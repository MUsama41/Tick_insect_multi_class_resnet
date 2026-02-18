from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
from torchvision import transforms
from PIL import Image
import io
import os

from model import get_model


CLASSES = ['hyalomma_female', 'hyalomma_male', 'rhipicephalus_female', 'rhipicephalus_male']
MODEL_PATH = 'tick_model.pth'
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class PredictionResponse(BaseModel):
    label: str
    confidence: float


class PathRequest(BaseModel):
    image_path: str


app = FastAPI(
    title="Tick Classification Inference API",
    description="FastAPI server for running inference with the trained tick classification model.",
    version="1.0.0",
)

# Allow CORS for frontend (you can restrict origins as needed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def load_trained_model(model_path: str, device: torch.device):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at path: {model_path}")

    model = get_model('resnet18', num_classes=len(CLASSES), pretrained=False)
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


inference_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


@app.on_event("startup")
def startup_event():
    """
    Load the model once when the server starts.
    """
    global model
    model = load_trained_model(MODEL_PATH, DEVICE)


def predict_image_bytes(image_bytes: bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    input_tensor = inference_transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
        confidence, predicted_idx = torch.max(probabilities, 0)

    label = CLASSES[predicted_idx.item()]
    score = confidence.item() * 100.0

    return label, score


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    """
    Accept an image file and return the predicted tick class and confidence.

    - **file**: image file (jpg, jpeg, png, etc.) sent as form-data.
    """
    if file.content_type is None or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image.")

    try:
        content = await file.read()
        label, confidence = predict_image_bytes(content)
        return PredictionResponse(label=label, confidence=confidence)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")


@app.post("/predict_path", response_model=list[str])
def predict_from_path(payload: PathRequest):
    """
    Accept an absolute image path on the server, load the image, and return [label, confidence_as_string].
    """
    image_path = payload.image_path

    if not os.path.exists(image_path):
        raise HTTPException(status_code=400, detail=f"Image path does not exist: {image_path}")

    try:
        image = Image.open(image_path).convert("RGB")
        input_tensor = inference_transform(image).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
            confidence, predicted_idx = torch.max(probabilities, 0)

        label = CLASSES[predicted_idx.item()]
        score = confidence.item() * 100.0

        # Return as list of strings: [label, confidence]
        return [label, f"{score:.2f}"]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference from path failed: {str(e)}")


# Optional: run with `python api_server.py` (for local testing)
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "api_server:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
    )


