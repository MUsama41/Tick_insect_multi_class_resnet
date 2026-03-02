from fastapi import FastAPI, UploadFile, File, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
from torchvision import transforms
from PIL import Image
import io
import os
import base64
from typing import List

from model import get_model


class PredictionListResponse(BaseModel):
    predictions: List[str]


CLASSES = ['hyalomma_female', 'hyalomma_male', 'rhipicephalus_female', 'rhipicephalus_male']
MODEL_PATH = 'tick_model.pth'
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


app = FastAPI(
    title="Tick Classification Inference API",
    description="FastAPI server for running inference with the trained tick classification model.",
    version="1.0.0",
)

# Allow CORS for frontend
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
    try:
        if not image_bytes:
            raise ValueError("Empty image bytes received")
            
        # Check for common magic numbers
        if image_bytes.startswith(b'\xff\xd8\xff'):
            fmt = "JPEG"
        elif image_bytes.startswith(b'\x89PNG\r\n\x1a\n'):
            fmt = "PNG"
        elif image_bytes.startswith(b'GIF87a') or image_bytes.startswith(b'GIF89a'):
            fmt = "GIF"
        else:
            fmt = "Unknown/Invalid"
            
        if fmt == "Unknown/Invalid":
             # Still try to open it, PIL might handle other formats, 
             # but we'll have this info for the error message
             pass

        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        input_tensor = inference_transform(image).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
            confidences, indices = torch.sort(probabilities, descending=True)

        # Ordered list of class labels by confidence (highest first)
        ordered_labels = [CLASSES[idx.item()] for idx in indices]
        top_confidence = confidences[0].item() * 100.0

        return ordered_labels, top_confidence
    except Exception as e:
        prefix = image_bytes[:20].hex() if image_bytes else "None"
        # Determine likely format based on prefix
        msg = f"Failed to identify image (len={len(image_bytes)}, prefix=0x{prefix}). "
        msg += "Data does not appear to be a valid JPEG, PNG, or GIF. "
        msg += f"Original error: {str(e)}"
        raise ValueError(msg)



@app.post("/predict_single", response_model=PredictionListResponse)
async def predict_single(file: bytes = File(...)):
    """
    Accepts a single image file (binary) and returns a PredictionResponse.
    
    - **file**: The raw binary image data.
    """
    try:
        labels, _ = predict_image_bytes(file)
        return PredictionListResponse(predictions=labels)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")


@app.post("/predict_multi", response_model=PredictionListResponse)
async def predict_multi(files: List[UploadFile] = File(...)):
    """
    Accepts up to 3 image files (binary), performs inference on each, and returns the one with the highest confidence.
    
    - **files**: A list of up to 3 image files.
    """
    if len(files) > 3:
        raise HTTPException(status_code=400, detail="Maximum 3 images allowed")
    if not files:
        raise HTTPException(status_code=400, detail="No images provided")

    predictions = []
    errors = []
    for i, file in enumerate(files):
        try:
            content = await file.read()
            labels, confidence = predict_image_bytes(content)
            predictions.append((labels, confidence))
        except Exception as e:
            errors.append(f"Image {file.filename} failed: {str(e)}")
            continue

    if not predictions:
        error_details = "; ".join(errors)
        raise HTTPException(status_code=400, detail=f"Inference failed for all provided images. Errors: {error_details}")

    # Return the predictions for the image with the highest top-1 confidence
    best_labels, _ = max(predictions, key=lambda x: x[1])
    return PredictionListResponse(predictions=best_labels)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "api_server:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
    )
