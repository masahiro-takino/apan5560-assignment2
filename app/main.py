from fastapi import FastAPI, File, UploadFile
from app.services.prediction_service import predict_image

app = FastAPI(
    title="APANPS5560 Assignment 2 CNN API",
    description="FastAPI application for CIFAR10 image classification using a PyTorch CNN.",
    version="1.0.0",
)

@app.get("/")
def read_root():
    return {"message": "APANPS5560 Assignment 2 CNN FastAPI server is running."}

@app.get("/health")
def health_check():
    return {"status": "healthy"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    result = predict_image(image_bytes)
    return result
