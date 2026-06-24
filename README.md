# APANPS5560 Assignment 2: CNN Image Classifier API

Student: Masahiro Takino

This project implements a Convolutional Neural Network (CNN) image classifier in PyTorch and deploys it through a FastAPI API with Docker support.

## GitHub Repository

https://github.com/masahiro-takino/apan5560-assignment2

## Main Endpoints

- `GET /` — confirms the API is running
- `GET /health` — health check endpoint
- `POST /predict` — accepts an uploaded image and returns the predicted CIFAR-10 class, class name, and confidence score

## Helper Library

The `helper_lib/` directory contains shared utilities for data loading, model training, and evaluation, reused across assignments.

## Quick Test (after starting the server)

```bash
curl -X POST http://localhost:8000/predict \
  -F "file=@sample_images/sample_car.png"
```

Expected response:

```json
{
  "class_id": 1,
  "class_name": "automobile",
  "confidence": 0.91
}
```

A sample test image is provided in `sample_images/sample_car.png`.

## Local Setup

```bash
uv sync
```

## Train the CNN on CIFAR-10

```bash
uv run python train_cifar10.py
```

This trains the model and saves the checkpoint to:

```text
checkpoints/cnn_cifar10.pt
```

> **Note:** The pre-trained checkpoint is already included in the repository. Running `train_cifar10.py` is optional — the API will load the existing checkpoint automatically.

## Run the API Locally

```bash
uv run uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Open:

```text
http://127.0.0.1:8000/docs
```

## Docker

```bash
docker build -t apan5560-assignment2 .
docker run -p 8000:8000 apan5560-assignment2
```

Then open:

```text
http://127.0.0.1:8000/docs
```

Or query directly:

```bash
curl -X POST http://localhost:8000/predict \
  -F "file=@sample_images/sample_car.png"
```
