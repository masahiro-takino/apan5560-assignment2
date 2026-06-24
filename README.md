# APANPS5560 Assignment 2: CNN Image Classifier API

Student: Masahiro Takino

This project implements a Convolutional Neural Network (CNN) image classifier in PyTorch and deploys it through a FastAPI API with Docker support.

## GitHub Repository

https://github.com/masahiro-takino/apan5560-assignment2

## Main Endpoints

- `GET /`
- `GET /health`
- `POST /predict`

## Local Setup

```bash
uv sync
```

## Train the CNN on CIFAR10

```bash
uv run python train_cifar10.py
```

This trains the model and saves the checkpoint to:

```text
checkpoints/cnn_cifar10.pt
```

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
