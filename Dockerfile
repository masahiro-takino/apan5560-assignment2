FROM python:3.12-slim

WORKDIR /app

RUN pip install --no-cache-dir uv

COPY pyproject.toml README.md ./

RUN uv pip install --system fastapi uvicorn python-multipart pillow numpy torch torchvision tqdm

COPY app ./app
COPY helper_lib ./helper_lib
COPY checkpoints ./checkpoints
COPY train_cifar10.py ./train_cifar10.py
COPY sample_images ./sample_images

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
