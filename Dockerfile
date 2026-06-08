# PokeGuess web app — CPU inference image for Hugging Face Spaces (Docker SDK).
FROM python:3.11-slim

WORKDIR /app

# CPU-only torch (the Spaces free tier has no GPU) — keeps the image small.
RUN pip install --no-cache-dir torch torchvision \
        --index-url https://download.pytorch.org/whl/cpu

COPY requirements-app.txt .
RUN pip install --no-cache-dir -r requirements-app.txt

# App code + model weights + class lists (see .dockerignore for exclusions).
COPY . .

# Hugging Face Spaces routes traffic to port 7860.
EXPOSE 7860
ENV PORT=7860
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
