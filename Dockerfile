FROM python:3.11-slim

# Install system dependencies including git
RUN apt-get update && apt-get install -y \
    git \
    libgl1 \
    libglu1-mesa \
    libxrender1 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p outputs

CMD ["python", "app.py"]