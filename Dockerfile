FROM python:3.10-slim

# Install system dependencies including git and build tools
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    libgl1 \
    libglu1-mesa \
    libxrender1 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Upgrade pip and install build tools
RUN pip install --upgrade pip setuptools wheel

# Copy and install requirements
COPY requirements.txt .
RUN pip install --no-cache-dir --verbose -r requirements.txt

# Copy app code
COPY . .

# Ensure outputs folder exists
RUN mkdir -p outputs

CMD ["python", "app.py"]