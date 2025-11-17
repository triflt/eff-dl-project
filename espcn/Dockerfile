FROM python:3.10-slim

WORKDIR /app

# Системные зависимости
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    wget \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Python зависимости
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Код проекта
COPY . .

# Создание директорий для данных и результатов
RUN mkdir -p data results

# Скачивание датасетов (опционально, можно примонтировать volume)
# RUN python -c "from dataset import preprocess_datasets; preprocess_datasets()"

ENV PYTHONUNBUFFERED=1
ENV OMP_NUM_THREADS=4

# По умолчанию запускается bash для интерактивной работы
CMD ["/bin/bash"]

