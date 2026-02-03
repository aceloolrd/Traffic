# Первый запуск (сборка + запуск)
# docker compose -p traffic_analyzer up -d --build

# Обычный перезапуск (без пересборки)
# docker compose -p traffic_analyzer up -d

# Остановка
# docker compose -p traffic_analyzer down

# Перезапуск после изменений в коде (без Dockerfile)
# docker compose -p traffic_analyzer up -d --build  # пересоберёт только traffic_analyzer

# Или просто restart без пересборки
# docker compose -p traffic_analyzer restart traffic_analyzer_camera_1 traffic_analyzer_camera_2

FROM python:3.10.13

# Dev зависимости для Cython + lap в одном RUN (кэш лучше)
RUN apt-get update && apt-get install -y \
    python3-dev \
    cython3 \
    build-essential \
    curl \
    software-properties-common \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Обновляем pip
RUN python3 -m pip install --no-cache-dir --upgrade pip

# ВСЕ зависимости в одном RUN (numpy -> cython -> cython_bbox/lap -> torch -> requirements)
COPY requirements.txt /app/
RUN pip3 install --no-cache-dir \
    "numpy<2" \
    cython \
    "cython_bbox==0.1.5" \
    "lap==0.4.0" \
    "torch==2.3.1" \
    "torchvision==0.18.1" \
    --index-url https://download.pytorch.org/whl/cu121 \
    -r requirements.txt

# Копируем код
COPY . /app
