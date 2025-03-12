# Usar Python 3.11 en lugar de Python 3.13
FROM python:3.11-slim

EXPOSE 5002

# Evita la generación de archivos .pyc
ENV PYTHONDONTWRITEBYTECODE=1

# Desactiva el buffering para logs en tiempo real
ENV PYTHONUNBUFFERED=1

# Instalar dependencias del sistema (ffmpeg para Whisper y git para clonar repositorios)
RUN apt-get update && apt-get install -y ffmpeg git

# Actualizar pip y setuptools
RUN python -m pip install --upgrade setuptools pip

# Instalar PyTorch (requerido por Whisper)
RUN python -m pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu

# Copiar e instalar dependencias de Python
COPY requirements.txt .
RUN python -m pip install -r requirements.txt

# Establecer el directorio de trabajo
WORKDIR /app
COPY . /app

# Crear un usuario no root
RUN adduser -u 1000 --disabled-password --gecos "" appuser && chown -R appuser /app
USER appuser

# Comando para ejecutar la aplicación
CMD ["gunicorn", "--bind", "0.0.0.0:5002", "app:app"]