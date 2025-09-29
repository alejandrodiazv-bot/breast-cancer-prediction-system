FROM python:3.9-slim

WORKDIR /app

# Copiar requirements primero
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar código fuente
COPY src/ ./src/

# Crear directorios necesarios
RUN mkdir -p logs models

# Entrenar el modelo DENTRO del contenedor (esto asegura compatibilidad)
RUN python src/train_model.py

EXPOSE 5000

CMD ["python", "src/app.py"]
