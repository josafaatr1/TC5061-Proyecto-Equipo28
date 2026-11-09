# Imagen OCI de la API de predicción de MiBici (FastAPI + modelos P2 de MLflow).
#
# Requiere haber exportado antes los modelos champion del Model Registry:
#   python -m mibici.api.export_models     (o: make export-models)
#
# Construcción y ejecución:
#   podman build -t mibici-api .
#   podman run --rm -p 8000:8000 mibici-api

# Imagen base fijada por digest (índice multiarquitectura de python:3.14-slim, Python 3.14.10,
# publicado el 2026-11-05): la misma imagen en cualquier máquina y en cualquier fecha.
FROM docker.io/library/python:3.14-slim@sha256:9aaa474097d094040649f2162a627c0049b408f52138d980916788d80f04802f

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    MLFLOW_DISABLE_AGENT_HINT=1

WORKDIR /app

# Usuario sin privilegios para ejecutar el servicio.
RUN useradd --create-home --uid 10001 --shell /usr/sbin/nologin app

# Dependencias primero: esta capa se reutiliza mientras requirements-api.txt no cambie.
COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

# Código del paquete y modelos exportados del Model Registry.
COPY mibici/ mibici/
COPY models/ models/

# Por defecto la API usa los modelos empaquetados. Para cargarlos del Model Registry,
# sobrescribir al ejecutar, por ejemplo:
#   -e MLFLOW_TRACKING_URI=http://host.containers.internal:5000
#   -e BIKE_MODEL_URI=models:/mibici-p2-bike-pressure@champion
#   -e DOCK_MODEL_URI=models:/mibici-p2-dock-pressure@champion
ENV BIKE_MODEL_URI=/app/models/p2-bike-pressure \
    DOCK_MODEL_URI=/app/models/p2-dock-pressure \
    PORT=8000

USER app

EXPOSE 8000

# La imagen slim no incluye curl; el chequeo usa Python. Podman solo conserva
# HEALTHCHECK si la imagen se construye con --format docker.
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD ["python", "-c", "import os, urllib.request; urllib.request.urlopen(f\"http://127.0.0.1:{os.environ['PORT']}/health\", timeout=3)"]

CMD ["sh", "-c", "exec uvicorn mibici.api.main:app --host 0.0.0.0 --port \"$PORT\""]
