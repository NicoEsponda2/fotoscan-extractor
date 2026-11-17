FROM python:3.12-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Esta capa solo cambia cuando cambian las dependencias y queda en cache.
COPY requirements.txt ./
RUN python -m pip install --no-cache-dir -r requirements.txt

COPY procesar_imagenes.py separador_fotos.py ./

RUN mkdir -p /datos/entrada /datos/salida

ENTRYPOINT ["python", "/app/procesar_imagenes.py"]
CMD ["--entrada", "/datos/entrada", "--salida", "/datos/salida", "--diagnostico"]
