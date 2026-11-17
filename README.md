<p align="center">
  <img src="https://www.python.org/static/community_logos/python-logo.png" alt="Logo de Python" width="240">
</p>

<h1 align="center">Separador de fotografías escaneadas</h1>

Extrae automáticamente entre 1 y 5 fotografías de cada escaneo. El programa
corrige la perspectiva, elimina bordes blancos y orienta las imágenes mediante
detección de rostros.

El procesamiento es completamente local: **ninguna imagen se envía a Internet**.

## Uso rápido con Docker

Solo necesitas [Docker Desktop](https://www.docker.com/products/docker-desktop/).

1. Copia los escaneos en `imagenes-pendientes/`.
2. Abre Docker Desktop.
3. Ejecuta `main.bat` o, desde PowerShell:

```powershell
.\main.bat
```

Los resultados quedarán en `imagenes-recortadas/`. La primera ejecución puede
tardar algunos minutos mientras Docker prepara la imagen.

También puedes ejecutar el servicio directamente:

```powershell
docker compose run --rm --build separador-fotos
```

## Uso con Python

Requiere Python 3.10 o posterior.

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python procesar_imagenes.py --diagnostico
```

En Linux o macOS, usa `python3 -m venv .venv` y luego
`source .venv/bin/activate`.

## Resultados

```text
imagenes-recortadas/
├── escaneo_foto_01.jpg
├── escaneo_foto_02.jpg
├── _diagnosticos/
│   └── escaneo_deteccion.jpg
└── resultado.json
```

- `*_foto_XX.jpg`: fotografías extraídas.
- `_diagnosticos/`: escaneos con las detecciones marcadas.
- `resultado.json`: resumen del procesamiento y datos de cada recorte.

Los archivos existentes no se reemplazan. Para regenerarlos con Docker:

```powershell
docker compose run --rm --build separador-fotos --entrada /datos/entrada --salida /datos/salida --diagnostico --sobrescribir
```

## Opciones principales

```text
--entrada CARPETA          Cambia la carpeta de entrada.
--salida CARPETA           Cambia la carpeta de salida.
--diagnostico              Genera imágenes con las detecciones marcadas.
--sin-rotacion             Desactiva la orientación mediante rostros.
--sobrescribir             Reemplaza resultados existentes.
--area-minima PORCENTAJE   Ajusta el tamaño mínimo de una foto.
--tolerancia-fondo VALOR   Ajusta la diferencia entre el papel y la foto.
```

Consulta todas las opciones con:

```powershell
python procesar_imagenes.py --help
```

## Pruebas

```powershell
python -m pip install -r requirements-dev.txt
python -m pytest -q
```
