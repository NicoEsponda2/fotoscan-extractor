<p align="center">
  <img src="https://www.python.org/static/community_logos/python-logo.png" alt="Logo de Python" width="290">
</p>

<h1 align="center">Separador de fotografías escaneadas</h1>

<p align="center">
  Detecta, endereza y extrae automáticamente entre 1 y 5 fotografías de cada escaneo.
</p>

El programa lee todos los escaneos de `imagenes-pendientes/` y guarda cada
fotografía como un archivo JPG independiente en `imagenes-recortadas/`. También
corrige la perspectiva, elimina márgenes blancos y puede orientar las fotos
mediante detección local de rostros.

Todo el procesamiento se realiza en el equipo: **las imágenes no se envían a
Internet**.

## Inicio rápido con Docker (recomendado)

### Requisitos

- [Docker Desktop](https://www.docker.com/products/docker-desktop/) instalado y abierto.
- Los escaneos que se quieran procesar dentro de `imagenes-pendientes/`.

No es necesario instalar Python ni las dependencias del proyecto.

### Ejecutar en Windows

1. Copia los escaneos en la carpeta `imagenes-pendientes/`.
2. Abre Docker Desktop y espera a que termine de iniciar.
3. Desde la carpeta del proyecto, haz doble clic en `main.bat` o ejecútalo en
   PowerShell:

```powershell
.\main.bat
```

4. Revisa los archivos generados en `imagenes-recortadas/`.

La primera ejecución construye la imagen de Docker y puede tardar algunos
minutos. Las siguientes reutilizan sus capas y normalmente inician más rápido.

El comando equivalente sin `main.bat` es:

```powershell
docker compose run --rm --build separador-fotos
```

## Ejecución directa con Python

Esta alternativa es útil para desarrollar, depurar o modificar el algoritmo.

### Requisitos

- Python 3.10 o posterior.
- `pip` disponible.

### Preparar el entorno en Windows

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Si PowerShell no permite activar el entorno virtual, se pueden ejecutar los
comandos con su intérprete directamente:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

### Preparar el entorno en Linux o macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### Procesar los escaneos

Coloca las imágenes en `imagenes-pendientes/` y ejecuta:

```bash
python procesar_imagenes.py --diagnostico
```

Después revisa los resultados en `imagenes-recortadas/`.

## Archivos generados

```text
imagenes-recortadas/
├── CCI_000016_foto_01.jpg
├── CCI_000016_foto_02.jpg
├── CCI_000016_foto_03.jpg
├── CCI_000016_foto_04.jpg
├── _diagnosticos/
│   └── CCI_000016_deteccion.jpg
└── resultado.json
```

- Los archivos `*_foto_XX.jpg` son las fotografías extraídas.
- `_diagnosticos/` contiene una copia de cada escaneo con las detecciones
  marcadas y numeradas. Solo se crea al usar `--diagnostico`; Docker lo activa
  de forma predeterminada.
- `resultado.json` registra la cantidad de fotos, sus dimensiones, coordenadas,
  rotaciones y la configuración utilizada.

Por seguridad, el programa conserva los recortes que ya existen. Usa
`--sobrescribir` cuando necesites regenerarlos.

## Opciones de ejecución

Para consultar todas las opciones disponibles:

```bash
python procesar_imagenes.py --help
```

| Opción | Descripción |
| --- | --- |
| `--entrada CARPETA` | Cambia la carpeta de escaneos. |
| `--salida CARPETA` | Cambia la carpeta de resultados. |
| `--diagnostico` | Guarda una copia de cada escaneo con las detecciones marcadas. |
| `--sin-rotacion` | No intenta orientar las fotos mediante rostros. |
| `--sobrescribir` | Reemplaza recortes existentes. |
| `--area-minima PORCENTAJE` | Cambia el área mínima de una foto; el valor predeterminado es `2.5`. |
| `--tolerancia-fondo VALOR` | Ajusta la separación entre el papel y la foto; el valor predeterminado es `18`. |

Ejemplo con carpetas personalizadas:

```bash
python procesar_imagenes.py --entrada ./escaneos --salida ./fotos --diagnostico
```

El equivalente dentro de Docker, usando las carpetas montadas por
`compose.yaml`, es:

```bash
docker compose run --rm --build separador-fotos \
  --entrada /datos/entrada \
  --salida /datos/salida \
  --diagnostico \
  --sobrescribir
```

### Ajustar la detección

- Si una foto pequeña no se detecta, prueba `--area-minima 1`.
- Si suciedad o texto del papel se detectan como fotos, prueba
  `--area-minima 4`.
- Para papel gris o amarillento, puede ayudar bajar
  `--tolerancia-fondo` a `12`.

## Estructura del proyecto

```text
separador_fotos.py       Algoritmo de detección, orientación y recorte.
procesar_imagenes.py     Interfaz de consola y procesamiento por lotes.
tests/                   Pruebas automatizadas.
imagenes-pendientes/     Carpeta de entrada para los escaneos.
imagenes-recortadas/     Carpeta de salida para recortes e informes.
Dockerfile               Definición de la imagen del proyecto.
compose.yaml             Configuración de carpetas y servicio Docker.
main.bat                 Inicio rápido con Docker para Windows.
requirements.txt         Dependencias de ejecución.
requirements-dev.txt     Dependencias adicionales para desarrollo.
```

## Usar como módulo de Python

El algoritmo también puede integrarse en otro programa:

```python
from pathlib import Path

from separador_fotos import guardar_imagen, leer_imagen, separar_fotos

escaneo = leer_imagen(Path("mi_escaneo.jpg"))
fotos, mascara = separar_fotos(escaneo)

for numero, foto in enumerate(fotos, start=1):
    guardar_imagen(Path(f"foto_{numero:02d}.jpg"), foto.imagen)
```

Cada elemento de `fotos` es una instancia de `FotoDetectada` e incluye el
recorte, las coordenadas originales, la rotación aplicada y la puntuación
geométrica. Los ajustes avanzados se agrupan en `ConfiguracionDeteccion`.

## Ejecutar las pruebas

Con el entorno virtual activo, instala las dependencias de desarrollo y ejecuta
Pytest:

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

Las pruebas generan imágenes sintéticas en memoria y no modifican los escaneos
reales.
