@echo off
setlocal
cd /d "%~dp0"

if not exist "imagenes-pendientes" mkdir "imagenes-pendientes"
if not exist "imagenes-recortadas" mkdir "imagenes-recortadas"

echo Procesando escaneos con Docker...
docker compose run --rm --build separador-fotos
if errorlevel 1 (
    echo.
    echo No se pudo completar el procesamiento.
    echo Comprueba que Docker Desktop este abierto y vuelve a intentarlo.
    exit /b 1
)

echo.
echo Proceso terminado. Revisa la carpeta imagenes-recortadas.
endlocal
