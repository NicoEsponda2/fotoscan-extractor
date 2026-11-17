"""Interfaz de consola para procesar una carpeta de escaneos por lotes.

Este módulo se ocupa de entrada/salida, argumentos, nombres de archivo e informe
JSON. El procesamiento visual vive en :mod:`separador_fotos`, lo que permite
probar y reutilizar el algoritmo sin depender de la consola ni de Docker.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from separador_fotos import (
    ConfiguracionDeteccion,
    crear_diagnostico,
    guardar_imagen,
    leer_imagen,
    listar_imagenes,
    separar_fotos,
)


RAIZ_PROYECTO = Path(__file__).resolve().parent


def _crear_argumentos() -> argparse.ArgumentParser:
    """Construye el analizador de argumentos de la aplicación."""
    parser = argparse.ArgumentParser(
        description=(
            "Detecta entre 1 y 5 fotografías en cada escaneo, las endereza y "
            "las guarda como archivos independientes."
        )
    )
    parser.add_argument(
        "--entrada",
        type=Path,
        default=RAIZ_PROYECTO / "imagenes-pendientes",
        help="Carpeta de escaneos (predeterminado: imagenes-pendientes).",
    )
    parser.add_argument(
        "--salida",
        type=Path,
        default=RAIZ_PROYECTO / "imagenes-recortadas",
        help="Carpeta donde se guardarán los recortes.",
    )
    parser.add_argument(
        "--diagnostico",
        action="store_true",
        help="Guarda también una copia con las detecciones numeradas.",
    )
    parser.add_argument(
        "--sin-rotacion",
        action="store_true",
        help="No intenta orientar las fotografías mediante rostros.",
    )
    parser.add_argument(
        "--sobrescribir",
        action="store_true",
        help="Permite reemplazar recortes que ya existen.",
    )
    parser.add_argument(
        "--area-minima",
        type=float,
        default=2.5,
        metavar="PORCENTAJE",
        help="Área mínima de una foto respecto al escaneo (predeterminado: 2.5).",
    )
    parser.add_argument(
        "--tolerancia-fondo",
        type=float,
        default=18.0,
        help="Diferencia mínima respecto al color del papel (predeterminado: 18).",
    )
    return parser


def _validar_argumentos(args: argparse.Namespace) -> None:
    """Valida opciones numéricas antes de leer o escribir archivos.

    Raises:
        ValueError: Si un porcentaje o una tolerancia está fuera de rango.
    """
    if not 0.1 <= args.area_minima <= 50:
        raise ValueError("--area-minima debe estar entre 0.1 y 50")
    if not 1 <= args.tolerancia_fondo <= 100:
        raise ValueError("--tolerancia-fondo debe estar entre 1 y 100")


def procesar_carpeta(args: argparse.Namespace) -> int:
    """Procesa todos los escaneos de entrada y escribe resultados e informe.

    Las fotografías terminadas se guardan juntas en la raíz de la carpeta de
    salida. Su nombre combina el nombre del escaneo con un número correlativo.
    Un error en un archivo se registra, pero no interrumpe el resto del lote.

    Args:
        args: Espacio de nombres creado por :func:`_crear_argumentos` o un
            objeto compatible con las mismas opciones.

    Returns:
        ``0`` si todo el lote termina correctamente y ``1`` si algún escaneo
        produce un error.
    """
    _validar_argumentos(args)
    entrada = args.entrada.resolve()
    salida = args.salida.resolve()
    entrada.mkdir(parents=True, exist_ok=True)
    salida.mkdir(parents=True, exist_ok=True)
    archivos = listar_imagenes(entrada)
    if not archivos:
        print(f"No se encontraron imágenes en: {entrada}")
        print("Copia allí uno o más escaneos y vuelve a ejecutar el programa.")
        return 0

    config = ConfiguracionDeteccion(
        proporcion_area_minima=args.area_minima / 100.0,
        tolerancia_fondo=args.tolerancia_fondo,
    )
    # El manifiesto permite auditar el lote sin tener que volver a analizar las
    # imágenes: conserva configuración, dimensiones, coordenadas y rotaciones.
    manifiesto: dict[str, object] = {
        "fecha": datetime.now().astimezone().isoformat(timespec="seconds"),
        "entrada": str(entrada),
        "salida": str(salida),
        "configuracion": asdict(config),
        "archivos": [],
    }
    total_fotos = 0
    errores = 0

    for indice_archivo, ruta in enumerate(archivos, start=1):
        print(f"[{indice_archivo}/{len(archivos)}] Analizando {ruta.name}...")
        registro: dict[str, object] = {"archivo": ruta.name}
        try:
            imagen = leer_imagen(ruta)
            fotos, _ = separar_fotos(
                imagen,
                config=config,
                corregir_orientacion=not args.sin_rotacion,
            )
            salidas: list[dict[str, object]] = []
            for numero, foto in enumerate(fotos, start=1):
                # Todos los resultados quedan juntos en la carpeta de salida.
                # El nombre del escaneo mantiene cada archivo identificable.
                ruta_salida = salida / f"{ruta.stem}_foto_{numero:02d}.jpg"
                if ruta_salida.exists() and not args.sobrescribir:
                    print(f"  - Ya existe, se conserva: {ruta_salida.name}")
                else:
                    guardar_imagen(ruta_salida, foto.imagen)
                    print(
                        f"  - Foto {numero}: {foto.imagen.shape[1]}x"
                        f"{foto.imagen.shape[0]} px -> {ruta_salida.name}"
                    )
                salidas.append(
                    {
                        "archivo": str(ruta_salida.relative_to(salida)),
                        "ancho": int(foto.imagen.shape[1]),
                        "alto": int(foto.imagen.shape[0]),
                        "rotacion_aplicada": foto.rotacion_aplicada,
                        "esquinas": foto.esquinas.round(2).tolist(),
                    }
                )

            if args.diagnostico:
                ruta_diagnostico = (
                    salida / "_diagnosticos" / f"{ruta.stem}_deteccion.jpg"
                )
                guardar_imagen(ruta_diagnostico, crear_diagnostico(imagen, fotos), 90)

            registro.update({"cantidad": len(fotos), "fotos": salidas})
            total_fotos += len(fotos)
            print(f"  Detectadas: {len(fotos)} fotografía(s).")
        except Exception as exc:  # continúa con el resto del lote
            errores += 1
            registro["error"] = str(exc)
            print(f"  ERROR: {exc}", file=sys.stderr)
        manifiesto["archivos"].append(registro)  # type: ignore[union-attr]

    ruta_manifiesto = salida / "resultado.json"
    ruta_manifiesto.write_text(
        json.dumps(manifiesto, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        f"\nListo: {total_fotos} fotografía(s) extraídas de {len(archivos)} "
        f"archivo(s). Errores: {errores}."
    )
    print(f"Informe: {ruta_manifiesto}")
    return 1 if errores else 0


def main() -> int:
    """Punto de entrada de la aplicación y del contenedor Docker."""
    parser = _crear_argumentos()
    args = parser.parse_args()
    try:
        return procesar_carpeta(args)
    except ValueError as exc:
        parser.error(str(exc))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
