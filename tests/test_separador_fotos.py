"""Pruebas unitarias del detector usando imágenes creadas completamente en memoria."""

from __future__ import annotations

import cv2
import numpy as np

from separador_fotos import (
    ConfiguracionDeteccion,
    eliminar_bordes_blancos,
    separar_fotos,
)


def _foto_sintetica(ancho: int, alto: int, color: tuple[int, int, int]) -> np.ndarray:
    """Crea contenido con textura y contraste suficientes para simular una foto."""
    foto = np.full((alto, ancho, 3), color, dtype=np.uint8)
    cv2.rectangle(foto, (20, 20), (ancho - 21, alto - 21), (15, 15, 15), 8)
    cv2.circle(foto, (ancho // 2, alto // 2), min(ancho, alto) // 5, (220, 180, 80), -1)
    cv2.putText(
        foto,
        "FOTO",
        (ancho // 8, alto * 4 // 5),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        (255, 255, 255),
        3,
        cv2.LINE_AA,
    )
    return foto


def test_detecta_cuatro_fotos_en_una_hoja() -> None:
    """Encuentra cuatro copias separadas y conserva dimensiones aprovechables."""
    hoja = np.full((1300, 1000, 3), (248, 248, 246), dtype=np.uint8)
    posiciones = [(70, 90), (540, 90), (70, 690), (540, 690)]
    colores = [(60, 85, 130), (120, 65, 70), (55, 120, 65), (100, 75, 125)]
    for (x, y), color in zip(posiciones, colores):
        foto = _foto_sintetica(390, 500, color)
        hoja[y : y + 500, x : x + 390] = foto

    fotos, _ = separar_fotos(hoja, corregir_orientacion=False)

    assert len(fotos) == 4
    assert all(foto.imagen.shape[0] > 450 for foto in fotos)
    assert all(foto.imagen.shape[1] > 340 for foto in fotos)


def test_una_imagen_sin_margen_se_conserva_como_una_foto() -> None:
    """Trata una imagen sin papel visible como una sola fotografía completa."""
    foto_original = _foto_sintetica(800, 600, (80, 90, 110))
    config = ConfiguracionDeteccion(proporcion_area_maxima=0.90)

    fotos, _ = separar_fotos(
        foto_original, config=config, corregir_orientacion=False
    )

    assert len(fotos) == 1
    assert abs(fotos[0].imagen.shape[1] - 800) <= 2
    assert abs(fotos[0].imagen.shape[0] - 600) <= 2


def test_elimina_bordes_blancos_en_los_cuatro_extremos() -> None:
    """Quita bandas de papel de distinto grosor en los cuatro lados."""
    contenido = _foto_sintetica(500, 360, (70, 90, 115))
    con_borde = cv2.copyMakeBorder(
        contenido,
        12,
        17,
        9,
        15,
        cv2.BORDER_CONSTANT,
        value=(248, 248, 248),
    )

    resultado = eliminar_bordes_blancos(con_borde)

    assert abs(resultado.shape[0] - contenido.shape[0]) <= 4
    assert abs(resultado.shape[1] - contenido.shape[1]) <= 4
    assert float(np.mean(resultado[0])) < 200
    assert float(np.mean(resultado[-1])) < 200
    assert float(np.mean(resultado[:, 0])) < 200
    assert float(np.mean(resultado[:, -1])) < 200


def test_elimina_papel_visible_en_esquinas_redondeadas() -> None:
    """Obtiene un rectángulo interior sin papel desde una copia redondeada."""
    alto, ancho, radio = 360, 500, 24
    contenido = _foto_sintetica(ancho, alto, (70, 90, 115))
    mascara = np.zeros((alto, ancho), dtype=np.uint8)
    cv2.rectangle(mascara, (radio, 0), (ancho - radio - 1, alto - 1), 255, -1)
    cv2.rectangle(mascara, (0, radio), (ancho - 1, alto - radio - 1), 255, -1)
    for centro in (
        (radio, radio),
        (ancho - radio - 1, radio),
        (radio, alto - radio - 1),
        (ancho - radio - 1, alto - radio - 1),
    ):
        cv2.circle(mascara, centro, radio, 255, -1)
    redondeada = np.full_like(contenido, 248)
    redondeada[mascara > 0] = contenido[mascara > 0]

    resultado = eliminar_bordes_blancos(redondeada)

    assert resultado.shape[0] < alto
    assert resultado.shape[1] < ancho
    assert all(float(np.mean(esquina)) < 200 for esquina in (
        resultado[0, 0],
        resultado[0, -1],
        resultado[-1, 0],
        resultado[-1, -1],
    ))
