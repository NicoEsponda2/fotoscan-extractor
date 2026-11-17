"""Núcleo de detección y recorte de fotografías escaneadas.

El flujo reduce temporalmente el escaneo para localizar regiones fotográficas,
proyecta cada región a un rectángulo, elimina el papel blanco residual y, de
forma opcional, determina su orientación mediante rostros. Todo el procesamiento
es local: OpenCV aporta los algoritmos y los clasificadores incluidos con el
paquete, por lo que no se descargan modelos ni se envían imágenes a Internet.

OpenCV representa las imágenes como arreglos ``numpy.ndarray`` en orden BGR.
Todas las funciones de este módulo que reciben o devuelven imágenes conservan
ese convenio.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Sequence

import cv2
import numpy as np
from PIL import Image, ImageOps


EXTENSIONES_ADMITIDAS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".tif",
    ".tiff",
    ".bmp",
    ".webp",
}


@dataclass(frozen=True)
class ConfiguracionDeteccion:
    """Parámetros del detector para escaneos de entre una y cinco fotos.

    Attributes:
        lado_maximo_analisis: Dimensión máxima de la copia usada para detectar.
            El recorte final siempre se realiza sobre la imagen original.
        proporcion_area_minima: Área mínima aceptada respecto a toda la hoja.
        proporcion_area_maxima: Área máxima aceptada respecto a toda la hoja.
        tolerancia_fondo: Distancia de color mínima entre el papel y una foto.
        proporcion_cierre: Tamaño relativo del cierre morfológico de la máscara.
        maximo_fotos: Número máximo de fotografías devueltas por escaneo.
        proporcion_maxima_borde_blanco: Profundidad máxima examinada al quitar
            una banda blanca recta de cada extremo.
    """

    lado_maximo_analisis: int = 2000
    proporcion_area_minima: float = 0.025
    proporcion_area_maxima: float = 0.97
    tolerancia_fondo: float = 18.0
    proporcion_cierre: float = 0.012
    maximo_fotos: int = 5
    proporcion_maxima_borde_blanco: float = 0.08


@dataclass
class FotoDetectada:
    """Fotografía extraída y metadatos de su ubicación original.

    Attributes:
        imagen: Recorte final en formato BGR.
        esquinas: Cuatro puntos del polígono en coordenadas del escaneo original.
        rotacion_aplicada: Giro horario aplicado, expresado en grados.
        puntuacion: Confianza geométrica relativa asignada durante la detección.
    """

    imagen: np.ndarray
    esquinas: np.ndarray
    rotacion_aplicada: int = 0
    puntuacion: float = 0.0


def leer_imagen(ruta: Path) -> np.ndarray:
    """Carga una imagen como BGR, respetando EXIF y rutas Unicode en Windows.

    Args:
        ruta: Archivo que se desea abrir.

    Returns:
        Imagen de tres canales en orden BGR.

    Raises:
        ValueError: Si Pillow no puede abrir o convertir el archivo.
    """

    try:
        with Image.open(ruta) as imagen_pil:
            imagen_pil = ImageOps.exif_transpose(imagen_pil).convert("RGB")
            rgb = np.asarray(imagen_pil)
    except (OSError, ValueError) as exc:
        raise ValueError(f"No se pudo abrir la imagen: {ruta}") from exc

    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def guardar_imagen(ruta: Path, imagen: np.ndarray, calidad_jpeg: int = 95) -> None:
    """Codifica y guarda una imagen BGR, admitiendo rutas Unicode en Windows.

    La extensión de ``ruta`` selecciona el codificador. Los directorios padre se
    crean automáticamente.

    Args:
        ruta: Destino, incluida una extensión admitida por OpenCV.
        imagen: Imagen BGR que se escribirá.
        calidad_jpeg: Calidad entre 0 y 100 cuando el destino es JPEG.

    Raises:
        OSError: Si OpenCV no puede codificar la imagen.
    """

    ruta.parent.mkdir(parents=True, exist_ok=True)
    extension = ruta.suffix.lower()
    parametros: list[int] = []
    if extension in {".jpg", ".jpeg"}:
        parametros = [cv2.IMWRITE_JPEG_QUALITY, calidad_jpeg]
    elif extension == ".png":
        parametros = [cv2.IMWRITE_PNG_COMPRESSION, 3]

    correcto, datos = cv2.imencode(extension, imagen, parametros)
    if not correcto:
        raise OSError(f"OpenCV no pudo codificar la imagen: {ruta}")
    datos.tofile(str(ruta))


def listar_imagenes(carpeta: Path) -> list[Path]:
    """Lista imágenes admitidas de una carpeta, ordenadas por nombre.

    No recorre subdirectorios. Si la carpeta no existe, devuelve una lista vacía.
    """

    if not carpeta.exists():
        return []
    return sorted(
        (
            ruta
            for ruta in carpeta.iterdir()
            if ruta.is_file() and ruta.suffix.lower() in EXTENSIONES_ADMITIDAS
        ),
        key=lambda ruta: ruta.name.casefold(),
    )


def _redimensionar_para_analisis(
    imagen: np.ndarray, lado_maximo: int
) -> tuple[np.ndarray, float]:
    """Crea una copia reducida y devuelve su escala respecto del original."""
    alto, ancho = imagen.shape[:2]
    escala = min(1.0, lado_maximo / float(max(alto, ancho)))
    if escala == 1.0:
        return imagen.copy(), 1.0
    reducida = cv2.resize(
        imagen,
        (max(1, round(ancho * escala)), max(1, round(alto * escala))),
        interpolation=cv2.INTER_AREA,
    )
    return reducida, escala


def _pixeles_del_borde(imagen: np.ndarray) -> np.ndarray:
    """Toma una muestra del perímetro para estimar el color del papel."""
    alto, ancho = imagen.shape[:2]
    grosor = max(2, round(min(alto, ancho) * 0.018))
    return np.concatenate(
        (
            imagen[:grosor, :, :].reshape(-1, 3),
            imagen[-grosor:, :, :].reshape(-1, 3),
            imagen[grosor:-grosor, :grosor, :].reshape(-1, 3),
            imagen[grosor:-grosor, -grosor:, :].reshape(-1, 3),
        ),
        axis=0,
    )


def _crear_mascara_contenido(
    imagen: np.ndarray, config: ConfiguracionDeteccion
) -> np.ndarray:
    """Separa el contenido fotográfico del papel usando el borde como muestra.

    Combina distancia de color, diferencia de luminosidad y bordes Canny. Las
    operaciones morfológicas posteriores cierran grietas dentro de una foto y
    eliminan partículas pequeñas sin unir copias separadas por papel.
    """

    suavizada = cv2.GaussianBlur(imagen, (5, 5), 0)
    laboratorio = cv2.cvtColor(suavizada, cv2.COLOR_BGR2LAB).astype(np.float32)
    borde_lab = cv2.cvtColor(
        _pixeles_del_borde(suavizada).reshape(-1, 1, 3), cv2.COLOR_BGR2LAB
    ).reshape(-1, 3)

    # La mediana resiste polvo, rayas y pequeños trozos de foto en el borde.
    color_papel = np.median(borde_lab, axis=0)
    diferencia_color = np.linalg.norm(laboratorio - color_papel, axis=2)
    diferencia_luz = color_papel[0] - laboratorio[:, :, 0]
    mascara_color = (diferencia_color > config.tolerancia_fondo) | (
        diferencia_luz > config.tolerancia_fondo * 0.55
    )

    gris = cv2.cvtColor(suavizada, cv2.COLOR_BGR2GRAY)
    bordes = cv2.Canny(gris, 35, 110)
    bordes = cv2.dilate(bordes, np.ones((3, 3), np.uint8), iterations=1)

    mascara = np.where(mascara_color | (bordes > 0), 255, 0).astype(np.uint8)
    lado_corto = min(mascara.shape)
    cierre = max(5, round(lado_corto * config.proporcion_cierre))
    if cierre % 2 == 0:
        cierre += 1
    apertura = max(3, cierre // 4)
    if apertura % 2 == 0:
        apertura += 1

    mascara = cv2.morphologyEx(
        mascara,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (cierre, cierre)),
        # Una sola pasada rellena pequeños cortes dentro de una foto sin unir
        # dos copias separadas por un margen angosto de papel.
        iterations=1,
    )
    mascara = cv2.morphologyEx(
        mascara,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (apertura, apertura)),
        iterations=1,
    )
    return mascara


def _interseccion_sobre_union(caja_a: np.ndarray, caja_b: np.ndarray) -> float:
    """Calcula IoU entre las cajas alineadas que envuelven dos polígonos."""
    x1a, y1a = np.min(caja_a, axis=0)
    x2a, y2a = np.max(caja_a, axis=0)
    x1b, y1b = np.min(caja_b, axis=0)
    x2b, y2b = np.max(caja_b, axis=0)
    inter_ancho = max(0.0, min(x2a, x2b) - max(x1a, x1b))
    inter_alto = max(0.0, min(y2a, y2b) - max(y1a, y1b))
    inter = inter_ancho * inter_alto
    union = (x2a - x1a) * (y2a - y1a) + (x2b - x1b) * (y2b - y1b) - inter
    return inter / union if union > 0 else 0.0


def _detectar_cajas(
    imagen_reducida: np.ndarray,
    config: ConfiguracionDeteccion,
) -> tuple[list[tuple[np.ndarray, float]], np.ndarray]:
    """Localiza candidatos rectangulares en la copia reducida del escaneo.

    Returns:
        Una lista de pares ``(esquinas, puntuacion)`` sin duplicados y la
        máscara binaria utilizada para encontrarlos.
    """
    mascara = _crear_mascara_contenido(imagen_reducida, config)
    contornos, _ = cv2.findContours(mascara, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    alto, ancho = mascara.shape
    area_total = float(alto * ancho)
    dimension_minima = min(alto, ancho) * 0.08
    candidatos: list[tuple[np.ndarray, float]] = []

    for contorno in contornos:
        area_contorno = cv2.contourArea(contorno)
        if area_contorno <= 0:
            continue
        rectangulo = cv2.minAreaRect(contorno)
        lado_a, lado_b = rectangulo[1]
        area_caja = float(lado_a * lado_b)
        if area_caja <= 0:
            continue
        proporcion_area = area_caja / area_total
        if not (
            config.proporcion_area_minima
            <= proporcion_area
            <= config.proporcion_area_maxima
        ):
            continue
        if min(lado_a, lado_b) < dimension_minima:
            continue
        relacion_lados = max(lado_a, lado_b) / max(1.0, min(lado_a, lado_b))
        if relacion_lados > 4.2:
            continue

        rectangularidad = min(1.0, area_contorno / area_caja)
        if rectangularidad < 0.45:
            continue
        caja = cv2.boxPoints(rectangulo).astype(np.float32)
        puntuacion = proporcion_area * (0.5 + 0.5 * rectangularidad)
        candidatos.append((caja, puntuacion))

    # Elimina duplicados que pueden aparecer alrededor de un borde grueso.
    sin_duplicados: list[tuple[np.ndarray, float]] = []
    for caja, puntuacion in sorted(candidatos, key=lambda item: item[1], reverse=True):
        if all(_interseccion_sobre_union(caja, previa[0]) < 0.72 for previa in sin_duplicados):
            sin_duplicados.append((caja, puntuacion))

    if len(sin_duplicados) > config.maximo_fotos:
        sin_duplicados = sin_duplicados[: config.maximo_fotos]
    return sin_duplicados, mascara


def _ordenar_esquinas(puntos: np.ndarray) -> np.ndarray:
    """Ordena cuatro puntos como superior izq./der. e inferior der./izq."""
    puntos = np.asarray(puntos, dtype=np.float32)
    suma = puntos.sum(axis=1)
    diferencia = np.diff(puntos, axis=1).reshape(-1)
    return np.array(
        [
            puntos[np.argmin(suma)],
            puntos[np.argmin(diferencia)],
            puntos[np.argmax(suma)],
            puntos[np.argmax(diferencia)],
        ],
        dtype=np.float32,
    )


def _recortar_con_perspectiva(imagen: np.ndarray, esquinas: np.ndarray) -> np.ndarray:
    """Rectifica un cuadrilátero y lo devuelve como una imagen rectangular."""
    superior_izq, superior_der, inferior_der, inferior_izq = _ordenar_esquinas(esquinas)
    ancho_superior = np.linalg.norm(superior_der - superior_izq)
    ancho_inferior = np.linalg.norm(inferior_der - inferior_izq)
    alto_izquierdo = np.linalg.norm(inferior_izq - superior_izq)
    alto_derecho = np.linalg.norm(inferior_der - superior_der)
    ancho = max(2, round(max(ancho_superior, ancho_inferior)))
    alto = max(2, round(max(alto_izquierdo, alto_derecho)))

    destino = np.array(
        [[0, 0], [ancho - 1, 0], [ancho - 1, alto - 1], [0, alto - 1]],
        dtype=np.float32,
    )
    matriz = cv2.getPerspectiveTransform(
        np.array([superior_izq, superior_der, inferior_der, inferior_izq]), destino
    )
    return cv2.warpPerspective(
        imagen,
        matriz,
        (ancho, alto),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )


def _ancho_borde_blanco(
    mascara_blanca: np.ndarray,
    luminosidad: np.ndarray,
    desde_inicio: bool,
    maximo: int,
) -> int:
    """Mide una banda blanca continua tolerando polvo y antialiasing.

    Args:
        mascara_blanca: Máscara booleana cuyas filas se inspeccionan.
        luminosidad: Canal de valor HSV alineado con la máscara.
        desde_inicio: Si es falso, examina desde la última fila hacia atrás.
        maximo: Cantidad máxima de filas que puede ocupar el borde.

    Returns:
        Número de filas consecutivas consideradas papel blanco.
    """

    if not desde_inicio:
        mascara_blanca = mascara_blanca[::-1]
        luminosidad = luminosidad[::-1]

    ultimo_borde = 0
    fallos_consecutivos = 0
    for indice in range(min(maximo, mascara_blanca.shape[0])):
        proporcion_blanca = float(np.mean(mascara_blanca[indice]))
        mediana_luz = float(np.median(luminosidad[indice]))
        # El primer criterio cubre papel limpio; el segundo incluye la franja
        # de antialiasing entre el papel y la fotografía.
        es_borde = (proporcion_blanca >= 0.68 and mediana_luz >= 215) or (
            ultimo_borde > 0 and proporcion_blanca >= 0.42 and mediana_luz >= 205
        )
        if es_borde:
            ultimo_borde = indice + 1
            fallos_consecutivos = 0
        else:
            fallos_consecutivos += 1
            if fallos_consecutivos >= 3:
                break
    return ultimo_borde


def eliminar_bordes_blancos(
    imagen: np.ndarray,
    proporcion_maxima: float = 0.08,
) -> np.ndarray:
    """Elimina papel blanco conectado a los cuatro extremos de una foto.

    Solo inspecciona una fracción limitada de cada lado y exige que la mayoría
    de la línea sea clara y poco saturada. Así evita confundir paredes, cielo o
    ropa clara con el papel que rodea la copia fotográfica. Una segunda etapa
    reconoce papel en al menos tres esquinas, lo que permite recortar copias
    antiguas con esquinas redondeadas.

    Args:
        imagen: Recorte BGR posiblemente rodeado por papel blanco.
        proporcion_maxima: Profundidad relativa máxima para bandas rectas. Las
            esquinas redondeadas usan un límite interno conservador del 15 %.

    Returns:
        Una copia recortada, o la imagen original si no se detecta papel.

    Raises:
        ValueError: Si ``imagen`` está vacía.
    """

    if imagen is None or imagen.size == 0:
        raise ValueError("La imagen de entrada esta vacia")
    alto, ancho = imagen.shape[:2]
    if alto < 20 or ancho < 20:
        return imagen

    suavizada = cv2.GaussianBlur(imagen, (3, 3), 0)
    hsv = cv2.cvtColor(suavizada, cv2.COLOR_BGR2HSV)
    luminosidad = hsv[:, :, 2]
    saturacion = hsv[:, :, 1]
    canal_minimo = np.min(suavizada, axis=2)
    mascara_blanca = (
        (luminosidad >= 218) & (saturacion <= 58) & (canal_minimo >= 195)
    )

    maximo_vertical = max(1, round(alto * proporcion_maxima))
    maximo_horizontal = max(1, round(ancho * proporcion_maxima))
    superior = _ancho_borde_blanco(
        mascara_blanca, luminosidad, True, maximo_vertical
    )
    inferior = _ancho_borde_blanco(
        mascara_blanca, luminosidad, False, maximo_vertical
    )
    izquierdo = _ancho_borde_blanco(
        mascara_blanca.T, luminosidad.T, True, maximo_horizontal
    )
    derecho = _ancho_borde_blanco(
        mascara_blanca.T, luminosidad.T, False, maximo_horizontal
    )

    # La máscara se calcula suavizada para resistir ruido; por ello, el último
    # píxel puro de papel puede quedar justo después del límite detectado.
    superior += 1 if superior else 0
    inferior += 1 if inferior else 0
    izquierdo += 1 if izquierdo else 0
    derecho += 1 if derecho else 0

    # Conserva siempre una zona central amplia, incluso ante una foto muy clara.
    if superior + inferior >= alto * 0.35:
        superior = inferior = 0
    if izquierdo + derecho >= ancho * 0.35:
        izquierdo = derecho = 0
    if any((superior, inferior, izquierdo, derecho)):
        imagen = imagen[
            superior : alto - inferior, izquierdo : ancho - derecho
        ].copy()

    # Una fotografía con esquinas redondeadas deja triángulos de papel aunque
    # no exista una banda blanca completa. Solo se consideran componentes
    # claros conectados a una esquina y de profundidad limitada.
    alto, ancho = imagen.shape[:2]
    if alto < 20 or ancho < 20:
        return imagen
    hsv = cv2.cvtColor(cv2.GaussianBlur(imagen, (3, 3), 0), cv2.COLOR_BGR2HSV)
    canal_minimo = np.min(imagen, axis=2)
    blanco_estricto = np.where(
        (hsv[:, :, 2] >= 222) & (hsv[:, :, 1] <= 48) & (canal_minimo >= 202),
        255,
        0,
    ).astype(np.uint8)
    # Una línea muy delgada de papel puede unir las cuatro esquinas a través
    # del perímetro. Se ignora esa franja antes de buscar los componentes para
    # que cada triángulo de una foto redondeada se evalúe por separado.
    margen_exterior = max(2, round(min(alto, ancho) * 0.006))
    blanco_estricto[:margen_exterior, :] = 0
    blanco_estricto[-margen_exterior:, :] = 0
    blanco_estricto[:, :margen_exterior] = 0
    blanco_estricto[:, -margen_exterior:] = 0
    cantidad, etiquetas, estadisticas, _ = cv2.connectedComponentsWithStats(
        blanco_estricto, connectivity=8
    )
    proporcion_maxima_esquina = max(0.15, proporcion_maxima)
    maximo_vertical = max(2, round(alto * proporcion_maxima_esquina))
    maximo_horizontal = max(2, round(ancho * proporcion_maxima_esquina))
    semilla = max(
        margen_exterior + 2,
        round(min(alto, ancho) * 0.035),
    )
    zonas_semilla = (
        etiquetas[margen_exterior:semilla, margen_exterior:semilla],
        etiquetas[margen_exterior:semilla, ancho - semilla : ancho - margen_exterior],
        etiquetas[alto - semilla : alto - margen_exterior, margen_exterior:semilla],
        etiquetas[
            alto - semilla : alto - margen_exterior,
            ancho - semilla : ancho - margen_exterior,
        ],
    )
    recortes_esquina = [0, 0, 0, 0]  # arriba, abajo, izquierda, derecha
    esquinas_detectadas = [False, False, False, False]
    for indice_zona, zona in enumerate(zonas_semilla):
        for etiqueta in np.unique(zona):
            if etiqueta == 0 or etiqueta >= cantidad:
                continue
            x, y, ancho_componente, alto_componente, _ = estadisticas[etiqueta]
            if (
                ancho_componente + margen_exterior > maximo_horizontal
                or alto_componente + margen_exterior > maximo_vertical
            ):
                continue
            if (
                indice_zona == 0
                and x <= margen_exterior
                and y <= margen_exterior
            ):
                esquinas_detectadas[0] = True
                recortes_esquina[0] = max(
                    recortes_esquina[0], y + alto_componente + 1
                )
                recortes_esquina[2] = max(
                    recortes_esquina[2], x + ancho_componente + 1
                )
            elif (
                indice_zona == 1
                and x + ancho_componente >= ancho - margen_exterior
                and y <= margen_exterior
            ):
                esquinas_detectadas[1] = True
                recortes_esquina[0] = max(
                    recortes_esquina[0], y + alto_componente + 1
                )
                recortes_esquina[3] = max(
                    recortes_esquina[3], ancho - x + 1
                )
            elif (
                indice_zona == 2
                and x <= margen_exterior
                and y + alto_componente >= alto - margen_exterior
            ):
                esquinas_detectadas[2] = True
                recortes_esquina[1] = max(
                    recortes_esquina[1], alto - y + 1
                )
                recortes_esquina[2] = max(
                    recortes_esquina[2], x + ancho_componente + 1
                )
            elif (
                indice_zona == 3
                and x + ancho_componente >= ancho - margen_exterior
                and y + alto_componente >= alto - margen_exterior
            ):
                esquinas_detectadas[3] = True
                recortes_esquina[1] = max(
                    recortes_esquina[1], alto - y + 1
                )
                recortes_esquina[3] = max(
                    recortes_esquina[3], ancho - x + 1
                )

    superior, inferior, izquierdo, derecho = recortes_esquina
    if (
        sum(esquinas_detectadas) >= 3
        and any(recortes_esquina)
        and superior + inferior < alto * 0.30
        and izquierdo + derecho < ancho * 0.30
    ):
        imagen = imagen[
            superior : alto - inferior, izquierdo : ancho - derecho
        ].copy()
    return imagen


@lru_cache(maxsize=1)
def _cargar_clasificadores_rostros() -> tuple[cv2.CascadeClassifier, ...]:
    """Carga una sola vez los clasificadores frontal y de perfil de OpenCV."""
    carpeta = Path(cv2.data.haarcascades)
    clasificadores: list[cv2.CascadeClassifier] = []
    for nombre in ("haarcascade_frontalface_default.xml", "haarcascade_profileface.xml"):
        clasificador = cv2.CascadeClassifier(str(carpeta / nombre))
        if not clasificador.empty():
            clasificadores.append(clasificador)
    return tuple(clasificadores)


@lru_cache(maxsize=1)
def _cargar_clasificador_ojos() -> cv2.CascadeClassifier | None:
    """Carga una sola vez el clasificador de ojos incluido con OpenCV."""
    ruta = Path(cv2.data.haarcascades) / "haarcascade_eye_tree_eyeglasses.xml"
    clasificador = cv2.CascadeClassifier(str(ruta))
    return None if clasificador.empty() else clasificador


def _puntuacion_orientacion(
    imagen: np.ndarray, clasificadores: Sequence[cv2.CascadeClassifier]
) -> float:
    """Puntúa una orientación según rostros, ojos y tamaño de detección."""
    alto, ancho = imagen.shape[:2]
    escala = min(1.0, 900.0 / max(alto, ancho))
    if escala < 1.0:
        imagen = cv2.resize(
            imagen,
            (round(ancho * escala), round(alto * escala)),
            interpolation=cv2.INTER_AREA,
        )
    gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    gris = cv2.equalizeHist(gris)
    lado_minimo = max(24, round(min(gris.shape) * 0.045))
    puntuacion = 0.0
    clasificador_ojos = _cargar_clasificador_ojos()
    for indice, clasificador in enumerate(clasificadores):
        rostros = clasificador.detectMultiScale(
            gris,
            scaleFactor=1.1,
            minNeighbors=5 if indice == 0 else 6,
            minSize=(lado_minimo, lado_minimo),
        )
        peso = 1.0 if indice == 0 else 0.25
        for x, y, ancho_rostro, alto_rostro in rostros:
            valor = 1.0 + (ancho_rostro * alto_rostro) / gris.size * 20.0
            if indice == 0 and clasificador_ojos is not None:
                # Un falso rostro es relativamente frecuente al girar fotos
                # antiguas. Uno o dos ojos en la mitad superior aportan una
                # señal mucho más fiable de que la cara está realmente derecha.
                region_superior = gris[
                    y : y + round(alto_rostro * 0.68), x : x + ancho_rostro
                ]
                lado_ojo = max(8, round(min(ancho_rostro, alto_rostro) * 0.10))
                ojos = clasificador_ojos.detectMultiScale(
                    region_superior,
                    scaleFactor=1.1,
                    minNeighbors=4,
                    minSize=(lado_ojo, lado_ojo),
                    maxSize=(
                        max(lado_ojo + 1, round(ancho_rostro * 0.48)),
                        max(lado_ojo + 1, round(alto_rostro * 0.38)),
                    ),
                )
                cantidad_ojos = min(2, len(ojos))
                if cantidad_ojos:
                    valor += 2.2 * cantidad_ojos
                else:
                    valor *= 0.30
            puntuacion += peso * valor
    return puntuacion


def orientar_por_rostros(imagen: np.ndarray) -> tuple[np.ndarray, int]:
    """Elige entre 0, 90, 180 y 270 grados usando rostros y ojos.

    Args:
        imagen: Fotografía BGR cuya orientación se desea corregir.

    Returns:
        Un par con la imagen orientada y el giro horario aplicado en grados. Si
        ninguna variante tiene una ventaja clara, conserva la orientación.
    """

    clasificadores = _cargar_clasificadores_rostros()
    if not clasificadores:
        return imagen, 0

    variantes = [
        (imagen, 0),
        (cv2.rotate(imagen, cv2.ROTATE_90_CLOCKWISE), 90),
        (cv2.rotate(imagen, cv2.ROTATE_180), 180),
        (cv2.rotate(imagen, cv2.ROTATE_90_COUNTERCLOCKWISE), 270),
    ]
    puntuaciones = [
        _puntuacion_orientacion(variante, clasificadores) for variante, _ in variantes
    ]
    mejor = int(np.argmax(puntuaciones))
    if puntuaciones[mejor] <= 0:
        return imagen, 0
    segunda_mejor = sorted(puntuaciones, reverse=True)[1]
    if mejor != 0 and segunda_mejor > 0 and puntuaciones[mejor] < segunda_mejor * 1.35:
        # Si dos orientaciones son casi equivalentes probablemente sólo hay
        # falsos positivos. En ese caso es más seguro conservar la posición
        # física del recorte que imponer un giro incierto.
        return imagen, 0
    return variantes[mejor]


def _orden_lectura(
    elementos: Iterable[tuple[np.ndarray, float]],
) -> list[tuple[np.ndarray, float]]:
    """Ordena detecciones por filas, de arriba abajo y de izquierda a derecha."""
    elementos = list(elementos)
    if not elementos:
        return []
    alturas = [np.max(caja[:, 1]) - np.min(caja[:, 1]) for caja, _ in elementos]
    tolerancia_fila = max(10.0, float(np.median(alturas)) * 0.35)
    por_y = sorted(elementos, key=lambda item: float(np.mean(item[0][:, 1])))
    filas: list[list[tuple[np.ndarray, float]]] = []
    centros_fila: list[float] = []

    for elemento in por_y:
        centro_y = float(np.mean(elemento[0][:, 1]))
        if not filas or abs(centro_y - centros_fila[-1]) > tolerancia_fila:
            filas.append([elemento])
            centros_fila.append(centro_y)
        else:
            filas[-1].append(elemento)
            centros_fila[-1] = float(
                np.mean([np.mean(item[0][:, 1]) for item in filas[-1]])
            )

    resultado: list[tuple[np.ndarray, float]] = []
    for fila in filas:
        resultado.extend(sorted(fila, key=lambda item: float(np.mean(item[0][:, 0]))))
    return resultado


def separar_fotos(
    imagen: np.ndarray,
    config: ConfiguracionDeteccion | None = None,
    corregir_orientacion: bool = True,
) -> tuple[list[FotoDetectada], np.ndarray]:
    """Detecta, rectifica y recorta todas las fotografías de un escaneo.

    Si no se encuentra un borde de papel alrededor del contenido, la entrada se
    considera una sola fotografía. Esto permite procesar en el mismo lote tanto
    hojas con varias copias como archivos que ya contienen una foto individual.

    Args:
        imagen: Escaneo original en formato BGR.
        config: Parámetros del detector. Usa valores conservadores si se omite.
        corregir_orientacion: Activa la selección de giro mediante rostros.

    Returns:
        Un par formado por las fotografías detectadas en orden de lectura y la
        máscara binaria reducida usada durante la detección.

    Raises:
        ValueError: Si ``imagen`` está vacía.
    """

    if imagen is None or imagen.size == 0:
        raise ValueError("La imagen de entrada esta vacia")
    config = config or ConfiguracionDeteccion()
    reducida, escala = _redimensionar_para_analisis(
        imagen, config.lado_maximo_analisis
    )
    candidatos, mascara = _detectar_cajas(reducida, config)

    # Un archivo sin margen visible se considera una única fotografía.
    if not candidatos:
        alto, ancho = imagen.shape[:2]
        candidatos = [
            (
                np.array(
                    [[0, 0], [ancho - 1, 0], [ancho - 1, alto - 1], [0, alto - 1]],
                    dtype=np.float32,
                )
                * escala,
                1.0,
            )
        ]

    candidatos = _orden_lectura(candidatos)
    resultado: list[FotoDetectada] = []
    for caja_reducida, puntuacion in candidatos:
        esquinas_original = caja_reducida / escala
        recorte = _recortar_con_perspectiva(imagen, esquinas_original)
        recorte = eliminar_bordes_blancos(
            recorte, config.proporcion_maxima_borde_blanco
        )
        rotacion = 0
        if corregir_orientacion:
            recorte, rotacion = orientar_por_rostros(recorte)
        resultado.append(
            FotoDetectada(
                imagen=recorte,
                esquinas=esquinas_original,
                rotacion_aplicada=rotacion,
                puntuacion=puntuacion,
            )
        )
    return resultado, mascara


def crear_diagnostico(
    imagen: np.ndarray, fotos: Sequence[FotoDetectada]
) -> np.ndarray:
    """Dibuja los polígonos detectados y su orden sobre el escaneo.

    Args:
        imagen: Escaneo BGR original.
        fotos: Resultados producidos por :func:`separar_fotos`.

    Returns:
        Copia del escaneo con contornos y números verdes; no modifica la entrada.
    """

    diagnostico = imagen.copy()
    grosor = max(2, round(max(imagen.shape[:2]) / 900))
    escala_texto = max(0.7, max(imagen.shape[:2]) / 1800)
    for numero, foto in enumerate(fotos, start=1):
        poligono = np.round(foto.esquinas).astype(np.int32)
        cv2.polylines(diagnostico, [poligono], True, (0, 220, 0), grosor)
        x, y = np.min(poligono, axis=0)
        cv2.putText(
            diagnostico,
            str(numero),
            (int(x + 12), int(y + 45 * escala_texto)),
            cv2.FONT_HERSHEY_SIMPLEX,
            escala_texto,
            (0, 0, 0),
            grosor * 4,
            cv2.LINE_AA,
        )
        cv2.putText(
            diagnostico,
            str(numero),
            (int(x + 12), int(y + 45 * escala_texto)),
            cv2.FONT_HERSHEY_SIMPLEX,
            escala_texto,
            (0, 255, 0),
            grosor * 2,
            cv2.LINE_AA,
        )
    return diagnostico
