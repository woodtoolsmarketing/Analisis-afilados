"""Region de interes, limpieza de mascaras y descarte de detecciones espurias.

El objetivo es quedarse solo con la pieza que se esta inspeccionando y tirar todo lo
demas: aserrin sobre la mesa, manchas de aceite, el borde del soporte, reflejos del
metal pulido. Cada descarte queda registrado en Deteccion.descartada_por para que el
informe de feedback pueda explicar por que la pieza no se conto.
"""

from __future__ import annotations

from typing import Sequence

import cv2
import numpy as np

from .config import FiltrosConfig, RoiConfig
from .tipos import Deteccion, Escala


def rect_roi(forma_frame: tuple[int, int], cfg: RoiConfig) -> tuple[int, int, int, int]:
    """Convierte el ROI normalizado (0..1) a pixeles (x1,y1,x2,y2) dentro del frame.

    Garantiza x2>x1 y y2>y1 aun con configuraciones absurdas, y nunca se sale del frame.
    Si el ROI no esta habilitado devuelve el frame completo.
    """
    alto = int(forma_frame[0])
    ancho = int(forma_frame[1])
    if ancho <= 0 or alto <= 0:
        return (0, 0, 0, 0)

    if not cfg.habilitado:
        return (0, 0, ancho, alto)

    inicio_x = _acotar(float(cfg.x), 0.0, 1.0)
    inicio_y = _acotar(float(cfg.y), 0.0, 1.0)
    ancho_rel = _acotar(float(cfg.w), 0.0, 1.0)
    alto_rel = _acotar(float(cfg.h), 0.0, 1.0)

    x1, x2 = _ordenar_y_acotar(
        int(round(inicio_x * ancho)),
        int(round((inicio_x + ancho_rel) * ancho)),
        ancho,
    )
    y1, y2 = _ordenar_y_acotar(
        int(round(inicio_y * alto)),
        int(round((inicio_y + alto_rel) * alto)),
        alto,
    )
    return (x1, y1, x2, y2)


def recortar_roi(frame: np.ndarray, roi: tuple[int, int, int, int]) -> np.ndarray:
    """Devuelve la vista del frame limitada al ROI, recortando el rectangulo al frame."""
    alto, ancho = int(frame.shape[0]), int(frame.shape[1])
    x1, y1, x2, y2 = (int(valor) for valor in roi)

    x1 = int(_acotar(x1, 0, ancho))
    x2 = int(_acotar(x2, 0, ancho))
    y1 = int(_acotar(y1, 0, alto))
    y2 = int(_acotar(y2, 0, alto))
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return frame[y1:y2, x1:x2]


def centro_en_roi(xyxy: Sequence[float], roi: tuple[int, int, int, int]) -> bool:
    """True si el centro de la caja cae dentro del ROI.

    Se usa el centro y no la caja entera porque una pieza bien encuadrada puede asomar
    unos pixeles fuera del ROI sin dejar de ser la pieza que se quiere medir.
    """
    x1, y1, x2, y2 = (float(valor) for valor in tuple(xyxy)[:4])
    centro_x = (x1 + x2) / 2.0
    centro_y = (y1 + y2) / 2.0

    roi_x1, roi_y1, roi_x2, roi_y2 = (float(valor) for valor in roi)
    if roi_x2 < roi_x1:
        roi_x1, roi_x2 = roi_x2, roi_x1
    if roi_y2 < roi_y1:
        roi_y1, roi_y2 = roi_y2, roi_y1

    return roi_x1 <= centro_x <= roi_x2 and roi_y1 <= centro_y <= roi_y2


def limpiar_mascara(mascara: np.ndarray, kernel: int, iteraciones: int) -> np.ndarray:
    """Apertura morfologica: erosion seguida de dilatacion.

    La apertura borra los puntos sueltos mas chicos que el kernel (polvo, aserrin,
    ruido de umbralizado) y devuelve la pieza real a su tamano original con la
    dilatacion posterior. Una erosion sola adelgazaria el diente y falsearia el ancho.
    Con kernel<=1 o iteraciones<=0 la mascara se devuelve intacta.
    """
    if kernel <= 1 or iteraciones <= 0:
        return mascara

    lado = int(kernel)
    if lado % 2 == 0:
        lado += 1
    elemento = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (lado, lado))
    return cv2.morphologyEx(mascara, cv2.MORPH_OPEN, elemento, iterations=int(iteraciones))


def aplicar_filtros(
    detecciones: list[Deteccion],
    roi: tuple[int, int, int, int],
    cfg_filtros: FiltrosConfig,
    escala: Escala,
) -> tuple[list[Deteccion], list[Deteccion]]:
    """Separa las detecciones en (aceptadas, descartadas), marcando el motivo del descarte.

    Orden: ROI -> area minima en px -> area minima/maxima en mm2.

    Los limites en mm2 son los unicos que expresan una regla fisica real ("un diente no
    puede medir 2 mm2"), pero solo se pueden aplicar si hay escala valida. Sin marcador
    la unica defensa contra el ruido es el umbral en pixeles, que depende de la
    resolucion y de la distancia de la camara: por eso es un piso conservador y no un
    reemplazo del filtro en mm2.

    Una deteccion sin medida (mascara vacia o no medible) solo se filtra por ROI.
    """
    aceptadas: list[Deteccion] = []
    descartadas: list[Deteccion] = []

    hay_escala = bool(escala is not None and escala.valida and escala.mm_por_px)

    for deteccion in detecciones:
        if not centro_en_roi(deteccion.xyxy, roi):
            descartadas.append(_descartar(deteccion, "roi"))
            continue

        medida = deteccion.medida
        if medida is None:
            aceptadas.append(deteccion)
            continue

        if medida.area_px < float(cfg_filtros.area_minima_px):
            descartadas.append(_descartar(deteccion, "area_minima"))
            continue

        if hay_escala and medida.area_mm2 is not None:
            if medida.area_mm2 < float(cfg_filtros.area_minima_mm2):
                descartadas.append(_descartar(deteccion, "area_minima"))
                continue
            if medida.area_mm2 > float(cfg_filtros.area_maxima_mm2):
                descartadas.append(_descartar(deteccion, "area_maxima"))
                continue

        deteccion.descartada_por = None
        aceptadas.append(deteccion)

    return aceptadas, descartadas


def _descartar(deteccion: Deteccion, motivo: str) -> Deteccion:
    deteccion.descartada_por = motivo
    return deteccion


def _acotar(valor: float, minimo: float, maximo: float) -> float:
    return max(minimo, min(maximo, valor))


def _ordenar_y_acotar(inicio: int, fin: int, limite: int) -> tuple[int, int]:
    """Ordena y recorta un intervalo al rango [0,limite] garantizando fin>inicio."""
    inicio, fin = min(inicio, fin), max(inicio, fin)
    inicio = max(0, min(inicio, limite - 1))
    fin = max(inicio + 1, min(fin, limite))
    return inicio, fin
