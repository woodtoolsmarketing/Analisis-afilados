"""Medicion geometrica de piezas a partir de mascaras o cajas de deteccion.

Convierte pixeles a milimetros usando la escala derivada del marcador ArUco. Si no
hay escala valida las magnitudes en milimetros quedan en None: es preferible no dar
una medida a dar una medida inventada.

Recordatorio del contexto fisico: la escala solo es correcta si el marcador esta a la
misma altura que la cara medida. Con el marcador apoyado en la mesa y una pieza de
espesor apreciable, la cara superior queda mas cerca del lente y toda medida sale
inflada (error de paralaje). Este modulo no puede detectar esa situacion.
"""

from __future__ import annotations

import math
from typing import Optional, Sequence

import cv2
import numpy as np

from .tipos import Escala, Medida


def contorno_principal(mascara: np.ndarray) -> Optional[np.ndarray]:
    """Devuelve el contorno externo de mayor area de la mascara, o None si no hay ninguno."""
    binaria = _binarizar(mascara)
    if binaria is None:
        return None

    # findContours devuelve 2 valores en OpenCV 4.x y 3 en OpenCV 3.x: tomar los
    # ultimos dos elementos funciona en ambas sin preguntar por la version.
    salida = cv2.findContours(binaria, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contornos = salida[-2]
    if not contornos:
        return None
    return max(contornos, key=cv2.contourArea)


def medir_contorno(contorno: np.ndarray, escala: Escala) -> Medida:
    """Mide un contorno con el rectangulo de area minima (rotado).

    Se usa minAreaRect y no la caja recta porque una fresa o un diente apoyado en
    diagonal quedaria envuelto por una caja recta mucho mayor que la pieza, y el largo
    saldria falseado. El area, en cambio, sale del contorno real (contourArea), no del
    rectangulo, para no contar el aire de las esquinas.
    """
    puntos = _puntos_de_contorno(contorno)
    if puntos is None or puntos.shape[0] < 3:
        return _medida_degradada(puntos, escala)

    rect = cv2.minAreaRect(puntos)
    (centro_x, centro_y), (lado_a, lado_b), _ = rect
    esquinas = cv2.boxPoints(rect)

    largo_px = float(max(lado_a, lado_b))
    ancho_px = float(min(lado_a, lado_b))
    area_px = abs(float(cv2.contourArea(puntos)))
    # El angulo se deriva del lado mas largo de la caja y no del que reporta
    # minAreaRect: esa convencion cambio entre versiones de OpenCV.
    angulo_deg = _angulo_del_lado_mayor(esquinas)
    caja_rotada_px = np.int32(np.round(esquinas))

    return _armar_medida(
        largo_px=largo_px,
        ancho_px=ancho_px,
        area_px=area_px,
        angulo_deg=angulo_deg,
        caja_rotada_px=caja_rotada_px,
        centro_px=(float(centro_x), float(centro_y)),
        escala=escala,
    )


def medir_desde_mascara(mascara: np.ndarray, escala: Escala) -> Optional[Medida]:
    """Mide la region mas grande de una mascara. Devuelve None si la mascara esta vacia."""
    contorno = contorno_principal(mascara)
    if contorno is None:
        return None
    return medir_contorno(contorno, escala)


def medir_desde_caja(xyxy: Sequence[float], escala: Escala) -> Medida:
    """Mide a partir de la caja recta de deteccion (sin mascara).

    Sin contorno no hay forma real: el area es la de la caja recta y el angulo es 0.
    La medida es una cota superior del tamano de la pieza.
    """
    x1, y1, x2, y2 = (float(valor) for valor in tuple(xyxy)[:4])
    izquierda, derecha = min(x1, x2), max(x1, x2)
    arriba, abajo = min(y1, y2), max(y1, y2)

    ancho_caja = derecha - izquierda
    alto_caja = abajo - arriba
    esquinas = np.array(
        [
            [izquierda, arriba],
            [derecha, arriba],
            [derecha, abajo],
            [izquierda, abajo],
        ],
        dtype=np.float32,
    )

    return _armar_medida(
        largo_px=float(max(ancho_caja, alto_caja)),
        ancho_px=float(min(ancho_caja, alto_caja)),
        area_px=float(ancho_caja * alto_caja),
        angulo_deg=0.0,
        caja_rotada_px=np.int32(np.round(esquinas)),
        centro_px=((izquierda + derecha) / 2.0, (arriba + abajo) / 2.0),
        escala=escala,
    )


def _binarizar(mascara: Optional[np.ndarray]) -> Optional[np.ndarray]:
    """Lleva cualquier mascara a uint8 0/255 de un solo canal, o None si no es usable."""
    if mascara is None:
        return None
    arreglo = np.asarray(mascara)
    if arreglo.size == 0:
        return None
    if arreglo.ndim == 3:
        if arreglo.shape[2] == 1:
            arreglo = arreglo[:, :, 0]
        else:
            arreglo = cv2.cvtColor(arreglo.astype(np.uint8), cv2.COLOR_BGR2GRAY)
    elif arreglo.ndim != 2:
        return None
    return np.where(arreglo > 0, np.uint8(255), np.uint8(0))


def _puntos_de_contorno(contorno: Optional[np.ndarray]) -> Optional[np.ndarray]:
    """Normaliza un contorno de OpenCV (N,1,2) a un arreglo float32 (N,2)."""
    if contorno is None:
        return None
    arreglo = np.asarray(contorno, dtype=np.float32)
    if arreglo.size == 0:
        return np.zeros((0, 2), dtype=np.float32)
    if arreglo.size % 2 != 0:
        return None
    return arreglo.reshape(-1, 2)


def _normalizar_angulo(grados: float) -> float:
    """Lleva cualquier angulo a [0,180): una pieza y la misma girada 180 grados miden igual."""
    angulo = math.fmod(float(grados), 180.0)
    if angulo < 0.0:
        angulo += 180.0
    # fmod de un negativo muy chico puede devolver exactamente 180.0 tras la suma.
    if angulo >= 180.0:
        angulo = 0.0
    return angulo


def _angulo_del_lado_mayor(esquinas: np.ndarray) -> float:
    """Angulo del lado mas largo de la caja, en grados [0,180).

    Medido en coordenadas de imagen (eje Y hacia abajo), igual que lo dibuja el overlay.
    """
    puntos = np.asarray(esquinas, dtype=np.float64).reshape(-1, 2)
    if puntos.shape[0] < 2:
        return 0.0

    mejor_largo = -1.0
    mejor_angulo = 0.0
    for indice in range(puntos.shape[0]):
        origen = puntos[indice]
        destino = puntos[(indice + 1) % puntos.shape[0]]
        delta_x = float(destino[0] - origen[0])
        delta_y = float(destino[1] - origen[1])
        largo = math.hypot(delta_x, delta_y)
        if largo > mejor_largo:
            mejor_largo = largo
            mejor_angulo = math.degrees(math.atan2(delta_y, delta_x))
    return _normalizar_angulo(mejor_angulo)


def _medida_degradada(puntos: Optional[np.ndarray], escala: Escala) -> Medida:
    """Medida coherente para contornos con menos de 3 puntos (no forman area)."""
    if puntos is None or puntos.shape[0] == 0:
        return _armar_medida(
            largo_px=0.0,
            ancho_px=0.0,
            area_px=0.0,
            angulo_deg=0.0,
            caja_rotada_px=np.zeros((4, 2), dtype=np.int32),
            centro_px=(0.0, 0.0),
            escala=escala,
        )

    if puntos.shape[0] == 1:
        unico = puntos[0]
        caja = np.int32(np.round(np.repeat(unico.reshape(1, 2), 4, axis=0)))
        return _armar_medida(
            largo_px=0.0,
            ancho_px=0.0,
            area_px=0.0,
            angulo_deg=0.0,
            caja_rotada_px=caja,
            centro_px=(float(unico[0]), float(unico[1])),
            escala=escala,
        )

    origen, destino = puntos[0], puntos[1]
    delta_x = float(destino[0] - origen[0])
    delta_y = float(destino[1] - origen[1])
    caja = np.int32(np.round(np.array([origen, destino, destino, origen], dtype=np.float32)))
    return _armar_medida(
        largo_px=math.hypot(delta_x, delta_y),
        ancho_px=0.0,
        area_px=0.0,
        angulo_deg=_normalizar_angulo(math.degrees(math.atan2(delta_y, delta_x))),
        caja_rotada_px=caja,
        centro_px=((float(origen[0]) + float(destino[0])) / 2.0,
                   (float(origen[1]) + float(destino[1])) / 2.0),
        escala=escala,
    )


def _armar_medida(
    largo_px: float,
    ancho_px: float,
    area_px: float,
    angulo_deg: float,
    caja_rotada_px: np.ndarray,
    centro_px: tuple[float, float],
    escala: Escala,
) -> Medida:
    """Crea la Medida y le agrega los milimetros solo si hay escala confiable."""
    medida = Medida(
        largo_px=float(largo_px),
        ancho_px=float(ancho_px),
        area_px=float(area_px),
        angulo_deg=float(angulo_deg),
        caja_rotada_px=caja_rotada_px,
        centro_px=centro_px,
    )

    if escala is None or not escala.valida or not escala.mm_por_px:
        return medida

    factor = float(escala.mm_por_px)
    medida.largo_mm = medida.largo_px * factor
    medida.ancho_mm = medida.ancho_px * factor
    medida.area_mm2 = medida.area_px * factor * factor
    medida.fiable = True
    return medida
