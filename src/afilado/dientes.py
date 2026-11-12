"""Separar los DIENTES que hay en la imagen, para analizar cada uno por su cuenta.

POR QUE HACE FALTA
------------------
El operario apoya la sierra y casi siempre entra un diente, pero a veces entran dos. Hasta
ahora el programa juzgaba la foto entera: con dos dientes adentro daba UN veredicto, que no
era de ninguno de los dos. Y el informe de la orden necesita un resultado POR DIENTE.

COMO SE SEPARAN, SIN MODELO ENTRENADO
--------------------------------------
No hay red que detecte dientes --la base de fotos se esta juntando recien-- asi que la
separacion sale de la imagen misma. Se probaron cinco caminos sobre las 19 fotos reales del
microscopio, y sobre fotos armadas pegando dos y tres dientes con un hueco en el medio, que
es lo que se ve cuando entran varios:

    metodo            1 diente da 1     2 dientes dan 2
    otsu por brillo      19 de 19          16 de 19      <-- el que quedo
    otsu invertido       19 de 19           3 de 19
    por textura          17 de 19           2 de 19
    umbral adaptativo     3 de 19           2 de 19
    por columnas          8 de 19           3 de 19

Gana el mas simple: la cara iluminada del diente es lo mas claro de la imagen, Otsu
encuentra ese corte solo --sin ningun numero puesto a mano-- y lo que queda arriba del corte
es el diente. Con tres dientes devuelve los tres 13 de 19 veces, y 2 las otras 6.

Tambien se probo repetir la busqueda tapando lo ya encontrado, para rescatar un segundo
diente mas oscuro. EMPEORA: sube de 16 a 17 de 19 en el caso de dos, pero en el de UN
diente empieza a inventar un segundo (pasa de 19 a 11 de 19). Inventar un diente es peor
que perderlo, porque el informe termina con un resultado de algo que no existe.

CUANDO NO ENCUENTRA NADA, EL DIENTE ES TODA LA FOTO. Con una imagen lisa --la tapa puesta,
una pared-- Otsu no deja ningun pedazo con forma de diente, y ahi el programa se comporta
como siempre: un solo diente, la foto entera. Es el modo en el que venia funcionando, asi
que no se pierde nada por no detectar.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

# Un pedazo mas chico que esto no es un diente: es una veta, un reflejo o una mota de
# aserrin. Medido sobre las fotos del taller, la cara iluminada de un diente ocupa entre el
# 15% y el 47% del cuadro (mediana 30%).
AREA_MINIMA = 0.04
# Y uno que ocupa casi todo tampoco es un diente: es el fondo tomado por el umbral.
AREA_MAXIMA = 0.96
# Un diente no es una tira: se descarta lo que sea mucho mas fino que el cuadro.
LADO_MINIMO = 0.12
# Cuanto aire se le deja alrededor al recortar. La punta del diente suele caer en la zona
# oscura que el umbral deja afuera, y sin ese aire el recorte corta justo el filo, que es
# lo unico que importa mirar.
MARGEN_RECORTE = 0.15

_NUCLEO = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))


@dataclass
class Diente:
    """Un diente encontrado en la imagen."""

    caja: tuple[int, int, int, int]      # x1, y1, x2, y2 en pixeles del frame
    area: float                          # que fraccion del frame ocupa
    entero: bool = False                 # True si es "no encontre nada, es toda la foto"

    @property
    def centro(self) -> tuple[int, int]:
        x1, y1, x2, y2 = self.caja
        return ((x1 + x2) // 2, (y1 + y2) // 2)

    def recortar(self, frame: np.ndarray, margen: float = MARGEN_RECORTE) -> np.ndarray:
        """El pedazo de imagen de este diente, con un poco de aire alrededor."""
        alto, ancho = frame.shape[:2]
        x1, y1, x2, y2 = self.caja
        dx = int((x2 - x1) * margen)
        dy = int((y2 - y1) * margen)
        x1 = max(0, x1 - dx)
        y1 = max(0, y1 - dy)
        x2 = min(ancho, x2 + dx)
        y2 = min(alto, y2 + dy)
        if x2 <= x1 or y2 <= y1:
            return frame
        return frame[y1:y2, x1:x2]


def detectar(frame: Optional[np.ndarray], maximo: int = 4) -> list[Diente]:
    """Los dientes que hay en la imagen, de izquierda a derecha.

    Nunca devuelve una lista vacia con una imagen valida: si no encuentra nada, devuelve un
    solo diente que es la foto entera. El programa tiene que poder seguir trabajando.
    """
    if frame is None or frame.size == 0:
        return []
    alto, ancho = frame.shape[:2]
    entera = Diente(caja=(0, 0, ancho, alto), area=1.0, entero=True)
    try:
        gris = frame if frame.ndim == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gris = cv2.GaussianBlur(gris, (7, 7), 0)
        _, mascara = cv2.threshold(gris, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        mascara = cv2.morphologyEx(mascara, cv2.MORPH_CLOSE, _NUCLEO)
        mascara = cv2.morphologyEx(mascara, cv2.MORPH_OPEN, _NUCLEO)
        encontrados = _componentes(mascara, (alto, ancho))
    except cv2.error:
        return [entera]

    if not encontrados:
        return [entera]
    # Los mas grandes primero para quedarse con los mejores, y despues de izquierda a
    # derecha, que es como los va a mirar el operario y como se numeran en la pantalla.
    encontrados.sort(key=lambda d: -d.area)
    encontrados = encontrados[: max(1, int(maximo))]
    encontrados.sort(key=lambda d: d.caja[0])
    return encontrados


def _componentes(mascara: np.ndarray, forma: tuple[int, int]) -> list[Diente]:
    alto, ancho = forma
    total = float(alto * ancho)
    cantidad, _, estadisticas, _ = cv2.connectedComponentsWithStats(mascara, 8)
    dientes: list[Diente] = []
    for indice in range(1, cantidad):
        x, y, an, al, area = estadisticas[indice]
        if not (AREA_MINIMA * total <= area <= AREA_MAXIMA * total):
            continue
        if min(an, al) < LADO_MINIMO * min(alto, ancho):
            continue
        dientes.append(Diente(caja=(int(x), int(y), int(x + an), int(y + al)),
                              area=float(area) / total))
    return dientes
