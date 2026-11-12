"""Separar los DIENTES que hay en la imagen, para analizar cada uno por su cuenta.

POR QUE HACE FALTA
------------------
El operario apoya la sierra y casi siempre entra un diente, pero a veces entran dos. Hasta
ahora el programa juzgaba la foto entera: con dos dientes adentro daba UN veredicto, que no
era de ninguno de los dos. Y el informe de la orden necesita un resultado POR DIENTE.

DOS VISTAS, DOS CAMINOS
-----------------------
El taller mira la sierra de dos maneras y la foto cambia por completo:

    DE CERCA   un diente llena el cuadro. Su cara iluminada ES lo mas claro de la foto,
               asi que Otsu la encuentra sola: 19 de 19 sobre las fotos del taller.
    DE LEJOS   se ven el diente, el disco y los huecos. Aca el cuerpo del diente es lo mas
               OSCURO, el disco es gris claro y granulado, y el hueco entre dientes esta
               lustrado y brilla casi tanto como la placa de widia. "Lo mas claro" deja de
               ser el diente, y el recuadro se iba al hueco: el taller lo vio en pantalla.

Para la vista de lejos se usa otra cosa: LA CARA RAYADA. La placa de widia sale del
esmeril con lineas marcadas, y ni el disco --granulado parejo-- ni el hueco --lustrado--
las tienen. Medido sobre las 3 fotos crudas de esa vista, mirando las manchas que deja el
10% mas brillante:

    la cara del diente     rayado 32.7   39.9   44.8
    el hueco iluminado     rayado 24.2   29.7   27.5
    el borde del disco     rayado 24.9   22.5    -

Se prueba primero el camino de siempre, y si devuelve un solo recuadro del tamano de la
foto --senal de que junto el diente con el disco-- se pasa al de la cara rayada.

OJO CON ESTO: el segundo camino esta medido sobre TRES fotos. Alcanza para que el recuadro
caiga sobre el diente en vez del hueco, pero no para darlo por cerrado; con diez o quince
fotos de esa vista hay que volver a correr la medicion y ajustar.

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


# Desde que tamano un recuadro deja de separar nada: si el unico que salio ocupa mas que
# esto del cuadro, el corte por brillo junto el diente con el disco y hay que probar el
# otro camino. Medido sobre las 3 fotos de la vista de lejos: el recuadro inutil ocupaba
# 77%, 79% y 81% del cuadro.
_AREA_SIN_SEPARAR = 0.60

# El corte del segundo camino. La cara de widia esta en el 10% mas brillante de la foto;
# Otsu, en cambio, corta por el medio y mete al disco en la misma bolsa que el diente.
_PERCENTIL_CARA = 90

# Cuanto puede bajar el puntaje de un recuadro respecto del mejor para seguir contando como
# diente. Sirve para quedarse con el segundo diente cuando hay dos, sin quedarse tambien
# con el reflejo del hueco, que puntua mucho mas bajo.
_RELACION_SEGUNDO = 0.55


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
        suave = cv2.GaussianBlur(gris, (7, 7), 0)
        encontrados = _por_brillo(suave, (alto, ancho))
        # SI EL CORTE POR BRILLO NO SEPARO NADA, se prueba por la CARA RAYADA. Con la
        # sierra de lejos, Otsu mete el diente y el disco en la misma bolsa y devuelve un
        # recuadro que es casi toda la foto. La cara de widia, en cambio, esta siempre en
        # el 10% mas brillante y es lo mas rayado de la imagen: sale del esmeril con
        # lineas marcadas, y ni el disco ni el hueco las tienen.
        if _no_separo(encontrados, alto * ancho):
            por_cara = _por_cara_rayada(gris, suave, (alto, ancho))
            if por_cara:
                encontrados = por_cara
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


def _por_brillo(suave: np.ndarray, forma: tuple[int, int]) -> list[Diente]:
    """El camino de siempre: Otsu busca solo el corte entre claro y oscuro.

    Es el que anda con el diente en primer plano, donde su cara iluminada ES lo mas claro
    de la foto: 19 de 19 sobre las fotos del taller.
    """
    _, mascara = cv2.threshold(suave, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mascara = cv2.morphologyEx(mascara, cv2.MORPH_CLOSE, _NUCLEO)
    mascara = cv2.morphologyEx(mascara, cv2.MORPH_OPEN, _NUCLEO)
    return _componentes(mascara, forma)


def _no_separo(encontrados: list[Diente], total: int) -> bool:
    """True si lo unico que salio es un recuadro del tamano de la foto."""
    if len(encontrados) != 1:
        return not encontrados
    x1, y1, x2, y2 = encontrados[0].caja
    return (x2 - x1) * (y2 - y1) >= _AREA_SIN_SEPARAR * total


def _por_cara_rayada(
    gris: np.ndarray, suave: np.ndarray, forma: tuple[int, int]
) -> list[Diente]:
    """El segundo camino: la cara de widia, que es lo mas claro Y lo mas rayado.

    Medido sobre las 3 fotos crudas de la vista de lejos que mando el taller, mirando las
    manchas que deja el 10% mas brillante:

        la cara del diente     rayado 32.7   39.9   44.8
        el hueco iluminado     rayado 24.2   29.7   27.5
        el borde del disco     rayado 24.9   22.5    -

    El rayado son las lineas que deja el esmeril en la placa: el disco es granulado parejo
    y el hueco esta lustrado, ninguno de los dos las tiene. Se puntua cada mancha por
    tamano x rayado y se queda con las que acompanan a la mejor.

    OJO: esto esta medido sobre TRES fotos de esta vista. Es mejor que lo que habia --que
    ponia el recuadro sobre el hueco-- pero tres fotos no son una medicion: cuando haya
    diez o quince de la misma vista hay que volver a correr esto.
    """
    alto, ancho = forma
    total = float(alto * ancho)
    rayado = cv2.GaussianBlur(np.abs(cv2.Sobel(gris, cv2.CV_32F, 1, 0, ksize=3)), (0, 0), 5)
    corte = float(np.percentile(gris, _PERCENTIL_CARA))
    mascara = (gris > corte).astype(np.uint8) * 255
    mascara = cv2.morphologyEx(mascara, cv2.MORPH_CLOSE, _NUCLEO)
    mascara = cv2.morphologyEx(mascara, cv2.MORPH_OPEN, _NUCLEO)

    candidatos: list[tuple[float, Diente]] = []
    for diente in _componentes(mascara, forma):
        x1, y1, x2, y2 = diente.caja
        puntaje = ((x2 - x1) * (y2 - y1) / total) * float(rayado[y1:y2, x1:x2].mean())
        candidatos.append((puntaje, diente))
    if not candidatos:
        return []
    mejor = max(p for p, _ in candidatos)
    if mejor <= 0:
        return []
    return [d for p, d in candidatos if p >= _RELACION_SEGUNDO * mejor]


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
