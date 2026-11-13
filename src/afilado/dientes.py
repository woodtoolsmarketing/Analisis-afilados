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
las tienen.

ESTA ES LA TABLA QUE SE REPRODUCE, medida con la mascara que usa el propio codigo --el
10% mas brillante, cerrado y abierto, y pasado por los filtros de area y de lado-- sobre
las 3 fotos de la vista de lejos:

    modelo_20261112_161319    2 candidatos    la cara 32.9    el hueco 24.4
    modelo_20261112_161340    1 candidato     28.3 (una sola mancha que junta cara y hueco)
    modelo_20261112_161523    1 candidato     42.5

Habia aca una tabla de tres columnas --cara 32.7/39.9/44.8 contra hueco 24.2/29.7/27.5--
que NO se reproduce con esta mascara: de las tres columnas solo coincide la primera. Se la
saco en vez de dejarla: un numero que no se puede volver a sacar es peor que ninguno. En
dos de las tres fotos no hay siquiera dos candidatos que comparar.

Se prueba primero el camino de siempre, y si devuelve un solo recuadro del tamano de la
foto --senal de que junto el diente con el disco-- se pasa al de la cara rayada.

EL "SEGUNDO" CAMINO ES EL QUE MAS TRABAJA, y conviene saberlo antes de tocar nada. Medido
sobre las 14 fotos que saco el taller con el microscopio (las de data/coleccion/afilado
del Escritorio, llevadas a 640x480), contando cual de los dos produce el recuadro final:

    el de la cara rayada    11 de 14
    otsu por brillo          3 de 14

O sea que la tabla de mas abajo --"otsu por brillo 19 de 19 <-- el que quedo"-- describe
otra epoca: esas 19 fotos eran de la camara anterior y de un encuadre distinto. Hoy, con
este microscopio, quien decide casi siempre es el camino que esta medido sobre TRES fotos.
Alcanza para que el recuadro caiga sobre el diente en vez del hueco --se ve en pantalla--
pero no para darlo por cerrado: con diez o quince fotos mas de esta vista hay que volver a
correr la medicion y ajustar, empezando por _PERCENTIL_CARA y _RELACION_SEGUNDO.

COMO SE SEPARAN, SIN MODELO ENTRENADO
--------------------------------------
No hay red que detecte dientes --la base de fotos se esta juntando recien-- asi que la
separacion sale de la imagen misma. Se probaron cinco caminos sobre las 19 fotos reales del
microscopio, y sobre fotos armadas pegando dos y tres dientes con un hueco en el medio, que
es lo que se ve cuando entran varios:

    metodo            1 diente da 1     2 dientes dan 2
    otsu por brillo      19 de 19          16 de 19      <-- el que gano ENTONCES
    otsu invertido       19 de 19           3 de 19
    por textura          17 de 19           2 de 19
    umbral adaptativo     3 de 19           2 de 19
    por columnas          8 de 19           3 de 19

Ganaba el mas simple: la cara iluminada del diente es lo mas claro de la imagen, Otsu
encuentra ese corte solo --sin ningun numero puesto a mano-- y lo que queda arriba del corte
es el diente. Con tres dientes devuelve los tres 13 de 19 veces, y 2 las otras 6.

Esas 19 fotos son de la camara y del encuadre de antes. Con el microscopio de hoy, Otsu
decide 3 de 14 (ver arriba): no esta mal, pero ya no es "el que quedo".

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

import logging
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

_log = logging.getLogger(__name__)

# Un pedazo mas chico que esto no es un diente: es una veta, un reflejo o una mota de
# aserrin.
#
# ESTE VALOR ESTA JUSTO EN EL BORDE y hay que saberlo. El 15%-47% que decia antes era con
# la camara anterior, que miraba el diente mucho mas de cerca. Medido sobre las 14 fotos
# del microscopio de hoy a 640x480, la cara del diente ocupa entre 4.1% y 6.4% del cuadro:
# las dos mas chicas dan 0.041, a UNA milesima del corte. Un encuadre un poco mas lejos y
# ese diente desaparece; ahi el programa no se rompe --vuelve a "el diente es la foto
# entera"-- pero deja de mirar el diente.
#
# Bajar el corte NO es gratis, medido sobre las mismas 14 fotos (recuadros por foto):
#
#     0.040   1 a 2   (media 1.07)   <-- el de hoy
#     0.030   1 a 2   (media 1.36)
#     0.025   1 a 3   (media 1.57)
#     0.015   1 a 3   (media 1.64)
#
# Cada recuadro de mas es un diente que el programa se inventa, y en este proyecto eso es
# peor que perder uno: el que se pierde lo ve el operario en pantalla, el que se inventa
# termina como un renglon de mas en el informe de un reclamo. Por eso queda en 0.04. Si el
# taller cambia el encuadre y empiezan a aparecer fotos sin separar, hay que volver a
# correr esta medicion con las fotos nuevas --no bajarlo a ojo.
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
    rayado: float = 0.0                  # cuanto se parece a una cara de widia; ver abajo

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
            # LA CAJA NO DEJA NI UN PIXEL: o no toca la imagen, o viene degenerada
            # --x2 == x1-- aunque caiga adentro. Devolver la foto entera callado es
            # archivar el cuadro completo como si fuera un diente. Con las cajas que arma
            # _componentes no pasa --el filtro de LADO_MINIMO ya exige 57 px de lado en
            # un cuadro de 640x480-- y sobre las 36 fotos reales por 9 inclinaciones no
            # se disparo nunca; queda el aviso para el dia que la caja venga de un cuadro
            # de otro tamano, que es lo unico que puede producirlo.
            _log.warning("recuadro fuera de la imagen: caja=%s imagen=%sx%s",
                         self.caja, ancho, alto)
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
# diente. Sirve para quedarse con el segundo diente cuando hay dos.
#
# NO DEJA AFUERA AL HUECO, aunque aca decia que si. Medido sobre las 36 fotos reales que
# hay --14 de cerca, 3 de lejos y 19 de la camara anterior--, solo 3 dan mas de un
# candidato, y el segundo puntua, respecto del mejor:
#
#     afilado_20261112_161626_660   0.86   el hueco, y ENTRA
#     modelo_20261112_161319        0.34   el hueco, y queda afuera
#     modelo_20261112_081056        0.25
#
# Y lo que lo deja afuera no es el rayado: por rayado solo esos segundos dan 0.70, 0.74 y
# 0.76 del mejor, los tres por encima de 0.55. Es el factor TAMANO, que en la vista de
# lejos los hunde a 0.34 y 0.25 y en la de cerca los SUBE (0.70 -> 0.86). Por eso no se
# toca el factor tamano sin volver a medir la vista de lejos, y por eso que el recuadro
# que se archiva no sea el hueco lo resuelve _medir_rayado y no este numero.
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
            por_cara = _por_cara_rayada(gris, (alto, ancho))
            # Y AL RESULTADO SE LE PREGUNTA LO MISMO. El segundo camino tambien puede
            # devolver un solo recuadro del tamano de la foto, y entonces no separo nada
            # el tampoco. Medido sobre las 36 fotos reales por 9 inclinaciones, pasa en 9
            # casos, todos de las 19 de la camara anterior: modelo_20261110_161627
            # devuelve (28,17,572,364), el 61% del cuadro, y lo mismo _164925 y _164931;
            # con las 17 de hoy no pasa en ningun angulo. revision.py no dibuja un
            # recuadro de 60% o mas, asi que el operario no ve nada en pantalla mientras
            # el programa archiva y juzga ese recorte como si fuera un diente.
            if por_cara and not _no_separo(por_cara, alto * ancho):
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
    if len(encontrados) > 1:
        _medir_rayado(gris, encontrados)
    return encontrados


def _medir_rayado(gris: np.ndarray, encontrados: list[Diente]) -> None:
    """Le pone a cada recuadro cuanto se parece a una cara de widia.

    POR QUE HACE FALTA: cuando salen varios recuadros, el primero de izquierda a derecha
    no es necesariamente el diente. En las fotos del taller el HUECO entre dos dientes
    --lustrado, brillante- pasa el filtro de area igual que la cara de widia, y si cae a
    la izquierda queda primero: es el que se archiva y el que va al informe. Pasa en 1 de
    las 14 fotos de cerca que mando el taller (afilado_20261112_161626_660).

    Lo que los separa es el rayado que deja el esmeril en la placa, que el hueco no tiene.
    Medido sobre esa foto de cerca: el hueco da 30.4 y la cara del diente 43.6. Y en la
    unica foto de la vista de lejos donde hay dos candidatos que comparar
    (modelo_20261112_161319): el hueco 24.4 y la cara 32.9. DOS casos, los dos en el
    mismo sentido. No alcanza para poner un corte fijo --por eso el numero no decide NADA
    por si solo, solo ordena-- pero si para elegir cual de los recuadros se mira primero.

    No se calcula cuando hay un solo recuadro: ahi no hay nada que elegir, y el Sobel con
    el desenfoque son unos milisegundos que en la PC del taller se notan.

    ESTE SOBEL ES SOLO EN X, asi que el puntaje se mueve con la inclinacion de la pieza.
    Antes de cambiarlo por la magnitud del gradiente, leer la medicion que esta en
    _por_cara_rayada: sobre las 17 fotos reales por 9 inclinaciones, la magnitud manda el
    recuadro al hueco en 6 casos y la derivada en x de hoy en ninguno.
    """
    try:
        mapa = cv2.GaussianBlur(
            np.abs(cv2.Sobel(gris, cv2.CV_32F, 1, 0, ksize=3)), (0, 0), 5
        )
    except cv2.error:
        return
    for diente in encontrados:
        x1, y1, x2, y2 = diente.caja
        pedazo = mapa[y1:y2, x1:x2]
        diente.rayado = float(pedazo.mean()) if pedazo.size else 0.0


def mas_parecido_a_diente(encontrados: list[Diente]) -> int:
    """Cual de los recuadros mirar primero: el mas rayado, que es el que tiene cara de
    widia. Con uno solo, ese. Ver _medir_rayado para los numeros."""
    if not encontrados:
        return 0
    return max(range(len(encontrados)), key=lambda i: encontrados[i].rayado)


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
    """True cuando el corte por brillo no separo nada, que son DOS casos distintos: o
    volvio UN solo recuadro del tamano de la foto --el umbral junto el diente con el
    disco-- o no volvio NINGUNO --el umbral no dejo nada con forma de diente--. Los dos
    mandan al segundo camino, y por eso los dos devuelven True.

    El caso vacio no se ve con fotos del microscopio: sobre las 36 reales _por_brillo
    devolvio siempre exactamente un recuadro. Se recorre con imagen lisa o con ruido, que
    es lo que hacen las pruebas de test_dientes.py.
    """
    if len(encontrados) != 1:
        return not encontrados
    x1, y1, x2, y2 = encontrados[0].caja
    return (x2 - x1) * (y2 - y1) >= _AREA_SIN_SEPARAR * total


def _por_cara_rayada(gris: np.ndarray, forma: tuple[int, int]) -> list[Diente]:
    """El segundo camino: la cara de widia, que es lo mas claro Y lo mas rayado.

    El rayado son las lineas que deja el esmeril en la placa: el disco es granulado parejo
    y el hueco esta lustrado, ninguno de los dos las tiene. Se puntua cada mancha por
    tamano x rayado y se queda con las que acompanan a la mejor.

    CUANTO SEPARA ESO, re-medido con esta misma mascara sobre las 3 fotos de la vista de
    lejos y contando solo las manchas que sobreviven al filtro de area y de lado:

        modelo_20261112_161319    2 candidatos    la cara 32.9    el hueco 24.4
        modelo_20261112_161340    1 candidato     28.3 (una mancha que junta cara y hueco)
        modelo_20261112_161523    1 candidato     42.5

    O sea que en 2 de las 3 no hay nada que separar: queda un solo candidato y se lo lleva
    mire el rayado que mire. En 161523 el hueco existe --da 27.8-- pero lo descarta el
    filtro de AREA (0.035 contra AREA_MINIMA 0.04), no el rayado; el 0.037 es la cara del
    diente de al lado, la mancha de 39.8.

    OJO: esto esta medido sobre TRES fotos de esta vista. Es mejor que lo que habia --que
    ponia el recuadro sobre el hueco-- pero tres fotos no son una medicion: cuando haya
    diez o quince de la misma vista hay que volver a correr esto.

    EL RAYADO SE MIDE CON LA DERIVADA HORIZONTAL SOLA (Sobel en x), asi que depende de
    como venga inclinada la pieza, y la sierra se sostiene con la mano. Medido girando las
    17 fotos reales --las 14 de cerca y las 3 de lejos, a 640x480-- con warpAffine de -20
    a +20 grados: en modelo_20261112_161319 el rayado de la cara cae de 32.9 a 25.4 con
    +5 grados (-23%) mientras el del hueco casi no se mueve, de 24.4 a 23.3 (-4%). El
    hueco pasa de puntuar 0.34 del mejor a 0.74, entra en la lista y aparece en pantalla
    como un segundo diente. A -5 grados no pasa (0.34): la caida es asimetrica, que es
    justo la firma de medir con una sola derivada.

    SE PROBO LA ALTERNATIVA Y ES PEOR; por eso sigue el Sobel en x solo. Sobre esas 17
    fotos por 9 inclinaciones, mirando donde cae el recuadro que elige
    mas_parecido_a_diente() y tomando como diente el recuadro de esa misma foto SIN
    inclinar, girado por la misma matriz --quedan 126 casos de 153: en los otros 27 el
    centro del diente se va afuera del cuadro--, con el borde en negro / replicado:

        derivada en x (hoy)         110 / 99 en el diente    0 / 1 en el hueco
        magnitud del gradiente      104 / 95                 6 / 5
        maximo entre |Sx| y |Sy|    104 / 95                 6 / 5

    Los casos que se van al hueco con el gradiente completo son
    afilado_20261112_161626_660 y afilado_20261112_163817_643 inclinadas: el hueco es una
    curva lustrada, su brillo cambia de arriba hacia abajo, y sumarle |Sy| le levanta el
    puntaje justo donde el de hoy lo deja afuera. En esa primera foto el rayado del hueco
    contra el de la cara es 30.4/43.6 = 0.70 con la derivada en x, 0.95 con la magnitud y
    0.94 con el maximo; con la derivada en y SOLA el hueco gana (1.49). O sea que sumar
    |Sy| achica de 30% a 5% el margen del que vive este metodo.

    LO QUE SI EMPEORA CON LA INCLINACION es que este camino deje de separar: 16 de 126
    casos (26 si se rota replicando el borde) vuelven con un recuadro de 60% o mas del
    cuadro, que es el caso de _no_separo.
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
