"""Anotar los dientes SOLO, mientras el operario pasa la sierra.

COMO TRABAJA EL TALLER
----------------------
Con una orden abierta hay que revisar cada diente de cada sierra. Pararse a apretar una
tecla por diente son veinticuatro pulsaciones por sierra, y ademas obliga a soltar la pieza.
Con esto el operario mueve la sierra y el programa anota solo: mira cinco fotos por
segundo --una cada 0.2 s, interfaz.segundos_entre_fotos-- las juzga, las tira, y cuando
CINCO seguidas son la misma foto con el mismo veredicto da ese diente por revisado y pasa
al siguiente. O sea que el operario tiene que apoyar cada diente UN SEGUNDO.

POR QUE CINCO IGUALES Y NO UNA
------------------------------
Una sola foto puede agarrar el momento en que la pieza se esta acomodando, el reflejo de
una lampara o la mano del operario cruzando. Cinco seguidas que dan LO MISMO quieren decir
dos cosas a la vez: que la pieza esta quieta y que el veredicto es estable. Cualquiera de
las dos sola no alcanza.

DESPUES DE ANOTAR, HAY QUE MOVER LA PIEZA
------------------------------------------
Esto es lo que evita el peor error posible: que una sierra quieta sobre la mesa se anote
veinticuatro veces sola y el informe diga que se reviso entera sin que nadie la haya
tocado. Una vez anotado un diente, no se anota nada mas hasta que la imagen CAMBIE de
verdad. Los dos cortes --"es la misma foto" y "es otro diente"-- estan medidos sobre las
14 fotos reales del taller llevadas a 640x480, con la firma que devuelve huella():

    solo ruido del sensor (+-2 niveles)           0.0015 - 0.0024
    la misma toma con otra exposicion             0.0000   (la firma va normalizada)
    la misma toma corrida  1 px                   0.0065 - 0.0113
    la misma toma corrida  2 px                   0.0132 - 0.0215
    la misma toma corrida  5 px                   0.0315 - 0.0525
    dos fotos DISTINTAS, el par mas parecido      0.0855
    dos fotos DISTINTAS, los 91 pares             0.0855 - 0.9476

HASTA DONDE LLEGA ESTO, MEDIDO. Apoyando seis dientes un segundo y medio cada uno, con el
temblor de una mano que sostiene la sierra, anota los seis con temblores de hasta 2 px y
ninguno con 4 px: hay que apoyar, no se puede ir pasando la sierra de largo. Y al reves:
una pieza que se desliza PAREJO y no se detiene nunca igual se anota, porque en una
ventana de un segundo se movio menos que el corte. Medido, 20 segundos de arrastre sin una
sola parada: a 1 px/s anota 3 dientes, a 5 px/s anota 8, a 10 px/s ninguno. Son todos el
mismo diente. El programa no puede distinguir "quieta" de "moviendose 4 px por segundo", y
lo que lo salva en el taller es que pasar al diente siguiente es un salto grande (0.0855
para arriba) y no un arrastre.

LAS FOTOS NO SE GUARDAN. Se miran y se tiran: lo que queda es el RESULTADO en la orden.
Juntar fotos para que el programa aprenda es otra cosa, y se sigue haciendo a mano con las
teclas, eligiendo en que concepto va cada una.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

# CUANTO SE PUEDE MOVER LA PIEZA ENTRE DOS FOTOS Y SEGUIR SIENDO EL MISMO DIENTE.
#
# Ya no es el mismo numero que usa almacen.AlmacenClasificacion para no guardar dos veces
# la misma foto, y la diferencia es a proposito: alla la pregunta es "este archivo ya
# esta", y conviene ser estricto. Aca la pregunta es otra, "el operario me sigue mostrando
# el mismo diente", y con la sierra en la mano eso nunca es pixel por pixel.
#
# Medido sobre las 14 fotos reales del taller llevadas a 640x480 (medir_tolerancia.py),
# con la firma suavizada que usa huella():
#
#     la misma toma corrida  1 px   0.0065 - 0.0113
#     la misma toma corrida  2 px   0.0132 - 0.0215
#     la misma toma corrida  3 px   0.0190 - 0.0317
#     la misma toma corrida  5 px   0.0315 - 0.0525
#     dos fotos DISTINTAS           0.0855 - 0.9476   (91 pares, el mas parecido 0.0855)
#
# El corte va en 0.04: aguanta unos 4 px de corrimiento --el temblor de una mano que
# sostiene la sierra-- y sigue estando a mas del doble de distancia del par de dientes
# distintos mas parecido que hay en la coleccion. Con el 0.015 de antes, medido en el
# bucle: apoyando seis dientes con un temblor de 0.5 px se anotaba UNO, y con 1 px ninguno.
DISTANCIA_MISMA_FOTO = 0.04

# Y CUANTO TIENE QUE CAMBIAR LA IMAGEN PARA QUE SEA OTRO DIENTE. No es el mismo numero que
# el de arriba, y tiene que ser mas grande: si fueran iguales, una pieza que se desliza
# despacio se aleja de a poco de donde quedo anotada y en cuanto pasa el corte el programa
# la toma por un diente nuevo --el mismo diente, cuatro pixeles mas alla-- y lo anota de
# nuevo. Medido en el bucle, 20 segundos de arrastre parejo con un solo corte de 0.04: a
# 5 px/s anotaba 11 dientes sin que la pieza se detuviera nunca.
#
# El valor sale de los 91 pares de fotos distintas de la coleccion del taller: el par mas
# parecido que hay --dos dientes afilados de la misma sierra-- da 0.0855, y NINGUNO de los
# 91 baja de 0.08. Con 0.07 todos los pares de dientes distintos lo pasan con margen, y en
# cambio la misma toma necesita correrse 6 u 8 px para pasarlo.
#
# EL FRENO NO SE PUEDE TRABAR, y esto es lo que mas habia que comprobar: si pasar al diente
# siguiente cambiara MENOS que este corte, el programa se quedaria clavado en el primer
# diente y no anotaria ninguno mas, que es peor que el defecto que el freno vino a tapar.
# Las 14 fotos del taller estan sacadas una tras otra, con 9 a 25 segundos entre ellas, que
# es justo lo que hace el operario: avanza la sierra un diente y saca otra. Midiendo cada
# par CONSECUTIVO --el paso de un diente al siguiente, de verdad--:
#
#     el paso mas chico de todos                    0.1048
#     mediana de los 13 pasos                       0.4527
#     pasos que NO soltarian el freno                0 de 13
#
# El paso mas chico esta a una vez y media del corte. Hay margen.
CAMBIO_DE_DIENTE = 0.07


@dataclass
class Resultado:
    """El veredicto de un diente, tal como lo va a anotar la orden."""

    estado: Optional[str] = None
    parecido: Optional[float] = None
    aprobado: Optional[bool] = None
    origen: str = ""

    def coincide_con(self, otro: Optional["Resultado"]) -> bool:
        """Si son el mismo veredicto. El parecido exacto no cuenta: nunca da igual dos
        veces, y lo que importa es que haya dicho lo mismo."""
        if otro is None:
            return False
        return self.estado == otro.estado and self.aprobado == otro.aprobado


def huella(frame: np.ndarray) -> Optional[np.ndarray]:
    """Firma chica de una imagen, sin brillo ni contraste propios y sin el filo de los bordes.

    EL SUAVIZADO NO ES ADORNO. Sin el, la firma mira los bordes tan de cerca que un
    corrimiento de un pixel la cambia tanto como cambiarla de diente: medido sobre las 14
    fotos del taller, la misma toma corrida 1 px daba 0.0148-0.0256, o sea por encima del
    corte. Desenfocando la firma antes de compararla, esa misma toma corrida 1 px da
    0.0065-0.0113 y el par de fotos DISTINTAS mas parecido sigue en 0.0855: lo que se
    pierde es sensibilidad al pixel, no la capacidad de distinguir un diente de otro.

    La normalizacion por media y desvio hace lo propio con la luz: un cambio de exposicion
    --el tubo fluorescente del taller, que parpadea-- no mueve la firma.
    """
    if frame is None or frame.size == 0:
        return None
    try:
        gris = frame if frame.ndim == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        chica = cv2.resize(gris, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32)
        chica = (chica - chica.mean()) / (chica.std() + 1e-6)
        return cv2.GaussianBlur(chica, (0, 0), 1.2)
    except cv2.error:
        return None


def distancia(a: Optional[np.ndarray], b: Optional[np.ndarray]) -> float:
    """Cuanto se diferencian dos huellas. 0 = la misma foto."""
    if a is None or b is None:
        return 1.0
    return float(np.abs(a - b).mean())


class CapturaAutomatica:
    """La maquina de estados que decide cuando un diente esta revisado."""

    def __init__(
        self,
        segundos_entre_fotos: float = 1.0,
        iguales_para_anotar: int = 5,
        distancia_misma_foto: float = DISTANCIA_MISMA_FOTO,
        cambio_de_diente: float = CAMBIO_DE_DIENTE,
    ) -> None:
        self._cada = max(0.05, float(segundos_entre_fotos))
        self._necesarias = max(1, int(iguales_para_anotar))
        self._corte = float(distancia_misma_foto)
        # El de soltar el freno nunca puede ser mas chico que el de la racha.
        self._cambio = max(float(cambio_de_diente), self._corte)
        self._proxima = 0.0
        self._huella: Optional[np.ndarray] = None
        self._resultado: Optional[Resultado] = None
        self._iguales = 0
        self._anotada: Optional[np.ndarray] = None   # la del ultimo diente anotado
        self._sobre_lo_anotado = False               # lo que se ve AHORA es ese diente

    # ------------------------------------------------------------------ lectura
    @property
    def iguales(self) -> int:
        """Cuantas fotos seguidas vienen dando lo mismo."""
        return self._iguales

    @property
    def necesarias(self) -> int:
        return self._necesarias

    @property
    def esperando_cambio(self) -> bool:
        """True si lo que se esta viendo AHORA es el diente que ya quedo anotado.

        Se mira la foto de este momento y no solo "si alguna vez anote": con lo segundo,
        el cartel se quedaba diciendo "pase al siguiente" mientras ya estaba confirmando
        otro diente.
        """
        return self._sobre_lo_anotado

    def progreso(self) -> str:
        """Una linea para la pantalla: en que esta."""
        if self.esperando_cambio:
            return "diente anotado - pase al siguiente"
        if self._iguales == 0:
            return "mirando"
        return f"confirmando: {self._iguales} de {self._necesarias}"

    # ------------------------------------------------------------------ el paso
    def reiniciar(self) -> None:
        """Olvida la RACHA. NO suelta el freno del diente ya anotado.

        Son dos cosas distintas y mezclarlas costo caro: mientras esto borraba tambien el
        freno, cada congelada, cada apertura del panel y cada anotacion a mano habilitaba
        que la MISMA pieza quieta se anotara de nuevo. Medido en el bucle de verdad con la
        sierra sin tocar: una pulsacion de barra espaciadora cada dos segundos dejaba seis
        renglones en el informe para un solo diente.
        """
        self._huella = None
        self._resultado = None
        self._iguales = 0

    def ya_anotado(self, frame: Optional[np.ndarray]) -> None:
        """Avisa que este diente quedo anotado por otro camino (a mano, o al archivarlo).

        Sin esto, el operario que aprieta 'a' o 'g' deja el diente anotado Y habilitado:
        un segundo despues la captura automatica cerraba su racha sobre la misma foto y lo
        anotaba otra vez, con dos renglones para un solo diente.
        """
        marca = huella(frame)
        if marca is not None:
            self._anotada = marca
            self._sobre_lo_anotado = True
        self.reiniciar()

    def es_lo_ya_anotado(self, frame: Optional[np.ndarray]) -> bool:
        """Si esta foto es el diente que ya quedo anotado. La pregunta el camino de a mano.

        Con la imagen CONGELADA el bucle no llama a mirar() --no hay cuadro nuevo que
        mirar-- asi que el freno de aca no se entera, y cada pulsacion de 'g' gastaba un
        numero de diente sobre LA MISMA foto. Medido en el bucle de verdad con una foto
        real del taller congelada y diez 'g': diez dientes anotados, los diez con el mismo
        parecido (0.9997) y el mismo instante. El informe decia diez dientes revisados y la
        foto era una.
        """
        if self._anotada is None:
            return False
        return distancia(huella(frame), self._anotada) < self._cambio

    def mirar(
        self, frame: Optional[np.ndarray], ahora: float, resultado: Optional[Resultado]
    ) -> Optional[Resultado]:
        """Una foto mas. Devuelve el resultado SOLO cuando el diente queda confirmado.

        `ahora` viene de afuera --time.monotonic()-- para que la cadencia se pueda probar
        sin esperar segundos de verdad.
        """
        if ahora < self._proxima:
            return None
        self._proxima = ahora + self._cada

        actual = huella(frame)
        if actual is None:
            return None

        # EL FRENO NO SE SUELTA CON UNA FOTO DISTINTA, se reemplaza recien cuando queda
        # confirmado OTRO diente. Antes bastaba una sola foto que se apartara --una mano
        # cruzando, el tubo fluorescente parpadeando, una vibracion-- para soltarlo, y
        # despues las cinco fotos quietas que cerraban la racha eran las del MISMO diente
        # ya anotado, sin que nadie las comparara contra el. Medido en el bucle de verdad
        # con la sierra sin tocar: una mano cruzando cada dos segundos durante cuarenta
        # segundos dejaba los 12 dientes de la sierra anotados, los 12 con el mismo
        # parecido, y el informe decia "12 revisados, 12 para rechazar". Ese es el papel
        # con el que se contesta un reclamo.
        #
        # Ahora, mientras lo que se ve siga pareciendose al diente anotado, no se cuenta
        # nada. Para anotar el siguiente hacen falta cinco fotos iguales entre si Y
        # distintas de la anterior, que es lo que de verdad significa "paso al siguiente".
        self._sobre_lo_anotado = (
            self._anotada is not None and distancia(actual, self._anotada) < self._cambio
        )
        if self._sobre_lo_anotado:
            self._huella, self._resultado, self._iguales = None, None, 0
            return None

        if resultado is None:
            # Sin veredicto no hay nada que confirmar; se descarta la racha para no
            # mezclar lo de antes con lo de despues del hueco.
            self._huella, self._resultado, self._iguales = None, None, 0
            return None

        # SE COMPARA CONTRA LA PRIMERA DE LA RACHA, NO CONTRA LA DE RECIEN. Encadenando
        # --cada foto contra la anterior-- una pieza que se desliza despacio y no se
        # detiene nunca cierra la racha igual, porque entre foto y foto se movio poco.
        # Medido en el bucle con la foto real del taller, 20 segundos de arrastre parejo:
        # a 1 px/s anotaba 12 dientes, a 2 px/s 15 y a 3 px/s 17, sin que la pieza se
        # hubiera parado una sola vez. Con el ancla, las cinco fotos tienen que estar cerca
        # de DONDE EMPEZO la racha, que es lo que de verdad quiere decir "la pieza esta
        # quieta": el recorrido total queda acotado por el corte, no el paso a paso.
        misma_foto = (
            self._huella is not None and distancia(actual, self._huella) < self._corte
        )
        if misma_foto and resultado.coincide_con(self._resultado):
            self._iguales += 1
        else:
            # Arranca una racha nueva: esta foto pasa a ser el ancla.
            self._iguales = 1
            self._huella = actual
        self._resultado = resultado

        if self._iguales >= self._necesarias:
            self._anotada = actual
            self._sobre_lo_anotado = True
            self._iguales = 0
            return resultado
        return None
