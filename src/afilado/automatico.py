"""Anotar los dientes SOLO, mientras el operario pasa la sierra.

COMO TRABAJA EL TALLER
----------------------
Con una orden abierta hay que revisar cada diente de cada sierra. Pararse a apretar una
tecla por diente son veinticuatro pulsaciones por sierra, y ademas obliga a soltar la pieza.
Con esto el operario mueve la sierra y el programa anota solo: mira una foto por segundo,
la juzga, la tira, y cuando CINCO seguidas son la misma foto con el mismo veredicto da ese
diente por revisado y pasa al siguiente.

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
verdad. El corte de "cambio" es el mismo que ya usa el almacen para no guardar dos veces
la misma foto, y esta medido sobre las fotos del taller:

    archivo identico                              0.0000
    mismo frame con otra exposicion               0.0063 - 0.0074
    tomas realmente distintas de la misma pieza   0.0315 - 0.3296

LAS FOTOS NO SE GUARDAN. Se miran y se tiran: lo que queda es el RESULTADO en la orden.
Juntar fotos para que el programa aprenda es otra cosa, y se sigue haciendo a mano con las
teclas, eligiendo en que concepto va cada una.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

# El mismo corte que almacen.AlmacenClasificacion: dos fotos por debajo de esto son LA
# MISMA foto. Ver la tabla del encabezado.
DISTANCIA_MISMA_FOTO = 0.015


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
    """Firma chica de una imagen, sin brillo ni contraste propios."""
    if frame is None or frame.size == 0:
        return None
    try:
        gris = frame if frame.ndim == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        chica = cv2.resize(gris, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32)
    except cv2.error:
        return None
    return (chica - chica.mean()) / (chica.std() + 1e-6)


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
    ) -> None:
        self._cada = max(0.05, float(segundos_entre_fotos))
        self._necesarias = max(1, int(iguales_para_anotar))
        self._corte = float(distancia_misma_foto)
        self._proxima = 0.0
        self._huella: Optional[np.ndarray] = None
        self._resultado: Optional[Resultado] = None
        self._iguales = 0
        self._anotada: Optional[np.ndarray] = None   # la del ultimo diente anotado

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
        """True si ya anoto y esta esperando que el operario mueva la pieza."""
        return self._anotada is not None

    def progreso(self) -> str:
        """Una linea para la pantalla: en que esta."""
        if self.esperando_cambio:
            return "diente anotado - pase al siguiente"
        if self._iguales == 0:
            return "mirando"
        return f"confirmando: {self._iguales} de {self._necesarias}"

    # ------------------------------------------------------------------ el paso
    def reiniciar(self) -> None:
        """Olvida lo que venia contando. Se llama al congelar o al cambiar de sierra."""
        self._huella = None
        self._resultado = None
        self._iguales = 0
        self._anotada = None

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

        # Recien anotado: no se cuenta nada hasta que la pieza se mueva de verdad.
        if self._anotada is not None:
            if distancia(actual, self._anotada) < self._corte:
                return None
            self._anotada = None
            self._huella, self._resultado, self._iguales = None, None, 0

        if resultado is None:
            # Sin veredicto no hay nada que confirmar; se descarta la racha para no
            # mezclar lo de antes con lo de despues del hueco.
            self._huella, self._resultado, self._iguales = None, None, 0
            return None

        misma_foto = (
            self._huella is not None and distancia(actual, self._huella) < self._corte
        )
        if misma_foto and resultado.coincide_con(self._resultado):
            self._iguales += 1
        else:
            self._iguales = 1
        self._huella, self._resultado = actual, resultado

        if self._iguales >= self._necesarias:
            self._anotada = actual
            self._iguales = 0
            return resultado
        return None
