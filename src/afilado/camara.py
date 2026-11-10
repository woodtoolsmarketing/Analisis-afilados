"""Captura de video para la inspeccion de herramientas de afilado.

Envuelve cv2.VideoCapture y resuelve las particularidades que hacen que una webcam
sirva o no para medir:

- BACKEND EN WINDOWS: CAP_DSHOW (DirectShow) es el unico backend que, en la practica,
  respeta CAP_PROP_FRAME_WIDTH/HEIGHT en la mayoria de las webcam UVC. Con CAP_ANY o
  CAP_MSMF la camara suele entregar su resolucion por defecto ignorando el pedido, y
  encima MSMF tarda varios segundos en abrir. Por eso "auto" => CAP_DSHOW en Windows.
- RESOLUCION NEGOCIADA: tras escribir CAP_PROP_FRAME_WIDTH/HEIGHT hay que RELEER las
  propiedades. La camara negocia el modo mas parecido que soporta (pedir 1280x720 puede
  devolver 640x480). El codigo reporta la resolucion REAL, no la pedida.
- ENFOQUE FIJO: la escala mm/px se calcula a partir del ArUco visto por el lente. Si el
  autofoco se mueve, cambia el campo de vision efectivo y TODAS las medidas se corren sin
  aviso. Por eso, salvo pedido explicito, se desactiva el autofoco y se fija el enfoque.
  Lo mismo vale para la exposicion cuando se la configura: el auto-exposicion persigue los
  brillos del metal pulido y hace latir la imagen.
"""

from __future__ import annotations

import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import Optional, Sequence

import cv2
import numpy as np

from .config import CamaraConfig

_log = logging.getLogger(__name__)

# Nombres de backend aceptados en CamaraConfig.backend (ademas de "auto").
_BACKENDS: dict[str, int] = {
    "any": cv2.CAP_ANY,
    "dshow": cv2.CAP_DSHOW,
    "msmf": cv2.CAP_MSMF,
    "v4l2": cv2.CAP_V4L2,
    "gstreamer": cv2.CAP_GSTREAMER,
    "ffmpeg": cv2.CAP_FFMPEG,
    "avfoundation": cv2.CAP_AVFOUNDATION,
}

_NOMBRES_BACKEND: dict[int, str] = {
    cv2.CAP_ANY: "ANY",
    cv2.CAP_DSHOW: "DSHOW",
    cv2.CAP_MSMF: "MSMF",
    cv2.CAP_V4L2: "V4L2",
    cv2.CAP_GSTREAMER: "GSTREAMER",
    cv2.CAP_FFMPEG: "FFMPEG",
    cv2.CAP_AVFOUNDATION: "AVFOUNDATION",
}

# Convencion de OpenCV para CAP_PROP_AUTO_EXPOSURE: 0.25 = manual, 0.75 = automatico.
_EXPOSICION_MANUAL = 0.25

# Backends que se prueban, EN ESTE ORDEN, cuando el config dice "auto".
#
# DSHOW va primero porque es el que mejor respeta CAP_PROP_FRAME_WIDTH/HEIGHT en las webcam
# UVC: pedirle 640x480 y que entregue 640x480. Pero hay camaras y situaciones donde DSHOW
# directamente no abre nada --una sesion de Escritorio Remoto con la camara redirigida es el
# caso tipico-- y ahi MSMF si funciona. Probar uno solo era la diferencia entre "anda" y
# "no se pudo abrir la camara", sin ninguna pista de por que.
_BACKENDS_AUTO_WINDOWS: tuple[int, ...] = (cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY)
_BACKENDS_AUTO_OTROS: tuple[int, ...] = (cv2.CAP_V4L2, cv2.CAP_ANY)

# Hasta que indice se busca cuando la fuente es "auto". Cuatro alcanza de sobra: en una PC
# con webcam integrada mas una USB, la USB suele quedar en 1 o 2. Subirlo hace mas lento el
# arranque sin ganar nada.
_MAX_INDICE_SONDEO = 3

# Cuantas veces se insiste en leer un frame antes de descartar una camara que abrio.
_INTENTOS_LECTURA_SONDEO = 8

# Segundos de respiro despues de cerrar una camara que SI se abrio, antes de tocar la
# siguiente. Sin esto el sondeo cuelga el dispositivo que acaba de probar.
_PAUSA_ENTRE_SONDEOS = 0.35


@dataclass(frozen=True)
class CamaraDetectada:
    """Una camara que se probo y ENTREGO un frame de verdad."""

    indice: int
    backend: int
    ancho: int
    alto: int

    @property
    def nombre_backend(self) -> str:
        return _NOMBRES_BACKEND.get(self.backend, str(self.backend))

    def __str__(self) -> str:
        return f"indice {self.indice} @{self.ancho}x{self.alto} ({self.nombre_backend})"


def _backends_de_sondeo() -> tuple[int, ...]:
    if sys.platform.startswith("win"):
        return _BACKENDS_AUTO_WINDOWS
    return _BACKENDS_AUTO_OTROS


def _probar(indice: int, backend: int) -> Optional[CamaraDetectada]:
    """Abre y LEE UN FRAME. Devuelve None si no sirve.

    Que isOpened() diga True no alcanza: hay drivers que abren y despues no entregan un
    solo frame --una camara tomada por otro programa es el caso mas comun--. La unica
    prueba que vale es leer.
    """
    captura = cv2.VideoCapture(indice, backend)
    abierta = captura.isOpened()
    try:
        if not abierta:
            return None
        # Estas camaras tardan varios frames en entregar el primero: leer una sola vez y
        # darla por muerta deja afuera camaras que funcionan perfecto.
        frame = None
        for _ in range(_INTENTOS_LECTURA_SONDEO):
            exito, leido = captura.read()
            if exito and leido is not None:
                frame = leido
                break
        if frame is None:
            return None
        alto, ancho = frame.shape[:2]
        return CamaraDetectada(indice=indice, backend=backend, ancho=int(ancho), alto=int(alto))
    except cv2.error:
        return None
    finally:
        captura.release()
        if abierta:
            # Abrir y cerrar seguido CUELGA a estas camaras: dejan de entregar frames en
            # todas las resoluciones hasta que pasan unos segundos. Verificado con el
            # microscopio del taller. Esta pausa es el precio de poder sondear sin romper
            # lo que se esta sondeando.
            time.sleep(_PAUSA_ENTRE_SONDEOS)


def enumerar_camaras(
    max_indice: int = _MAX_INDICE_SONDEO,
    backends: Optional[Sequence[int]] = None,
) -> list[CamaraDetectada]:
    """Devuelve las camaras que responden, probando indices y backends.

    Se recorre por INDICE y despues por backend, de modo que el indice mas bajo gana: si la
    misma camara responde por DSHOW y por MSMF, se informa una sola vez, con el backend
    preferido. Para cada indice se corta en el primer backend que anda.

    OpenCV imprime advertencias en rojo por cada intento fallido. Son normales: es el precio
    de preguntar. No indican que haya nada roto.
    """
    a_probar = tuple(backends) if backends is not None else _backends_de_sondeo()
    encontradas: list[CamaraDetectada] = []
    for indice in range(max(0, int(max_indice)) + 1):
        for backend in a_probar:
            hallada = _probar(indice, backend)
            if hallada is not None:
                encontradas.append(hallada)
                break
    return encontradas


def es_fuente_automatica(fuente: int | str) -> bool:
    """True si el config pide buscar la camara sola ("auto")."""
    return isinstance(fuente, str) and fuente.strip().lower() == "auto"


def _resolver_fuente(fuente: int | str) -> tuple[int | str, bool]:
    """Devuelve (fuente_para_VideoCapture, es_camara).

    Un str numerico ("0") se trata como indice de camara: es lo que llega desde la linea
    de comandos o desde el YAML sin comillas mal puestas. Un str que apunta a un archivo o
    carpeta existente se usa como video. Cualquier otro str se pasa tal cual (URL RTSP/HTTP).
    "auto" es un caso aparte: la fuente real se decide al abrir, sondeando.
    """
    if isinstance(fuente, int):
        return fuente, True
    texto = str(fuente).strip()
    if es_fuente_automatica(texto):
        return "auto", True
    if texto.lstrip("+-").isdigit():
        return int(texto), True
    if Path(texto).exists():
        return str(Path(texto)), False
    return texto, False


def _resolver_backend(nombre: str, es_camara: bool) -> int:
    """Traduce el nombre del backend a la constante de OpenCV."""
    clave = (nombre or "auto").strip().lower()
    if clave == "auto":
        if not es_camara:
            # Archivos y streams los resuelve mejor el backend por defecto (FFMPEG).
            return cv2.CAP_ANY
        return cv2.CAP_DSHOW if sys.platform.startswith("win") else cv2.CAP_ANY
    if clave not in _BACKENDS:
        opciones = ", ".join(sorted(_BACKENDS))
        raise ValueError(
            f"Backend de camara desconocido: '{nombre}'. Use 'auto' o uno de: {opciones}."
        )
    return _BACKENDS[clave]


class Camara:
    """Fuente de video: webcam, archivo de video o stream."""

    def __init__(self, cfg: CamaraConfig) -> None:
        self._cfg = cfg
        self._fuente, self._es_camara = _resolver_fuente(cfg.fuente)
        self._auto = es_fuente_automatica(cfg.fuente)
        # Si el config nombra un backend, se respeta y no se prueba ningun otro: alguien que
        # escribio "msmf" a mano sabe por que lo hizo.
        self._backend_explicito = (cfg.backend or "auto").strip().lower() != "auto"
        self._backend = _resolver_backend(cfg.backend, self._es_camara)
        self._captura: Optional[cv2.VideoCapture] = None
        self._ancho_real = 0
        self._alto_real = 0

    def _backends_candidatos(self) -> tuple[int, ...]:
        if self._backend_explicito:
            return (self._backend,)
        return _backends_de_sondeo()

    def _candidatos(self) -> list[tuple[int, int]]:
        """Pares (indice, backend) a probar, en orden de preferencia.

        Con fuente "auto" se recorren los indices; con un indice fijo se prueba SOLO ese,
        pero con todos los backends. Cambiar de backend es transparente --es la misma
        camara--, cambiar de indice no: podria ser otra camara, y eso solo se hace si el
        config lo pidio explicitamente.
        """
        backends = self._backends_candidatos()
        if self._auto:
            return [
                (indice, backend)
                for indice in range(_MAX_INDICE_SONDEO + 1)
                for backend in backends
            ]
        return [(int(self._fuente), backend) for backend in backends]

    def _abrir_camara(self) -> cv2.VideoCapture:
        """Abre la webcam probando hasta que una ENTREGUE un frame.

        Se lee un frame de prueba a proposito: isOpened() devuelve True con backends que
        despues no entregan nada, y en ese caso el bucle en vivo arrancaria para morirse
        treinta frames mas tarde con "la camara dejo de entregar frames", que manda a
        revisar el cable cuando el problema era el backend.
        """
        pedido = (self._fuente, self._backend)
        for indice, backend in self._candidatos():
            captura = cv2.VideoCapture(indice, backend)
            if captura.isOpened():
                # Se insiste varias veces, igual que en el sondeo. El microscopio USB del
                # taller tarda hasta media docena de frames en entregar el primero: leer una
                # sola vez descartaba una camara que funciona perfecto, y el programa
                # terminaba diciendo "no se pudo abrir" con la camara sana y enchufada.
                exito, frame = False, None
                for _ in range(_INTENTOS_LECTURA_SONDEO):
                    exito, frame = captura.read()
                    if exito and frame is not None:
                        break
                if exito and frame is not None:
                    if (indice, backend) != pedido:
                        _log.info(
                            "camara encontrada en el indice %d con el backend %s "
                            "(el config pedia %s)",
                            indice,
                            _NOMBRES_BACKEND.get(backend, str(backend)),
                            "auto" if self._auto else str(pedido[0]),
                        )
                    self._fuente, self._backend = indice, backend
                    return captura
            captura.release()
        raise RuntimeError(self._mensaje_no_abre())

    def abrir(self) -> None:
        """Abre la fuente y aplica la configuracion. Lanza RuntimeError si no abre."""
        if self._captura is not None and self._captura.isOpened():
            return

        if self._es_camara:
            captura = self._abrir_camara()
        else:
            captura = cv2.VideoCapture(self._fuente, self._backend)
            if not captura.isOpened():
                captura.release()
                raise RuntimeError(self._mensaje_no_abre())

        self._captura = captura
        if self._es_camara:
            self._aplicar_formato()
            self._aplicar_enfoque()
            self._aplicar_exposicion()
            self._aplicar_brillo_y_ganancia()
        self._releer_resolucion()

        if self._es_camara and (
            self._ancho_real != self._cfg.ancho or self._alto_real != self._cfg.alto
        ):
            _log.warning(
                "La camara negocio %dx%d en lugar de los %dx%d pedidos; se usara la real.",
                self._ancho_real,
                self._alto_real,
                self._cfg.ancho,
                self._cfg.alto,
            )

    def _mensaje_no_abre(self) -> str:
        nombre_backend = _NOMBRES_BACKEND.get(self._backend, str(self._backend))
        if self._es_camara:
            # Antes de rendirse, mirar que hay: si el operario puso el indice equivocado,
            # decirle cual es el bueno vale mas que mandarlo a revisar el cable.
            otras = enumerar_camaras()
            if otras:
                listado = "; ".join(str(c) for c in otras)
                if self._auto:
                    return (
                        "Se encontraron camaras (" + listado + ") pero ninguna entrego "
                        "imagen al abrirla. Suele ser que otro programa la tiene tomada "
                        "(Zoom, Teams, la app de Camara de Windows): cerralo y proba de nuevo."
                    )
                return (
                    f"No se pudo abrir la camara {self._fuente}, pero SI hay camara: "
                    f"{listado}. Corregi 'camara.fuente' en el config con ese indice, o "
                    "pone 'auto' para que la busque sola."
                )
            return (
                f"No se pudo abrir la camara {self._fuente} con el backend {nombre_backend}, "
                "y el sondeo no encontro ninguna otra. Verifique que este conectada, que "
                "ningun otro programa la tenga tomada (Zoom, Teams, la app de Camara de "
                "Windows) y, si es USB, que el cable haga contacto."
            )
        return (
            f"No se pudo abrir la fuente de video '{self._fuente}' con el backend "
            f"{nombre_backend}. Verifique que la ruta exista y que el formato sea legible."
        )

    def _aplicar_formato(self) -> None:
        assert self._captura is not None
        # MJPG antes que la resolucion: sin el, muchas webcam UVC limitan 1280x720 a 5 fps
        # porque intentan entregar YUYV sin comprimir.
        self._captura.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self._captura.set(cv2.CAP_PROP_FRAME_WIDTH, float(self._cfg.ancho))
        self._captura.set(cv2.CAP_PROP_FRAME_HEIGHT, float(self._cfg.alto))
        self._captura.set(cv2.CAP_PROP_FPS, float(self._cfg.fps))

    def _aplicar_enfoque(self) -> None:
        assert self._captura is not None
        if self._cfg.autofocus:
            self._captura.set(cv2.CAP_PROP_AUTOFOCUS, 1.0)
            _log.warning(
                "Autofoco ACTIVO: cada reenfoque cambia la escala mm/px y falsea las medidas. "
                "Para medir, use autofocus=false y fije 'enfoque'."
            )
            return
        if not self._captura.set(cv2.CAP_PROP_AUTOFOCUS, 0.0):
            _log.warning(
                "La camara no acepta desactivar el autofoco por software; "
                "desactivelo desde el panel del fabricante o las medidas van a derivar."
            )
        if self._cfg.enfoque is not None:
            if not self._captura.set(cv2.CAP_PROP_FOCUS, float(self._cfg.enfoque)):
                _log.warning(
                    "La camara no acepta fijar el enfoque en %s por software.",
                    self._cfg.enfoque,
                )

    def _aplicar_exposicion(self) -> None:
        assert self._captura is not None
        if self._cfg.exposicion is None:
            return
        self._captura.set(cv2.CAP_PROP_AUTO_EXPOSURE, _EXPOSICION_MANUAL)
        if not self._captura.set(cv2.CAP_PROP_EXPOSURE, float(self._cfg.exposicion)):
            _log.warning(
                "La camara no acepta fijar la exposicion en %s por software.",
                self._cfg.exposicion,
            )

    def _aplicar_brillo_y_ganancia(self) -> None:
        """Sube brillo y ganancia si el config los pide.

        En el microscopio USB del taller la imagen llega casi negra con los valores de
        fabrica --brillo medio 13 sobre 255, con el 7% de la imagen en negro puro-- y el
        detalle del filo, que es lo unico que importa, queda enterrado. Con brillo 64 sube a
        89 y agregando ganancia 64 llega a 144, con el doble de nitidez medida y sin quemar.

        Son controles del sensor, no un ajuste de la imagen despues: lo que se guarda para
        reentrenar ya sale bien expuesto. Corregirlo en software despues solo estiraria el
        ruido de una imagen que nunca tuvo la informacion.
        """
        assert self._captura is not None
        for nombre, propiedad, valor in (
            ("brillo", cv2.CAP_PROP_BRIGHTNESS, self._cfg.brillo),
            ("ganancia", cv2.CAP_PROP_GAIN, self._cfg.ganancia),
        ):
            if valor is None:
                continue
            if not self._captura.set(propiedad, float(valor)):
                _log.warning(
                    "La camara no acepta fijar %s en %s por software.", nombre, valor
                )
                continue
            leido = self._captura.get(propiedad)
            # Estas camaras clampean en silencio: piden 192 y se quedan en 64. Decirlo
            # evita media hora de subir el numero esperando que cambie algo.
            if leido >= 0 and abs(leido - float(valor)) > 1.0:
                _log.info(
                    "La camara limito %s a %.0f (se pidio %s): es su maximo.",
                    nombre,
                    leido,
                    valor,
                )

    def _releer_resolucion(self) -> None:
        """Lee la resolucion realmente entregada (puede diferir de la pedida)."""
        assert self._captura is not None
        self._ancho_real = int(round(self._captura.get(cv2.CAP_PROP_FRAME_WIDTH)))
        self._alto_real = int(round(self._captura.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    def leer(self) -> Optional[np.ndarray]:
        """Devuelve el frame siguiente, o None si la fuente se agoto o fallo."""
        if self._captura is None or not self._captura.isOpened():
            return None
        exito, frame = self._captura.read()
        if not exito or frame is None:
            return None
        if self._cfg.voltear_horizontal:
            frame = cv2.flip(frame, 1)
        if frame.shape[1] != self._ancho_real or frame.shape[0] != self._alto_real:
            self._alto_real, self._ancho_real = frame.shape[0], frame.shape[1]
        return frame

    def cerrar(self) -> None:
        """Libera la fuente. Es idempotente."""
        if self._captura is not None:
            self._captura.release()
            self._captura = None

    def __enter__(self) -> "Camara":
        self.abrir()
        return self

    def __exit__(
        self,
        tipo_exc: Optional[type[BaseException]],
        exc: Optional[BaseException],
        traza: Optional[TracebackType],
    ) -> None:
        self.cerrar()

    @property
    def abierta(self) -> bool:
        return self._captura is not None and self._captura.isOpened()

    @property
    def descripcion(self) -> str:
        """Ej: 'webcam 0 @1280x720 (DSHOW)' o 'video muestras/fresa.mp4 @1920x1080 (ANY)'."""
        nombre_backend = _NOMBRES_BACKEND.get(self._backend, str(self._backend))
        tipo = "webcam" if self._es_camara else "video"
        return (
            f"{tipo} {self._fuente} @{self._ancho_real}x{self._alto_real} ({nombre_backend})"
        )

    @property
    def resolucion_real(self) -> tuple[int, int]:
        """(ancho, alto) en px realmente entregados. (0, 0) si todavia no se abrio."""
        return self._ancho_real, self._alto_real
