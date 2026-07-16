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
from pathlib import Path
from types import TracebackType
from typing import Optional

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


def _resolver_fuente(fuente: int | str) -> tuple[int | str, bool]:
    """Devuelve (fuente_para_VideoCapture, es_camara).

    Un str numerico ("0") se trata como indice de camara: es lo que llega desde la linea
    de comandos o desde el YAML sin comillas mal puestas. Un str que apunta a un archivo o
    carpeta existente se usa como video. Cualquier otro str se pasa tal cual (URL RTSP/HTTP).
    """
    if isinstance(fuente, int):
        return fuente, True
    texto = str(fuente).strip()
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
        self._backend = _resolver_backend(cfg.backend, self._es_camara)
        self._captura: Optional[cv2.VideoCapture] = None
        self._ancho_real = 0
        self._alto_real = 0

    def abrir(self) -> None:
        """Abre la fuente y aplica la configuracion. Lanza RuntimeError si no abre."""
        if self._captura is not None and self._captura.isOpened():
            return

        captura = cv2.VideoCapture(self._fuente, self._backend)
        if not captura.isOpened():
            captura.release()
            raise RuntimeError(self._mensaje_no_abre())

        self._captura = captura
        if self._es_camara:
            self._aplicar_formato()
            self._aplicar_enfoque()
            self._aplicar_exposicion()
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
            return (
                f"No se pudo abrir la camara {self._fuente} con el backend {nombre_backend}. "
                "Verifique que este conectada, que ningun otro programa la tenga tomada "
                "(Zoom, Teams, la app de Camara de Windows) y que el indice sea el correcto."
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
