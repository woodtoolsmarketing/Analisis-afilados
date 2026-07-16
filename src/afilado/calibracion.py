"""Calibracion de escala pixel<->milimetro a partir de un marcador ArUco impreso.

El marcador es la unica referencia metrica del sistema: su lado fisico es conocido
(cfg.lado_mm) y su lado en pixeles se mide en cada frame. De ahi sale mm_por_px.

LIMITACION FISICA (error de paralaje): el marcador debe estar SIEMPRE a la misma
altura que la cara de la pieza que se mide. Si el marcador esta apoyado en la mesa
y la pieza tiene espesor, la cara superior queda mas cerca del lente, se ve mas
grande y la medida sale inflada. Ninguna correccion de software arregla esto con
una sola camara 2D: hay que colocar el marcador a la altura correcta.
"""

from __future__ import annotations

import logging
from typing import Optional

import cv2
import numpy as np

from .config import ArucoConfig
from .tipos import Escala

_log = logging.getLogger(__name__)


def obtener_diccionario(nombre: str) -> "cv2.aruco.Dictionary":
    """Devuelve el diccionario ArUco cuyo nombre es una constante de cv2.aruco.

    Args:
        nombre: por ejemplo "DICT_4X4_50".

    Raises:
        ValueError: si el nombre no existe en cv2.aruco.
    """
    if not hasattr(cv2, "aruco"):
        raise ValueError(
            "OpenCV no trae el modulo aruco. Instala 'opencv-contrib-python'."
        )
    constante = getattr(cv2.aruco, nombre, None)
    if constante is None or not isinstance(constante, int):
        raise ValueError(
            f"Diccionario ArUco desconocido: '{nombre}'. "
            "Usa un nombre valido como 'DICT_4X4_50' o 'DICT_5X5_100'."
        )
    if hasattr(cv2.aruco, "getPredefinedDictionary"):
        return cv2.aruco.getPredefinedDictionary(constante)
    return cv2.aruco.Dictionary_get(constante)


def dibujar_marcador(diccionario: str, id_marcador: int, lado_px: int) -> np.ndarray:
    """Genera la imagen en escala de grises de un marcador ArUco.

    Args:
        diccionario: nombre del diccionario, ej "DICT_4X4_50".
        id_marcador: id dentro del diccionario.
        lado_px: lado de la imagen resultante en pixeles.

    Raises:
        ValueError: si el diccionario no existe, el id es negativo o lado_px < 1.
    """
    if lado_px < 1:
        raise ValueError(f"lado_px debe ser >= 1, se recibio {lado_px}.")
    if id_marcador < 0:
        raise ValueError(f"id_marcador debe ser >= 0, se recibio {id_marcador}.")
    dic = obtener_diccionario(diccionario)
    if hasattr(cv2.aruco, "generateImageMarker"):
        return cv2.aruco.generateImageMarker(dic, id_marcador, lado_px)
    return cv2.aruco.drawMarker(dic, id_marcador, lado_px)


def _a_gris(frame: np.ndarray) -> np.ndarray:
    """Convierte a escala de grises solo si hace falta."""
    if frame.ndim == 2:
        return frame
    if frame.shape[2] == 3:
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if frame.shape[2] == 4:
        return cv2.cvtColor(frame, cv2.COLOR_BGRA2GRAY)
    raise ValueError(
        f"Formato de frame no soportado: forma {frame.shape}. "
        "Se esperaba gris, BGR o BGRA."
    )


def _lados_del_marcador(esquinas: np.ndarray) -> np.ndarray:
    """Longitudes en px de los 4 lados del cuadrilatero (0-1, 1-2, 2-3, 3-0)."""
    puntos = esquinas.reshape(4, 2).astype(np.float64)
    siguiente = np.roll(puntos, -1, axis=0)
    return np.linalg.norm(siguiente - puntos, axis=1)


class CalibradorAruco:
    """Estima la escala mm/px buscando un marcador ArUco en cada frame.

    Detecta una sola vez, en el constructor, cual API de ArUco trae la version de
    OpenCV instalada (ArucoDetector desde 4.7, detectMarkers legacy antes) y guarda
    la estrategia. Hacer hasattr por frame costaria tiempo en el bucle en vivo.
    """

    def __init__(self, cfg: ArucoConfig) -> None:
        self._cfg = cfg
        self._escala = Escala()
        self._px_por_mm_suavizado: Optional[float] = None
        self._detector = None
        self._diccionario = None
        self._parametros = None
        self._api_nueva = False

        if not cfg.habilitado:
            return

        self._diccionario = obtener_diccionario(cfg.diccionario)
        if hasattr(cv2.aruco, "ArucoDetector"):
            self._api_nueva = True
            self._parametros = cv2.aruco.DetectorParameters()
            self._detector = cv2.aruco.ArucoDetector(self._diccionario, self._parametros)
        else:
            self._api_nueva = False
            self._parametros = cv2.aruco.DetectorParameters_create()

    @property
    def ultima_escala(self) -> Escala:
        """Ultima escala calculada (o Escala invalida si aun no hubo ninguna)."""
        return self._escala

    def reiniciar(self) -> None:
        """Olvida la escala previa y el suavizado. Util al mover camara o marcador."""
        self._escala = Escala()
        self._px_por_mm_suavizado = None

    def _detectar(self, gris: np.ndarray) -> tuple[list[np.ndarray], Optional[np.ndarray]]:
        if self._api_nueva:
            esquinas, ids, _ = self._detector.detectMarkers(gris)
        else:
            esquinas, ids, _ = cv2.aruco.detectMarkers(
                gris, self._diccionario, parameters=self._parametros
            )
        return list(esquinas) if esquinas is not None else [], ids

    def _elegir_marcador(
        self, esquinas: list[np.ndarray], ids: np.ndarray
    ) -> Optional[tuple[int, np.ndarray]]:
        """Devuelve (id, esquinas) del marcador de referencia, o None si no esta."""
        planos = ids.flatten().tolist()
        if self._cfg.id_referencia is None:
            indice = int(np.argmin(planos))
            return int(planos[indice]), esquinas[indice]
        for indice, valor in enumerate(planos):
            if int(valor) == self._cfg.id_referencia:
                return int(valor), esquinas[indice]
        return None

    def _sin_marcador(self) -> Escala:
        """Extrapola la ultima escala mientras la ausencia sea breve.

        Que la mano del operario tape el marcador un instante no debe borrar la
        escala: se conserva unos frames marcada como extrapolada, y recien pasado
        el limite se declara invalida.
        """
        previa = self._escala
        frames = previa.frames_sin_marcador + 1
        tenia_escala = previa.px_por_mm is not None
        if tenia_escala and frames <= self._cfg.max_frames_sin_marcador:
            self._escala = Escala(
                mm_por_px=previa.mm_por_px,
                px_por_mm=previa.px_por_mm,
                valida=True,
                fuente="aruco_extrapolado",
                id_marcador=previa.id_marcador,
                esquinas=previa.esquinas,
                error_lados=previa.error_lados,
                frames_sin_marcador=frames,
            )
        else:
            self._escala = Escala(
                valida=False, fuente="ninguna", frames_sin_marcador=frames
            )
            self._px_por_mm_suavizado = None
        return self._escala

    def estimar(self, frame: np.ndarray) -> Escala:
        """Busca el marcador en el frame completo y devuelve la escala resultante.

        Args:
            frame: imagen en gris, BGR o BGRA. El frame no se modifica.

        Returns:
            La escala vigente. Si el marcador no aparece, la ultima escala
            extrapolada mientras no se supere max_frames_sin_marcador.
        """
        if not self._cfg.habilitado:
            self._escala = Escala(valida=False, fuente="ninguna")
            return self._escala
        if frame is None or getattr(frame, "size", 0) == 0:
            return self._sin_marcador()

        gris = _a_gris(frame)
        esquinas, ids = self._detectar(gris)
        if ids is None or len(esquinas) == 0:
            return self._sin_marcador()

        elegido = self._elegir_marcador(esquinas, ids)
        if elegido is None:
            return self._sin_marcador()

        id_marcador, esquinas_marcador = elegido
        lados = _lados_del_marcador(esquinas_marcador)
        promedio_px = float(lados.mean())
        if promedio_px <= 0.0:
            _log.warning("Marcador %d detectado con lado nulo; se ignora.", id_marcador)
            return self._sin_marcador()

        error_lados = float(lados.std() / promedio_px)
        px_por_mm_medido = promedio_px / self._cfg.lado_mm

        # EMA solo sobre px_por_mm: mm_por_px se recalcula como su inversa. Suavizar
        # ambos por separado los haria divergir (la media de inversas no es la inversa
        # de la media). El suavizado evita que el numero baile por ruido sub-pixel.
        if self._px_por_mm_suavizado is None:
            px_por_mm = px_por_mm_medido
        else:
            a = self._cfg.suavizado
            px_por_mm = a * px_por_mm_medido + (1.0 - a) * self._px_por_mm_suavizado
        self._px_por_mm_suavizado = px_por_mm

        if error_lados > self._cfg.max_error_lados:
            _log.debug(
                "Marcador %d con lados desparejos (error %.3f > %.3f): camara inclinada.",
                id_marcador,
                error_lados,
                self._cfg.max_error_lados,
            )

        self._escala = Escala(
            mm_por_px=1.0 / px_por_mm,
            px_por_mm=px_por_mm,
            valida=True,
            fuente="aruco",
            id_marcador=id_marcador,
            esquinas=esquinas_marcador.reshape(4, 2).astype(np.float32),
            error_lados=error_lados,
            frames_sin_marcador=0,
        )
        return self._escala
