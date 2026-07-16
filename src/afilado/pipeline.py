"""Orquestador de un frame: calibrar, detectar, medir, filtrar.

Es el unico lugar donde se fija el ORDEN de los pasos, y ese orden importa:
la escala tiene que estar resuelta antes de medir, y las medidas antes de filtrar
por area en mm2. Ningun paso escribe sobre el frame que entra.
"""

from __future__ import annotations

import time
from collections import deque
from typing import Optional

import numpy as np

from .calibracion import CalibradorAruco
from .config import AppConfig
from .detector import crear_detector
from .filtros import aplicar_filtros, rect_roi
from .medicion import medir_desde_caja, medir_desde_mascara
from .tipos import Deteccion, Detector, Escala, Medida, ResultadoFrame


class Pipeline:
    """Encadena calibracion, deteccion, medicion y filtrado sobre cada frame."""

    def __init__(self, cfg: AppConfig, detector: Optional[Detector] = None) -> None:
        self._cfg = cfg
        self._detector: Detector = detector if detector is not None else crear_detector(cfg)
        self._calibrador = CalibradorAruco(cfg.aruco)
        # Media movil de ~30 frames: el FPS instantaneo salta con cada hipo del sistema
        # operativo y seria ilegible en el HUD.
        self._duraciones: deque[float] = deque(maxlen=30)

    @property
    def descripcion_modelo(self) -> str:
        return self._detector.descripcion

    def reiniciar_calibracion(self) -> None:
        self._calibrador.reiniciar()

    def procesar(self, frame: np.ndarray) -> ResultadoFrame:
        """Procesa un frame y devuelve el resultado. No modifica el frame recibido."""
        inicio = time.perf_counter()

        # El ArUco se busca en el frame COMPLETO: el marcador suele estar apoyado al costado
        # de la pieza, fuera del ROI, y aun asi define la escala de todo lo que se mide.
        escala = self._calibrador.estimar(frame)

        detecciones = self._detector.predecir(frame)
        for det in detecciones:
            det.medida = self._medir(det, escala)

        forma_frame = (int(frame.shape[0]), int(frame.shape[1]))
        roi = rect_roi(forma_frame, self._cfg.roi)
        aceptadas, descartadas = aplicar_filtros(detecciones, roi, self._cfg.filtros, escala)

        self._duraciones.append(time.perf_counter() - inicio)
        return ResultadoFrame(
            detecciones=aceptadas,
            descartadas=descartadas,
            escala=escala,
            roi=roi,
            fps=self._fps(),
            forma_frame=forma_frame,
            modelo=self._detector.descripcion,
        )

    def _medir(self, det: Deteccion, escala: Escala) -> Medida:
        """Mide por mascara si la hay; si no (o si la mascara vino vacia), por caja."""
        if det.mascara is not None:
            medida = medir_desde_mascara(det.mascara, escala)
            if medida is not None:
                return medida
        return medir_desde_caja(det.xyxy, escala)

    def _fps(self) -> float:
        if not self._duraciones:
            return 0.0
        promedio = sum(self._duraciones) / len(self._duraciones)
        if promedio <= 0.0:
            return 0.0
        return 1.0 / promedio
