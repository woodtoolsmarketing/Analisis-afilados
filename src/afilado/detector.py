"""Detectores de estado de herramientas de afilado.

Expone dos implementaciones del protocolo `Detector`:

- `DetectorYolo`: envoltorio sobre un modelo YOLO de ultralytics (deteccion o segmentacion).
- `DetectorGeometrico`: fallback por umbralizacion de Otsu que NO necesita modelo entrenado.
  Permite medir piezas y, sobre todo, RECOLECTAR DATOS desde el primer dia: sin el, el bucle de
  auto-entrenamiento no podria arrancar (no hay modelo => no hay detecciones => no hay imagenes
  que guardar => nunca hay modelo).

`ultralytics` y `torch` se importan de forma PEREZOSA (dentro de los metodos, en try/except).
El resto del paquete debe poder importarse en una maquina sin torch instalado.

Ninguna implementacion asigna `medida` a las detecciones: medir es responsabilidad del pipeline,
que es quien conoce la escala del frame.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np

from .config import AppConfig, DetectorConfig, FiltrosConfig, ruta_absoluta
from .filtros import limpiar_mascara
from .tipos import Deteccion, Detector

logger = logging.getLogger(__name__)

_MENSAJE_SIN_ULTRALYTICS = (
    "No se pudo importar 'ultralytics'. Instalalo con: pip install ultralytics\n"
    "Mientras tanto el sistema sigue funcionando con el detector geometrico: podes medir "
    "y recolectar imagenes para entrenar, solo que sin clasificar el estado de la pieza."
)


def _resolver_dispositivo(dispositivo: str) -> str:
    """Traduce "auto" a "cuda:0" o "cpu" segun lo que haya disponible."""
    if dispositivo != "auto":
        return dispositivo
    try:
        import torch
    except ImportError:
        return "cpu"
    return "cuda:0" if torch.cuda.is_available() else "cpu"


class DetectorYolo:
    """Detector basado en un modelo YOLO entrenado (ultralytics)."""

    def __init__(self, cfg: DetectorConfig, clases: list[str]) -> None:
        self._cfg = cfg
        self._clases = list(clases)
        self._modelo: Optional[Any] = None
        self._dispositivo = "cpu"
        self._nombre_modelo = Path(cfg.pesos).stem or "yolo"
        self._nombres_modelo: dict[int, str] = {}
        self._aviso_clases_emitido = False

        try:
            from ultralytics import YOLO
        except ImportError:
            logger.error(_MENSAJE_SIN_ULTRALYTICS)
            return

        ruta_pesos = ruta_absoluta(cfg.pesos)
        if not ruta_pesos.is_file():
            logger.error("No existe el archivo de pesos: %s", ruta_pesos)
            return

        self._dispositivo = _resolver_dispositivo(cfg.dispositivo)
        try:
            self._modelo = YOLO(str(ruta_pesos))
        except Exception as exc:  # el .pt puede estar corrupto o ser de otra version
            logger.error("No se pudo cargar el modelo %s: %s", ruta_pesos, exc)
            return

        nombres = getattr(self._modelo, "names", None)
        if isinstance(nombres, dict):
            self._nombres_modelo = {int(k): str(v) for k, v in nombres.items()}
        elif isinstance(nombres, (list, tuple)):
            self._nombres_modelo = {i: str(v) for i, v in enumerate(nombres)}
        logger.info(
            "Modelo cargado: %s en %s (clases: %s)",
            ruta_pesos,
            self._dispositivo,
            ", ".join(self._nombres_modelo.values()) or "desconocidas",
        )

    @property
    def disponible(self) -> bool:
        return self._modelo is not None

    @property
    def descripcion(self) -> str:
        if self._modelo is None:
            return "YOLO no disponible"
        return f"{self._nombre_modelo} ({self._dispositivo})"

    def _nombre_de_clase(self, clase_id: int) -> str:
        """Prioriza los nombres del MODELO: el config puede estar desincronizado del .pt."""
        if clase_id in self._nombres_modelo:
            return self._nombres_modelo[clase_id]
        if 0 <= clase_id < len(self._clases):
            return self._clases[clase_id]
        return f"clase_{clase_id}"

    def _avisar_si_difieren_clases(self) -> None:
        if self._aviso_clases_emitido or not self._nombres_modelo:
            return
        self._aviso_clases_emitido = True
        del_modelo = [self._nombres_modelo[k] for k in sorted(self._nombres_modelo)]
        if del_modelo != self._clases:
            logger.warning(
                "Las clases del modelo %s no coinciden con las del config %s. "
                "Se usan las del modelo; revisa configs/config.yaml.",
                del_modelo,
                self._clases,
            )

    def _mascaras_al_frame(
        self, mascaras: np.ndarray, alto: int, ancho: int
    ) -> list[np.ndarray]:
        """Lleva las mascaras de la resolucion del modelo (imgsz) a la del frame.

        INTER_NEAREST y no INTER_LINEAR: interpolar valores de una mascara binaria inventa
        bordes intermedios que luego falsean el area medida.
        """
        salida: list[np.ndarray] = []
        for mascara in mascaras:
            binaria = (mascara > 0.5).astype(np.uint8) * 255
            if binaria.shape[:2] != (alto, ancho):
                binaria = cv2.resize(
                    binaria, (ancho, alto), interpolation=cv2.INTER_NEAREST
                )
            salida.append(binaria)
        return salida

    def predecir(self, frame: np.ndarray) -> list[Deteccion]:
        if self._modelo is None:
            return []

        try:
            resultados = self._modelo.predict(
                frame,
                conf=self._cfg.confianza,
                iou=self._cfg.iou,
                imgsz=self._cfg.imgsz,
                device=self._dispositivo,
                max_det=self._cfg.max_detecciones,
                verbose=False,
            )
        except Exception as exc:
            logger.error("Fallo la inferencia: %s", exc)
            return []

        if not resultados:
            return []
        resultado = resultados[0]
        self._avisar_si_difieren_clases()

        cajas = getattr(resultado, "boxes", None)
        if cajas is None or len(cajas) == 0:
            return []

        xyxy = cajas.xyxy.cpu().numpy()
        confianzas = cajas.conf.cpu().numpy()
        clases_id = cajas.cls.cpu().numpy().astype(int)

        alto, ancho = frame.shape[:2]
        mascaras: list[np.ndarray] = []
        objeto_mascaras = getattr(resultado, "masks", None)
        if objeto_mascaras is not None and objeto_mascaras.data is not None:
            mascaras = self._mascaras_al_frame(
                objeto_mascaras.data.cpu().numpy(), alto, ancho
            )

        detecciones: list[Deteccion] = []
        for indice in range(len(xyxy)):
            x1, y1, x2, y2 = (float(v) for v in xyxy[indice])
            clase_id = int(clases_id[indice])
            detecciones.append(
                Deteccion(
                    clase_id=clase_id,
                    clase=self._nombre_de_clase(clase_id),
                    confianza=float(confianzas[indice]),
                    xyxy=(x1, y1, x2, y2),
                    mascara=mascaras[indice] if indice < len(mascaras) else None,
                )
            )
        return detecciones


class DetectorGeometrico:
    """Fallback sin modelo: separa la pieza del fondo por umbral de Otsu.

    Asume PIEZA OSCURA SOBRE FONDO CLARO (THRESH_BINARY_INV): el metal a contraluz o sobre mesa
    blanca queda como silueta. Si en tu taller la pieza es clara sobre mesa oscura, la mascara
    sale invertida (detectara el fondo como pieza): hay que cambiar THRESH_BINARY_INV por
    THRESH_BINARY.
    """

    def __init__(self, cfg: FiltrosConfig) -> None:
        self._cfg = cfg

    @property
    def descripcion(self) -> str:
        return "geometrico (sin modelo entrenado)"

    def predecir(self, frame: np.ndarray) -> list[Deteccion]:
        if frame.ndim == 3:
            gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gris = frame
        # El blur previo evita que el ruido del sensor parta el histograma que usa Otsu.
        suavizado = cv2.GaussianBlur(gris, (5, 5), 0)
        _, binaria = cv2.threshold(
            suavizado, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU
        )
        binaria = limpiar_mascara(
            binaria, self._cfg.kernel_morfologico, self._cfg.iteraciones_morfologicas
        )

        contornos, _ = cv2.findContours(
            binaria, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        alto, ancho = binaria.shape[:2]
        detecciones: list[Deteccion] = []
        for contorno in contornos:
            if cv2.contourArea(contorno) < self._cfg.area_minima_px:
                continue
            mascara = np.zeros((alto, ancho), dtype=np.uint8)
            cv2.drawContours(mascara, [contorno], -1, 255, thickness=cv2.FILLED)
            x, y, w, h = cv2.boundingRect(contorno)
            detecciones.append(
                Deteccion(
                    clase_id=-1,
                    clase="sin_clasificar",
                    confianza=0.0,
                    xyxy=(float(x), float(y), float(x + w), float(y + h)),
                    mascara=mascara,
                )
            )
        return detecciones


def crear_detector(cfg: AppConfig) -> Detector:
    """Elige el mejor detector disponible: YOLO si hay pesos y ultralytics, si no el geometrico."""
    ruta_pesos = ruta_absoluta(cfg.detector.pesos)
    if not ruta_pesos.is_file():
        logger.warning(
            "No se encontraron los pesos en %s. Se usa el detector geometrico: mide y permite "
            "recolectar datos, pero no clasifica el estado.",
            ruta_pesos,
        )
        return DetectorGeometrico(cfg.filtros)

    detector = DetectorYolo(cfg.detector, cfg.clases)
    if not detector.disponible:
        logger.warning(
            "Hay pesos en %s pero el modelo no se pudo cargar (falta ultralytics/torch o el "
            "archivo no es valido). Se usa el detector geometrico.",
            ruta_pesos,
        )
        return DetectorGeometrico(cfg.filtros)
    return detector
