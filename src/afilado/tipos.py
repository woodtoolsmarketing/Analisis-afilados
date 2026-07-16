"""Tipos de datos compartidos por todo el paquete.

Este modulo es la fuente de verdad del contrato entre modulos: define las
estructuras que viajan desde la calibracion y el detector hasta el overlay y
el almacen de feedback. No debe importar nada del propio paquete para evitar
ciclos, ni dependencias pesadas (torch/ultralytics).
"""

from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Protocol, Sequence
import numpy as np

@dataclass
class Escala:
    """Relacion pixel<->milimetro derivada del marcador de referencia."""
    mm_por_px: Optional[float] = None
    px_por_mm: Optional[float] = None
    valida: bool = False
    fuente: str = "ninguna"            # "aruco" | "aruco_extrapolado" | "manual" | "ninguna"
    id_marcador: Optional[int] = None
    esquinas: Optional[np.ndarray] = None      # (4,2) float32 en px
    error_lados: float = 0.0           # desvio relativo entre los 4 lados; alto => camara inclinada
    frames_sin_marcador: int = 0

@dataclass
class Medida:
    largo_px: float
    ancho_px: float
    area_px: float
    angulo_deg: float
    caja_rotada_px: np.ndarray         # (4,2) int32, esquinas del rectangulo de area minima
    centro_px: tuple[float, float]
    largo_mm: Optional[float] = None
    ancho_mm: Optional[float] = None
    area_mm2: Optional[float] = None
    fiable: bool = False               # False si no hay escala valida

@dataclass
class Deteccion:
    clase_id: int
    clase: str
    confianza: float
    xyxy: tuple[float, float, float, float]
    mascara: Optional[np.ndarray] = None   # uint8 0/255, mismo alto/ancho que el frame
    medida: Optional[Medida] = None
    descartada_por: Optional[str] = None   # "roi" | "area_minima" | "area_maxima" | "confianza"

@dataclass
class ResultadoFrame:
    detecciones: list[Deteccion] = field(default_factory=list)   # las que pasaron los filtros
    descartadas: list[Deteccion] = field(default_factory=list)   # con descartada_por seteado
    escala: Escala = field(default_factory=Escala)
    roi: tuple[int, int, int, int] = (0, 0, 0, 0)                # x1,y1,x2,y2 en px
    fps: float = 0.0
    forma_frame: tuple[int, int] = (0, 0)                        # (alto, ancho)
    modelo: str = "ninguno"
    instante: datetime = field(default_factory=datetime.now)

class Detector(Protocol):
    """Interfaz que cumplen DetectorYolo y DetectorGeometrico."""
    @property
    def descripcion(self) -> str: ...
    def predecir(self, frame: np.ndarray) -> list[Deteccion]: ...
