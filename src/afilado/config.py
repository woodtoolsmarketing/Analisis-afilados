"""Configuracion de la aplicacion: dataclasses, carga y validacion del YAML.

La configuracion se declara con dataclasses tipadas y se serializa a YAML con
las mismas claves que los nombres de campo. La construccion es estricta con las
claves desconocidas: un typo en el YAML (por ejemplo "confianzza") se rechaza
con ValueError en vez de ignorarse en silencio, porque el operario creeria haber
cambiado un umbral que en realidad quedo en su valor por defecto.

Solo depende de stdlib y pyyaml: se puede importar sin torch ni ultralytics.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Optional, Union

import yaml

CLASES_POR_DEFECTO: list[str] = ["ok", "desgastado", "fisura", "astillado", "oxido"]
CLASES_DEFECTO_POR_DEFECTO: list[str] = ["desgastado", "fisura", "astillado", "oxido"]

TAREAS_VALIDAS: frozenset[str] = frozenset({"detect", "segment"})


@dataclass
class CamaraConfig:
    """Parametros de captura de la webcam cenital."""

    fuente: Union[int, str] = 0
    ancho: int = 1280
    alto: int = 720
    fps: int = 30
    backend: str = "auto"
    autofocus: bool = False
    enfoque: Optional[int] = None
    exposicion: Optional[float] = None
    voltear_horizontal: bool = False


@dataclass
class ArucoConfig:
    """Parametros del marcador de referencia que fija la escala mm/px."""

    habilitado: bool = True
    diccionario: str = "DICT_4X4_50"
    lado_mm: float = 30.0
    id_referencia: Optional[int] = 0
    suavizado: float = 0.3
    max_frames_sin_marcador: int = 90
    max_error_lados: float = 0.06


@dataclass
class RoiConfig:
    """Region de interes en coordenadas normalizadas 0..1 sobre el frame."""

    habilitado: bool = True
    x: float = 0.15
    y: float = 0.15
    w: float = 0.7
    h: float = 0.7


@dataclass
class DetectorConfig:
    """Parametros del modelo YOLO que clasifica el estado de la pieza."""

    pesos: str = "models/afilado_best.pt"
    tarea: str = "segment"
    confianza: float = 0.80
    iou: float = 0.45
    imgsz: int = 640
    dispositivo: str = "auto"
    max_detecciones: int = 20


@dataclass
class FiltrosConfig:
    """Umbrales geometricos que separan la pieza real del polvo y el aserrin."""

    area_minima_mm2: float = 20.0
    area_maxima_mm2: float = 100000.0
    area_minima_px: int = 400
    kernel_morfologico: int = 3
    iteraciones_morfologicas: int = 1


@dataclass
class FeedbackConfig:
    """Destino y formato de las capturas del bucle de auto-entrenamiento."""

    habilitado: bool = True
    directorio: str = "data/feedback"
    guardar_json: bool = True
    guardar_anotada: bool = True
    calidad_jpg: int = 95


@dataclass
class AppConfig:
    """Configuracion completa de la aplicacion."""

    camara: CamaraConfig = field(default_factory=CamaraConfig)
    aruco: ArucoConfig = field(default_factory=ArucoConfig)
    roi: RoiConfig = field(default_factory=RoiConfig)
    detector: DetectorConfig = field(default_factory=DetectorConfig)
    filtros: FiltrosConfig = field(default_factory=FiltrosConfig)
    feedback: FeedbackConfig = field(default_factory=FeedbackConfig)
    clases: list[str] = field(default_factory=lambda: list(CLASES_POR_DEFECTO))
    clases_defecto: list[str] = field(
        default_factory=lambda: list(CLASES_DEFECTO_POR_DEFECTO)
    )


_SECCIONES: dict[str, type] = {
    "camara": CamaraConfig,
    "aruco": ArucoConfig,
    "roi": RoiConfig,
    "detector": DetectorConfig,
    "filtros": FiltrosConfig,
    "feedback": FeedbackConfig,
}


def raiz_repo() -> Path:
    """Devuelve la raiz del repositorio: la primera carpeta ascendente con configs/."""
    actual = Path(__file__).resolve()
    for carpeta in actual.parents:
        if (carpeta / "configs").is_dir():
            return carpeta
    # Sin configs/ (instalacion como paquete), la raiz razonable es la que contiene src/.
    return actual.parents[2]


def ruta_absoluta(valor: Union[str, Path]) -> Path:
    """Resuelve una ruta relativa contra la raiz del repo; deja intactas las absolutas."""
    ruta = Path(valor)
    if ruta.is_absolute():
        return ruta
    return (raiz_repo() / ruta).resolve()


def _campos_de(tipo: type) -> set[str]:
    return {f.name for f in fields(tipo)}


def _construir_seccion(nombre: str, tipo: type, datos: Any) -> Any:
    """Construye una dataclass de seccion rechazando claves desconocidas."""
    if datos is None:
        return tipo()
    if not isinstance(datos, dict):
        raise ValueError(
            f"La seccion '{nombre}' del config debe ser un mapa de claves, "
            f"se recibio {type(datos).__name__}."
        )
    validas = _campos_de(tipo)
    for clave in datos:
        if clave not in validas:
            esperadas = ", ".join(sorted(validas))
            raise ValueError(
                f"Clave desconocida '{clave}' en la seccion '{nombre}' del config. "
                f"Claves validas: {esperadas}."
            )
    return tipo(**datos)


def _validar(cfg: AppConfig) -> None:
    """Valida invariantes que las dataclasses no pueden expresar por si solas."""
    roi = cfg.roi
    for nombre, valor in (("x", roi.x), ("y", roi.y), ("w", roi.w), ("h", roi.h)):
        if not 0.0 <= float(valor) <= 1.0:
            raise ValueError(
                f"roi.{nombre} debe estar entre 0 y 1 (normalizado), se recibio {valor}."
            )
    if roi.x + roi.w > 1.0:
        raise ValueError(
            f"roi.x + roi.w no puede superar 1: {roi.x} + {roi.w} = {roi.x + roi.w}. "
            "El ROI se saldria del borde derecho del frame."
        )
    if roi.y + roi.h > 1.0:
        raise ValueError(
            f"roi.y + roi.h no puede superar 1: {roi.y} + {roi.h} = {roi.y + roi.h}. "
            "El ROI se saldria del borde inferior del frame."
        )

    if not 0.0 <= float(cfg.detector.confianza) <= 1.0:
        raise ValueError(
            f"detector.confianza debe estar entre 0 y 1, se recibio {cfg.detector.confianza}."
        )
    if cfg.detector.tarea not in TAREAS_VALIDAS:
        validas = ", ".join(sorted(TAREAS_VALIDAS))
        raise ValueError(
            f"detector.tarea debe ser una de: {validas}. Se recibio '{cfg.detector.tarea}'."
        )

    if float(cfg.aruco.lado_mm) <= 0:
        raise ValueError(
            f"aruco.lado_mm debe ser mayor que 0, se recibio {cfg.aruco.lado_mm}. "
            "Es el lado fisico real del marcador impreso, medido con calibre."
        )
    if not 0.0 <= float(cfg.aruco.suavizado) <= 1.0:
        raise ValueError(
            f"aruco.suavizado debe estar entre 0 y 1, se recibio {cfg.aruco.suavizado}."
        )

    kernel = int(cfg.filtros.kernel_morfologico)
    if kernel < 1 or kernel % 2 == 0:
        raise ValueError(
            f"filtros.kernel_morfologico debe ser impar y >= 1, se recibio {kernel}. "
            "Un kernel par no tiene pixel central y desplazaria la mascara."
        )

    if not cfg.clases:
        raise ValueError("clases no puede estar vacia: el sistema necesita al menos una clase.")
    faltantes = [c for c in cfg.clases_defecto if c not in cfg.clases]
    if faltantes:
        raise ValueError(
            "clases_defecto debe ser un subconjunto de clases. "
            f"No estan declaradas en clases: {', '.join(faltantes)}."
        )


def cargar_config(ruta: Optional[Union[str, Path]] = None) -> AppConfig:
    """Carga la configuracion desde YAML, valida y devuelve un AppConfig.

    Si ruta es None usa configs/config.yaml de la raiz del repo. Si el archivo no
    existe devuelve los valores por defecto. Lanza ValueError en espanol ante
    claves desconocidas o valores incoherentes.
    """
    destino = ruta_absoluta(ruta) if ruta is not None else raiz_repo() / "configs" / "config.yaml"

    if not destino.is_file():
        cfg = AppConfig()
        _validar(cfg)
        return cfg

    with open(destino, "r", encoding="utf-8") as manejador:
        crudo = yaml.safe_load(manejador)

    if crudo is None:
        crudo = {}
    if not isinstance(crudo, dict):
        raise ValueError(
            f"El config '{destino}' debe contener un mapa de claves en la raiz, "
            f"se leyo {type(crudo).__name__}."
        )

    validas = _campos_de(AppConfig)
    for clave in crudo:
        if clave not in validas:
            esperadas = ", ".join(sorted(validas))
            raise ValueError(
                f"Clave desconocida '{clave}' en la raiz del config '{destino}'. "
                f"Claves validas: {esperadas}."
            )

    argumentos: dict[str, Any] = {}
    for nombre, tipo in _SECCIONES.items():
        if nombre in crudo:
            try:
                argumentos[nombre] = _construir_seccion(nombre, tipo, crudo[nombre])
            except TypeError as error:
                raise ValueError(
                    f"No se pudo construir la seccion '{nombre}' del config '{destino}': {error}"
                ) from error

    for nombre in ("clases", "clases_defecto"):
        if nombre in crudo:
            valor = crudo[nombre]
            if valor is None:
                continue
            if not isinstance(valor, list) or not all(isinstance(c, str) for c in valor):
                raise ValueError(
                    f"'{nombre}' debe ser una lista de textos, se recibio {type(valor).__name__}."
                )
            argumentos[nombre] = list(valor)

    cfg = AppConfig(**argumentos)
    _validar(cfg)
    return cfg


def guardar_config(cfg: AppConfig, ruta: Union[str, Path]) -> None:
    """Serializa la configuracion a YAML, creando la carpeta destino si falta."""
    if not is_dataclass(cfg):
        raise ValueError("guardar_config espera una instancia de AppConfig.")
    _validar(cfg)
    destino = ruta_absoluta(ruta)
    destino.parent.mkdir(parents=True, exist_ok=True)
    with open(destino, "w", encoding="utf-8") as manejador:
        yaml.safe_dump(
            asdict(cfg),
            manejador,
            allow_unicode=True,
            sort_keys=False,
            default_flow_style=False,
        )
