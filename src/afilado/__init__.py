"""Sistema de vision artificial para inspeccion de herramientas de afilado.

Expone la configuracion y los objetos principales del pipeline. Pipeline,
crear_detector y Camara se resuelven con __getattr__ perezoso (PEP 562) para que
"import afilado" y "from afilado import cargar_config" funcionen en maquinas sin
torch ni ultralytics instalados: importar detector a nivel de modulo arrastraria
esas dependencias en cadena y romperia herramientas que solo leen el config.
"""

from __future__ import annotations

from typing import Any

from .config import AppConfig, cargar_config

__version__ = "0.1.0"

__all__ = [
    "AppConfig",
    "cargar_config",
    "Pipeline",
    "crear_detector",
    "Camara",
    "__version__",
]

_PEREZOSOS: dict[str, str] = {
    "Pipeline": "afilado.pipeline",
    "crear_detector": "afilado.detector",
    "Camara": "afilado.camara",
}


def __getattr__(nombre: str) -> Any:
    modulo = _PEREZOSOS.get(nombre)
    if modulo is None:
        raise AttributeError(f"module 'afilado' no tiene el atributo '{nombre}'")
    from importlib import import_module

    return getattr(import_module(modulo), nombre)


def __dir__() -> list[str]:
    return sorted(__all__)
