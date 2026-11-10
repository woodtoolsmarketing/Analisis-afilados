"""Galeria de fotos de referencia: varias piezas modelo, y una seleccionada.

POR QUE VARIAS
--------------
Cada modelo de sierra tiene su geometria de diente. Comparar un LU3F-0300 contra la foto de
un R01 no dice nada: el operario necesita tener al lado la foto del modelo que esta mirando.
Con una sola referencia, cada cambio de modelo obligaba a sacar la foto de nuevo y se perdia
la anterior.

COMO SE ORGANIZA
----------------
Una carpeta con una imagen por modelo. El nombre del archivo es el nombre que se muestra, y
el orden es alfabetico, asi que numerarlas (01-LU3F.jpg, 02-R01.jpg) alcanza para fijar el
orden en que se ciclan.

Las imagenes se cargan cuando se las necesita, no todas juntas: una carpeta con cincuenta
fotos de microscopio son varios cientos de MB, y la PC del taller tiene que andar con poco.

Se recuerda cual estaba elegida la ultima vez, en un archivo .ultima dentro de la misma
carpeta. Sin eso, el operario que trabaja todo el dia con el mismo modelo tendria que
ciclar hasta encontrarlo en cada arranque.
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from .config import ruta_absoluta

_log = logging.getLogger(__name__)

_EXTENSIONES = {".jpg", ".jpeg", ".png", ".bmp"}
_ARCHIVO_ULTIMA = ".ultima"


class GaleriaReferencias:
    """Las fotos de referencia de una carpeta, con una seleccionada."""

    def __init__(self, ruta_config: str, calidad_jpg: int = 95) -> None:
        """`ruta_config` puede ser una CARPETA o un ARCHIVO.

        Que acepte un archivo no es por comodidad: es lo que tenian los configs anteriores,
        cuando habia una sola referencia. En ese caso se usa su carpeta como galeria y ese
        archivo queda preseleccionado, asi que una instalacion vieja sigue andando igual sin
        que nadie toque el config.
        """
        ruta = ruta_absoluta(ruta_config)
        if ruta.suffix.lower() in _EXTENSIONES:
            self._carpeta = ruta.parent
            preferida: Optional[str] = ruta.name
        else:
            self._carpeta = ruta
            preferida = None

        self._calidad = int(calidad_jpg)
        self._nombres: list[str] = []
        self._indice = 0
        self._cache: Optional[tuple[str, np.ndarray]] = None
        self.recargar(preferida)

    # ---------------------------------------------------------------- lectura
    @property
    def carpeta(self) -> Path:
        return self._carpeta

    @property
    def cantidad(self) -> int:
        return len(self._nombres)

    @property
    def nombre_actual(self) -> Optional[str]:
        """Nombre de la referencia elegida, sin extension. None si la galeria esta vacia."""
        if not self._nombres:
            return None
        return Path(self._nombres[self._indice]).stem

    @property
    def posicion(self) -> tuple[int, int]:
        """(cual, cuantas), empezando en 1. (0, 0) si no hay ninguna."""
        if not self._nombres:
            return (0, 0)
        return (self._indice + 1, len(self._nombres))

    def imagen_actual(self) -> Optional[np.ndarray]:
        """La imagen elegida, leida del disco la primera vez que se la pide."""
        if not self._nombres:
            return None
        nombre = self._nombres[self._indice]
        if self._cache is not None and self._cache[0] == nombre:
            return self._cache[1]
        imagen = cv2.imread(str(self._carpeta / nombre))
        if imagen is None:
            _log.warning("no se pudo leer la referencia %s", self._carpeta / nombre)
            return None
        self._cache = (nombre, imagen)
        return imagen

    # ---------------------------------------------------------------- cambios
    def recargar(self, preferida: Optional[str] = None) -> None:
        """Relee la carpeta. Mantiene la seleccion si el archivo sigue estando."""
        elegida = preferida or (self._nombres[self._indice] if self._nombres else None)
        if elegida is None:
            elegida = self._leer_ultima()

        if self._carpeta.is_dir():
            self._nombres = sorted(
                p.name
                for p in self._carpeta.iterdir()
                if p.is_file() and p.suffix.lower() in _EXTENSIONES
            )
        else:
            self._nombres = []

        self._indice = self._nombres.index(elegida) if elegida in self._nombres else 0
        self._cache = None

    def siguiente(self) -> Optional[str]:
        """Pasa a la referencia siguiente, dando la vuelta. Devuelve su nombre."""
        if len(self._nombres) > 1:
            self._indice = (self._indice + 1) % len(self._nombres)
            self._guardar_ultima()
        return self.nombre_actual

    def anterior(self) -> Optional[str]:
        if len(self._nombres) > 1:
            self._indice = (self._indice - 1) % len(self._nombres)
            self._guardar_ultima()
        return self.nombre_actual

    def agregar(self, frame: np.ndarray, nombre: Optional[str] = None) -> Path:
        """Guarda `frame` como una referencia NUEVA y la deja seleccionada.

        No pisa la anterior: agregar un modelo no puede costar perder otro. Sin nombre, se
        usa la fecha y hora, que al menos ordena las fotos por cuando se sacaron; el
        operario despues renombra el archivo con el modelo de sierra y listo.
        """
        self._carpeta.mkdir(parents=True, exist_ok=True)
        if nombre:
            base = _sanear(nombre)
        else:
            base = "modelo_" + datetime.now().strftime("%Y%m%d_%H%M%S")

        destino = self._carpeta / f"{base}.jpg"
        repeticion = 1
        while destino.exists():
            destino = self._carpeta / f"{base}_{repeticion}.jpg"
            repeticion += 1

        if not cv2.imwrite(str(destino), frame, [cv2.IMWRITE_JPEG_QUALITY, self._calidad]):
            raise OSError(f"no se pudo escribir {destino}")

        self.recargar(preferida=destino.name)
        self._guardar_ultima()
        return destino

    # ---------------------------------------------------------------- memoria
    def _leer_ultima(self) -> Optional[str]:
        archivo = self._carpeta / _ARCHIVO_ULTIMA
        if not archivo.is_file():
            return None
        try:
            return archivo.read_text(encoding="utf-8").strip() or None
        except OSError:
            return None

    def _guardar_ultima(self) -> None:
        """Recuerda la eleccion. Que falle no es grave: solo se pierde la memoria."""
        if not self._nombres:
            return
        try:
            self._carpeta.mkdir(parents=True, exist_ok=True)
            (self._carpeta / _ARCHIVO_ULTIMA).write_text(
                self._nombres[self._indice], encoding="utf-8"
            )
        except OSError as error:
            _log.debug("no se pudo recordar la referencia elegida: %s", error)


def _sanear(nombre: str) -> str:
    """Deja un nombre de archivo usable, sin caracteres que Windows rechace."""
    limpio = "".join(c if c.isalnum() or c in "-_ " else "_" for c in nombre).strip()
    return limpio.replace(" ", "_") or "modelo"
