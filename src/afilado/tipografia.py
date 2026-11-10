"""Texto con tipografia de verdad sobre imagenes de OpenCV.

POR QUE EXISTE ESTE MODULO
--------------------------
cv2.putText solo sabe dibujar las fuentes Hershey: trazos vectoriales de los anos 60,
pensados para plotters. Se ven como un plano tecnico, no tienen acentos --cada tilde sale
como '?'-- y a tamano grande se notan angulosas. En una pantalla que el operario mira todo
el dia, eso importa.

Pillow sabe dibujar TrueType. Este modulo envuelve las dos opciones detras de la misma
interfaz y elige sola: si Pillow y una fuente del sistema estan disponibles, usa esa; si no,
cae a Hershey y el programa sigue funcionando igual, solo que mas feo.

ESA CAIDA NO ES TEORICA: el .exe del taller se arma desde un entorno liviano, y si algun
dia alguien lo rearma sin Pillow, la pantalla tiene que seguir siendo legible en vez de
reventar al primer texto.

CONVERSION DE IMAGEN
--------------------
Dibujar con Pillow obliga a pasar el frame de BGR (OpenCV) a RGB (Pillow) y volver. Para
evitar hacerlo una vez por cada texto --serian veinte conversiones de 1280x672 por frame--
se dibuja TODO el texto de una pasada: se abre un lienzo de Pillow, se dibujan las lineas
que hagan falta y se vuelve a OpenCV una sola vez.
"""

from __future__ import annotations

import os
from typing import Optional

import cv2
import numpy as np

try:  # Pillow es opcional: sin el, todo cae a Hershey
    from PIL import Image, ImageDraw, ImageFont

    _HAY_PILLOW = True
except ImportError:  # pragma: no cover - depende del entorno
    _HAY_PILLOW = False

# Fuentes de Windows en orden de preferencia. Segoe UI viene con Windows desde Vista, asi
# que en la PC del taller va a estar; las demas son redes de contencion.
_CANDIDATAS_NORMAL = ("segoeui.ttf", "calibri.ttf", "arial.ttf", "tahoma.ttf", "verdana.ttf")
_CANDIDATAS_NEGRITA = ("segoeuib.ttf", "calibrib.ttf", "arialbd.ttf", "tahomabd.ttf")

_FUENTE_HERSHEY = cv2.FONT_HERSHEY_DUPLEX
# Cuantos px de alto da Hershey por unidad de escala. Sirve para traducir un tamano en px
# a la escala que espera cv2.putText, y que el texto mida parecido con las dos rutas.
_PX_POR_ESCALA_HERSHEY = 22.0

_cache: dict[tuple[str, int, bool], object] = {}


def hay_tipografia() -> bool:
    """True si se va a dibujar con TrueType (Pillow + una fuente encontrada)."""
    return _HAY_PILLOW and _ruta_fuente(False) is not None


def _ruta_fuente(negrita: bool) -> Optional[str]:
    carpeta = os.path.join(os.environ.get("WINDIR", r"C:\Windows"), "Fonts")
    for nombre in _CANDIDATAS_NEGRITA if negrita else _CANDIDATAS_NORMAL:
        ruta = os.path.join(carpeta, nombre)
        if os.path.isfile(ruta):
            return ruta
    # Fuera de Windows se prueban las rutas tipicas de Linux, por si alguna vez corre ahi.
    for ruta in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if negrita else
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    ):
        if os.path.isfile(ruta):
            return ruta
    return None


def _fuente(tamano: int, negrita: bool):
    clave = ("ttf", tamano, negrita)
    if clave in _cache:
        return _cache[clave]
    ruta = _ruta_fuente(negrita)
    objeto = None
    if ruta is not None:
        try:
            objeto = ImageFont.truetype(ruta, tamano)
        except OSError:  # pragma: no cover - fuente corrupta
            objeto = None
    _cache[clave] = objeto
    return objeto


def medir(texto: str, tamano: int, negrita: bool = False) -> tuple[int, int]:
    """(ancho, alto) en px que va a ocupar el texto."""
    if not texto:
        return (0, 0)
    fuente = _fuente(tamano, negrita) if _HAY_PILLOW else None
    if fuente is not None:
        izq, arriba, der, abajo = fuente.getbbox(texto)
        return (int(der - izq), int(abajo - arriba))
    escala = tamano / _PX_POR_ESCALA_HERSHEY
    (ancho, alto), _ = cv2.getTextSize(texto, _FUENTE_HERSHEY, escala, max(1, int(escala * 1.5)))
    return (int(ancho), int(alto))


class Pincel:
    """Dibuja varios textos sobre una imagen BGR con UNA sola conversion de color.

    Se usa como contexto:

        with Pincel(lienzo) as pincel:
            pincel.texto("APROBADO", (40, 100), (90, 200, 90), 52, negrita=True)

    Al salir del bloque, el lienzo recibe todo lo dibujado. Si Pillow no esta, cada texto
    se dibuja directo con cv2.putText y el contexto no hace nada especial.
    """

    def __init__(self, imagen: np.ndarray) -> None:
        self._imagen = imagen
        self._pil = None
        self._dibujo = None

    def __enter__(self) -> "Pincel":
        if _HAY_PILLOW and _fuente(20, False) is not None:
            self._pil = Image.fromarray(cv2.cvtColor(self._imagen, cv2.COLOR_BGR2RGB))
            self._dibujo = ImageDraw.Draw(self._pil)
        return self

    def __exit__(self, *_excepcion: object) -> None:
        if self._pil is not None:
            self._imagen[:, :] = cv2.cvtColor(np.array(self._pil), cv2.COLOR_RGB2BGR)

    def texto(
        self,
        texto: str,
        org: tuple[int, int],
        color: tuple[int, int, int],
        tamano: int,
        negrita: bool = False,
    ) -> None:
        """Escribe 'texto' con su esquina SUPERIOR IZQUIERDA en 'org'.

        Se usa la esquina y no la linea base --que es lo que espera cv2.putText-- porque
        apilar y centrar bloques es sumar altos, sin arrastrar el descuento de la baseline
        en cada cuenta.
        """
        if not texto:
            return
        if self._dibujo is not None:
            fuente = _fuente(tamano, negrita)
            if fuente is not None:
                izq, arriba, _, _ = fuente.getbbox(texto)
                self._dibujo.text(
                    (org[0] - izq, org[1] - arriba),
                    texto,
                    font=fuente,
                    fill=(int(color[2]), int(color[1]), int(color[0])),  # BGR -> RGB
                )
                return
        escala = tamano / _PX_POR_ESCALA_HERSHEY
        grosor = max(1, int(escala * 1.5))
        _, alto = medir(texto, tamano, negrita)
        cv2.putText(
            self._imagen, texto, (org[0], org[1] + alto), _FUENTE_HERSHEY, escala,
            color, grosor, cv2.LINE_AA,
        )
