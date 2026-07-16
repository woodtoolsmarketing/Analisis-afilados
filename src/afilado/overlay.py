"""Dibujo de la capa de informacion sobre el frame (HUD, cajas, etiquetas, avisos).

Este modulo NUNCA muestra ventanas: solo devuelve una imagen nueva. La ventana la abre
run_live. Y NUNCA escribe sobre el frame recibido: la imagen limpia es la materia prima del
reentrenamiento y no puede llevar ni un pixel dibujado.

Todo el texto se dibuja con un rectangulo relleno detras (texto_con_fondo). Sobre widia
pulido o diamante hay reflejos blancos que dejan ilegible cualquier texto sin fondo, y el
operario tiene que poder leer las medidas de un vistazo mientras sostiene la pieza.
"""

from __future__ import annotations

from typing import Optional, Sequence

import cv2
import numpy as np

from .config import AppConfig
from .tipos import Deteccion, ResultadoFrame

_FUENTE = cv2.FONT_HERSHEY_SIMPLEX
_MARGEN = 3          # px de aire entre el texto y el borde de su fondo
_INTERLINEA = 6      # px entre lineas apiladas del HUD

_VERDE = (0, 200, 0)
_ROJO = (0, 0, 235)
_AMARILLO = (0, 215, 255)
_GRIS = (170, 170, 170)
_CIAN = (255, 255, 0)
_NEGRO = (0, 0, 0)
_BLANCO = (255, 255, 255)

_CLASE_SIN_CLASIFICAR = "sin_clasificar"

_AYUDA: tuple[str, ...] = (
    "q/ESC salir | e marcar ERROR de la IA | g guardar ejemplo BUENO | espacio capturar",
    "r recalibrar ArUco | p pausa | c ciclar ROI | h ocultar/mostrar ayuda",
)


def _color_texto(fondo: tuple[int, int, int]) -> tuple[int, int, int]:
    """Negro sobre fondos claros, blanco sobre oscuros (luminancia ITU-R BT.601, BGR)."""
    azul, verde, rojo = fondo
    luminancia = 0.114 * azul + 0.587 * verde + 0.299 * rojo
    return _NEGRO if luminancia > 140 else _BLANCO


def _tamano_texto(texto: str, escala_fuente: float) -> tuple[int, int, int]:
    """Devuelve (ancho, alto, grosor) del bloque con fondo que ocuparia el texto."""
    grosor = max(1, int(round(escala_fuente * 2)))
    (ancho, alto), base = cv2.getTextSize(texto, _FUENTE, escala_fuente, grosor)
    return ancho + 2 * _MARGEN, alto + base + 2 * _MARGEN, grosor


def texto_con_fondo(
    img: np.ndarray,
    texto: str,
    org: tuple[int, int],
    color: tuple[int, int, int],
    escala_fuente: float = 0.5,
) -> None:
    """Escribe 'texto' con un rectangulo relleno de 'color' detras.

    'org' es la esquina SUPERIOR IZQUIERDA del bloque, no la baseline de cv2.putText: asi
    apilar lineas es sumar altos, sin arrastrar el descuento de la baseline en cada llamada.
    El bloque se recorta contra los bordes de 'img': un texto fuera del frame no se ve.
    Modifica 'img' en el lugar.
    """
    if not texto:
        return
    alto_img, ancho_img = img.shape[:2]
    ancho_bloque, alto_bloque, grosor = _tamano_texto(texto, escala_fuente)

    x = max(0, min(int(org[0]), max(0, ancho_img - ancho_bloque)))
    y = max(0, min(int(org[1]), max(0, alto_img - alto_bloque)))
    x2 = min(ancho_img, x + ancho_bloque)
    y2 = min(alto_img, y + alto_bloque)
    if x2 <= x or y2 <= y:
        return

    cv2.rectangle(img, (x, y), (x2, y2), color, cv2.FILLED)
    base_y = y + alto_bloque - _MARGEN - max(1, int(round(escala_fuente * 4)))
    cv2.putText(
        img,
        texto,
        (x + _MARGEN, base_y),
        _FUENTE,
        escala_fuente,
        _color_texto(color),
        grosor,
        cv2.LINE_AA,
    )


def color_de_clase(clase: str, cfg: AppConfig) -> tuple[int, int, int]:
    """Verde para clases correctas, rojo para defectos, amarillo para 'sin_clasificar'."""
    nombre = (clase or "").strip().lower()
    if nombre == _CLASE_SIN_CLASIFICAR:
        return _AMARILLO
    if nombre in {c.strip().lower() for c in cfg.clases_defecto}:
        return _ROJO
    if nombre in {c.strip().lower() for c in cfg.clases}:
        return _VERDE
    return _GRIS


def _etiqueta_de(det: Deteccion) -> str:
    """Ej: 'desgastado 92% | 20.4 x 5.1 mm'. Cae a px si no hay escala fiable."""
    partes = [det.clase]
    if det.confianza > 0.0:
        partes[0] = f"{det.clase} {det.confianza * 100:.0f}%"
    medida = det.medida
    if medida is not None:
        if medida.fiable and medida.largo_mm is not None and medida.ancho_mm is not None:
            partes.append(f"{medida.largo_mm:.1f} x {medida.ancho_mm:.1f} mm")
        else:
            partes.append(f"{medida.largo_px:.0f} x {medida.ancho_px:.0f} px")
    return " | ".join(partes)


def _puntos_de(det: Deteccion) -> np.ndarray:
    """Poligono a dibujar: la caja rotada si hay medida, si no la caja recta del detector."""
    if det.medida is not None:
        return np.asarray(det.medida.caja_rotada_px, dtype=np.int32).reshape(-1, 1, 2)
    x1, y1, x2, y2 = (int(round(v)) for v in det.xyxy)
    esquinas = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
    return np.asarray(esquinas, dtype=np.int32).reshape(-1, 1, 2)


def _dibujar_etiqueta(
    img: np.ndarray, texto: str, puntos: np.ndarray, color: tuple[int, int, int]
) -> None:
    """Coloca la etiqueta arriba de la caja; si no entra, adentro de la caja."""
    xs = puntos[:, 0, 0]
    ys = puntos[:, 0, 1]
    x = int(xs.min())
    y_arriba = int(ys.min())
    _, alto_bloque, _ = _tamano_texto(texto, 0.5)
    y = y_arriba - alto_bloque - 2
    if y < 0:
        # La pieza toca el borde superior del frame: la etiqueta afuera no se veria.
        y = y_arriba + 2
    texto_con_fondo(img, texto, (x, y), color, 0.5)


def _dibujar_roi(img: np.ndarray, roi: tuple[int, int, int, int]) -> None:
    x1, y1, x2, y2 = (int(v) for v in roi)
    if x2 <= x1 or y2 <= y1:
        return
    alto, ancho = img.shape[:2]
    if (x1, y1, x2, y2) == (0, 0, ancho, alto):
        return
    cv2.rectangle(img, (x1, y1), (x2 - 1, y2 - 1), _GRIS, 1)
    texto_con_fondo(img, "ROI", (x1, y1 - 18), _GRIS, 0.4)


def _dibujar_aruco(img: np.ndarray, esquinas: Optional[np.ndarray], id_marcador) -> None:
    if esquinas is None:
        return
    puntos = np.asarray(esquinas, dtype=np.int32).reshape(-1, 1, 2)
    if puntos.shape[0] < 3:
        return
    cv2.polylines(img, [puntos], True, _CIAN, 2, cv2.LINE_AA)
    etiqueta = "ArUco" if id_marcador is None else f"ArUco #{id_marcador}"
    x = int(puntos[:, 0, 0].min())
    y = int(puntos[:, 0, 1].min()) - 18
    texto_con_fondo(img, etiqueta, (x, y), _CIAN, 0.4)


def _lineas_hud(resultado: ResultadoFrame, capturas: int) -> list[str]:
    escala = resultado.escala
    if escala.valida and escala.mm_por_px:
        detalle = f"{escala.mm_por_px:.4f} mm/px ({escala.fuente})"
    else:
        detalle = "SIN ESCALA"
    return [
        f"FPS {resultado.fps:.1f}",
        f"Modelo: {resultado.modelo}",
        f"Escala: {detalle}",
        f"Piezas: {len(resultado.detecciones)}  descartadas: {len(resultado.descartadas)}",
        f"Capturas guardadas: {capturas}",
    ]


def _dibujar_hud(img: np.ndarray, lineas: Sequence[str]) -> int:
    """Apila el HUD arriba a la izquierda. Devuelve la y libre debajo del bloque."""
    y = 8
    for linea in lineas:
        _, alto_bloque, _ = _tamano_texto(linea, 0.5)
        texto_con_fondo(img, linea, (8, y), _NEGRO, 0.5)
        y += alto_bloque + 2
    return y


def _dibujar_ayuda(img: np.ndarray, lineas: Sequence[str]) -> None:
    """Apila la ayuda de teclas abajo a la izquierda, de la ultima linea hacia arriba."""
    alto = img.shape[0]
    y = alto - 8
    for linea in reversed(lineas):
        _, alto_bloque, _ = _tamano_texto(linea, 0.45)
        y -= alto_bloque + 2
        texto_con_fondo(img, linea, (8, y), _NEGRO, 0.45)


def dibujar(
    frame: np.ndarray,
    resultado: ResultadoFrame,
    cfg: AppConfig,
    mostrar_ayuda: bool = True,
    mensaje: Optional[str] = None,
) -> np.ndarray:
    """Devuelve una COPIA anotada del frame. El frame original no se toca.

    'mensaje' es la confirmacion efimera que run_live muestra tras guardar una captura.
    El contador de capturas se lee de resultado.capturas_guardadas si run_live lo adjunta;
    ResultadoFrame es contrato cerrado y no puede declarar el campo.
    """
    lienzo = frame.copy()
    capturas = int(getattr(resultado, "capturas_guardadas", 0) or 0)

    _dibujar_roi(lienzo, resultado.roi)
    _dibujar_aruco(lienzo, resultado.escala.esquinas, resultado.escala.id_marcador)

    for det in resultado.detecciones:
        color = color_de_clase(det.clase, cfg)
        puntos = _puntos_de(det)
        cv2.drawContours(lienzo, [puntos], -1, color, 2, cv2.LINE_AA)
        _dibujar_etiqueta(lienzo, _etiqueta_de(det), puntos, color)

    y = _dibujar_hud(lienzo, _lineas_hud(resultado, capturas))

    avisos: list[str] = []
    if not resultado.escala.valida:
        avisos.append("SIN ESCALA - medidas no fiables")
    if resultado.escala.error_lados > cfg.aruco.max_error_lados:
        avisos.append("CAMARA INCLINADA - enderece el cabezal")
    for aviso in avisos:
        _, alto_bloque, _ = _tamano_texto(aviso, 0.6)
        texto_con_fondo(lienzo, aviso, (8, y + 4), _ROJO, 0.6)
        y += alto_bloque + _INTERLINEA

    if mensaje:
        ancho_bloque, _, _ = _tamano_texto(mensaje, 0.7)
        x = max(0, (lienzo.shape[1] - ancho_bloque) // 2)
        texto_con_fondo(lienzo, mensaje, (x, y + _INTERLINEA), _VERDE, 0.7)

    if mostrar_ayuda:
        _dibujar_ayuda(lienzo, _AYUDA)

    return lienzo
