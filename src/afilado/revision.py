"""Pantalla de revision: lo que ve la camara al lado de una pieza aprobada.

POR QUE ESTA PANTALLA Y NO LA DE SIEMPRE
----------------------------------------
Con el microscopio, la imagen solo sirve cuando el diente esta bien encuadrado y enfocado.
Juzgar todos los frames --como hace el modo clasico-- significa juzgar sobre todo imagenes
movidas, a medio encuadrar, de la mesa o del dedo del operario. El veredicto parpadearia
sin sentido y el operario aprenderia a ignorarlo, que es la peor forma de perder una
herramienta de control.

Aca el programa no opina hasta que se lo piden: el operario acomoda la pieza, aprieta
REVISAR, y RECIEN AHI se congela ese frame y se analiza. Lo que queda en pantalla es una
foto quieta con su veredicto, que se puede mirar con calma y comparar con la referencia de
al lado.

Este modulo NO abre ventanas ni lee teclas: solo compone la imagen y devuelve donde quedaron
los botones, para que run_live resuelva los clics. Asi se puede probar sin pantalla.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np

from .config import AppConfig
from .overlay import _color_texto, _tamano_texto, texto_con_fondo
from .tipos import Veredicto

# Ancho al que se compone la pantalla. Fijo a proposito: el monitor del taller es chico y
# una pantalla de 1620 px --dos paneles de 800 lado a lado-- no entraria. Cada panel se
# escala para entrar aca, sea cual sea la resolucion de la camara.
ANCHO_LIENZO = 1280

_MARGEN = 12
_ALTO_BANDA = 78          # banda de veredicto arriba
_ALTO_BARRA = 104         # barra de abajo: una fila de botones y una linea de ayuda

_VERDE = (0, 200, 0)
_ROJO = (0, 0, 235)
_GRIS = (60, 60, 60)
_GRIS_CLARO = (170, 170, 170)
_NEGRO = (0, 0, 0)
_BLANCO = (255, 255, 255)
_AMARILLO = (0, 215, 255)
_FUENTE = cv2.FONT_HERSHEY_SIMPLEX

Rect = tuple[int, int, int, int]  # x1, y1, x2, y2


@dataclass
class EstadoRevision:
    """Todo lo que la pantalla necesita saber para dibujarse."""

    congelado: bool = False
    veredicto: Optional[Veredicto] = None
    mensaje: Optional[str] = None
    capturas: int = 0
    referencia_faltante: bool = False


@dataclass
class Pantalla:
    """La imagen compuesta y donde quedaron los botones."""

    imagen: np.ndarray
    botones: dict[str, Rect] = field(default_factory=dict)

    def boton_en(self, x: int, y: int) -> Optional[str]:
        """Devuelve el nombre del boton que hay en (x, y), o None."""
        for nombre, (x1, y1, x2, y2) in self.botones.items():
            if x1 <= x < x2 and y1 <= y < y2:
                return nombre
        return None


def _encajar(imagen: np.ndarray, ancho: int, alto: int) -> np.ndarray:
    """Escala la imagen para entrar en ancho x alto SIN deformarla, con bandas negras.

    Deformar la imagen del microscopio para llenar el panel haria que el operario compare
    un diente estirado contra una referencia sin estirar. La proporcion se respeta.
    """
    panel = np.zeros((alto, ancho, 3), np.uint8)
    if imagen is None or imagen.size == 0:
        return panel
    if imagen.ndim == 2:
        imagen = cv2.cvtColor(imagen, cv2.COLOR_GRAY2BGR)

    alto_img, ancho_img = imagen.shape[:2]
    escala = min(ancho / ancho_img, alto / alto_img)
    nuevo = (max(1, int(ancho_img * escala)), max(1, int(alto_img * escala)))
    interpolacion = cv2.INTER_AREA if escala < 1.0 else cv2.INTER_LINEAR
    redimensionada = cv2.resize(imagen, nuevo, interpolation=interpolacion)

    x0 = (ancho - nuevo[0]) // 2
    y0 = (alto - nuevo[1]) // 2
    panel[y0 : y0 + nuevo[1], x0 : x0 + nuevo[0]] = redimensionada
    return panel


def _titulo_panel(lienzo: np.ndarray, texto: str, x: int, y: int, color) -> None:
    texto_con_fondo(lienzo, texto, (x, y), color, 0.6)


_RELLENO_BOTON = 22       # px de aire a cada lado del texto dentro del boton


def _dibujar_boton(
    lienzo: np.ndarray, x: int, y1: int, y2: int, texto: str, color, escala: float = 0.85
) -> Rect:
    """Dibuja un boton que se ADAPTA a su texto y devuelve el rectangulo que ocupo.

    El ancho sale de medir el texto, no de un numero fijo: con anchos fijos, cambiar
    "APROBADA" por "GUARDAR APROBADA" hacia que las letras se salieran de la caja y se
    montaran sobre el boton de al lado, que es exactamente lo que no puede pasar en un
    boton que decide si una sierra se tira o no.
    """
    ancho_texto, alto_texto, grosor = _tamano_texto(texto, escala)
    x2 = x + ancho_texto + 2 * _RELLENO_BOTON
    cv2.rectangle(lienzo, (x, y1), (x2, y2), color, cv2.FILLED)
    cv2.rectangle(lienzo, (x, y1), (x2, y2), _BLANCO, 2)
    cy = y1 + (y2 - y1 + alto_texto) // 2 - 6
    cv2.putText(
        lienzo,
        texto,
        (x + _RELLENO_BOTON, cy),
        _FUENTE,
        escala,
        _color_texto(color),
        grosor,
        cv2.LINE_AA,
    )
    return (x, y1, x2, y2)


def _banda_veredicto(lienzo: np.ndarray, estado: EstadoRevision, cfg: AppConfig) -> None:
    ancho = lienzo.shape[1]
    if not estado.congelado:
        cv2.rectangle(lienzo, (0, 0), (ancho, _ALTO_BANDA), _GRIS, cv2.FILLED)
        cv2.putText(
            lienzo,
            "ACOMODE LA PIEZA Y APRIETE REVISAR",
            (16, 50),
            _FUENTE,
            0.9,
            _GRIS_CLARO,
            2,
            cv2.LINE_AA,
        )
        return

    veredicto = estado.veredicto
    if veredicto is None:
        cv2.rectangle(lienzo, (0, 0), (ancho, _ALTO_BANDA), _AMARILLO, cv2.FILLED)
        cv2.putText(
            lienzo,
            "SIN MODELO - no hay veredicto todavia",
            (16, 52),
            _FUENTE,
            0.9,
            _NEGRO,
            2,
            cv2.LINE_AA,
        )
        return

    es_defecto = veredicto.clase in cfg.clases_defecto
    color = _ROJO if es_defecto else _VERDE
    cv2.rectangle(lienzo, (0, 0), (ancho, _ALTO_BANDA), color, cv2.FILLED)
    titulo = f"{veredicto.clase.upper()}   {veredicto.confianza * 100:.0f}%"
    cv2.putText(lienzo, titulo, (16, 56), _FUENTE, 1.3, _color_texto(color), 3, cv2.LINE_AA)

    if veredicto.probabilidades:
        partes = [
            f"{nombre}: {valor * 100:.0f}%"
            for nombre, valor in sorted(
                veredicto.probabilidades.items(), key=lambda kv: kv[1], reverse=True
            )
        ]
        detalle = "   |   ".join(partes)
        ancho_detalle, _, _ = _tamano_texto(detalle, 0.6)
        cv2.putText(
            lienzo,
            detalle,
            (max(16, ancho - ancho_detalle - 16), 50),
            _FUENTE,
            0.6,
            _color_texto(color),
            2,
            cv2.LINE_AA,
        )


def componer(
    vivo: np.ndarray,
    referencia: Optional[np.ndarray],
    cfg: AppConfig,
    estado: EstadoRevision,
) -> Pantalla:
    """Arma la pantalla de revision. No modifica ninguna de las imagenes que recibe."""
    ancho_panel = (ANCHO_LIENZO - 3 * _MARGEN) // 2
    alto_img, ancho_img = vivo.shape[:2]
    alto_panel = max(240, int(ancho_panel * alto_img / max(1, ancho_img)))

    alto_total = _ALTO_BANDA + alto_panel + 2 * _MARGEN + _ALTO_BARRA
    lienzo = np.full((alto_total, ANCHO_LIENZO, 3), 28, np.uint8)

    y_panel = _ALTO_BANDA + _MARGEN
    x_izq, x_der = _MARGEN, 2 * _MARGEN + ancho_panel

    lienzo[y_panel : y_panel + alto_panel, x_izq : x_izq + ancho_panel] = _encajar(
        vivo, ancho_panel, alto_panel
    )

    if referencia is not None:
        lienzo[y_panel : y_panel + alto_panel, x_der : x_der + ancho_panel] = _encajar(
            referencia, ancho_panel, alto_panel
        )
    else:
        # Sin referencia se explica que falta y como conseguirla, en vez de un hueco negro
        # que parece un error del programa.
        cv2.rectangle(
            lienzo,
            (x_der, y_panel),
            (x_der + ancho_panel, y_panel + alto_panel),
            (45, 45, 45),
            cv2.FILLED,
        )
        for indice, linea in enumerate(
            (
                "Falta la foto de referencia.",
                "",
                "Pone una pieza APROBADA bajo el",
                "microscopio y apreta la tecla  f",
                "para guardarla como referencia.",
            )
        ):
            if not linea:
                continue
            cv2.putText(
                lienzo,
                linea,
                (x_der + 24, y_panel + 70 + indice * 34),
                _FUENTE,
                0.62,
                _GRIS_CLARO,
                1,
                cv2.LINE_AA,
            )

    _banda_veredicto(lienzo, estado, cfg)

    etiqueta_izq = "PIEZA CONGELADA" if estado.congelado else "EN VIVO"
    _titulo_panel(lienzo, etiqueta_izq, x_izq + 6, y_panel + 6, _NEGRO)
    _titulo_panel(lienzo, "REFERENCIA APROBADA", x_der + 6, y_panel + 6, _NEGRO)

    # --- barra de botones ---
    y_barra = y_panel + alto_panel + _MARGEN
    cv2.rectangle(lienzo, (0, y_barra), (ANCHO_LIENZO, alto_total), (18, 18, 18), cv2.FILLED)

    alto_boton = 52
    y1 = y_barra + 8
    y2 = y1 + alto_boton
    botones: dict[str, Rect] = {}

    if estado.congelado:
        botones["seguir"] = _dibujar_boton(lienzo, _MARGEN, y1, y2, "OTRA PIEZA", _GRIS)
        x = botones["seguir"][2] + _MARGEN
        botones["aprobada"] = _dibujar_boton(lienzo, x, y1, y2, "GUARDAR APROBADA", _VERDE)
        x = botones["aprobada"][2] + _MARGEN
        botones["rechazada"] = _dibujar_boton(lienzo, x, y1, y2, "GUARDAR RECHAZADA", _ROJO)
    else:
        botones["revisar"] = _dibujar_boton(
            lienzo, _MARGEN, y1, y2, "REVISAR", (150, 90, 0), 1.0
        )

    ancho_salir, _, _ = _tamano_texto("SALIR", 0.85)
    botones["salir"] = _dibujar_boton(
        lienzo,
        ANCHO_LIENZO - _MARGEN - ancho_salir - 2 * _RELLENO_BOTON,
        y1,
        y2,
        "SALIR",
        (40, 40, 120),
    )

    # La ayuda va en su PROPIO renglon, debajo de los botones. Antes compartia la franja y
    # se montaba encima de ellos.
    ayuda = "espacio o clic = REVISAR   |   a aprobada   |   r rechazada   |   f fijar referencia   |   q salir"
    if estado.capturas:
        ayuda += f"   |   guardadas: {estado.capturas}"
    cv2.putText(
        lienzo,
        ayuda,
        (_MARGEN, y2 + 24),
        _FUENTE,
        0.46,
        _GRIS_CLARO,
        1,
        cv2.LINE_AA,
    )

    if estado.mensaje:
        ancho_msg, alto_msg, _ = _tamano_texto(estado.mensaje, 0.8)
        texto_con_fondo(
            lienzo,
            estado.mensaje,
            ((ANCHO_LIENZO - ancho_msg) // 2, y_panel + alto_panel - alto_msg - 14),
            _VERDE,
            0.8,
        )

    return Pantalla(imagen=lienzo, botones=botones)


# ---------------------------------------------------------------------------
# MODO SIMPLE: dos recuadros y un cartel RESULTADO enorme abajo.
#
# Pensado para que lo lea cualquiera, de un vistazo y desde lejos: sin
# porcentajes por clase, sin nombres de modelo, sin jerga. Dos palabras.
# ---------------------------------------------------------------------------

_ALTO_RESULTADO = 150     # el cartel de abajo, el elemento mas grande de la pantalla
_ALTO_TITULO = 42         # la franja con SIERRA / MODELO APROBADO


def coincidencia(veredicto: Optional[Veredicto], cfg: AppConfig) -> Optional[float]:
    """Cuanto se parece la pieza a una APROBADA, entre 0 y 1. None si no hay veredicto.

    QUE ES: la probabilidad que el modelo le da al conjunto de clases sanas. Como se lo
    entreno con fotos de piezas aprobadas y rechazadas, ese numero se lee como "cuanto se
    parece esto a las piezas buenas".

    QUE NO ES: una comparacion pixel a pixel contra la foto de referencia. Esa comparacion
    estaria dominada por la luz y el encuadre, no por el estado del filo --dos fotos del
    mismo diente con distinta iluminacion dan muy distinto, y dos dientes distintos con la
    misma luz dan casi igual--, y devolveria un porcentaje con cara de dato que no mide
    nada. La foto de referencia esta para que la compare el ojo del operario.

    Se suman TODAS las clases sanas, no solo la primera, para que siga andando si algun dia
    el esquema pasa de aprobada/rechazada a varios defectos con nombre propio.
    """
    if veredicto is None:
        return None
    sanas = [c for c in cfg.clases if c not in cfg.clases_defecto]
    if not sanas:
        return None
    if veredicto.probabilidades:
        return float(sum(veredicto.probabilidades.get(clase, 0.0) for clase in sanas))
    # Un veredicto sin desglose igual dice algo: si la clase que gano es sana, la confianza
    # ES la coincidencia; si gano una de defecto, lo que queda para las sanas es el resto.
    if veredicto.clase in sanas:
        return float(veredicto.confianza)
    return float(1.0 - veredicto.confianza)


def _cartel_resultado(
    lienzo: np.ndarray,
    y: int,
    estado: EstadoRevision,
    cfg: AppConfig,
) -> None:
    """Dibuja la franja RESULTADO y, adentro, APROBADO o DESAPROBADO."""
    alto_total = lienzo.shape[0]
    ancho = lienzo.shape[1]

    texto_con_fondo(lienzo, "RESULTADO", (_MARGEN, y), _NEGRO, 0.7)
    y_caja = y + 34
    caja = (_MARGEN, y_caja, ancho - _MARGEN, min(alto_total - 4, y_caja + _ALTO_RESULTADO))

    valor = coincidencia(estado.veredicto, cfg) if estado.congelado else None

    if not estado.congelado:
        color, titulo, detalle = (55, 55, 55), "ESPERANDO", "apriete REVISAR"
    elif valor is None:
        color, titulo = (0, 170, 220), "SIN MODELO"
        detalle = "no hay modelo entrenado todavia"
    elif valor >= float(cfg.interfaz.umbral_aprobacion):
        color, titulo = _VERDE, "APROBADO"
        detalle = f"coincidencia {valor * 100:.0f}%  (minimo {cfg.interfaz.umbral_aprobacion * 100:.0f}%)"
    else:
        color, titulo = _ROJO, "DESAPROBADO"
        detalle = f"coincidencia {valor * 100:.0f}%  (minimo {cfg.interfaz.umbral_aprobacion * 100:.0f}%)"

    cv2.rectangle(lienzo, (caja[0], caja[1]), (caja[2], caja[3]), color, cv2.FILLED)
    cv2.rectangle(lienzo, (caja[0], caja[1]), (caja[2], caja[3]), _BLANCO, 3)

    escala_titulo = 2.6
    ancho_t, alto_t, grosor_t = _tamano_texto(titulo, escala_titulo)
    # Si la palabra no entra a lo ancho, se achica hasta que entre: "DESAPROBADO" es larga
    # y el lienzo depende de la camara. Un cartel cortado por el borde no se puede leer.
    while ancho_t > (caja[2] - caja[0]) - 40 and escala_titulo > 0.8:
        escala_titulo -= 0.1
        ancho_t, alto_t, grosor_t = _tamano_texto(titulo, escala_titulo)

    cx = caja[0] + ((caja[2] - caja[0]) - ancho_t) // 2 + 3
    cy = caja[1] + ((caja[3] - caja[1]) + alto_t) // 2 - 22
    cv2.putText(
        lienzo, titulo, (cx, cy), _FUENTE, escala_titulo, _color_texto(color), grosor_t + 1,
        cv2.LINE_AA,
    )
    ancho_d, _, _ = _tamano_texto(detalle, 0.66)
    cv2.putText(
        lienzo,
        detalle,
        (caja[0] + ((caja[2] - caja[0]) - ancho_d) // 2 + 3, caja[3] - 18),
        _FUENTE,
        0.66,
        _color_texto(color),
        2,
        cv2.LINE_AA,
    )


def componer_simple(
    vivo: np.ndarray,
    referencia: Optional[np.ndarray],
    cfg: AppConfig,
    estado: EstadoRevision,
) -> Pantalla:
    """La pantalla del taller: la sierra, el modelo, y abajo el resultado en dos palabras."""
    ancho_panel = (ANCHO_LIENZO - 3 * _MARGEN) // 2
    alto_img, ancho_img = vivo.shape[:2]
    alto_panel = max(220, int(ancho_panel * alto_img / max(1, ancho_img)))

    alto_total = (
        _ALTO_TITULO + alto_panel + _MARGEN + 34 + _ALTO_RESULTADO + _MARGEN + 70
    )
    lienzo = np.full((alto_total, ANCHO_LIENZO, 3), 24, np.uint8)

    x_izq, x_der = _MARGEN, 2 * _MARGEN + ancho_panel

    # --- titulos de cada recuadro, grandes ---
    for x, titulo in ((x_izq, "SIERRA"), (x_der, "MODELO APROBADO")):
        ancho_t, alto_t, _ = _tamano_texto(titulo, 0.9)
        cv2.putText(
            lienzo,
            titulo,
            (x + max(0, (ancho_panel - ancho_t) // 2), _ALTO_TITULO - 12),
            _FUENTE,
            0.9,
            _BLANCO,
            2,
            cv2.LINE_AA,
        )

    y_panel = _ALTO_TITULO
    lienzo[y_panel : y_panel + alto_panel, x_izq : x_izq + ancho_panel] = _encajar(
        vivo, ancho_panel, alto_panel
    )
    if referencia is not None:
        lienzo[y_panel : y_panel + alto_panel, x_der : x_der + ancho_panel] = _encajar(
            referencia, ancho_panel, alto_panel
        )
    else:
        cv2.rectangle(
            lienzo,
            (x_der, y_panel),
            (x_der + ancho_panel, y_panel + alto_panel),
            (45, 45, 45),
            cv2.FILLED,
        )
        for indice, linea in enumerate(
            ("Falta la foto del modelo.", "", "Pone una pieza buena y", "apreta la tecla  f")
        ):
            if not linea:
                continue
            cv2.putText(
                lienzo,
                linea,
                (x_der + 24, y_panel + 60 + indice * 40),
                _FUENTE,
                0.75,
                _GRIS_CLARO,
                2,
                cv2.LINE_AA,
            )

    if estado.congelado:
        texto_con_fondo(lienzo, "CONGELADA", (x_izq + 6, y_panel + 6), _NEGRO, 0.6)

    # --- el cartel RESULTADO, abajo de los DOS recuadros ---
    _cartel_resultado(lienzo, y_panel + alto_panel + _MARGEN, estado, cfg)

    # --- botones ---
    y1 = alto_total - 62
    y2 = y1 + 50
    botones: dict[str, Rect] = {}
    if estado.congelado:
        botones["seguir"] = _dibujar_boton(lienzo, _MARGEN, y1, y2, "OTRA PIEZA", _GRIS)
        x = botones["seguir"][2] + _MARGEN
        botones["aprobada"] = _dibujar_boton(lienzo, x, y1, y2, "GUARDAR BUENA", _VERDE)
        x = botones["aprobada"][2] + _MARGEN
        botones["rechazada"] = _dibujar_boton(lienzo, x, y1, y2, "GUARDAR MALA", _ROJO)
    else:
        botones["revisar"] = _dibujar_boton(
            lienzo, _MARGEN, y1, y2, "REVISAR", (150, 90, 0), 1.0
        )

    ancho_salir, _, _ = _tamano_texto("SALIR", 0.85)
    botones["salir"] = _dibujar_boton(
        lienzo,
        ANCHO_LIENZO - _MARGEN - ancho_salir - 2 * _RELLENO_BOTON,
        y1,
        y2,
        "SALIR",
        (40, 40, 120),
    )

    if estado.mensaje:
        ancho_msg, alto_msg, _ = _tamano_texto(estado.mensaje, 0.7)
        texto_con_fondo(
            lienzo,
            estado.mensaje,
            ((ANCHO_LIENZO - ancho_msg) // 2, y_panel + alto_panel - alto_msg - 10),
            _VERDE,
            0.7,
        )

    return Pantalla(imagen=lienzo, botones=botones)
