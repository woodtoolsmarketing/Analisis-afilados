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

from . import tipografia
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
    # Nombre del modelo de referencia que se esta mostrando, y cual es de cuantos. Sin esto
    # el operario no sabe contra que esta comparando, que con varios modelos cargados es
    # justamente lo que necesita saber.
    nombre_referencia: Optional[str] = None
    posicion_referencia: tuple[int, int] = (0, 0)


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


# Todo lo que ocupa el lienzo ADEMAS de los recuadros de imagen: titulos, la palabra
# RESULTADO, el cartel y la fila de botones. Se usa para despejar cuanto alto le queda a los
# recuadros cuando hay que entrar en una pantalla chica.
# Medidas de la pantalla simple. Estan juntas aca porque el alto total depende de ellas:
# cuando el lienzo no entra en el monitor, lo que se achica es el RECUADRO de imagen y
# estas franjas se respetan enteras.
_ALTO_CABECERA = 46       # la fila con SIERRA / MODELO APROBADO
_ALTO_RESULTADO = 136     # la tarjeta del veredicto
_ALTO_BOTONES = 76        # la fila de botones
_MARGEN_SIMPLE = 18

_ALTO_FIJO_SIMPLE = (
    _ALTO_CABECERA + 3 * _MARGEN_SIMPLE + _ALTO_RESULTADO + _ALTO_BOTONES
)

# Por debajo de esto los recuadros no muestran nada util: antes que seguir achicandolos
# --o peor, achicar el cartel de resultado y los botones, que es lo que hay que leer y
# apretar-- la pantalla se queda en este piso aunque el monitor sea mas chico.
_ALTO_MINIMO_PANEL = 180

# Alto minimo del lienzo del modo simple. Es el piso real: pedirle menos no lo achica mas.
ALTO_MINIMO_LIENZO = _ALTO_FIJO_SIMPLE + _ALTO_MINIMO_PANEL


# ---------------------------------------------------------------------------
# PALETA Y PRIMITIVAS DEL REDISENO
# ---------------------------------------------------------------------------

# Todo en BGR, que es como los quiere OpenCV.
_FONDO = (26, 21, 17)            # azul petroleo muy oscuro, no negro puro
_SUPERFICIE = (40, 33, 28)       # las tarjetas, apenas mas claras que el fondo
_BORDE = (62, 52, 44)            # borde sutil, para separar sin gritar
_TEXTO = (240, 238, 232)
_TEXTO_TENUE = (168, 155, 139)
_VERDE_OK = (88, 170, 70)        # verde contenido: se lee de lejos sin vibrar
_ROJO_NO = (60, 50, 220)
_AMBAR = (20, 160, 235)
_AZUL = (200, 130, 40)
_FUENTE_TITULO = cv2.FONT_HERSHEY_DUPLEX   # doble trazo: menos "dibujo tecnico"


def _rect_redondeado(
    img: np.ndarray,
    pt1: tuple[int, int],
    pt2: tuple[int, int],
    color,
    radio: int = 12,
    grosor: int = -1,
) -> None:
    """Rectangulo con esquinas redondeadas. OpenCV no trae uno."""
    x1, y1 = pt1
    x2, y2 = pt2
    radio = max(0, min(radio, (x2 - x1) // 2, (y2 - y1) // 2))
    if radio == 0:
        cv2.rectangle(img, pt1, pt2, color, grosor, cv2.LINE_AA)
        return
    if grosor < 0:
        cv2.rectangle(img, (x1 + radio, y1), (x2 - radio, y2), color, -1, cv2.LINE_AA)
        cv2.rectangle(img, (x1, y1 + radio), (x2, y2 - radio), color, -1, cv2.LINE_AA)
    else:
        cv2.line(img, (x1 + radio, y1), (x2 - radio, y1), color, grosor, cv2.LINE_AA)
        cv2.line(img, (x1 + radio, y2), (x2 - radio, y2), color, grosor, cv2.LINE_AA)
        cv2.line(img, (x1, y1 + radio), (x1, y2 - radio), color, grosor, cv2.LINE_AA)
        cv2.line(img, (x2, y1 + radio), (x2, y2 - radio), color, grosor, cv2.LINE_AA)
    for cx, cy, a1, a2 in (
        (x1 + radio, y1 + radio, 180, 270),
        (x2 - radio, y1 + radio, 270, 360),
        (x1 + radio, y2 - radio, 90, 180),
        (x2 - radio, y2 - radio, 0, 90),
    ):
        cv2.ellipse(img, (cx, cy), (radio, radio), 0, a1, a2, color, grosor, cv2.LINE_AA)


def _tarjeta(img: np.ndarray, pt1, pt2, relleno=_SUPERFICIE, radio: int = 14) -> None:
    """Una superficie elevada: relleno mas claro que el fondo, con borde sutil."""
    _rect_redondeado(img, pt1, pt2, relleno, radio, -1)
    _rect_redondeado(img, pt1, pt2, _BORDE, radio, 1)


def _icono_veredicto(img: np.ndarray, centro, radio: int, aprobado: Optional[bool], color) -> None:
    """Tilde, cruz o reloj segun el veredicto.

    La forma importa tanto como el color: a tres metros se reconoce antes que la palabra,
    y para alguien que no distingue bien rojo de verde es la unica senal que queda.
    """
    cx, cy = centro
    cv2.circle(img, (cx, cy), radio, color, -1, cv2.LINE_AA)
    cv2.circle(img, (cx, cy), radio, _TEXTO, 2, cv2.LINE_AA)
    grosor = max(3, radio // 6)
    if aprobado is True:
        cv2.line(img, (cx - radio // 2, cy), (cx - radio // 8, cy + radio // 2),
                 _TEXTO, grosor, cv2.LINE_AA)
        cv2.line(img, (cx - radio // 8, cy + radio // 2), (cx + radio // 2, cy - radio // 2),
                 _TEXTO, grosor, cv2.LINE_AA)
    elif aprobado is False:
        d = radio // 2
        cv2.line(img, (cx - d, cy - d), (cx + d, cy + d), _TEXTO, grosor, cv2.LINE_AA)
        cv2.line(img, (cx + d, cy - d), (cx - d, cy + d), _TEXTO, grosor, cv2.LINE_AA)
    else:
        cv2.line(img, (cx, cy - radio // 2), (cx, cy), _TEXTO, grosor, cv2.LINE_AA)
        cv2.line(img, (cx, cy), (cx + radio // 3, cy + radio // 3), _TEXTO, grosor, cv2.LINE_AA)


def _barra_coincidencia(
    img: np.ndarray, x1: int, y: int, x2: int, valor: float, umbral: float, color
) -> None:
    """Barra de coincidencia con una marca en el umbral.

    Dice algo que el porcentaje suelto no decia: si paso raspando o de sobra.
    """
    alto = 14
    _rect_redondeado(img, (x1, y), (x2, y + alto), _FONDO, alto // 2, -1)
    ancho = x2 - x1
    lleno = int(ancho * max(0.0, min(1.0, valor)))
    if lleno > alto:
        _rect_redondeado(img, (x1, y), (x1 + lleno, y + alto), color, alto // 2, -1)
    marca = x1 + int(ancho * max(0.0, min(1.0, umbral)))
    cv2.line(img, (marca, y - 5), (marca, y + alto + 5), _TEXTO, 2, cv2.LINE_AA)


def estado_resultado(
    estado: EstadoRevision, cfg: AppConfig
) -> tuple[str, str, tuple, Optional[bool]]:
    """(titulo, detalle, color, aprobado) del cartel de resultado.

    Aparte para poder probar la DECISION sin tener que leer pixeles de una imagen.
    """
    if not estado.congelado:
        return "ESPERANDO", "apriete REVISAR", _SUPERFICIE, None
    valor = coincidencia(estado.veredicto, cfg)
    if valor is None:
        return "SIN MODELO", "todavia no hay modelo entrenado", _AMBAR, None
    umbral = float(cfg.interfaz.umbral_aprobacion)
    if valor >= umbral:
        return "APROBADO", f"coincidencia {valor * 100:.0f}%", _VERDE_OK, True
    return "DESAPROBADO", f"coincidencia {valor * 100:.0f}%", _ROJO_NO, False


def _boton_lindo(
    img: np.ndarray, x: int, y1: int, y2: int, texto: str, color, tamano: int = 21
) -> Rect:
    """Boton redondeado que se adapta a su texto. NO dibuja el texto: eso va despues."""
    ancho_texto, _ = tipografia.medir(texto, tamano, negrita=True)
    x2 = x + ancho_texto + 2 * _RELLENO_BOTON
    _rect_redondeado(img, (x, y1), (x2, y2), color, 10, -1)
    return (x, y1, x2, y2)


def componer_simple(
    vivo: np.ndarray,
    referencia: Optional[np.ndarray],
    cfg: AppConfig,
    estado: EstadoRevision,
    alto_maximo: Optional[int] = None,
) -> Pantalla:
    """La pantalla del taller: la sierra, el modelo y el resultado en dos palabras.

    Se dibuja en dos tiempos: primero TODAS las formas sobre el lienzo de OpenCV, y recien
    despues todo el texto, en una sola pasada de Pillow. Dibujar texto de a uno obligaria a
    convertir la imagen entera de BGR a RGB y de vuelta una vez por linea: veinte
    conversiones de 1280x672 en cada frame.
    """
    margen = _MARGEN_SIMPLE
    alto_cabecera, alto_resultado, alto_botones = (
        _ALTO_CABECERA, _ALTO_RESULTADO, _ALTO_BOTONES,
    )
    ancho_panel = (ANCHO_LIENZO - 3 * margen) // 2
    alto_img, ancho_img = vivo.shape[:2]
    alto_panel = max(220, int(ancho_panel * alto_img / max(1, ancho_img)))

    if alto_maximo is not None:
        alto_panel = max(
            _ALTO_MINIMO_PANEL, min(alto_panel, int(alto_maximo) - _ALTO_FIJO_SIMPLE)
        )

    alto_total = _ALTO_FIJO_SIMPLE + alto_panel
    lienzo = np.full((alto_total, ANCHO_LIENZO, 3), _FONDO, np.uint8)

    x_izq, x_der = margen, 2 * margen + ancho_panel
    y_panel = alto_cabecera

    # ---------------- formas ----------------
    for x, imagen in ((x_izq, vivo), (x_der, referencia)):
        _tarjeta(lienzo, (x - 3, y_panel - 3), (x + ancho_panel + 3, y_panel + alto_panel + 3))
        if imagen is not None:
            lienzo[y_panel : y_panel + alto_panel, x : x + ancho_panel] = _encajar(
                imagen, ancho_panel, alto_panel
            )

    if estado.congelado:
        ancho_c, _ = tipografia.medir("CONGELADA", 17, negrita=True)
        _rect_redondeado(
            lienzo, (x_izq + 10, y_panel + 10),
            (x_izq + 10 + ancho_c + 24, y_panel + 44), _FONDO, 8, -1,
        )

    titulo, detalle, color, aprobado = estado_resultado(estado, cfg)
    y_res = y_panel + alto_panel + margen
    _tarjeta(lienzo, (margen, y_res), (ANCHO_LIENZO - margen, y_res + alto_resultado),
             _SUPERFICIE, 16)

    radio = alto_resultado // 2 - 22
    cx = margen + 30 + radio
    cy = y_res + alto_resultado // 2
    _icono_veredicto(lienzo, (cx, cy), radio, aprobado, color)

    x_texto = cx + radio + 34
    valor = coincidencia(estado.veredicto, cfg) if estado.congelado else None
    if valor is not None:
        _barra_coincidencia(lienzo, x_texto, cy + 36, ANCHO_LIENZO - margen - 30,
                            valor, float(cfg.interfaz.umbral_aprobacion), color)

    y1 = alto_total - alto_botones + 14
    y2 = y1 + 48
    botones: dict[str, Rect] = {}
    etiquetas: list[tuple[str, Rect, tuple]] = []
    if estado.congelado:
        for clave, texto_boton, color_boton in (
            ("seguir", "OTRA PIEZA", _SUPERFICIE),
            ("aprobada", "GUARDAR BUENA", _VERDE_OK),
            ("rechazada", "GUARDAR MALA", _ROJO_NO),
        ):
            x = margen if not botones else max(r[2] for r in botones.values()) + 12
            rect = _boton_lindo(lienzo, x, y1, y2, texto_boton, color_boton)
            botones[clave] = rect
            etiquetas.append((texto_boton, rect, color_boton))
    else:
        rect = _boton_lindo(lienzo, margen, y1, y2, "REVISAR", _AZUL, 24)
        botones["revisar"] = rect
        etiquetas.append(("REVISAR", rect, _AZUL))

    ancho_salir, _ = tipografia.medir("SALIR", 21, negrita=True)
    x_salir = ANCHO_LIENZO - margen - ancho_salir - 2 * _RELLENO_BOTON
    rect_salir = _boton_lindo(lienzo, x_salir, y1, y2, "SALIR", _SUPERFICIE)
    botones["salir"] = rect_salir
    etiquetas.append(("SALIR", rect_salir, _SUPERFICIE))

    # Cambiar de modelo solo tiene sentido si hay mas de uno cargado. Se ubica pegado a
    # SALIR, calculando hacia la izquierda, para que no choque con los de guardar.
    if estado.posicion_referencia[1] > 1:
        texto_modelo = "OTRO MODELO"
        ancho_mod, _ = tipografia.medir(texto_modelo, 21, negrita=True)
        x_mod = x_salir - 12 - ancho_mod - 2 * _RELLENO_BOTON
        ocupado = max((r[2] for r in botones.values() if r[0] < x_mod), default=0)
        if x_mod > ocupado + 12:
            rect_mod = _boton_lindo(lienzo, x_mod, y1, y2, texto_modelo, _SUPERFICIE)
            botones["modelo"] = rect_mod
            etiquetas.append((texto_modelo, rect_mod, _SUPERFICIE))

    rect_mensaje = None
    if estado.mensaje:
        ancho_m, alto_m = tipografia.medir(estado.mensaje, 22, negrita=True)
        x_m = (ANCHO_LIENZO - ancho_m) // 2
        y_m = y_panel + alto_panel - alto_m - 30
        _rect_redondeado(lienzo, (x_m - 18, y_m - 10), (x_m + ancho_m + 18, y_m + alto_m + 12),
                         _VERDE_OK, 12, -1)
        rect_mensaje = (x_m, y_m)

    # ---------------- texto, todo de una pasada ----------------
    with tipografia.Pincel(lienzo) as pincel:
        titulo_modelo = "MODELO APROBADO"
        if estado.nombre_referencia:
            titulo_modelo = f"MODELO: {estado.nombre_referencia.upper()}"
            cual, cuantos = estado.posicion_referencia
            if cuantos > 1:
                titulo_modelo += f"   ({cual} de {cuantos})"
        for x, titulo_col in ((x_izq, "SIERRA"), (x_der, titulo_modelo)):
            ancho_t, alto_t = tipografia.medir(titulo_col, 20, negrita=True)
            pincel.texto(titulo_col, (x + (ancho_panel - ancho_t) // 2,
                                      alto_cabecera - alto_t - 12), _TEXTO_TENUE, 20, True)

        if referencia is None:
            for i, linea in enumerate(("No hay fotos de modelo.", "",
                                       "Pone una pieza buena", "y apreta la tecla  f")):
                if linea:
                    pincel.texto(linea, (x_der + 26, y_panel + 52 + i * 34), _TEXTO_TENUE, 20)

        if estado.congelado:
            pincel.texto("CONGELADA", (x_izq + 22, y_panel + 19), _TEXTO, 17, True)

        ancho_v, alto_v = tipografia.medir(titulo, 54, negrita=True)
        tamano_v = 54
        while x_texto + ancho_v > ANCHO_LIENZO - margen - 40 and tamano_v > 22:
            tamano_v -= 2
            ancho_v, alto_v = tipografia.medir(titulo, tamano_v, negrita=True)
        pincel.texto(titulo, (x_texto, cy - alto_v - 10),
                     color if aprobado is not None else _TEXTO, tamano_v, True)
        pincel.texto(detalle, (x_texto + 2, cy + 6), _TEXTO_TENUE, 19)

        if valor is not None:
            texto_umbral = f"minimo {cfg.interfaz.umbral_aprobacion * 100:.0f}%"
            ancho_u, _ = tipografia.medir(texto_umbral, 16)
            pincel.texto(texto_umbral, (ANCHO_LIENZO - margen - 30 - ancho_u, cy + 60),
                         _TEXTO_TENUE, 16)

        for texto_boton, rect, color_boton in etiquetas:
            tamano = 24 if texto_boton == "REVISAR" else 21
            ancho_b, alto_b = tipografia.medir(texto_boton, tamano, negrita=True)
            pincel.texto(
                texto_boton,
                (rect[0] + ((rect[2] - rect[0]) - ancho_b) // 2,
                 rect[1] + ((rect[3] - rect[1]) - alto_b) // 2),
                _color_texto(color_boton), tamano, True,
            )

        if rect_mensaje is not None and estado.mensaje:
            pincel.texto(estado.mensaje, rect_mensaje, _TEXTO, 22, True)

    return Pantalla(imagen=lienzo, botones=botones)
