"""El panel de CONCEPTOS: elegir en que carpeta va una foto, y manejar esas carpetas.

POR QUE EXISTE
--------------
Hasta ahora el programa tenia dos destinos fijos --la tecla 'a' guardaba en "aprobada" y la
'r' en "rechazada"-- y una tercera carpeta aparte para las fotos de modelo. Cambiar esa
lista obligaba a editar configs/config.yaml a mano, que en la PC del taller no lo va a
hacer nadie. El resultado se vio en el registro: el operario sacaba fotos con 'f' --el
unico destino que no pedia decidir entre dos palabras-- y la coleccion quedaba vacia.

Aca se ve la lista entera, cuantas fotos tiene cada concepto, cuanto acierta la memoria en
cada uno, y se puede agregar uno nuevo, borrar, mirar las fotos y tirar las que no sirven.

SE DIBUJA, NO DECIDE
--------------------
Igual que revision.py: este modulo arma la imagen y devuelve donde quedaron los botones.
Las teclas y los clics los resuelve run_live. Asi se puede probar sin pantalla.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from . import tipografia
from .conceptos import Concepto
# Se importan los trazos de la pantalla principal --no se copian-- para que el panel y la
# pantalla del taller sean el mismo programa a la vista: mismos colores, mismas esquinas.
from .revision import (
    ANCHO_LIENZO,
    Pantalla,
    Rect,
    _AMBAR,
    _AZUL,
    _BORDE,
    _FONDO,
    _RELLENO_BOTON,
    _ROJO_NO,
    _SUPERFICIE,
    _TEXTO,
    _TEXTO_TENUE,
    _VERDE_OK,
    _boton_lindo,
    _encajar,
    _rect_redondeado,
    _tarjeta,
)
from .overlay import _color_texto

MODO_LISTA = "lista"
MODO_NUEVO = "nuevo"
MODO_FOTOS = "fotos"
MODO_BORRAR = "borrar"

_MARGEN = 18
_ALTO_FILA = 56
_ALTO_CABECERA = 92
_ALTO_BOTONES = 76
_POR_PAGINA = 8          # miniaturas por pagina en el modo fotos
_COLUMNAS = 4


@dataclass
class EstadoPanel:
    """Todo lo que el panel necesita para dibujarse."""

    conceptos: list[Concepto] = field(default_factory=list)
    elegido: int = 0
    modo: str = MODO_LISTA
    escribiendo: str = ""
    aviso: Optional[str] = None
    resumen: str = ""
    aciertos: dict[str, tuple[int, int]] = field(default_factory=dict)
    # Si el acierto se midio sobre un recorte de las fotos y no sobre todas. Sin esto, la
    # fila diria "acierta 9 de 10" cuando el concepto tiene 300 fotos y se midieron 10.
    recortada: bool = False
    # La foto que se va a archivar, si se llego aca con una foto en la mano.
    pendiente: Optional[np.ndarray] = None
    # Modo fotos: las rutas del concepto elegido y en que pagina se esta.
    fotos: list[Path] = field(default_factory=list)
    pagina: int = 0
    foto_elegida: int = 0

    @property
    def concepto_elegido(self) -> Optional[Concepto]:
        if not self.conceptos:
            return None
        indice = max(0, min(self.elegido, len(self.conceptos) - 1))
        return self.conceptos[indice]

    def mover(self, paso: int) -> None:
        if self.conceptos:
            self.elegido = (self.elegido + paso) % len(self.conceptos)


class Miniaturas:
    """Las fotos chiquitas del modo fotos, leidas una sola vez.

    Leer ocho JPG del disco en cada cuadro serian 40 ms de los 30 que dura el cuadro: la
    pantalla se arrastraria. Se leen cuando cambia la pagina y se quedan en memoria.
    """

    def __init__(self, lado: int = 180, tope: int = 48) -> None:
        self._lado = int(lado)
        self._tope = int(tope)
        self._cache: dict[str, Optional[np.ndarray]] = {}

    def obtener(self, ruta: Path) -> Optional[np.ndarray]:
        clave = str(ruta)
        if clave in self._cache:
            return self._cache[clave]
        imagen = cv2.imread(clave)
        chica = None
        if imagen is not None:
            alto, ancho = imagen.shape[:2]
            escala = self._lado / max(1, max(alto, ancho))
            chica = cv2.resize(
                imagen,
                (max(1, int(ancho * escala)), max(1, int(alto * escala))),
                interpolation=cv2.INTER_AREA,
            )
        if len(self._cache) >= self._tope:
            self._cache.clear()
        self._cache[clave] = chica
        return chica

    def olvidar(self, ruta: Optional[Path] = None) -> None:
        if ruta is None:
            self._cache.clear()
        else:
            self._cache.pop(str(ruta), None)


def _texto_aciertos(estado: EstadoPanel, concepto: Concepto) -> str:
    """Que tan bien reconoce la memoria este concepto, con sus propias fotos."""
    if concepto.especial:
        # La galeria de modelos no entra en la memoria: no dice como esta el filo, dice
        # contra que foto se compara. Mostrarle un acierto seria inventarle un sentido.
        return f"{concepto.fotos} fotos  -  es la foto con la que se compara, no un concepto"
    if concepto.fotos == 0:
        return "sin fotos todavia"
    par = estado.aciertos.get(concepto.nombre)
    if not par or par[1] == 0:
        return f"{concepto.fotos} fotos"
    bien, total = par
    # Se dice "de las N que midio" y no "de N": con muchas fotos la autoevaluacion se mide
    # sobre un recorte, y un "acierta 9 de 10" al lado de "300 fotos" se lee como si
    # hubiera medido las 300.
    if estado.recortada or total < concepto.fotos:
        return f"{concepto.fotos} fotos  -  acierta {bien} de las {total} que midio"
    return f"{concepto.fotos} fotos  -  la memoria acierta {bien} de {total}"


def componer(
    estado: EstadoPanel,
    alto: int,
    miniaturas: Optional[Miniaturas] = None,
) -> Pantalla:
    """Arma el panel al alto que se le pida, para que la ventana no cambie de tamano."""
    alto = max(420, int(alto))
    lienzo = np.full((alto, ANCHO_LIENZO, 3), _FONDO, np.uint8)
    botones: dict[str, Rect] = {}
    etiquetas: list[tuple[str, Rect, tuple]] = []

    if estado.modo == MODO_FOTOS:
        lineas = _dibujar_fotos(lienzo, estado, botones, etiquetas, miniaturas)
    elif estado.modo == MODO_BORRAR:
        lineas = _dibujar_borrar(lienzo, estado, botones, etiquetas)
    elif estado.modo == MODO_NUEVO:
        lineas = _dibujar_nuevo(lienzo, estado, botones, etiquetas)
    else:
        lineas = _dibujar_lista(lienzo, estado, botones, etiquetas)

    with tipografia.Pincel(lienzo) as pincel:
        for texto, posicion, color, tamano, negrita in lineas:
            pincel.texto(texto, posicion, color, tamano, negrita)
        for texto, rect, color in etiquetas:
            ancho_b, alto_b = tipografia.medir(texto, 20, negrita=True)
            pincel.texto(
                texto,
                (rect[0] + ((rect[2] - rect[0]) - ancho_b) // 2,
                 rect[1] + ((rect[3] - rect[1]) - alto_b) // 2),
                _color_texto(color), 20, True,
            )

    return Pantalla(imagen=lienzo, botones=botones)


def _cabecera(titulo: str, bajada: str, lineas: list) -> None:
    """El titulo y su bajada. Solo agrega texto: el panel se pinta de un solo color."""
    lineas.append((titulo, (_MARGEN + 4, 18), _TEXTO, 30, True))
    if bajada:
        lineas.append((bajada, (_MARGEN + 6, 58), _TEXTO_TENUE, 19, False))


def _fila_de_botones(
    lienzo, alto: int, definiciones: list[tuple[str, str, tuple]],
    botones: dict, etiquetas: list,
) -> None:
    """Dibuja la fila de abajo. El ultimo boton se pega a la derecha."""
    y1 = alto - _ALTO_BOTONES + 14
    y2 = y1 + 48
    x = _MARGEN
    for clave, texto, color in definiciones[:-1]:
        rect = _boton_lindo(lienzo, x, y1, y2, texto, color, 20)
        botones[clave] = rect
        etiquetas.append((texto, rect, color))
        x = rect[2] + 12

    clave, texto, color = definiciones[-1]
    ancho, _ = tipografia.medir(texto, 20, negrita=True)
    x_final = max(x, ANCHO_LIENZO - _MARGEN - ancho - 2 * _RELLENO_BOTON)
    rect = _boton_lindo(lienzo, x_final, y1, y2, texto, color, 20)
    botones[clave] = rect
    etiquetas.append((texto, rect, color))


def _dibujar_lista(lienzo, estado: EstadoPanel, botones: dict, etiquetas: list) -> list:
    alto = lienzo.shape[0]
    lineas: list = []
    _cabecera(
        "CONCEPTOS",
        estado.resumen or "cada foto que guardes se suma a la memoria al instante",
        lineas,
    )

    # A la derecha, la foto que se va a archivar: el operario tiene que ver QUE esta
    # mandando a donde, no solo el nombre de la carpeta.
    ancho_lista = ANCHO_LIENZO - 2 * _MARGEN
    if estado.pendiente is not None:
        ancho_vista = 340
        ancho_lista = ANCHO_LIENZO - 3 * _MARGEN - ancho_vista
        x_vista = ANCHO_LIENZO - _MARGEN - ancho_vista
        alto_vista = min(260, alto - _ALTO_CABECERA - _ALTO_BOTONES - _MARGEN)
        _tarjeta(lienzo, (x_vista - 3, _ALTO_CABECERA - 3),
                 (x_vista + ancho_vista + 3, _ALTO_CABECERA + alto_vista + 3))
        lienzo[_ALTO_CABECERA:_ALTO_CABECERA + alto_vista,
               x_vista:x_vista + ancho_vista] = _encajar(
            estado.pendiente, ancho_vista, alto_vista
        )
        lineas.append(("ESTA FOTO VA A IR A:", (x_vista + 6, _ALTO_CABECERA + alto_vista + 12),
                       _TEXTO_TENUE, 17, True))
        elegido = estado.concepto_elegido
        if elegido is not None:
            lineas.append((elegido.nombre.upper(),
                           (x_vista + 6, _ALTO_CABECERA + alto_vista + 36),
                           _VERDE_OK, 24, True))

    y = _ALTO_CABECERA
    tope = alto - _ALTO_BOTONES - _MARGEN - 30
    caben = max(1, (tope - _ALTO_CABECERA) // _ALTO_FILA)

    # LA LISTA SE CORRE PARA QUE EL ELEGIDO SIEMPRE SE VEA. Antes se dibujaban las filas que
    # entraban y el resto se anunciaba con un "...y 12 mas": los que no se veian no se
    # podian elegir de ninguna forma, ni con el mouse ni con las teclas, asi que un taller
    # con muchos conceptos se quedaba sin poder usar los ultimos.
    primera = 0
    if len(estado.conceptos) > caben and estado.elegido >= caben:
        primera = min(estado.elegido - caben + 1, len(estado.conceptos) - caben)
    hay_mas = len(estado.conceptos) > caben

    for posicion, concepto in enumerate(estado.conceptos[primera:primera + caben]):
        indice = primera + posicion
        elegido = indice == estado.elegido
        _tarjeta(
            lienzo, (_MARGEN, y), (_MARGEN + ancho_lista, y + _ALTO_FILA - 8),
            _SUPERFICIE if not elegido else _BORDE, 10,
        )
        if elegido:
            _rect_redondeado(lienzo, (_MARGEN, y), (_MARGEN + ancho_lista, y + _ALTO_FILA - 8),
                             _AZUL, 10, 2)
        # La marca de la izquierda dice de un vistazo que es cada fila: circulo lleno rojo
        # para los conceptos de rechazo, verde para los sanos, y un cuadrado hueco para la
        # galeria de modelos, que no es ni una cosa ni la otra.
        if concepto.especial:
            cv2.rectangle(lienzo, (_MARGEN + 18, y + 15), (_MARGEN + 34, y + 31),
                          _TEXTO_TENUE, 2, cv2.LINE_AA)
        else:
            color_punto = _ROJO_NO if concepto.defecto else _VERDE_OK
            cv2.circle(lienzo, (_MARGEN + 26, y + 23), 9, color_punto, -1, cv2.LINE_AA)
            # EL PUNTO SE PUEDE APRETAR y cambia de verde a rojo. Es lo que decide si la
            # pantalla pinta APROBADO o DESAPROBADO cuando la memoria elige ese concepto:
            # sin esto, todo concepto que creara el taller nacia "sano" y un diente
            # astillado salia en verde. Va ANTES que la fila en el diccionario, porque
            # boton_en se queda con el primero que encuentra.
            botones[f"defecto:{indice}"] = (_MARGEN + 10, y + 6, _MARGEN + 44, y + 40)

        # El numero solo se dibuja si tiene tecla: del decimo en adelante no hay digito que
        # lo alcance y un numero sin tecla es una promesa que no se cumple.
        if indice < 9:
            lineas.append((f"{indice + 1}", (_MARGEN + 44, y + 12), _TEXTO_TENUE, 17, False))
        lineas.append((concepto.nombre, (_MARGEN + 68, y + 8), _TEXTO, 24, True))
        lineas.append((_texto_aciertos(estado, concepto),
                       (_MARGEN + 68 + 300, y + 13), _TEXTO_TENUE, 18, False))
        botones[f"elegir:{indice}"] = (_MARGEN, y, _MARGEN + ancho_lista, y + _ALTO_FILA - 8)
        y += _ALTO_FILA

    if not estado.conceptos:
        lineas.append(("No hay ningun concepto todavia.", (_MARGEN + 16, y + 10),
                       _TEXTO_TENUE, 22, False))
        lineas.append(("Apreta NUEVO y escribi uno: 'aprobada', 'rechazada', lo que uses.",
                       (_MARGEN + 16, y + 44), _TEXTO_TENUE, 19, False))

    if estado.aviso:
        lineas.append((estado.aviso, (_MARGEN + 4, alto - _ALTO_BOTONES - 48),
                       _AMBAR, 20, True))

    # Todo lo que hay en los botones tiene tecla, y la tecla se dice. En el taller el
    # mouse esta siempre con aserrin y el operario termina usando el teclado.
    teclas = ("teclas:   1-9 elegir   ENTER guardar aca   n nuevo   d bueno/malo   "
              "v ver fotos   b borrar   ESC volver")
    if hay_mas:
        teclas += f"   |   {estado.elegido + 1} de {len(estado.conceptos)}, w y s para moverse"
    lineas.append((teclas, (_MARGEN + 4, alto - _ALTO_BOTONES - 22), _TEXTO_TENUE, 16, False))

    definiciones = []
    if estado.pendiente is not None and estado.conceptos:
        definiciones.append(("guardar_aqui", "GUARDAR ACA", _VERDE_OK))
    definiciones += [
        ("nuevo", "NUEVO CONCEPTO", _AZUL),
        ("fotos", "VER FOTOS", _SUPERFICIE),
        ("borrar", "BORRAR CONCEPTO", _SUPERFICIE),
        ("volver", "VOLVER", _SUPERFICIE),
    ]
    _fila_de_botones(lienzo, alto, definiciones, botones, etiquetas)
    return lineas


def _dibujar_nuevo(lienzo, estado: EstadoPanel, botones: dict, etiquetas: list) -> list:
    alto = lienzo.shape[0]
    lineas: list = []
    _cabecera("CONCEPTO NUEVO",
              "escribi el nombre y apreta ENTER. ESC para dejarlo.", lineas)

    y = _ALTO_CABECERA + 30
    _tarjeta(lienzo, (_MARGEN, y), (ANCHO_LIENZO - _MARGEN, y + 96), _SUPERFICIE, 14)
    texto = estado.escribiendo or ""
    lineas.append((texto + "_", (_MARGEN + 28, y + 26), _TEXTO, 44, True))

    lineas.append(("Va a quedar como una carpeta con ese nombre adentro de data/coleccion.",
                   (_MARGEN + 6, y + 120), _TEXTO_TENUE, 19, False))
    lineas.append(("Letras, numeros y guiones. Los espacios pasan a ser _.",
                   (_MARGEN + 6, y + 150), _TEXTO_TENUE, 19, False))
    if estado.aviso:
        lineas.append((estado.aviso, (_MARGEN + 6, y + 186), _AMBAR, 20, True))

    _fila_de_botones(
        lienzo, alto,
        [("confirmar_nuevo", "CREAR", _VERDE_OK), ("volver_lista", "CANCELAR", _SUPERFICIE)],
        botones, etiquetas,
    )
    return lineas


def _dibujar_borrar(lienzo, estado: EstadoPanel, botones: dict, etiquetas: list) -> list:
    alto = lienzo.shape[0]
    lineas: list = []
    concepto = estado.concepto_elegido
    nombre = concepto.nombre if concepto else "?"
    cuantas = concepto.fotos if concepto else 0
    _cabecera("BORRAR UN CONCEPTO", "", lineas)

    y = _ALTO_CABECERA + 20
    _tarjeta(lienzo, (_MARGEN, y), (ANCHO_LIENZO - _MARGEN, y + 180), _SUPERFICIE, 14)
    lineas.append((f"Sacar '{nombre}' de la lista", (_MARGEN + 28, y + 24), _TEXTO, 32, True))
    lineas.append((f"Tiene {cuantas} foto(s).", (_MARGEN + 28, y + 76), _TEXTO_TENUE, 22, False))
    lineas.append(("No se pierden: la carpeta se mueve a data/coleccion/_papelera y de ahi",
                   (_MARGEN + 28, y + 110), _TEXTO_TENUE, 19, False))
    lineas.append(("se puede volver a sacar con el Explorador de Windows.",
                   (_MARGEN + 28, y + 138), _TEXTO_TENUE, 19, False))
    # Esta pantalla es la unica donde una tecla hace algo que no se puede deshacer con otra
    # tecla, asi que las dos tienen que estar escritas.
    lineas.append(("teclas:   s = si, borralo      n o ESC = no, dejalo",
                   (_MARGEN + 4, alto - _ALTO_BOTONES - 22), _TEXTO_TENUE, 16, False))

    _fila_de_botones(
        lienzo, alto,
        [("confirmar_borrado", "SI, BORRAR", _ROJO_NO),
         ("volver_lista", "NO, DEJALO", _SUPERFICIE)],
        botones, etiquetas,
    )
    return lineas


def _dibujar_fotos(
    lienzo, estado: EstadoPanel, botones: dict, etiquetas: list,
    miniaturas: Optional[Miniaturas],
) -> list:
    alto = lienzo.shape[0]
    lineas: list = []
    concepto = estado.concepto_elegido
    nombre = concepto.nombre if concepto else "?"
    total = len(estado.fotos)
    paginas = max(1, (total + _POR_PAGINA - 1) // _POR_PAGINA)
    pagina = max(0, min(estado.pagina, paginas - 1))
    _cabecera(
        f"FOTOS DE {nombre.upper()}",
        f"{total} en total  -  pagina {pagina + 1} de {paginas}  -  "
        "elegi una y apreta BORRAR ESTA",
        lineas,
    )

    desde = pagina * _POR_PAGINA
    visibles = estado.fotos[desde:desde + _POR_PAGINA]
    filas = 2
    ancho_celda = (ANCHO_LIENZO - _MARGEN * (_COLUMNAS + 1)) // _COLUMNAS
    # Se reservan los dos renglones de abajo --el aviso y la linea de teclas-- igual que en
    # la lista. Sin reservarlos, el texto se dibujaba ENCIMA de la segunda fila de
    # miniaturas y no se podia leer ninguno de los dos.
    alto_disponible = alto - _ALTO_CABECERA - _ALTO_BOTONES - _MARGEN - 56
    alto_celda = max(90, (alto_disponible - _MARGEN * (filas - 1)) // filas)

    for posicion, ruta in enumerate(visibles):
        fila, columna = divmod(posicion, _COLUMNAS)
        x = _MARGEN + columna * (ancho_celda + _MARGEN)
        y = _ALTO_CABECERA + fila * (alto_celda + _MARGEN)
        elegida = (desde + posicion) == estado.foto_elegida
        _tarjeta(lienzo, (x - 3, y - 3), (x + ancho_celda + 3, y + alto_celda + 3),
                 _BORDE if elegida else _SUPERFICIE)
        chica = miniaturas.obtener(ruta) if miniaturas is not None else None
        alto_imagen = alto_celda - 26
        if chica is not None:
            lienzo[y:y + alto_imagen, x:x + ancho_celda] = _encajar(
                chica, ancho_celda, alto_imagen
            )
        if elegida:
            _rect_redondeado(lienzo, (x - 3, y - 3), (x + ancho_celda + 3, y + alto_celda + 3),
                             _AZUL, 14, 3)
        lineas.append((ruta.name[:26], (x + 4, y + alto_imagen + 4), _TEXTO_TENUE, 15, False))
        botones[f"foto:{desde + posicion}"] = (x, y, x + ancho_celda, y + alto_celda)

    if not visibles:
        lineas.append((f"'{nombre}' no tiene fotos todavia.",
                       (_MARGEN + 16, _ALTO_CABECERA + 20), _TEXTO_TENUE, 22, False))

    if estado.aviso:
        lineas.append((estado.aviso, (_MARGEN + 4, alto - _ALTO_BOTONES - 48),
                       _AMBAR, 20, True))
    lineas.append((
        "teclas:   1-8 elegir   b borrar la elegida   espacio ver mas   ESC volver",
        (_MARGEN + 4, alto - _ALTO_BOTONES - 22), _TEXTO_TENUE, 16, False,
    ))

    definiciones = []
    if visibles:
        definiciones.append(("borrar_foto", "BORRAR ESTA", _ROJO_NO))
    if paginas > 1:
        definiciones.append(("siguiente_pagina", "SIGUIENTES", _SUPERFICIE))
    definiciones.append(("volver_lista", "VOLVER", _SUPERFICIE))
    _fila_de_botones(lienzo, alto, definiciones, botones, etiquetas)
    return lineas
