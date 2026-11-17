"""El panel de pantalla completa: los CONCEPTOS, la ORDEN de afilado y el HISTORIAL.

POR QUE EXISTE
--------------
Hasta ahora el programa tenia dos destinos fijos --la tecla 'a' guardaba en "aprobada" y la
'r' en "rechazada"-- y una tercera carpeta aparte para las fotos de modelo. Cambiar esa
lista obligaba a editar configs/config.yaml a mano, que en la PC del taller no lo va a
hacer nadie. El resultado se vio en el registro: el operario sacaba fotos con 'f' --el
unico destino que no pedia decidir entre dos palabras-- y la coleccion quedaba vacia.

Aca se ve la lista entera, cuantas fotos tiene cada concepto, cuanto acierta la memoria en
cada uno, y se puede agregar uno nuevo, borrar, mirar las fotos y tirar las que no sirven.

EL HISTORIAL DE ORDENES
-----------------------
Despues entro la orden de afilado, y con ella el historial: todas las ordenes guardadas,
la mas nueva arriba, para retomar una, exportarla a un zip ante un reclamo o mandarla a la
papelera. Conceptos y ordenes son dos cosas hermanas --una es con que se compara, la otra
es para quien se trabaja-- asi que las dos se entran desde la pantalla del taller y
ninguna cuelga de la otra.

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
from .ordenes import Orden, Resumen, sanear_numero
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
MODO_ORDEN = "orden"
MODO_ORDENES = "ordenes"            # el historial: todas las ordenes guardadas
MODO_BORRAR_ORDEN = "borrar_orden"  # la confirmacion de mandar una a la papelera

# Los datos de la orden de afilado, en el orden en que se teclean. El tipo dice que teclas
# se aceptan: en "cuantas sierras" una letra no es un error de dedo, es un dato que despues
# rompe el informe.
CAMPOS_ORDEN: tuple[tuple[str, str], ...] = (
    ("Orden de afilado", "texto"),
    ("Nota de pedido", "texto"),
    ("Cliente", "texto"),
    ("Cuantas sierras", "numero"),
    # "el mismo para todas" no es un detalle de redaccion: es lo unico que hay. Orden.nueva
    # pone este numero en TODAS las sierras de la orden, asi que una orden con sierras de
    # 24 y de 36 no se puede cargar y hay que partirla en dos ordenes.
    # MAS LARGO NO ENTRA: la etiqueta se dibuja en x = _MARGEN + 22 y el valor en
    # x = _MARGEN + 330, o sea 308 px de lugar. Medido con tipografia.medir(texto, 20):
    # "Dientes por sierra" 153 px, con "(el mismo)" 249 px, y con "(el mismo para todas)"
    # 347 px, que se dibuja ENCIMA del numero que teclea el operario.
    ("Dientes por sierra (el mismo)", "numero"),
)

_MARGEN = 18
_ALTO_FILA = 56
_ALTO_CABECERA = 92
_ALTO_BOTONES = 76
_POR_PAGINA = 8          # miniaturas por pagina en el modo fotos
_COLUMNAS = 4
# TOPE de filas del historial por pagina: 8, porque 8 es el ultimo digito que tiene sentido
# ofrecer con las teclas 1-8. Cuantas se dibujan DE VERDAD lo dice filas_de_ordenes(), que
# mide el lienzo: este numero es solo el techo.
_POR_PAGINA_ORDENES = 8


def filas_de_ordenes(alto: int) -> int:
    """Cuantas filas del historial entran en un lienzo de este alto. LA UNICA CUENTA.

    Antes esta cuenta estaba adentro de _dibujar_ordenes y run_live paginaba con el tope
    _POR_PAGINA_ORDENES, que es 8 fijo. Las dos coincidian en esta PC con la camara en
    640x480 y dejaban de coincidir con cualquier otra: MEDIDO componiendo la pantalla del
    taller de verdad (modo simple, alto util de esta PC = 672 px) y contando las filas que
    _dibujar_ordenes dibuja,

        camara 640x480   -> lienzo 672 px -> 8 filas
        camara 1280x720  -> lienzo 656 px -> 7 filas
        camara 1920x1080 -> lienzo 656 px -> 7 filas

    y con una orden abierta y la barra de dientes en pantalla el lienzo sube a 672 px en las
    tres --el recuadro de la imagen se achica para dejarles lugar-- asi que las filas pasan
    a 8. O sea que el numero CAMBIA mientras el programa anda: no se puede escribir a mano
    en ningun lado. Con la camara en 1280x720 y 8 fijo, la tecla 8 elegia la fila 7, que no
    estaba dibujada, y la 'b' siguiente se llevaba una orden que el operario nunca vio.

    Los otros altos medidos, para saber donde estan los escalones: 664 px -> 8, 663 -> 7,
    656 -> 7, 600 -> 6, 520 -> 5.

    Se reservan los dos renglones de abajo --el aviso y la linea de teclas-- igual que en la
    lista de conceptos, y por eso se descuentan _ALTO_BOTONES, un margen y 30 px mas.
    """
    alto = max(420, int(alto))
    tope = alto - _ALTO_BOTONES - _MARGEN - 30
    caben = max(1, (tope - _ALTO_CABECERA) // _ALTO_FILA)
    return max(1, min(_POR_PAGINA_ORDENES, caben))


# Lugar de cada columna de una fila del historial, en px desde el borde del lienzo.
# Son fijos y no calculados para que las cuatro ordenes queden en columna y el ojo
# baje derecho. Medido con tipografia.medir sobre las 4 ordenes de verdad del taller:
# el numero mas ancho ('5113546' en 22 negrita) da 91 px de los 160 del cajon y la fecha
# mas ancha 167 px. La columna de la prueba va pegada a la derecha y su texto mas largo
# paso a ser '320 CON PRUEBA, 320 NO ESTAN' con 286 px, de los 914 que hay entre la
# columna del cliente y el borde: mira _texto_prueba().
_ANCHO_NUMERO_ORDEN = 160


@dataclass
class EstadoPanel:
    """Todo lo que el panel necesita para dibujarse."""

    conceptos: list[Concepto] = field(default_factory=list)
    elegido: int = 0
    modo: str = MODO_LISTA
    escribiendo: str = ""
    aviso: Optional[str] = None
    # ESTE AVISO NO SE LIMPIA CON LA TECLA SIGUIENTE. Lo pone run_live cuando el aviso dice
    # algo que todavia esta pasando y que no se puede volver a averiguar: hoy, el borrado que
    # quedo a medias, que es el unico lugar donde se nombra la carpeta de papelera con la
    # mitad de la orden adentro.
    # QUE SE ROMPIA: la 'b' del historial limpiaba pan.aviso, asi que el operario que
    # apretaba 'b' para reintentar el borrado perdia ese cartel antes de poder leerlo.
    # REPRODUCIDO con la orden 1234 y 3 pruebas, trabando una foto con msvcrt.locking --lo
    # que pasa si el operario la tiene abierta en el visor de Windows--: el aviso decia "la
    # orden 1234 quedo A MEDIAS: no se movio pruebas/1234. El resto se fue a
    # _papelera/20261116_164709" y despues de la 'b' pan.aviso era None.
    # Dura mientras el panel siga parado en el historial: si el operario sale y vuelve a
    # entrar, _abrir_historial lo limpia, y de ahi en mas lo que queda es el registro.
    aviso_pegado: bool = False
    resumen: str = ""
    aciertos: dict[str, tuple[int, int]] = field(default_factory=dict)
    # Si el acierto se midio sobre un recorte de las fotos y no sobre todas. Sin esto, la
    # fila diria "acierta 9 de 10" cuando el concepto tiene 300 fotos y se midieron 10.
    recortada: bool = False
    # Cuantas fotos necesita un concepto para poder ser un veredicto. 0 = no se sabe, y
    # entonces la fila no dice nada de esto.
    # ESTA PANTALLA ES DONDE SE ARREGLA: un concepto que no llego al minimo no aparece en
    # `aciertos` --la memoria no lo mide, porque no decide con el-- asi que su fila se
    # quedaba sin explicacion, mostrando "2 fotos" y nada mas. El operario tiene que poder
    # leer aca que le falta UNA foto, que es todo lo que hay que hacer.
    minimo_fotos: int = 0
    # La foto que se va a archivar, si se llego aca con una foto en la mano.
    pendiente: Optional[np.ndarray] = None
    # Modo fotos: las rutas del concepto elegido y en que pagina se esta.
    fotos: list[Path] = field(default_factory=list)
    pagina: int = 0
    foto_elegida: int = 0
    # Modo orden: la orden abierta (si hay), y los datos que se estan tecleando.
    orden: Optional[Orden] = None
    cargando_orden: bool = False
    valores: list[str] = field(default_factory=lambda: ["" for _ in CAMPOS_ORDEN])
    campo: int = 0
    # El formulario que ya se ofrecio retomar, para que el segundo ENTER confirme y
    # cualquier cambio en los datos vuelva a preguntar.
    retomar: str = ""
    # Modo ordenes: el historial leido del disco (una fila Resumen por orden), cual esta
    # elegida --el indice es absoluto, contra toda la lista, no contra la pagina-- y en que
    # pagina se esta. Se guarda la lista ya leida y no el objeto Ordenes porque el panel se
    # dibuja en cada cuadro y leer el disco 30 veces por segundo seria absurdo: medido,
    # armar el historial de las 4 ordenes de verdad cuesta 2,2 ms.
    historial: list = field(default_factory=list)   # list[Resumen]
    orden_elegida: int = 0
    pagina_ordenes: int = 0
    # LAS FILAS QUE EL HISTORIAL DIBUJO DE VERDAD. Lo escribe _dibujar_ordenes con lo que
    # midio del lienzo, y es lo que run_live usa para los digitos y para las paginas: el que
    # dibuja no decide nada, solo deja anotado cuanto le entro. Arranca en el tope porque
    # hasta que el panel se dibuje una vez no hay lienzo que medir, y run_live lo pone antes
    # con panel.filas_de_ordenes(alto) para que el primer cuadro tampoco mienta.
    filas_por_pagina: int = _POR_PAGINA_ORDENES

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

    def __init__(self, lado: int = 300, tope: int = 48) -> None:
        # 300 y no 180: la celda de la grilla es mas ancha que eso, asi que con 180 la
        # miniatura se AGRANDABA para llenarla y la foto se veia mas borrosa de lo que es.
        # Con el tope de 48 el cache ocupa unos 10 MB, que en esta PC no es problema.
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
    # LE FALTAN FOTOS PARA PODER DECIDIR, y se dice cuantas. Mientras no llegue, la memoria
    # no lo nombra en ningun veredicto: lo reconoce y avisa que falta. Decirlo aca es lo
    # que convierte un "no estoy seguro" en una tarea de un minuto.
    if estado.minimo_fotos and concepto.fotos < estado.minimo_fotos:
        faltan = estado.minimo_fotos - concepto.fotos
        cuantas = "1 foto mas" if faltan == 1 else f"{faltan} fotos mas"
        return (f"{concepto.fotos} fotos  -  le {'falta' if faltan == 1 else 'faltan'} "
                f"{cuantas} para que la memoria pueda decidir con el")
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

    if estado.modo == MODO_ORDEN:
        lineas = _dibujar_orden(lienzo, estado, botones, etiquetas)
    elif estado.modo == MODO_ORDENES:
        lineas = _dibujar_ordenes(lienzo, estado, botones, etiquetas)
    elif estado.modo == MODO_BORRAR_ORDEN:
        lineas = _dibujar_borrar_orden(lienzo, estado, botones, etiquetas)
    elif estado.modo == MODO_FOTOS:
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
            # Se achica hasta que entre: un nombre largo se cortaba contra el borde del
            # lienzo, y lo que quedaba a la vista podia ser otro concepto.
            nombre = elegido.nombre.upper()
            tamano = 24
            while tipografia.medir(nombre, tamano, True)[0] > ancho_vista - 12 and tamano > 13:
                tamano -= 2
            lineas.append((nombre, (x_vista + 6, _ALTO_CABECERA + alto_vista + 36),
                           _VERDE_OK, tamano, True))

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
        # La segunda columna arranca DESPUES del nombre, no en una posicion fija: un
        # nombre de 24 letras se montaba encima del texto de los aciertos.
        ancho_nombre, _ = tipografia.medir(concepto.nombre, 24, negrita=True)
        x_aciertos = max(_MARGEN + 68 + 300, _MARGEN + 68 + ancho_nombre + 20)
        lineas.append((_texto_aciertos(estado, concepto),
                       (x_aciertos, y + 13), _TEXTO_TENUE, 18, False))
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
    # LA ORDEN DE AFILADO YA NO SE ENTRA DESDE ACA. Estaba como un boton mas de esta fila
    # y con la tecla 'o', como si una orden fuera una cosa de los conceptos. No lo es: el
    # concepto dice contra que se compara el diente y la orden dice para quien se esta
    # trabajando. Ahora las dos se entran desde la pantalla del taller. Medido con
    # tipografia.medir: la fila de botones media 1249 px de los 1280 --31 px de hueco, no
    # entraba uno mas-- y sin "ORDEN DE AFILADO" queda en 1001 px.
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


def _dibujar_orden(lienzo, estado: EstadoPanel, botones: dict, etiquetas: list) -> list:
    """La orden de afilado: el formulario para cargarla, o el avance si ya esta abierta."""
    alto = lienzo.shape[0]
    lineas: list = []

    if estado.orden is None or estado.cargando_orden:
        _cabecera(
            "ORDEN DE AFILADO",
            "escribi cada dato y apreta ENTER para pasar al siguiente. ESC para dejarlo.",
            lineas,
        )
        y = _ALTO_CABECERA + 6
        for indice, (etiqueta, tipo) in enumerate(CAMPOS_ORDEN):
            activo = indice == estado.campo
            _tarjeta(lienzo, (_MARGEN, y), (ANCHO_LIENZO - _MARGEN, y + 54),
                     _BORDE if activo else _SUPERFICIE, 10)
            if activo:
                _rect_redondeado(lienzo, (_MARGEN, y), (ANCHO_LIENZO - _MARGEN, y + 54),
                                 _AZUL, 10, 2)
            valor = estado.valores[indice] if indice < len(estado.valores) else ""
            lineas.append((etiqueta, (_MARGEN + 22, y + 16), _TEXTO_TENUE, 20, False))
            lineas.append((valor + ("_" if activo else ""),
                           (_MARGEN + 330, y + 12), _TEXTO, 26, True))
            if tipo == "numero" and not valor:
                lineas.append(("(un numero)", (_MARGEN + 330, y + 18), _TEXTO_TENUE, 17, False))
            botones[f"campo:{indice}"] = (_MARGEN, y, ANCHO_LIENZO - _MARGEN, y + 54)
            y += 60

        if estado.aviso:
            lineas.append((estado.aviso, (_MARGEN + 4, y + 10), _AMBAR, 20, True))
        else:
            lineas.append((
                "Estos datos hoy se cargan a mano. Van tal cual al informe que se imprime "
                "si hay un reclamo.",
                (_MARGEN + 4, y + 10), _TEXTO_TENUE, 18, False,
            ))
        # EL BOTON DE RETOMAR APARECE SOLO CUANDO HAY UNA GUARDADA SIN CERRAR con este
        # numero, que es cuando estado.retomar queda puesto. Antes esa decision se contestaba
        # con un SEGUNDO ENTER, y dos ENTER seguidos en el ultimo campo retomaban la orden de
        # ayer sin que el operario llegara a leer el aviso: ver _cargar_orden() en run_live.
        # Un boton pintado que no se puede apretar es lo mismo que una tecla escrita que no
        # hace nada, asi que cuando no hay nada que retomar no esta.
        # MEDIDO con tipografia.medir en 20 negrita, que es como se dibujan: los tres
        # botones dan 739 px de los 1.280 del lienzo, contando los 12 px entre cada uno y los
        # dos margenes de 18.
        definiciones = [("crear_orden", "EMPEZAR LA ORDEN", _VERDE_OK)]
        if estado.retomar:
            definiciones.append(("retomar_la_guardada", "RETOMAR LA GUARDADA", _AMBAR))
        definiciones.append(("volver_orden", "CANCELAR", _SUPERFICIE))
        _fila_de_botones(lienzo, alto, definiciones, botones, etiquetas)
        return lineas

    orden = estado.orden
    sierra = orden.sierra()
    _cabecera("ORDEN DE AFILADO", orden.resumen(), lineas)

    y = _ALTO_CABECERA + 6
    _tarjeta(lienzo, (_MARGEN, y), (ANCHO_LIENZO - _MARGEN, y + 190), _SUPERFICIE, 14)
    datos = [
        f"Nota de pedido:  {orden.nota_pedido or '-'}",
        f"Cliente:  {orden.cliente or '-'}",
        f"Abierta:  {orden.abierta}",
        f"Sierras:  {len(orden.sierras)}   -   dientes en total: {orden.total_dientes}",
        f"Revisados:  {orden.total_revisados} de {orden.total_dientes}"
        f"   -   para rechazar: {orden.total_rechazados}",
    ]
    for indice, texto in enumerate(datos):
        lineas.append((texto, (_MARGEN + 24, y + 16 + indice * 34), _TEXTO, 20, False))

    y += 210
    if sierra is not None:
        lineas.append((
            f"SIERRA {sierra.numero}:  {sierra.cuantos_revisados} de {sierra.dientes} "
            f"dientes revisados   -   {sierra.rechazados} para rechazar",
            (_MARGEN + 4, y), _TEXTO_TENUE, 20, True,
        ))
        # Una barra por diente: de un vistazo se ve cuanto falta y donde estan los malos.
        y += 34
        ancho = ANCHO_LIENZO - 2 * _MARGEN
        paso = max(3, min(26, ancho // max(1, sierra.dientes)))
        for indice, diente in enumerate(sierra.revisados):
            x = _MARGEN + indice * paso
            if x + paso - 2 > ANCHO_LIENZO - _MARGEN:
                break
            if not diente.revisado:
                color = _BORDE
            elif diente.aprobado is True:
                color = _VERDE_OK
            elif diente.aprobado is False:
                color = _ROJO_NO
            else:
                color = _AMBAR
            cv2.rectangle(lienzo, (x, y), (x + paso - 3, y + 26), color, -1, cv2.LINE_AA)

    if estado.aviso:
        lineas.append((estado.aviso, (_MARGEN + 4, alto - _ALTO_BOTONES - 48),
                       _AMBAR, 20, True))
    lineas.append((
        "teclas:   s siguiente sierra   x saltear este diente   i informe   "
        "n orden nueva   ESC volver",
        (_MARGEN + 4, alto - _ALTO_BOTONES - 22), _TEXTO_TENUE, 16, False,
    ))

    _fila_de_botones(
        lienzo, alto,
        [("siguiente_sierra", "SIGUIENTE SIERRA", _SUPERFICIE),
         ("saltear_diente", "SALTEAR DIENTE", _SUPERFICIE),
         ("informe", "GUARDAR INFORME", _AZUL),
         ("nueva_orden", "ORDEN NUEVA", _SUPERFICIE),
         ("cerrar_orden", "CERRAR ORDEN", _ROJO_NO),
         ("volver_orden", "VOLVER", _SUPERFICIE)],
        botones, etiquetas,
    )
    return lineas


def _acortar(texto: str, ancho: int, tamano: int, negrita: bool = False) -> str:
    """Corta el texto hasta que entre en 'ancho' px, y avisa con '..' que esta cortado.

    Cortar a una cantidad fija de letras --como hace el modo fotos con el nombre del
    archivo-- no alcanza aca: "Aserradero del Litoral" y "WWWWWWWWWWWWWWWWWWWWWW" tienen las
    mismas 22 letras y miden 188 y 418 px en tamano 20. Y el aviso de que esta cortado no es
    un adorno: sin el, el operario lee un nombre de cliente que no es el que escribio.
    """
    if tipografia.medir(texto, tamano, negrita=negrita)[0] <= ancho:
        return texto
    recortado = texto
    while recortado and tipografia.medir(recortado + "..", tamano, negrita=negrita)[0] > ancho:
        recortado = recortado[:-1]
    return recortado + ".." if recortado else ""


def _orden_del_historial(estado: EstadoPanel) -> Optional[Resumen]:
    """La fila del historial que esta elegida, o None si no hay ninguna orden.

    El indice se acota igual que en concepto_elegido: el historial se vuelve a leer del
    disco cada vez que se entra, y una orden que se fue a la papelera deja el indice
    apuntando afuera de la lista.
    """
    if not estado.historial:
        return None
    indice = max(0, min(estado.orden_elegida, len(estado.historial) - 1))
    return estado.historial[indice]


def _ordenes_con_prueba(historial: list) -> int:
    """Cuantas filas del historial tienen al menos una prueba guardada.

    Se cuenta sobre lo que ya esta leido --el historial que el panel tiene en la mano-- asi
    que no cuesta ningun acceso al disco, y se puede llamar en cada cuadro: MEDIDO en esta
    PC, mediana 0,5 microsegundos con las 4 filas del taller y 4,7 con cien filas (max 11,7).
    Contra los 33 ms que dura un cuadro, no se mide.
    """
    return sum(1 for fila in historial if getattr(fila, "con_prueba", 0) > 0)


def hay_que_nombrar_el_archivo(fila: Resumen) -> bool:
    """Si el nombre del .json hace falta para saber de QUE fila se esta hablando.

    ES LA MISMA COMPARACION QUE HACE EL ENTER DEL HISTORIAL --fila.archivo contra el stem
    que le toca al numero, que es sanear_numero()-- y esta aca para que la pantalla, los
    carteles y esa decision no puedan decir tres cosas distintas. Ordenes.ruta() arma el
    nombre con sanear_numero(), asi que esto es preguntar "el archivo esta donde el numero
    lo manda?".

    POR QUE HACE FALTA. Medido con la orden 1234 del taller y su copia de Windows "1234 -
    copia.json": las dos pantallas de confirmacion de borrado salian PALABRA POR PALABRA
    iguales --comparadas renglon por renglon, a == b daba True-- las dos diciendo "Sacar la
    orden 1234 del historial", "Cliente: 1234" y "7 de 80 dientes revisados". Y el borrado
    va por el NOMBRE DEL ARCHIVO, que no aparecia en ninguna de las dos: el operario no
    tenia con que saber cual de las dos filas estaba por borrar. Los dos carteles del
    borrado tambien decian lo mismo, "ORDEN 1234 A LA PAPELERA".

    Cuando coinciden NO se nombra: para las 4 ordenes de verdad del taller archivo y
    sanear_numero(numero) son el mismo texto, y un renglon mas que repite el numero con
    otras palabras es un renglon que el operario deja de leer.
    """
    archivo = (fila.archivo or "").strip()
    return bool(archivo) and archivo != sanear_numero(fila.numero)


def nombre_de_la_fila(fila: Resumen) -> str:
    """Como se nombra esta orden donde el operario la va a leer: el numero, y ademas el
    archivo cuando el archivo no es el que al numero le toca.

    Lo usa el panel y lo usan los carteles de run_live, que es el punto: la pantalla de
    confirmacion y el cartel de "a la papelera" tienen que nombrar la MISMA orden.
    """
    if hay_que_nombrar_el_archivo(fila):
        return f"{fila.numero} del archivo {fila.archivo}.json"
    return fila.numero or (fila.archivo or "").strip() or "sin numero"


def carpeta_de_pruebas_de_la_fila(fila: Resumen) -> str:
    """La carpeta de pruebas que esta fila tiene adentro de data, para poder nombrarla.

    SE ARMA CON EL ARCHIVO, igual que Ordenes._deposito(), que es la carpeta que el borrado
    mueve de verdad: el borrado va por carpeta_pruebas(numero, archivo) y archivo manda. Si
    la pantalla nombrara pruebas/<numero> estaria nombrando una carpeta que el movimiento no
    toca, que es justo el enredo que se arreglo esta vuelta.
    """
    stem = (fila.archivo or "").strip() or sanear_numero(fila.numero)
    return f"pruebas/{stem or 'sin_numero'}"


def _texto_prueba(fila: Resumen) -> tuple[str, tuple]:
    """Que dice la columna de la prueba de una fila, y de que color.

    ACA SE CUMPLE LA REGLA: NO SE MUESTRA COMO PRUEBA ALGO QUE NO ES PRUEBA. Las ordenes
    viejas solo guardan Diente.foto --el nombre pelado de la foto que entro a la coleccion,
    sin carpeta-- y eso no es la prueba de nada: medido, las 14 fotos que nombraban las
    ordenes 123 y 1234 estaban todas en data/coleccion/_papelera y ninguna viva en la
    coleccion, asi que ese nombre no abria ningun archivo. (El 17/11/2026 esas 14 se
    rescataron y se copiaron al deposito de pruebas de cada orden, asi que esas dos ya
    tienen prueba de verdad; el nombre pelado sigue siendo lo unico que tiene cualquier
    otra orden vieja.) Por eso esta columna cuenta
    solo `con_prueba` --los dientes con la foto en el deposito-- y cuando no hay ninguna lo
    dice con todas las letras en vez de mostrar un cero al lado de la palabra prueba.

    Y LA PRUEBA PERDIDA SE DICE APARTE, EN ROJO. No es lo mismo no haber tenido prueba que
    haberla tenido y haberla perdido: en el primer caso no hay nada que buscar, en el
    segundo el .json nombra una foto que alguien tiene que ir a buscar --a la papelera de la
    coleccion, a la PC de la que se copio la carpeta, al respaldo-- antes de contestar el
    reclamo. Hasta ahora las dos filas decian lo MISMO, "SIN PRUEBA GUARDADA" en ambar, y el
    zip salia sin las fotos sin que nada avisara.

    Cuando hay de las dos cosas se dicen las dos: una fila que dijera solo "15 CON PRUEBA"
    con tres fotos faltando esta contando como prueba lo que no esta.

    REPRODUCIDO con una orden de 4 sierras de 80 dientes y las 320 pruebas puestas en el
    disco, usando una foto de verdad del microscopio: con las fotos la fila dice
    '320 CON PRUEBA' en verde y su zip pesa 26,9 MiB; borrando la carpeta de pruebas la
    misma fila decia 'SIN PRUEBA GUARDADA' en ambar --IGUAL que una orden que nunca tuvo
    prueba-- y el zip bajaba a 7,0 KiB sin que nada en la pantalla cambiara. Ahora dice
    '320 PRUEBAS QUE NO ESTAN' en rojo. Con la mitad de las fotos borradas:
    '160 CON PRUEBA, 160 NO ESTAN'.

    EN LAS 4 ORDENES DE VERDAD DEL TALLER, medido el 17/11/2026 despues de mudar al
    deposito las 14 fotos rescatadas: las cuatro dan prueba_perdida=0; la 123 y la 1234
    dicen '7 CON PRUEBA' en verde, y la 213132 y la 5113546 'SIN PRUEBA GUARDADA' en ambar.
    Hasta esa mudanza las cuatro decian 'SIN PRUEBA GUARDADA', porque ninguna tenia el
    campo prueba cargado y lo unico que habia eran nombres de foto sueltos.

    MEDIDO con tipografia.medir en 18 negrita, que es como se dibuja esta columna: el texto
    mas largo que puede salir es '320 CON PRUEBA, 320 NO ESTAN' con 286 px, '320 PRUEBAS
    QUE NO ESTAN' da 254 y 'SIN PRUEBA GUARDADA' daba 214. Entre la columna del cliente
    (x=334) y el borde derecho (x=1248) hay 914 px: entran los tres, y el cliente se acorta
    contra lo que queda, como siempre.
    """
    if fila.ilegible:
        return "NO SE PUDO LEER", _ROJO_NO
    if fila.con_prueba > 0 and fila.prueba_perdida > 0:
        # Ambar y no verde: la mitad de la prueba de esta orden no esta en el disco.
        return f"{fila.con_prueba} CON PRUEBA, {fila.prueba_perdida} NO ESTAN", _AMBAR
    if fila.con_prueba > 0:
        # EL VERDE ES "ESTA ORDEN TIENE SU PRUEBA", Y NO "TIENE ALGO DE PRUEBA". Si quedan
        # dientes con nada mas que el nombre pelado de una foto --que no es prueba-- la
        # fila va en ambar, igual que cuando hay pruebas perdidas: son dos formas de la
        # misma cosa, que es prueba que falta. REPRODUCIDO con la orden 1234 del taller, que
        # tiene 7 dientes con el nombre suelto, anotandole UNA prueba de verdad: la fila
        # pasaba a '1 CON PRUEBA' en VERDE y el renglon de abajo se callaba los 7. Una orden
        # con 1 prueba y 7 nombres sueltos no es una orden con prueba.
        # El texto no cambia --el numero sigue siendo el de las pruebas que estan-- lo que
        # cambia es el color, que es lo unico que el operario mira de lejos.
        return (f"{fila.con_prueba} CON PRUEBA",
                _AMBAR if fila.con_nombre_suelto > 0 else _VERDE_OK)
    if fila.prueba_perdida > 0:
        if fila.prueba_perdida == 1:
            return "1 PRUEBA QUE NO ESTA", _ROJO_NO
        return f"{fila.prueba_perdida} PRUEBAS QUE NO ESTAN", _ROJO_NO
    return "SIN PRUEBA GUARDADA", _AMBAR


def _segunda_linea(fila: Resumen) -> str:
    """El renglon chico de una fila: las fechas, lo revisado y los nombres sueltos.

    Van en un segundo renglon y no en mas columnas porque las columnas de arriba ya llegan
    al borde: medido, este renglon completo con las ordenes de verdad da 1.142 px de los
    1.186 utiles, y repartirlo en columnas obligaria a acortar el numero o el cliente, que
    son lo que el operario busca con la vista.

    MEDIDO con tipografia.medir en 17, que es como se dibuja, contra esos 1.186 px: con la
    fecha de la orden (las 4 del taller) da 777 px; con la fecha estimada y el campo vacio,
    934 px; con el archivo ilegible, 853 px. El unico que se pasa es el caso raro de un
    campo `abierta` con texto que no es una fecha: 1.126 px, y si ademas la orden tiene
    cliente largo y nombres sueltos se acorta y se le pone el '..'. Se acorta desde el
    final, asi que lo que se cae es el nombre del archivo y lo de la fecha queda, que es lo
    que no se puede perder.
    """
    if fila.ilegible:
        # Y SE DICE QUE LA FECHA ES LA DEL ARCHIVO: de un .json que no se pudo leer no sale
        # ninguna fecha de orden, asi que la fila se ubica en la lista por el mtime, que se
        # corre solo cada vez que alguien copia la carpeta.
        return ("el archivo esta pero no se pudo leer: hay que abrirlo a mano con el "
                "Explorador   -   va por la fecha del ARCHIVO")
    # LA FECHA ESTIMADA SE VE COMO ESTIMADA. Cuando la orden no dejo una fecha que se
    # entienda, la fila se ordena por el mtime del .json, que NO es la fecha de la orden: se
    # corre sola cada vez que alguien copia la carpeta Afilado a otra PC. Medido con la 123
    # del taller a la que se le vacio el campo abierta: su clave paso a ser la del archivo y
    # la fila se fue ARRIBA del historial, haciendose pasar por la orden mas nueva. Si eso
    # no se dice, el operario busca "la de ayer" arriba y lo que encuentra es la que alguien
    # copio ayer.
    if fila.clave_estimada:
        if fila.abierta.strip():
            # Hay texto pero no se entiende como fecha: se muestra tal cual, porque es lo
            # que el .json dice, y se aclara que no es lo que ordena la lista.
            primera = (f"abierta {fila.abierta} (no es una fecha; va por la fecha del "
                       "ARCHIVO)")
        else:
            primera = "sin fecha anotada (va por la fecha del ARCHIVO)"
        partes = [primera]
    else:
        partes = [f"abierta {fila.abierta or 'sin fecha anotada'}"]
    if fila.esta_abierta:
        partes.append("SIGUE ABIERTA")
    elif fila.cerrada:
        partes.append(f"cerrada {fila.cerrada}")
    else:
        # Ni abierta ni con fecha de cierre. Pasa si se abrio otra orden sin cerrar esta:
        # no se puede saber cuando se dejo, y es mejor decirlo que poner una fecha.
        partes.append("no quedo anotado cuando se cerro")
    partes.append(f"{fila.revisados} de {fila.total_dientes} revisados")
    partes.append(f"{fila.rechazados} para rechazar")
    # LOS NOMBRES SUELTOS SE NOMBRAN SIEMPRE, tenga o no tenga pruebas de verdad. Antes se
    # decian solo con con_prueba == 0, y entonces la PRIMERA prueba que conseguia una orden
    # hacia desaparecer el aviso de todas las demas: medido con la orden 1234 del taller
    # --7 dientes con el nombre pelado de una foto-- anotandole UNA prueba, la fila pasaba a
    # '1 CON PRUEBA' en verde y el renglon de abajo dejaba de mencionar los 7. Esos 7 no son
    # prueba: medido, las 14 fotos que nombran las ordenes 123 y 1234 estan todas en
    # data/coleccion/_papelera y ninguna viva en la coleccion.
    # ENTRA: medido con tipografia.medir en 17, que es como se dibuja, el renglon de esa
    # orden con la prueba y los 7 sueltos da 1.142 px de los 1.186 utiles, y el caso de 15
    # pruebas con 7 sueltos 1.160 px. El unico que se pasa sigue siendo el campo `abierta`
    # con texto que no es fecha, y de eso se encarga _acortar.
    if fila.archivo and fila.archivo != fila.numero:
        # El nombre del archivo solo aparece cuando NO es el numero que tecleo el operario
        # (una orden cargada como "OA/123" se guarda en OA_123.json): es el unico caso en
        # que hace falta para encontrarla con el Explorador.
        #
        # Y VA ANTES QUE LOS NOMBRES SUELTOS, al reves que hasta ahora. Lo ultimo que se
        # arma es lo primero que _acortar se lleva, y desde que los nombres sueltos se dicen
        # SIEMPRE --no solo cuando no hay prueba-- el renglon se pasa de ancho justo en el
        # caso en que las dos cosas estan: MEDIDO con la 1234 del taller renombrada a
        # "1234 - copia", el renglon entero da 1.368 px de los 1.186 utiles, asi que una de
        # las dos se corta --queda "... archivo 1234 - copia.json - 7 nombres de fo.."--. Se corta la de los nombres sueltos, que ademas sale entera en la
        # pantalla de borrado; el nombre del archivo es lo UNICO que distingue una fila de
        # su tocaya, y sobre el archivo caen la 'b' y la 'e'.
        partes.append(f"archivo {fila.archivo}.json")
    if fila.con_nombre_suelto > 0:
        partes.append(
            f"{fila.con_nombre_suelto} nombres de foto sueltos, no son prueba"
        )
    return "   -   ".join(partes)


def _dibujar_ordenes(lienzo, estado: EstadoPanel, botones: dict, etiquetas: list) -> list:
    """EL HISTORIAL: todas las ordenes guardadas, la mas nueva arriba.

    POR QUE ESTA PANTALLA
    ---------------------
    Hasta ahora una orden cerrada desaparecia del programa: quedaba el .json en el disco y
    nada mas. Si el cliente llamaba por una sierra de la semana pasada, el operario tenia
    que abrir la carpeta con el Explorador. Desde aca se la encuentra, se la retoma, se la
    exporta a un zip para mandar por mail o se la manda a la papelera.

    PAGINADA Y NO CON SCROLL, al contrario que la lista de conceptos: las filas del
    historial llevan dos renglones y no entra media fila, asi que correr la lista de a una
    dejaria la ultima cortada. El paginado es el mismo del modo fotos.
    """
    alto = lienzo.shape[0]
    lineas: list = []
    total = len(estado.historial)

    # UNA SOLA CUENTA DE FILAS, y queda anotada en el estado para que run_live use ESTE
    # numero --el que se dibujo-- y no un 8 fijo: mira filas_de_ordenes().
    por_pagina = filas_de_ordenes(alto)
    estado.filas_por_pagina = por_pagina
    paginas = max(1, (total + por_pagina - 1) // por_pagina)
    pagina = max(0, min(estado.pagina_ordenes, paginas - 1))
    elegida = _orden_del_historial(estado)

    if total == 0:
        # Sin bajada: lo que hay que decir --que no hay ninguna y que se carga con ORDEN
        # NUEVA-- va en el medio de la pantalla, donde el operario esta mirando.
        bajada = ""
    else:
        bajada = (f"{total} orden(es)  -  pagina {pagina + 1} de {paginas}  -  "
                  "la mas nueva arriba")
        if elegida is not None:
            # El numero de la elegida se escribe arriba porque EXPORTAR y BORRAR trabajan
            # sobre ella, y con el paginado puede estar en otra pagina que la que se ve.
            bajada += f"  -  elegida: {elegida.numero}"
    _cabecera("ORDENES DE AFILADO", bajada, lineas)

    ancho_lista = ANCHO_LIENZO - 2 * _MARGEN
    x_numero = _MARGEN + 68
    x_abierta = x_numero + _ANCHO_NUMERO_ORDEN + 10
    x_cliente = x_abierta + 78            # 78 = los 70 px de "ABIERTA" en 17 negrita y aire
    x_derecha = _MARGEN + ancho_lista - 14
    desde = pagina * por_pagina
    y = _ALTO_CABECERA

    for posicion, fila in enumerate(estado.historial[desde:desde + por_pagina]):
        indice = desde + posicion
        esta = indice == estado.orden_elegida
        _tarjeta(
            lienzo, (_MARGEN, y), (_MARGEN + ancho_lista, y + _ALTO_FILA - 8),
            _SUPERFICIE if not esta else _BORDE, 10,
        )
        if esta:
            _rect_redondeado(lienzo, (_MARGEN, y), (_MARGEN + ancho_lista, y + _ALTO_FILA - 8),
                             _AZUL, 10, 2)
        # La marca de la izquierda, con el mismo idioma que la lista de conceptos: circulo
        # lleno verde para la orden que esta abierta, hueco para las cerradas y lleno rojo
        # para la que no se pudo leer. De un vistazo se ve en que anda el taller.
        if fila.ilegible:
            cv2.circle(lienzo, (_MARGEN + 26, y + 23), 9, _ROJO_NO, -1, cv2.LINE_AA)
        elif fila.esta_abierta:
            cv2.circle(lienzo, (_MARGEN + 26, y + 23), 9, _VERDE_OK, -1, cv2.LINE_AA)
        else:
            cv2.circle(lienzo, (_MARGEN + 26, y + 23), 9, _TEXTO_TENUE, 2, cv2.LINE_AA)

        # EL NUMERO DE LA TECLA ES EL DE LA PAGINA, no el de la lista entera: las teclas
        # 1-N eligen una fila de lo que se esta viendo. Si dijera el numero de la lista, en
        # la pagina 2 la primera fila diria 9 y no habria tecla que la alcance.
        lineas.append((f"{posicion + 1}", (_MARGEN + 44, y + 6), _TEXTO_TENUE, 17, False))

        numero = fila.numero or fila.archivo or "sin numero"
        tamano = 22
        while (tipografia.medir(numero, tamano, True)[0] > _ANCHO_NUMERO_ORDEN
               and tamano > 14):
            tamano -= 2
        lineas.append((_acortar(numero, _ANCHO_NUMERO_ORDEN, tamano, True),
                       (x_numero, y + 4), _TEXTO, tamano, True))

        if fila.esta_abierta:
            lineas.append(("ABIERTA", (x_abierta, y + 8), _VERDE_OK, 17, True))

        # La columna de la prueba va PEGADA A LA DERECHA y en negrita: es el dato por el que
        # se abre esta pantalla cuando hay un reclamo, y asi cae siempre en el mismo lugar.
        texto_prueba, color_prueba = _texto_prueba(fila)
        ancho_prueba, _ = tipografia.medir(texto_prueba, 18, negrita=True)
        x_prueba = x_derecha - ancho_prueba
        lineas.append((texto_prueba, (x_prueba, y + 6), color_prueba, 18, True))

        if fila.ilegible:
            cliente, color_cliente = "", _TEXTO_TENUE
        elif fila.cliente.strip():
            cliente, color_cliente = fila.cliente.strip(), _TEXTO
        else:
            cliente, color_cliente = "sin cliente anotado", _TEXTO_TENUE
        if cliente:
            lineas.append((_acortar(cliente, max(40, x_prueba - 20 - x_cliente), 20),
                           (x_cliente, y + 7), color_cliente, 20, False))

        # El renglon chico se acorta contra el ancho que queda. Lo ultimo que se arma es lo
        # primero que se cae --el nombre del archivo-- y el aviso de los nombres sueltos va
        # antes, porque es el que no se puede perder.
        lineas.append((_acortar(_segunda_linea(fila), x_derecha - (_MARGEN + 44), 17),
                       (_MARGEN + 44, y + 26), _TEXTO_TENUE, 17, False))

        botones[f"elegir_orden:{indice}"] = (
            _MARGEN, y, _MARGEN + ancho_lista, y + _ALTO_FILA - 8
        )
        y += _ALTO_FILA

    if total == 0:
        lineas.append(("Todavia no hay ninguna orden guardada.", (_MARGEN + 16, y + 10),
                       _TEXTO_TENUE, 22, False))
        lineas.append((
            "Apreta ORDEN NUEVA y carga la primera: numero, cliente y cuantas sierras.",
            (_MARGEN + 16, y + 44), _TEXTO_TENUE, 19, False,
        ))

    # EL AVISO SE ACORTA CONTRA EL ANCHO QUE HAY. Antes se dibujaba tal cual y el que no
    # entraba se cortaba contra el borde del lienzo, sin que nada lo dijera: medido en 20
    # negrita, el renglon tiene 1.258 px desde x=22, y un aviso que nombra un archivo largo
    # --el nombre del .json entra en estos avisos-- se pasa sin aviso. Con _acortar queda el
    # '..' que dice que hay mas, que es lo que distingue un texto cortado de un texto que
    # terminaba ahi. Los avisos que se escriben a proposito ya entran: el mas largo de los
    # que hay mide 1.133 px.
    ancho_aviso = ANCHO_LIENZO - (_MARGEN + 4) - 18
    if estado.aviso:
        lineas.append((_acortar(estado.aviso, ancho_aviso, 20, True),
                       (_MARGEN + 4, alto - _ALTO_BOTONES - 48), _AMBAR, 20, True))
    elif elegida is not None and elegida.ilegible:
        # Si no se pudo leer el .json no hay nada que retomar ni que exportar, y los botones
        # tampoco estan. Decir por que es mejor que dejar la fila de botones mas corta sin
        # explicacion.
        lineas.append(("Esa orden no se puede leer: solo se puede mandar a la papelera y "
                       "mirarla a mano.", (_MARGEN + 4, alto - _ALTO_BOTONES - 48),
                       _AMBAR, 20, True))
    elif _ordenes_con_prueba(estado.historial):
        # LOS .exe DE RESPALDO YA NO ABREN ESTAS ORDENES, Y SE DICE UNA VEZ. Al lado de los
        # datos, en Desktop/Afilado, hay cuatro .exe: AnalisisAfilado_anterior_20261112_1648,
        # _20261112_1944, _20261113_1047 y el AnalisisAfilado.exe de hoy, todos escritos
        # entre el 12/11 y el 13/11 a las 17:03. Son el plan B del taller si este programa
        # falla, y hasta ahora la pantalla no decia en ningun lado que dejan de abrir la
        # orden.
        #
        # MEDIDO, y por eso se puede afirmar: el codigo con que se armaron (commit 1cc3c2b
        # y los anteriores) arma cada diente con Diente(**d) y ese Diente NO tiene campo
        # prueba, asi que UNA clave de mas adentro de un diente le hace ilegible la ORDEN
        # ENTERA --no el campo: la orden--. Se le dio a ese desde_datos() las 4 ordenes de
        # verdad del taller: las 4 se abren hoy (7, 7, 0 y 15 dientes revisados), y con UN
        # solo diente con prueba las 4 dan TypeError "unexpected keyword argument 'prueba'".
        # CUANTAS QUEDAN AFECTADAS HOY: 2 de 4. El 17/11/2026 se mudaron al deposito de
        # pruebas las 14 fotos rescatadas de las ordenes 123 y 1234 --7 cada una-- asi que
        # esas dos ya tienen dientes con el campo prueba y este renglon se dibuja. Las
        # otras dos, la 213132 y la 5113546, no tienen ninguna. Antes aca decia 0 de 4, que
        # era cierto hasta esa mudanza.
        #
        # VA EN EL RENGLON DEL AVISO, de ultimo: primero lo que acaba de pasar --un borrado
        # a medias, por ejemplo-- y despues lo de la fila elegida. Este dato no es un evento,
        # es como son las cosas, asi que puede esperar a que no haya nada mas urgente. Y va
        # UNA VEZ y no en cada fila: lo que se repite en ocho renglones se deja de leer.
        # Y NO TIENE RENGLON PROPIO porque no hay: con la pagina llena las filas llegan
        # hasta donde arranca el aviso. MEDIDO al alto del taller (672 px, 8 filas): las
        # filas terminan en el pixel 540 y el renglon del aviso se dibuja en el 548, o sea
        # que entre los dos no entra nada.
        # No se dice "N de las M": el total ya esta arriba, en la bajada. Y el singular va
        # aparte porque "1 de estas 1 ordenes" no es castellano de ninguna parte.
        # MEDIDO con tipografia.medir en 20 negrita, que es como se dibuja: 861 px de los
        # 1.240 que entran en este renglon, con cuatro ordenes y con veinticinco.
        cuantas = _ordenes_con_prueba(estado.historial)
        if cuantas == 1:
            aviso_exe = ("1 orden ya tiene prueba guardada: a esa los .exe de respaldo del "
                         "taller ya no la abren.")
        else:
            aviso_exe = (f"{cuantas} ordenes ya tienen prueba guardada: a esas los .exe de "
                         "respaldo del taller ya no las abren.")
        lineas.append((_acortar(aviso_exe, ancho_aviso, 20, True),
                       (_MARGEN + 4, alto - _ALTO_BOTONES - 48), _AMBAR, 20, True))

    # Solo se nombran las teclas que hacen algo. Con el historial vacio no hay nada que
    # elegir, retomar, exportar ni borrar, y una tecla escrita que no hace nada es lo mismo
    # que un boton pintado que no se puede apretar.
    if total == 0:
        teclas = "teclas:   n orden nueva   ESC volver"
    else:
        # EL CARTEL DICE LAS FILAS QUE HAY, no un 1-8 fijo. Ofrecer una tecla que elige una
        # fila que no esta en pantalla es la regla del taller al reves: lo que se aprieta
        # tiene que ser lo que se ve. Con una sola fila dibujada no se dice "1-1" sino "1".
        # Y no se nombran mas digitos que ordenes: con 3 ordenes en una pagina de 7, el 5 no
        # elige nada.
        hasta = min(por_pagina, total - desde)
        cuales = "1" if hasta <= 1 else f"1-{hasta}"
        teclas = (f"teclas:   {cuales} elegir   ENTER retomar   e exportar   b borrar   "
                  "n orden nueva   espacio ver mas   ESC volver")
    if paginas > 1:
        teclas += f"   |   {min(estado.orden_elegida + 1, total)} de {total}, w y s para moverse"
    lineas.append((teclas, (_MARGEN + 4, alto - _ALTO_BOTONES - 22), _TEXTO_TENUE, 16, False))

    # Medido con tipografia.medir: 271 + 239 + 126 + 188 + 120 px de botones, mas 12 px entre
    # cada uno y los dos margenes de 18, dan 1.028 px de los 1.280 del lienzo.
    definiciones = []
    if elegida is not None and not elegida.ilegible:
        definiciones.append(("abrir_orden", "RETOMAR ESTA ORDEN", _VERDE_OK))
        definiciones.append(("exportar_orden", "EXPORTAR A UN ZIP", _SUPERFICIE))
    if elegida is not None:
        definiciones.append(("borrar_orden", "BORRAR", _SUPERFICIE))
    definiciones.append(("nueva_orden_historial", "ORDEN NUEVA", _AZUL))
    definiciones.append(("volver_ordenes", "VOLVER", _SUPERFICIE))
    _fila_de_botones(lienzo, alto, definiciones, botones, etiquetas)
    return lineas


def _renglones_de_la_prueba(fila: Resumen) -> list[tuple[str, tuple, bool]]:
    """QUE PRUEBA SE VA Y QUE PRUEBA NO, con la cuenta que usa el movimiento.

    ES LA MISMA CUENTA QUE EL BORRADO, y eso es todo el arreglo. El movimiento va por
    Ordenes.carpeta_pruebas(numero, archivo) --o sea data/pruebas/<archivo>-- y con la regla
    nueva `con_prueba` cuenta exactamente los dientes cuya foto esta ADENTRO de esa carpeta.
    Antes eran dos cuentas distintas y no coincidian.

    REPRODUCIDO, con el manejador de teclas de verdad: la orden 1234 del taller con 3
    pruebas de verdad, renombrada a mano a "1234 viejo.json" con el Explorador. La 'b'
    dibujaba "Se lleva las 3 fotos de prueba que tiene guardadas." y la 's' devolvia
    completo=True con movidos=['1234 viejo.json'] mas sus dos informes, dejando las 3 fotos
    donde estaban. Hoy esa fila da con_prueba=0 y prueba_perdida=3 --las fotos estan en
    pruebas/1234 y el archivo se llama "1234 viejo"-- asi que esta pantalla no promete
    ninguna foto, que es lo que de verdad se lleva.

    Y SE NOMBRA LA CARPETA. Es el dato con el que el operario puede ir a mirar con el
    Explorador antes de apretar, y es lo que hace la promesa comprobable en vez de una
    palabra: "se lleva las 3 fotos ... en pruebas/1234 viejo" se puede verificar; "se lleva
    las fotos" no. Y es importante que se nombre la CARPETA porque lo que se mueve es la
    carpeta entera: si adentro hay fotos huerfanas --las que ningun diente nombra, de un
    guardado que fallo-- esas tambien viajan a la papelera y aca no estan contadas.
    MEDIDO comparando lo que la pantalla promete contra lo que llega a la papelera, en
    cuatro casos: 3 pruebas con el archivo bien, promete 3 y se mueven 3; renombrado a mano,
    promete 0 y se mueven 0; sin pruebas, promete 0 y se mueven 0; y 3 pruebas con 2
    huerfanas al lado, promete 3 y se mueven 5. O sea que la cuenta nunca promete de mas,
    que es el lado que importa: lo que se promete se mueve siempre.

    LO QUE YA NO SE AFIRMA: antes decia "esas ya no estan" y "ya no estan en el disco" de
    las pruebas que el archivo nombra y no aparecieron, y eso es mas de lo que se midio. Lo
    medido es "no estan donde el archivo las nombra": en la reproduccion de arriba las 3
    fotos estaban perfectamente en el disco, en data/pruebas/1234, y medido en el taller las
    14 fotos que nombran las ordenes 123 y 1234 estan TODAS en data/coleccion/_papelera.
    Existen; no estan donde el archivo dice. Un "ya no estan" a secas es dejar de buscar.
    Ahora el "ya no estan" queda ATADO A LA CARPETA que se nombra en el mismo renglon, y el
    renglon siguiente dice con todas las letras que eso no quiere decir que no existan y
    donde mirar. El que arme el reclamo tiene que ir a buscarlas, no darlas por perdidas.
    (La frase "ya no estan" sigue escrita porque dos pruebas del repo la fijan ---
    test_la_pantalla_de_borrado_declara_la_prueba_que_ya_no_esta y
    test_la_pantalla_de_borrado_no_calla_la_prueba_perdida_cuando_hay_de_las_dos--- y esta
    vuelta no se tocan las pruebas. Lo que cambio es de que se afirma: de "no estan" a "no
    estan ahi".)

    MEDIDO con tipografia.medir en 20, que es como se dibujan, contra los 1.216 px que hay
    desde x=46, pasandole a esta funcion los ocho casos con los numeros mas grandes que
    salen del taller (320 dientes) y con los nombres de archivo mas largos: entran todos, y
    el mas ancho es el de los nombres sueltos con 902 px. Los otros: 888 px el de la prueba
    que no esta ahi, 855 el del archivo ilegible, 800 el que dice donde buscarla, 799 el de
    "no se lleva ninguna foto", 710 el de "se lleva las N" y 380 el de la orden sin prueba.
    """
    carpeta = carpeta_de_pruebas_de_la_fila(fila)
    if fila.ilegible:
        # DE UN ARCHIVO QUE NO SE PUDO LEER NO SE SABE QUE PRUEBA NOMBRA, y decir que no
        # tiene ninguna seria afirmar lo que nadie pudo contar. Lo que SI es seguro es que
        # el borrado mueve la carpeta si esta: no mira el .json para eso.
        return [(f"No se pudo contar su prueba: si hay una carpeta {carpeta}, se va con la "
                 "orden.", _AMBAR, True)]
    if fila.con_prueba > 0 and fila.prueba_perdida > 0:
        # LAS DOS COSAS JUNTAS, porque son dos: unas se van y las otras no estan ahi.
        return [
            (f"Se lleva las {fila.con_prueba} fotos de prueba que tiene guardadas en "
             f"{carpeta}.", _AMBAR, True),
            (f"Otras {fila.prueba_perdida} que el archivo nombra ya no estan ahi: pueden "
             "estar en la papelera de la coleccion.", _AMBAR, True),
        ]
    if fila.con_prueba > 0:
        return [(f"Se lleva las {fila.con_prueba} fotos de prueba que tiene guardadas en "
                 f"{carpeta}.", _AMBAR, True)]
    if fila.prueba_perdida > 0:
        return [
            (f"No se lleva ninguna foto: el archivo nombra {fila.prueba_perdida} y ya no "
             f"estan en {carpeta}.", _ROJO_NO, True),
            ("Que no esten ahi no quiere decir que no existan: pueden estar en la papelera "
             "de la coleccion.", _TEXTO_TENUE, False),
        ]
    if fila.con_nombre_suelto > 0:
        return [(f"No tiene ninguna foto de prueba guardada: sus {fila.con_nombre_suelto} "
                 "dientes solo tienen anotado el nombre de una foto.", _TEXTO_TENUE, False)]
    return [("No tiene ninguna foto de prueba guardada.", _TEXTO_TENUE, False)]


def _dibujar_borrar_orden(lienzo, estado: EstadoPanel, botones: dict, etiquetas: list) -> list:
    """La confirmacion de mandar una orden a la papelera.

    La orden se nombra con su numero, CON EL NOMBRE DEL ARCHIVO cuando no es el que al
    numero le toca, y con lo que se lleva puesto --el informe y las fotos de prueba-- porque
    el que aprieta esto puede estar borrando la unica prueba que hay de un trabajo, y porque
    el indice de la fila elegida se mueve solo cuando el historial se vuelve a leer: lo que
    vale es lo que esta escrito aca.

    EL ARCHIVO SE NOMBRA PORQUE EL BORRADO VA POR EL ARCHIVO. Medido con la orden 1234 del
    taller y su copia de Windows "1234 - copia.json": las dos pantallas salian palabra por
    palabra iguales (a == b daba True, ningun renglon distinto) y ninguna nombraba el
    archivo. Mira hay_que_nombrar_el_archivo().

    LOS RENGLONES SE ARMAN Y DESPUES SE MIDE LA TARJETA, no al reves. Antes cada renglon
    tenia su y escrito a mano --y + 108, y + 138, y + 170 + extra-- y agregar uno obligaba a
    correr todos los de abajo. Ahora cada renglon dice cuanto baja el que sigue y la tarjeta
    se dibuja del alto que hace falta.
    MEDIDO: el caso mas cargado son nueve renglones --archivo nombrado, las dos cosas de la
    prueba y la orden abierta-- y al alto del taller (672 px) el ultimo cae en el pixel 396
    contra un renglon de aviso que arranca en el 548. Al alto minimo que usa run_live antes
    de medir la pantalla (492 px) hace falta apretar el espaciado: mira el comentario de
    abajo, donde estan las dos cuentas.
    """
    alto = lienzo.shape[0]
    lineas: list = []
    fila = _orden_del_historial(estado)
    _cabecera("BORRAR UNA ORDEN", "", lineas)

    y = _ALTO_CABECERA + 20
    # (texto, cuanto baja el renglon siguiente, color, tamano, negrita)
    cuerpo: list[tuple[str, int, tuple, int, bool]] = []
    if fila is None:
        cuerpo.append(("No hay ninguna orden elegida.", 52, _TEXTO, 32, True))
        cuerpo.append(("Volve al historial y elegi una.", 0, _TEXTO_TENUE, 22, False))
    else:
        cliente = fila.cliente.strip() or "sin cliente anotado"
        cuerpo.append((f"Sacar la orden {fila.numero} del historial", 50, _TEXTO, 32, True))
        if hay_que_nombrar_el_archivo(fila):
            # EN AMBAR Y ARRIBA DE TODO lo demas: es lo unico que distingue esta fila de la
            # otra que lleva el mismo numero adentro, y es por lo que se guia el borrado.
            cuerpo.append((
                f"El archivo que se va es {fila.archivo}.json, que no se llama como su "
                "numero", 30, _AMBAR, 20, True,
            ))
        cuerpo.append((
            _acortar(f"Cliente: {cliente}   -   abierta {fila.abierta or '-'}",
                     ANCHO_LIENZO - 2 * _MARGEN - 56, 22),
            32, _TEXTO_TENUE, 22, False,
        ))
        if fila.ilegible:
            cuerpo.append(("El archivo no se pudo leer, asi que no se sabe que tiene "
                           "adentro.", 30, _AMBAR, 20, True))
        else:
            cuerpo.append((f"{fila.revisados} de {fila.total_dientes} dientes revisados, "
                           f"{fila.rechazados} para rechazar", 30, _TEXTO_TENUE, 20, False))
        # QUE PRUEBA SE PIERDE. Es lo unico de esta pantalla que no se puede volver a hacer:
        # los dientes se pueden revisar otra vez, la foto del diente de la semana pasada no.
        for texto, color, negrita in _renglones_de_la_prueba(fila):
            cuerpo.append((texto, 30, color, 20, negrita))
        if fila.esta_abierta:
            # Es la orden con la que se esta trabajando: al irse, el programa queda sin
            # ninguna orden abierta y los dientes que se revisen despues no se anotan en
            # ningun lado hasta que se abra otra. Eso hay que saberlo ANTES de apretar.
            cuerpo.append(("ES LA ORDEN ABIERTA: el programa va a quedar sin ninguna orden "
                           "abierta.", 30, _AMBAR, 20, True))
        # EL PIE PROMETE LAS FOTOS SOLO SI SE VAN FOTOS. Con la orden renombrada a mano, el
        # borrado se llevo el .json y los dos informes y NINGUNA foto, mientras este renglon
        # decia "con su informe y sus fotos". Se promete lo que se mueve.
        if fila.con_prueba > 0:
            cuerpo.append(("No se pierde: se mueve a data/ordenes/_papelera con su informe "
                           "y sus", 28, _TEXTO_TENUE, 19, False))
            cuerpo.append(("fotos, y de ahi se puede volver a sacar con el Explorador de "
                           "Windows.", 0, _TEXTO_TENUE, 19, False))
        else:
            cuerpo.append(("No se pierde: se mueve a data/ordenes/_papelera con su informe, "
                           "y de ahi", 28, _TEXTO_TENUE, 19, False))
            cuerpo.append(("se puede volver a sacar con el Explorador de Windows.",
                           0, _TEXTO_TENUE, 19, False))

    # EL ESPACIADO SE APRIETA SOLO SI NO ENTRA, y en la PC del taller no se aprieta nunca.
    # Esta pantalla ahora puede llegar a nueve renglones --el archivo nombrado, las dos
    # cosas de la prueba y el aviso de la orden abierta-- y el lienzo no siempre tiene el
    # alto del taller: run_live arranca con revision.ALTO_MINIMO_LIENZO, que son 492 px, y
    # recien despues de componer la pantalla principal una vez sabe cuanto mide de verdad.
    # MEDIDO, el caso de nueve renglones:
    #   alto 672 (la PC del taller) -> entran 392 px, se piden 260: NO se aprieta nada, el
    #                                  ultimo renglon cae en el pixel 396 y el renglon del
    #                                  aviso arranca en el 548.
    #   alto 492 (el minimo)        -> entran 212 px y se piden 260, asi que se aprieta a
    #                                  0,82 --saltos de 40 y 24 px en vez de 50 y 30-- y el
    #                                  ultimo renglon cae en el 346, justo arriba del aviso,
    #                                  que ahi arranca en el 368.
    # Sin esto, a 492 px los dos ultimos renglones de la tarjeta se dibujaban ENCIMA del
    # renglon del aviso y de la linea de teclas.
    # El tope de abajo es el renglon del aviso --alto - _ALTO_BOTONES - 48-- y no la fila de
    # botones: el aviso es el que lleva lo del borrado a medias, y taparlo seria repetir el
    # problema que se acaba de arreglar.
    saltos = [salto for _t, salto, *_r in cuerpo]
    hasta_el_aviso = (alto - _ALTO_BOTONES - 48) - (y + 24) - 20
    if sum(saltos) > hasta_el_aviso > 0:
        factor = hasta_el_aviso / sum(saltos)
        # 24 px de piso: es lo que mide de alto un renglon de 20 px mas un pelo de aire. Y
        # el ultimo salto es siempre 0 --no hay renglon despues-- asi que no se le pone piso.
        saltos = [0 if salto == 0 else max(24, int(salto * factor)) for salto in saltos]

    alto_tarjeta = 24 + sum(saltos) + 30
    _tarjeta(lienzo, (_MARGEN, y), (ANCHO_LIENZO - _MARGEN, y + alto_tarjeta),
             _SUPERFICIE, 14)
    cursor = y + 24
    for (texto, _pedido, color, tamano, negrita), salto in zip(cuerpo, saltos):
        lineas.append((texto, (_MARGEN + 28, cursor), color, tamano, negrita))
        cursor += salto

    if estado.aviso:
        lineas.append((estado.aviso, (_MARGEN + 4, alto - _ALTO_BOTONES - 48),
                       _AMBAR, 20, True))
    # Las dos teclas escritas, igual que al borrar un concepto: es la otra pantalla donde una
    # tecla hace algo que no se deshace con otra tecla.
    lineas.append(("teclas:   s = si, borrala      n o ESC = no, dejala",
                   (_MARGEN + 4, alto - _ALTO_BOTONES - 22), _TEXTO_TENUE, 16, False))

    _fila_de_botones(
        lienzo, alto,
        [("confirmar_borrado_orden", "SI, BORRAR", _ROJO_NO),
         ("cancelar_borrado_orden", "NO, DEJALA", _SUPERFICIE)],
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
