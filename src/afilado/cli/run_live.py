"""Inspeccion en vivo con la webcam cenital y bucle de feedback del operario.

PRINCIPIO DE LA MEMORIA DUPLICADA (el nucleo del auto-entrenamiento)
--------------------------------------------------------------------
Cada frame vive dos veces y nunca se mezclan:

    limpio  = lo que salio de la camara. JAMAS se dibuja un pixel encima.
    anotado = overlay.dibujar(limpio, ...) -> COPIA. Es lo unico que se muestra
              en pantalla y lo unico que se graba con --grabar.

Al almacen de feedback va SIEMPRE el frame limpio. Si se guardara la imagen con
las cajas ya dibujadas, al reentrenar la red aprenderia a detectar rectangulos
verdes en vez de desgaste real: la anotacion seria la senal mas facil de
aprender y el modelo colapsaria sobre ella. La copia anotada se guarda aparte,
en revision/, solo para que un humano audite que penso la IA.

El operario corrige a la IA con una tecla: 'e' marca un fallo (lo que la IA erro),
'g' marca un acierto y el espacio captura sin emitir juicio. Cada pulsacion deja
en disco imagen limpia + pre-etiquetado YOLO + informe, listo para corregir en
Roboflow y reentrenar.

Este es el UNICO modulo del paquete autorizado a usar cv2.imshow/waitKey.
"""

from __future__ import annotations

import argparse
import logging
import logging.handlers
import sys
import time
from pathlib import Path
from typing import Optional, Union

import cv2
import numpy as np

from .. import overlay, panel, revision
from ..referencias import GaleriaReferencias
from ..almacen import AlmacenClasificacion, AlmacenFeedback
from ..camara import Camara, enumerar_camaras
from ..conceptos import (
    EXTENSIONES,
    LARGO_MAXIMO,
    Concepto,
    Conceptos,
    caracter_valido,
    sanear,
)
from ..config import AppConfig, cargar_config, raiz_repo, ruta_absoluta
from ..detector import crear_clasificador
from ..memoria import Memoria
from ..ordenes import Orden, Ordenes
from ..pipeline import Pipeline
from ..tipos import ResultadoFrame, Veredicto

# Carpeta donde el modo clasificacion archiva las fotos por clase. Su estructura es
# exactamente la que consume scripts/construir_clasificacion.py: <dir>/<clase>/*.jpg
_DIR_COLECCION = "data/coleccion"

# La galeria de fotos de modelo aparece en el panel como un destino mas. No es un concepto
# --no dice si el filo esta sano, dice que modelo de sierra es-- pero ES una carpeta a la
# que el operario manda fotos, y el registro del taller muestra que es la que mas usa. Si
# no estuviera en la misma lista, el panel contestaria a medias la pregunta "a donde va
# esta foto".
_DESTINO_MODELO = "foto de modelo"

# Donde quedan las ordenes de afilado y sus informes, al lado del programa.
_DIR_ORDENES = "data/ordenes"

# Donde queda el registro de lo que paso. Abierto con doble clic NO hay consola --esa es
# justamente la idea-- asi que si el log solo fuera a pantalla se perderia entero, y es lo
# unico con que se diagnostica un problema de camara en el taller. Rota para no crecer sin
# limite: dos vueltas de 1 MB alcanzan de sobra para ver la sesion de hoy y la anterior.
_ARCHIVO_REGISTRO = "registro.txt"
_BYTES_REGISTRO = 1_000_000
_VUELTAS_REGISTRO = 2

_registro = logging.getLogger(__name__)

_VENTANA = "Analisis de afilado"

# Una webcam devuelve None sueltos cuando el driver hipa; eso no es una averia.
# Pero decenas seguidas si: el cable se solto o la camara se apago. Sin este tope
# el bucle se quedaria mudo girando en vacio para siempre.
_MAX_FRAMES_NULOS = 30

# Duracion de la confirmacion verde tras guardar. Suficiente para leerla sin
# taparle la pieza al operario.
_SEGUNDOS_MENSAJE = 1.5

_ESC = 27

# Presets que cicla la tecla 'c'. El primero se sustituye en tiempo de ejecucion
# por el ROI que traiga el config, para que ciclar siempre pueda volver a el.
_PRESETS_ROI: tuple[tuple[str, bool, float, float, float, float], ...] = (
    ("configurado", True, 0.15, 0.15, 0.7, 0.7),
    ("centro estrecho", True, 0.30, 0.30, 0.4, 0.4),
    ("frame completo", False, 0.0, 0.0, 1.0, 1.0),
)

_EPILOGO_CODIGOS = (
    "codigos de salida: 0 exito, 1 error de arranque, 2 la fuente dejo de entregar frames"
)


def _interpretar_fuente(texto: str) -> Union[int, str]:
    """Convierte el --fuente de la linea de comandos al tipo que espera CamaraConfig.

    Un texto numerico es un indice de camara; cualquier otra cosa es ruta o URL.
    """
    limpio = texto.strip()
    if limpio.lstrip("+-").isdigit():
        return int(limpio)
    return limpio


def _es_camara(fuente: Union[int, str]) -> bool:
    """Replica la regla de camara._resolver_fuente para distinguir webcam de archivo.

    Se duplica a proposito y no se importa: esa funcion es privada del modulo camara.
    Importa la distincion porque un archivo que se termina es un final NORMAL (codigo 0),
    mientras que una webcam que deja de responder es una averia (codigo 2).
    """
    if isinstance(fuente, int):
        return True
    texto = str(fuente).strip()
    if texto.lower() == "auto":
        return True
    if texto.lstrip("+-").isdigit():
        return True
    return False


def _construir_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="run_live",
        description="Inspeccion en vivo de herramientas de afilado con webcam cenital.",
        epilog=_EPILOGO_CODIGOS,
    )
    parser.add_argument("--config", default=None, help="ruta del YAML de configuracion")
    parser.add_argument(
        "--fuente",
        default=None,
        help=(
            "indice de webcam (0, 1, ...), 'auto' para que la busque sola, ruta de video "
            "o URL de stream"
        ),
    )
    parser.add_argument(
        "--listar-camaras",
        action="store_true",
        help="muestra las camaras que responden en esta maquina y termina",
    )
    parser.add_argument("--pesos", default=None, help="ruta del .pt del modelo YOLO")
    parser.add_argument(
        "--conf", type=float, default=None, help="umbral de confianza 0..1"
    )
    parser.add_argument(
        "--sin-roi",
        action="store_true",
        help="analiza el frame completo, ignorando la region de interes",
    )
    parser.add_argument(
        "--grabar",
        default=None,
        metavar="RUTA.mp4",
        help="graba el video ANOTADO (lo que se ve en pantalla) en la ruta indicada",
    )
    return parser


def _aplicar_argumentos(cfg: AppConfig, args: argparse.Namespace) -> None:
    """Vuelca los argumentos de linea de comandos sobre el config ya cargado."""
    if args.fuente is not None:
        cfg.camara.fuente = _interpretar_fuente(args.fuente)
    if args.pesos is not None:
        cfg.detector.pesos = args.pesos
    if args.conf is not None:
        if not 0.0 <= args.conf <= 1.0:
            raise ValueError(
                f"--conf debe estar entre 0 y 1, se recibio {args.conf}."
            )
        cfg.detector.confianza = args.conf
    if args.sin_roi:
        cfg.roi.habilitado = False


def _presets_con_config(cfg: AppConfig) -> list[tuple[str, bool, float, float, float, float]]:
    """Lista de presets de ROI cuyo primer elemento es el ROI real del config."""
    presets = list(_PRESETS_ROI)
    presets[0] = (
        "configurado",
        cfg.roi.habilitado,
        cfg.roi.x,
        cfg.roi.y,
        cfg.roi.w,
        cfg.roi.h,
    )
    return presets


def _aplicar_preset_roi(
    cfg: AppConfig, preset: tuple[str, bool, float, float, float, float]
) -> str:
    """Muta cfg.roi en el sitio. Pipeline relee cfg.roi en cada frame, asi que
    el cambio entra en vigor en el frame siguiente sin reconstruir nada."""
    nombre, habilitado, x, y, w, h = preset
    cfg.roi.habilitado = habilitado
    cfg.roi.x, cfg.roi.y, cfg.roi.w, cfg.roi.h = x, y, w, h
    return nombre


def _abrir_grabador(ruta: Path, frame: np.ndarray, fps: int) -> Optional[cv2.VideoWriter]:
    """Abre el VideoWriter mp4v con el tamano REAL del frame entregado.

    Se abre con el primer frame y no con cfg.camara.ancho/alto porque la webcam
    puede negociar una resolucion distinta a la pedida; si el tamano del writer no
    coincide con el del frame, OpenCV descarta cada write en silencio y el mp4
    sale vacio.
    """
    alto, ancho = frame.shape[:2]
    ruta.parent.mkdir(parents=True, exist_ok=True)
    codec = cv2.VideoWriter_fourcc(*"mp4v")
    grabador = cv2.VideoWriter(str(ruta), codec, float(max(fps, 1)), (ancho, alto))
    if not grabador.isOpened():
        _registro.error(
            "no se pudo abrir el grabador en %s; se continua sin grabar", ruta
        )
        return None
    _registro.info("grabando video anotado en %s (%dx%d @%d fps)", ruta, ancho, alto, fps)
    return grabador


def _describir_escala(resultado: ResultadoFrame) -> str:
    escala = resultado.escala
    if escala.valida and escala.mm_por_px:
        return (
            f"{escala.mm_por_px:.4f} mm/px (fuente={escala.fuente}, "
            f"id={escala.id_marcador}, error_lados={escala.error_lados:.3f})"
        )
    return "SIN ESCALA - no se vio el marcador ArUco, las medidas no seran fiables"


def _guardar(
    almacen: AlmacenFeedback,
    limpio: np.ndarray,
    resultado: ResultadoFrame,
    anotado: np.ndarray,
    etiqueta: str,
) -> str:
    """Persiste la captura y devuelve el mensaje de confirmacion para el overlay."""
    try:
        ruta = almacen.guardar(limpio, resultado, etiqueta=etiqueta, frame_anotado=anotado)
    except OSError as error:
        _registro.error("no se pudo guardar la captura: %s", error)
        return "ERROR AL GUARDAR - revise el disco"
    return f"GUARDADO {etiqueta}: {ruta.name}"


def _clases_de_teclas(cfg: AppConfig) -> dict[int, str]:
    """Mapea 'a' a la primera clase sana y 'r' a la primera clase de defecto.

    Con el esquema binario aprobada/rechazada esto queda a=aprobada, r=rechazada.

    UNA TECLA SIN DESTINO NO HACE NADA. Antes, si no habia ninguna clase sana, la 'a' caia
    en cfg.clases[0] --que podia ser un concepto de RECHAZO-- y el operario que apretaba
    GUARDAR BUENA archivaba la foto como mala sin enterarse. Una foto mal etiquetada no se
    nota: entra al dataset, se entrena con ella y el modelo sale raro meses despues. Mejor
    que la tecla no haga nada, igual que el boton que ya desaparecio de la barra.
    """
    sanas = [c for c in cfg.clases if c not in cfg.clases_defecto]
    defectuosas = [c for c in cfg.clases if c in cfg.clases_defecto]
    teclas: dict[int, str] = {}
    if sanas:
        teclas[ord("a")] = sanas[0]
    if defectuosas:
        teclas[ord("r")] = defectuosas[0]
    return teclas


# Etiquetas de taller para los dos conceptos de siempre. "GUARDAR BUENA" se lee de un
# vistazo desde lejos; "GUARDAR APROBADA" no es mas claro y ocupa mas. Para un concepto
# que invento el taller se usa su propio nombre, que es el unico que significa algo.
_ETIQUETAS_AMIGAS = {"aprobada": "GUARDAR BUENA", "rechazada": "GUARDAR MALA"}

_TECLAS_DIGITO = {ord(str(numero)): numero for numero in range(1, 10)}


def _volcar_conceptos(cfg: AppConfig, conceptos: Conceptos) -> None:
    """Hace que cfg.clases sea lo que hay en disco, no lo que diga el YAML.

    El config queda como semilla del primer arranque. Despues la lista vive en las carpetas
    y en el panel, que es donde el operario la puede cambiar.
    """
    nombres = conceptos.nombres()
    if not nombres and not conceptos.hubo_indice:
        # Primer arranque, sin nada en disco: se respeta el config, que es la semilla.
        return
    # Pero si el operario BORRO todos los conceptos, la lista vacia es la verdad: dejar la
    # del config haria aparecer botones y teclas de conceptos que ya no existen, y
    # apretarlos los resucitaria con una carpeta nueva.
    cfg.clases = nombres
    cfg.clases_defecto = [n for n in conceptos.defectos() if n in nombres]


def _destinos_rapidos(cfg: AppConfig) -> list[tuple[str, str, bool]]:
    """Los dos botones de guardado directo: el primer concepto sano y el primero de defecto."""
    sanas = [c for c in cfg.clases if c not in cfg.clases_defecto]
    defectuosas = [c for c in cfg.clases if c in cfg.clases_defecto]
    destinos: list[tuple[str, str, bool]] = []
    for nombre, es_defecto in ((sanas[:1], False), (defectuosas[:1], True)):
        for clase in nombre:
            destinos.append(
                (clase, _ETIQUETAS_AMIGAS.get(clase, f"GUARDAR {clase.upper()}"), es_defecto)
            )
    return destinos


def _lista_para_panel(conceptos: Conceptos, galeria: GaleriaReferencias) -> list[Concepto]:
    """Los conceptos mas la galeria de modelos, que es el otro destino posible de una foto."""
    modelo = Concepto(
        nombre=_DESTINO_MODELO,
        carpeta=galeria.carpeta,
        fotos=galeria.cantidad,
        defecto=False,
        especial=True,
    )
    return [modelo] + conceptos.listar()


def _fotos_del_destino(
    destino: Optional[Concepto], conceptos: Conceptos, galeria: GaleriaReferencias
) -> list[Path]:
    if destino is None:
        return []
    if destino.especial:
        carpeta = galeria.carpeta
        if not carpeta.is_dir():
            return []
        return sorted(
            (p for p in carpeta.iterdir()
             if p.is_file() and p.suffix.lower() in EXTENSIONES),
            key=lambda p: p.name,
            reverse=True,
        )
    return conceptos.fotos(destino.nombre)


def _atender_panel(
    accion: Optional[str],
    tecla: int,
    pan: panel.EstadoPanel,
    conceptos: Conceptos,
    galeria: GaleriaReferencias,
    miniaturas: panel.Miniaturas,
    archivar,
    refrescar,
    ordenes,
) -> tuple[bool, Optional[str]]:
    """Resuelve una tecla o un clic del panel. Devuelve (sigue_abierto, mensaje).

    Vive en run_live y no en panel.py por la misma razon que revision.py no lee teclas:
    el que dibuja no decide. Aca se puede leer de corrido que hace cada tecla.
    """
    if pan.modo == panel.MODO_NUEVO:
        return _panel_nuevo(accion, tecla, pan, conceptos, refrescar)
    if pan.modo == panel.MODO_BORRAR:
        return _panel_borrar(accion, tecla, pan, conceptos, miniaturas, refrescar)
    if pan.modo == panel.MODO_ORDEN:
        return _panel_orden(accion, tecla, pan, ordenes)
    if pan.modo == panel.MODO_FOTOS:
        return _panel_fotos(accion, tecla, pan, conceptos, galeria, miniaturas, refrescar)
    return _panel_lista(accion, tecla, pan, conceptos, galeria, miniaturas, archivar, refrescar)


def _panel_orden(accion, tecla, pan, ordenes) -> tuple[bool, Optional[str]]:
    """La orden de afilado: cargarla a mano, ver el avance, sacar el informe."""
    cargando = pan.orden is None or pan.cargando_orden

    # La 'q' vuelve igual que ESC, pero SOLO con la orden ya cargada: adentro del
    # formulario es una letra mas --hay clientes que se llaman "Quiroga"-- y ahi la unica
    # salida es ESC, que es lo que dice la pantalla.
    if accion == "volver_orden" or tecla == _ESC or (tecla == ord("q") and not cargando):
        if cargando and pan.orden is not None:
            pan.cargando_orden, pan.aviso = False, None    # vuelve al avance
            return True, None
        pan.modo, pan.cargando_orden, pan.aviso = panel.MODO_LISTA, False, None
        return True, None

    if cargando:
        return _cargar_orden(accion, tecla, pan, ordenes)

    orden = pan.orden
    if accion == "nueva_orden" or tecla == ord("n"):
        pan.cargando_orden = True
        pan.valores = ["" for _ in panel.CAMPOS_ORDEN]
        pan.campo, pan.aviso = 0, None
    elif accion == "siguiente_sierra" or tecla == ord("s"):
        numero = orden.siguiente_sierra()
        ordenes.guardar()
        return True, f"SIERRA {numero} de {len(orden.sierras)}"
    elif accion == "saltear_diente" or tecla == ord("x"):
        diente = orden.saltear()
        ordenes.guardar()
        if diente is None:
            pan.aviso = "esta sierra ya tiene todos los dientes revisados"
            return True, None
        return True, f"DIENTE {diente.numero} SALTEADO"
    elif accion == "informe" or tecla == ord("i"):
        texto, _ = ordenes.escribir_informe()
        if texto is None:
            pan.aviso = "no se pudo escribir el informe; revise el disco"
            return True, None
        return True, f"INFORME EN {texto.name}"
    elif accion == "cerrar_orden":
        numero = orden.numero
        ordenes.escribir_informe()
        ordenes.cerrar()
        pan.orden = None
        return True, f"ORDEN {numero} CERRADA - el informe quedo guardado"
    return True, None


def _cargar_orden(accion, tecla, pan, ordenes) -> tuple[bool, Optional[str]]:
    """El formulario. Un campo por vez, ENTER pasa al siguiente."""
    if accion is not None and accion.startswith("campo:"):
        pan.campo = int(accion.split(":", 1)[1])
        pan.aviso = None
        return True, None

    if accion == "crear_orden" or (tecla == 13 and pan.campo >= len(panel.CAMPOS_ORDEN) - 1):
        valores = [v.strip() for v in pan.valores]
        try:
            orden = Orden.nueva(
                numero=valores[0],
                cantidad_sierras=int(valores[3] or 0),
                dientes_por_sierra=int(valores[4] or 0),
                nota_pedido=valores[1],
                cliente=valores[2],
            )
        except ValueError as error:
            pan.aviso = str(error)
            return True, None
        if ordenes.cargar(orden.numero) is not None:
            pan.aviso = f"ya hay una orden {orden.numero} guardada; usa otro numero"
            return True, None
        ordenes.abrir(orden)
        pan.orden, pan.cargando_orden, pan.aviso = orden, False, None
        return True, f"ORDEN {orden.numero}: {len(orden.sierras)} sierras de {orden.sierras[0].dientes} dientes"

    if tecla == 13:
        pan.campo = min(pan.campo + 1, len(panel.CAMPOS_ORDEN) - 1)
        pan.aviso = None
    elif tecla in (8, 127):
        pan.valores[pan.campo] = pan.valores[pan.campo][:-1]
    elif 32 <= tecla <= 126:
        letra = chr(tecla)
        _, tipo = panel.CAMPOS_ORDEN[pan.campo]
        if tipo == "numero" and not letra.isdigit():
            # Una letra en "cuantas sierras" no es un error de dedo que se arregla solo:
            # es un dato que despues rompe el informe.
            pan.aviso = "aca va un numero"
        elif len(pan.valores[pan.campo]) < 40:
            pan.valores[pan.campo] += letra
            pan.aviso = None
    return True, None


def _panel_lista(
    accion, tecla, pan, conceptos, galeria, miniaturas, archivar, refrescar
) -> tuple[bool, Optional[str]]:
    if accion == "volver" or tecla in (_ESC, ord("q")):
        return False, None

    # El punto de color: si este concepto significa "pieza para tirar". Es lo que decide si
    # la pantalla pinta verde o rojo cuando la memoria lo elige.
    if accion is not None and accion.startswith("defecto:"):
        pan.elegido = int(accion.split(":", 1)[1])
        tecla = ord("d")
    if tecla == ord("d"):
        elegido = pan.concepto_elegido
        if elegido is None or elegido.especial:
            pan.aviso = "'foto de modelo' no es un concepto: no es ni bueno ni malo"
            return True, None
        try:
            conceptos.marcar_defecto(elegido.nombre, not elegido.defecto)
        except (ValueError, OSError) as error:
            _registro.error("no se pudo marcar '%s': %s", elegido.nombre, error)
            pan.aviso = "no se pudo guardar el cambio; revise el disco"
            return True, None
        refrescar(recalcular=False)
        ahora = pan.concepto_elegido
        es_malo = ahora.defecto if ahora is not None else not elegido.defecto
        return True, (
            f"'{elegido.nombre}' es una pieza PARA TIRAR" if es_malo
            else f"'{elegido.nombre}' es una pieza BUENA"
        )

    if accion is not None and accion.startswith("elegir:"):
        pan.elegido = int(accion.split(":", 1)[1])
        pan.aviso = None
    elif tecla in _TECLAS_DIGITO:
        indice = _TECLAS_DIGITO[tecla] - 1
        if indice < len(pan.conceptos):
            pan.elegido = indice
    elif tecla == ord("s"):
        pan.mover(1)
    elif tecla == ord("w"):
        pan.mover(-1)

    elegido = pan.concepto_elegido
    if accion == "nuevo" or tecla == ord("n"):
        pan.modo, pan.escribiendo, pan.aviso = panel.MODO_NUEVO, "", None
    elif (accion == "borrar" or tecla == ord("b")) and elegido is not None:
        if elegido.especial:
            pan.aviso = (
                "'foto de modelo' no es un concepto que se pueda borrar; sus fotos se "
                "sacan de a una desde VER FOTOS"
            )
        else:
            pan.modo = panel.MODO_BORRAR
    elif accion == "orden" or tecla == ord("o"):
        pan.modo, pan.aviso = panel.MODO_ORDEN, None
        pan.cargando_orden = pan.orden is None
        if pan.cargando_orden:
            pan.valores = ["" for _ in panel.CAMPOS_ORDEN]
            pan.campo = 0
    elif (accion == "fotos" or tecla == ord("v")) and elegido is not None:
        pan.fotos = _fotos_del_destino(elegido, conceptos, galeria)
        pan.modo, pan.pagina, pan.foto_elegida, pan.aviso = panel.MODO_FOTOS, 0, 0, None
        miniaturas.olvidar()
    elif (accion == "guardar_aqui" or tecla == 13) and elegido is not None:
        try:
            return False, archivar(elegido.nombre)
        except (OSError, ValueError) as error:
            _registro.error("no se pudo archivar en '%s': %s", elegido.nombre, error)
            pan.aviso = "no se pudo guardar; revise el disco"
    return True, None


def _panel_nuevo(accion, tecla, pan, conceptos, refrescar) -> tuple[bool, Optional[str]]:
    if accion == "volver_lista" or tecla == _ESC:
        pan.modo, pan.escribiendo, pan.aviso = panel.MODO_LISTA, "", None
        return True, None

    if accion == "confirmar_nuevo" or tecla == 13:
        nombre = sanear(pan.escribiendo)
        if not nombre:
            pan.aviso = "escribi un nombre para el concepto"
            return True, None
        try:
            nuevo = conceptos.agregar(nombre)
        except (ValueError, OSError) as error:
            pan.aviso = str(error)
            return True, None
        refrescar()
        pan.modo, pan.escribiendo, pan.aviso = panel.MODO_LISTA, "", None
        for indice, concepto in enumerate(pan.conceptos):
            if concepto.nombre == nuevo.nombre:
                pan.elegido = indice
                break
        return True, f"CONCEPTO NUEVO: {nuevo.nombre}"

    if tecla in (8, 127):  # retroceso
        pan.escribiendo = pan.escribiendo[:-1]
    elif 32 <= tecla <= 126 and len(pan.escribiendo) < LARGO_MAXIMO:
        letra = chr(tecla)
        if caracter_valido(letra):
            pan.escribiendo += letra
        else:
            pan.aviso = "solo letras, numeros, espacios y guiones"
    return True, None


def _panel_borrar(accion, tecla, pan, conceptos, miniaturas, refrescar):
    """La confirmacion de borrado. ENTER NO alcanza, a proposito.

    En la lista ENTER es "guardar aca", la tecla que el operario aprieta todo el dia. Si
    aca tambien confirmara, un 'b' de mas seguido del ENTER de siempre se llevaria puesto
    un concepto entero. Hay que decir que si: el boton, o la 's'.
    """
    elegido = pan.concepto_elegido
    if accion == "confirmar_borrado" or tecla == ord("s"):
        if elegido is None:
            pan.modo = panel.MODO_LISTA
            return True, None
        nombre = elegido.nombre
        try:
            cuantas = conceptos.eliminar(nombre)
        except (ValueError, OSError) as error:
            _registro.error("no se pudo borrar el concepto '%s': %s", nombre, error)
            pan.modo, pan.aviso = panel.MODO_LISTA, "no se pudo borrar; revise el disco"
            return True, None
        miniaturas.olvidar()
        refrescar()
        pan.modo = panel.MODO_LISTA
        return True, f"'{nombre}' y sus {cuantas} foto(s) fueron a _papelera"
    if accion == "volver_lista" or tecla in (_ESC, ord("n"), ord("q")):
        pan.modo = panel.MODO_LISTA
    return True, None


def _panel_fotos(accion, tecla, pan, conceptos, galeria, miniaturas, refrescar):
    if accion == "volver_lista" or tecla in (_ESC, ord("q"), ord("v")):
        pan.modo, pan.aviso = panel.MODO_LISTA, None
        miniaturas.olvidar()
        return True, None

    paginas = max(1, (len(pan.fotos) + panel._POR_PAGINA - 1) // panel._POR_PAGINA)
    if accion is not None and accion.startswith("foto:"):
        pan.foto_elegida = int(accion.split(":", 1)[1])
    elif tecla in _TECLAS_DIGITO and _TECLAS_DIGITO[tecla] <= panel._POR_PAGINA:
        # Solo los digitos de ESTA pagina. En pantalla hay ocho miniaturas; el 9 elegia una
        # que no se ve, y la 'b' siguiente la mandaba a la papelera a ciegas.
        indice = pan.pagina * panel._POR_PAGINA + _TECLAS_DIGITO[tecla] - 1
        if indice < len(pan.fotos):
            pan.foto_elegida = indice
    elif accion == "siguiente_pagina" or tecla == ord(" "):
        pan.pagina = (pan.pagina + 1) % paginas
        pan.foto_elegida = pan.pagina * panel._POR_PAGINA

    if accion == "borrar_foto" or tecla == ord("b"):
        if not 0 <= pan.foto_elegida < len(pan.fotos):
            return True, None
        ruta = pan.fotos[pan.foto_elegida]
        elegido = pan.concepto_elegido
        try:
            conceptos.eliminar_foto(ruta)
        except (ValueError, OSError) as error:
            _registro.error("no se pudo borrar la foto %s: %s", ruta, error)
            pan.aviso = "no se pudo borrar la foto; revise el disco"
            return True, None
        miniaturas.olvidar(ruta)
        if elegido is not None and elegido.especial:
            galeria.recargar()
        refrescar()
        pan.fotos = _fotos_del_destino(pan.concepto_elegido, conceptos, galeria)
        pan.foto_elegida = max(0, min(pan.foto_elegida, len(pan.fotos) - 1))
        pan.pagina = min(
            pan.pagina,
            max(0, (len(pan.fotos) + panel._POR_PAGINA - 1) // panel._POR_PAGINA - 1),
        )
        return True, f"FOTO A LA PAPELERA: {ruta.name}"
    return True, None


def _ejecutar_clasificacion(cfg: AppConfig, args: argparse.Namespace) -> int:
    """Bucle del modo clasificacion: muestra el veredicto y archiva fotos por clase.

    Si no hay modelo entrenado, entra en modo recoleccion: no juzga, pero el operario puede
    archivar cada sierra en su clase con las teclas, para juntar el dataset y entrenar despues.
    """
    clasificador = crear_clasificador(cfg)
    almacen = AlmacenClasificacion(_DIR_COLECCION, cfg.clases, cfg.feedback.calidad_jpg)
    teclas_clase = _clases_de_teclas(cfg)
    cada_n_frames = cfg.detector.clasificar_cada_n_frames
    fuente_es_camara = _es_camara(cfg.camara.fuente)
    ruta_grabar = ruta_absoluta(args.grabar) if args.grabar else None
    grabador: Optional[cv2.VideoWriter] = None

    camara = Camara(cfg.camara)
    try:
        camara.abrir()
    except (RuntimeError, ValueError) as error:
        _registro.error("%s", error)
        return 1

    _registro.info("MODO CLASIFICACION | camara: %s", camara.descripcion)
    _registro.info(
        "modelo: %s",
        clasificador.descripcion if clasificador else "SIN MODELO (modo recoleccion)",
    )
    if clasificador is not None:
        _registro.info(
            "clasificando 1 de cada %d frames (detector.clasificar_cada_n_frames)",
            cada_n_frames,
        )
    _registro.info("archivando fotos en: %s", ruta_absoluta(_DIR_COLECCION))
    _registro.info(
        "teclas: q/ESC salir | a guardar APROBADA | r guardar RECHAZADA | p pausa | h ayuda",
    )

    codigo_salida = 0
    nulos_seguidos = 0
    frames_vistos = 0
    mostrar_ayuda = True
    en_pausa = False
    mensaje: Optional[str] = None
    expira_mensaje = 0.0
    limpio: Optional[np.ndarray] = None
    veredicto: Optional[Veredicto] = None

    try:
        cv2.namedWindow(_VENTANA, cv2.WINDOW_NORMAL)
        while True:
            if not en_pausa:
                frame = camara.leer()
                if frame is None:
                    nulos_seguidos += 1
                    if nulos_seguidos >= _MAX_FRAMES_NULOS:
                        if fuente_es_camara:
                            _registro.error(
                                "la camara dejo de entregar frames (%d fallos seguidos).",
                                nulos_seguidos,
                            )
                            codigo_salida = 2
                        else:
                            _registro.info("el video termino")
                        break
                    continue
                nulos_seguidos = 0
                limpio = frame
                # La red se consulta 1 de cada N frames; en el medio se sigue mostrando el
                # ultimo veredicto. Es lo unico caro del bucle (35 ms contra 1 ms del resto),
                # y una sierra apoyada en la mesa no cambia de estado 30 veces por segundo.
                # El frame que se ve y el que se archiva con 'a'/'r' siguen siendo el actual:
                # lo que se reusa es el juicio, no la imagen.
                if clasificador is not None:
                    if frames_vistos % cada_n_frames == 0:
                        veredicto = clasificador.clasificar(limpio)
                    frames_vistos += 1

            if limpio is None:
                continue

            if mensaje is not None and time.monotonic() >= expira_mensaje:
                mensaje = None
            texto_mensaje = mensaje
            if en_pausa and texto_mensaje is None:
                texto_mensaje = "PAUSA - p para reanudar"

            anotado = overlay.dibujar_veredicto(
                limpio, veredicto, cfg,
                mostrar_ayuda=mostrar_ayuda, mensaje=texto_mensaje,
                capturas=almacen.total_guardados,
            )

            if ruta_grabar is not None and not en_pausa:
                if grabador is None:
                    grabador = _abrir_grabador(ruta_grabar, anotado, cfg.camara.fps)
                    if grabador is None:
                        ruta_grabar = None
                if grabador is not None:
                    grabador.write(anotado)

            cv2.imshow(_VENTANA, anotado)
            tecla = cv2.waitKey(1) & 0xFF

            if tecla in (ord("q"), _ESC):
                break
            if tecla == 255:
                continue
            if tecla in teclas_clase:
                clase = teclas_clase[tecla]
                motivo = motivo_para_no_archivar(camara, limpio, cfg, almacen)
                if motivo is not None:
                    _registro.warning("no se archivo la foto: %s", motivo)
                    mensaje = motivo
                else:
                    try:
                        ruta = almacen.guardar(limpio, clase)
                        mensaje = f"GUARDADA como {clase}: {ruta.name}"
                    except (OSError, ValueError) as error:
                        _registro.error("no se pudo guardar: %s", error)
                        mensaje = "ERROR AL GUARDAR - revise el disco"
                expira_mensaje = time.monotonic() + _SEGUNDOS_MENSAJE
            elif tecla == ord("p"):
                en_pausa = not en_pausa
                mensaje = None
            elif tecla == ord("h"):
                mostrar_ayuda = not mostrar_ayuda
    except KeyboardInterrupt:
        _registro.info("interrumpido por el operario")
    finally:
        if grabador is not None:
            grabador.release()
        camara.cerrar()
        cv2.destroyAllWindows()
        cv2.waitKey(1)

    _registro.info("fotos archivadas en esta sesion: %d", almacen.total_guardados)
    return codigo_salida


def _listar_camaras() -> int:
    """Muestra las camaras que responden. Es la herramienta de diagnostico del taller.

    Cuando el programa dice que no puede abrir la camara, esto contesta la unica pregunta
    que importa: hay camara, y en que indice. OpenCV va a imprimir advertencias en rojo por
    cada intento fallido; son normales y por eso se avisa antes.
    """
    print("Buscando camaras... (los avisos en rojo de OpenCV son normales)")
    print("Puede tardar unos segundos: se prueba indice por indice, con pausas.")
    print()
    camaras = enumerar_camaras()
    if not camaras:
        print("No respondio ninguna camara.")
        print()
        print("Cosas para revisar:")
        print("  - que este enchufada y que el cable USB haga contacto")
        print("  - que ningun otro programa la tenga tomada (Zoom, Teams, Camara de Windows)")
        print("  - en una notebook, que no este desactivada por una tecla de funcion")
        return 2

    print(f"{len(camaras)} camara(s):")
    for camara in camaras:
        print(f"  - {camara}")
    print()
    print("Si despues de esto alguna camara deja de responder, desenchufala y volve a")
    print("enchufarla: hay camaras USB baratas que se cuelgan cuando se las abre y cierra")
    print("varias veces seguidas. Windows las sigue mostrando como OK igual.")
    print()
    print("En configs/config.yaml, la seccion camara acepta:")
    print("    fuente: auto        <- la busca sola, es lo recomendado")
    print(f"    fuente: {camaras[0].indice}           <- fija esta en particular")
    return 0


def _tamano_pantalla() -> Optional[tuple[int, int]]:
    """(ancho, alto) del escritorio, o None si no se puede averiguar.

    Sin dependencias nuevas: ctypes es de la biblioteca estandar. Fuera de Windows se
    devuelve None y la ventana se abre del tamano del lienzo sin recortar.
    """
    if not sys.platform.startswith("win"):
        return None
    try:
        import ctypes

        usuario = ctypes.windll.user32
        usuario.SetProcessDPIAware()

        # SPI_GETWORKAREA da el escritorio SIN la barra de tareas, que es lo que de verdad
        # hay disponible: en este monitor son 1360x720 contra los 1360x768 de la pantalla,
        # y esos 48 px son los que se comerian los botones de abajo.
        class _Rect(ctypes.Structure):
            _fields_ = [
                ("left", ctypes.c_long),
                ("top", ctypes.c_long),
                ("right", ctypes.c_long),
                ("bottom", ctypes.c_long),
            ]

        area = _Rect()
        if usuario.SystemParametersInfoW(0x0030, 0, ctypes.byref(area), 0):
            ancho = int(area.right - area.left)
            alto = int(area.bottom - area.top)
        else:
            ancho = int(usuario.GetSystemMetrics(0))
            alto = int(usuario.GetSystemMetrics(1))
    except Exception:  # noqa: BLE001 - si falla, se sigue sin recortar
        return None
    if ancho <= 0 or alto <= 0:
        return None
    return ancho, alto


# Alto que se le va el marco de la ventana y la barra de titulo, que no son area de dibujo.
_ALTO_MARCO_VENTANA = 48


def _alto_util_pantalla() -> Optional[int]:
    """Alto en px que puede ocupar el DIBUJO, descontando marco y barra de tareas."""
    pantalla = _tamano_pantalla()
    if pantalla is None:
        return None
    return max(240, pantalla[1] - _ALTO_MARCO_VENTANA)


def _ajustar_ventana(ventana: str, ancho: int, alto: int) -> None:
    """Deja la ventana justo del tamano del lienzo, sin que OpenCV tenga que reescalar."""
    try:
        cv2.resizeWindow(ventana, ancho, alto)
    except cv2.error as error:  # algun backend de GUI puede no soportarlo
        _registro.debug("no se pudo redimensionar la ventana: %s", error)


def _ejecutar_revision(cfg: AppConfig, args: argparse.Namespace) -> int:
    """Modo revision: el operario encuadra, aprieta REVISAR, y recien ahi se juzga.

    A diferencia del modo clasico, aca NO se clasifica cada frame. Con el microscopio la
    imagen solo sirve cuando la pieza esta bien puesta y enfocada; juzgar todo el tiempo
    seria juzgar sobre todo imagenes movidas y a medio encuadrar, y un veredicto que
    parpadea es un veredicto que el operario aprende a ignorar.
    """
    clasificador = crear_clasificador(cfg)
    # Los conceptos mandan sobre la lista de clases del config: son las carpetas que hay de
    # verdad en disco, y el operario las cambia desde el panel sin tocar el YAML.
    conceptos = Conceptos(_DIR_COLECCION, cfg.clases, cfg.clases_defecto)
    _volcar_conceptos(cfg, conceptos)
    almacen = AlmacenClasificacion(_DIR_COLECCION, cfg.clases, cfg.feedback.calidad_jpg)
    teclas_clase = _clases_de_teclas(cfg)
    fuente_es_camara = _es_camara(cfg.camara.fuente)
    galeria = GaleriaReferencias(cfg.interfaz.referencia_aprobada, cfg.feedback.calidad_jpg)
    if galeria.cantidad == 0:
        _registro.warning(
            "no hay fotos de modelo en %s; sacala con la tecla 'f'", galeria.carpeta
        )
    else:
        _registro.info(
            "%d foto(s) de modelo en %s (tecla 'm' para cambiar)",
            galeria.cantidad,
            galeria.carpeta,
        )
    # El bucle es el mismo en los dos modos --encuadrar, REVISAR, congelar, juzgar--; lo
    # unico que cambia es como se dibuja. "simple" es la pantalla del taller.
    # El lienzo se arma para entrar en la pantalla: si se deja que OpenCV lo escale, la
    # fila de botones queda cortada abajo.
    alto_util = _alto_util_pantalla()
    if cfg.interfaz.modo == "simple":

        def componer(vivo, referencia, cfg, estado):  # type: ignore[no-redef]
            return revision.componer_simple(vivo, referencia, cfg, estado, alto_util)

    else:
        componer = revision.componer  # type: ignore[assignment]

    camara = Camara(cfg.camara)
    try:
        camara.abrir()
    except (RuntimeError, ValueError) as error:
        _registro.error("%s", error)
        return 1

    _registro.info(
        "MODO %s | camara: %s", cfg.interfaz.modo.upper(), camara.descripcion
    )
    _registro.info(
        "modelo: %s",
        clasificador.descripcion if clasificador else "SIN MODELO (modo recoleccion)",
    )
    _registro.info("archivando fotos en: %s", ruta_absoluta(_DIR_COLECCION))
    if cfg.interfaz.evaluar_en_vivo:
        _registro.info(
            "evaluacion EN VIVO: nitidez minima %.0f, %d evaluaciones seguidas para "
            "confirmar un veredicto",
            cfg.interfaz.nitidez_minima,
            cfg.interfaz.frames_para_confirmar,
        )
    _registro.info(
        "espacio o clic en REVISAR congela | a/r guardan la pieza | "
        "c abre los conceptos | f agrega foto de modelo | m cambia de modelo | q sale"
    )

    # Las ordenes de afilado. Se retoma la que haya quedado abierta: el taller corta a
    # mitad de una sierra para atender el mostrador, y perder el conteo seria volver a
    # empezar la sierra.
    ordenes = Ordenes(_DIR_ORDENES)
    if ordenes.abierta is not None:
        _registro.info("se retoma la %s", ordenes.abierta.resumen())

    # La memoria: lo que el programa aprendio de las fotos ya archivadas. Se pone al dia
    # antes de empezar, leyendo del disco solo lo que haya cambiado.
    memoria: Optional[Memoria] = None
    if cfg.memoria.habilitada:
        memoria = Memoria(
            cfg.memoria.directorio,
            candidatas_por_concepto=cfg.memoria.candidatas_por_concepto,
            vecinos_por_concepto=cfg.memoria.vecinos_por_concepto,
            maximo_autoevaluacion=cfg.memoria.maximo_autoevaluacion,
        )
        memoria.sincronizar(conceptos.mapa())
        _registro.info("%s", memoria.resumen())

    # Los clics llegan por un callback de OpenCV, que corre fuera del bucle: se encolan y se
    # atienden en el frame siguiente. Con la ventana redimensionable, OpenCV entrega las
    # coordenadas ya llevadas al espacio de la imagen. Igual todo tiene tecla equivalente,
    # asi que si en alguna maquina el mouse fallara, el programa se sigue usando.
    clics: list[tuple[int, int]] = []

    def _al_hacer_clic(evento: int, x: int, y: int, flags: int, param: object) -> None:
        if evento == cv2.EVENT_LBUTTONDOWN:
            clics.append((x, y))

    codigo_salida = 0
    nulos_seguidos = 0
    ventana_ajustada = False
    estado = revision.EstadoRevision()
    # Se cuenta UNA vez al arrancar y despues de cada guardado, no en cada frame: son
    # lecturas de disco y el bucle corre a 17 fps.
    estado.conteos = almacen.conteo_por_clase()

    # El panel de conceptos. None = cerrado; mientras este abierto, la camara no se lee y
    # la pantalla es el panel.
    pan: Optional[panel.EstadoPanel] = None
    miniaturas = panel.Miniaturas()
    alto_lienzo = revision.ALTO_MINIMO_LIENZO

    def refrescar(recalcular: bool = True) -> None:
        """Pone al dia todo lo que depende de que fotos y conceptos hay.

        Se llama cuando algo CAMBIO --se guardo una foto, se creo o borro un concepto--,
        nunca en cada cuadro: son lecturas de disco.
        """
        conceptos.recargar()
        _volcar_conceptos(cfg, conceptos)
        almacen.clases = cfg.clases
        teclas_clase.clear()
        teclas_clase.update(_clases_de_teclas(cfg))
        estado.conteos = almacen.conteo_por_clase()
        estado.botones_guardar = _destinos_rapidos(cfg)
        if memoria is not None:
            memoria.sincronizar(conceptos.mapa(), guardar=not recalcular)
            if recalcular:
                # Medirse cuesta, asi que se hace aca --justo despues de un cambio, con el
                # operario mirando un cartel-- y no en medio de un veredicto.
                memoria.autoevaluacion()
            estado.resumen_memoria = memoria.resumen()
        if pan is not None:
            # La seleccion se sigue por NOMBRE y no por posicion: si se borro un concepto
            # de mas arriba, el indice apuntaria a otro y el siguiente BORRAR se llevaria
            # puesto al que no era.
            anterior = pan.concepto_elegido.nombre if pan.concepto_elegido else None
            pan.conceptos = _lista_para_panel(conceptos, galeria)
            pan.elegido = next(
                (i for i, c in enumerate(pan.conceptos) if c.nombre == anterior), 0
            )
            pan.elegido = max(0, min(pan.elegido, len(pan.conceptos) - 1))
            pan.resumen = estado.resumen_memoria
            auto = memoria.autoevaluacion() if memoria is not None else None
            pan.aciertos = auto.por_concepto if auto is not None else {}
            pan.recortada = bool(auto.recortada) if auto is not None else False

    def archivar(destino: str, imagen: np.ndarray) -> str:
        """Manda la foto al destino elegido y devuelve el mensaje para la pantalla."""
        if destino == _DESTINO_MODELO:
            # La foto de modelo no pasa por el filtro de nitidez --el operario elige a ojo
            # cual quiere de referencia, y una un poco movida puede servirle-- pero SI por
            # el de camara congelada: con el microscopio desenchufado, lo que se guardaria
            # como "modelo" es el cartel de DroidCam, y entonces todo lo que venga despues
            # se compara contra un cartel.
            if getattr(camara, "congelada", False):
                _registro.warning("no se guardo la foto de modelo: la camara esta congelada")
                return "CAMARA CONGELADA - conecte el microscopio"
            ruta = galeria.agregar(imagen)
            galeria.recargar(preferida=ruta.name)
            miniaturas.olvidar()
            _registro.info("nueva foto de modelo en %s", ruta)
            return f"MODELO AGREGADO: {ruta.stem}"
        motivo = motivo_para_no_archivar(camara, imagen, cfg, almacen)
        if motivo is not None:
            _registro.warning("no se archivo la foto: %s", motivo)
            return motivo
        ruta = almacen.guardar(imagen, destino)
        estado.capturas = almacen.total_guardados
        refrescar()
        miniaturas.olvidar()
        _registro.info("foto archivada en '%s': %s", destino, ruta.name)
        mensaje = f"GUARDADA como {destino}: {ruta.name}"

        # Con una orden abierta, archivar un diente tambien lo ANOTA: el conteo avanza y
        # queda en el informe. Lo que vale como estado es lo que dijo el operario --el
        # archivo donde lo puso-- y lo que opinaba el programa queda al lado, para que un
        # reclamo pueda ver las dos cosas.
        orden = ordenes.abierta
        if orden is not None:
            opinion = ""
            if estado.juicio is not None and estado.juicio.decidio:
                opinion = f" (la memoria decia {estado.juicio.concepto})"
            diente = orden.anotar(
                estado=destino,
                parecido=estado.juicio.parecido if estado.juicio is not None else None,
                aprobado=destino not in cfg.clases_defecto,
                origen=f"lo archivo el operario{opinion}",
                foto=ruta.name,
            )
            ordenes.guardar()
            actualizar_orden()
            if diente is None:
                mensaje = "ESTA SIERRA YA ESTA COMPLETA - pase a la siguiente con 'o'"
            else:
                sierra = orden.sierra()
                mensaje = (
                    f"DIENTE {diente.numero} de {sierra.dientes}: {destino}"
                    if sierra is not None else mensaje
                )
        return mensaje

    def actualizar_orden() -> None:
        """Pone en la pantalla en que diente va la orden, si hay una abierta."""
        orden = ordenes.abierta
        estado.orden_resumen = orden.resumen() if orden is not None else ""
        if pan is not None:
            pan.orden = orden

    estado.botones_guardar = _destinos_rapidos(cfg)
    actualizar_orden()
    if memoria is not None:
        estado.resumen_memoria = memoria.resumen()
    # Evaluacion en vivo: se clasifica cada N frames, pero el veredicto que se MUESTRA solo
    # cambia cuando varias evaluaciones seguidas coinciden. Sin esa confirmacion el cartel
    # parpadea entre aprobado y rechazado con cada temblor de la mano, y un cartel que
    # parpadea es un cartel que el operario deja de mirar.
    en_vivo = bool(cfg.interfaz.evaluar_en_vivo)
    cada_n_frames = cfg.detector.clasificar_cada_n_frames
    frames_vistos = 0
    # La imagen congelada se juzga una sola vez, al congelarla; ver mas abajo.
    juzgado = False
    clase_candidata: Optional[str] = None
    repeticiones = 0
    vivo: Optional[np.ndarray] = None
    congelado: Optional[np.ndarray] = None
    expira_mensaje = 0.0
    pantalla: Optional[revision.Pantalla] = None

    try:
        cv2.namedWindow(_VENTANA, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(_VENTANA, _al_hacer_clic)
        while True:
            if not estado.congelado and pan is None:
                frame = camara.leer()
                if frame is None:
                    nulos_seguidos += 1
                    if nulos_seguidos >= _MAX_FRAMES_NULOS:
                        if fuente_es_camara:
                            _registro.error(
                                "la camara dejo de entregar frames (%d fallos seguidos).",
                                nulos_seguidos,
                            )
                            codigo_salida = 2
                        else:
                            _registro.info("el video termino")
                        break
                    continue
                nulos_seguidos = 0
                vivo = frame

                if en_vivo and not estado.congelado:
                    # La nitidez se mide SIEMPRE, con modelo o sin el. Sin modelo el
                    # programa no juzga, pero el operario esta juntando el dataset con
                    # a/r, y una foto movida que entra al dataset es una muestra
                    # envenenada: queda etiquetada como buena o mala cuando en realidad
                    # no se ve nada. El medidor de foco es justo lo que hace falta ahi.
                    estado.nitidez = revision.nitidez(vivo)
                    if estado.nitidez < float(cfg.interfaz.nitidez_minima):
                        # Imagen movida o fuera de foco: no se opina. Y se reinicia la
                        # confirmacion, para que al volver el foco no se arrastre un
                        # veredicto formado con frames que ya no valen.
                        estado.esperando_foco = True
                        estado.veredicto = None
                        clase_candidata, repeticiones = None, 0
                    else:
                        estado.esperando_foco = False
                        if clasificador is not None and frames_vistos % cada_n_frames == 0:
                            nuevo = clasificador.clasificar(vivo)
                            if nuevo is not None:
                                if nuevo.clase == clase_candidata:
                                    repeticiones += 1
                                else:
                                    clase_candidata, repeticiones = nuevo.clase, 1
                                if repeticiones >= int(cfg.interfaz.frames_para_confirmar):
                                    estado.veredicto = nuevo
                    frames_vistos += 1

            if vivo is None:
                continue

            if estado.mensaje is not None and time.monotonic() >= expira_mensaje:
                estado.mensaje = None

            a_mostrar = congelado if estado.congelado and congelado is not None else vivo
            referencia = galeria.imagen_actual()
            # Quien opina cuando no hay modelo entrenado: primero la MEMORIA, que compara
            # contra todas las fotos archivadas; si todavia no le alcanza, el parecido
            # contra la unica foto de modelo que este elegida. Las dos cuestan unos ms y se
            # respeta cada_n_frames: no hay razon para gastarlo en cada cuadro.
            #
            # LA IMAGEN CONGELADA SE JUZGA UNA SOLA VEZ, y esto no es una optimizacion: es
            # correctitud. Antes se volvia a juzgar en cada cuadro, asi que en cuanto el
            # operario apretaba GUARDAR, esa misma foto entraba a la memoria y el cuadro
            # siguiente la encontraba ahi: el cartel pasaba a decir 100% del concepto que
            # el operario acababa de elegir. El programa se daba la razon a si mismo.
            if clasificador is None and not estado.esperando_foco and pan is None:
                if (estado.congelado and not juzgado) or (
                    not estado.congelado and frames_vistos % cada_n_frames == 0
                ):
                    juzgado = estado.congelado
                    estado.juicio = None
                    if memoria is not None and memoria.lista_para_juzgar(
                        cfg.memoria.minimo_por_concepto
                    ):
                        estado.juicio = memoria.juzgar(
                            a_mostrar,
                            parecido_minimo=cfg.memoria.parecido_minimo,
                            confianza_minima=cfg.memoria.confianza_minima,
                            minimo_por_concepto=cfg.memoria.minimo_por_concepto,
                        )
                    estado.parecido = revision.parecido(a_mostrar, referencia)
            estado.referencia_faltante = referencia is None
            estado.nombre_referencia = galeria.nombre_actual
            estado.posicion_referencia = galeria.posicion

            if pan is not None:
                pantalla = panel.componer(pan, alto_lienzo, miniaturas)
            else:
                pantalla = componer(a_mostrar, referencia, cfg, estado)
                alto_lienzo = pantalla.imagen.shape[0]
            cv2.imshow(_VENTANA, pantalla.imagen)
            if not ventana_ajustada:
                # Recien aca se sabe cuanto mide el lienzo: depende de la resolucion que
                # haya negociado la camara.
                _ajustar_ventana(_VENTANA, pantalla.imagen.shape[1], pantalla.imagen.shape[0])
                ventana_ajustada = True

            tecla = cv2.waitKey(30) & 0xFF
            accion: Optional[str] = None
            while clics:
                x, y = clics.pop(0)
                pulsado = pantalla.boton_en(x, y)
                if pulsado is not None:
                    accion = pulsado

            # Con el panel abierto las teclas son suyas: adentro se escriben nombres de
            # conceptos, y ahi la 'q' es una letra, no la orden de salir del programa.
            if pan is not None:
                sigue, mensaje = _atender_panel(
                    accion, tecla, pan, conceptos, galeria, miniaturas,
                    lambda destino: archivar(destino, pan.pendiente
                                             if pan.pendiente is not None else a_mostrar),
                    refrescar,
                    ordenes,
                )
                actualizar_orden()
                if mensaje is not None:
                    estado.mensaje = mensaje
                    expira_mensaje = time.monotonic() + _SEGUNDOS_MENSAJE
                    if sigue:
                        # El panel no dibuja estado.mensaje: si se queda abierto --borrar
                        # una foto, crear un concepto-- la unica forma de que el operario
                        # vea que paso es el aviso del propio panel.
                        pan.aviso = mensaje
                if not sigue:
                    pan = None
                    miniaturas.olvidar()
                continue

            if tecla in (ord("q"), _ESC):
                accion = "salir"
            elif tecla in (ord(" "), 13):
                accion = "seguir" if estado.congelado else "revisar"
            elif tecla in teclas_clase:
                accion = teclas_clase[tecla]
            elif tecla == ord("f"):
                accion = "referencia"
            elif tecla == ord("m"):
                accion = "modelo"
            elif tecla == ord("c"):
                accion = "conceptos"
            elif tecla == ord("o"):
                accion = "orden"

            if accion is None:
                continue
            if accion == "salir":
                break
            if accion in ("conceptos", "orden"):
                pan = panel.EstadoPanel(
                    conceptos=_lista_para_panel(conceptos, galeria),
                    pendiente=a_mostrar.copy(),
                    resumen=estado.resumen_memoria,
                    aciertos=(
                        memoria.autoevaluacion().por_concepto if memoria is not None else {}
                    ),
                    recortada=(
                        memoria.autoevaluacion().recortada if memoria is not None else False
                    ),
                    orden=ordenes.abierta,
                )
                if accion == "orden":
                    pan.modo = panel.MODO_ORDEN
                    pan.cargando_orden = ordenes.abierta is None
                continue

            if accion == "revisar":
                congelado = vivo.copy()
                estado.congelado = True
                juzgado = False
                estado.esperando_foco = False
                # En vivo ya hay un veredicto confirmado para esta imagen: se congela ese,
                # en vez de volver a clasificar y arriesgar que salga otro distinto justo
                # al apretar el boton.
                if not (en_vivo and estado.veredicto is not None):
                    estado.veredicto = (
                        clasificador.clasificar(congelado) if clasificador else None
                    )
            elif accion == "seguir":
                estado.congelado = False
                juzgado = False
                estado.veredicto = None
                estado.nitidez = None
                estado.esperando_foco = False
                clase_candidata, repeticiones = None, 0
                congelado = None
            elif accion == "referencia":
                base = congelado if estado.congelado and congelado is not None else vivo
                try:
                    estado.mensaje = archivar(_DESTINO_MODELO, base)
                except OSError as error:
                    _registro.error("no se pudo guardar la foto de modelo: %s", error)
                    estado.mensaje = "ERROR AL GUARDAR - revise el disco"
                expira_mensaje = time.monotonic() + _SEGUNDOS_MENSAJE
            elif accion == "modelo":
                # Se relee la carpeta antes de pasar al siguiente: asi aparecen las fotos
                # que alguien haya copiado ahi con el programa abierto, sin reiniciar.
                galeria.recargar()
                nombre = galeria.siguiente()
                estado.mensaje = (
                    f"MODELO: {nombre}" if nombre else "NO HAY FOTOS DE MODELO"
                )
                expira_mensaje = time.monotonic() + _SEGUNDOS_MENSAJE
            elif accion in cfg.clases:
                base = congelado if estado.congelado and congelado is not None else vivo
                try:
                    estado.mensaje = archivar(accion, base)
                except (OSError, ValueError) as error:
                    _registro.error("no se pudo guardar: %s", error)
                    estado.mensaje = "ERROR AL GUARDAR - revise el disco"
                expira_mensaje = time.monotonic() + _SEGUNDOS_MENSAJE
    except KeyboardInterrupt:
        _registro.info("interrumpido por el operario")
    finally:
        camara.cerrar()
        cv2.destroyAllWindows()
        cv2.waitKey(1)

    _registro.info("fotos archivadas en esta sesion: %d", almacen.total_guardados)
    return codigo_salida


def motivo_para_no_archivar(camara, frame, cfg: AppConfig, almacen=None) -> Optional[str]:
    """Por que esta foto NO deberia entrar al dataset. None si puede entrar.

    Una foto mal etiquetada no se nota. Entra al dataset, se entrena con ella y el modelo
    sale raro, sin que nadie pueda decir por que. Frenarla al guardar es mucho mas barato
    que descubrirla meses despues entre cientos.

    Los dos casos que ya pasaron de verdad en este proyecto:

    1. CAMARA CONGELADA. Con el microscopio desenchufado el programa cae en DroidCam, que
       muestra un cartel fijo. Alguien archivo ese cartel como sierra APROBADA. Ojo: eso NO
       lo atrapa un control de foco --el cartel es texto blanco nitido sobre negro y mide
       nitidez 642, la mas alta de todo el dataset-- asi que hace falta preguntarle a la
       camara, no a la imagen.

    2. IMAGEN MOVIDA. El mismo diente enfocado mide 101 y con un desenfoque de 5x5 cae a
       3.7. Una foto asi queda etiquetada buena o mala sin que se vea nada.

    3. LA MISMA FOTO OTRA VEZ. Con la imagen congelada, cada pulsacion de GUARDAR escribia
       de nuevo el MISMO frame. Quedaron 50 archivos por clase que en realidad eran 2 fotos,
       y una estaba archivada como buena Y como mala, byte por byte el mismo archivo.
    """
    if getattr(camara, "congelada", False):
        return "CAMARA CONGELADA - conecte el microscopio"
    medida = revision.nitidez(frame)
    if medida < float(cfg.interfaz.nitidez_minima):
        return f"IMAGEN MOVIDA (nitidez {medida:.0f}) - enfoque y repita"
    if almacen is not None:
        anterior = almacen.ya_guardada(frame)
        if anterior is not None:
            return f"ESTA FOTO YA LA GUARDASTE como {anterior} - mueva la pieza"
    return None


def configurar_registro() -> Optional[Path]:
    """Manda el log a un archivo al lado del programa, y a la consola si es que hay.

    Abierto con doble clic no hay consola ninguna: sys.stderr es None y un StreamHandler
    revienta al primer mensaje. Pero el log es lo UNICO que queda cuando la camara falla en
    el taller --asi se diagnosticaron todos los problemas hasta ahora-- asi que no puede
    depender de que alguien lo haya arrancado desde una terminal.

    Devuelve la ruta del archivo, o None si no se pudo escribir (disco lleno, carpeta de
    solo lectura). Que falle no frena el programa: se sigue sin registro.
    """
    formato = logging.Formatter(
        "%(asctime)s %(levelname)s [%(name)s] %(message)s", "%Y-%m-%d %H:%M:%S"
    )
    raiz = logging.getLogger()
    raiz.setLevel(logging.INFO)

    archivo: Optional[Path] = None
    try:
        archivo = raiz_repo() / _ARCHIVO_REGISTRO
        a_disco = logging.handlers.RotatingFileHandler(
            archivo,
            maxBytes=_BYTES_REGISTRO,
            backupCount=_VUELTAS_REGISTRO,
            encoding="utf-8",
        )
        a_disco.setFormatter(formato)
        raiz.addHandler(a_disco)
    except OSError:
        archivo = None

    # Solo si hay consola de verdad. Con --windowed, PyInstaller deja stderr en None.
    if sys.stderr is not None:
        a_pantalla = logging.StreamHandler()
        a_pantalla.setFormatter(formato)
        raiz.addHandler(a_pantalla)

    return archivo


def main(argv: Optional[list[str]] = None) -> int:
    """Punto de entrada del bucle en vivo. Devuelve 0 en exito."""
    archivo_registro = configurar_registro()
    if archivo_registro is not None:
        _registro.info("registro de esta sesion: %s", archivo_registro)
    args = _construir_parser().parse_args(argv)

    if args.listar_camaras:
        return _listar_camaras()

    try:
        cfg = cargar_config(args.config)
        _aplicar_argumentos(cfg, args)
    except (ValueError, OSError) as error:
        _registro.error("configuracion invalida: %s", error)
        return 1

    # Modo clasificacion: veredicto de la sierra entera (aprobada/rechazada), sin cajas ni
    # medidas. Es un camino aparte que no toca el pipeline de deteccion.
    if cfg.detector.tarea == "classify":
        if cfg.interfaz.modo in ("revision", "simple"):
            return _ejecutar_revision(cfg, args)
        return _ejecutar_clasificacion(cfg, args)

    presets_roi = _presets_con_config(cfg)
    indice_roi = 0

    try:
        pipeline = Pipeline(cfg)
    except (ValueError, OSError) as error:
        _registro.error("no se pudo preparar el pipeline: %s", error)
        return 1

    almacen = AlmacenFeedback(cfg.feedback, cfg.clases)
    fuente_es_camara = _es_camara(cfg.camara.fuente)
    ruta_grabar = ruta_absoluta(args.grabar) if args.grabar else None
    grabador: Optional[cv2.VideoWriter] = None

    camara = Camara(cfg.camara)
    try:
        camara.abrir()
    except (RuntimeError, ValueError) as error:
        _registro.error("%s", error)
        return 1

    _registro.info("camara: %s", camara.descripcion)
    _registro.info("modelo: %s", pipeline.descripcion_modelo)
    _registro.info(
        "feedback: %s en %s",
        "activo" if cfg.feedback.habilitado else "DESACTIVADO",
        ruta_absoluta(cfg.feedback.directorio),
    )
    _registro.info("teclas: q/ESC salir | e error de la IA | g ejemplo bueno | espacio capturar")

    codigo_salida = 0
    nulos_seguidos = 0
    escala_registrada = False
    mostrar_ayuda = True
    en_pausa = False
    mensaje: Optional[str] = None
    expira_mensaje = 0.0
    limpio: Optional[np.ndarray] = None
    resultado: Optional[ResultadoFrame] = None

    try:
        cv2.namedWindow(_VENTANA, cv2.WINDOW_NORMAL)
        while True:
            if not en_pausa:
                frame = camara.leer()
                if frame is None:
                    nulos_seguidos += 1
                    if nulos_seguidos >= _MAX_FRAMES_NULOS:
                        if fuente_es_camara:
                            _registro.error(
                                "la camara dejo de entregar frames (%d fallos seguidos). "
                                "Revise el cable USB, que ninguna otra aplicacion la tenga "
                                "tomada y que siga alimentada.",
                                nulos_seguidos,
                            )
                            codigo_salida = 2
                        else:
                            _registro.info("el video termino")
                        break
                    continue
                nulos_seguidos = 0

                # limpio es la memoria intacta: no se dibuja nunca sobre el.
                limpio = frame
                resultado = pipeline.procesar(limpio)

                if not escala_registrada:
                    _registro.info("escala inicial: %s", _describir_escala(resultado))
                    escala_registrada = True

            if limpio is None or resultado is None:
                continue

            if mensaje is not None and time.monotonic() >= expira_mensaje:
                mensaje = None

            # La pausa se reafirma en cada frame: sin esto el aviso caducaria a los
            # 1.5s y el operario se quedaria mirando una imagen congelada sin saber
            # por que. Una confirmacion de guardado reciente tiene prioridad.
            texto_mensaje = mensaje
            if en_pausa and texto_mensaje is None:
                texto_mensaje = "PAUSA - p para reanudar"

            # overlay lee el contador por getattr: ResultadoFrame es contrato cerrado
            # y no puede declarar el campo, pero el HUD debe mostrarlo.
            setattr(resultado, "capturas_guardadas", almacen.total_guardados)
            anotado = overlay.dibujar(
                limpio, resultado, cfg, mostrar_ayuda=mostrar_ayuda, mensaje=texto_mensaje
            )

            if ruta_grabar is not None and not en_pausa:
                if grabador is None:
                    grabador = _abrir_grabador(ruta_grabar, anotado, cfg.camara.fps)
                    if grabador is None:
                        ruta_grabar = None
                if grabador is not None:
                    grabador.write(anotado)

            cv2.imshow(_VENTANA, anotado)
            tecla = cv2.waitKey(1) & 0xFF

            if tecla in (ord("q"), _ESC):
                break
            if tecla == 255:
                continue

            if tecla in (ord("e"), ord("g"), ord(" ")):
                if not cfg.feedback.habilitado:
                    mensaje = "FEEDBACK DESACTIVADO en el config"
                else:
                    etiqueta = {ord("e"): "fallo", ord("g"): "bueno", ord(" "): "captura"}[
                        tecla
                    ]
                    mensaje = _guardar(almacen, limpio, resultado, anotado, etiqueta)
                expira_mensaje = time.monotonic() + _SEGUNDOS_MENSAJE
            elif tecla == ord("r"):
                pipeline.reiniciar_calibracion()
                mensaje = "CALIBRACION REINICIADA - muestre el marcador"
                expira_mensaje = time.monotonic() + _SEGUNDOS_MENSAJE
            elif tecla == ord("p"):
                en_pausa = not en_pausa
                mensaje = None
            elif tecla == ord("h"):
                mostrar_ayuda = not mostrar_ayuda
            elif tecla == ord("c"):
                indice_roi = (indice_roi + 1) % len(presets_roi)
                nombre = _aplicar_preset_roi(cfg, presets_roi[indice_roi])
                mensaje = f"ROI: {nombre}"
                expira_mensaje = time.monotonic() + _SEGUNDOS_MENSAJE
    except KeyboardInterrupt:
        _registro.info("interrumpido por el operario")
    finally:
        if grabador is not None:
            grabador.release()
            _registro.info("video anotado guardado en %s", ruta_grabar)
        camara.cerrar()
        cv2.destroyAllWindows()
        # En algunos backends la ventana solo se destruye si el bucle de eventos
        # de GUI corre una vez mas despues del destroy.
        cv2.waitKey(1)

    _registro.info("capturas guardadas en esta sesion: %d", almacen.total_guardados)
    return codigo_salida


if __name__ == "__main__":
    raise SystemExit(main())
