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
import sys
import time
from pathlib import Path
from typing import Optional, Union

import cv2
import numpy as np

from .. import overlay, revision
from ..referencias import GaleriaReferencias
from ..almacen import AlmacenClasificacion, AlmacenFeedback
from ..camara import Camara, enumerar_camaras
from ..config import AppConfig, cargar_config, ruta_absoluta
from ..detector import crear_clasificador
from ..pipeline import Pipeline
from ..tipos import ResultadoFrame, Veredicto

# Carpeta donde el modo clasificacion archiva las fotos por clase. Su estructura es
# exactamente la que consume scripts/construir_clasificacion.py: <dir>/<clase>/*.jpg
_DIR_COLECCION = "data/coleccion"

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
    """
    sanas = [c for c in cfg.clases if c not in cfg.clases_defecto]
    clase_sana = sanas[0] if sanas else cfg.clases[0]
    clase_defecto = cfg.clases_defecto[0] if cfg.clases_defecto else cfg.clases[-1]
    return {ord("a"): clase_sana, ord("r"): clase_defecto}


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
    _registro.info(
        "espacio o clic en REVISAR analiza | a/r guardan la pieza | "
        "f agrega foto de modelo | m cambia de modelo | q sale"
    )

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
    vivo: Optional[np.ndarray] = None
    congelado: Optional[np.ndarray] = None
    expira_mensaje = 0.0
    pantalla: Optional[revision.Pantalla] = None

    try:
        cv2.namedWindow(_VENTANA, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(_VENTANA, _al_hacer_clic)
        while True:
            if not estado.congelado:
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

            if vivo is None:
                continue

            if estado.mensaje is not None and time.monotonic() >= expira_mensaje:
                estado.mensaje = None

            a_mostrar = congelado if estado.congelado and congelado is not None else vivo
            referencia = galeria.imagen_actual()
            estado.referencia_faltante = referencia is None
            estado.nombre_referencia = galeria.nombre_actual
            estado.posicion_referencia = galeria.posicion
            pantalla = componer(a_mostrar, referencia, cfg, estado)
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

            if accion is None:
                continue
            if accion == "salir":
                break

            if accion == "revisar":
                congelado = vivo.copy()
                estado.congelado = True
                estado.veredicto = (
                    clasificador.clasificar(congelado) if clasificador else None
                )
            elif accion == "seguir":
                estado.congelado = False
                estado.veredicto = None
                congelado = None
            elif accion == "referencia":
                base = congelado if estado.congelado and congelado is not None else vivo
                try:
                    ruta = galeria.agregar(base)
                    estado.mensaje = f"MODELO AGREGADO: {ruta.stem}"
                    _registro.info(
                        "nueva foto de modelo en %s (renombrala con el modelo de sierra "
                        "para reconocerla despues)",
                        ruta,
                    )
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
                    ruta = almacen.guardar(base, accion)
                    estado.capturas = almacen.total_guardados
                    estado.mensaje = f"GUARDADA como {accion}: {ruta.name}"
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


def main(argv: Optional[list[str]] = None) -> int:
    """Punto de entrada del bucle en vivo. Devuelve 0 en exito."""
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s [%(name)s] %(message)s"
    )
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
