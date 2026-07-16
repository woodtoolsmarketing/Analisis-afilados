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
import time
from pathlib import Path
from typing import Optional, Union

import cv2
import numpy as np

from .. import overlay
from ..almacen import AlmacenFeedback
from ..camara import Camara
from ..config import AppConfig, cargar_config, ruta_absoluta
from ..pipeline import Pipeline
from ..tipos import ResultadoFrame

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
        help="indice de webcam (0, 1, ...), ruta de video o URL de stream",
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


def main(argv: Optional[list[str]] = None) -> int:
    """Punto de entrada del bucle en vivo. Devuelve 0 en exito."""
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s [%(name)s] %(message)s"
    )
    args = _construir_parser().parse_args(argv)

    try:
        cfg = cargar_config(args.config)
        _aplicar_argumentos(cfg, args)
    except (ValueError, OSError) as error:
        _registro.error("configuracion invalida: %s", error)
        return 1

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
