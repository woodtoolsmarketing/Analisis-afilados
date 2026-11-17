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

from .. import dientes, overlay, panel, revision
from ..automatico import CapturaAutomatica, Resultado
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

# Cuantos dientes se buscan en una imagen. Lo normal es uno; dos pasa seguido y tres a
# veces. Mas que esto, en una foto de microscopio, es que algo se entendio mal: antes de
# analizar seis pedazos conviene que el operario acerque y mire de a uno.
_MAX_DIENTES = 4

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
# LO QUE SE LE DICE AL OPERARIO CUANDO LA ORDEN NO LLEGO AL DISCO. Antes no se le decia
# nada: ordenes.guardar() devolvia None, nadie lo miraba, y el operario seguia anotando
# dientes toda la tarde contra una orden que no existia en ningun archivo. Reproducido con
# el disco lleno: el programa salia con codigo 0, cuatro guardados fallidos, y en disco no
# habia orden. Con el .exe sin consola, el log en un archivo no lo lee nadie.
_NO_SE_GUARDO = "NO SE PUDO GUARDAR LA ORDEN - revise el disco"

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


def _cuanto_pesa(archivo: Path) -> str:
    """Lo que ocupa un archivo, para decirlo en el cartel de la pantalla.

    En KiB hasta el mega y en MiB de ahi para arriba, con una coma de decimal: lo que el
    operario necesita saber es si lo que acaba de escribir son dos mil bytes o treinta
    megas, no el byte exacto. Son las mismas unidades con las que esta escrito el resto del
    programa --el Explorador de Windows muestra lo mismo-- asi que los numeros se pueden
    comparar con lo que se ve en la carpeta.

    MEDIDO exportando las 4 ordenes de verdad del taller (copiadas al costado: en
    Desktop/Afilado no se escribio nada) y el caso gordo armado con una foto de verdad del
    microscopio de 88.051 bytes:

        213132  (0 de 96 dientes)      2.706 bytes ->  "2,6 KiB"
        123     (7 de 80)              2.913 bytes ->  "2,8 KiB"
        1234    (7 de 80)              2.918 bytes ->  "2,8 KiB"
        5113546 (15 de 80)             3.275 bytes ->  "3,2 KiB"
        4 sierras de 80 con 320 pruebas   28.178.007 bytes -> "26,9 MiB"   (847 ms)
        la misma, sin las fotos en el disco     7.153 bytes ->  "7,0 KiB"  (7 ms)

    O sea: lo que pesa es la prueba, y entre el zip mas liviano y el mas pesado hay diez mil
    veces. Diez 'e' de la orden gorda en un dia son 269 MiB que nadie borra.

    Si el archivo no se puede medir se dice que no se pudo, no se pone un cero: un cero ahi
    se lee como "el zip salio vacio", que es otra cosa.
    """
    try:
        bytes_ = archivo.stat().st_size
    except OSError:
        return "no se pudo medir cuanto pesa"
    if bytes_ >= 1024 * 1024:
        return f"{bytes_ / (1024 * 1024):.1f} MiB".replace(".", ",")
    return f"{bytes_ / 1024:.1f} KiB".replace(".", ",")


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


def _analizar_dientes(
    frame, memoria, cfg: AppConfig, referencia, ejemplos: dict, referencia_entera=None,
    todas_las_modelo=None,
) -> list[revision.DienteEnPantalla]:
    """Separa los dientes de la imagen y juzga CADA UNO por su cuenta.

    El porcentaje de cada diente sale de la misma fuente que el de la pantalla entera
    --primero la memoria, si no el parecido contra la foto de modelo-- pero medido sobre
    el recorte de ese diente y no sobre la foto completa. Con dos dientes adentro, juzgar
    la foto entera daba un numero que no era de ninguno de los dos.

    CON interfaz.recuadros_de_diente EN FALSE --como esta de fabrica-- no se separa nada:
    la pieza es LA FOTO. Ver el comentario del config para el por que, que es que el
    detector busca lo mas claro y con la luz a contraluz de hoy el diente es lo mas
    oscuro, asi que marcaba el cuadro entero.
    """
    if not getattr(cfg.interfaz, "recuadros_de_diente", False):
        alto, ancho = frame.shape[:2]
        encontrados = [dientes.Diente(caja=(0, 0, ancho, alto), area=1.0, entero=True)]
    else:
        encontrados = dientes.detectar(frame, maximo=_MAX_DIENTES)
    if not encontrados:
        return []
    # CON UN SOLO DIENTE A LA VISTA, LA PIEZA ES LA FOTO. Ahi se compara cuadro contra
    # cuadro, que es lo que el taller viene mirando y sobre lo que esta puesto el umbral
    # del 70%. Con VARIOS no hay un numero unico posible, asi que cada uno se compara
    # recortado contra el diente de la foto de modelo. Son dos escalas distintas --medido,
    # el mismo par da 42% cuadro contra cuadro y 45% recorte contra recorte-- pero nunca
    # se muestran juntas: con varios dientes el cartel pasa a ser el resumen.
    uno_solo = len(encontrados) == 1
    # CUAL SE MIRA PRIMERO. No el de la izquierda: el mas parecido a una cara de widia.
    # En las fotos del taller el HUECO entre dos dientes pasa el filtro de area igual que
    # la cara, y cuando cae a la izquierda quedaba primero --o sea que era el que se
    # archivaba y el que iba al informe-- con el diente de verdad al lado, sin elegir.
    # Pasa en 1 de las 14 fotos de cerca. Ver dientes._medir_rayado para los numeros.
    por_defecto = dientes.mas_parecido_a_diente(encontrados)

    listos = memoria is not None and memoria.lista_para_juzgar(
        cfg.memoria.minimo_por_concepto
    )
    # TODAVIA NO PUEDE JUZGAR PERO YA TIENE FOTOS. Es lo que pasa el primer dia: todas las
    # fotos en un solo concepto, asi que no hay contra que comparar y un veredicto es
    # imposible. Pero esas fotos dicen muchisimo mas que la unica foto de modelo elegida,
    # asi que el parecido se mide contra TODAS ellas y se muestra el de la que mas se
    # parece. Medido sobre las fotos del taller, contra una foto suelta el numero se mueve
    # entre 42% y 73% segun como se haya apoyado la pieza.
    con_memoria = (not listos) and memoria is not None and memoria.cantidad > 0
    en_pantalla: list[revision.DienteEnPantalla] = []
    for indice, encontrado in enumerate(encontrados):
        recorte = encontrado.recortar(frame)
        parecido = aprobado = concepto = ejemplo = modelo = None
        juicio = None
        if listos:
            juicio = memoria.juzgar(
                recorte,
                parecido_minimo=cfg.memoria.parecido_minimo,
                confianza_minima=cfg.memoria.confianza_minima,
                minimo_por_concepto=cfg.memoria.minimo_por_concepto,
            )
            parecido = juicio.parecido or None
            if juicio.decidio:
                concepto = juicio.concepto
                aprobado = concepto not in cfg.clases_defecto
                ejemplo = _foto_de_ejemplo(juicio, ejemplos)
        elif con_memoria:
            vecino = memoria.mejor_parecido(recorte if not uno_solo else frame)
            if vecino is not None:
                parecido = vecino.parecido
                concepto = vecino.concepto
                # El veredicto sale de un PARECIDO, no de la memoria juzgando: si la
                # llave que apaga eso esta en false, aca tampoco se pinta. Antes esta rama
                # no la miraba, asi que poner veredicto_por_parecido: false apagaba el
                # cartel pero dejaba el recuadro verde y la orden anotando APROBADO.
                if getattr(cfg.interfaz, "veredicto_por_parecido", False):
                    aprobado = (
                        parecido >= float(cfg.interfaz.umbral_parecido)
                        and concepto not in cfg.clases_defecto
                    )
                ejemplo = _foto_por_ruta(vecino.archivo, ejemplos)
        else:
            # CONTRA EL DIENTE DE LA FOTO DE MODELO, no contra la foto entera. Comparar un
            # recorte de diente contra un cuadro completo es comparar dos cosas distintas:
            # medido sobre las fotos del taller, el mismo par daba 42% mirando cuadro
            # contra cuadro, 45% recorte contra recorte y 25% recorte contra cuadro. Ese
            # 25% es el que aparecia en rojo sobre el diente mientras el cartel decia
            # APROBADO en verde, con los dos numeros sacados de la misma pieza.
            mirar = frame if uno_solo else recorte
            if uno_solo and todas_las_modelo:
                # Contra TODAS las fotos de modelo, quedandose con la mejor. Y SE GUARDA
                # CUAL GANO, porque la cabecera nombra la que eligio el operario con 'm' y
                # casi nunca es la misma: medido con una foto del taller y los tres
                # modelos reales, el cartel decia APROBADO 80% --que sale de
                # modelo_20261112_161523-- mientras contra la foto nombrada arriba
                # (modelo_20261112_161319) el parecido era 62%, o sea DESAPROBADO.
                medidos = []
                for nombre_modelo, foto_modelo in todas_las_modelo:
                    valor = revision.parecido(mirar, foto_modelo)
                    if valor is not None:
                        medidos.append((valor, nombre_modelo))
                if medidos:
                    parecido, modelo = max(medidos)
            else:
                contra = (
                    referencia_entera if uno_solo and referencia_entera is not None
                    else referencia
                )
                parecido = revision.parecido(mirar, contra)
            if parecido is not None and getattr(cfg.interfaz, "veredicto_por_parecido", False):
                aprobado = parecido >= float(cfg.interfaz.umbral_parecido)
        en_pantalla.append(
            revision.DienteEnPantalla(
                caja=encontrado.caja, parecido=parecido, aprobado=aprobado,
                concepto=concepto, ejemplo=ejemplo, elegido=indice == por_defecto,
                juicio=juicio, modelo=modelo, entero=encontrado.entero,
            )
        )
    return en_pantalla


def _foto_por_ruta(ruta: str, ejemplos: dict):
    """Una foto guardada, leida una sola vez."""
    if ruta not in ejemplos:
        if len(ejemplos) > 24:
            ejemplos.clear()
        ejemplos[ruta] = cv2.imread(ruta)
    return ejemplos[ruta]


def _foto_de_ejemplo(juicio, ejemplos: dict):
    """La foto guardada a la que se parecio este diente. Se lee una sola vez.

    Es lo que contesta "como es un diente bien afilado" sin que el operario tenga que
    ir a buscar la carpeta: al lado del diente que esta mirando queda el que la memoria
    tomo de ejemplo.
    """
    propio = next((v for v in juicio.vecinos if v.concepto == juicio.concepto), None)
    if propio is None:
        return None
    if propio.archivo not in ejemplos:
        if len(ejemplos) > 24:
            ejemplos.clear()
        ejemplos[propio.archivo] = cv2.imread(propio.archivo)
    return ejemplos[propio.archivo]


def _barra_de_la_sierra(orden) -> list[str]:
    """Un cuadrito por diente de la sierra que se esta revisando, para la pantalla."""
    if orden is None:
        return []
    sierra = orden.sierra()
    if sierra is None:
        return []
    barra = []
    for diente in sierra.revisados:
        if not diente.revisado:
            barra.append("falta")
        elif diente.aprobado is True:
            barra.append("bien")
        elif diente.aprobado is False:
            barra.append("mal")
        else:
            barra.append("duda")
    return barra


def _resultado_del_diente(
    estado: revision.EstadoRevision, cfg: AppConfig, quien: str = "anotado solo"
) -> Optional[Resultado]:
    """El veredicto que se va a anotar, o None si no hay.

    Se usa el diente ELEGIDO, que con un solo diente a la vista --lo normal-- es el unico
    que hay. Antes se usaba siempre `dientes[0]`: con dos dientes en la imagen, el operario
    apretaba '2' para que la pantalla marcara el de la derecha, apretaba 'g', y el informe
    anotaba el de la IZQUIERDA. El recuadro marcado y el renglon del informe hablaban de
    dientes distintos, y eso recien se ve cuando llega el reclamo.
    """
    if not estado.dientes:
        return None
    diente = next((d for d in estado.dientes if d.elegido), estado.dientes[0])
    if diente.aprobado is None and diente.concepto is None:
        return None
    # DE DONDE SALIO EL NUMERO, dicho como fue. El informe es el papel con el que se
    # contesta un reclamo, y antes decia "comparando con la foto de modelo" en los tres
    # casos: tambien cuando el numero lo habia puesto la memoria contra las fotos
    # archivadas, que es una comparacion distinta y contra otras fotos.
    if diente.juicio is not None and diente.juicio.decidio:
        origen = "la memoria"
    elif diente.concepto is not None:
        origen = "la foto archivada mas parecida"
        # EL ESTADO Y EL RESULTADO DE ESTE RENGLON SALEN DE DOS CRITERIOS DISTINTOS, y en
        # el papel con el que se contesta un reclamo hay que decirlo: el estado es el
        # concepto de la foto archivada que mas se parecio, y el APROBADO/RECHAZADO sale de
        # si ese parecido llega al umbral. Reproducido en el bucle con el taller de hoy
        # --14 fotos en 'afilado', un solo concepto--: el renglon salia "afilado RECHAZADO
        # 32%" sin nada que lo explicara, y la leyenda del propio informe dice que 'estado'
        # es en que quedo cada diente.
        if diente.aprobado is False and diente.concepto not in cfg.clases_defecto:
            origen += (
                f", que es de {diente.concepto}, pero al "
                f"{(diente.parecido or 0.0) * 100:.0f}%: hace falta "
                f"{float(cfg.interfaz.umbral_parecido) * 100:.0f}% para aprobar"
            )
    else:
        origen = "comparando con las fotos de modelo"
    return Resultado(
        estado=diente.concepto,
        parecido=diente.parecido,
        aprobado=diente.aprobado,
        origen=f"{origen}, {quien}",
    )


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
    if pan.modo == panel.MODO_ORDENES:
        return _panel_ordenes(accion, tecla, pan, ordenes)
    if pan.modo == panel.MODO_BORRAR_ORDEN:
        return _panel_borrar_orden(accion, tecla, pan, ordenes)
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
        # SE VUELVE AL HISTORIAL, no a los conceptos. La jerarquia es taller ->
        # ORDENES -> esta orden: subir un piso desde una orden es la lista de ordenes, y
        # caer en la lista de conceptos --que no tiene nada que ver con lo que se estaba
        # haciendo-- dejaba al operario sin forma de volver a la orden de al lado.
        # El historial se vuelve a leer porque adentro pudo abrirse, retomarse o cerrarse
        # una orden, y la lista de antes mostraria el avance viejo.
        _abrir_historial(pan, ordenes)
        return True, None

    if cargando:
        return _cargar_orden(accion, tecla, pan, ordenes)

    orden = pan.orden
    if accion == "nueva_orden" or tecla == ord("n"):
        pan.cargando_orden = True
        pan.valores = ["" for _ in panel.CAMPOS_ORDEN]
        # El ofrecimiento de retomar se apaga con el formulario nuevo, igual que cuando se
        # entra desde el historial: si quedara puesto, el panel dibujaria el boton RETOMAR LA
        # GUARDADA sobre un formulario vacio, y un boton que no corresponde es lo mismo que
        # un boton que no hace nada.
        pan.campo, pan.aviso, pan.retomar = 0, None, ""
    elif accion == "siguiente_sierra" or tecla == ord("s"):
        posicion = orden.siguiente_sierra()
        if ordenes.guardar() is None:
            return True, _NO_SE_GUARDO
        # Se nombra el NUMERO de la sierra, no la posicion: es el que sale en el informe.
        # Para una orden cargada a mano son el mismo.
        sierra = orden.sierra()
        return True, f"SIERRA {posicion if sierra is None else sierra.numero} de {len(orden.sierras)}"
    elif accion == "saltear_diente" or tecla == ord("x"):
        diente = orden.saltear()
        if ordenes.guardar() is None:
            return True, _NO_SE_GUARDO
        if diente is None:
            pan.aviso = "esta sierra ya tiene todos los dientes revisados"
            return True, None
        return True, f"DIENTE {diente.numero} SALTEADO"
    elif accion == "informe" or tecla == ord("i"):
        texto, planilla = ordenes.escribir_informe()
        if texto is None and planilla is None:
            pan.aviso = "no se pudo escribir el informe; revise el disco"
            return True, None
        if texto is None:
            # QUE MITAD SALIO SE DICE. El .txt es el papel que se entrega y el .csv el que
            # se abre en Excel; ahora se escriben por separado, asi que puede salir uno y
            # no el otro. Callarlo deja al operario creyendo que el informe esta completo,
            # con el anterior todavia en disco para confundirlo.
            pan.aviso = "el .txt del informe no salio; revise el disco"
            return True, f"PLANILLA EN {planilla.name}"
        if planilla is None:
            pan.aviso = "la planilla .csv no salio; revise el disco"
        return True, f"INFORME EN {texto.name}"
    elif accion == "cerrar_orden":
        numero = orden.numero
        # PRIMERO SE CIERRA Y DESPUES SE ESCRIBE. Al reves, el informe que se le entrega
        # al cliente en la mano llevaba impreso "Cerrada: sigue abierta", porque se
        # generaba un instante antes de poner la fecha de cierre. cerrar() devuelve la
        # orden ya cerrada, que es la que hay que imprimir.
        cerrada = ordenes.cerrar()
        texto, planilla = ordenes.escribir_informe(cerrada)
        pan.orden = None
        if texto is None:
            return True, f"ORDEN {numero} CERRADA - NO SE PUDO ESCRIBIR EL INFORME"
        if planilla is None:
            return True, f"ORDEN {numero} CERRADA - el .txt quedo guardado, la planilla NO"
        return True, f"ORDEN {numero} CERRADA - el informe quedo guardado"
    return True, None


def _es_campo_de_numero(pan) -> bool:
    """Si el cursor del formulario esta en un campo que solo acepta digitos.

    IMPORTA PARA SABER DONDE UNA LETRA NO ES UN DATO. En "cuantas sierras" y en "dientes por
    sierra" el formulario ya contesta "aca va un numero" a cualquier letra, asi que ahi se le
    puede dar un significado a la 'r' sin robarsela a nadie. En los campos de texto la 'r'
    sigue siendo una 'r': hay clientes que se llaman Rodriguez, y una tecla que a veces
    escribe y a veces retoma una orden es peor que no tenerla.
    """
    if not 0 <= pan.campo < len(panel.CAMPOS_ORDEN):
        return False
    return panel.CAMPOS_ORDEN[pan.campo][1] == "numero"


def _retomar_la_guardada(pan, ordenes) -> tuple[bool, Optional[str]]:
    """Retoma la orden sin cerrar que el formulario encontro con ese numero.

    SE VUELVE A MIRAR QUE LOS DATOS SEAN LOS MISMOS que cuando se ofrecio. pan.retomar
    guarda la huella de los cinco campos: si el operario cambio alguno desde entonces, lo
    que hay guardado con ese numero puede ser otra cosa, asi que no se retoma nada y se
    vuelve a preguntar. Es el mismo recaudo que habia antes, con la diferencia de que ahora
    la respuesta no es un ENTER.
    """
    valores = [v.strip() for v in pan.valores]
    if not pan.retomar or "|".join(valores) != pan.retomar:
        pan.retomar = ""
        pan.aviso = ("los datos del formulario cambiaron: apreta ENTER para ver que hay "
                     "guardado con ese numero")
        return True, None
    # El mismo texto que Orden.nueva() deja en numero: se junta los espacios y nada mas. El
    # saneado para el nombre del archivo lo hace Ordenes.
    numero = " ".join(str(valores[0]).split())
    archivo = ordenes.ruta(numero).name
    pan.retomar = ""
    retomada = ordenes.retomar(numero)
    if retomada is None or ordenes.fallo_al_guardar:
        pan.aviso = f"no se pudo retomar {archivo}; revise el disco"
        return True, None
    pan.orden, pan.cargando_orden, pan.aviso = retomada, False, None
    return True, (
        f"ORDEN {retomada.numero} RETOMADA: "
        f"{retomada.total_revisados} de {retomada.total_dientes} dientes"
    )


def _cargar_orden(accion, tecla, pan, ordenes) -> tuple[bool, Optional[str]]:
    """El formulario. Un campo por vez, ENTER pasa al siguiente."""
    if accion is not None and accion.startswith("campo:"):
        pan.campo = int(accion.split(":", 1)[1])
        pan.aviso = None
        return True, None

    # LA 'r' RETOMA LA ORDEN GUARDADA. EL ENTER NO LA RETOMA MAS, Y ESO ES EL ARREGLO.
    # QUE SE ROMPIA: el ENTER preguntaba Y contestaba. El primer ENTER del ultimo campo
    # ofrecia "ENTER otra vez para RETOMARLA" y el segundo la retomaba, asi que dos ENTER
    # seguidos --el dedo apoyado, o la costumbre de apretar ENTER por cada campo-- retomaban
    # la orden de ayer sin que el operario llegara a leer el aviso.
    # REPRODUCIDO con el manejador de teclas de verdad: en el disco, la orden 1234 del taller
    # sin cerrar, con 7 de 80 dientes. El operario teclea 1234, "NP-998 urgente", "Aserradero
    # del Litoral", 4 sierras y 80 dientes, y aprieta ENTER dos veces: quedaba abierta la
    # orden de ayer --1 sierra de 80, cliente '1234', nota '1234'-- y las 4 sierras y el
    # cliente que acababa de teclear no iban a ninguna parte.
    # AHORA EL ENTER SOLO PREGUNTA, cuantas veces se lo apriete, y la respuesta es otra
    # tecla. La 'r' se acepta donde una letra no es un dato --los campos de numero, que es
    # donde el ENTER del ultimo campo deja el cursor-- y el boton RETOMAR LA GUARDADA sirve
    # siempre, tambien si se llego aca con un clic desde un campo de texto.
    if accion == "retomar_la_guardada" or (
        pan.retomar and tecla == ord("r") and _es_campo_de_numero(pan)
    ):
        if not pan.retomar:
            # Un clic sobre un boton que ya no esta dibujado: no hay nada ofrecido.
            return True, None
        return _retomar_la_guardada(pan, ordenes)

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
        if ordenes.existe(orden.numero):
            # existe() y no cargar(): un .json roto no se puede leer, pero esta, y
            # escribirle encima borra la unica copia del trabajo de ese dia.
            # Y SE NOMBRA EL ARCHIVO, no el numero tecleado: dos numeros distintos caen en
            # el mismo archivo --"OA 123" y "OA/123" dan los dos OA_123.json-- asi que
            # decir "ya hay una orden OA/123" seria mentir sobre lo que esta guardado.
            archivo = ordenes.ruta(orden.numero).name
            guardada = ordenes.cargar(orden.numero)
            if guardada is None or guardada.cerrada:
                pan.aviso = f"ya hay una orden guardada en {archivo}; usa otro numero"
                return True, None
            # SIN CERRAR: SE OFRECE RETOMARLA, Y NADA MAS. Antes esto era un callejon
            # --"usa otro numero"-- cuando lo que el operario casi siempre quiere es seguir
            # la orden que ya habia empezado, y el programa no tenia como volver a ella. No
            # se retoma sola porque el formulario trae sierras, dientes y cliente nuevos que
            # se perderian sin avisar.
            # ESTE ENTER NO RETOMA, LO APRIETE LAS VECES QUE SEA: solo deja la pregunta en
            # pantalla y arma el boton. Contesta la 'r' o el boton; mira la rama de arriba.
            # Y la huella queda guardada para que, si se cambia cualquier dato, la respuesta
            # no valga para los datos nuevos.
            pan.retomar = "|".join(valores)
            # EL AVISO NOMBRA LA TECLA QUE DE VERDAD VA A ANDAR DONDE ESTA EL CURSOR. En un
            # campo de texto la 'r' es una letra, asi que ahi se nombra el boton.
            # MEDIDO con tipografia.medir en 20 negrita, que es como el formulario lo dibuja:
            # 1.019 px la version de la 'r' y 1.155 px la del boton, de los 1.240 que entran.
            if _es_campo_de_numero(pan):
                como = "Apreta 'r' para RETOMARLA, o cambia el numero"
            else:
                como = "Apreta el boton RETOMAR LA GUARDADA, o cambia el numero"
            pan.aviso = (
                f"ya hay una orden sin cerrar en {archivo}: "
                f"{guardada.total_revisados} de {guardada.total_dientes} dientes. {como}"
            )
            return True, None
        ordenes.abrir(orden)
        if ordenes.fallo_al_guardar:
            # No se anuncia una orden que no llego al disco: el operario la daria por
            # abierta y se pasaria la tarde anotando dientes contra un archivo inexistente.
            pan.aviso = _NO_SE_GUARDO
            return True, None
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


def _limpiar_aviso(pan) -> None:
    """Borra el aviso del panel, SALVO el que esta pegado.

    Un aviso normal cuenta algo que acaba de pasar y la tecla siguiente lo tapa, que es lo
    correcto: si no, el operario arrastra carteles viejos. Pero el del borrado a medias no
    es eso: es el UNICO lugar donde se nombra la carpeta de papelera que quedo con la mitad
    de la orden adentro, y el que lo hacia desaparecer era justamente la tecla del
    reintento. Mira panel.EstadoPanel.aviso_pegado.
    """
    if not getattr(pan, "aviso_pegado", False):
        pan.aviso = None


def _decir(pan, texto: str) -> None:
    """Deja un aviso nuevo en el panel y APAGA el pegado.

    Porque pegado quiere decir "este texto de aca no se limpia solo", no "este renglon de la
    pantalla no se limpia nunca": si el aviso de un borrado a medias quedara pegando un
    "no se pudo exportar" que vino despues, el operario se quedaria arrastrando un cartel
    viejo que ninguna tecla saca. El unico que vuelve a prenderlo es el borrado.
    """
    pan.aviso = texto
    pan.aviso_pegado = False


def _abrir_historial(pan, ordenes) -> None:
    """Deja el panel en el historial, con la orden abierta ya elegida.

    La orden abierta se busca y se elige porque el que aprieta 'o' con una orden en curso
    casi siempre quiere esa: sin esto caia parado en la primera fila --la mas nueva, que
    puede ser otra-- y un ENTER de reflejo retomaba la que no era.

    EL HISTORIAL SE LEE DEL DISCO CADA VEZ QUE SE ENTRA, y tambien al volver de una orden,
    al retomar y al borrar: medido sobre una copia de las 4 ordenes de verdad del taller,
    mediana 2,42 ms y maximo 7,12 ms en 50 vueltas. Al lado de los 1.200 ms que dura apoyar
    un diente no se nota, y una lista vieja muestra el avance de antes o una orden que ya
    no esta. Con las 4 de hoy no hay ninguna abierta --5113546 se cerro el 16/11/2026
    12:11:25-- asi que queda elegida la primera fila, que es la mas nueva.
    """
    pan.modo, pan.aviso = panel.MODO_ORDENES, None
    # Entrar de nuevo al historial empieza de nuevo: hasta el aviso pegado se va. Lo que
    # quedo de un borrado a medias sigue escrito en el registro, que es donde se mira
    # despues.
    pan.aviso_pegado = False
    pan.cargando_orden = False
    pan.historial = ordenes.historial()
    pan.orden_elegida, pan.pagina_ordenes = 0, 0
    for indice, fila in enumerate(pan.historial):
        if fila.esta_abierta:
            pan.orden_elegida = indice
            # LA PAGINA SE CUENTA CON LAS FILAS QUE SE DIBUJAN, que es lo que dejo anotado
            # _dibujar_ordenes --o lo que midio run_live antes de abrir el panel-- y no un 8
            # fijo: mira panel.filas_de_ordenes().
            pan.pagina_ordenes = indice // max(1, pan.filas_por_pagina)
            break


def _panel_ordenes(accion, tecla, pan, ordenes) -> tuple[bool, Optional[str]]:
    """El historial: todas las ordenes del taller, para retomar, exportar o borrar.

    Hasta ahora la unica orden que se podia tocar era la marcada como abierta: listar() no
    se llamaba desde ninguna parte del programa, asi que una orden de la semana pasada --la
    del reclamo que entra por telefono-- no se podia ni mirar sin abrir el Explorador.

    ESTA PANTALLA NO BORRA NADA DE UN SAQUE: la 'b' lleva a la confirmacion, igual que con
    los conceptos. Lo unico que decide aca mismo es exportar, que solo agrega un .zip.
    """
    if accion == "volver_ordenes" or tecla in (_ESC, ord("q")):
        # ESC cierra el panel entero y vuelve a la pantalla del taller: el historial ES el
        # primer piso, no hay nada mas arriba a donde subir.
        return False, None

    # LAS FILAS QUE SE DIBUJARON, no un 8 fijo. _dibujar_ordenes las mide contra el alto del
    # lienzo y las deja anotadas en el estado, y el alto del lienzo depende de la camara:
    # medido en el modo del taller con alto util de 672 px, la camara en 640x480 da un
    # lienzo de 672 px y 8 filas, y en 1280x720 y 1920x1080 da 656 px y 7 filas. Con 8 fijo
    # y la camara en 1280x720, la tecla 8 dejaba elegida la fila 7 --que no estaba
    # dibujada-- y la 'b' siguiente se llevaba una orden que el operario nunca vio.
    # Reproducido corriendo el bucle con el guion de teclas o,8,b,s.
    por_pagina = max(1, pan.filas_por_pagina)

    if accion is not None and accion.startswith("elegir_orden:"):
        pan.orden_elegida = int(accion.split(":", 1)[1])
        _limpiar_aviso(pan)
    elif tecla in _TECLAS_DIGITO and _TECLAS_DIGITO[tecla] <= por_pagina:
        # Solo los digitos de ESTA pagina, igual que en el modo fotos: un digito de mas
        # elegiria una orden que no se ve, que es justo la que no hay que borrar a ciegas.
        indice = pan.pagina_ordenes * por_pagina + _TECLAS_DIGITO[tecla] - 1
        if indice < len(pan.historial):
            pan.orden_elegida = indice
            _limpiar_aviso(pan)
    elif tecla in (ord("s"), ord("w")) and pan.historial:
        paso = 1 if tecla == ord("s") else -1
        pan.orden_elegida = (pan.orden_elegida + paso) % len(pan.historial)
        # LA PAGINA SIGUE AL ELEGIDO. Sin esto, bajar desde la ultima fila de la pagina
        # marcaba la orden siguiente sin cambiar de pagina: la marca desaparecia de la
        # pantalla y las teclas siguientes --e, b-- trabajaban sobre una fila que el
        # operario no veia.
        pan.pagina_ordenes = pan.orden_elegida // por_pagina
    elif tecla == ord(" "):
        paginas = max(1, (len(pan.historial) + por_pagina - 1) // por_pagina)
        pan.pagina_ordenes = (pan.pagina_ordenes + 1) % paginas
        pan.orden_elegida = min(
            pan.pagina_ordenes * por_pagina, max(0, len(pan.historial) - 1)
        )

    # LA FILA SE BUSCA DESPUES DE MOVERSE, no antes: con la 's' y la fila elegida leidas
    # en el mismo orden al reves, un clic que elige y una tecla que actua en la misma vuelta
    # trabajaban sobre la fila vieja.
    fila = None
    if 0 <= pan.orden_elegida < len(pan.historial):
        fila = pan.historial[pan.orden_elegida]

    if accion == "nueva_orden_historial" or tecla == ord("n"):
        pan.modo, pan.cargando_orden, pan.aviso = panel.MODO_ORDEN, True, None
        pan.aviso_pegado = False
        pan.valores = ["" for _ in panel.CAMPOS_ORDEN]
        # El 'retomar' se limpia aca y no en el formulario: es la huella del "ENTER otra vez
        # para RETOMARLA" de la vez anterior, y si quedara puesta el primer ENTER de este
        # formulario retomaria una orden sin preguntar nada.
        pan.campo, pan.retomar = 0, ""
        return True, None

    if accion == "abrir_orden" or tecla == 13:
        if fila is None:
            return True, None
        if fila.ilegible:
            _decir(pan, f"el archivo {fila.archivo}.json esta pero no se puede leer: "
                        "hay que mirarlo a mano, el programa no lo va a abrir")
            return True, None
        if fila.cerrada:
            # NO SE REABRE UNA ORDEN CERRADA, y se dice por que: su informe ya se imprimio
            # y se entrego, asi que seguir anotando dientes deja el papel del cliente
            # diciendo menos de lo que el archivo guarda. Sin este aviso, retomar()
            # devolvia None y la pantalla culpaba al disco, que no tenia nada que ver.
            _decir(pan, f"la orden {fila.numero} se cerro el {fila.cerrada}: su informe "
                        "ya salio. Para exportarla apreta 'e'")
            return True, None
        # SE RETOMA EL ARCHIVO QUE LEYO EL HISTORIAL, no el numero de adentro. Pero
        # retomar() deja la orden abierta y, de ahi en adelante, cada diente se guarda en
        # ruta(numero) --el numero saneado-- asi que una orden cuyo archivo NO se llama como
        # su numero se leeria de un archivo y se escribiria en otro. Eso es peor que no
        # retomarla: el operario se pasaria la tarde anotando dientes encima de la orden del
        # cliente. Medido con "123 - copia.json" (numero 123 adentro, que es lo que deja
        # Windows al copiar): el archivo es '123 - copia' y sanear_numero da '123_-_copia',
        # mientras que lo que se escribiria es 123.json. Asi que se mira primero y, si no
        # coinciden, no se retoma y se dice por que: exportar y borrar SI andan, porque esos
        # ya van por el archivo.
        donde_escribiria = ordenes.ruta(fila.numero).stem
        if fila.archivo != donde_escribiria:
            # MEDIDO en 20 negrita, que es como el panel dibuja el aviso: este texto con los
            # nombres mas largos que hay en el taller da 1.133 px de los 1.258 que entran.
            # Un archivo con nombre mas largo lo acorta _dibujar_ordenes.
            _decir(pan, f"{fila.archivo}.json no se llama como su numero "
                        f"({fila.numero}): anotar ahi escribiria en "
                        f"{donde_escribiria}.json. Exportala con 'e' o borrala con 'b'")
            return True, None
        retomada = ordenes.retomar(fila.numero)
        if retomada is None or ordenes.fallo_al_guardar:
            _decir(pan, f"no se pudo retomar {fila.archivo}; revise el disco")
            return True, None
        pan.orden, pan.cargando_orden, pan.aviso = retomada, False, None
        pan.aviso_pegado = False
        pan.modo = panel.MODO_ORDEN
        # El historial se vuelve a leer porque esta orden acaba de pasar a ser la abierta:
        # la fila que queda atras diria que no lo es.
        pan.historial = ordenes.historial()
        return True, (
            f"ORDEN {retomada.numero} RETOMADA: "
            f"{retomada.total_revisados} de {retomada.total_dientes} dientes"
        )

    if accion == "exportar_orden" or tecla == ord("e"):
        if fila is None:
            return True, None
        if fila.ilegible:
            _decir(pan, f"el archivo {fila.archivo}.json no se puede leer: no se "
                        "exporta como prueba algo que el programa no pudo abrir")
            return True, None
        # EL ARCHIVO QUE LEYO EL HISTORIAL, no el numero de adentro: mira el comentario de
        # abajo, en el borrado. Reproducido: con "123.json" (cliente EL CLIENTE BUENO) y
        # "copia_vieja.json" (cliente UNA COPIA VIEJA), las dos con numero 123 adentro,
        # exportar la fila copia_vieja armaba un zip con la orden.json de EL CLIENTE BUENO.
        donde = ordenes.exportar(fila.numero, archivo=fila.archivo)
        if donde is None:
            _decir(pan, "no se pudo exportar; revise el disco")
            return True, None
        # Se nombra la carpeta ademas del archivo --el operario tiene que ir a buscar ese
        # .zip con el Explorador para copiarlo al pendrive-- Y CUANTO PESA. Cada 'e' escribe
        # un zip NUEVO, sin tope y sin limpieza: medido, la orden de verdad 5113546 (15
        # dientes, ninguna prueba) da 3.275 bytes, y una de 4 sierras de 80 dientes con las
        # 320 fotos de prueba adentro da 28.178.007 bytes --26,9 MiB-- en 847 ms. Diez 'e'
        # de esas en un dia son 269 MiB que nadie borra, y sin el tamanio en pantalla el
        # operario no tiene de donde darse cuenta de que esta llenando el disco. El cartel
        # mas largo que sale asi mide 761 px de los 1.244 que entran: medido con
        # tipografia.medir en 22 negrita, que es como se dibuja.
        return True, (f"EXPORTADA EN {donde.parent.name}/{donde.name} "
                      f"- {_cuanto_pesa(donde)}")

    if accion == "borrar_orden" or tecla == ord("b"):
        if fila is None:
            return True, None
        # LA 'b' DEL REINTENTO NO SE LLEVA EL AVISO DEL BORRADO A MEDIAS. Antes este renglon
        # ponia pan.aviso en None de una, y ese aviso era el unico lugar donde se nombraba
        # la carpeta de papelera con la mitad de la orden adentro: el operario que apretaba
        # 'b' para reintentar lo perdia antes de poder leerlo. REPRODUCIDO con la orden 1234
        # y 3 pruebas, trabando una foto con msvcrt.locking: el aviso nombraba
        # _papelera/20261116_164709 y despues de la 'b' pan.aviso era None.
        pan.modo = panel.MODO_BORRAR_ORDEN
        _limpiar_aviso(pan)
    return True, None


def _panel_borrar_orden(accion, tecla, pan, ordenes) -> tuple[bool, Optional[str]]:
    """La confirmacion de borrado de una orden. ENTER NO alcanza, a proposito.

    Igual que con los conceptos: en el historial ENTER es "retomar esta orden", la tecla
    con la que se entra a trabajar. Si aca tambien confirmara, una 'b' de mas seguida del
    ENTER de siempre se llevaria la orden --y sus pruebas-- a la papelera.
    """
    fila = None
    if 0 <= pan.orden_elegida < len(pan.historial):
        fila = pan.historial[pan.orden_elegida]

    if accion == "confirmar_borrado_orden" or tecla == ord("s"):
        if fila is None:
            pan.modo = panel.MODO_ORDENES
            return True, None
        era_la_abierta = fila.esta_abierta
        # SE BORRA EL ARCHIVO QUE LEYO EL HISTORIAL, no el numero de adentro del .json.
        # Antes iba fila.numero y Ordenes le volvia a aplicar sanear_numero(), asi que la
        # fila elegida y el archivo que se iba podian ser dos ordenes distintas.
        # REPRODUCIDO DOS VECES, con el manejador de teclas de verdad:
        #   (a) "123.json" (cliente EL CLIENTE BUENO) y "copia_vieja.json" (cliente UNA
        #       COPIA VIEJA), las dos con numero 123 adentro. Se elige la fila copia_vieja,
        #       se aprieta b+s: el que se fue a la papelera fue 123.json, el del cliente, y
        #       copia_vieja.json quedo en su lugar.
        #   (b) "123 - copia.json", que es como lo llama Windows al copiar y lo que la
        #       pantalla de borrado le sugiere al operario. sanear_numero("123 - copia") da
        #       "123_-_copia", asi que pasar fila.archivo por `numero` TAMPOCO alcanzaba:
        #       hace falta el parametro archivo=, que entra sin sanear.
        donde = ordenes.borrar(fila.numero, archivo=fila.archivo)
        pan.modo = panel.MODO_ORDENES
        # EL HISTORIAL SE RELEE SIEMPRE, salga bien o salga mal: un borrado que fallo a
        # mitad de camino ya movio parte de los archivos, asi que la lista de antes es
        # tan vieja en un caso como en el otro. Y se recorta la eleccion igual que al
        # borrar una foto, o la fila elegida queda apuntando a una orden que no esta.
        pan.historial = ordenes.historial()
        pan.orden_elegida = max(0, min(pan.orden_elegida, len(pan.historial) - 1))
        # Con las filas que se dibujaron, no con un 8 fijo: mira panel.filas_de_ordenes().
        por_pagina = max(1, pan.filas_por_pagina)
        ultima = max(0, (len(pan.historial) + por_pagina - 1) // por_pagina - 1)
        pan.pagina_ordenes = min(pan.pagina_ordenes, ultima)
        if donde is None:
            # SE NOMBRA LA ORDEN COMO LA NOMBRA LA PANTALLA, con el archivo cuando hace
            # falta: "no se pudo borrar la orden" a secas, con dos filas que dicen 1234, no
            # dice cual de las dos sigue ahi. Y queda PEGADO: la orden no se borro, eso no es
            # una novedad que la tecla siguiente pueda tapar.
            pan.aviso = (f"no se pudo borrar la orden {panel.nombre_de_la_fila(fila)}; "
                         "revise el disco")
            pan.aviso_pegado = True
            return True, None
        if era_la_abierta:
            # Ya no hay orden en curso: borrar() suelta el .abierta. Se dice tambien aca
            # para que el panel no siga mostrando el avance de una orden que se fue. SE LE
            # PREGUNTA A ordenes, no se pone None a mano: un borrado que se traba antes de
            # llegar al .json deja el marcador puesto --a proposito, para que el operario no
            # pierda la orden abierta por una foto que no se pudo mover-- y ahi el panel
            # tiene que seguir mostrando esa orden, no vaciarse.
            pan.orden, pan.cargando_orden = ordenes.abierta, False
        if not donde.completo:
            # QUEDO A MEDIAS Y SE DICE CON TODAS LAS LETRAS. Antes esto era un "no se pudo
            # borrar la orden" que era falso justo cuando mas importaba: medido con una foto
            # trabada --lo que pasa si el operario la tiene abierta en el visor de Windows--
            # el .json y los dos informes YA estaban en la papelera y el historial habia
            # quedado vacio, asi que el operario buscaba la orden en la unica pantalla desde
            # donde se la podia exportar. Con el orden nuevo de los movimientos el .json es
            # lo ultimo que se va, asi que lo normal es que la orden siga ENTERA en la lista
            # y lo que falte sea una foto; igual hay que nombrar la carpeta de la papelera,
            # que es a donde hay que ir a mirar.
            #
            # VA COMO AVISO DEL PANEL Y NO COMO CARTEL: el panel se queda abierto en el
            # historial, no dibuja estado.mensaje, y el cartel verde del taller no se ve
            # hasta que se cierre el panel. Se devuelve None para que el bucle no pise este
            # aviso con un texto mas corto.
            #
            # Y QUEDA PEGADO: la 'b' del reintento lo limpiaba, y es el unico lugar donde se
            # nombra la carpeta de papelera. Mira panel.EstadoPanel.aviso_pegado.
            #
            # SE NOMBRA EL ARCHIVO Y NO EL NUMERO: lo que quedo en la papelera se llama como
            # el archivo, y con dos filas que dicen 1234 el numero no alcanza para saber cual
            # se fue a medias. La carpeta de papelera va PRIMERO porque es a donde hay que ir
            # a mirar, y porque si el texto se pasara de ancho _acortar corta por el final.
            # MEDIDO en 20 negrita, que es como el panel lo dibuja: con los nombres mas largos
            # del taller da 1.000 px de los 1.258 que entran. Nombrando el numero Y el archivo
            # daba 1.319 px y se cortaba justo la carpeta.
            pan.aviso = (
                f"el borrado de {fila.archivo}.json quedo A MEDIAS en "
                f"_papelera/{donde.destino.name}: falto {donde.falto}"
            )
            pan.aviso_pegado = True
            return True, None
        # SALIO COMPLETO: se limpia hasta el aviso pegado, porque lo que decia --que habia
        # quedado a medias-- ya no es cierto.
        pan.aviso, pan.aviso_pegado = None, False
        # EL CARTEL NOMBRA LA MISMA ORDEN QUE NOMBRO LA PANTALLA, con el archivo cuando no se
        # llama como su numero. Medido con la orden 1234 del taller y su copia de Windows: los
        # dos borrados devolvian el mismo cartel, "ORDEN 1234 A LA PAPELERA", asi que el
        # operario no tenia con que saber cual de las dos se habia ido.
        # MEDIDO con tipografia.medir en 22 negrita, que es como se dibuja el cartel del
        # taller: 784 px de los 1.244 que entran con el nombre mas largo.
        return True, (f"ORDEN {panel.nombre_de_la_fila(fila)} A LA PAPELERA: "
                      f"{donde.destino.name}")

    if accion == "cancelar_borrado_orden" or tecla in (_ESC, ord("n"), ord("q")):
        # Tampoco al decir "no, dejala": lo que el aviso pegado dice --que un borrado quedo a
        # medias y donde esta la mitad-- sigue siendo cierto despues de cancelar, y el que
        # cancela es justamente el que va a ir a mirar esa carpeta.
        pan.modo = panel.MODO_ORDENES
        _limpiar_aviso(pan)
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
            # El mismo minimo con el que se juzga: la memoria lo necesita para medirse con
            # la regla con la que decide. Ver Memoria.autoevaluacion().
            minimo_por_concepto=cfg.memoria.minimo_por_concepto,
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
    # Las fotos de ejemplo que se muestran al lado de cada diente, leidas una sola vez.
    ejemplos: dict = {}
    # La captura automatica: con una orden abierta, el programa anota los dientes solo.
    automatica = CapturaAutomatica(
        segundos_entre_fotos=cfg.interfaz.segundos_entre_fotos,
        iguales_para_anotar=cfg.interfaz.fotos_iguales_para_anotar,
    )
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
                # Medirse cuesta y se paga ACA, con el aviso de GUARDADA ya pintado en
                # pantalla por archivar(): medido, 118 ms con 15 fotos del microscopio,
                # 575 ms con 41 y 1554 ms con 121. Ojo que resumen(), dos lineas mas
                # abajo, tambien llama a autoevaluacion(), asi que la cuenta se paga
                # igual: esta linea solo deja dicho donde se paga.
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

    def diente_elegido() -> Optional[revision.DienteEnPantalla]:
        return next((d for d in estado.dientes if d.elegido), None)

    def recorte_a_archivar(imagen: np.ndarray) -> np.ndarray:
        """Lo que se guarda es EL DIENTE, no la foto entera.

        La memoria juzga el recorte de cada diente, asi que lo que se archiva tiene que
        ser lo mismo que se juzgo: si se guardara la foto completa, lo que el programa
        aprende y lo que el programa compara serian dos cosas distintas.
        """
        elegido = diente_elegido()
        if elegido is None:
            return imagen
        # EL RECUADRO Y LOS PIXELES, DE LA MISMA FOTO. El recuadro se midio en el ultimo
        # cuadro que paso por el analisis --uno de cada cinco-- y recortarlo de la foto de
        # AHORA lo deja corrido lo que se movio la mano mientras tanto. Reproducido en el
        # bucle con una pieza que corre 12 px por cuadro: apretando 'a' cuatro cuadros
        # despues del analisis, el recuadro era (87,158,302,342) y la pieza ya estaba en
        # (135,158,350,342) --48 px a la derecha-- y en el recorte archivado la pieza baja
        # del 48% al 44% de los pixeles; a 20 px por cuadro cae al 36%. Eso entra corrido
        # al dataset y a la memoria, que despues compara recortes asi contra recortes
        # centrados. La foto que se le pasa a la captura automatica sigue siendo la de
        # ahora: su freno compara huellas contra el cuadro en vivo.
        cuadro = cuadro_analizado if cuadro_analizado is not None else imagen
        return dientes.Diente(caja=elegido.caja, area=0.0).recortar(cuadro)

    def prueba_del_diente(imagen: np.ndarray) -> tuple[str, str]:
        """Guarda la foto que respalda al diente que SIGUE. Devuelve (ruta, falta).

        SE LLAMA ANTES DE orden.anotar(), y el orden no es un gusto: anotar() es lo que
        ASIGNA el numero de diente y devuelve None cuando la sierra ya esta completa, asi
        que el numero no se puede averiguar despues sin haber escrito ya la foto, ni la
        foto antes sin saber el numero. La salida es ESPIAR primero: orden.sierra() y
        sierra.proximo() dicen cual viene SIN tocar nada, se escribe la foto con ese
        numero puesto, y recien ahi se anota con la ruta en la mano. Asi el .json nunca
        nombra una foto que no esta --es lo que paso con las 14 fotos de las ordenes 123 y
        1234, que hoy estan en la papelera mientras el informe_123.csv ya entregado las
        sigue nombrando--. Al reves no le miente a nadie: una foto de mas en el deposito.

        SI NO SE PUDO ESCRIBIR, la ruta vuelve vacia y `falta` dice que paso: el diente se
        anota IGUAL, con prueba vacia y esa nota. Perder la foto es malo, perder el diente
        es peor, y un renglon que dice que no tiene prueba no inventa nada; el diente que
        no esta en el informe es el diente que nadie reviso.

        LA PRUEBA NO ENTRA A LA COLECCION y por eso esto no cuesta lo que cuesta
        archivar: verificado leyendo ordenes.guardar_prueba(), que solo comprime y escribe
        en data/pruebas/ y no llama ni a refrescar() ni a sincronizar() ni mira el almacen.
        Archivar una foto arrastra refrescar(), que va de 209 ms con 15 fotos a 3.153 ms
        con 320; esto no.

        MEDIDO EN ESTA PC con una foto de verdad del microscopio (640x480, 95.648 bytes),
        80 dientes seguidos --una sierra entera-- por el camino completo
        "espiar + guardar_prueba + anotar + guardar()": mediana 4,98 ms y p95 7,18 ms
        (tres corridas: medianas 4,95 / 4,98 / 5,07 ms). El mismo camino SIN prueba, que es
        lo que hace el programa hoy, da mediana 2,27 ms: la prueba agrega 2,7 ms. Contra
        los 1.200 ms que el operario apoya cada diente, el camino entero es el 0,4 % del
        tiempo. El max salto a 13, 37 y 63 ms segun la corrida, pero el camino sin prueba
        tambien salto a 46 ms: es el disco tosiendo, no la foto.

        LO QUE OCUPA, medido escribiendo las 80 de verdad: 7.670.480 bytes = 7,32 MiB la
        sierra entera, y el .json de esa orden queda en 30.770 bytes. Una orden de 4
        sierras de 80 dientes son 29,3 MiB. En el C: de esta PC hay 116,7 GiB libres:
        entran mas de 4.000 ordenes asi antes de que el disco diga basta.
        """
        orden = ordenes.abierta
        if orden is None or imagen is None:
            return "", ""
        sierra = orden.sierra()
        que_viene = sierra.proximo() if sierra is not None else None
        if que_viene is None:
            # La sierra ya esta completa: anotar() va a devolver None y no hay diente que
            # respaldar. No se escribe ninguna foto --una prueba de un diente que no se
            # anoto no respalda nada-- y el camino sigue como seguia hasta hoy.
            return "", ""
        ruta = ordenes.guardar_prueba(
            orden.numero, sierra.numero, que_viene.numero, imagen
        )
        if ruta is None:
            return "", "no se pudo guardar la foto de prueba"
        return ruta, ""

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
        completa = imagen
        imagen = recorte_a_archivar(imagen)
        motivo = motivo_para_no_archivar(camara, imagen, cfg, almacen)
        if motivo is not None:
            _registro.warning("no se archivo la foto: %s", motivo)
            return motivo
        ruta = almacen.guardar(imagen, destino)
        estado.capturas = almacen.total_guardados
        _registro.info("foto archivada en '%s': %s", destino, ruta.name)
        mensaje = f"GUARDADA como {destino}: {ruta.name}"
        # EL CARTEL PRIMERO Y LA CUENTA DESPUES. refrescar() vuelve a medir la memoria
        # contra si misma y eso TARDA: medido con fotos del microscopio a 640x480, 118 ms
        # con 15 fotos repartidas en dos conceptos, 575 ms con 41 y 1554 ms con 121 --ahi
        # lo frena
        # maximo_autoevaluacion=120-- mas 7 a 26 ms de sincronizar. Mientras eso corre el
        # bucle no vuelve a dibujar: la ventana se queda con el cuadro anterior y SIN
        # aviso, asi que el operario ve que el programa se clava y vuelve a apretar la
        # tecla. Con el panel abierto no se repinta, porque la pantalla es el panel y el
        # panel avisa por su cuenta.
        if pan is None:
            estado.mensaje = mensaje
            cv2.imshow(
                _VENTANA, componer(completa, galeria.imagen_actual(), cfg, estado).imagen
            )
            cv2.pollKey()
        refrescar()
        miniaturas.olvidar()

        # Con una orden abierta, archivar un diente tambien lo ANOTA: el conteo avanza y
        # queda en el informe. Lo que vale como estado es lo que dijo el operario --el
        # archivo donde lo puso-- y lo que opinaba el programa queda al lado, para que un
        # reclamo pueda ver las dos cosas.
        orden = ordenes.abierta
        if orden is not None:
            elegido = diente_elegido()
            juicio = elegido.juicio if elegido is not None else estado.juicio
            opinion = ""
            if juicio is not None and juicio.decidio:
                opinion = f" (la memoria decia {juicio.concepto})"
            elif elegido is not None and elegido.concepto and elegido.parecido is not None:
                # LO QUE MOSTRABA LA PANTALLA TIENE QUE QUEDAR ESCRITO. Hoy el veredicto no
                # lo da la memoria juzgando sino el parecido contra las fotos archivadas,
                # asi que juicio es None y este renglon se iba al informe sin una sola
                # cifra. Medido en el bucle con el taller de hoy: el cartel decia
                # "APROBADO 100% - se parece a afilado" y el renglon quedaba con la columna
                # parecido VACIA y nada mas. En la columna no puede ir --ahi va cuanto se
                # parecio al estado que eligio el operario, que es otra cosa-- pero en el
                # origen si, y asi el informe no pierde lo que se vio en pantalla.
                opinion = (
                    f" (en pantalla decia {elegido.concepto} al "
                    f"{elegido.parecido * 100:.0f}%)"
                )
            # EL PORCENTAJE TIENE QUE SER DEL ESTADO QUE ESTA EN LA MISMA FILA. El informe
            # dice que la columna 'parecido' es cuanto se parecio AL ESTADO que figura al
            # lado, y por este camino el estado lo pone el operario mientras el porcentaje
            # salia de la memoria, que podia estar opinando otra cosa. Paso en el bucle con
            # las fotos del taller: la memoria decia mal_afilado al 66%, el operario lo
            # archivaba como afilado, y el informe quedaba "afilado APROBADO 66%" --un
            # numero que nadie midio sobre 'afilado'-- en el papel que contesta un reclamo.
            # Si no coinciden, el porcentaje no va: lo que opinaba el programa queda dicho
            # en el origen, con todas las letras.
            coinciden = (
                juicio is not None and juicio.decidio and juicio.concepto == destino
            )
            # LA FOTO DE PRUEBA VA ANTES DEL RENGLON: mira prueba_del_diente(). Se le
            # pasa `completa` --el cuadro entero que el programa juzgo-- y NO el recorte
            # que se acaba de archivar: el recorte es para ensenarle a la memoria, y lo
            # que contesta un reclamo es lo que se vio en la pantalla. Son dos fotos con
            # dos trabajos distintos, y por eso tambien son dos campos distintos del
            # diente: `foto` es el nombre pelado de la que entro a la coleccion y `prueba`
            # es la ruta de la que respalda el veredicto.
            prueba, falta_prueba = prueba_del_diente(completa)
            diente = orden.anotar(
                estado=destino,
                parecido=juicio.parecido if coinciden else None,
                aprobado=destino not in cfg.clases_defecto,
                origen=f"lo archivo el operario{opinion}",
                foto=ruta.name,
                nota=falta_prueba,
                prueba=prueba,
            )
            guardado = ordenes.guardar()
            actualizar_orden()
            # ESTE DIENTE YA QUEDO ANOTADO: hay que decirselo a la captura automatica o lo
            # anota de nuevo ella sola. Medido en el bucle con una orden abierta y la pieza
            # quieta: una sola 'a' dejaba DOS renglones para el mismo diente, el primero
            # con el estado que eligio el operario y el segundo con el que opinaba el
            # programa un segundo despues. Se le pasa la foto COMPLETA y no el recorte,
            # porque la huella del freno se toma sobre el cuadro entero.
            automatica.ya_anotado(completa)
            if guardado is None:
                # La FOTO si se guardo --eso ya esta en disco-- pero el renglon de la
                # orden no, y es lo que hay que decir: lo otro el operario lo da por hecho.
                mensaje = _NO_SE_GUARDO
            elif diente is None:
                # LA 'o' NO PASA A LA SIGUIENTE SIERRA: abre el HISTORIAL. Pasar de
                # sierra son tres teclas, y el cartel las dice en el orden en que se
                # aprietan: 'o' abre el historial con la orden en curso ya elegida, ENTER la
                # retoma y entra a su pantalla, y ahi la 's' es SIGUIENTE SIERRA. Comprobado
                # apretandolas con el manejador de teclas de verdad.
                mensaje = ("ESTA SIERRA YA ESTA COMPLETA - para la que sigue: "
                           "'o', ENTER y 's'")
            else:
                sierra = orden.sierra()
                mensaje = (
                    f"DIENTE {diente.numero} de {sierra.dientes}: {destino}"
                    if sierra is not None else mensaje
                )
                # EL OPERARIO TIENE QUE ENTERARSE EN EL MOMENTO. El diente quedo anotado
                # --eso es lo importante-- pero sin la foto que lo respalda, y eso recien
                # se nota el dia del reclamo si nadie lo dice ahora. Cuando la orden
                # tampoco se guardo manda el otro cartel: ahi no hay renglon ni prueba.
                if falta_prueba:
                    mensaje += " - SIN FOTO DE PRUEBA"
        return mensaje

    recortes_modelo: dict = {}
    fotos_modelo: dict = {}

    def todas_las_de_modelo() -> list:
        """Las PRIMERAS 24 fotos de modelo de la galeria, leidas una sola vez.

        EL TOPE DE 24 ES DE COSTO, y lo que queda afuera no se compara contra nada: con 50
        fotos en la carpeta se miran 24 y las otras 26 no se miran nunca. Medido con
        cuadros de 640x480: el parecido contra 24 fotos cuesta 58 ms y contra 50 cuesta
        117 ms --2.3 ms por foto-- asi que un analisis pasa de 21 ms con 3 fotos a 74 ms
        con 24. El bucle espera 30 ms por tecla en cada cuadro y analiza uno de cada cinco;
        la PC del taller es mas lenta que la de desarrollo, donde se midio esto, asi que
        subir el tope hay que medirlo alla. Las 24 quedan en memoria: 0.9 MB cada una.

        OJO CON UNA FOTO ILEGIBLE: si un archivo de la carpeta no se puede leer --un .jpg
        a medio copiar-- la cache queda con menos entradas que nombres y no valida nunca,
        asi que las demas vuelven del disco en cada analisis, tambien con menos de 24.
        Medido con 3 fotos mas una cortada: 49 lecturas en 60 cuadros contra 4, y 22.6 ms
        por cuadro contra 15.2.

        El parecido se mide contra la MEJOR de todas y no contra la que este elegida:
        medido sobre las fotos del taller, contra una sola foto ningun diente bueno pasaba
        el 70% --mediana 57%-- y contra las tres pasaban diez de catorce, con mediana 78%.
        El numero no cambio porque el filo cambiara: cambio porque antes se lo comparaba
        contra el encuadre equivocado.
        """
        carpeta = galeria.carpeta
        if not carpeta.is_dir():
            return []
        nombres = sorted(
            p.name for p in carpeta.iterdir()
            if p.is_file() and p.suffix.lower() in EXTENSIONES
        )
        # LA CACHE SE COMPARA CONTRA LAS 24 QUE GUARDA, no contra todo lo que hay en la
        # carpeta: arriba de 24 los dos conjuntos nunca eran iguales, asi que las 24 fotos
        # se volvian a leer del disco en CADA analisis. Medido con 60 cuadros de 640x480 y
        # clasificar_cada_n_frames=5: con 25 fotos en la carpeta, 289 cv2.imread y 21.6 ms
        # por cuadro; con esta linea, 25 imread y 13.5 ms. Con 24 fotos ya andaba bien
        # (25 imread, 14.8 ms), por eso en el taller --que tiene 3-- no se nota.
        if set(nombres[:24]) != set(fotos_modelo):
            fotos_modelo.clear()
            for nombre in nombres[:24]:
                imagen = cv2.imread(str(carpeta / nombre))
                if imagen is not None:
                    fotos_modelo[nombre] = imagen
        # Con el NOMBRE de cada una: la que gana es la que hay que escribir en el cartel.
        return list(fotos_modelo.items())

    def recorte_modelo(referencia):
        """El DIENTE de la foto de modelo, recortado igual que el de la camara.

        Se guarda por nombre de foto: detectar el diente cuesta unos milisegundos y la
        foto de modelo no cambia entre cuadro y cuadro.
        """
        if referencia is None:
            return None
        # LA CLAVE LLEVA LA EXTENSION: nombre_actual viene sin ella, asi que modelo.jpg y
        # modelo.png compartian el recorte. Reproducido en el bucle con dos dientes a la
        # vista: con modelo.jpg (un diente) y modelo.png (dos dientes), la 'm' dejaba los
        # parecidos clavados en (0.6823, 0.6173) --el recorte de la primera-- y la cabecera
        # decia "modelo" las dos veces, asi que no habia como notarlo. Con nombres
        # distintos el mismo par da (1.0000, 0.6328) sobre la que corresponde.
        clave = galeria.archivo_actual or ""
        if clave not in recortes_modelo:
            if len(recortes_modelo) > 8:
                recortes_modelo.clear()
            # CON LA MISMA REGLA QUE LA FOTO DE LA CAMARA. Con maximo=1 el que queda
            # es el de MASCARA mas grande, y en las fotos del taller esa es la del hueco:
            # medido sobre afilado_20261112_161626_660, maximo=1 devuelve
            # (84,300,261,447) --el hueco, mascara 0.0443-- y la pantalla elige
            # (237,63,323,308) --la cara de widia, mascara 0.0425 y rayado 43.6 contra
            # 30.4--. Si la foto de modelo se recorta por el hueco, todo lo que se compare
            # contra ella se mide contra un hueco. Pasa en 1 de las 36 fotos reales; en
            # las otras 35 el recorte es el mismo.
            encontrados = dientes.detectar(referencia, maximo=_MAX_DIENTES)
            if encontrados:
                cual = dientes.mas_parecido_a_diente(encontrados)
                recortes_modelo[clave] = encontrados[cual].recortar(referencia)
            else:
                recortes_modelo[clave] = referencia
        return recortes_modelo[clave]

    def actualizar_orden() -> None:
        """Pone en la pantalla en que diente va la orden, si hay una abierta."""
        orden = ordenes.abierta
        estado.orden_resumen = orden.resumen() if orden is not None else ""
        estado.barra_dientes = _barra_de_la_sierra(orden)
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
    # EL CUADRO SOBRE EL QUE SE MIDIERON LOS RECUADROS. El analisis corre uno de cada
    # cada_n_frames cuadros --5 en el config del taller-- asi que cuando el operario
    # aprieta 'a' el recuadro casi nunca es de la foto que la camara acaba de entregar.
    cuadro_analizado: Optional[np.ndarray] = None
    # CUAL DE LOS DIENTES ELIGIO EL OPERARIO, por numero y no por objeto. El analisis
    # vuelve a armar la lista de dientes en cada cuadro, asi que si la eleccion viviera
    # solo dentro de esos objetos se perderia al cuadro siguiente: en vivo, apretar '2'
    # marcaba el diente de la derecha por unos milisegundos y despues la marca volvia sola
    # al primero. None = el operario no eligio, y manda el que el analisis marco por
    # defecto, que es el que mas cara de diente tiene y no el que esta mas a la izquierda.
    elegido_diente: Optional[int] = None
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
                        # Y SE OLVIDA EL CUADRO ANALIZADO. Con la imagen borrosa el
                        # analisis no corre, asi que ese cuadro se queda viejo sin
                        # limite; recortar de el al archivar saltea el control de
                        # nitidez, que es el unico que frena una foto movida. Medido
                        # con el microscopio tapado (nitidez 0.2, minimo 30): la "a"
                        # pasaba de "IMAGEN MOVIDA - enfoque y repita" y cero fotos a
                        # archivar una de cuatro cuadros antes, con nitidez 343, y con
                        # una orden abierta a anotar ese diente en el informe.
                        cuadro_analizado = None
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
            # EL VEREDICTO ES DE ESTE CUADRO O NO SE USA PARA ANOTAR. Se enciende recien
            # abajo, cuando el analisis corre de verdad. Lo de antes era: el analisis se
            # saltea --imagen fuera de foco, panel abierto, 1 de cada N cuadros-- pero
            # estado.dientes se queda con lo ULTIMO que se vio, y la captura automatica
            # seguia contando con eso. Medido en el bucle: tapando el microscopio con la
            # mano durante cinco segundos, el informe anotaba el diente con el parecido de
            # la foto de antes de taparlo.
            #
            # EL DIA QUE APAREZCA UN MODELO ENTRENADO, ESTO DEJA DE ANOTAR SOLO, y conviene
            # saberlo de antemano para no buscarlo como si fuera una rotura nueva. Tanto
            # juicio_fresco como estado.dientes se llenan unicamente en la rama de abajo,
            # que corre cuando NO hay clasificador; con un .onnx entrenado la rama no corre,
            # _resultado_del_diente no encuentra dientes y devuelve None, y la captura
            # automatica no confirma nunca. No es una rotura de esta tanda --estado.dientes
            # ya se llenaba solo ahi-- pero no estaba dicho en ningun lado. Para que ande
            # con modelo hay que hacer dos cosas: que _resultado_del_diente sepa leer
            # tambien estado.veredicto, y encender juicio_fresco en la rama del
            # clasificador, donde hoy se decide clase_candidata.
            juicio_fresco = False
            if clasificador is None and not estado.esperando_foco and pan is None:
                if (estado.congelado and not juzgado) or (
                    not estado.congelado and frames_vistos % cada_n_frames == 0
                ):
                    juzgado = estado.congelado
                    juicio_fresco = True
                    estado.juicio = None
                    estado.dientes = _analizar_dientes(
                        a_mostrar, memoria, cfg, recorte_modelo(referencia), ejemplos,
                        referencia_entera=referencia,
                        todas_las_modelo=todas_las_de_modelo(),
                    )
                    cuadro_analizado = a_mostrar
                    if estado.dientes and elegido_diente is not None:
                        cual = min(elegido_diente, len(estado.dientes) - 1)
                        for numero, uno in enumerate(estado.dientes):
                            uno.elegido = numero == cual
                    # Con UN solo diente --lo normal-- el cartel grande sigue siendo el de
                    # siempre, con lo que ese diente dio. Con varios, cada uno tiene lo
                    # suyo sobre su recuadro y el cartel pasa a ser el resumen.
                    estado.parecido = revision.parecido(a_mostrar, referencia)
                    if len(estado.dientes) == 1:
                        # EL CARTEL Y EL RECUADRO TIENEN QUE DECIR LO MISMO. Con un solo
                        # diente son la misma pieza: si el cartel mostrara el parecido del
                        # cuadro entero y el recuadro el del diente, la pantalla tendria
                        # dos numeros --y dos colores-- para una sola sierra. Paso: 84% en
                        # verde abajo y 40% en rojo arriba, a la vez.
                        estado.juicio = estado.dientes[0].juicio
                        if estado.dientes[0].parecido is not None:
                            estado.parecido = estado.dientes[0].parecido
            # LA CAPTURA AUTOMATICA. Con una orden abierta y la imagen en vivo, el
            # programa mira cinco fotos por segundo, las juzga, las tira, y cuando varias
            # seguidas dan lo mismo anota ese diente y pasa al siguiente. El operario
            # mueve la sierra y no toca una tecla.
            #
            # LA GUARDA DE FOCO NO LE CIERRA LA PUERTA A LAS FOTOS BUENAS, y habia que
            # comprobarlo antes de ponerla: si las fotos del microscopio no llegaran al
            # minimo de nitidez, esto dejaria de anotar para siempre y seria peor que el
            # defecto que vino a tapar. Medido sobre las 14 fotos reales del taller a
            # 640x480 contra el minimo de 30 que pide el config desplegado:
            #
            #     las 14 como salieron       nitidez 85.2 a 408.6 (mediana 279.7)   pasan 14
            #     las mismas, desenfocadas   sigma 1.0: pasa 1 de 14; sigma 2.0: ninguna
            #
            # O sea que la puerta la cruzan todas las buenas y ninguna movida.
            if (ordenes.abierta is not None and not estado.congelado and pan is None
                    and not estado.esperando_foco and juicio_fresco
                    and not getattr(camara, "congelada", False)
                    and cfg.interfaz.segundos_entre_fotos > 0):
                confirmado = automatica.mirar(
                    vivo, time.monotonic(), _resultado_del_diente(estado, cfg)
                )
                estado.progreso_automatico = automatica.progreso()
                if confirmado is not None:
                    orden = ordenes.abierta
                    # ESTE ES EL CAMINO QUE DEJABA LOS DIENTES SIN PRUEBA. Medido en la
                    # orden de verdad 5113546: 15 dientes anotados, los 15 con origen "la
                    # memoria, anotado solo" y CERO con foto. El operario no toca una
                    # tecla, asi que si la foto no se guarda aca no se guarda en ninguna
                    # parte. Se guarda `vivo`, que es el cuadro que cerro la racha y el
                    # mismo que se juzgo: la captura automatica solo corre con la imagen
                    # en vivo (not estado.congelado) y con el juicio fresco.
                    prueba, falta_prueba = prueba_del_diente(vivo)
                    anotado = orden.anotar(
                        estado=confirmado.estado,
                        parecido=confirmado.parecido,
                        aprobado=confirmado.aprobado,
                        origen=confirmado.origen,
                        nota=falta_prueba,
                        prueba=prueba,
                    )
                    sin_guardar = ordenes.guardar() is None
                    actualizar_orden()
                    if sin_guardar:
                        estado.mensaje = _NO_SE_GUARDO
                    elif anotado is None:
                        # 'o' abre el historial, ENTER retoma la orden y ahi la 's' es
                        # SIGUIENTE SIERRA. La 'o' sola no cambia de sierra.
                        estado.mensaje = ("SIERRA COMPLETA - para la que sigue: "
                                          "'o', ENTER y 's'")
                    else:
                        sierra = orden.sierra()
                        cuantos = sierra.dientes if sierra is not None else 0
                        estado.mensaje = (
                            f"DIENTE {anotado.numero} de {cuantos}: "
                            f"{confirmado.estado or 'sin decidir'}"
                        )
                        if falta_prueba:
                            estado.mensaje += " - SIN FOTO DE PRUEBA"
                    expira_mensaje = time.monotonic() + _SEGUNDOS_MENSAJE
            else:
                estado.progreso_automatico = ""

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
                accion = "ordenes"
            elif tecla == ord("g"):
                accion = "anotar"
            elif tecla in _TECLAS_DIGITO and len(estado.dientes) > 1:
                # Con varios dientes a la vista hay que decir cual se archiva: marcar los
                # dos con el mismo estado de un saque seria etiquetar mal el que no se
                # miro, y una foto mal etiquetada no se nota hasta que el modelo sale raro.
                indice = _TECLAS_DIGITO[tecla] - 1
                if indice < len(estado.dientes):
                    elegido_diente = indice
                    for numero, diente in enumerate(estado.dientes):
                        diente.elegido = numero == indice
                continue

            if accion is None:
                continue
            if accion == "salir":
                break
            if accion == "anotar":
                # ANOTAR EL DIENTE A MANO, sin archivar la foto. Son dos cosas distintas:
                # archivar le ensena al programa --y mete la foto en el dataset-- mientras
                # que anotar dice "este diente ya lo mire" y llena el informe. El taller
                # pidio poder hacer lo segundo sin lo primero.
                orden = ordenes.abierta
                if orden is None:
                    # LA 'o' NO ABRE UNA ORDEN: abre el historial, y ahi hay dos
                    # caminos distintos. ENTER retoma la orden elegida --y solo si no esta
                    # cerrada: una orden cerrada ya entrego su informe y el programa lo
                    # dice-- y la 'n' carga una nueva con el formulario. El cartel nombra
                    # los dos porque el que aprieta 'g' sin orden abierta puede querer
                    # cualquiera de los dos.
                    estado.mensaje = ("NO HAY ORDEN ABIERTA - apreta 'o': ahi se retoma "
                                      "una o se carga con 'n'")
                elif automatica.es_lo_ya_anotado(a_mostrar):
                    # LA MISMA FOTO NO SE ANOTA DOS VECES. El freno es por "la imagen no
                    # cambio" y no por "ya apretaste g": con la sierra en la mano cada
                    # cuadro es distinto y hay que poder anotar diente tras diente, pero
                    # con la imagen congelada --o con la pieza apoyada sin tocar-- la foto
                    # es la misma y ese diente ya esta contado. Medido en el bucle con una
                    # foto real del taller congelada y diez 'g': diez renglones, los diez
                    # con el mismo parecido y el mismo instante.
                    estado.mensaje = "ESTE DIENTE YA SE ANOTO - mueva la sierra"
                else:
                    # EL ORIGEN DICE DE DONDE SALIO EL VEREDICTO, no quien apreto la
                    # tecla: asi esta documentado en ordenes.Diente.origen. Decia "lo anoto
                    # el operario" al lado de un "APROBADO 84%" que habia puesto el
                    # programa, y en el .csv eso se lee igual que el renglon de 'a', donde
                    # el estado SI lo eligio el operario. Y cuando no hay veredicto en
                    # pantalla el renglon se iba en blanco --"sin decidir"-- contado como
                    # revisado y sin una nota que dijera por que.
                    resultado = _resultado_del_diente(
                        estado, cfg, quien="lo anoto el operario"
                    )
                    # La prueba, antes del renglon: mira prueba_del_diente(). Se
                    # guarda `a_mostrar`, que es el cuadro del que salio el veredicto
                    # --el congelado si el operario lo congelo-- y es el mismo contra el
                    # que se acaba de comprobar que este diente no estaba ya anotado.
                    prueba, falta_prueba = prueba_del_diente(a_mostrar)
                    # LOS DOS FALTANTES VAN JUNTOS EN LA NOTA. Que no hubiera veredicto en
                    # pantalla y que la prueba no se pudiera escribir son dos cosas
                    # distintas, y las dos tienen que poder quedar escritas en el mismo
                    # renglon: pisar una con la otra seria perder la mitad de lo que se
                    # sabe del diente justo en el papel que contesta un reclamo.
                    faltantes = [
                        texto for texto in (
                            "" if resultado else "no habia veredicto en pantalla",
                            falta_prueba,
                        ) if texto
                    ]
                    anotado = orden.anotar(
                        estado=resultado.estado if resultado else None,
                        parecido=resultado.parecido if resultado else None,
                        aprobado=resultado.aprobado if resultado else None,
                        origen=(
                            resultado.origen if resultado
                            else "lo anoto el operario, sin veredicto en pantalla"
                        ),
                        nota="; ".join(faltantes),
                        prueba=prueba,
                    )
                    sin_guardar = ordenes.guardar() is None
                    actualizar_orden()
                    # Igual que al archivar: el diente quedo anotado, asi que la captura
                    # automatica tiene que frenarse sobre el. Con reiniciar() sola --que
                    # solo olvida la racha-- un segundo despues lo anotaba otra vez.
                    automatica.ya_anotado(vivo)
                    if sin_guardar:
                        estado.mensaje = _NO_SE_GUARDO
                    elif anotado is None:
                        # 'o' abre el historial, ENTER retoma la orden y ahi la 's' es
                        # SIGUIENTE SIERRA. La 'o' sola no cambia de sierra.
                        estado.mensaje = ("SIERRA COMPLETA - para la que sigue: "
                                          "'o', ENTER y 's'")
                    else:
                        sierra = orden.sierra()
                        estado.mensaje = (
                            f"DIENTE {anotado.numero} de "
                            f"{sierra.dientes if sierra else '?'}: "
                            f"{anotado.estado or 'sin decidir'}"
                        )
                        if falta_prueba:
                            estado.mensaje += " - SIN FOTO DE PRUEBA"
                expira_mensaje = time.monotonic() + _SEGUNDOS_MENSAJE
                continue

            # "ordenes" es el boton nuevo de la barra de abajo y la tecla 'o'; "orden" se
            # deja aceptado porque es el nombre que tenia la accion hasta ahora y hace lo
            # mismo: las dos abren el historial.
            if accion in ("conceptos", "orden", "ordenes"):
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
                    # El mismo minimo con el que la memoria decide: esta pantalla es donde
                    # el operario ve cual concepto todavia no llega y cuantas fotos le
                    # faltan. Ver panel._texto_aciertos().
                    minimo_fotos=cfg.memoria.minimo_por_concepto,
                    orden=ordenes.abierta,
                    # LAS FILAS QUE VAN A ENTRAR, medidas antes de abrir el panel: el
                    # historial se dibuja al alto del lienzo del taller, que es el que se
                    # acaba de usar, y sin esto el PRIMER cuadro del historial paginaba con
                    # el tope de 8 mientras dibujaba 7 filas. Despues lo mantiene al dia
                    # _dibujar_ordenes, que es el que dibuja de verdad.
                    filas_por_pagina=panel.filas_de_ordenes(alto_lienzo),
                )
                automatica.reiniciar()
                if accion in ("orden", "ordenes"):
                    # LA 'o' Y EL BOTON ORDENES ABREN EL HISTORIAL, no el formulario. Antes
                    # caia derecho en "orden nueva" cuando no habia ninguna abierta, asi que
                    # las ordenes ya guardadas no se podian mirar desde el programa: para la
                    # orden de la semana pasada --la del reclamo que entra por telefono--
                    # habia que abrir el Explorador. De aca se entra a una con ENTER o se
                    # carga una nueva con 'n'.
                    _abrir_historial(pan, ordenes)
                continue

            if accion == "revisar":
                congelado = vivo.copy()
                estado.congelado = True
                juzgado = False
                automatica.reiniciar()
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
                automatica.reiniciar()
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
                # CAMBIO EL MODELO, ASI QUE HAY QUE VOLVER A JUZGAR. Con la imagen
                # congelada el veredicto se calcula una sola vez, y esa bandera dejaba la
                # cabecera nombrando el modelo nuevo con el porcentaje del anterior.
                # Reproducido con dos dientes a la vista y tres modelos: congelado, dos
                # 'm' movian la cabecera de mod0 a mod2 y los parecidos quedaban clavados
                # en (0.6823, 0.6173), mientras que en vivo las mismas teclas los movian a
                # (0.7258, 0.7333) y (0.6906, 0.7680). La bandera sigue frenando el
                # OJO, NO TAPA EL CASO 'a' Y DESPUES 'm': ahi la foto ya entro a la
                # memoria y volver a juzgar la encuentra a si misma. Medido congelado con
                # una foto archivada: el cartel pasa de '28% - la que mas se parece de las
                # fotos de modelo' a '20% - se parece a aprobada, la foto archivada mas
                # parecida'. No anota nada --con la imagen congelada la captura automatica
                # no corre-- y el cartel dice de donde sale el numero, pero el operario ve
                # moverse un numero que no tiene que ver con el modelo que eligio.
                juzgado = False
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
