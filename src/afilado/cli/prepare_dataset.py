"""Preparacion del dataset: extraccion de frames, division train/val y fusion del feedback.

Tres subcomandos que cubren el camino desde el video crudo hasta el data.yaml que
consume el entrenamiento:

  extraer   video -> frames utiles (descarta borrosos y casi duplicados)
  dividir   carpeta de imagenes+etiquetas -> images/{train,val} + labels/{train,val} + data.yaml
  fusionar  data/feedback -> dataset crudo (para corregir las cajas en Roboflow)

Por que se descartan frames al extraer: un video de 30 fps de una pieza quieta entrega
900 fotos casi identicas por cada 30 segundos. Etiquetar 900 clones cuesta lo mismo que
etiquetar 900 fotos distintas pero no aporta variedad; peor aun, infla el set y sesga la
validacion (si un clon cae en train y su gemelo en val, la metrica miente porque la red
esta puntuando una foto que ya vio). Se filtra por nitidez (varianza del Laplaciano: un
frame movido tiene pocos bordes y por lo tanto poca varianza) y por diferencia contra el
ultimo frame GUARDADO, no contra el ultimo leido: comparar contra el leido dejaria pasar
una deriva lenta que acumula cientos de imagenes practicamente iguales.

Solo depende de numpy, cv2, pyyaml y stdlib: se puede ejecutar sin torch ni ultralytics.
"""

from __future__ import annotations

import argparse
import logging
import random
import shutil
import sys
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import yaml

from ..config import cargar_config, ruta_absoluta

_registro = logging.getLogger("afilado.prepare_dataset")

EXTENSIONES_IMAGEN: tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
EXTENSIONES_VIDEO: tuple[str, ...] = (
    ".mp4",
    ".avi",
    ".mov",
    ".mkv",
    ".mpg",
    ".mpeg",
    ".wmv",
    ".m4v",
)

# Subcarpetas admitidas al buscar los pares imagen/etiqueta en el subcomando dividir.
_SUBCARPETAS_IMAGEN: tuple[str, ...] = ("images", "imagenes")
_SUBCARPETAS_ETIQUETA: tuple[str, ...] = ("labels", "etiquetas")


def _configurar_registro(verboso: bool) -> None:
    """Deja el log legible para el operario: sin ruido de modulo ni de nivel."""
    logging.basicConfig(
        level=logging.DEBUG if verboso else logging.INFO,
        format="%(message)s",
        stream=sys.stdout,
    )


def _escribir_imagen(destino: Path, imagen: np.ndarray, calidad: int) -> bool:
    """Guarda un JPG. Usa imencode+tofile porque cv2.imwrite no admite rutas no ASCII en Windows."""
    exito, buffer = cv2.imencode(".jpg", imagen, [int(cv2.IMWRITE_JPEG_QUALITY), int(calidad)])
    if not exito:
        return False
    buffer.tofile(str(destino))
    return True


def _listar_por_extension(carpeta: Path, extensiones: tuple[str, ...]) -> list[Path]:
    """Devuelve los archivos de la carpeta (no recursivo) con esas extensiones, ordenados."""
    if not carpeta.is_dir():
        return []
    encontrados = [
        ruta
        for ruta in carpeta.iterdir()
        if ruta.is_file() and ruta.suffix.lower() in extensiones
    ]
    return sorted(encontrados, key=lambda ruta: ruta.name.lower())


def _tiene_contenido(carpeta: Path) -> bool:
    """True si la carpeta existe y contiene algo."""
    return carpeta.is_dir() and any(carpeta.iterdir())


def _nitidez(gris: np.ndarray) -> float:
    """Varianza del Laplaciano: mide cuanta energia de borde hay. Bajo => frame movido."""
    return float(cv2.Laplacian(gris, cv2.CV_64F).var())


# ----------------------------------------------------------------------------- extraer


def _extraer_de_video(
    video: Path,
    salida: Path,
    cada: int,
    nitidez_minima: float,
    diferencia_minima: float,
    calidad: int,
    forzar: bool,
) -> dict[str, int]:
    """Extrae frames utiles de un video. Devuelve el conteo por motivo de decision."""
    conteo = {"leidos": 0, "guardados": 0, "borrosos": 0, "duplicados": 0, "existentes": 0}

    captura = cv2.VideoCapture(str(video))
    if not captura.isOpened():
        _registro.warning("  No se pudo abrir el video (codec no soportado?): %s", video.name)
        return conteo

    # Referencia de duplicados: el ultimo frame GUARDADO, no el ultimo leido.
    # Contra el ultimo leido, una deriva lenta pasaria el umbral en cada paso y
    # terminaria guardando cientos de imagenes casi identicas.
    gris_guardado: Optional[np.ndarray] = None
    indice = 0
    try:
        while True:
            exito, frame = captura.read()
            if not exito or frame is None:
                break
            conteo["leidos"] += 1
            indice_actual = indice
            indice += 1

            if indice_actual % cada != 0:
                continue

            gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            if _nitidez(gris) < nitidez_minima:
                conteo["borrosos"] += 1
                continue

            if gris_guardado is not None and gris_guardado.shape == gris.shape:
                diferencia = float(np.mean(cv2.absdiff(gris, gris_guardado)))
                if diferencia < diferencia_minima:
                    conteo["duplicados"] += 1
                    continue

            destino = salida / f"{video.stem}_{indice_actual:06d}.jpg"
            if destino.exists() and not forzar:
                conteo["existentes"] += 1
                gris_guardado = gris
                continue

            if not _escribir_imagen(destino, frame, calidad):
                _registro.warning("  No se pudo escribir %s", destino.name)
                continue

            gris_guardado = gris
            conteo["guardados"] += 1
    finally:
        captura.release()

    return conteo


def _comando_extraer(args: argparse.Namespace) -> int:
    videos = ruta_absoluta(args.videos)
    salida = ruta_absoluta(args.salida)

    if not videos.is_dir():
        _registro.error("La carpeta de videos no existe: %s", videos)
        return 1
    if args.cada < 1:
        _registro.error("--cada debe ser >= 1, se recibio %d.", args.cada)
        return 1

    archivos = _listar_por_extension(videos, EXTENSIONES_VIDEO)
    if not archivos:
        _registro.error(
            "No hay videos en %s. Extensiones reconocidas: %s",
            videos,
            ", ".join(EXTENSIONES_VIDEO),
        )
        return 1

    salida.mkdir(parents=True, exist_ok=True)

    total = {"leidos": 0, "guardados": 0, "borrosos": 0, "duplicados": 0, "existentes": 0}
    _registro.info("Extrayendo de %d video(s) hacia %s", len(archivos), salida)
    for video in archivos:
        _registro.info("- %s", video.name)
        conteo = _extraer_de_video(
            video=video,
            salida=salida,
            cada=args.cada,
            nitidez_minima=args.nitidez_minima,
            diferencia_minima=args.diferencia_minima,
            calidad=args.calidad_jpg,
            forzar=args.forzar,
        )
        for clave, valor in conteo.items():
            total[clave] += valor
        _registro.info(
            "  leidos %d | guardados %d | borrosos %d | duplicados %d",
            conteo["leidos"],
            conteo["guardados"],
            conteo["borrosos"],
            conteo["duplicados"],
        )

    _registro.info("")
    _registro.info("Frames leidos:      %d", total["leidos"])
    _registro.info("Frames guardados:   %d  -> %s", total["guardados"], salida)
    _registro.info("Descartados borrosos (nitidez < %.1f): %d", args.nitidez_minima, total["borrosos"])
    _registro.info(
        "Descartados duplicados (diferencia < %.1f): %d",
        args.diferencia_minima,
        total["duplicados"],
    )
    if total["existentes"]:
        _registro.warning(
            "Se respetaron %d archivo(s) ya existentes. Usa --forzar para sobrescribirlos.",
            total["existentes"],
        )
    if total["guardados"] == 0:
        _registro.warning(
            "No se guardo ningun frame. Baja --nitidez-minima o --diferencia-minima si el "
            "filtro quedo demasiado exigente para tu iluminacion."
        )
    return 0


# ----------------------------------------------------------------------------- dividir


def _buscar_etiqueta(imagen: Path, origen: Path) -> Optional[Path]:
    """Busca el .txt de una imagen: junto a ella o en la carpeta de etiquetas hermana."""
    candidatos = [imagen.with_suffix(".txt")]
    for nombre in _SUBCARPETAS_ETIQUETA:
        candidatos.append(origen / nombre / f"{imagen.stem}.txt")
        candidatos.append(imagen.parent.parent / nombre / f"{imagen.stem}.txt")
    for candidato in candidatos:
        if candidato.is_file():
            return candidato
    return None


def _recopilar_imagenes(origen: Path) -> list[Path]:
    """Reune las imagenes de la carpeta raiz y de sus subcarpetas de imagenes conocidas."""
    imagenes = _listar_por_extension(origen, EXTENSIONES_IMAGEN)
    for nombre in _SUBCARPETAS_IMAGEN:
        imagenes.extend(_listar_por_extension(origen / nombre, EXTENSIONES_IMAGEN))
    return imagenes


def _clases_de_etiqueta(etiqueta: Path) -> Optional[frozenset[int]]:
    """Lee la primera columna de cada linea del .txt. None si el archivo esta corrupto."""
    clases: set[int] = set()
    try:
        contenido = etiqueta.read_text(encoding="utf-8")
    except OSError:
        return None
    for linea in contenido.splitlines():
        partes = linea.split()
        if not partes:
            continue
        try:
            clases.add(int(float(partes[0])))
        except ValueError:
            return None
    return frozenset(clases)


def _clave_estrato(clases: Optional[frozenset[int]], etiqueta: Optional[Path]) -> str:
    """Etiqueta de estrato para el reparto: mismo contenido de clases => mismo grupo."""
    if etiqueta is None:
        return "sin_etiqueta"
    if clases is None:
        return "etiqueta_invalida"
    if not clases:
        return "fondo"
    return ",".join(str(c) for c in sorted(clases))


def _repartir(cantidad: int, fraccion_val: float) -> int:
    """Cuantos elementos de un estrato van a validacion, dejando siempre train no vacio."""
    if cantidad <= 1 or fraccion_val <= 0:
        return 0
    n_val = int(round(cantidad * float(fraccion_val)))
    return max(1, min(n_val, cantidad - 1))


def _escribir_data_yaml(salida: Path, clases: list[str]) -> Path:
    """Genera el data.yaml con rutas absolutas: ultralytics lo resuelve desde su propio cwd."""
    destino = salida / "data.yaml"
    contenido = {
        "path": str(salida),
        "train": str(salida / "images" / "train"),
        "val": str(salida / "images" / "val"),
        "nc": len(clases),
        "names": list(clases),
    }
    with open(destino, "w", encoding="utf-8") as manejador:
        yaml.safe_dump(contenido, manejador, allow_unicode=True, sort_keys=False)
    return destino


def _comando_dividir(args: argparse.Namespace) -> int:
    origen = ruta_absoluta(args.origen)
    salida = ruta_absoluta(args.salida)

    if not origen.is_dir():
        _registro.error("La carpeta de origen no existe: %s", origen)
        return 1
    if not 0.0 <= args.val < 1.0:
        _registro.error("--val debe estar en [0, 1), se recibio %s.", args.val)
        return 1

    try:
        clases = list(cargar_config(args.config).clases)
    except ValueError as error:
        _registro.error("Config invalido: %s", error)
        return 1

    imagenes = _recopilar_imagenes(origen)
    if not imagenes:
        _registro.error("No hay imagenes en %s (ni en sus subcarpetas images/ o imagenes/).", origen)
        return 1

    vistos: dict[str, Path] = {}
    duplicados: list[Path] = []
    unicas: list[Path] = []
    for imagen in imagenes:
        if imagen.stem in vistos:
            duplicados.append(imagen)
            continue
        vistos[imagen.stem] = imagen
        unicas.append(imagen)
    if duplicados:
        _registro.warning(
            "Se ignoraron %d imagen(es) con nombre repetido (el destino es plano y se pisarian): %s",
            len(duplicados),
            ", ".join(sorted(r.name for r in duplicados[:5])),
        )

    pares: list[tuple[Path, Optional[Path]]] = []
    estratos: dict[str, list[int]] = {}
    sin_etiqueta: list[Path] = []
    invalidas: list[Path] = []
    fuera_de_rango: set[int] = set()

    for imagen in unicas:
        etiqueta = _buscar_etiqueta(imagen, origen)
        ids = _clases_de_etiqueta(etiqueta) if etiqueta is not None else None
        if etiqueta is None:
            sin_etiqueta.append(imagen)
        elif ids is None:
            invalidas.append(etiqueta)
        else:
            fuera_de_rango.update(c for c in ids if c < 0 or c >= len(clases))
        indice = len(pares)
        pares.append((imagen, etiqueta))
        estratos.setdefault(_clave_estrato(ids, etiqueta), []).append(indice)

    # Etiquetas sin imagen: quedarian fuera del dataset en silencio.
    stems_imagen = set(vistos)
    huerfanas: list[Path] = []
    carpetas_etiqueta = [origen] + [origen / nombre for nombre in _SUBCARPETAS_ETIQUETA]
    for carpeta in carpetas_etiqueta:
        for etiqueta in _listar_por_extension(carpeta, (".txt",)):
            if etiqueta.stem not in stems_imagen:
                huerfanas.append(etiqueta)

    if sin_etiqueta:
        _registro.warning(
            "%d imagen(es) sin archivo de etiquetas: entraran al dataset como fondo (sin objetos). "
            "Ejemplos: %s",
            len(sin_etiqueta),
            ", ".join(r.name for r in sin_etiqueta[:5]),
        )
    if invalidas:
        _registro.warning(
            "%d etiqueta(s) ilegibles o con formato invalido: %s",
            len(invalidas),
            ", ".join(r.name for r in invalidas[:5]),
        )
    if huerfanas:
        _registro.warning(
            "%d etiqueta(s) huerfanas (sin imagen con el mismo nombre): %s",
            len(huerfanas),
            ", ".join(sorted(r.name for r in huerfanas[:5])),
        )
    if fuera_de_rango:
        _registro.warning(
            "Hay etiquetas con id de clase fuera de las %d clases del config (%s): %s. "
            "Corregilas antes de entrenar o YOLO fallara.",
            len(clases),
            ", ".join(clases),
            ", ".join(str(c) for c in sorted(fuera_de_rango)),
        )

    con_etiqueta = len(pares) - len(sin_etiqueta)
    hay_estratos = con_etiqueta > 0 and len(estratos) > 1
    aleatorio = random.Random(args.semilla)

    indices_val: set[int] = set()
    if hay_estratos:
        # Estratificar mantiene la proporcion de cada clase en val: con pocas fisuras,
        # un reparto ciego puede dejarlas todas en train y la metrica de fisura sale vacia.
        for clave in sorted(estratos):
            grupo = list(estratos[clave])
            aleatorio.shuffle(grupo)
            indices_val.update(grupo[: _repartir(len(grupo), args.val)])
    else:
        grupo = list(range(len(pares)))
        aleatorio.shuffle(grupo)
        indices_val.update(grupo[: _repartir(len(grupo), args.val)])

    destinos = [
        salida / "images" / "train",
        salida / "images" / "val",
        salida / "labels" / "train",
        salida / "labels" / "val",
    ]
    ocupadas = [carpeta for carpeta in destinos if _tiene_contenido(carpeta)]
    if (ocupadas or (salida / "data.yaml").is_file()) and not args.forzar:
        _registro.error(
            "El dataset de salida ya tiene contenido: %s. "
            "Un reparto nuevo mezclaria imagenes viejas de val en train y arruinaria la "
            "validacion. Usa --forzar para borrar el reparto anterior y rehacerlo.",
            salida,
        )
        return 1

    for carpeta in destinos:
        if carpeta.exists() and args.forzar:
            shutil.rmtree(carpeta)
        carpeta.mkdir(parents=True, exist_ok=True)

    copiadas = {"train": 0, "val": 0}
    etiquetas_copiadas = {"train": 0, "val": 0}
    for indice, (imagen, etiqueta) in enumerate(pares):
        parte = "val" if indice in indices_val else "train"
        shutil.copy2(imagen, salida / "images" / parte / imagen.name)
        copiadas[parte] += 1
        if etiqueta is not None:
            shutil.copy2(etiqueta, salida / "labels" / parte / f"{imagen.stem}.txt")
            etiquetas_copiadas[parte] += 1

    ruta_yaml = _escribir_data_yaml(salida, clases)

    _registro.info("")
    _registro.info("Reparto %s (semilla %d)", "estratificado por clase" if hay_estratos else "simple", args.semilla)
    _registro.info("  train: %d imagenes (%d con etiqueta)", copiadas["train"], etiquetas_copiadas["train"])
    _registro.info("  val:   %d imagenes (%d con etiqueta)", copiadas["val"], etiquetas_copiadas["val"])
    _registro.info("data.yaml: %s", ruta_yaml)
    _registro.info("Entrenar con: python -m afilado.cli.train --datos \"%s\"", ruta_yaml)
    return 0


# ---------------------------------------------------------------------------- fusionar


def _comando_fusionar(args: argparse.Namespace) -> int:
    feedback = ruta_absoluta(args.feedback)
    salida = ruta_absoluta(args.salida)

    if not feedback.is_dir():
        _registro.error("La carpeta de feedback no existe: %s", feedback)
        return 1

    dias = sorted(carpeta for carpeta in feedback.iterdir() if carpeta.is_dir())
    if not dias:
        _registro.error("No hay capturas en %s: el bucle de feedback todavia no guardo nada.", feedback)
        return 1

    salida.mkdir(parents=True, exist_ok=True)

    copiadas = 0
    sin_etiqueta = 0
    existentes = 0
    for dia in dias:
        carpeta_imagenes = dia / "imagenes"
        carpeta_etiquetas = dia / "etiquetas"
        imagenes = _listar_por_extension(carpeta_imagenes, EXTENSIONES_IMAGEN)
        if not imagenes:
            continue
        for imagen in imagenes:
            destino_imagen = salida / imagen.name
            if destino_imagen.exists() and not args.forzar:
                existentes += 1
                continue
            shutil.copy2(imagen, destino_imagen)
            copiadas += 1
            # El pre-etiquetado viaja junto a la imagen: en Roboflow solo hay que corregir
            # las cajas que la IA erro, no dibujarlas de cero.
            etiqueta = carpeta_etiquetas / f"{imagen.stem}.txt"
            if etiqueta.is_file():
                shutil.copy2(etiqueta, salida / f"{imagen.stem}.txt")
            else:
                sin_etiqueta += 1
        _registro.info("- %s: %d imagen(es)", dia.name, len(imagenes))

    _registro.info("")
    _registro.info("Copiadas %d imagen(es) con su pre-etiquetado a %s", copiadas, salida)
    if sin_etiqueta:
        _registro.warning(
            "%d imagen(es) sin pre-etiquetado: habra que dibujar sus cajas desde cero.",
            sin_etiqueta,
        )
    if existentes:
        _registro.warning(
            "Se omitieron %d imagen(es) ya presentes en el destino. Usa --forzar para sobrescribirlas.",
            existentes,
        )
    if copiadas == 0 and existentes == 0:
        _registro.warning("No habia imagenes en %s/<fecha>/imagenes.", feedback)
        return 1
    _registro.info(
        "Siguiente paso: subir %s a Roboflow, CORREGIR las cajas erradas y exportar en formato YOLO.",
        salida,
    )
    return 0


# ------------------------------------------------------------------------------- cli


def _construir_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="prepare_dataset",
        description="Prepara el dataset de afilado: extraer frames, dividir train/val y fusionar el feedback.",
    )
    parser.add_argument("--verboso", action="store_true", help="Muestra el detalle de cada decision.")
    subparsers = parser.add_subparsers(dest="subcomando", required=True)

    extraer = subparsers.add_parser(
        "extraer",
        help="Extrae frames utiles de videos, descartando borrosos y casi duplicados.",
    )
    extraer.add_argument("--videos", required=True, help="Carpeta con los videos de entrada.")
    extraer.add_argument("--salida", default="data/dataset/crudo", help="Carpeta destino de los frames.")
    extraer.add_argument("--cada", type=int, default=15, help="Analiza uno de cada N frames.")
    extraer.add_argument(
        "--nitidez-minima",
        type=float,
        default=60.0,
        dest="nitidez_minima",
        help="Varianza minima del Laplaciano; por debajo el frame se considera movido.",
    )
    extraer.add_argument(
        "--diferencia-minima",
        type=float,
        default=8.0,
        dest="diferencia_minima",
        help="Diferencia media en gris contra el ultimo frame guardado; por debajo es un duplicado.",
    )
    extraer.add_argument(
        "--calidad-jpg", type=int, default=95, dest="calidad_jpg", help="Calidad JPG de los frames (1-100)."
    )
    extraer.add_argument("--forzar", action="store_true", help="Sobrescribe los frames ya existentes.")
    extraer.set_defaults(funcion=_comando_extraer)

    dividir = subparsers.add_parser(
        "dividir",
        help="Arma images/{train,val}, labels/{train,val} y data.yaml respetando los pares.",
    )
    dividir.add_argument("--origen", required=True, help="Carpeta con las imagenes y sus etiquetas.")
    dividir.add_argument("--salida", default="data/dataset", help="Carpeta raiz del dataset resultante.")
    dividir.add_argument("--val", type=float, default=0.2, help="Fraccion de imagenes para validacion.")
    dividir.add_argument("--semilla", type=int, default=42, help="Semilla del reparto (reproducible).")
    dividir.add_argument("--config", default=None, help="Ruta del config del que se leen las clases.")
    dividir.add_argument("--forzar", action="store_true", help="Borra el reparto anterior y lo rehace.")
    dividir.set_defaults(funcion=_comando_dividir)

    fusionar = subparsers.add_parser(
        "fusionar",
        help="Copia las capturas del bucle de feedback (imagen + pre-etiquetado) al dataset crudo.",
    )
    fusionar.add_argument("--feedback", default="data/feedback", help="Carpeta raiz del feedback.")
    fusionar.add_argument("--salida", default="data/dataset/crudo", help="Carpeta destino.")
    fusionar.add_argument("--forzar", action="store_true", help="Sobrescribe las capturas ya copiadas.")
    fusionar.set_defaults(funcion=_comando_fusionar)

    return parser


def main(argv: Optional[list[str]] = None) -> int:
    """Punto de entrada de la CLI de preparacion del dataset. Devuelve 0 en exito."""
    parser = _construir_parser()
    args = parser.parse_args(argv)
    _configurar_registro(args.verboso)
    try:
        return int(args.funcion(args))
    except KeyboardInterrupt:
        _registro.warning("Interrumpido por el operario.")
        return 1
    except OSError as error:
        _registro.error("Error de disco: %s", error)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
