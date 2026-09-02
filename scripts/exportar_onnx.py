"""Exporta el modelo entrenado (.pt) a ONNX, para la PC del taller.

PARA QUE SIRVE
--------------
El .pt solo se puede correr con torch + ultralytics: 1.2 GB de entorno y 440 MB de RAM.
El mismo modelo exportado a .onnx corre con opencv, que este proyecto ya necesita si o si
por el modulo aruco. Medido en un i5-12400 sin placa de video:

    RAM del proceso      440 MB  ->  80 MB de pico (48 MB en regimen)
    entorno en disco     1.2 GB  ->  ~250 MB
    cargar el modelo     1.2 s   ->  0.05 s
    inferencia a 224 px  12 ms   ->  5.8 ms (cv2.dnn) o 1.8 ms (onnxruntime)

O sea: la PC del taller no necesita instalar torch NUNCA. Se entrena en la maquina buena,
se corre este script, y al taller viajan dos archivos chiquitos: el .onnx y su .json.

COMO SE USA
-----------
    python scripts/exportar_onnx.py

Eso toma models/afilado_best.pt y escribe models/afilado_best.onnx (mas el .onnx.json con
las clases). Despues, en configs/config.yaml del taller:

    detector:
      pesos: models/afilado_best.onnx

y listo: run_live lo levanta solo, sin torch.

QUE VERIFICA
------------
Antes de dar el export por bueno, el script corre el .pt y el .onnx sobre las MISMAS fotos
del dataset y compara los veredictos. Si no coinciden, termina con error y no deja el
archivo publicado. Un export silenciosamente distinto del original es exactamente el tipo
de fallo que en un taller no se detecta hasta que ya se tiro una sierra buena.

DEPENDENCIA
-----------
Necesita el paquete 'onnx' (solo en la maquina de entrenamiento, ~20 MB):

    pip install onnx
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

RAIZ = Path(__file__).resolve().parents[1]
if str(RAIZ / "src") not in sys.path:
    sys.path.insert(0, str(RAIZ / "src"))

from afilado.config import cargar_config, ruta_absoluta  # noqa: E402

_PESOS_POR_DEFECTO = "models/afilado_best.pt"
# opset 12 a proposito: es el ultimo que el importador ONNX de OpenCV lee sin problemas.
# Un opset mas nuevo puede exportar bien y despues fallar solo en la PC del taller, que es
# justo donde no hay como diagnosticarlo.
_OPSET = 12
# Umbral de tolerancia entre .pt y .onnx. La diferencia real medida es ~0.001; 0.05 deja
# margen para el ruido del interpolador sin dejar pasar un export roto.
_TOLERANCIA = 0.05


def _construir_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="exportar_onnx",
        description=(
            "Exporta el modelo de clasificacion .pt a .onnx para correrlo sin torch, "
            "y verifica que el veredicto no cambie."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--pesos",
        default=_PESOS_POR_DEFECTO,
        help="Checkpoint .pt entrenado que se quiere exportar.",
    )
    parser.add_argument(
        "--salida",
        default=None,
        help="Ruta del .onnx a escribir. Por defecto, el mismo nombre que --pesos.",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=0,
        help="Lado de entrada en px. 0 = el que diga detector.imgsz de configs/config.yaml.",
    )
    parser.add_argument(
        "--fotos",
        default=None,
        help=(
            "Carpeta con imagenes para verificar el export. Por defecto usa "
            "data/clasificacion/val y, si no existe, data/clasificacion/train."
        ),
    )
    parser.add_argument(
        "--max-fotos",
        type=int,
        default=40,
        help="Cuantas fotos como maximo se usan en la verificacion.",
    )
    parser.add_argument(
        "--sin-verificar",
        action="store_true",
        help="Exporta sin comparar contra el .pt. No recomendado: es la unica red de seguridad.",
    )
    return parser


def _buscar_fotos(carpeta: Optional[str], maximo: int) -> list[Path]:
    """Junta las fotos con las que se va a comparar .pt contra .onnx."""
    candidatas: list[Path] = []
    if carpeta:
        candidatas = [ruta_absoluta(carpeta)]
    else:
        candidatas = [
            ruta_absoluta("data/clasificacion/val"),
            ruta_absoluta("data/clasificacion/train"),
            ruta_absoluta("data/coleccion"),
        ]
    for base in candidatas:
        if not base.is_dir():
            continue
        fotos = sorted(
            p
            for p in base.rglob("*")
            if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
        )
        if fotos:
            return fotos[:maximo]
    return []


def _verificar(
    modelo: Any, ruta_onnx: Path, clases: list[str], lado: int, fotos: list[Path]
) -> bool:
    """Compara el veredicto del .pt contra el del .onnx TAL COMO lo va a correr el taller.

    No reimplementa nada: instancia el mismo DetectorClasificadorOnnx que usa run_live, sobre
    el archivo ya publicado y con su .json al lado. Asi lo que se valida es la cadena completa
    que se entrega --preprocesado, metadatos y runtime incluidos--, y no una aproximacion que
    podria coincidir aca y fallar alla.
    """
    import cv2

    from afilado.config import DetectorConfig
    from afilado.detector import DetectorClasificadorOnnx

    clasificador = DetectorClasificadorOnnx(
        DetectorConfig(pesos=str(ruta_onnx), tarea="classify", imgsz=lado), clases
    )
    if not clasificador.disponible:
        print("ERROR: el .onnx exportado no se pudo abrir para verificarlo.")
        return False

    desacuerdos: list[str] = []
    peor_diferencia = 0.0
    comparadas = 0

    for foto in fotos:
        imagen = cv2.imread(str(foto))
        if imagen is None:
            continue
        referencia = modelo.predict(imagen, imgsz=lado, device="cpu", verbose=False)[0]
        clase_pt = str(referencia.names[int(referencia.probs.top1)])
        confianza_pt = float(referencia.probs.top1conf)

        veredicto = clasificador.clasificar(imagen)
        if veredicto is None:
            print(f"ERROR: el .onnx no pudo clasificar {foto.name}.")
            return False

        comparadas += 1
        peor_diferencia = max(peor_diferencia, abs(veredicto.confianza - confianza_pt))
        if veredicto.clase != clase_pt:
            desacuerdos.append(f"{foto.name} ({clase_pt} -> {veredicto.clase})")

    if comparadas == 0:
        print("AVISO: no se pudo leer ninguna foto; el export queda SIN verificar.")
        return True

    print(f"Verificacion: {comparadas} fotos, mismo camino que corre run_live")
    print(f"  veredictos distintos:          {len(desacuerdos)}")
    print(f"  peor diferencia de confianza:  {peor_diferencia:.6f} (tolerancia {_TOLERANCIA})")
    if desacuerdos:
        print("  en desacuerdo: " + ", ".join(desacuerdos[:8]))
        return False
    if peor_diferencia > _TOLERANCIA:
        print("  las clases coinciden pero la confianza se corrio demasiado.")
        return False
    return True


def _reponer(resguardo: Optional[Path], destino: Path) -> None:
    """Devuelve a su lugar el .onnx que se habia apartado antes de exportar."""
    if resguardo is None or not resguardo.is_file():
        return
    destino.unlink(missing_ok=True)
    resguardo.rename(destino)


def _ruta_para_config(ruta: Path) -> str:
    """Devuelve la ruta como la escribiria uno en el config: relativa al repo si esta
    adentro, absoluta si no. Exportar a un pendrive para llevarlo al taller es un uso
    normal, y ahi no hay ninguna ruta relativa que tenga sentido."""
    try:
        return ruta.relative_to(RAIZ).as_posix()
    except ValueError:
        return ruta.as_posix()


def main(argv: Optional[list[str]] = None) -> int:
    args = _construir_parser().parse_args(argv)

    ruta_pt = ruta_absoluta(args.pesos)
    if not ruta_pt.is_file():
        print(f"ERROR: no existe el modelo '{ruta_pt}'.")
        print("Entrena primero:  python -m afilado.cli.train --datos data/clasificacion")
        return 2

    try:
        cfg = cargar_config(None)
    except (ValueError, OSError) as error:
        print(f"ERROR: no se pudo leer configs/config.yaml: {error}")
        return 2

    lado = args.imgsz or int(cfg.detector.imgsz)
    ruta_onnx = ruta_absoluta(args.salida) if args.salida else ruta_pt.with_suffix(".onnx")
    # crear_clasificador despacha POR EXTENSION. Un archivo que no termine en .onnx se
    # mandaria al camino de ultralytics y, en una PC sin torch, el programa entraria en modo
    # recoleccion sin explicar por que. Mas vale frenar aca.
    if ruta_onnx.suffix.lower() != ".onnx":
        print(f"ERROR: --salida tiene que terminar en .onnx, se recibio '{ruta_onnx.name}'.")
        print("       El programa elige como correr el modelo por la extension del archivo.")
        return 2

    try:
        from ultralytics import YOLO
    except ImportError:
        print("ERROR: falta 'ultralytics'. Este script corre en la PC de ENTRENAMIENTO.")
        return 3

    modelo = YOLO(str(ruta_pt))
    nombres = getattr(modelo, "names", None)
    if isinstance(nombres, dict):
        clases = [str(nombres[k]) for k in sorted(nombres)]
    elif isinstance(nombres, (list, tuple)):
        clases = [str(v) for v in nombres]
    else:
        clases = list(cfg.clases)

    print(f"Modelo:  {ruta_pt}")
    print(f"Salida:  {ruta_onnx}")
    print(f"imgsz:   {lado}   opset: {_OPSET}")
    print(f"Clases:  {', '.join(clases)}")
    if clases != list(cfg.clases):
        print("AVISO: las clases del modelo NO son las de configs/config.yaml")
        print(f"       config: {', '.join(cfg.clases)}")
        print("       manda el modelo; el .onnx.json va a llevar las del modelo.")

    # ultralytics escribe el .onnx al lado del .pt y no deja elegir el destino. Si --salida
    # apunta a otro lado (llevarlo en un pendrive al taller es el caso tipico), ese archivo
    # intermedio pisaria un .onnx ya existente y despues se lo llevaria al mover. Se lo
    # aparta antes y se lo repone al final: nadie deberia quedarse sin el modelo del taller
    # por exportar una copia a otra carpeta.
    intermedio = ruta_pt.with_suffix(".onnx")
    resguardo: Optional[Path] = None
    if intermedio.is_file() and intermedio.resolve() != ruta_onnx.resolve():
        resguardo = Path(str(intermedio) + ".previo")
        resguardo.unlink(missing_ok=True)
        intermedio.rename(resguardo)

    try:
        generado = Path(
            modelo.export(format="onnx", imgsz=lado, opset=_OPSET, simplify=False)
        )
    except Exception as error:
        _reponer(resguardo, intermedio)
        print(f"ERROR durante el export: {error}")
        print("Si se queja por el paquete 'onnx', instalalo con:  pip install onnx")
        return 4

    ruta_onnx.parent.mkdir(parents=True, exist_ok=True)
    if generado.resolve() != ruta_onnx.resolve():
        shutil.move(str(generado), str(ruta_onnx))
    # Recien ahora, con el generado ya en su destino, se repone el que estaba.
    _reponer(resguardo, intermedio)

    metadatos = {
        "clases": clases,
        "imgsz": lado,
        "tarea": "classify",
        "origen": ruta_pt.name,
        "exportado": datetime.now().isoformat(timespec="seconds"),
        "opset": _OPSET,
    }
    ruta_json = Path(str(ruta_onnx) + ".json")
    with open(ruta_json, "w", encoding="utf-8") as manejador:
        json.dump(metadatos, manejador, ensure_ascii=False, indent=2)

    if not args.sin_verificar:
        fotos = _buscar_fotos(args.fotos, args.max_fotos)
        if not fotos and args.fotos:
            # Si el operador pidio una carpeta concreta y ahi no hay nada, casi seguro se
            # equivoco de ruta. Saltear la verificacion en silencio seria dejarlo creer que
            # el export quedo validado.
            print(f"ERROR: no se encontraron imagenes en '{args.fotos}'.")
            print("       Corregi la ruta, o usa --sin-verificar si de verdad no queres validar.")
            ruta_onnx.unlink(missing_ok=True)
            ruta_json.unlink(missing_ok=True)
            return 2
        if not fotos:
            print("AVISO: no se encontraron fotos para verificar; el export queda sin validar.")
        elif not _verificar(modelo, ruta_onnx, clases, lado, fotos):
            print()
            print("EXPORT RECHAZADO: el .onnx no da el mismo veredicto que el .pt.")
            print("Se borraron los archivos generados para que nadie los lleve al taller.")
            print("Revisa que --imgsz sea el mismo con el que se entreno.")
            ruta_onnx.unlink(missing_ok=True)
            ruta_json.unlink(missing_ok=True)
            return 5

    tamano = ruta_onnx.stat().st_size / 2**20
    print()
    print(f"Listo: {ruta_onnx}  ({tamano:.1f} MB)")
    print(f"       {ruta_json.name}  <- clases y tamano de entrada; viaja SIEMPRE con el .onnx")
    print()
    print("Para usarlo en el taller, en configs/config.yaml:")
    print("    detector:")
    print(f"      pesos: {_ruta_para_config(ruta_onnx)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
