"""Arma un dataset de CLASIFICACION (aprobada/rechazada/...) a partir de carpetas de fotos.

Cada clase es una carpeta de fotos: vos separas las sierras buenas de las que no sirven en
carpetas distintas, y este script las convierte en la estructura que YOLO necesita para
entrenar, aplicando el mismo filtro de nitidez y deduplicacion que el resto del sistema.

Estructura que genera:
    <salida>/train/<clase>/*.jpg
    <salida>/val/<clase>/*.jpg

Ejemplo (las dos tandas de la primera prueba):
    python scripts/construir_clasificacion.py ^
        --clase aprobada  "C:/ruta/tanda_sierras_buenas" ^
        --clase rechazada "C:/ruta/tanda_no_afilables"

Despues:
    python -m afilado.cli.train --datos data/clasificacion --epocas 100
"""
from __future__ import annotations

import argparse
import random
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

_EXTENSIONES = (".jpg", ".jpeg", ".png", ".bmp")


def _nitidez(imagen: np.ndarray) -> float:
    """Varianza del Laplaciano: baja = foto movida/desenfocada."""
    return float(cv2.Laplacian(cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var())


def _firma(imagen: np.ndarray) -> np.ndarray:
    """Miniatura en gris para comparar cuadros casi identicos."""
    return cv2.resize(cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY), (64, 114))


def _fotos(carpeta: Path) -> list[Path]:
    return sorted(f for f in carpeta.iterdir() if f.suffix.lower() in _EXTENSIONES)


def _construir_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="construir_clasificacion",
        description="Convierte carpetas de fotos por clase en un dataset de clasificacion YOLO.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--clase",
        action="append",
        nargs=2,
        metavar=("NOMBRE", "CARPETA"),
        required=True,
        help="Nombre de la clase y carpeta con sus fotos. Repetir por cada clase.",
    )
    parser.add_argument("--salida", default="data/clasificacion", help="Carpeta de salida.")
    parser.add_argument("--val", type=float, default=0.2, help="Fraccion para validacion (0..1).")
    parser.add_argument("--semilla", type=int, default=42, help="Semilla del reparto (reproducible).")
    parser.add_argument(
        "--nitidez-minima",
        type=float,
        default=60.0,
        help="Descarta fotos con varianza del Laplaciano por debajo de este valor.",
    )
    parser.add_argument(
        "--dedup",
        type=float,
        default=6.0,
        help="Diferencia media minima para considerar dos fotos distintas (0 = no deduplicar).",
    )
    parser.add_argument(
        "--forzar",
        action="store_true",
        help="Sobrescribe la carpeta de salida si ya existe.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _construir_parser().parse_args(argv)

    if not 0.0 < args.val < 1.0:
        print("ERROR: --val debe estar entre 0 y 1 (exclusivo).")
        return 2

    salida = Path(args.salida)
    if salida.exists():
        if not args.forzar:
            print(f"ERROR: '{salida}' ya existe. Usa --forzar para sobrescribirla.")
            return 2
        shutil.rmtree(salida)

    nombres = [c[0] for c in args.clase]
    if len(set(nombres)) != len(nombres):
        print("ERROR: hay nombres de clase repetidos.")
        return 2

    rng = random.Random(args.semilla)
    total_train = total_val = 0
    print("%-14s %6s %8s %7s %7s %5s" % ("clase", "fotos", "nitidas", "unicas", "train", "val"))
    for nombre, carpeta_str in args.clase:
        carpeta = Path(carpeta_str)
        if not carpeta.is_dir():
            print(f"ERROR: la carpeta de la clase '{nombre}' no existe: {carpeta}")
            return 2

        nitidas: list[np.ndarray] = []
        total = 0
        for f in _fotos(carpeta):
            total += 1
            im = cv2.imread(str(f))
            if im is None or _nitidez(im) < args.nitidez_minima:
                continue
            nitidas.append(im)

        # Deduplicar contra las ya aceptadas
        unicas: list[np.ndarray] = []
        for im in nitidas:
            s = _firma(im)
            if args.dedup <= 0 or all(np.mean(cv2.absdiff(s, _firma(u))) > args.dedup for u in unicas):
                unicas.append(im)

        if not unicas:
            print(f"AVISO: la clase '{nombre}' se quedo sin fotos utiles (todas movidas o duplicadas).")
            continue

        indices = list(range(len(unicas)))
        rng.shuffle(indices)
        n_val = max(1, round(len(unicas) * args.val)) if len(unicas) > 1 else 0
        val_idx = set(indices[:n_val])

        n_tr = n_va = 0
        for i, im in enumerate(unicas):
            sub = "val" if i in val_idx else "train"
            destino = salida / sub / nombre
            destino.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(destino / f"{nombre}_{i + 1:03d}.jpg"), im, [cv2.IMWRITE_JPEG_QUALITY, 95])
            if sub == "val":
                n_va += 1
            else:
                n_tr += 1
        total_train += n_tr
        total_val += n_va
        print("%-14s %6d %8d %7d %7d %5d" % (nombre, total, len(nitidas), len(unicas), n_tr, n_va))

    if total_train == 0:
        print("ERROR: no quedo ninguna imagen de entrenamiento.")
        return 1

    print(f"\nDataset en: {salida}")
    print(f"TOTAL train={total_train}  val={total_val}")
    print(f"Entrenar con: python -m afilado.cli.train --datos {salida} --epocas 100")
    if total_val < len(nombres) * 5:
        print("AVISO: hay muy pocas fotos. Para un modelo que sirva de verdad se necesitan\n"
              "       cientos por clase, con la misma luz y encuadre que en produccion.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
