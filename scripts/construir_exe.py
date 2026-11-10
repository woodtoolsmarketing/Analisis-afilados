"""Arma AnalisisAfilado.exe: el programa del taller en un solo archivo.

PARA QUE
--------
La PC del taller no instala nada. Ni Python, ni librerias, ni este repositorio: se le copia
un unico .exe y se hace doble clic. Todo --el interprete, OpenCV, el modelo y el config--
viaja adentro.

COMO SE USA (en la maquina de desarrollo)
-----------------------------------------
    python scripts/construir_exe.py

Conviene armarlo desde un entorno LIVIANO (el de requirements-taller.txt). PyInstaller mete
en el .exe todo lo que encuentra importable: si lo corres desde el entorno completo, te vas a
llevar torch adentro y el ejecutable va a pesar mas de 2 GB. El script avisa si detecta eso.

QUE QUEDA AFUERA DEL .EXE, A PROPOSITO
--------------------------------------
Lo que el programa ESCRIBE y lo que puede cambiar sin rearmar nada:

    <carpeta del .exe>\\data\\coleccion\\      las fotos que archiva el operario
    <carpeta del .exe>\\models\\               el modelo .onnx, cuando llegue
    <carpeta del .exe>\\configs\\config.yaml   opcional: si existe, pisa al de adentro

Eso NO es un capricho: en modo onefile, lo que viaja adentro se descomprime en una carpeta
temporal que Windows borra al cerrar el programa. Si las fotos se guardaran ahi, cada sesion
de recoleccion se perderia. Por eso config.raiz_repo() apunta a la carpeta del ejecutable.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Optional

RAIZ = Path(__file__).resolve().parents[1]

_NOMBRE = "AnalisisAfilado"

# Paquetes que no tienen nada que hacer en el taller. Se excluyen explicitamente y no solo
# por no estar instalados: si alguien arma el .exe desde el entorno de entrenamiento, sin
# esto se llevaria torch (533 MB) adentro sin darse cuenta.
_EXCLUIR = (
    "torch",
    "torchvision",
    "ultralytics",
    "matplotlib",
    "scipy",
    "pandas",
    "polars",
    "sympy",
    "networkx",
    # PIL NO va aca: Pillow es lo que dibuja la tipografia de la pantalla. Excluirlo
    # dejaba el .exe cayendo a las fuentes Hershey, que es lo que se quiso evitar.
    "tkinter",
    "pytest",
    "IPython",
)

# El modelo del taller es el .onnx, no el .pt: alla no hay torch para leer un checkpoint.
_PESOS_TALLER = "models/afilado_best.onnx"


def _construir_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="construir_exe",
        description="Arma el ejecutable del taller en un solo archivo.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--salida",
        default=str(RAIZ / "dist"),
        help="Carpeta donde dejar el .exe terminado.",
    )
    parser.add_argument(
        "--nombre",
        default=_NOMBRE,
        help="Nombre del ejecutable, sin la extension.",
    )
    parser.add_argument(
        "--modelo",
        default=None,
        help=(
            "Ruta a un .onnx para meter DENTRO del .exe. Por defecto no se incluye ninguno: "
            "el modelo se copia despues a models/ al lado del ejecutable."
        ),
    )
    parser.add_argument(
        "--carpeta",
        action="store_true",
        help=(
            "Arma una carpeta en vez de un solo archivo. Arranca mucho mas rapido (no "
            "descomprime en cada corrida), pero son cientos de archivos en vez de uno."
        ),
    )
    return parser


def _avisar_si_el_entorno_es_pesado() -> None:
    """Un .exe armado desde el entorno de entrenamiento se lleva torch adentro."""
    import importlib.util

    pesados = [m for m in ("torch", "ultralytics") if importlib.util.find_spec(m)]
    if not pesados:
        return
    print()
    print("AVISO: este entorno tiene " + ", ".join(pesados) + ".")
    print("       Se excluyen del .exe explicitamente, pero PyInstaller puede tardar mucho")
    print("       mas y el resultado suele salir mas grande. Para el .exe mas chico y")
    print("       predecible, arma desde un entorno instalado con requirements-taller.txt.")
    print()


def _config_del_taller(destino: Path) -> Path:
    """Escribe el config que viaja dentro del .exe, apuntando al modelo .onnx.

    Se genera aca en vez de meter el del repo tal cual porque el repo apunta al .pt, que es
    lo correcto en la maquina de entrenamiento y no se puede leer en el taller.
    """
    original = (RAIZ / "configs" / "config.yaml").read_text(encoding="utf-8")
    if "  pesos: models/afilado_best.pt" not in original:
        print("AVISO: no se encontro la linea de 'pesos' esperada en configs/config.yaml;")
        print("       el config se empaqueta tal cual esta.")
        texto = original
    else:
        texto = original.replace(
            "  pesos: models/afilado_best.pt", f"  pesos: {_PESOS_TALLER}"
        )
    destino.parent.mkdir(parents=True, exist_ok=True)
    destino.write_text(texto, encoding="utf-8")
    return destino


def _comando(args: argparse.Namespace, temporal: Path, modelo: Optional[Path]) -> list[str]:
    separador = ";"  # PyInstaller usa ; en Windows y : en el resto
    if not sys.platform.startswith("win"):
        separador = ":"

    comando = [
        sys.executable,
        "-m",
        "PyInstaller",
        "--noconfirm",
        "--clean",
        "--onedir" if args.carpeta else "--onefile",
        "--console",
        "--name",
        args.nombre,
        "--paths",
        str(RAIZ / "src"),
        "--distpath",
        str(Path(args.salida).resolve()),
        "--workpath",
        str(temporal / "build"),
        "--specpath",
        str(temporal),
        "--add-data",
        f"{temporal / 'configs' / 'config.yaml'}{separador}configs",
    ]
    if modelo is not None:
        comando += ["--add-data", f"{modelo}{separador}models"]
    for paquete in _EXCLUIR:
        comando += ["--exclude-module", paquete]
    comando.append(str(RAIZ / "scripts" / "lanzador_taller.py"))
    return comando


def main(argv: Optional[list[str]] = None) -> int:
    args = _construir_parser().parse_args(argv)

    try:
        import PyInstaller  # noqa: F401
    except ImportError:
        print("ERROR: falta PyInstaller. Instalalo con:  pip install pyinstaller")
        return 3

    modelo: Optional[Path] = None
    if args.modelo:
        modelo = Path(args.modelo).resolve()
        if not modelo.is_file():
            print(f"ERROR: no existe el modelo '{modelo}'.")
            return 2
        if modelo.suffix.lower() != ".onnx":
            print("ERROR: el modelo que va adentro del .exe tiene que ser un .onnx.")
            print("       El .pt necesita torch, que no viaja en el ejecutable.")
            return 2

    _avisar_si_el_entorno_es_pesado()

    with tempfile.TemporaryDirectory(prefix="afilado-exe-") as tmp:
        temporal = Path(tmp)
        _config_del_taller(temporal / "configs" / "config.yaml")

        print(f"Armando {args.nombre}.exe ({'carpeta' if args.carpeta else 'un solo archivo'})")
        print(f"  entrada:  scripts/lanzador_taller.py")
        print(f"  config:   configs/config.yaml  (con pesos: {_PESOS_TALLER})")
        print(f"  modelo:   {modelo.name if modelo else 'no se incluye; va en models/ al lado del .exe'}")
        print(f"  salida:   {args.salida}")
        print("  Esto tarda unos minutos.")
        print()

        resultado = subprocess.run(_comando(args, temporal, modelo))
        if resultado.returncode != 0:
            print()
            print("ERROR: PyInstaller fallo. El detalle esta mas arriba.")
            return 4

    if args.carpeta:
        destino = Path(args.salida).resolve() / args.nombre / f"{args.nombre}.exe"
    else:
        destino = Path(args.salida).resolve() / f"{args.nombre}.exe"

    if not destino.is_file():
        print(f"ERROR: PyInstaller termino bien pero no aparecio '{destino}'.")
        return 4

    print()
    print(f"Listo: {destino}  ({destino.stat().st_size / 2**20:.0f} MB)")
    print()
    print("En la PC del taller:")
    print("  1. Copiar el .exe a una carpeta con permiso de escritura (el Escritorio sirve).")
    print("     NO dejarlo en Archivos de programa: ahi no va a poder guardar las fotos.")
    print("  2. Doble clic. Sin modelo arranca en modo recoleccion: 'a' archiva APROBADA,")
    print("     'r' archiva RECHAZADA, 'q' sale.")
    print("  3. Las fotos quedan en  data\\coleccion\\  al lado del .exe.")
    print("  4. Cuando haya modelo, copiar afilado_best.onnx y afilado_best.onnx.json a")
    print("     una carpeta  models\\  al lado del .exe. El programa los toma solo.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
