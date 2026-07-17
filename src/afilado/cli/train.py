"""Entrenamiento del modelo YOLO que clasifica el estado de las herramientas.

CONCEPTOS, EN CASTELLANO LLANO
------------------------------
EPOCA: una pasada completa por TODAS las imagenes del dataset. Si tenes 500 fotos
etiquetadas, una epoca son esas 500 fotos vistas una vez. La red no aprende de una
sola pasada: necesita ver el mismo diente desgastado muchas veces, desde el mismo
angulo, hasta que el patron "esto es desgaste" se le graba en los pesos. Por eso el
valor por defecto son 150 epocas: 150 vueltas al album de fotos.

LOTE (batch): la red no mira las 500 fotos de golpe (no entrarian en la memoria de
la placa de video). Las agarra de a montoncitos. Con batch=16 procesa 16 imagenes,
calcula cuanto se equivoco en esas 16, ajusta los pesos, y sigue con las 16
siguientes. Una epoca de 500 fotos con batch=16 son ~32 ajustes de pesos.
Batch grande => entrena mas rapido y con gradientes mas estables, pero come mas VRAM.
Si la placa tira "CUDA out of memory", bajar el batch (8, 4) es lo primero que hay
que probar.

PACIENCIA (patience): cuantas epocas seguidas se tolera que el modelo NO mejore en
validacion antes de cortar por lo sano. Evita quemar horas de GPU entrenando algo
que ya dejo de aprender (y que, peor, empieza a memorizar en vez de generalizar).

mAP50 y mAP50-95: la nota del boletin. mAP50 mide si la red encontro el objeto con
un solapamiento razonable (50%) contra la caja real; mAP50-95 es mucho mas exigente
(promedia solapamientos del 50% al 95%) y castiga las cajas mal ajustadas. Para este
sistema mAP50 dice "vio el diente", mAP50-95 dice "vio el diente Y lo delineo bien".

`ultralytics` y `torch` se importan de forma PEREZOSA: este modulo debe poder
listarse e inspeccionarse en una maquina sin torch instalado.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Any, Optional

from ..config import ruta_absoluta

_MENSAJE_SIN_ULTRALYTICS = (
    "No se pudo importar 'ultralytics'. Instalalo con: pip install ultralytics\n"
    "Si el entorno virtual esta activado y aun asi falla, revisa que estes usando el python "
    "del venv (.venv\\Scripts\\python.exe) y no el del sistema."
)

_DESTINO_PESOS = "models/afilado_best.pt"


def _construir_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="afilado-train",
        description=(
            "Entrena el modelo YOLO de analisis de afilado sobre un dataset en formato "
            "ultralytics y copia el mejor checkpoint a " + _DESTINO_PESOS + "."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--datos",
        default="data/dataset/data.yaml",
        help="Ruta al data.yaml del dataset (train/val/nc/names).",
    )
    parser.add_argument(
        "--modelo",
        default="yolo11n-seg.pt",
        help="Modelo base del que partir. Se descarga solo si no esta en disco.",
    )
    parser.add_argument(
        "--epocas",
        type=int,
        default=150,
        help="Pasadas completas por el dataset.",
    )
    parser.add_argument("--imgsz", type=int, default=640, help="Lado de la imagen de entrada en px.")
    parser.add_argument(
        "--batch",
        type=int,
        default=16,
        help="Imagenes por lote. Bajalo si la GPU se queda sin memoria.",
    )
    parser.add_argument(
        "--dispositivo",
        default="auto",
        help='Dispositivo de entrenamiento: "auto", "cpu", "0", "cuda:0"...',
    )
    parser.add_argument(
        "--nombre",
        default="afilado",
        help="Nombre de la corrida; define la carpeta runs/<tarea>/<nombre>.",
    )
    parser.add_argument(
        "--reanudar",
        action="store_true",
        help="Reanuda la ultima corrida interrumpida en vez de empezar de cero.",
    )
    parser.add_argument(
        "--paciencia",
        type=int,
        default=50,
        help="Epocas sin mejora en validacion antes de cortar el entrenamiento.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Procesos que cargan imagenes en paralelo.",
    )
    parser.add_argument(
        "--si",
        action="store_true",
        help="Responde que si a la confirmacion de entrenar en CPU.",
    )
    return parser


def _resolver_dispositivo(dispositivo: str) -> tuple[str, bool]:
    """Traduce el dispositivo pedido y avisa si el entrenamiento sera por CPU.

    Devuelve (dispositivo_resuelto, es_cpu).
    """
    if dispositivo != "auto":
        return dispositivo, str(dispositivo).strip().lower() == "cpu"
    try:
        import torch
    except ImportError:
        return "cpu", True
    if torch.cuda.is_available():
        return "0", False
    return "cpu", True


def _confirmar_cpu(asumir_si: bool) -> bool:
    """Pide confirmacion antes de entrenar sin GPU: puede tardar dias, no horas."""
    print(
        "AVISO: no se detecto GPU con CUDA. El entrenamiento correra en CPU y sera LENTISIMO\n"
        "       (dias en vez de horas para un dataset de tamano real).",
    )
    if asumir_si:
        print("       Se continua igual por --si.")
        return True
    try:
        respuesta = input("Entrenar en CPU de todas formas? [s/N]: ")
    except EOFError:
        print("Sin entrada interactiva disponible. Usa --si para confirmar explicitamente.")
        return False
    return respuesta.strip().lower() in {"s", "si", "y", "yes"}


def _metrica(resultados: Any, claves: tuple[str, ...]) -> Optional[float]:
    """Lee una metrica del dict results_dict de ultralytics tolerando cambios de nombre."""
    diccionario = getattr(resultados, "results_dict", None)
    if not isinstance(diccionario, dict):
        return None
    for clave in claves:
        valor = diccionario.get(clave)
        if valor is not None:
            return float(valor)
    return None


def _formatear_metrica(valor: Optional[float]) -> str:
    return f"{valor:.4f}" if valor is not None else "no disponible"


def _copiar_mejor(directorio_corrida: Path) -> Optional[Path]:
    """Copia el best.pt de la corrida a models/afilado_best.pt."""
    origen = directorio_corrida / "weights" / "best.pt"
    if not origen.is_file():
        print(f"ERROR: no se encontro el checkpoint entrenado en '{origen}'.")
        return None
    destino = ruta_absoluta(_DESTINO_PESOS)
    destino.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(origen, destino)
    return destino


def main(argv: Optional[list[str]] = None) -> int:
    """Punto de entrada del entrenamiento. Devuelve 0 en exito."""
    args = _construir_parser().parse_args(argv)

    ruta_datos = ruta_absoluta(args.datos)
    if not ruta_datos.is_file():
        print(
            f"ERROR: no existe el dataset '{ruta_datos}'.\n"
            "Genera el data.yaml con: python -m afilado.cli.prepare_dataset dividir "
            "--origen <carpeta> --salida data/dataset"
        )
        return 2

    try:
        from ultralytics import YOLO
    except ImportError:
        print(f"ERROR: {_MENSAJE_SIN_ULTRALYTICS}")
        return 3

    dispositivo, es_cpu = _resolver_dispositivo(args.dispositivo)
    if es_cpu and not _confirmar_cpu(args.si):
        print("Entrenamiento cancelado.")
        return 1

    print(f"Dataset:     {ruta_datos}")
    print(f"Modelo base: {args.modelo}")
    print(f"Dispositivo: {dispositivo}")
    print(f"Epocas: {args.epocas} | batch: {args.batch} | imgsz: {args.imgsz}")

    try:
        modelo = YOLO(args.modelo)
        resultados_entrenamiento = modelo.train(
            data=str(ruta_datos),
            epochs=args.epocas,
            imgsz=args.imgsz,
            batch=args.batch,
            device=dispositivo,
            name=args.nombre,
            resume=args.reanudar,
            patience=args.paciencia,
            workers=args.workers,
        )
        metricas = modelo.val(data=str(ruta_datos), imgsz=args.imgsz, device=dispositivo)
    except KeyboardInterrupt:
        print(
            "\nEntrenamiento interrumpido por el operario. "
            "Podes retomarlo con --reanudar."
        )
        return 1
    except Exception as error:  # noqa: BLE001 - la CLI no debe volcar un traceback crudo
        print(f"ERROR durante el entrenamiento: {error}")
        return 4

    map50 = _metrica(metricas, ("metrics/mAP50(B)", "metrics/mAP50(M)"))
    map50_95 = _metrica(metricas, ("metrics/mAP50-95(B)", "metrics/mAP50-95(M)"))

    print("\n--- Metricas de validacion ---")
    print(f"mAP50:    {_formatear_metrica(map50)}")
    print(f"mAP50-95: {_formatear_metrica(map50_95)}")

    directorio_corrida = Path(str(getattr(resultados_entrenamiento, "save_dir", "")))
    if not directorio_corrida.name:
        print("ERROR: ultralytics no informo el directorio de la corrida (save_dir).")
        return 4

    destino = _copiar_mejor(directorio_corrida)
    if destino is None:
        return 4

    print(f"\nCorrida:      {directorio_corrida}")
    print(f"Pesos listos: {destino}")
    print("El sistema en vivo ya toma estos pesos: python -m afilado.cli.run_live")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
