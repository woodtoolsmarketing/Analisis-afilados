"""Genera un marcador ArUco imprimible con tamano fisico exacto.

El marcador es la unica referencia de escala del sistema: todo el calculo de
milimetros depende de que el lado impreso mida EXACTAMENTE los milimetros
declarados en la configuracion. Por eso este script no dibuja "un marcador
grande" sino una imagen cuyo lado en pixeles corresponde milimetro a milimetro
con los DPI de impresion elegidos.

Uso tipico:
    python scripts/make_aruco.py --diccionario DICT_4X4_50 --id 0 --lado-mm 30 --dpi 300
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

PULGADA_EN_MM: float = 25.4


def mm_a_px(milimetros: float, dpi: int) -> int:
    """Convierte milimetros a pixeles para una densidad de impresion dada."""
    return int(round(milimetros / PULGADA_EN_MM * dpi))


def obtener_diccionario(nombre: str) -> "cv2.aruco.Dictionary":
    """Devuelve el diccionario ArUco por nombre (ej "DICT_4X4_50").

    Lanza ValueError si el nombre no existe en cv2.aruco. Requiere
    opencv-contrib-python: el modulo aruco no viene en opencv-python.
    """
    if not hasattr(cv2, "aruco"):
        raise ValueError(
            "cv2.aruco no esta disponible. Instala opencv-contrib-python "
            "(desinstala antes opencv-python: no pueden convivir)."
        )
    constante = getattr(cv2.aruco, nombre, None)
    if constante is None:
        disponibles = sorted(n for n in dir(cv2.aruco) if n.startswith("DICT_"))
        raise ValueError(
            f"Diccionario ArUco desconocido: {nombre!r}. "
            f"Disponibles: {', '.join(disponibles)}"
        )
    if hasattr(cv2.aruco, "getPredefinedDictionary"):
        return cv2.aruco.getPredefinedDictionary(constante)
    return cv2.aruco.Dictionary_get(constante)


def dibujar_marcador(diccionario: str, id_marcador: int, lado_px: int) -> np.ndarray:
    """Genera la imagen cruda del marcador (sin margen ni leyenda), en escala de grises."""
    dicc = obtener_diccionario(diccionario)
    if hasattr(cv2.aruco, "generateImageMarker"):
        return cv2.aruco.generateImageMarker(dicc, id_marcador, lado_px)
    return cv2.aruco.drawMarker(dicc, id_marcador, lado_px)


def _componer_hoja(
    marcador: np.ndarray,
    margen_px: int,
    leyenda: str,
) -> np.ndarray:
    """Pega el marcador sobre fondo blanco con margen y escribe la leyenda al pie.

    El margen blanco (zona silenciosa) no es decorativo: el detector necesita
    contraste alrededor del borde negro para encontrar el cuadrilatero.
    """
    lado_px = marcador.shape[0]
    alto_leyenda = max(margen_px, mm_a_px(6.0, 300))
    ancho = lado_px + 2 * margen_px
    alto = lado_px + 2 * margen_px + alto_leyenda

    hoja = np.full((alto, ancho), 255, dtype=np.uint8)
    hoja[margen_px : margen_px + lado_px, margen_px : margen_px + lado_px] = marcador

    escala_fuente = max(0.35, ancho / 1400.0)
    grosor = max(1, int(round(escala_fuente * 2)))
    (ancho_texto, alto_texto), _ = cv2.getTextSize(
        leyenda, cv2.FONT_HERSHEY_SIMPLEX, escala_fuente, grosor
    )
    origen_x = max(2, (ancho - ancho_texto) // 2)
    origen_y = margen_px + lado_px + margen_px + alto_texto
    origen_y = min(origen_y, alto - 2)
    cv2.putText(
        hoja,
        leyenda,
        (origen_x, origen_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        escala_fuente,
        (0,),
        grosor,
        cv2.LINE_AA,
    )
    return hoja


def _embeber_dpi(ruta: Path, dpi: int) -> bool:
    """Reescribe el PNG con la densidad fisica embebida (chunk pHYs).

    Devuelve True si se pudo. Es opcional: el PNG ya tiene los pixeles
    correctos, pero sin pHYs muchos visores imprimen "a lo que les parezca".
    """
    try:
        from PIL import Image
    except ImportError:
        return False
    with Image.open(ruta) as imagen:
        imagen.save(ruta, dpi=(dpi, dpi))
    return True


def construir_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Genera un marcador ArUco imprimible con tamano fisico exacto.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--diccionario", type=str, default="DICT_4X4_50")
    parser.add_argument("--id", dest="id_marcador", type=int, default=0)
    parser.add_argument("--lado-mm", dest="lado_mm", type=float, default=30.0)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--salida", type=str, default="marcador.png")
    parser.add_argument("--margen-mm", dest="margen_mm", type=float, default=5.0)
    return parser


def _imprimir_instrucciones(
    ruta: Path, lado_mm: float, lado_px: int, dpi: int, dpi_embebido: bool
) -> None:
    print(f"\nMarcador escrito en: {ruta}")
    print(f"  Lado del marcador: {lado_mm} mm = {lado_px} px @ {dpi} DPI")
    if dpi_embebido:
        print(f"  DPI embebido en el PNG: {dpi}")
    else:
        print(
            "  AVISO: Pillow no esta instalado, no se pudo embeber el DPI en el PNG.\n"
            "         El archivo IGUAL tiene la cantidad exacta de pixeles: basta con\n"
            f"         imprimir al 100% indicando {dpi} DPI en el dialogo de impresion."
        )
    print("\nCOMO IMPRIMIRLO (de esto depende toda la precision del sistema):")
    print("  1. Imprimir al 100% / 'Tamano real'. NUNCA 'Ajustar a pagina': te cambia la escala.")
    print("  2. Papel MATE. El papel brillante espeja bajo el aro LED y ciega al detector.")
    print("  3. NO plastificar. El plastico agrega reflejo y el calor deforma el papel.")
    print(f"  4. Verificar con calibre que el lado negro mida {lado_mm} mm. Si no, reimprimir.")
    print("  5. Pegarlo bien plano, sin arrugas ni burbujas.")
    print("  6. Colocarlo a la MISMA ALTURA que la cara de la pieza que se mide.")
    print("     Si el marcador queda en la mesa y la pieza tiene espesor, la cara superior")
    print("     esta mas cerca del lente, se ve mas grande y la medida sale INFLADA")
    print("     (error de paralaje). Es la limitacion principal del metodo.")


def main(argv: Optional[list[str]] = None) -> int:
    args = construir_parser().parse_args(argv)

    if args.lado_mm <= 0:
        print("ERROR: --lado-mm debe ser mayor que 0.", file=sys.stderr)
        return 2
    if args.margen_mm < 0:
        print("ERROR: --margen-mm no puede ser negativo.", file=sys.stderr)
        return 2
    if args.dpi <= 0:
        print("ERROR: --dpi debe ser mayor que 0.", file=sys.stderr)
        return 2
    if args.id_marcador < 0:
        print("ERROR: --id no puede ser negativo.", file=sys.stderr)
        return 2

    lado_px = mm_a_px(args.lado_mm, args.dpi)
    if lado_px < 8:
        print(
            f"ERROR: {args.lado_mm} mm a {args.dpi} DPI dan {lado_px} px: "
            "el marcador seria ilegible. Sube --dpi o --lado-mm.",
            file=sys.stderr,
        )
        return 2

    try:
        marcador = dibujar_marcador(args.diccionario, args.id_marcador, lado_px)
    except ValueError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2
    except cv2.error as error:
        print(
            f"ERROR: OpenCV rechazo el id {args.id_marcador} para {args.diccionario}. "
            f"Revisa que el id este dentro del rango del diccionario. Detalle: {error}",
            file=sys.stderr,
        )
        return 2

    leyenda = f"{args.diccionario}  id={args.id_marcador}  lado={args.lado_mm:g}mm  {args.dpi}dpi"
    hoja = _componer_hoja(marcador, mm_a_px(args.margen_mm, args.dpi), leyenda)

    ruta = Path(args.salida).expanduser().resolve()
    ruta.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(ruta), hoja):
        print(f"ERROR: no se pudo escribir {ruta}", file=sys.stderr)
        return 1

    _imprimir_instrucciones(ruta, args.lado_mm, lado_px, args.dpi, _embeber_dpi(ruta, args.dpi))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
