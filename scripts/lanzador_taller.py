"""Punto de entrada del ejecutable del taller (AnalisisAfilado.exe).

Existe porque afilado/cli/run_live.py usa imports relativos y no se puede arrancar como
script suelto, que es lo que necesita PyInstaller. Aparte se ocupa de lo unico que le falta
al programa cuando se lo abre con doble clic: que un error de arranque se VEA.

EL EJECUTABLE SE ARMA SIN CONSOLA
---------------------------------
El taller abre el programa con doble clic sobre un acceso directo. Una ventana negra de
consola al lado de la ventana del programa no le dice nada a nadie, y encima si alguien la
cierra se lleva el programa puesto. Asi que el .exe se arma con --windowed.

Eso tiene una consecuencia que hay que atender aca: sin consola, sys.stdout y sys.stderr
son None. print() no imprime en ningun lado, input() levanta excepcion, y un traceback se
va al vacio. Un error de arranque --la camara tomada por otro programa, un modelo corrupto,
un config mal escrito-- se veria como un parpadeo y nada mas, que es exactamente lo que
pasaba antes de que existiera este archivo.

Por eso, cuando algo falla:
  1. se deja el detalle completo en registro.txt, al lado del .exe;
  2. se muestra un cartel de Windows diciendo que paso y donde esta ese archivo.

El cartel es lo que convierte "no anda" en algo que se puede arreglar sin estar ahi.
"""

from __future__ import annotations

import sys
import traceback
from typing import Optional

_TITULO = "Analisis de afilado"


def _hay_consola() -> bool:
    """False cuando el .exe se armo con --windowed: no hay donde imprimir."""
    return sys.stderr is not None


def _cartel(mensaje: str, titulo: str = _TITULO) -> None:
    """Muestra un cartel de Windows. Si no se puede, no pasa nada: es el ultimo recurso."""
    try:
        import ctypes

        # 0x10 = icono de error, 0x40000 = adelante de todo (si no, queda tapado por la
        # ventana del programa y el operario no lo ve nunca).
        ctypes.windll.user32.MessageBoxW(None, mensaje, titulo, 0x10 | 0x40000)
    except Exception:  # noqa: BLE001 - avisar nunca puede romper mas que el error original
        pass


def _ruta_del_registro() -> Optional[str]:
    try:
        from afilado.cli.run_live import _ARCHIVO_REGISTRO
        from afilado.config import raiz_repo

        return str(raiz_repo() / _ARCHIVO_REGISTRO)
    except Exception:  # noqa: BLE001
        return None


def _ultimo_error_del_registro() -> Optional[str]:
    """La ultima linea de ERROR que se escribio, para poder mostrarla en el cartel.

    Cuando el programa termina mal sin excepcion --la camara tomada por otro programa es el
    caso tipico-- lo unico que hay es un codigo de salida, y "termino con el codigo 1" no le
    sirve a nadie. La causa real ya quedo en el registro unas lineas antes; esto la saca de
    ahi para ponerla en el cartel.
    """
    ruta = _ruta_del_registro()
    if not ruta:
        return None
    try:
        with open(ruta, encoding="utf-8", errors="replace") as archivo:
            errores = [l.strip() for l in archivo if " ERROR " in l or " WARNING " in l]
    except OSError:
        return None
    if not errores:
        return None
    # Se recorta el prefijo de fecha y modulo: en un cartel ocupa media linea y no aporta.
    ultimo = errores[-1]
    for marca in (" ERROR ", " WARNING "):
        if marca in ultimo:
            resto = ultimo.split(marca, 1)[1]
            return resto.split("] ", 1)[-1] if "] " in resto else resto
    return ultimo


def _avisar_del_fallo(detalle: str) -> None:
    """Deja el detalle donde se pueda leer despues y lo pone en pantalla."""
    registro = _ruta_del_registro()

    # Al registro primero: es lo que sobrevive a que el operario cierre el cartel.
    try:
        import logging

        logging.getLogger("afilado.lanzador").error("fallo al arrancar:\n%s", detalle)
    except Exception:  # noqa: BLE001
        pass

    if _hay_consola():
        print()
        print("=" * 70)
        print(" El programa termino con un error.")
        print(" Anota o sacale una foto a lo que dice mas arriba: eso es lo que hace")
        print(" falta para arreglarlo.")
        print("=" * 70)
        try:
            input(" Enter para cerrar...")
        except (EOFError, OSError):
            pass
        return

    ultimas = [linea for linea in detalle.strip().splitlines() if linea.strip()]
    resumen = ultimas[-1] if ultimas else "error desconocido"
    texto = [
        "El programa no pudo arrancar.",
        "",
        resumen,
    ]
    if registro:
        texto += [
            "",
            "El detalle completo quedo en:",
            registro,
            "",
            "Sacale una foto a ese archivo o mandalo: eso es lo que hace falta",
            "para arreglarlo.",
        ]
    _cartel("\n".join(texto))


def main() -> int:
    try:
        from afilado.cli.run_live import main as correr

        codigo = correr()
    except KeyboardInterrupt:
        return 130
    except BaseException:  # noqa: BLE001 - se informa entero, no se traga nada
        _avisar_del_fallo(traceback.format_exc())
        return 1

    if codigo != 0:
        causa = _ultimo_error_del_registro()
        _avisar_del_fallo(
            causa
            or (
                "El programa termino con el codigo %d y no dejo ningun error en el "
                "registro." % codigo
            )
        )
    return codigo


if __name__ == "__main__":
    raise SystemExit(main())
