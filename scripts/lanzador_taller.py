"""Punto de entrada del ejecutable del taller (AnalisisAfilado.exe).

Existe porque afilado/cli/run_live.py usa imports relativos y no se puede arrancar como
script suelto, que es lo que necesita PyInstaller. Aparte agrega lo unico que le falta al
programa cuando se lo abre con doble clic en vez de desde una consola: que la ventana NO se
cierre de golpe si algo falla al arrancar.

Sin esa pausa, un error de arranque --la camara tomada por otro programa, un modelo corrupto,
un config mal escrito-- se ve como un parpadeo negro y nada mas. El operario no tiene forma
de saber que paso ni que contarle a quien lo instalo.
"""

from __future__ import annotations

import sys
import traceback


def main() -> int:
    try:
        from afilado.cli.run_live import main as correr

        codigo = correr()
    except KeyboardInterrupt:
        return 130
    except BaseException:  # noqa: BLE001 - se muestra entero y se pausa, no se traga nada
        traceback.print_exc()
        codigo = 1

    if codigo != 0:
        print()
        print("=" * 70)
        print(" El programa termino con un error (codigo %d)." % codigo)
        print(" Anota o sacale una foto a lo que dice mas arriba: eso es lo que hace")
        print(" falta para arreglarlo.")
        print("=" * 70)
        try:
            input(" Enter para cerrar...")
        except (EOFError, OSError):
            pass
    return codigo


if __name__ == "__main__":
    raise SystemExit(main())
