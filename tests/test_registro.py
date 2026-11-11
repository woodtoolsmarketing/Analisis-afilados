"""Pruebas del registro en archivo.

POR QUE IMPORTA ESTO
--------------------
El ejecutable del taller se arma SIN consola: se abre con doble clic desde un acceso en el
Escritorio y no hay ninguna ventana negra donde mirar. Eso deja al archivo de registro como
lo UNICO que queda cuando algo falla alla. Todos los problemas de camara que se arreglaron
hasta ahora --el 0x0, la camara virtual, el handle colgado-- se diagnosticaron leyendo ese
log. Si deja de escribirse, el taller se vuelve una caja negra.

Y hay una trampa concreta: con --windowed, PyInstaller deja sys.stderr en None. Un
StreamHandler construido sobre eso revienta al primer mensaje, y el programa se cae por el
log en vez de por un problema de verdad.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from afilado.cli import run_live


@pytest.fixture(autouse=True)
def _limpiar_el_logger():
    """configurar_registro() toca el logger raiz, que es global a todo el proceso."""
    raiz = logging.getLogger()
    antes, nivel = list(raiz.handlers), raiz.level
    raiz.handlers = []
    yield
    for manejador in raiz.handlers:
        manejador.close()
    raiz.handlers, raiz.level = antes, nivel


def _apuntar_a(monkeypatch, carpeta: Path) -> None:
    monkeypatch.setattr(run_live, "raiz_repo", lambda: carpeta)


def test_el_registro_queda_en_un_archivo(monkeypatch, tmp_path: Path) -> None:
    _apuntar_a(monkeypatch, tmp_path)

    archivo = run_live.configurar_registro()
    logging.getLogger("afilado.prueba").info("la camara entrego 60 frames")

    assert archivo == tmp_path / "registro.txt"
    assert "la camara entrego 60 frames" in archivo.read_text(encoding="utf-8")


def test_el_registro_lleva_fecha_y_hora(monkeypatch, tmp_path: Path) -> None:
    """Sin hora no se puede saber si lo que se esta leyendo es de hoy o de la semana pasada."""
    _apuntar_a(monkeypatch, tmp_path)

    archivo = run_live.configurar_registro()
    logging.getLogger("afilado.prueba").warning("la camara no acepta fijar brillo")

    linea = archivo.read_text(encoding="utf-8").strip().splitlines()[-1]
    assert linea.startswith("20"), f"no empieza con la fecha: {linea!r}"
    assert "WARNING" in linea and "brillo" in linea


def test_sin_consola_no_se_arma_un_manejador_de_pantalla(monkeypatch, tmp_path: Path) -> None:
    """Es el caso del .exe con --windowed: sys.stderr es None.

    Un StreamHandler sobre None no falla al construirse, falla al PRIMER mensaje. Por eso
    aca no alcanza con que configurar_registro() no explote: hay que escribir algo.
    """
    _apuntar_a(monkeypatch, tmp_path)
    monkeypatch.setattr("sys.stderr", None)

    archivo = run_live.configurar_registro()
    logging.getLogger("afilado.prueba").info("arranco sin consola")

    de_pantalla = [
        h for h in logging.getLogger().handlers if type(h) is logging.StreamHandler
    ]
    assert de_pantalla == [], "con sys.stderr en None no puede haber manejador de pantalla"
    assert "arranco sin consola" in archivo.read_text(encoding="utf-8")


def test_con_consola_se_escribe_en_los_dos_lados(monkeypatch, tmp_path: Path, capsys) -> None:
    _apuntar_a(monkeypatch, tmp_path)

    archivo = run_live.configurar_registro()
    logging.getLogger("afilado.prueba").info("corriendo desde una terminal")

    assert "corriendo desde una terminal" in archivo.read_text(encoding="utf-8")
    assert "corriendo desde una terminal" in capsys.readouterr().err


def test_si_no_se_puede_escribir_el_programa_sigue(monkeypatch, tmp_path: Path) -> None:
    """Disco lleno o carpeta de solo lectura. Sin registro se trabaja; sin programa no."""
    inexistente = tmp_path / "no" / "existe" / "y" / "no" / "se" / "puede" / "crear"
    _apuntar_a(monkeypatch, inexistente)

    def _explota(*args, **kwargs):
        raise OSError("carpeta de solo lectura")

    monkeypatch.setattr(logging.handlers.RotatingFileHandler, "__init__", _explota)

    archivo = run_live.configurar_registro()

    assert archivo is None
    logging.getLogger("afilado.prueba").info("esto no tiene que romper nada")


def test_el_registro_rota_y_no_crece_sin_limite(monkeypatch, tmp_path: Path) -> None:
    """Una PC de taller prendida todo el dia, meses seguidos, llenaria el disco."""
    _apuntar_a(monkeypatch, tmp_path)

    run_live.configurar_registro()
    de_disco = [
        h for h in logging.getLogger().handlers
        if isinstance(h, logging.handlers.RotatingFileHandler)
    ]

    assert len(de_disco) == 1
    assert de_disco[0].maxBytes > 0, "sin maxBytes no rota nunca"
    assert de_disco[0].backupCount > 0, "sin backupCount se pierde lo anterior al rotar"


# ---------------------------------------------------------------------------
# EL CARTEL DE ERROR del lanzador. Sin consola, es lo unico que el operario ve
# cuando el programa no arranca. Si dice "codigo 1" no sirve de nada; tiene que
# decir QUE paso, y eso se saca del registro.
# ---------------------------------------------------------------------------


def _lanzador():
    """Importa scripts/lanzador_taller.py, que vive fuera del paquete."""
    import importlib.util

    ruta = Path(__file__).resolve().parents[1] / "scripts" / "lanzador_taller.py"
    spec = importlib.util.spec_from_file_location("lanzador_taller", ruta)
    modulo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(modulo)
    return modulo


def test_el_cartel_muestra_la_causa_real_no_el_codigo(monkeypatch, tmp_path: Path) -> None:
    """El caso tipico del taller: la camara tomada por otro programa.

    run_live devuelve 1 sin levantar excepcion, asi que lo unico que tiene el lanzador es
    ese 1. La causa de verdad quedo escrita unas lineas antes.
    """
    registro = tmp_path / "registro.txt"
    registro.write_text(
        "2026-11-11 09:14:02 INFO [afilado.camara] probando el indice 0\n"
        "2026-11-11 09:14:05 ERROR [afilado.cli.run_live] no se pudo abrir la camara\n",
        encoding="utf-8",
    )
    lanzador = _lanzador()
    monkeypatch.setattr(lanzador, "_ruta_del_registro", lambda: str(registro))

    assert lanzador._ultimo_error_del_registro() == "no se pudo abrir la camara"


def test_el_cartel_se_queda_con_el_ultimo_problema(monkeypatch, tmp_path: Path) -> None:
    registro = tmp_path / "registro.txt"
    registro.write_text(
        "2026-11-11 09:14:02 WARNING [afilado.camara] no acepta fijar brillo en 16\n"
        "2026-11-11 09:14:09 ERROR [afilado.camara] la camara dejo de entregar frames\n",
        encoding="utf-8",
    )
    lanzador = _lanzador()
    monkeypatch.setattr(lanzador, "_ruta_del_registro", lambda: str(registro))

    assert lanzador._ultimo_error_del_registro() == "la camara dejo de entregar frames"


def test_sin_registro_el_cartel_no_rompe(monkeypatch, tmp_path: Path) -> None:
    lanzador = _lanzador()
    monkeypatch.setattr(lanzador, "_ruta_del_registro", lambda: str(tmp_path / "no-esta.txt"))

    assert lanzador._ultimo_error_del_registro() is None


def test_un_registro_sin_errores_no_inventa_una_causa(monkeypatch, tmp_path: Path) -> None:
    registro = tmp_path / "registro.txt"
    registro.write_text(
        "2026-11-11 09:14:02 INFO [afilado.cli.run_live] MODO SIMPLE\n", encoding="utf-8"
    )
    lanzador = _lanzador()
    monkeypatch.setattr(lanzador, "_ruta_del_registro", lambda: str(registro))

    assert lanzador._ultimo_error_del_registro() is None


def test_el_lanzador_sabe_que_no_hay_consola(monkeypatch) -> None:
    """Es lo que decide entre un print que nadie ve y un cartel de Windows."""
    lanzador = _lanzador()

    monkeypatch.setattr("sys.stderr", None)
    assert lanzador._hay_consola() is False
