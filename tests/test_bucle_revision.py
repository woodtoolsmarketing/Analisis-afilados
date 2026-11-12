"""El bucle entero, de punta a punta, con una camara de mentira y teclas escritas a mano.

POR QUE HACE FALTA ESTO ADEMAS DE LAS PRUEBAS DE CADA MODULO
-------------------------------------------------------------
Los modulos estan probados de a uno, pero los errores que llegaron al taller no fueron de
un modulo: fueron de como se enganchan. Un cierre que lee una variable que despues se
reasigna, una lambda que captura la foto equivocada, una funcion que cambio de parametros
y su unico llamador quedo viejo. Nada de eso lo ve una prueba de unidad, y todo eso lo
pagaba el operario abriendo el programa.

Aca se corre _ejecutar_revision de verdad --el mismo bucle que corre en la PC del
taller-- con la camara, la ventana y el teclado de mentira, y se mira que quedo en disco.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import pytest

from afilado import config as config_mod
from afilado.cli import run_live

_ESC = 27
_ENTER = 13
_ESPACIO = 32


def _diente(semilla: int = 0, lado: int = 240) -> np.ndarray:
    """Una imagen con estructura: el ruido puro no se parece a lo que da el microscopio."""
    img = np.full((lado, lado, 3), 60, np.uint8)
    cv2.fillPoly(img, [np.array([[10, lado - 20], [10, 120 + semilla * 4],
                                 [lado - 60, 40], [lado - 20, 70],
                                 [lado - 20, lado - 20]])], (190, 190, 190))
    cv2.line(img, (10, 120 + semilla * 4), (lado - 60, 40), (235, 235, 235), 6)
    for i in range(0, lado, 7):
        cv2.line(img, (i, 0), (i + 40 + semilla * 9, lado), (0, 0, 0), 1)
    return cv2.GaussianBlur(img, (5, 5), 0)


class _CamaraFalsa:
    """Entrega una foto distinta cada vez: si entregara siempre la misma, el guardado se
    frenaria solo por duplicada y la prueba no probaria nada."""

    congelada = False
    descripcion = "camara de prueba"

    def __init__(self, *_args, **_kwargs) -> None:
        self.leidos = 0
        self.cerrada = False

    def abrir(self) -> None:
        pass

    def leer(self) -> Optional[np.ndarray]:
        self.leidos += 1
        return _diente(self.leidos % 7)

    def cerrar(self) -> None:
        self.cerrada = True


@pytest.fixture
def taller(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Un taller de mentira: todas las rutas relativas caen adentro de tmp_path."""
    monkeypatch.setattr(config_mod, "raiz_repo", lambda: tmp_path)
    monkeypatch.setattr(run_live, "Camara", _CamaraFalsa)
    monkeypatch.setattr(run_live, "crear_clasificador", lambda cfg: None)

    # La ventana, el teclado y el mouse, de mentira.
    monkeypatch.setattr(cv2, "namedWindow", lambda *a, **k: None)
    monkeypatch.setattr(cv2, "setMouseCallback", lambda *a, **k: None)
    monkeypatch.setattr(cv2, "imshow", lambda *a, **k: None)
    monkeypatch.setattr(cv2, "resizeWindow", lambda *a, **k: None)
    monkeypatch.setattr(cv2, "destroyAllWindows", lambda: None)
    return tmp_path


def _correr(teclas: list[int], monkeypatch: pytest.MonkeyPatch) -> int:
    """Corre el bucle mandandole esas teclas. Al quedarse sin teclas, sale."""
    pendientes = list(teclas)

    def _waitKey(_espera: int = 0) -> int:
        # Al terminar el guion se sale SIEMPRE: sin esto, una prueba que no llegue a la
        # 'q' deja el bucle girando para siempre y lo unico que se ve es que pytest se
        # colgo. Ya paso.
        return pendientes.pop(0) if pendientes else ord("q")

    monkeypatch.setattr(cv2, "waitKey", _waitKey)

    cfg = config_mod.AppConfig(clases=["aprobada", "rechazada"], clases_defecto=["rechazada"])
    cfg.interfaz.modo = "simple"
    cfg.interfaz.evaluar_en_vivo = True
    cfg.interfaz.nitidez_minima = 1.0
    cfg.detector.tarea = "classify"
    cfg.memoria.minimo_por_concepto = 1
    return run_live._ejecutar_revision(cfg, argparse.Namespace(grabar=None))


def _fotos(base: Path, concepto: str) -> list[Path]:
    carpeta = base / "data" / "coleccion" / concepto
    return sorted(carpeta.glob("*.jpg")) if carpeta.is_dir() else []


# ---------------------------------------------------------------------------


def test_arranca_y_sale_limpio(taller: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    assert _correr([ord("q")], monkeypatch) == 0


def test_guardar_una_foto_la_deja_en_su_carpeta(taller: Path, monkeypatch) -> None:
    _correr([_ESPACIO, ord("a"), ord("q")], monkeypatch)

    assert len(_fotos(taller, "aprobada")) == 1


def test_la_memoria_se_arma_sola_con_lo_que_se_va_guardando(taller: Path, monkeypatch) -> None:
    """Esto es lo que pidio el taller: que aprenda de cada foto, sin entrenar nada."""
    _correr(
        [_ESPACIO, ord("a"), ord(" "), ord("a"), ord(" "), ord("r"), ord("q")],
        monkeypatch,
    )

    cache = taller / "data" / "memoria" / "memoria.npz"
    assert cache.is_file()
    # Las fotos quedaron repartidas en sus dos conceptos.
    assert len(_fotos(taller, "aprobada")) >= 1
    assert len(_fotos(taller, "rechazada")) >= 1


def test_crear_un_concepto_desde_el_panel_y_archivar_ahi(taller: Path, monkeypatch) -> None:
    """El camino completo del pedido: abrir el panel, crear un concepto y mandar la foto."""
    teclas = [_ESPACIO, ord("c"), ord("n")]
    teclas += [ord(c) for c in "astillado"]
    teclas += [_ENTER, _ENTER, ord("q")]

    _correr(teclas, monkeypatch)

    assert (taller / "data" / "coleccion" / "astillado").is_dir()
    assert len(_fotos(taller, "astillado")) == 1


def test_la_q_adentro_del_nombre_no_cierra_el_programa(taller: Path, monkeypatch) -> None:
    """Si la 'q' se escapara al manejador de siempre, el programa se cerraria en la mitad
    de la palabra y el operario no entenderia por que."""
    teclas = [ord("c"), ord("n")]
    teclas += [ord(c) for c in "quemado"]
    teclas += [_ENTER, _ESC, ord("q")]

    _correr(teclas, monkeypatch)

    assert (taller / "data" / "coleccion" / "quemado").is_dir()


def test_borrar_un_concepto_desde_el_panel(taller: Path, monkeypatch) -> None:
    _correr([_ESPACIO, ord("a"), ord("q")], monkeypatch)
    assert len(_fotos(taller, "aprobada")) == 1

    # 2 elige 'aprobada' (1 es la galeria de modelos), b pide borrar, s confirma.
    _correr([ord("c"), ord("2"), ord("b"), ord("s"), ord("q")], monkeypatch)

    assert not (taller / "data" / "coleccion" / "aprobada").exists()
    assert list((taller / "data" / "coleccion" / "_papelera").rglob("*.jpg"))


def test_el_concepto_nuevo_se_puede_usar_sin_reiniciar(taller: Path, monkeypatch) -> None:
    """El almacen valida contra su lista de clases: si no se entera del concepto nuevo,
    guardar ahi falla con 'clase desconocida' hasta cerrar y volver a abrir."""
    teclas = [ord("c"), ord("n")] + [ord(c) for c in "rebaba"] + [_ENTER, _ESC]
    # Ya fuera del panel: congelar y guardar en el concepto nuevo desde el panel otra vez.
    teclas += [_ESPACIO, ord("c"), ord("4"), _ENTER, ord("q")]

    _correr(teclas, monkeypatch)

    assert len(_fotos(taller, "rebaba")) == 1


def test_borrar_una_foto_desde_el_panel(taller: Path, monkeypatch) -> None:
    _correr([_ESPACIO, ord("a"), ord("q")], monkeypatch)
    guardada = _fotos(taller, "aprobada")[0]

    # c abre, 2 elige aprobada, v ve sus fotos, b borra la elegida.
    _correr([ord("c"), ord("2"), ord("v"), ord("b"), ord("q")], monkeypatch)

    assert not guardada.exists()
    assert _fotos(taller, "aprobada") == []


def test_la_foto_de_modelo_tambien_se_elige_desde_el_panel(taller: Path, monkeypatch) -> None:
    """Es el destino que el taller mas usa: tiene que estar en la misma lista."""
    _correr([_ESPACIO, ord("c"), ord("1"), _ENTER, ord("q")], monkeypatch)

    modelos = list((taller / "data" / "referencia" / "microscopio").glob("*.jpg"))
    assert len(modelos) == 1


def test_no_se_archiva_dos_veces_la_misma_foto_congelada(taller: Path, monkeypatch) -> None:
    """El error que dejo 50 copias por clase: con la imagen congelada, cada pulsacion de
    GUARDAR reescribia el MISMO frame."""
    _correr([_ESPACIO, ord("a"), ord("a"), ord("a"), ord("q")], monkeypatch)

    assert len(_fotos(taller, "aprobada")) == 1
