"""Pruebas de la eleccion de fuente y backend de la camara.

Ninguna abre una camara de verdad: la suite tiene que correr en una maquina sin webcam, y
en la de integracion continua. Lo que se prueba aca es la LOGICA de que se va a intentar y
en que orden, que es donde estaba el problema: el programa probaba un solo backend y, si esa
camara no lo hablaba, decia "no se pudo abrir la camara" y ahi terminaba.
"""

from __future__ import annotations

import cv2
import pytest

from afilado.camara import (
    Camara,
    _backends_de_sondeo,
    _resolver_backend,
    _resolver_fuente,
    es_fuente_automatica,
)
from afilado.config import CamaraConfig


@pytest.mark.parametrize("valor", ["auto", "AUTO", "  Auto  "])
def test_reconoce_la_fuente_automatica(valor: str) -> None:
    assert es_fuente_automatica(valor) is True


@pytest.mark.parametrize("valor", [0, 1, "0", "video.mp4", "rtsp://camara/1", ""])
def test_no_confunde_otras_fuentes_con_auto(valor) -> None:
    assert es_fuente_automatica(valor) is False


def test_auto_se_resuelve_como_camara() -> None:
    """'auto' no es una ruta ni una URL: es una camara todavia sin elegir."""
    fuente, es_camara = _resolver_fuente("auto")

    assert fuente == "auto"
    assert es_camara is True


@pytest.mark.parametrize(
    "entrada, esperado", [(0, 0), (2, 2), ("0", 0), ("3", 3), ("-1", -1)]
)
def test_los_indices_siguen_siendo_indices(entrada, esperado) -> None:
    fuente, es_camara = _resolver_fuente(entrada)

    assert (fuente, es_camara) == (esperado, True)


def test_una_ruta_no_es_camara(tmp_path) -> None:
    video = tmp_path / "grabacion.mp4"
    video.write_bytes(b"")

    fuente, es_camara = _resolver_fuente(str(video))

    assert es_camara is False


def test_con_fuente_auto_se_recorren_varios_indices() -> None:
    """Sin esto, una camara que no queda en el indice 0 era invisible."""
    camara = Camara(CamaraConfig(fuente="auto", backend="auto"))

    candidatos = camara._candidatos()
    indices = sorted({indice for indice, _ in candidatos})

    assert len(indices) > 1, "con 'auto' hay que mirar mas de un indice"
    assert indices[0] == 0
    assert len({backend for _, backend in candidatos}) == len(_backends_de_sondeo())


def test_con_indice_fijo_se_prueban_backends_pero_no_otros_indices() -> None:
    """Cambiar de backend es transparente: es la misma camara. Cambiar de indice no:
    podria ser otra camara, y eso solo se hace si el config lo pidio con 'auto'."""
    camara = Camara(CamaraConfig(fuente=1, backend="auto"))

    candidatos = camara._candidatos()

    assert {indice for indice, _ in candidatos} == {1}
    assert len(candidatos) == len(_backends_de_sondeo())


def test_un_backend_nombrado_a_mano_se_respeta() -> None:
    """Quien escribio 'msmf' en el config sabe por que; no se le prueba nada mas."""
    camara = Camara(CamaraConfig(fuente=0, backend="msmf"))

    candidatos = camara._candidatos()

    assert candidatos == [(0, cv2.CAP_MSMF)]


def test_el_sondeo_prueba_dshow_antes_que_msmf_en_windows(monkeypatch) -> None:
    """DSHOW respeta mejor la resolucion pedida, asi que va primero. Pero no puede ser el
    unico: hay camaras --y sesiones de Escritorio Remoto-- donde no abre nada y MSMF si."""
    monkeypatch.setattr("sys.platform", "win32")

    backends = _backends_de_sondeo()

    assert backends[0] == cv2.CAP_DSHOW
    assert cv2.CAP_MSMF in backends
    assert cv2.CAP_ANY in backends


def test_un_backend_desconocido_se_rechaza_con_las_opciones() -> None:
    """Un typo en el config no puede quedar en silencio usando otro backend."""
    with pytest.raises(ValueError, match="dshhow"):
        _resolver_backend("dshhow", True)
