"""Pruebas del filtro que decide si una foto puede entrar al dataset.

POR QUE EXISTE ESTO
-------------------
En la colección del taller aparecieron fotos que no eran sierras. La peor: el cartel fijo
"Start DroidCam", archivado como sierra APROBADA. Entró porque el microscopio estaba
desenchufado, el programa cayó en la cámara virtual --avisándolo en el registro, que nadie
mira-- y la tecla 'a' guarda lo que haya en pantalla sin preguntar nada.

Una foto mal etiquetada no se nota: entra, se entrena con ella y el modelo sale raro sin
que nadie pueda decir por qué.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from afilado.cli import run_live
from afilado.config import AppConfig


class _CamaraFalsa:
    def __init__(self, congelada: bool = False) -> None:
        self.congelada = congelada


@pytest.fixture
def cfg() -> AppConfig:
    c = AppConfig(clases=["aprobada", "rechazada"], clases_defecto=["rechazada"])
    c.interfaz.nitidez_minima = 30.0
    return c


def _nitida() -> np.ndarray:
    return np.random.default_rng(5).integers(0, 255, (240, 320, 3), dtype=np.uint8)


def _movida() -> np.ndarray:
    return cv2.GaussianBlur(_nitida(), (31, 31), 0)


def test_una_foto_nitida_de_una_camara_viva_se_archiva(cfg: AppConfig) -> None:
    assert run_live.motivo_para_no_archivar(_CamaraFalsa(), _nitida(), cfg) is None


def test_no_se_archiva_con_la_camara_congelada(cfg: AppConfig) -> None:
    """El caso DroidCam. Lo que se ve no es la sierra, sea cual sea la tecla que se apriete."""
    motivo = run_live.motivo_para_no_archivar(_CamaraFalsa(congelada=True), _nitida(), cfg)

    assert motivo is not None
    assert "CONGELADA" in motivo


def test_el_control_de_foco_NO_alcanza_para_la_camara_congelada(cfg: AppConfig) -> None:
    """Esta es la razon por la que se le pregunta a la camara y no a la imagen.

    El cartel de DroidCam es texto blanco nitido sobre negro: medido sobre la foto real que
    quedo en la coleccion, da nitidez 642 --la mas alta de todo el dataset, mas que
    cualquier diente-- asi que pasa cualquier umbral de foco con comodidad.
    """
    cartel = np.zeros((240, 320, 3), np.uint8)
    cv2.putText(cartel, "Start DroidCam", (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                (255, 255, 255), 2)

    assert run_live.revision.nitidez(cartel) > cfg.interfaz.nitidez_minima, (
        "si esto falla, el cartel dejo de ser nitido y el test perdio sentido"
    )
    assert run_live.motivo_para_no_archivar(_CamaraFalsa(congelada=True), cartel, cfg)


def test_no_se_archiva_una_imagen_movida(cfg: AppConfig) -> None:
    motivo = run_live.motivo_para_no_archivar(_CamaraFalsa(), _movida(), cfg)

    assert motivo is not None
    assert "MOVIDA" in motivo


def test_el_motivo_dice_que_hacer_no_solo_que_fallo(cfg: AppConfig) -> None:
    """El operario tiene que poder arreglarlo sin llamar a nadie."""
    congelada = run_live.motivo_para_no_archivar(_CamaraFalsa(congelada=True), _nitida(), cfg)
    movida = run_live.motivo_para_no_archivar(_CamaraFalsa(), _movida(), cfg)

    assert "conecte" in congelada.lower()
    assert "enfoque" in movida.lower()


def test_el_motivo_de_la_imagen_movida_trae_el_numero(cfg: AppConfig) -> None:
    """Sin el numero no se sabe si falto poco o esta totalmente fuera de foco."""
    motivo = run_live.motivo_para_no_archivar(_CamaraFalsa(), _movida(), cfg)

    assert any(c.isdigit() for c in motivo)


def test_un_video_no_se_considera_congelado(cfg: AppConfig) -> None:
    """Las fuentes que no son camara no tienen el atributo; no pueden bloquear el archivado."""

    class _SinAtributo:
        pass

    assert run_live.motivo_para_no_archivar(_SinAtributo(), _nitida(), cfg) is None


def test_el_umbral_sale_del_config(cfg: AppConfig) -> None:
    """Un taller con otra iluminacion puede necesitar otro umbral."""
    cfg.interfaz.nitidez_minima = 1e9

    assert run_live.motivo_para_no_archivar(_CamaraFalsa(), _nitida(), cfg) is not None


# ---------------------------------------------------------------------------
# EL CONTEO ACUMULADO. total_guardados solo cuenta la sesion actual, que no
# sirve para saber si ya alcanza para entrenar: el dataset se junta de a ratos.
# ---------------------------------------------------------------------------


def test_cuenta_las_fotos_que_ya_estaban_en_disco(tmp_path) -> None:
    from afilado.almacen import AlmacenClasificacion

    for clase, cuantas in (("aprobada", 3), ("rechazada", 2)):
        carpeta = tmp_path / clase
        carpeta.mkdir()
        for i in range(cuantas):
            cv2.imwrite(str(carpeta / f"{clase}_{i}.jpg"), np.zeros((8, 8, 3), np.uint8))

    almacen = AlmacenClasificacion(str(tmp_path), ["aprobada", "rechazada"])

    assert almacen.total_guardados == 0, "esta sesion no guardo nada"
    assert almacen.conteo_por_clase() == {"aprobada": 3, "rechazada": 2}


def test_una_clase_sin_carpeta_cuenta_cero(tmp_path) -> None:
    from afilado.almacen import AlmacenClasificacion

    almacen = AlmacenClasificacion(str(tmp_path), ["aprobada", "rechazada"])

    assert almacen.conteo_por_clase() == {"aprobada": 0, "rechazada": 0}


def test_el_conteo_sube_al_guardar(tmp_path) -> None:
    from afilado.almacen import AlmacenClasificacion

    almacen = AlmacenClasificacion(str(tmp_path), ["aprobada", "rechazada"])
    almacen.guardar(np.zeros((8, 8, 3), np.uint8), "aprobada")

    assert almacen.conteo_por_clase()["aprobada"] == 1


def test_no_cuenta_archivos_que_no_son_fotos(tmp_path) -> None:
    """Un .txt o un .ultima tirado en la carpeta no es una muestra de entrenamiento."""
    from afilado.almacen import AlmacenClasificacion

    carpeta = tmp_path / "aprobada"
    carpeta.mkdir()
    cv2.imwrite(str(carpeta / "buena.jpg"), np.zeros((8, 8, 3), np.uint8))
    (carpeta / "notas.txt").write_text("no soy una foto", encoding="utf-8")

    almacen = AlmacenClasificacion(str(tmp_path), ["aprobada"])

    assert almacen.conteo_por_clase() == {"aprobada": 1}
