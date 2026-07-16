"""Pruebas del ROI, la limpieza morfologica y el descarte de detecciones."""

from __future__ import annotations

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from afilado.config import FiltrosConfig, RoiConfig
from afilado.filtros import (
    aplicar_filtros,
    centro_en_roi,
    limpiar_mascara,
    rect_roi,
    recortar_roi,
)
from afilado.tipos import Deteccion, Escala, Medida

FORMA_HD = (720, 1280)  # (alto, ancho)
MM_POR_PX = 0.2         # 5 px/mm


@pytest.fixture
def roi_centrado() -> RoiConfig:
    """ROI por defecto: 70% central del frame."""
    return RoiConfig(habilitado=True, x=0.15, y=0.15, w=0.7, h=0.7)


@pytest.fixture
def escala_valida() -> Escala:
    return Escala(mm_por_px=MM_POR_PX, px_por_mm=1.0 / MM_POR_PX, valida=True, fuente="aruco")


@pytest.fixture
def escala_invalida() -> Escala:
    return Escala()


def _medida(area_px: float, escala: Escala) -> Medida:
    """Medida sintetica coherente con la escala: los mm2 solo existen si la escala vale."""
    lado = float(np.sqrt(area_px))
    medida = Medida(
        largo_px=lado,
        ancho_px=lado,
        area_px=float(area_px),
        angulo_deg=0.0,
        caja_rotada_px=np.zeros((4, 2), dtype=np.int32),
        centro_px=(0.0, 0.0),
    )
    if escala.valida and escala.mm_por_px:
        medida.largo_mm = lado * escala.mm_por_px
        medida.ancho_mm = lado * escala.mm_por_px
        medida.area_mm2 = float(area_px) * escala.mm_por_px**2
        medida.fiable = True
    return medida


def _deteccion(
    centro: tuple[float, float],
    area_px: float,
    escala: Escala,
    clase: str = "desgastado",
) -> Deteccion:
    cx, cy = centro
    return Deteccion(
        clase_id=1,
        clase=clase,
        confianza=0.9,
        xyxy=(cx - 20.0, cy - 10.0, cx + 20.0, cy + 10.0),
        medida=_medida(area_px, escala),
    )


def test_rect_roi_convierte_normalizado_a_pixeles(roi_centrado: RoiConfig) -> None:
    x1, y1, x2, y2 = rect_roi(FORMA_HD, roi_centrado)

    assert (x1, y1) == (192, 108)      # 0.15 * 1280, 0.15 * 720
    assert (x2, y2) == (1088, 612)     # 0.85 * 1280, 0.85 * 720


def test_rect_roi_deshabilitado_devuelve_el_frame_completo() -> None:
    cfg = RoiConfig(habilitado=False, x=0.15, y=0.15, w=0.7, h=0.7)

    assert rect_roi(FORMA_HD, cfg) == (0, 0, 1280, 720)


def test_rect_roi_no_se_sale_del_frame() -> None:
    """Un ROI que pide mas de lo que hay se recorta a los bordes, no revienta."""
    cfg = RoiConfig(habilitado=True, x=0.9, y=0.9, w=1.0, h=1.0)

    x1, y1, x2, y2 = rect_roi(FORMA_HD, cfg)

    assert 0 <= x1 < x2 <= 1280
    assert 0 <= y1 < y2 <= 720


def test_rect_roi_con_frame_vacio() -> None:
    assert rect_roi((0, 0), RoiConfig()) == (0, 0, 0, 0)


def test_recortar_roi_devuelve_la_region_pedida() -> None:
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    frame[108:612, 192:1088] = 255

    recorte = recortar_roi(frame, (192, 108, 1088, 612))

    assert recorte.shape == (504, 896, 3)
    assert bool((recorte == 255).all())


def test_recortar_roi_acota_un_rectangulo_desbordado() -> None:
    frame = np.zeros((100, 100, 3), dtype=np.uint8)

    recorte = recortar_roi(frame, (-50, -50, 500, 500))

    assert recorte.shape == (100, 100, 3)


def test_centro_en_roi_dentro_y_fuera() -> None:
    roi = (192, 108, 1088, 612)

    assert centro_en_roi((600.0, 300.0, 700.0, 400.0), roi) is True   # centro (650,350)
    assert centro_en_roi((0.0, 0.0, 100.0, 100.0), roi) is False      # centro (50,50)
    assert centro_en_roi((1200.0, 650.0, 1260.0, 700.0), roi) is False


def test_centro_en_roi_admite_la_pieza_que_asoma_del_borde() -> None:
    """Con el centro adentro la pieza cuenta, aunque los bordes se salgan del ROI."""
    roi = (192, 108, 1088, 612)

    assert centro_en_roi((150.0, 300.0, 400.0, 400.0), roi) is True   # centro x=275, adentro


def test_centro_en_roi_incluye_el_borde() -> None:
    roi = (100, 100, 200, 200)

    assert centro_en_roi((140.0, 140.0, 260.0, 260.0), roi) is True   # centro exacto (200,200)


def test_limpiar_mascara_borra_el_polvo_y_conserva_la_pieza() -> None:
    """Un punto de 3 px (aserrin) desaparece; un blob de 60 px (pieza) sobrevive intacto."""
    mascara = np.zeros((200, 200), dtype=np.uint8)
    mascara[10:13, 10:13] = 255        # polvo: 3x3 px
    mascara[100:160, 100:160] = 255    # pieza: 60x60 px

    limpia = limpiar_mascara(mascara, kernel=5, iteraciones=1)

    assert int(limpia[10:13, 10:13].max()) == 0
    assert int(limpia[130, 130]) == 255
    area_pieza = int((limpia[95:165, 95:165] > 0).sum())
    # La apertura solo redondea las esquinas: el area no puede caer mas de un 5%.
    assert area_pieza >= 0.95 * 60 * 60


def test_limpiar_mascara_no_encoge_la_pieza() -> None:
    """La dilatacion posterior devuelve el blob a su tamano: una erosion sola lo adelgazaria."""
    mascara = np.zeros((200, 200), dtype=np.uint8)
    mascara[100:160, 100:160] = 255

    limpia = limpiar_mascara(mascara, kernel=5, iteraciones=1)
    solo_erosionada = cv2.erode(
        mascara, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)), iterations=1
    )

    assert int((limpia > 0).sum()) > int((solo_erosionada > 0).sum())


@pytest.mark.parametrize("kernel,iteraciones", [(1, 1), (0, 1), (5, 0), (3, -2)])
def test_limpiar_mascara_sin_efecto_devuelve_la_mascara_intacta(
    kernel: int, iteraciones: int
) -> None:
    mascara = np.zeros((50, 50), dtype=np.uint8)
    mascara[10:13, 10:13] = 255

    limpia = limpiar_mascara(mascara, kernel=kernel, iteraciones=iteraciones)

    assert np.array_equal(limpia, mascara)


def test_aplicar_filtros_descarta_por_roi(escala_valida: Escala) -> None:
    roi = (192, 108, 1088, 612)
    adentro = _deteccion((640.0, 360.0), area_px=10_000.0, escala=escala_valida)
    afuera = _deteccion((50.0, 50.0), area_px=10_000.0, escala=escala_valida)

    aceptadas, descartadas = aplicar_filtros(
        [adentro, afuera], roi, FiltrosConfig(), escala_valida
    )

    assert aceptadas == [adentro]
    assert adentro.descartada_por is None
    assert [d.descartada_por for d in descartadas] == ["roi"]


def test_aplicar_filtros_descarta_por_area_minima_en_px(escala_invalida: Escala) -> None:
    roi = (0, 0, 1280, 720)
    cfg = FiltrosConfig(area_minima_px=400)
    ruido = _deteccion((640.0, 360.0), area_px=100.0, escala=escala_invalida)

    aceptadas, descartadas = aplicar_filtros([ruido], roi, cfg, escala_invalida)

    assert aceptadas == []
    assert descartadas[0].descartada_por == "area_minima"


def test_aplicar_filtros_descarta_por_area_minima_en_mm2(escala_valida: Escala) -> None:
    """Supera el piso en px pero no la regla fisica: 500 px2 son 20 mm2 a 5 px/mm."""
    roi = (0, 0, 1280, 720)
    cfg = FiltrosConfig(area_minima_px=400, area_minima_mm2=50.0)
    chica = _deteccion((640.0, 360.0), area_px=500.0, escala=escala_valida)

    assert chica.medida is not None
    assert chica.medida.area_mm2 == pytest.approx(20.0)

    aceptadas, descartadas = aplicar_filtros([chica], roi, cfg, escala_valida)

    assert aceptadas == []
    assert descartadas[0].descartada_por == "area_minima"


def test_aplicar_filtros_descarta_por_area_maxima(escala_valida: Escala) -> None:
    roi = (0, 0, 1280, 720)
    cfg = FiltrosConfig(area_maxima_mm2=100.0)
    enorme = _deteccion((640.0, 360.0), area_px=10_000.0, escala=escala_valida)

    aceptadas, descartadas = aplicar_filtros([enorme], roi, cfg, escala_valida)

    assert aceptadas == []
    assert descartadas[0].descartada_por == "area_maxima"


def test_sin_escala_los_limites_en_mm2_no_se_aplican(
    escala_valida: Escala, escala_invalida: Escala
) -> None:
    """Sin marcador no se puede filtrar por mm2: la unica defensa es el umbral en px.

    La misma deteccion de 10.000 px2 se descarta por area_maxima cuando hay escala
    (son 400 mm2, muy por encima del limite de 100) y se acepta cuando no la hay,
    porque supera el piso en pixeles y no existe forma de saber cuantos mm mide.
    """
    roi = (0, 0, 1280, 720)
    cfg = FiltrosConfig(area_minima_px=400, area_minima_mm2=50.0, area_maxima_mm2=100.0)

    con_escala = _deteccion((640.0, 360.0), area_px=10_000.0, escala=escala_valida)
    aceptadas_con, descartadas_con = aplicar_filtros([con_escala], roi, cfg, escala_valida)

    sin_escala = _deteccion((640.0, 360.0), area_px=10_000.0, escala=escala_invalida)
    assert sin_escala.medida is not None
    assert sin_escala.medida.area_mm2 is None
    aceptadas_sin, descartadas_sin = aplicar_filtros([sin_escala], roi, cfg, escala_invalida)

    assert aceptadas_con == [] and descartadas_con[0].descartada_por == "area_maxima"
    assert aceptadas_sin == [sin_escala] and descartadas_sin == []


def test_aplicar_filtros_respeta_el_orden_roi_antes_que_area(escala_valida: Escala) -> None:
    """Una deteccion que falla ROI y area a la vez se reporta como descarte por ROI."""
    roi = (192, 108, 1088, 612)
    cfg = FiltrosConfig(area_minima_px=400)
    doble_falla = _deteccion((50.0, 50.0), area_px=10.0, escala=escala_valida)

    _, descartadas = aplicar_filtros([doble_falla], roi, cfg, escala_valida)

    assert descartadas[0].descartada_por == "roi"


def test_aplicar_filtros_deteccion_sin_medida_solo_pasa_por_roi(escala_valida: Escala) -> None:
    roi = (192, 108, 1088, 612)
    sin_medida = Deteccion(
        clase_id=0, clase="ok", confianza=0.9, xyxy=(600.0, 300.0, 700.0, 400.0), medida=None
    )
    sin_medida_afuera = Deteccion(
        clase_id=0, clase="ok", confianza=0.9, xyxy=(0.0, 0.0, 50.0, 50.0), medida=None
    )

    aceptadas, descartadas = aplicar_filtros(
        [sin_medida, sin_medida_afuera], roi, FiltrosConfig(), escala_valida
    )

    assert aceptadas == [sin_medida]
    assert descartadas[0].descartada_por == "roi"


def test_aplicar_filtros_sin_detecciones(escala_valida: Escala) -> None:
    aceptadas, descartadas = aplicar_filtros([], (0, 0, 100, 100), FiltrosConfig(), escala_valida)

    assert aceptadas == []
    assert descartadas == []


def test_aplicar_filtros_limpia_el_motivo_de_una_deteccion_reciclada(
    escala_valida: Escala,
) -> None:
    """Una deteccion aceptada no puede arrastrar un descartada_por de un frame anterior."""
    roi = (0, 0, 1280, 720)
    reciclada = _deteccion((640.0, 360.0), area_px=10_000.0, escala=escala_valida)
    reciclada.descartada_por = "roi"

    aceptadas, _ = aplicar_filtros([reciclada], roi, FiltrosConfig(), escala_valida)

    assert aceptadas == [reciclada]
    assert reciclada.descartada_por is None
