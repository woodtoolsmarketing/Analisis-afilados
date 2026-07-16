"""Pruebas de la medicion geometrica.

El nucleo de este archivo es la prueba de la pieza rotada: verifica que el
rectangulo de area minima mide la pieza real y, en el mismo test, que la caja
recta MIENTE. Esa comparacion es la que justifica el uso de minAreaRect.
"""

from __future__ import annotations

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from afilado.medicion import (
    contorno_principal,
    medir_contorno,
    medir_desde_caja,
    medir_desde_mascara,
)
from afilado.tipos import Escala

# Un rectangulo de 100x50 px a 5 px/mm es una pieza de 20x10 mm.
PX_POR_MM = 5.0
LARGO_PX = 100.0
ANCHO_PX = 50.0
LARGO_MM_ESPERADO = 20.0
ANCHO_MM_ESPERADO = 10.0

# La rasterizacion de la mascara y el redondeo de minAreaRect valen +-1 px sobre
# lados de 50-100 px: cualquier tolerancia mas fina probaria el antialiasing de
# OpenCV, no la matematica del modulo.
TOLERANCIA_RECTO = 0.03
TOLERANCIA_ROTADO = 0.04


@pytest.fixture
def escala_valida() -> Escala:
    """Escala de 5 px/mm proveniente de un marcador visible."""
    return Escala(
        mm_por_px=1.0 / PX_POR_MM,
        px_por_mm=PX_POR_MM,
        valida=True,
        fuente="aruco",
        id_marcador=0,
    )


@pytest.fixture
def escala_invalida() -> Escala:
    """Sin marcador a la vista: no hay conversion posible a milimetros."""
    return Escala()


def _mascara_rectangulo(
    angulo_deg: float,
    largo_px: float = LARGO_PX,
    ancho_px: float = ANCHO_PX,
    lado_lienzo: int = 400,
) -> np.ndarray:
    """Pinta un rectangulo macizo de tamano exacto, girado angulo_deg, en un lienzo negro."""
    mascara = np.zeros((lado_lienzo, lado_lienzo), dtype=np.uint8)
    centro = (lado_lienzo / 2.0, lado_lienzo / 2.0)
    esquinas = cv2.boxPoints((centro, (largo_px, ancho_px), float(angulo_deg)))
    cv2.fillPoly(mascara, [np.int32(np.round(esquinas))], 255)
    return mascara


def _diferencia_angular(uno: float, otro: float) -> float:
    """Distancia entre dos angulos en el circulo de 180 grados (una pieza girada 180 es la misma)."""
    bruta = abs(float(uno) - float(otro)) % 180.0
    return min(bruta, 180.0 - bruta)


def test_rectangulo_recto_mide_20x10_mm(escala_valida: Escala) -> None:
    mascara = _mascara_rectangulo(angulo_deg=0.0)

    medida = medir_desde_mascara(mascara, escala_valida)

    assert medida is not None
    assert medida.fiable is True
    assert medida.largo_mm == pytest.approx(LARGO_MM_ESPERADO, rel=TOLERANCIA_RECTO)
    assert medida.ancho_mm == pytest.approx(ANCHO_MM_ESPERADO, rel=TOLERANCIA_RECTO)
    assert medida.area_mm2 == pytest.approx(
        LARGO_MM_ESPERADO * ANCHO_MM_ESPERADO, rel=2 * TOLERANCIA_RECTO
    )
    assert _diferencia_angular(medida.angulo_deg, 0.0) < 2.0


def test_rectangulo_rotado_mide_igual_y_la_caja_recta_miente(escala_valida: Escala) -> None:
    """La misma pieza girada 30 grados sigue midiendo 20x10 mm con minAreaRect.

    La caja recta que la envuelve mide 111.6 x 93.3 px (20 x 10 px girados 30 grados),
    o sea 22.3 x 18.7 mm: el ancho saldria casi al doble del real. Por eso se mide con
    el rectangulo de area minima y no con la caja alineada a los ejes.
    """
    mascara = _mascara_rectangulo(angulo_deg=30.0)

    rotada = medir_desde_mascara(mascara, escala_valida)

    assert rotada is not None
    assert rotada.largo_mm == pytest.approx(LARGO_MM_ESPERADO, rel=TOLERANCIA_ROTADO)
    assert rotada.ancho_mm == pytest.approx(ANCHO_MM_ESPERADO, rel=TOLERANCIA_ROTADO)
    assert _diferencia_angular(rotada.angulo_deg, 30.0) < 3.0

    contorno = contorno_principal(mascara)
    assert contorno is not None
    x, y, ancho_caja, alto_caja = cv2.boundingRect(contorno)
    recta = medir_desde_caja((x, y, x + ancho_caja, y + alto_caja), escala_valida)

    assert recta.ancho_mm is not None
    assert recta.largo_mm is not None
    # La caja recta infla el ancho a ~18.7 mm: mas de un 50% por encima del real.
    assert recta.ancho_mm > 15.0
    assert abs(recta.ancho_mm - ANCHO_MM_ESPERADO) > abs(rotada.ancho_mm - ANCHO_MM_ESPERADO)
    assert abs(recta.largo_mm - LARGO_MM_ESPERADO) > abs(rotada.largo_mm - LARGO_MM_ESPERADO)


@pytest.mark.parametrize("angulo", [0.0, 15.0, 30.0, 45.0, 60.0, 75.0])
def test_minarearect_es_invariante_al_giro(escala_valida: Escala, angulo: float) -> None:
    """Girar la pieza no puede cambiar cuanto mide."""
    medida = medir_desde_mascara(_mascara_rectangulo(angulo_deg=angulo), escala_valida)

    assert medida is not None
    assert medida.largo_mm == pytest.approx(LARGO_MM_ESPERADO, rel=TOLERANCIA_ROTADO)
    assert medida.ancho_mm == pytest.approx(ANCHO_MM_ESPERADO, rel=TOLERANCIA_ROTADO)


def test_sin_escala_valida_no_hay_milimetros(escala_invalida: Escala) -> None:
    """Nunca se inventan milimetros: sin marcador solo hay pixeles."""
    medida = medir_desde_mascara(_mascara_rectangulo(angulo_deg=0.0), escala_invalida)

    assert medida is not None
    assert medida.largo_mm is None
    assert medida.ancho_mm is None
    assert medida.area_mm2 is None
    assert medida.fiable is False
    # Los pixeles se miden igual: lo unico que falta es la conversion.
    assert medida.largo_px == pytest.approx(LARGO_PX, rel=TOLERANCIA_RECTO)
    assert medida.ancho_px == pytest.approx(ANCHO_PX, rel=TOLERANCIA_RECTO)


def test_escala_presente_pero_no_valida_tampoco_convierte() -> None:
    """valida=False manda, aunque el mm_por_px este cargado (marcador perdido hace rato)."""
    escala = Escala(mm_por_px=0.2, px_por_mm=5.0, valida=False, fuente="ninguna")

    medida = medir_desde_mascara(_mascara_rectangulo(angulo_deg=0.0), escala)

    assert medida is not None
    assert medida.largo_mm is None
    assert medida.fiable is False


def test_area_sale_del_contorno_real_y_no_del_rectangulo(escala_valida: Escala) -> None:
    """Un triangulo ocupa la mitad de su caja: el area debe reflejarlo."""
    mascara = np.zeros((400, 400), dtype=np.uint8)
    triangulo = np.array([[100, 300], [300, 300], [200, 100]], dtype=np.int32)
    cv2.fillPoly(mascara, [triangulo], 255)

    medida = medir_desde_mascara(mascara, escala_valida)

    assert medida is not None
    area_triangulo_px = 200.0 * 200.0 / 2.0
    assert medida.area_px == pytest.approx(area_triangulo_px, rel=0.03)
    assert medida.area_mm2 == pytest.approx(
        area_triangulo_px / (PX_POR_MM**2), rel=0.03
    )


def test_conversion_de_area_usa_el_factor_al_cuadrado(escala_valida: Escala) -> None:
    medida = medir_desde_mascara(_mascara_rectangulo(angulo_deg=20.0), escala_valida)

    assert medida is not None
    assert medida.area_mm2 == pytest.approx(
        medida.area_px * escala_valida.mm_por_px**2, rel=1e-6
    )
    assert medida.largo_mm == pytest.approx(medida.largo_px * escala_valida.mm_por_px, rel=1e-6)


def test_contorno_principal_elige_el_de_mayor_area() -> None:
    """El aserrin al lado de la pieza no debe robarle el protagonismo."""
    mascara = np.zeros((400, 400), dtype=np.uint8)
    cv2.rectangle(mascara, (20, 20), (29, 29), 255, -1)      # ruido: 100 px2
    cv2.rectangle(mascara, (150, 150), (249, 249), 255, -1)  # pieza: 10000 px2

    contorno = contorno_principal(mascara)

    assert contorno is not None
    assert cv2.contourArea(contorno) == pytest.approx(100.0 * 100.0, rel=0.05)


def test_contorno_principal_con_mascara_vacia_devuelve_none() -> None:
    assert contorno_principal(np.zeros((100, 100), dtype=np.uint8)) is None


def test_medir_desde_mascara_vacia_devuelve_none(escala_valida: Escala) -> None:
    assert medir_desde_mascara(np.zeros((100, 100), dtype=np.uint8), escala_valida) is None


def test_medir_desde_caja_usa_el_area_de_la_caja_recta(escala_valida: Escala) -> None:
    """Sin mascara no hay forma real: el area es la de la caja y el angulo es 0."""
    medida = medir_desde_caja((10.0, 20.0, 110.0, 70.0), escala_valida)

    assert medida.largo_px == pytest.approx(100.0)
    assert medida.ancho_px == pytest.approx(50.0)
    assert medida.area_px == pytest.approx(100.0 * 50.0)
    assert medida.angulo_deg == 0.0
    assert medida.centro_px == pytest.approx((60.0, 45.0))
    assert medida.largo_mm == pytest.approx(LARGO_MM_ESPERADO)
    assert medida.ancho_mm == pytest.approx(ANCHO_MM_ESPERADO)


def test_medir_desde_caja_acepta_esquinas_invertidas(escala_valida: Escala) -> None:
    normal = medir_desde_caja((10.0, 20.0, 110.0, 70.0), escala_valida)
    invertida = medir_desde_caja((110.0, 70.0, 10.0, 20.0), escala_valida)

    assert invertida.largo_px == pytest.approx(normal.largo_px)
    assert invertida.ancho_px == pytest.approx(normal.ancho_px)
    assert invertida.area_px == pytest.approx(normal.area_px)


def test_caja_rotada_tiene_cuatro_esquinas_enteras(escala_valida: Escala) -> None:
    medida = medir_desde_mascara(_mascara_rectangulo(angulo_deg=30.0), escala_valida)

    assert medida is not None
    assert medida.caja_rotada_px.shape == (4, 2)
    assert medida.caja_rotada_px.dtype == np.int32


def test_angulo_siempre_normalizado_entre_0_y_180(escala_valida: Escala) -> None:
    for angulo in (-150.0, -30.0, 0.0, 95.0, 170.0, 200.0):
        medida = medir_desde_mascara(_mascara_rectangulo(angulo_deg=angulo), escala_valida)
        assert medida is not None
        assert 0.0 <= medida.angulo_deg < 180.0


def test_medir_contorno_acepta_el_formato_nativo_de_opencv(escala_valida: Escala) -> None:
    """findContours entrega (N,1,2); medir_contorno debe tragarlo sin reshape previo."""
    mascara = _mascara_rectangulo(angulo_deg=0.0)
    salida = cv2.findContours(mascara, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contorno = max(salida[-2], key=cv2.contourArea)

    medida = medir_contorno(contorno, escala_valida)

    assert medida.largo_mm == pytest.approx(LARGO_MM_ESPERADO, rel=TOLERANCIA_RECTO)


def test_centro_medido_coincide_con_el_centro_pintado(escala_valida: Escala) -> None:
    medida = medir_desde_mascara(_mascara_rectangulo(angulo_deg=30.0, lado_lienzo=400), escala_valida)

    assert medida is not None
    assert medida.centro_px[0] == pytest.approx(200.0, abs=2.0)
    assert medida.centro_px[1] == pytest.approx(200.0, abs=2.0)
