"""Pruebas del calibrador ArUco: escala recuperada, extrapolacion y suavizado EMA.

Los marcadores se generan con dibujar_marcador y se pegan en un lienzo blanco con
un margen conocido, de modo que el lado en pixeles es exacto por construccion y la
escala esperada se conoce de antemano.
"""

from __future__ import annotations

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

pytestmark = pytest.mark.skipif(
    not hasattr(cv2, "aruco"),
    reason="cv2.aruco solo viene en opencv-contrib-python",
)

from afilado.calibracion import CalibradorAruco, dibujar_marcador, obtener_diccionario
from afilado.config import ArucoConfig

DICCIONARIO = "DICT_4X4_50"
LADO_MM = 30.0
MARGEN_PX = 80  # zona blanca de silencio: sin ella detectMarkers no engancha el marcador

# El detector trabaja con esquinas enteras (sin refinado sub-pixel), asi que el lado
# medido puede desviarse ~1 px sobre 200: menos del 1%.
TOLERANCIA_ESCALA = 0.02


def _lienzo_blanco(lado: int) -> np.ndarray:
    return np.full((lado, lado, 3), 255, dtype=np.uint8)


def _lienzo_con_marcador(lado_px: int = 200, id_marcador: int = 0) -> np.ndarray:
    """Marcador de lado_px exactos, centrado en un lienzo blanco con margen."""
    marcador = dibujar_marcador(DICCIONARIO, id_marcador, lado_px)
    lienzo = _lienzo_blanco(lado_px + 2 * MARGEN_PX)
    lienzo[
        MARGEN_PX : MARGEN_PX + lado_px, MARGEN_PX : MARGEN_PX + lado_px
    ] = cv2.cvtColor(marcador, cv2.COLOR_GRAY2BGR)
    return lienzo


def _lienzo_con_dos_marcadores(ids: tuple[int, int], lado_px: int = 160) -> np.ndarray:
    alto = lado_px + 2 * MARGEN_PX
    ancho = 2 * lado_px + 3 * MARGEN_PX
    lienzo = np.full((alto, ancho, 3), 255, dtype=np.uint8)
    for indice, id_marcador in enumerate(ids):
        inicio_x = MARGEN_PX + indice * (lado_px + MARGEN_PX)
        marcador = dibujar_marcador(DICCIONARIO, id_marcador, lado_px)
        lienzo[
            MARGEN_PX : MARGEN_PX + lado_px, inicio_x : inicio_x + lado_px
        ] = cv2.cvtColor(marcador, cv2.COLOR_GRAY2BGR)
    return lienzo


def _cfg(**cambios: object) -> ArucoConfig:
    base: dict[str, object] = {
        "habilitado": True,
        "diccionario": DICCIONARIO,
        "lado_mm": LADO_MM,
        "id_referencia": 0,
        "suavizado": 1.0,  # sin memoria salvo que el test la pida explicitamente
        "max_frames_sin_marcador": 90,
        "max_error_lados": 0.06,
    }
    base.update(cambios)
    return ArucoConfig(**base)  # type: ignore[arg-type]


def test_estimar_recupera_la_escala_con_menos_de_2_por_ciento_de_error() -> None:
    """Marcador de 30 mm dibujado con 200 px de lado => 0.15 mm/px."""
    lado_px = 200
    calibrador = CalibradorAruco(_cfg())

    escala = calibrador.estimar(_lienzo_con_marcador(lado_px=lado_px))

    assert escala.valida is True
    assert escala.fuente == "aruco"
    assert escala.id_marcador == 0
    assert escala.mm_por_px == pytest.approx(LADO_MM / lado_px, rel=TOLERANCIA_ESCALA)
    assert escala.px_por_mm == pytest.approx(lado_px / LADO_MM, rel=TOLERANCIA_ESCALA)
    assert escala.frames_sin_marcador == 0


def test_mm_por_px_y_px_por_mm_son_inversos() -> None:
    calibrador = CalibradorAruco(_cfg())

    escala = calibrador.estimar(_lienzo_con_marcador())

    assert escala.mm_por_px is not None and escala.px_por_mm is not None
    assert escala.mm_por_px * escala.px_por_mm == pytest.approx(1.0, rel=1e-9)


@pytest.mark.parametrize("lado_px", [120, 200, 320])
def test_la_escala_escala_con_el_tamano_aparente(lado_px: int) -> None:
    """Acercar la camara agranda el marcador y achica el mm/px en la misma proporcion."""
    calibrador = CalibradorAruco(_cfg())

    escala = calibrador.estimar(_lienzo_con_marcador(lado_px=lado_px))

    assert escala.mm_por_px == pytest.approx(LADO_MM / lado_px, rel=TOLERANCIA_ESCALA)


def test_marcador_frontal_no_reporta_camara_inclinada() -> None:
    """Los 4 lados de un marcador plano y perpendicular miden casi lo mismo."""
    calibrador = CalibradorAruco(_cfg())

    escala = calibrador.estimar(_lienzo_con_marcador())

    assert escala.error_lados < 0.02


def test_esquinas_devueltas_encuadran_el_marcador_dibujado() -> None:
    lado_px = 200
    calibrador = CalibradorAruco(_cfg())

    escala = calibrador.estimar(_lienzo_con_marcador(lado_px=lado_px))

    assert escala.esquinas is not None
    assert escala.esquinas.shape == (4, 2)
    assert escala.esquinas[:, 0].min() == pytest.approx(MARGEN_PX, abs=3.0)
    assert escala.esquinas[:, 0].max() == pytest.approx(MARGEN_PX + lado_px, abs=3.0)


def test_frame_en_gris_tambien_sirve() -> None:
    calibrador = CalibradorAruco(_cfg())
    gris = cv2.cvtColor(_lienzo_con_marcador(), cv2.COLOR_BGR2GRAY)

    assert calibrador.estimar(gris).valida is True


def test_estimar_no_modifica_el_frame() -> None:
    calibrador = CalibradorAruco(_cfg())
    frame = _lienzo_con_marcador()
    copia = frame.copy()

    calibrador.estimar(frame)

    assert np.array_equal(frame, copia)


def test_id_referencia_none_elige_el_marcador_de_menor_id() -> None:
    calibrador = CalibradorAruco(_cfg(id_referencia=None))

    escala = calibrador.estimar(_lienzo_con_dos_marcadores(ids=(5, 2)))

    assert escala.valida is True
    assert escala.id_marcador == 2


def test_se_elige_el_id_de_referencia_y_no_otro() -> None:
    calibrador = CalibradorAruco(_cfg(id_referencia=5))

    escala = calibrador.estimar(_lienzo_con_dos_marcadores(ids=(5, 2)))

    assert escala.id_marcador == 5


def test_marcador_presente_pero_de_otro_id_no_calibra() -> None:
    """Que se cuele un ArUco ajeno en la escena no puede fijar la escala."""
    calibrador = CalibradorAruco(_cfg(id_referencia=7))

    escala = calibrador.estimar(_lienzo_con_marcador(id_marcador=0))

    assert escala.valida is False
    assert escala.fuente == "ninguna"
    assert escala.frames_sin_marcador == 1


def test_sin_marcador_extrapola_y_luego_invalida() -> None:
    """La mano del operario tapa el marcador: la escala aguanta unos frames y despues cae."""
    limite = 3
    calibrador = CalibradorAruco(_cfg(max_frames_sin_marcador=limite))
    original = calibrador.estimar(_lienzo_con_marcador())
    assert original.valida is True
    mm_por_px_original = original.mm_por_px
    vacio = _lienzo_blanco(360)

    for numero in range(1, limite + 1):
        escala = calibrador.estimar(vacio)
        assert escala.valida is True
        assert escala.fuente == "aruco_extrapolado"
        assert escala.frames_sin_marcador == numero
        assert escala.mm_por_px == pytest.approx(mm_por_px_original)

    caida = calibrador.estimar(vacio)

    assert caida.valida is False
    assert caida.fuente == "ninguna"
    assert caida.mm_por_px is None


def test_sin_marcador_desde_el_arranque_no_extrapola_nada() -> None:
    calibrador = CalibradorAruco(_cfg())

    escala = calibrador.estimar(_lienzo_blanco(360))

    assert escala.valida is False
    assert escala.fuente == "ninguna"
    assert escala.mm_por_px is None


def test_reaparecer_el_marcador_repone_la_escala() -> None:
    calibrador = CalibradorAruco(_cfg(max_frames_sin_marcador=1))
    calibrador.estimar(_lienzo_con_marcador())
    calibrador.estimar(_lienzo_blanco(360))
    calibrador.estimar(_lienzo_blanco(360))

    recuperada = calibrador.estimar(_lienzo_con_marcador())

    assert recuperada.valida is True
    assert recuperada.fuente == "aruco"
    assert recuperada.frames_sin_marcador == 0


def test_la_primera_medicion_no_se_suaviza() -> None:
    """Sin valor previo el EMA no tiene con que promediar: toma el medido tal cual."""
    lado_px = 200
    calibrador = CalibradorAruco(_cfg(suavizado=0.1))

    escala = calibrador.estimar(_lienzo_con_marcador(lado_px=lado_px))

    assert escala.px_por_mm == pytest.approx(lado_px / LADO_MM, rel=TOLERANCIA_ESCALA)


def test_el_ema_amortigua_el_salto_y_converge() -> None:
    """Con a=0.5 el primer frame tras el cambio queda a mitad de camino y luego converge.

    Justifica el suavizado: la escala no salta de golpe ante una medicion nueva, pero
    tampoco se queda anclada en la vieja.
    """
    calibrador = CalibradorAruco(_cfg(suavizado=0.5))
    inicial = calibrador.estimar(_lienzo_con_marcador(lado_px=200)).px_por_mm
    assert inicial is not None

    frame_chico = _lienzo_con_marcador(lado_px=100)
    objetivo = 100 / LADO_MM

    primero = calibrador.estimar(frame_chico).px_por_mm
    assert primero is not None
    assert primero == pytest.approx((inicial + objetivo) / 2.0, rel=0.05)
    assert objetivo < primero < inicial

    previo = primero
    for _ in range(20):
        actual = calibrador.estimar(frame_chico).px_por_mm
        assert actual is not None
        assert actual <= previo + 1e-9  # se acerca al objetivo sin rebotar
        previo = actual

    assert previo == pytest.approx(objetivo, rel=TOLERANCIA_ESCALA)


def test_suavizado_1_sigue_a_la_medicion_sin_memoria() -> None:
    calibrador = CalibradorAruco(_cfg(suavizado=1.0))
    calibrador.estimar(_lienzo_con_marcador(lado_px=200))

    escala = calibrador.estimar(_lienzo_con_marcador(lado_px=100))

    assert escala.px_por_mm == pytest.approx(100 / LADO_MM, rel=TOLERANCIA_ESCALA)


def test_reiniciar_olvida_la_escala_y_el_suavizado() -> None:
    calibrador = CalibradorAruco(_cfg(suavizado=0.5))
    calibrador.estimar(_lienzo_con_marcador(lado_px=200))

    calibrador.reiniciar()

    assert calibrador.ultima_escala.valida is False
    assert calibrador.ultima_escala.mm_por_px is None
    # Sin memoria, la primera medicion posterior vuelve a tomarse sin suavizar.
    escala = calibrador.estimar(_lienzo_con_marcador(lado_px=100))
    assert escala.px_por_mm == pytest.approx(100 / LADO_MM, rel=TOLERANCIA_ESCALA)


def test_reiniciar_impide_extrapolar_la_escala_vieja() -> None:
    calibrador = CalibradorAruco(_cfg(max_frames_sin_marcador=90))
    calibrador.estimar(_lienzo_con_marcador())

    calibrador.reiniciar()
    escala = calibrador.estimar(_lienzo_blanco(360))

    assert escala.valida is False


def test_ultima_escala_refleja_la_ultima_estimacion() -> None:
    calibrador = CalibradorAruco(_cfg())

    assert calibrador.ultima_escala.valida is False
    escala = calibrador.estimar(_lienzo_con_marcador())
    assert calibrador.ultima_escala is escala


def test_calibrador_deshabilitado_no_calibra() -> None:
    calibrador = CalibradorAruco(_cfg(habilitado=False))

    escala = calibrador.estimar(_lienzo_con_marcador())

    assert escala.valida is False
    assert escala.fuente == "ninguna"
    assert escala.mm_por_px is None


def test_dibujar_marcador_devuelve_la_imagen_del_tamano_pedido() -> None:
    imagen = dibujar_marcador(DICCIONARIO, 0, 120)

    assert imagen.shape == (120, 120)
    assert imagen.dtype == np.uint8
    # Un marcador ArUco es binario: solo negro y blanco.
    assert set(np.unique(imagen).tolist()) <= {0, 255}


def test_dibujar_marcador_rechaza_argumentos_imposibles() -> None:
    with pytest.raises(ValueError):
        dibujar_marcador(DICCIONARIO, 0, 0)
    with pytest.raises(ValueError):
        dibujar_marcador(DICCIONARIO, -1, 100)


def test_obtener_diccionario_conocido_y_desconocido() -> None:
    assert obtener_diccionario(DICCIONARIO) is not None

    with pytest.raises(ValueError, match="[Dd]iccionario"):
        obtener_diccionario("DICT_INEXISTENTE_9X9")
