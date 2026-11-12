"""Pruebas de la comparacion foto contra foto.

QUE RESUELVE ESTO
-----------------
Durante semanas la pantalla mostro un porcentaje que el operario leia como "cuanto se
parece la sierra a la foto de al lado", y no era eso: era la probabilidad que un modelo
entrenado con otras fotos le daba a la clase sana. Por eso aparecio un DESAPROBADO 36% con
las dos imagenes practicamente iguales, y por eso cambiar la foto de referencia no movia el
numero ni un punto.

Ahora, mientras no haya modelo, el numero SI sale de las dos fotos que estan a la vista.

EL COMENTARIO QUE HABIA EN EL CODIGO DECIA QUE ESTO NO SE PODIA HACER, porque "la luz y el
encuadre dominan". Era cierto para una resta pixel a pixel. Normalizando se va la luz y con
matchTemplate se va el encuadre, y eso esta medido abajo.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from afilado import revision
from afilado.config import AppConfig
from afilado.tipos import Veredicto


@pytest.fixture
def cfg() -> AppConfig:
    c = AppConfig(clases=["aprobada", "rechazada"], clases_defecto=["rechazada"])
    c.interfaz.umbral_aprobacion = 0.70
    return c


def _textura(semilla: int = 3, lado: int = 320) -> np.ndarray:
    rng = np.random.default_rng(semilla)
    base = rng.integers(0, 255, (lado, lado, 3), dtype=np.uint8)
    return cv2.GaussianBlur(base, (5, 5), 0)


# --------------------------------------------------------------------- la medida


def test_una_foto_contra_si_misma_da_el_maximo() -> None:
    img = _textura()

    assert revision.parecido(img, img) == pytest.approx(1.0, abs=1e-3)


def test_la_luz_no_cambia_el_resultado() -> None:
    """Lo que hacia inviable comparar fotos del microscopio.

    Medido sobre las fotos reales del taller: 01-LU3F-0300, 02-R01 y 03-FREUD-250 son el
    mismo diente en la misma posicion con distinta exposicion --diferentes en el 100% de
    los pixeles, con una diferencia media de 19 a 39 niveles-- y el parecido da 1.000.
    """
    img = _textura()
    mas_clara = np.clip(img.astype(np.float32) * 1.4 + 30, 0, 255).astype(np.uint8)
    mas_oscura = np.clip(img.astype(np.float32) * 0.6 - 20, 0, 255).astype(np.uint8)

    assert revision.parecido(img, mas_clara) > 0.97
    assert revision.parecido(img, mas_oscura) > 0.97


def test_correr_la_pieza_no_derrumba_el_parecido() -> None:
    """La pieza se apoya a mano: nunca cae dos veces en el mismo lugar."""
    img = _textura()
    M = np.float32([[1, 0, 14], [0, 1, -11]])
    corrida = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]), borderMode=cv2.BORDER_REPLICATE)

    assert revision.parecido(img, corrida) > 0.85


def test_dos_piezas_distintas_dan_bastante_menos() -> None:
    assert revision.parecido(_textura(1), _textura(99)) < 0.5


def test_sin_foto_o_sin_modelo_no_hay_numero() -> None:
    """Mejor nada que un porcentaje inventado: es toda la leccion de este proyecto."""
    img = _textura()

    assert revision.parecido(None, img) is None
    assert revision.parecido(img, None) is None
    assert revision.parecido(None, None) is None


def test_funciona_en_gris_y_en_color() -> None:
    color = _textura()
    gris = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)

    assert revision.parecido(gris, color) == pytest.approx(1.0, abs=1e-2)


def test_el_resultado_siempre_queda_entre_0_y_1() -> None:
    """Se dibuja como porcentaje y como barra: un -0.3 romperia las dos cosas."""
    for a, b in ((_textura(1), _textura(2)), (_textura(5), 255 - _textura(5))):
        v = revision.parecido(a, b)
        assert 0.0 <= v <= 1.0


# --------------------------------------------------------------------- en pantalla


def test_sin_modelo_se_muestra_el_parecido_pero_no_como_veredicto(cfg: AppConfig) -> None:
    """La decision de diseno central de este modo.

    Esta MEDIDO que el parecido no alcanza para juzgar: sobre las fotos del taller, dos
    tomas de la misma pieza dan 78% y dos de piezas distintas 79%. Lo unico que distingue
    con confianza es "es exactamente la misma toma" (100%). Pintarlo de verde o rojo seria
    repetir el DESAPROBADO 36%: un numero con cara de veredicto que no mide lo que el
    operario cree que mide.
    """
    estado = revision.EstadoRevision(congelado=True, veredicto=None, parecido=0.94)

    titulo, detalle, color, aprobado = revision.estado_resultado(estado, cfg)

    assert "94%" in titulo, "el numero se muestra, bien grande"
    assert aprobado is None, "pero no es un aprobado ni un desaprobado"
    assert color == revision._AMBAR
    assert "no es un veredicto" in detalle


def test_un_parecido_bajo_tampoco_desaprueba(cfg: AppConfig) -> None:
    estado = revision.EstadoRevision(congelado=True, veredicto=None, parecido=0.42)

    titulo, _, _, aprobado = revision.estado_resultado(estado, cfg)

    assert "42%" in titulo and aprobado is None


def test_se_puede_encender_el_veredicto_por_parecido(cfg: AppConfig) -> None:
    """Para quien quiera probarlo igual: una llave en el config, no el comportamiento base."""
    cfg.interfaz.veredicto_por_parecido = True
    cfg.interfaz.umbral_parecido = 0.85

    alto = revision.EstadoRevision(congelado=True, parecido=0.94)
    bajo = revision.EstadoRevision(congelado=True, parecido=0.42)

    assert revision.estado_resultado(alto, cfg)[3] is True
    assert revision.estado_resultado(bajo, cfg)[3] is False


def test_la_pantalla_SIEMPRE_dice_de_donde_salio_el_numero(cfg: AppConfig) -> None:
    """La regla que nace del DESAPROBADO 36%.

    Un porcentaje sin su origen es indistinguible de uno inventado: el operario no tiene
    como saber si lo produjo el modelo entrenado o la foto que tiene al lado.
    """
    cfg.interfaz.veredicto_por_parecido = True
    por_foto = revision.EstadoRevision(congelado=True, parecido=0.94)
    por_modelo = revision.EstadoRevision(
        congelado=True,
        veredicto=Veredicto(
            clase="aprobada", clase_id=0, confianza=0.94,
            probabilidades={"aprobada": 0.94, "rechazada": 0.06},
        ),
    )

    _, detalle_foto, _, _ = revision.estado_resultado(por_foto, cfg)
    _, detalle_modelo, _, _ = revision.estado_resultado(por_modelo, cfg)

    assert "foto de modelo" in detalle_foto
    assert "modelo entrenado" in detalle_modelo
    assert detalle_foto != detalle_modelo


def test_con_modelo_entrenado_manda_el_modelo(cfg: AppConfig) -> None:
    """El parecido es el respaldo, no un voto que se promedia con el modelo."""
    estado = revision.EstadoRevision(
        congelado=True,
        parecido=0.99,
        veredicto=Veredicto(
            clase="rechazada", clase_id=1, confianza=0.95,
            probabilidades={"aprobada": 0.05, "rechazada": 0.95},
        ),
    )

    titulo, detalle, _, _ = revision.estado_resultado(estado, cfg)

    assert titulo == "DESAPROBADO"
    assert "5%" in detalle and "modelo entrenado" in detalle


def test_sin_parecido_ni_modelo_sigue_diciendo_sin_modelo(cfg: AppConfig) -> None:
    estado = revision.EstadoRevision(congelado=True, parecido=None, veredicto=None)

    titulo, _, _, aprobado = revision.estado_resultado(estado, cfg)

    assert (titulo, aprobado) == ("SIN MODELO", None)


def test_la_comparacion_de_fotos_usa_SU_umbral(cfg: AppConfig) -> None:
    """Dos escalas distintas no pueden compartir el mismo numero.

    umbral_aprobacion mide una probabilidad del modelo; umbral_parecido mide cuanto se
    parecen dos fotos. Medido sobre las fotos del taller, dos dientes DISTINTOS dan entre
    60% y 71%, asi que con el umbral del modelo (0.70) una pieza equivocada aprobaba.
    """
    cfg.interfaz.umbral_aprobacion = 0.70
    cfg.interfaz.umbral_parecido = 0.85
    cfg.interfaz.veredicto_por_parecido = True
    estado = revision.EstadoRevision(congelado=True, parecido=0.71)

    titulo, _, _, aprobado = revision.estado_resultado(estado, cfg)

    assert (titulo, aprobado) == ("DESAPROBADO", False), "71% es otra pieza, no la misma"


def test_el_umbral_del_modelo_no_se_toca(cfg: AppConfig) -> None:
    """Con modelo entrenado sigue mandando umbral_aprobacion."""
    cfg.interfaz.umbral_aprobacion = 0.70
    cfg.interfaz.umbral_parecido = 0.95
    estado = revision.EstadoRevision(
        congelado=True,
        veredicto=Veredicto(clase="aprobada", clase_id=0, confianza=0.80,
                            probabilidades={"aprobada": 0.80, "rechazada": 0.20}),
    )

    titulo, _, _, _ = revision.estado_resultado(estado, cfg)

    assert titulo == "APROBADO"
