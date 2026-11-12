"""Pruebas del dibujo de texto.

Lo que importa aca es que la pantalla siga siendo legible en las DOS rutas: con Pillow y una
fuente TrueType del sistema, y sin ellos. La segunda no es teorica: el .exe del taller se
arma desde un entorno liviano, y si alguna vez se rearma sin Pillow la pantalla tiene que
seguir dibujandose, no reventar al primer texto.
"""

from __future__ import annotations

import numpy as np
import pytest

from afilado import tipografia


def _lienzo(ancho: int = 400, alto: int = 120) -> np.ndarray:
    return np.zeros((alto, ancho, 3), np.uint8)


def test_medir_devuelve_algo_razonable() -> None:
    ancho, alto = tipografia.medir("APROBADO", 40, negrita=True)

    assert ancho > 0 and alto > 0
    assert ancho > alto, "una palabra de ocho letras tiene que ser mas ancha que alta"


def test_el_texto_vacio_no_mide_ni_dibuja() -> None:
    assert tipografia.medir("", 40) == (0, 0)

    lienzo = _lienzo()
    with tipografia.Pincel(lienzo) as pincel:
        pincel.texto("", (10, 10), (255, 255, 255), 40)

    assert int(lienzo.max()) == 0


def test_mas_grande_mide_mas() -> None:
    chico, _ = tipografia.medir("DESAPROBADO", 20, negrita=True)
    grande, _ = tipografia.medir("DESAPROBADO", 50, negrita=True)

    assert grande > chico


def test_el_texto_se_dibuja_de_verdad() -> None:
    lienzo = _lienzo()

    with tipografia.Pincel(lienzo) as pincel:
        pincel.texto("APROBADO", (10, 30), (255, 255, 255), 36, negrita=True)

    assert int(lienzo.max()) > 200, "no quedo nada dibujado"


def test_el_texto_arranca_donde_se_lo_pide() -> None:
    """org es la esquina SUPERIOR IZQUIERDA, no la linea base de cv2.putText.

    Apilar y centrar bloques es sumar altos; con la baseline habria que arrastrar un
    descuento en cada cuenta y es donde se cuelan los solapamientos.
    """
    lienzo = _lienzo()
    org = (40, 30)
    ancho, alto = tipografia.medir("HOLA", 34, negrita=True)

    with tipografia.Pincel(lienzo) as pincel:
        pincel.texto("HOLA", org, (255, 255, 255), 34, negrita=True)

    pintados = np.argwhere(lienzo.max(axis=2) > 60)
    assert pintados.size > 0
    arriba, izquierda = pintados[:, 0].min(), pintados[:, 1].min()
    abajo, derecha = pintados[:, 0].max(), pintados[:, 1].max()

    assert abs(int(izquierda) - org[0]) <= 4, f"empieza en x={izquierda}, se pidio {org[0]}"
    assert abs(int(arriba) - org[1]) <= 6, f"empieza en y={arriba}, se pidio {org[1]}"
    assert derecha - izquierda <= ancho + 6
    assert abajo - arriba <= alto + 6


def test_sin_pillow_sigue_dibujando(monkeypatch: pytest.MonkeyPatch) -> None:
    """La caida a Hershey tiene que dibujar algo legible, no fallar.

    Es el escenario de un .exe rearmado sin Pillow: feo, pero funcionando.
    """
    monkeypatch.setattr(tipografia, "_HAY_PILLOW", False)
    tipografia._cache.clear()
    try:
        lienzo = _lienzo()

        ancho, alto = tipografia.medir("APROBADO", 40, negrita=True)
        with tipografia.Pincel(lienzo) as pincel:
            pincel.texto("APROBADO", (10, 30), (255, 255, 255), 40, negrita=True)

        assert ancho > 0 and alto > 0
        assert int(lienzo.max()) > 200, "sin Pillow no dibujo nada"
    finally:
        tipografia._cache.clear()


def test_la_pantalla_entera_se_dibuja_sin_pillow(monkeypatch: pytest.MonkeyPatch) -> None:
    """El modo simple completo tiene que armarse igual sin Pillow."""
    from afilado import revision
    from afilado.config import AppConfig

    monkeypatch.setattr(tipografia, "_HAY_PILLOW", False)
    tipografia._cache.clear()
    try:
        cfg = AppConfig(clases=["aprobada", "rechazada"], clases_defecto=["rechazada"])
        imagen = np.full((480, 640, 3), 120, np.uint8)

        pantalla = revision.componer_simple(
            imagen, imagen, cfg, revision.EstadoRevision(congelado=True), 672
        )

        assert pantalla.imagen.shape[1] == revision.ANCHO_LIENZO
        assert set(pantalla.botones) == {"seguir", "aprobada", "rechazada", "conceptos", "salir"}
    finally:
        tipografia._cache.clear()
