"""¿La memoria dice lo MISMO si el operario vuelve a apoyar la pieza?

Es la pregunta de precision que mas le importa al taller, y no es la misma que "cuanto
acierta": un veredicto que cambia porque la sierra se apoyo tres milimetros mas alla es
inservible aunque acierte seguido. Quien lo usa pierde la confianza a la segunda vez que le
pasa, y despues ya no la recupera.

MEDIDO SOBRE LAS 19 FOTOS DE MICROSCOPIO DEL TALLER, perturbandolas como pasa de verdad al
volver a apoyar la pieza:

    quieta (control)        19 de 19     parecido 1.000
    corrida 10 px           19 de 19              0.991
    corrida 25 px           19 de 19              0.971
    corrida 50 px           19 de 19              0.923
    girada 3 grados         17 de 19              0.953
    girada 7 grados         16 de 19              0.867
    girada 15 grados        10 de 19              0.778
    mas luz (+25%)          19 de 19              0.974
    menos luz (-25%)        19 de 19              1.000
    reapoyada de verdad     17 de 19              0.914   (14 px + 4 grados + 12% de luz)

La conclusion practica, que esta escrita tambien en el README: correr la pieza y cambiar la
luz NO mueven el veredicto; GIRARLA si. Conviene apoyarla siempre con la misma orientacion.

Estas pruebas repiten la medicion con imagenes sinteticas --las fotos del taller no viajan
con el repositorio-- para que si alguien cambia la formula del parecido y rompe esta
propiedad, se entere aca y no en el taller.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from afilado.memoria import Memoria


def _diente(semilla: int = 0, lado: int = 240) -> np.ndarray:
    img = np.full((lado, lado, 3), 60, np.uint8)
    cv2.fillPoly(img, [np.array([[10, lado - 20], [10, 120 + semilla * 4],
                                 [lado - 60, 40], [lado - 20, 70],
                                 [lado - 20, lado - 20]])], (190, 190, 190))
    cv2.line(img, (10, 120 + semilla * 4), (lado - 60, 40), (235, 235, 235), 6)
    for i in range(0, lado, 7):
        cv2.line(img, (i, 0), (i + 40 + semilla * 9, lado), (0, 0, 0), 1)
    return cv2.GaussianBlur(img, (5, 5), 0)


def _astillado(semilla: int = 0, lado: int = 240) -> np.ndarray:
    img = np.full((lado, lado, 3), 200, np.uint8)
    cv2.fillPoly(img, [np.array([[0, 0], [lado, 0], [lado, 90 + semilla * 5],
                                 [0, 60 + semilla * 5]])], (40, 40, 40))
    for i in range(0, lado, 11):
        cv2.circle(img, (i, 150 + (i * semilla) % 40), 6, (90, 90, 90), -1)
    return cv2.GaussianBlur(img, (5, 5), 0)


def _reapoyar(img: np.ndarray, dx: int = 0, dy: int = 0, grados: float = 0.0,
              luz: float = 1.0, suma: float = 0.0) -> np.ndarray:
    """La misma pieza, vuelta a poner bajo el microscopio."""
    alto, ancho = img.shape[:2]
    M = cv2.getRotationMatrix2D((ancho / 2, alto / 2), grados, 1.0)
    M[0, 2] += dx
    M[1, 2] += dy
    movida = cv2.warpAffine(img, M, (ancho, alto), borderMode=cv2.BORDER_REPLICATE)
    return np.clip(movida.astype(np.float32) * luz + suma, 0, 255).astype(np.uint8)


@pytest.fixture
def memoria(tmp_path: Path) -> Memoria:
    base = tmp_path / "coleccion"
    mapa: dict[str, list[Path]] = {"aprobada": [], "rechazada": []}
    for i in range(5):
        for concepto, imagen in (("aprobada", _diente(i)), ("rechazada", _astillado(i))):
            ruta = base / concepto / f"{i}.jpg"
            ruta.parent.mkdir(parents=True, exist_ok=True)
            assert cv2.imwrite(str(ruta), imagen)
            mapa[concepto].append(ruta)
    mem = Memoria(str(tmp_path / "memoria"))
    mem.sincronizar(mapa)
    return mem


def _concepto(mem: Memoria, imagen: np.ndarray) -> str | None:
    return mem.juzgar(imagen, minimo_por_concepto=2, confianza_minima=0.0).concepto


def test_el_veredicto_es_el_mismo_dos_veces_seguidas(memoria: Memoria) -> None:
    """El control. Si esto fallara, nada de lo demas significaria algo."""
    pieza = _diente(2)

    primero = _concepto(memoria, pieza)
    segundo = _concepto(memoria, pieza.copy())

    assert primero == segundo == "aprobada"


def test_correr_la_pieza_no_cambia_el_veredicto(memoria: Memoria) -> None:
    """La pieza se apoya a mano: nunca cae en el mismo lugar."""
    pieza = _diente(2)
    esperado = _concepto(memoria, pieza)

    for dx, dy in ((10, -6), (25, -18), (-30, 20)):
        assert _concepto(memoria, _reapoyar(pieza, dx=dx, dy=dy)) == esperado


def test_cambiar_la_luz_no_cambia_el_veredicto(memoria: Memoria) -> None:
    """La lampara del microscopio no es estable, y el filo no cambia porque cambie la luz."""
    pieza = _diente(2)
    esperado = _concepto(memoria, pieza)

    assert _concepto(memoria, _reapoyar(pieza, luz=1.25, suma=15)) == esperado
    assert _concepto(memoria, _reapoyar(pieza, luz=0.75, suma=-10)) == esperado


def test_un_giro_chico_todavia_lo_aguanta(memoria: Memoria) -> None:
    pieza = _diente(2)
    esperado = _concepto(memoria, pieza)

    assert _concepto(memoria, _reapoyar(pieza, grados=3.0)) == esperado


def test_el_parecido_baja_al_girar_mas_que_al_correr(memoria: Memoria) -> None:
    """La limitacion conocida, puesta por escrito en una prueba: no se corrige rotacion.

    Medido sobre las fotos del taller: corrida 50 px el parecido queda en 0.923 y el
    veredicto no cambia en ninguna de las 19; girada 15 grados cae a 0.778 y cambia en 9.
    Si algun dia se agrega busqueda de angulo, esta prueba es la que tiene que cambiar.
    """
    pieza = _diente(2)
    corrida = memoria.juzgar(_reapoyar(pieza, dx=30, dy=-20), minimo_por_concepto=2,
                             confianza_minima=0.0)
    girada = memoria.juzgar(_reapoyar(pieza, grados=15.0), minimo_por_concepto=2,
                            confianza_minima=0.0)

    assert girada.parecido < corrida.parecido
