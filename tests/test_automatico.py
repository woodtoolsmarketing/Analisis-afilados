"""Pruebas de la captura automatica: anotar los dientes solo, mientras pasa la sierra.

LO QUE MAS SE CUIDA ACA es que una sierra QUIETA sobre la mesa no se anote sola. Es el
peor error posible de esta funcion: el informe diria que la sierra se reviso entera, diente
por diente, sin que nadie la haya tocado, y ese informe es justamente el que se usa para
contestar un reclamo.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from afilado.automatico import CapturaAutomatica, Resultado, distancia, huella


def _diente(semilla: int = 0, lado: int = 240) -> np.ndarray:
    img = np.full((lado, lado, 3), 60, np.uint8)
    cv2.fillPoly(img, [np.array([[10, lado - 20], [10, 120 + semilla * 7],
                                 [lado - 60, 40], [lado - 20, 70],
                                 [lado - 20, lado - 20]])], (190, 190, 190))
    cv2.line(img, (10, 120 + semilla * 7), (lado - 60, 40), (235, 235, 235), 6)
    for i in range(0, lado, 7):
        cv2.line(img, (i, 0), (i + 40 + semilla * 11, lado), (0, 0, 0), 1)
    return cv2.GaussianBlur(img, (5, 5), 0)


def _temblor(imagen: np.ndarray) -> np.ndarray:
    """La misma pieza quieta: la camara igual no entrega dos frames identicos."""
    ruido = np.random.default_rng(7).integers(-2, 3, imagen.shape, dtype=np.int16)
    return np.clip(imagen.astype(np.int16) + ruido, 0, 255).astype(np.uint8)


BIEN = Resultado(estado="afilado", parecido=0.92, aprobado=True, origen="la memoria")
MAL = Resultado(estado="roto", parecido=0.88, aprobado=False, origen="la memoria")


def _pasar(captura: CapturaAutomatica, imagen, resultado, veces: int, desde: float = 0.0):
    """Le pasa `veces` fotos, una por segundo. Devuelve lo que confirmo en cada una."""
    salidas = []
    for i in range(veces):
        salidas.append(captura.mirar(imagen, desde + i * 1.0, resultado))
    return salidas


# ---------------------------------------------------------------------------
# Confirmar un diente
# ---------------------------------------------------------------------------


def test_cinco_fotos_iguales_dan_el_diente_por_revisado() -> None:
    captura = CapturaAutomatica()

    salidas = _pasar(captura, _diente(), BIEN, 5)

    assert salidas[:4] == [None, None, None, None]
    assert salidas[4] is BIEN


def test_con_cuatro_no_alcanza() -> None:
    captura = CapturaAutomatica()

    assert all(s is None for s in _pasar(captura, _diente(), BIEN, 4))


def test_si_el_veredicto_cambia_se_empieza_de_nuevo() -> None:
    """Un veredicto que baila no es un diente revisado: es una imagen que no se decide."""
    captura = CapturaAutomatica()
    imagen = _diente()

    captura.mirar(imagen, 0.0, BIEN)
    captura.mirar(imagen, 1.0, BIEN)
    captura.mirar(imagen, 2.0, MAL)          # cambio de opinion: se reinicia
    captura.mirar(imagen, 3.0, MAL)
    assert captura.mirar(imagen, 4.0, MAL) is None   # recien va 3 de 5
    assert captura.mirar(imagen, 5.0, MAL) is None
    assert captura.mirar(imagen, 6.0, MAL) is MAL


def test_si_la_pieza_se_mueve_se_empieza_de_nuevo() -> None:
    captura = CapturaAutomatica()

    captura.mirar(_diente(0), 0.0, BIEN)
    captura.mirar(_diente(0), 1.0, BIEN)
    captura.mirar(_diente(5), 2.0, BIEN)     # otra pieza
    assert captura.mirar(_diente(5), 3.0, BIEN) is None
    assert captura.iguales == 2


def test_el_temblor_de_la_camara_no_corta_la_racha() -> None:
    """Un sensor real nunca entrega dos frames identicos; eso no es mover la pieza."""
    captura = CapturaAutomatica()
    imagen = _diente()

    salidas = [captura.mirar(_temblor(imagen), i * 1.0, BIEN) for i in range(5)]

    assert salidas[4] is BIEN


def test_sin_veredicto_no_se_anota_nada() -> None:
    captura = CapturaAutomatica()

    assert all(s is None for s in _pasar(captura, _diente(), None, 8))


# ---------------------------------------------------------------------------
# LA PRUEBA QUE MAS IMPORTA: una sierra quieta no se anota sola
# ---------------------------------------------------------------------------


def test_una_sierra_quieta_no_anota_el_mismo_diente_dos_veces() -> None:
    """Sin esto, una sierra olvidada sobre la mesa llenaria el informe sola."""
    captura = CapturaAutomatica()
    imagen = _diente()

    anotados = [s for s in _pasar(captura, imagen, BIEN, 60) if s is not None]

    assert len(anotados) == 1


def test_recien_anotado_hay_que_mover_la_pieza() -> None:
    captura = CapturaAutomatica()
    assert _pasar(captura, _diente(0), BIEN, 5)[4] is BIEN
    assert captura.esperando_cambio

    # Mas fotos de lo mismo: nada.
    assert all(s is None for s in _pasar(captura, _diente(0), BIEN, 10, desde=10.0))
    # Se mueve la pieza y arranca de cero.
    assert captura.mirar(_diente(6), 30.0, BIEN) is None
    assert not captura.esperando_cambio
    assert captura.iguales == 1


def test_el_diente_siguiente_se_anota_cuando_corresponde() -> None:
    captura = CapturaAutomatica()
    _pasar(captura, _diente(0), BIEN, 5)

    salidas = _pasar(captura, _diente(6), MAL, 6, desde=10.0)

    # La primera foto de la pieza nueva ya cuenta como la primera de las cinco: el cambio
    # se detecta y se arranca a contar en la misma pasada.
    assert salidas[4] is MAL
    assert sum(1 for s in salidas if s is not None) == 1


# ---------------------------------------------------------------------------
# La cadencia
# ---------------------------------------------------------------------------


def test_mira_una_foto_por_segundo_no_todas() -> None:
    """El bucle corre a 17 cuadros por segundo; mirarlos todos seria confirmar un diente
    cada tres decimas y llenar el informe con la primera pieza que se apoye."""
    captura = CapturaAutomatica(segundos_entre_fotos=1.0)
    imagen = _diente()

    # treinta cuadros en un segundo: solo cuenta el primero
    for i in range(30):
        captura.mirar(imagen, i * 0.033, BIEN)

    assert captura.iguales == 1


def test_la_cadencia_se_puede_cambiar() -> None:
    captura = CapturaAutomatica(segundos_entre_fotos=0.25, iguales_para_anotar=2)
    imagen = _diente()

    assert captura.mirar(imagen, 0.0, BIEN) is None
    assert captura.mirar(imagen, 0.1, BIEN) is None      # todavia no paso el cuarto
    assert captura.mirar(imagen, 0.3, BIEN) is BIEN


def test_reiniciar_olvida_la_racha() -> None:
    captura = CapturaAutomatica()
    _pasar(captura, _diente(), BIEN, 3)

    captura.reiniciar()

    assert captura.iguales == 0 and not captura.esperando_cambio


def test_el_progreso_se_puede_mostrar() -> None:
    captura = CapturaAutomatica()
    assert "mirando" in captura.progreso()

    _pasar(captura, _diente(), BIEN, 2)
    assert "2 de 5" in captura.progreso()

    _pasar(captura, _diente(), BIEN, 3, desde=2.0)
    assert "siguiente" in captura.progreso()


# ---------------------------------------------------------------------------
# Las piezas sueltas
# ---------------------------------------------------------------------------


def test_la_huella_aguanta_un_cambio_de_luz() -> None:
    """La lampara del microscopio no es estable; eso no es otra pieza."""
    imagen = _diente()
    mas_clara = np.clip(imagen.astype(np.float32) * 1.1 + 6, 0, 255).astype(np.uint8)

    assert distancia(huella(imagen), huella(mas_clara)) < 0.015


def test_dos_piezas_distintas_estan_lejos() -> None:
    assert distancia(huella(_diente(0)), huella(_diente(6))) > 0.015


def test_sin_imagen_no_hay_huella() -> None:
    assert huella(None) is None
    assert distancia(None, huella(_diente())) == 1.0


def test_dos_veredictos_se_comparan_por_lo_que_dicen() -> None:
    uno = Resultado(estado="afilado", parecido=0.90, aprobado=True)
    otro = Resultado(estado="afilado", parecido=0.93, aprobado=True)
    distinto = Resultado(estado="roto", parecido=0.90, aprobado=False)

    # El parecido nunca da igual dos veces: lo que tiene que coincidir es el veredicto.
    assert uno.coincide_con(otro)
    assert not uno.coincide_con(distinto)
    assert not uno.coincide_con(None)
