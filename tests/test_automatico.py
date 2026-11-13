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


# ---------------------------------------------------------------------------
# LO QUE LA PRUEBA DE "SIERRA QUIETA" NO MIRABA. La de arriba usa una imagen
# perfectamente quieta, y en ese mundo no pasa nada: el freno nunca se ponia a
# prueba. En el taller pasan cosas delante del microscopio todo el tiempo --una
# mano, el tubo fluorescente, la vibracion del motor de al lado-- y cada una de
# esas alcanzaba para soltar el freno y dejar que la MISMA pieza quieta se
# anotara de nuevo. Medido en el bucle de verdad: una mano cruzando cada dos
# segundos llenaba los doce dientes de la sierra, los doce con el mismo numero.
# ---------------------------------------------------------------------------


def _mano_cruzando(imagen: np.ndarray) -> np.ndarray:
    """Alguien pasa la mano por delante del microscopio.

    MEDIDO: este es el caso que fallaba. Con el freno de antes --que se soltaba con la
    primera foto distinta-- una mano cruzando cada dos segundos dejaba anotada la sierra
    entera: doce dientes en el bucle de verdad, seis en la prueba de punta a punta.
    """
    tapada = imagen.copy()
    tapada[:, 40:160] = 20
    return tapada


def _tubo_parpadeando(imagen: np.ndarray) -> np.ndarray:
    """El fluorescente del taller, que no es continuo.

    MEDIDO: este caso ya pasaba antes del arreglo, y hay que decirlo. La huella se toma
    sobre la imagen normalizada por media y desvio, asi que un cambio de exposicion no la
    mueve: el parpadeo nunca soltaba el freno. Queda igual como guardia, para que
    cualquiera que toque la huella se entere si pierde esa propiedad.
    """
    return np.clip(imagen.astype(np.float32) * 0.55, 0, 255).astype(np.uint8)


@pytest.mark.parametrize("estorbo", [_mano_cruzando, _tubo_parpadeando])
def test_lo_que_cruza_por_delante_no_habilita_anotar_de_nuevo(estorbo) -> None:
    captura = CapturaAutomatica()
    quieta = _diente()
    anotados = []
    reloj = 0.0

    for vuelta in range(12):
        # cinco fotos de la pieza quieta...
        for _ in range(5):
            r = captura.mirar(quieta, reloj, BIEN)
            reloj += 1.0
            if r is not None:
                anotados.append(r)
        # ...y algo que cruza por delante
        anotados += [x for x in [captura.mirar(estorbo(quieta), reloj, BIEN)] if x]
        reloj += 1.0

    assert len(anotados) == 1, f"se anoto {len(anotados)} veces la misma pieza quieta"


def test_el_freno_sobrevive_a_reiniciar() -> None:
    """reiniciar() se llama al congelar, al abrir el panel y al anotar a mano. Si soltara
    el freno, cada una de esas tres cosas habilitaria anotar otra vez el mismo diente."""
    captura = CapturaAutomatica()
    quieta = _diente()
    assert _pasar(captura, quieta, BIEN, 5)[4] is BIEN

    captura.reiniciar()

    assert captura.esperando_cambio, "reiniciar solto el freno"
    assert all(s is None for s in _pasar(captura, quieta, BIEN, 10, desde=10.0))


def test_anotar_por_otro_camino_tambien_frena() -> None:
    """El operario aprieta 'a' o 'g': ese diente queda anotado, y la captura automatica
    tiene que enterarse o lo anota de nuevo un segundo despues."""
    captura = CapturaAutomatica()
    quieta = _diente()
    _pasar(captura, quieta, BIEN, 2)          # racha a medio hacer

    captura.ya_anotado(quieta)

    assert captura.esperando_cambio
    assert all(s is None for s in _pasar(captura, quieta, BIEN, 10, desde=5.0))
    # Y el diente siguiente se anota normal.
    assert _pasar(captura, _diente(6), BIEN, 5, desde=20.0)[4] is BIEN


def test_despues_del_estorbo_el_diente_siguiente_se_anota_igual() -> None:
    """El freno no puede volverse un tapon: con una pieza NUEVA tiene que contar."""
    captura = CapturaAutomatica()
    assert _pasar(captura, _diente(0), BIEN, 5)[4] is BIEN
    captura.mirar(_mano_cruzando(_diente(0)), 10.0, BIEN)

    salidas = _pasar(captura, _diente(6), BIEN, 5, desde=20.0)

    assert salidas[4] is BIEN


# ---------------------------------------------------------------------------
# CON LA SIERRA EN LA MANO. El corte de antes --0.015 sobre la firma cruda-- aguantaba
# medio pixel de corrimiento entre foto y foto. Medido sobre las 14 fotos reales del
# taller: la misma toma corrida 1 px daba 0.0148-0.0256, o sea por encima del corte, y en
# el bucle, apoyando seis dientes con un temblor de mano de 0.5 px se anotaba UNO solo y
# con 1 px ninguno. La captura automatica andaba unicamente con la pieza apoyada y quieta,
# que es justo lo que el taller no hace.
# ---------------------------------------------------------------------------


def _corrida(imagen: np.ndarray, dx: float, dy: float = 0.0) -> np.ndarray:
    matriz = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(imagen, matriz, (imagen.shape[1], imagen.shape[0]),
                          borderMode=cv2.BORDER_REPLICATE)


@pytest.mark.parametrize("temblor", [0.0, 0.5, 1.0])
def test_con_la_mano_temblando_el_diente_se_anota_igual(temblor: float) -> None:
    """OJO CON EL NUMERO. Sobre las fotos REALES del taller a 640x480 esto aguanta hasta
    2 px de temblor (medido: seis dientes de seis). Aca se llega a 1 px y no mas, porque
    la imagen de prueba tiene lineas cada 7 px --mucho mas fina que cualquier cosa que
    entregue el microscopio-- y un corrimiento de 2 px le cambia medio dibujo. El 2 px del
    comentario de DISTANCIA_MISMA_FOTO sale de las fotos de verdad, no de esta."""
    captura = CapturaAutomatica()
    quieta = _diente()
    rng = np.random.default_rng(7)

    salidas = [
        captura.mirar(
            _corrida(quieta, rng.normal(0, temblor), rng.normal(0, temblor)), i * 1.0, BIEN
        )
        for i in range(8)
    ]

    assert any(s is BIEN for s in salidas), f"con {temblor} px de temblor no anoto"


def test_pasar_la_sierra_de_largo_no_anota() -> None:
    """El otro lado de lo mismo: si la pieza no se detiene, no hay diente que revisar."""
    captura = CapturaAutomatica()
    quieta = _diente()

    salidas = [captura.mirar(_corrida(quieta, i * 12.0), i * 1.0, BIEN) for i in range(20)]

    assert all(s is None for s in salidas)


def test_la_racha_se_mide_contra_su_primera_foto() -> None:
    """Encadenando --cada foto contra la anterior-- una pieza que se desliza despacio
    cierra la racha igual, porque entre foto y foto se movio poco. Medido en el bucle con
    la foto real del taller: 20 s de arrastre parejo a 3 px/s anotaban 17 dientes sin que
    la pieza se detuviera una sola vez."""
    captura = CapturaAutomatica()
    quieta = _diente()
    paso = 2.0        # por debajo del corte entre foto y foto, muy por encima en total

    salidas = [captura.mirar(_corrida(quieta, i * paso), i * 1.0, BIEN) for i in range(6)]

    assert all(s is None for s in salidas)


def test_otro_diente_suelta_el_freno_y_el_mismo_corrido_no() -> None:
    """Los dos cortes son distintos a proposito: 'es la misma foto' (0.04) y 'es otro
    diente' (0.07). Sobre los 91 pares de fotos distintas del taller, el par mas parecido
    da 0.0855 y ninguno baja de 0.08."""
    captura = CapturaAutomatica()
    quieta = _diente()
    assert _pasar(captura, quieta, BIEN, 5)[4] is BIEN

    # El mismo diente, corrido un poco: sigue frenado.
    assert captura.mirar(_corrida(quieta, 3.0), 10.0, BIEN) is None
    assert captura.esperando_cambio
    # Otro diente: cuenta.
    assert _pasar(captura, _diente(6), BIEN, 5, desde=20.0)[4] is BIEN


def test_el_freno_no_se_traba_con_la_sierra_entera() -> None:
    """Lo contrario del defecto que el freno vino a tapar, y tanto o mas grave: si el
    freno no se soltara al pasar al diente siguiente, el programa se clavaria en el
    primero y no anotaria ninguno mas.

    Sobre las 14 fotos reales del taller, el paso de un diente al siguiente mas chico que
    hay da 0.1048 contra un corte de 0.07. Aca se simula la sierra entera: seis dientes,
    uno tras otro, cada uno apoyado un rato."""
    captura = CapturaAutomatica()
    anotados, reloj = [], 0.0

    for numero in range(6):
        for _ in range(7):
            salida = captura.mirar(_diente(numero * 5), reloj, BIEN)
            reloj += 1.0
            if salida is not None:
                anotados.append(numero)

    assert anotados == [0, 1, 2, 3, 4, 5], f"se anotaron los dientes {anotados}"
