"""Pruebas de la separacion de dientes.

LO QUE MAS SE CUIDA: que no INVENTE un diente. Perder el segundo diente de una foto se ve
--el operario mira la pantalla y lo vuelve a encuadrar-- pero inventar uno que no existe
termina como un resultado de mas en el informe de una orden, y eso no se nota hasta que
llega el reclamo.

Por eso tambien existe la regla de que nunca devuelva una lista vacia: con una imagen que
no entiende, el diente es la foto entera y el programa sigue andando como venia.
"""

from __future__ import annotations

import cv2
import numpy as np

from afilado import dientes as D


def _diente(semilla: int = 0, lado: int = 240) -> np.ndarray:
    """Un diente: la cara iluminada clara sobre un fondo oscuro, como en el microscopio."""
    img = np.full((lado, lado, 3), 40, np.uint8)
    cv2.fillPoly(img, [np.array([[30, lado - 40], [30, 90 + semilla * 4],
                                 [lado - 70, 40], [lado - 40, 70],
                                 [lado - 40, lado - 40]])], (195, 195, 195))
    cv2.line(img, (30, 90 + semilla * 4), (lado - 70, 40), (240, 240, 240), 5)
    for i in range(0, lado, 9):
        cv2.line(img, (i, 0), (i + 30 + semilla * 6, lado), (150, 150, 150), 1)
    return cv2.GaussianBlur(img, (5, 5), 0)


def _pegar(*imagenes: np.ndarray, hueco: int = 40) -> np.ndarray:
    """Varios dientes uno al lado del otro, con la sombra del hueco en el medio."""
    alto = min(i.shape[0] for i in imagenes)
    partes: list[np.ndarray] = []
    for indice, imagen in enumerate(imagenes):
        if indice:
            partes.append(np.full((alto, hueco, 3), 15, np.uint8))
        escala = alto / imagen.shape[0]
        partes.append(cv2.resize(imagen, (int(imagen.shape[1] * escala), alto)))
    return np.hstack(partes)


def _centros(encontrados: list[D.Diente]) -> list[int]:
    return [d.centro[0] for d in encontrados]


# ---------------------------------------------------------------------------


def test_un_diente_da_un_diente() -> None:
    encontrados = D.detectar(_diente())

    assert len(encontrados) == 1
    assert not encontrados[0].entero          # lo encontro de verdad, no es el fallback
    assert 0.04 < encontrados[0].area < 0.96


def test_dos_dientes_dan_dos_cajas_cada_una_de_su_lado() -> None:
    imagen = _pegar(_diente(0), _diente(3))

    encontrados = D.detectar(imagen)

    assert len(encontrados) == 2
    mitad = imagen.shape[1] / 2
    izquierdo, derecho = _centros(encontrados)
    assert izquierdo < mitad < derecho


def test_vienen_ordenados_de_izquierda_a_derecha() -> None:
    """La pantalla los numera 1 y 2: si el orden cambiara, el 1 seria un dia uno y al dia
    siguiente el otro."""
    encontrados = D.detectar(_pegar(_diente(0), _diente(2), _diente(5)))

    assert _centros(encontrados) == sorted(_centros(encontrados))


def test_tres_dientes_se_pueden_separar() -> None:
    encontrados = D.detectar(_pegar(_diente(0), _diente(2), _diente(5)))

    assert len(encontrados) == 3


def test_no_devuelve_mas_de_los_que_se_le_piden() -> None:
    encontrados = D.detectar(_pegar(_diente(0), _diente(2), _diente(5)), maximo=2)

    assert len(encontrados) == 2


def test_una_imagen_lisa_no_inventa_dientes() -> None:
    """La tapa puesta, una pared, el microscopio apuntando al aire."""
    for imagen in (np.full((240, 320, 3), 128, np.uint8),
                   np.zeros((240, 320, 3), np.uint8),
                   np.full((240, 320, 3), 255, np.uint8)):
        encontrados = D.detectar(imagen)

        assert len(encontrados) == 1
        assert encontrados[0].entero        # el fallback: el diente es la foto entera


def test_el_ruido_tampoco_inventa_dientes() -> None:
    ruido = np.random.default_rng(3).integers(0, 255, (240, 320, 3), dtype=np.uint8)

    encontrados = D.detectar(ruido)

    assert len(encontrados) == 1 and encontrados[0].entero


def test_una_veta_fina_no_cuenta_como_diente() -> None:
    imagen = np.full((240, 320, 3), 30, np.uint8)
    cv2.line(imagen, (0, 120), (320, 120), (230, 230, 230), 4)

    encontrados = D.detectar(imagen)

    assert encontrados[0].entero


def test_sin_imagen_no_hay_dientes() -> None:
    assert D.detectar(None) == []
    assert D.detectar(np.zeros((0, 0, 3), np.uint8)) == []


def test_el_recorte_cae_sobre_el_diente() -> None:
    imagen = _pegar(_diente(0), _diente(3))
    encontrados = D.detectar(imagen)

    recortes = [d.recortar(imagen) for d in encontrados]

    assert len(recortes) == 2
    for recorte in recortes:
        assert recorte.size > 0
        # Un recorte de diente tiene parte clara: si cayera en el hueco seria todo oscuro.
        assert recorte.mean() > 60
    # Y son pedazos distintos de la imagen, no el mismo dos veces.
    assert recortes[0].shape != recortes[1].shape or not np.array_equal(*recortes)


def test_el_recorte_deja_aire_alrededor() -> None:
    """La punta del diente cae en la zona oscura que el umbral deja afuera; sin aire, el
    recorte corta justo el filo, que es lo unico que importa mirar."""
    imagen = _diente()
    encontrado = D.detectar(imagen)[0]

    justo = encontrado.recortar(imagen, margen=0.0)
    con_aire = encontrado.recortar(imagen, margen=0.2)

    assert con_aire.shape[0] > justo.shape[0]
    assert con_aire.shape[1] > justo.shape[1]


def test_el_recorte_no_se_sale_de_la_foto() -> None:
    imagen = _diente()
    entero = D.Diente(caja=(0, 0, imagen.shape[1], imagen.shape[0]), area=1.0)

    recorte = entero.recortar(imagen, margen=0.5)

    assert recorte.shape == imagen.shape


# ---------------------------------------------------------------------------
# EL FRENO DE MANO. Con la sierra mas lejos --varios dientes con sus huecos-- lo
# mas claro de la imagen puede ser el hueco iluminado y no el diente: el taller
# lo vio en pantalla, con el recuadro puesto sobre el hueco. Cuando pasa eso es
# mejor analizar la foto entera, como se venia haciendo, que recortar el pedazo
# equivocado: un recorte del hueco archivado como "afilado" envenena la memoria
# sin que nadie se entere.
# ---------------------------------------------------------------------------


def _sierra_de_lejos(ancho: int = 480, alto: int = 360) -> np.ndarray:
    """Un diente OSCURO sobre el disco claro, con el hueco al lado bien iluminado.

    Es la estructura de la foto que mando el taller: el cuerpo del diente es lo mas
    oscuro, la placa de arriba lo mas claro, y el hueco de al lado tambien brilla.
    """
    # Los brillos salen de la foto del taller: el disco ronda 135, el fondo 95, el cuerpo
    # del diente 45, el hueco lustrado llega a 185 y la placa de widia a 250.
    img = np.full((alto, ancho, 3), 95, np.uint8)          # fondo
    cv2.rectangle(img, (0, 220), (ancho, alto), (135, 135, 135), -1)   # el disco
    cv2.ellipse(img, (120, 250), (90, 60), 0, 0, 180, (165, 165, 165), -1)  # hueco lustrado
    cv2.rectangle(img, (230, 110), (350, 250), (45, 45, 45), -1)       # cuerpo del diente
    # La placa sobresale hacia la derecha, como en la foto del taller, y es lo mas claro.
    cv2.rectangle(img, (228, 90), (408, 124), (250, 250, 250), -1)     # placa de widia
    return cv2.GaussianBlur(img, (5, 5), 0)


def test_que_sobren_dientes_para_el_tope_no_dispara_el_freno() -> None:
    """Pedir 2 cuando hay 3 no quiere decir que los recuadros esten mal puestos."""
    encontrados = D.detectar(_pegar(_diente(0), _diente(2), _diente(5)), maximo=2)

    assert len(encontrados) == 2
    assert not any(d.entero for d in encontrados)


def test_en_la_vista_de_lejos_el_recuadro_no_cae_en_el_hueco() -> None:
    """Lo que vio el taller: con la sierra de lejos, el hueco entre dientes esta lustrado
    y el corte por brillo lo elegia a el. Ahora decide la cara rayada."""
    imagen = _sierra_de_lejos()
    encontrados = D.detectar(imagen)

    assert len(encontrados) >= 1
    x1, y1, x2, y2 = encontrados[0].caja
    if encontrados[0].entero:
        return                                   # no separo nada: no inventa un recuadro
    centro_x = (x1 + x2) / 2
    assert centro_x > 200, "el recuadro se fue al hueco de la izquierda"
