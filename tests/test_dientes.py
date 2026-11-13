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


# ---------------------------------------------------------------------------
# CUAL DE LOS RECUADROS SE MIRA PRIMERO
# ---------------------------------------------------------------------------


def _cara_y_hueco() -> np.ndarray:
    """Dos manchas claras del mismo tamano: a la izquierda el HUECO entre dientes, que
    esta lustrado y sale liso, y a la derecha la cara de widia, que sale del esmeril con
    lineas marcadas. Es lo que se ve en afilado_20261112_161626_660.jpg, donde el hueco
    medido da 30.4 de rayado y la cara 43.6."""
    img = np.full((480, 640, 3), 70, np.uint8)
    cv2.rectangle(img, (60, 150), (260, 380), (185, 185, 185), -1)   # el hueco, liso
    cv2.rectangle(img, (360, 150), (560, 380), (180, 180, 180), -1)  # la cara...
    for x in range(360, 560, 6):                                     # ...rayada
        cv2.line(img, (x, 150), (x, 380), (235, 235, 235), 2)
    return cv2.GaussianBlur(img, (5, 5), 0)


def test_cuando_hay_hueco_y_diente_se_elige_el_diente() -> None:
    """El hueco pasa el filtro de area igual que la cara, y cuando cae a la izquierda
    quedaba PRIMERO: era el que se archivaba y el que iba al informe, con el diente de
    verdad al lado sin elegir. Pasa en 1 de las 14 fotos de cerca del taller."""
    encontrados = D.detectar(_cara_y_hueco())

    assert len(encontrados) == 2
    izquierda, derecha = encontrados
    assert izquierda.caja[0] < derecha.caja[0]          # el orden en pantalla no cambia
    assert derecha.rayado > izquierda.rayado
    assert D.mas_parecido_a_diente(encontrados) == 1


def test_con_un_solo_recuadro_no_se_gasta_en_medir_el_rayado() -> None:
    """Son unos milisegundos de Sobel que en la PC del taller se notan, y con un solo
    recuadro no hay nada que elegir."""
    encontrados = D.detectar(_diente())

    assert len(encontrados) == 1
    assert encontrados[0].rayado == 0.0
    assert D.mas_parecido_a_diente(encontrados) == 0


def _inclinar(imagen: np.ndarray, grados: float) -> np.ndarray:
    """La sierra la sostiene la mano del operario: no entra derecha."""
    alto, ancho = imagen.shape[:2]
    matriz = cv2.getRotationMatrix2D((ancho / 2.0, alto / 2.0), grados, 1.0)
    return cv2.warpAffine(imagen, matriz, (ancho, alto), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REPLICATE)


def test_con_la_sierra_inclinada_el_recuadro_sigue_en_la_cara_rayada() -> None:
    """Con la sierra torcida, el que gana tiene que seguir siendo la cara rayada. Medido
    sobre las 17 fotos reales por 9 inclinaciones de -20 a +20 grados, el recuadro
    elegido cae en el diente en 110 de 126 casos y en el hueco en ninguno.

    LO QUE ESTA PRUEBA NO CUIDA: aca las dos manchas se separan por rayado 4.5 contra
    84.8, y en la foto real el margen es 30.4 contra 43.6. Medido, esta prueba pasa igual
    midiendo el rayado con la magnitud del gradiente, con el maximo de |Sx| y |Sy| y
    hasta con la derivada en y SOLA --que en la foto real le da el recuadro al hueco--.
    Avisa si el recuadro se va al hueco con la sierra torcida; el margen no lo cuida
    nadie: eso esta medido a mano en el docstring de _por_cara_rayada.
    """
    for grados in (-15, -10, -5, 5, 10, 15):
        encontrados = D.detectar(_inclinar(_cara_y_hueco(), grados))

        assert len(encontrados) == 2, f"a {grados} grados no salieron las dos manchas"
        elegido = encontrados[D.mas_parecido_a_diente(encontrados)]
        assert elegido.centro[0] > 320, f"a {grados} grados el recuadro se fue al hueco"


def _sierra_que_llena_el_cuadro() -> np.ndarray:
    """La sierra tan de cerca que el cuerpo del disco ocupa casi todo: el corte por
    brillo junta todo en un recuadro del tamano del cuadro, y el de la cara rayada
    tampoco separa nada --devuelve uno del 62%--."""
    azar = np.random.default_rng(7)
    img = np.full((480, 640, 3), 30, np.uint8)
    img[20:450, 20:620] = azar.normal(140, 14, (430, 600, 3)).clip(0, 255).astype(np.uint8)
    img[45:425, 60:580] = azar.normal(195, 30, (380, 520, 3)).clip(0, 255).astype(np.uint8)
    return cv2.GaussianBlur(img, (5, 5), 0)


def test_si_el_segundo_camino_tampoco_separa_no_se_queda_con_su_recuadro() -> None:
    """Un recuadro de 60% o mas no lo dibuja revision.py, asi que el operario no ve nada
    mientras el programa archiva y juzga ese recorte. Pasa en 9 de 324 casos medidos (36
    fotos reales por 9 inclinaciones): el 2do camino devuelve el 61%-65% del cuadro y se
    lo queda. Si ninguno de los dos separo, que vuelva el del corte por brillo."""
    imagen = _sierra_que_llena_el_cuadro()
    alto, ancho = imagen.shape[:2]

    encontrados = D.detectar(imagen)

    assert len(encontrados) == 1
    x1, y1, x2, y2 = encontrados[0].caja
    assert (x2 - x1) * (y2 - y1) >= 0.80 * alto * ancho


def test_un_recuadro_que_no_toca_la_imagen_queda_en_el_log(caplog) -> None:
    """recortar() devuelve la foto entera cuando la caja no la toca --una caja medida en
    un cuadro de otro tamano--. Callado, eso se archiva como si fuera un diente. Con las
    cajas de _componentes no pasa: sobre las 36 fotos reales por 9 inclinaciones no se
    disparo nunca."""
    imagen = np.zeros((100, 100, 3), np.uint8)

    recorte = D.Diente(caja=(200, 200, 300, 300), area=0.0).recortar(imagen)

    assert recorte.shape == imagen.shape
    assert "fuera de la imagen" in caplog.text


def test_sin_ningun_recuadro_tambien_se_prueba_el_segundo_camino() -> None:
    """_no_separo dice True en DOS casos opuestos: el umbral junto todo en un recuadro
    del tamano de la foto, o no dejo nada. Los dos tienen que mandar al 2do camino."""
    entera = [D.Diente(caja=(0, 0, 640, 480), area=1.0, entero=True)]

    assert D._no_separo(entera, 640 * 480) is True
    assert D._no_separo([], 640 * 480) is True
