"""Pruebas de la galeria de fotos de modelo.

Lo que se protege aca es que agregar un modelo nunca cueste perder otro, y que la galeria
siga andando con una carpeta vacia o con archivos basura adentro, que es como va a estar el
dia 1 y despues de que alguien copie cosas a mano.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from afilado.referencias import GaleriaReferencias


def _foto(valor: int = 120) -> np.ndarray:
    return np.full((60, 80, 3), valor, np.uint8)


def _poblar(carpeta: Path, nombres: tuple[str, ...]) -> None:
    carpeta.mkdir(parents=True, exist_ok=True)
    for indice, nombre in enumerate(nombres):
        cv2.imwrite(str(carpeta / nombre), _foto(40 + indice * 30))


def test_galeria_vacia_no_explota(tmp_path: Path) -> None:
    """El dia 1 no hay ninguna foto y el programa tiene que arrancar igual."""
    galeria = GaleriaReferencias(str(tmp_path / "modelos"))

    assert galeria.cantidad == 0
    assert galeria.nombre_actual is None
    assert galeria.imagen_actual() is None
    assert galeria.posicion == (0, 0)
    assert galeria.siguiente() is None


def test_lee_la_carpeta_en_orden_alfabetico(tmp_path: Path) -> None:
    """El orden es el del nombre: numerar los archivos alcanza para ordenarlos."""
    _poblar(tmp_path, ("02-R01.jpg", "01-LU3F.jpg", "03-otra.png"))

    galeria = GaleriaReferencias(str(tmp_path))

    assert galeria.cantidad == 3
    assert galeria.nombre_actual == "01-LU3F"
    assert galeria.posicion == (1, 3)


def test_ignora_lo_que_no_sea_imagen(tmp_path: Path) -> None:
    _poblar(tmp_path, ("modelo.jpg",))
    (tmp_path / "notas.txt").write_text("no soy una foto", encoding="utf-8")
    (tmp_path / "datos.yaml").write_text("tampoco", encoding="utf-8")

    galeria = GaleriaReferencias(str(tmp_path))

    assert galeria.cantidad == 1


def test_ciclar_da_la_vuelta(tmp_path: Path) -> None:
    _poblar(tmp_path, ("a.jpg", "b.jpg", "c.jpg"))
    galeria = GaleriaReferencias(str(tmp_path))

    assert galeria.siguiente() == "b"
    assert galeria.siguiente() == "c"
    assert galeria.siguiente() == "a", "tiene que volver al principio"
    assert galeria.anterior() == "c", "y para atras tambien"


def test_con_una_sola_foto_ciclar_no_hace_nada(tmp_path: Path) -> None:
    _poblar(tmp_path, ("unica.jpg",))
    galeria = GaleriaReferencias(str(tmp_path))

    assert galeria.siguiente() == "unica"
    assert galeria.posicion == (1, 1)


def test_agregar_no_pisa_las_anteriores(tmp_path: Path) -> None:
    """Agregar un modelo no puede costar perder otro: es la razon de que haya galeria."""
    _poblar(tmp_path, ("viejo.jpg",))
    galeria = GaleriaReferencias(str(tmp_path))

    ruta = galeria.agregar(_foto(200))

    assert ruta.is_file()
    assert (tmp_path / "viejo.jpg").is_file(), "se perdio la referencia anterior"
    assert galeria.cantidad == 2


def test_agregar_deja_la_nueva_seleccionada(tmp_path: Path) -> None:
    """El operario acaba de sacar la foto: espera verla, no seguir en otra."""
    _poblar(tmp_path, ("aaa.jpg",))
    galeria = GaleriaReferencias(str(tmp_path))

    ruta = galeria.agregar(_foto(200), nombre="LU3F-0300")

    assert galeria.nombre_actual == ruta.stem == "LU3F-0300"
    assert int(galeria.imagen_actual().mean()) == 200


def test_agregar_dos_veces_con_el_mismo_nombre_no_pisa(tmp_path: Path) -> None:
    galeria = GaleriaReferencias(str(tmp_path))

    primera = galeria.agregar(_foto(100), nombre="R01")
    segunda = galeria.agregar(_foto(200), nombre="R01")

    assert primera != segunda
    assert primera.is_file() and segunda.is_file()
    assert galeria.cantidad == 2


@pytest.mark.parametrize("nombre", ["con/barra", "dos:puntos", "  espacios  ", "*raro?"])
def test_el_nombre_se_sanea(tmp_path: Path, nombre: str) -> None:
    """Windows rechaza varios caracteres; un nombre invalido no puede tumbar el guardado."""
    galeria = GaleriaReferencias(str(tmp_path))

    ruta = galeria.agregar(_foto(), nombre=nombre)

    assert ruta.is_file()


def test_recuerda_cual_estaba_elegida(tmp_path: Path) -> None:
    """Quien trabaja todo el dia con el mismo modelo no deberia ciclar en cada arranque."""
    _poblar(tmp_path, ("a.jpg", "b.jpg", "c.jpg"))
    primera = GaleriaReferencias(str(tmp_path))
    primera.siguiente()
    primera.siguiente()
    assert primera.nombre_actual == "c"

    segunda = GaleriaReferencias(str(tmp_path))

    assert segunda.nombre_actual == "c"


def test_un_archivo_en_el_config_sigue_andando(tmp_path: Path) -> None:
    """Los configs anteriores apuntaban a UN archivo. No pueden romperse."""
    _poblar(tmp_path, ("aprobada.jpg", "zzz.jpg"))

    galeria = GaleriaReferencias(str(tmp_path / "aprobada.jpg"))

    assert galeria.carpeta == tmp_path
    assert galeria.nombre_actual == "aprobada", "el archivo del config manda al arrancar"
    assert galeria.cantidad == 2, "pero se ven todas las de la carpeta"


def test_recargar_ve_las_fotos_copiadas_a_mano(tmp_path: Path) -> None:
    """Alguien copia fotos a la carpeta con el programa abierto; 'm' las tiene que ver."""
    _poblar(tmp_path, ("a.jpg",))
    galeria = GaleriaReferencias(str(tmp_path))
    assert galeria.cantidad == 1

    _poblar(tmp_path, ("b.jpg", "c.jpg"))
    galeria.recargar()

    assert galeria.cantidad == 3
    assert galeria.nombre_actual == "a", "la que estaba elegida sigue elegida"


def test_una_imagen_ilegible_no_tumba_la_galeria(tmp_path: Path) -> None:
    """Un .jpg truncado por una copia a medias es un escenario real en un pendrive."""
    _poblar(tmp_path, ("buena.jpg",))
    (tmp_path / "rota.jpg").write_bytes(b"esto no es un jpg")
    galeria = GaleriaReferencias(str(tmp_path))
    galeria.recargar(preferida="rota.jpg")

    assert galeria.imagen_actual() is None, "no deberia inventar una imagen"
    assert galeria.siguiente() == "buena", "y se puede pasar a la que si sirve"
