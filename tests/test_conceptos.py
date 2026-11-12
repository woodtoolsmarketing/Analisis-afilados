"""Pruebas de la lista de conceptos: las carpetas donde el operario archiva cada foto.

POR QUE IMPORTA QUE ESTO ESTE BIEN
-----------------------------------
Hasta ahora la lista vivia en configs/config.yaml y cambiarla pedia editar un YAML en la PC
del taller. El registro mostro la consecuencia: el operario sacaba fotos con la unica tecla
que no lo obligaba a elegir entre dos palabras fijas, y la coleccion quedaba vacia.

Ahora la lista se maneja desde el programa. Eso significa que borrar es un clic, y por eso
borrar no borra: mueve a _papelera. Esa es la invariante que mas se cuida aca.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from afilado.conceptos import Conceptos, sanear


def _foto(carpeta: Path, nombre: str) -> Path:
    carpeta.mkdir(parents=True, exist_ok=True)
    ruta = carpeta / nombre
    ruta.write_bytes(b"no es un jpg de verdad, pero ocupa un lugar")
    return ruta


# ---------------------------------------------------------------------------
# De donde sale la lista
# ---------------------------------------------------------------------------


def test_la_primera_vez_siembra_los_conceptos_del_config(tmp_path: Path) -> None:
    conceptos = Conceptos(str(tmp_path / "coleccion"), ["aprobada", "rechazada"], ["rechazada"])

    assert conceptos.nombres() == ["aprobada", "rechazada"]
    assert conceptos.defectos() == ["rechazada"]
    # Y les crea la carpeta: la verdad vive en el disco, no en una lista aparte.
    assert (tmp_path / "coleccion" / "aprobada").is_dir()


def test_una_carpeta_copiada_a_mano_aparece_como_concepto(tmp_path: Path) -> None:
    """Alguien copia una carpeta con el Explorador: al abrir el programa tiene que estar."""
    base = tmp_path / "coleccion"
    _foto(base / "aprobada", "a.jpg")
    _foto(base / "filo_quemado", "b.jpg")

    conceptos = Conceptos(str(base))

    assert set(conceptos.nombres()) == {"aprobada", "filo_quemado"}


def test_las_carpetas_con_guion_bajo_no_son_conceptos(tmp_path: Path) -> None:
    """_papelera vive adentro de la coleccion y NO puede aparecer como una clase mas."""
    base = tmp_path / "coleccion"
    _foto(base / "aprobada", "a.jpg")
    _foto(base / "_papelera" / "rechazada", "vieja.jpg")

    conceptos = Conceptos(str(base))

    assert conceptos.nombres() == ["aprobada"]


def test_el_orden_no_cambia_solo(tmp_path: Path) -> None:
    """El panel se maneja con las teclas 1..9: si la lista se reacomoda, el 3 cambia de
    significado entre un dia y el otro."""
    base = tmp_path / "coleccion"
    conceptos = Conceptos(str(base), ["aprobada", "rechazada"], ["rechazada"])
    conceptos.agregar("astillado")
    orden = conceptos.nombres()

    conceptos.marcar_defecto("astillado", True)
    conceptos.recargar()

    assert conceptos.nombres() == orden
    assert Conceptos(str(base)).nombres() == orden


def test_cuenta_solo_imagenes(tmp_path: Path) -> None:
    base = tmp_path / "coleccion"
    _foto(base / "aprobada", "a.jpg")
    _foto(base / "aprobada", "b.PNG")
    _foto(base / "aprobada", "notas.txt")

    conceptos = Conceptos(str(base))

    assert conceptos.listar()[0].fotos == 2


# ---------------------------------------------------------------------------
# Agregar
# ---------------------------------------------------------------------------


def test_agregar_crea_la_carpeta_y_lo_recuerda(tmp_path: Path) -> None:
    base = tmp_path / "coleccion"
    conceptos = Conceptos(str(base), ["aprobada"], [])

    conceptos.agregar("diente astillado", defecto=True)

    assert (base / "diente_astillado").is_dir()
    assert "diente_astillado" in conceptos.defectos()
    # Y sigue estando en el proximo arranque, aunque todavia no tenga ni una foto.
    assert "diente_astillado" in Conceptos(str(base)).nombres()


@pytest.mark.parametrize("nombre", ["", "   ", "///", "___"])
def test_un_nombre_que_no_deja_nada_se_rechaza(tmp_path: Path, nombre: str) -> None:
    conceptos = Conceptos(str(tmp_path / "coleccion"), ["aprobada"], [])

    with pytest.raises(ValueError):
        conceptos.agregar(nombre)


def test_no_se_puede_repetir_un_concepto(tmp_path: Path) -> None:
    conceptos = Conceptos(str(tmp_path / "coleccion"), ["aprobada"], [])

    with pytest.raises(ValueError, match="ya hay"):
        conceptos.agregar("aprobada")


def test_un_concepto_no_puede_terminar_siendo_la_papelera(tmp_path: Path) -> None:
    """Si pudiera, el concepto nuevo se comeria todo lo que alguien borro antes."""
    base = tmp_path / "coleccion"
    conceptos = Conceptos(str(base), ["aprobada"], [])
    _foto(base / "aprobada", "vieja.jpg")
    conceptos.recargar()
    conceptos.eliminar("aprobada")

    nuevo = conceptos.agregar("_papelera")

    assert nuevo.nombre == "papelera"
    assert nuevo.carpeta == base / "papelera"
    # Lo que se habia borrado sigue intacto en su carpeta, sin mezclarse con el concepto.
    assert list((base / "_papelera").rglob("*.jpg"))


def test_el_nombre_queda_usable_como_carpeta(tmp_path: Path) -> None:
    conceptos = Conceptos(str(tmp_path / "coleccion"), ["aprobada"], [])

    nuevo = conceptos.agregar("filo: quemado/roto")

    assert nuevo.nombre == "filo__quemado_roto"
    assert nuevo.carpeta.is_dir()


# ---------------------------------------------------------------------------
# Borrar: NUNCA se pierde una foto
# ---------------------------------------------------------------------------


def test_borrar_un_concepto_manda_sus_fotos_a_la_papelera(tmp_path: Path) -> None:
    base = tmp_path / "coleccion"
    _foto(base / "rechazada", "una.jpg")
    _foto(base / "rechazada", "otra.jpg")
    conceptos = Conceptos(str(base))

    cuantas = conceptos.eliminar("rechazada")

    assert cuantas == 2
    assert "rechazada" not in conceptos.nombres()
    assert not (base / "rechazada").exists()
    sobrevivientes = list((base / "_papelera").rglob("*.jpg"))
    assert len(sobrevivientes) == 2


def test_borrar_una_foto_la_manda_a_la_papelera(tmp_path: Path) -> None:
    base = tmp_path / "coleccion"
    ruta = _foto(base / "aprobada", "sale.jpg")
    _foto(base / "aprobada", "queda.jpg")
    conceptos = Conceptos(str(base))

    destino = conceptos.eliminar_foto(ruta)

    assert not ruta.exists()
    assert destino.is_file()
    assert "_papelera" in destino.parts
    assert conceptos.listar()[0].fotos == 1


def test_borrar_una_foto_que_no_esta_avisa(tmp_path: Path) -> None:
    conceptos = Conceptos(str(tmp_path / "coleccion"), ["aprobada"], [])

    with pytest.raises(ValueError):
        conceptos.eliminar_foto(tmp_path / "no_existe.jpg")


def test_borrar_dos_veces_el_mismo_nombre_no_se_pisa(tmp_path: Path) -> None:
    """Se crea 'aprobada', se borra, se vuelve a crear y se borra: nada se sobreescribe."""
    base = tmp_path / "coleccion"
    conceptos = Conceptos(str(base), ["aprobada"], [])
    _foto(base / "aprobada", "primera.jpg")
    conceptos.recargar()
    conceptos.eliminar("aprobada")

    conceptos.agregar("aprobada")
    _foto(base / "aprobada", "segunda.jpg")
    conceptos.recargar()
    conceptos.eliminar("aprobada")

    guardadas = {p.name for p in (base / "_papelera").rglob("*.jpg")}
    assert guardadas == {"primera.jpg", "segunda.jpg"}


# ---------------------------------------------------------------------------
# El indice no puede dejar al taller sin conceptos
# ---------------------------------------------------------------------------


def test_un_indice_roto_no_borra_los_conceptos(tmp_path: Path) -> None:
    base = tmp_path / "coleccion"
    _foto(base / "aprobada", "a.jpg")
    _foto(base / "rechazada", "b.jpg")
    (base / "conceptos.json").write_text("{ esto no es json", encoding="utf-8")

    conceptos = Conceptos(str(base))

    assert set(conceptos.nombres()) == {"aprobada", "rechazada"}


def test_una_carpeta_borrada_a_mano_deja_de_estar(tmp_path: Path) -> None:
    base = tmp_path / "coleccion"
    conceptos = Conceptos(str(base), ["aprobada", "rechazada"], ["rechazada"])
    (base / "rechazada").rmdir()

    conceptos.recargar()

    assert conceptos.nombres() == ["aprobada"]


def test_mapa_es_lo_que_la_memoria_tiene_que_recordar(tmp_path: Path) -> None:
    base = tmp_path / "coleccion"
    _foto(base / "aprobada", "a.jpg")
    _foto(base / "rechazada", "b.jpg")
    _foto(base / "rechazada", "c.jpg")

    mapa = Conceptos(str(base)).mapa()

    assert sorted(mapa) == ["aprobada", "rechazada"]
    assert len(mapa["rechazada"]) == 2


def test_sanear_deja_nombres_de_carpeta_validos() -> None:
    assert sanear("Diente Astillado") == "Diente_Astillado"
    assert sanear("  filo  quemado  ") == "filo_quemado"
    assert sanear("a" * 80) == "a" * 24
    assert sanear("!!!") == ""
