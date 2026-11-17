"""Pruebas del panel de conceptos: lo que ve y lo que aprieta el operario.

El panel no abre ventanas ni lee teclas --eso lo hace run_live-- asi que las dos mitades
se prueban por separado y sin pantalla: aca se arma la imagen y se revisan los botones que
quedaron, y mas abajo se le mandan teclas a los manejadores de run_live y se mira que
cambio en disco.

LO QUE MAS SE CUIDA
-------------------
1. Que escribiendo el nombre de un concepto, la 'q' sea una letra y no la orden de salir
   del programa. Es el error clasico de un modo de texto encima de un bucle de teclas, y
   aca costaria cerrar el programa en la mitad de una palabra.
2. Que borrar un concepto borre EL QUE ESTA ELEGIDO, incluso despues de que la lista haya
   cambiado debajo.
"""

from __future__ import annotations

import json
import re
import shutil
import zipfile
from pathlib import Path

import cv2
import numpy as np
import pytest

from afilado import panel
from afilado.cli import run_live
from afilado.conceptos import Concepto, Conceptos
from afilado.ordenes import Orden, Ordenes, Resumen

_ESC = 27
_ENTER = 13
_ALTO = 720


def _foto(carpeta: Path, nombre: str) -> Path:
    carpeta.mkdir(parents=True, exist_ok=True)
    ruta = carpeta / nombre
    assert cv2.imwrite(str(ruta), np.full((80, 120, 3), 128, np.uint8))
    return ruta


class _GaleriaFalsa:
    """La galeria de fotos de modelo, que en el panel es un destino mas."""

    def __init__(self, carpeta: Path) -> None:
        self.carpeta = carpeta
        self.recargada = 0

    @property
    def cantidad(self) -> int:
        return len(list(self.carpeta.glob("*.jpg"))) if self.carpeta.is_dir() else 0

    def recargar(self, preferida: str | None = None) -> None:
        self.recargada += 1


@pytest.fixture
def taller(tmp_path: Path):
    """Un taller con dos conceptos con fotos y una galeria de modelos."""
    base = tmp_path / "coleccion"
    _foto(base / "aprobada", "a1.jpg")
    _foto(base / "aprobada", "a2.jpg")
    _foto(base / "rechazada", "r1.jpg")
    galeria = _GaleriaFalsa(tmp_path / "referencia")
    _foto(galeria.carpeta, "modelo.jpg")
    conceptos = Conceptos(str(base))
    return conceptos, galeria


def _estado(conceptos: Conceptos, galeria: _GaleriaFalsa, **kwargs) -> panel.EstadoPanel:
    return panel.EstadoPanel(
        conceptos=run_live._lista_para_panel(conceptos, galeria), **kwargs
    )


# ---------------------------------------------------------------------------
# Lo que se dibuja
# ---------------------------------------------------------------------------


def test_la_lista_muestra_todos_los_destinos(taller) -> None:
    conceptos, galeria = taller
    pantalla = panel.componer(_estado(conceptos, galeria), _ALTO)

    assert pantalla.imagen.shape[0] == _ALTO
    assert pantalla.imagen.shape[1] == panel.ANCHO_LIENZO
    # Una fila por destino, mas los botones de abajo.
    assert {"elegir:0", "elegir:1", "elegir:2"} <= set(pantalla.botones)
    assert {"nuevo", "borrar", "fotos", "volver"} <= set(pantalla.botones)


def test_el_panel_mide_lo_mismo_que_la_pantalla_principal(taller) -> None:
    """Si cambiara de tamano, la ventana saltaria al abrirlo y los clics caerian corridos."""
    conceptos, galeria = taller
    for alto in (520, 672, 900):
        assert panel.componer(_estado(conceptos, galeria), alto).imagen.shape[0] == alto


def test_sin_foto_pendiente_no_ofrece_guardar(taller) -> None:
    conceptos, galeria = taller

    pantalla = panel.componer(_estado(conceptos, galeria), _ALTO)

    assert "guardar_aqui" not in pantalla.botones


def test_con_foto_pendiente_ofrece_guardar(taller) -> None:
    conceptos, galeria = taller
    estado = _estado(conceptos, galeria, pendiente=np.full((60, 80, 3), 200, np.uint8))

    pantalla = panel.componer(estado, _ALTO)

    assert "guardar_aqui" in pantalla.botones


def test_el_modo_fotos_arma_una_grilla(taller) -> None:
    conceptos, galeria = taller
    estado = _estado(conceptos, galeria, elegido=1, modo=panel.MODO_FOTOS)
    estado.fotos = conceptos.fotos("aprobada")

    pantalla = panel.componer(estado, _ALTO, panel.Miniaturas())

    assert {"foto:0", "foto:1", "borrar_foto", "volver_lista"} <= set(pantalla.botones)


def test_el_modo_nuevo_no_ofrece_borrar_nada(taller) -> None:
    conceptos, galeria = taller
    estado = _estado(conceptos, galeria, modo=panel.MODO_NUEVO, escribiendo="filo")

    pantalla = panel.componer(estado, _ALTO)

    assert set(pantalla.botones) == {"confirmar_nuevo", "volver_lista"}


def test_el_modo_borrar_pide_confirmacion(taller) -> None:
    conceptos, galeria = taller
    estado = _estado(conceptos, galeria, elegido=1, modo=panel.MODO_BORRAR)

    pantalla = panel.componer(estado, _ALTO)

    assert set(pantalla.botones) == {"confirmar_borrado", "volver_lista"}


def test_se_dibuja_sin_un_solo_concepto(tmp_path: Path) -> None:
    """Recien instalado, o despues de borrar todo: no puede quedar una pantalla en blanco."""
    estado = panel.EstadoPanel(conceptos=[])

    pantalla = panel.componer(estado, _ALTO)

    assert pantalla.imagen.shape[0] == _ALTO
    assert "nuevo" in pantalla.botones


def test_las_miniaturas_se_leen_una_sola_vez(tmp_path: Path) -> None:
    ruta = _foto(tmp_path, "una.jpg")
    minis = panel.Miniaturas()

    primera = minis.obtener(ruta)
    ruta.unlink()
    segunda = minis.obtener(ruta)

    assert primera is not None and segunda is not None
    assert np.array_equal(primera, segunda)


def test_una_foto_ilegible_no_rompe_la_grilla(tmp_path: Path) -> None:
    rota = tmp_path / "rota.jpg"
    rota.write_bytes(b"no soy una imagen")
    estado = panel.EstadoPanel(
        conceptos=[Concepto(nombre="aprobada", carpeta=tmp_path, fotos=1)],
        modo=panel.MODO_FOTOS,
        fotos=[rota],
    )

    pantalla = panel.componer(estado, _ALTO, panel.Miniaturas())

    assert "foto:0" in pantalla.botones


# ---------------------------------------------------------------------------
# Las teclas y los clics (run_live)
# ---------------------------------------------------------------------------


def _manejar(pan, conceptos, galeria, accion=None, tecla=255, archivadas=None,
             ordenes=None):
    minis = panel.Miniaturas()

    def archivar(destino: str) -> str:
        if archivadas is not None:
            archivadas.append(destino)
        return f"GUARDADA como {destino}"

    def refrescar(recalcular: bool = True) -> None:
        conceptos.recargar()
        anterior = pan.concepto_elegido.nombre if pan.concepto_elegido else None
        pan.conceptos = run_live._lista_para_panel(conceptos, galeria)
        pan.elegido = next(
            (i for i, c in enumerate(pan.conceptos) if c.nombre == anterior), 0
        )

    return run_live._atender_panel(
        accion, tecla, pan, conceptos, galeria, minis, archivar, refrescar,
        ordenes if ordenes is not None else Ordenes(str(conceptos.carpeta / "_ordenes")),
    )


def test_la_q_cierra_el_panel(taller) -> None:
    conceptos, galeria = taller
    pan = _estado(conceptos, galeria)

    sigue, _ = _manejar(pan, conceptos, galeria, tecla=ord("q"))

    assert sigue is False


def test_escribiendo_un_nombre_la_q_es_una_letra(taller) -> None:
    """La prueba que justifica que el panel se atienda antes que las teclas de siempre."""
    conceptos, galeria = taller
    pan = _estado(conceptos, galeria, modo=panel.MODO_NUEVO)

    for letra in "quemado":
        sigue, _ = _manejar(pan, conceptos, galeria, tecla=ord(letra))
        assert sigue is True

    assert pan.escribiendo == "quemado"


def test_crear_un_concepto_desde_el_panel(taller) -> None:
    conceptos, galeria = taller
    pan = _estado(conceptos, galeria)

    _manejar(pan, conceptos, galeria, tecla=ord("n"))
    for letra in "filo quemado":
        _manejar(pan, conceptos, galeria, tecla=ord(letra))
    sigue, mensaje = _manejar(pan, conceptos, galeria, tecla=_ENTER)

    assert sigue is True
    assert "filo_quemado" in (mensaje or "")
    assert "filo_quemado" in conceptos.nombres()
    assert pan.modo == panel.MODO_LISTA
    # Y queda elegido el que se acaba de crear, que es lo que el operario va a usar ahora.
    assert pan.concepto_elegido.nombre == "filo_quemado"


def test_un_nombre_vacio_no_crea_nada(taller) -> None:
    conceptos, galeria = taller
    pan = _estado(conceptos, galeria, modo=panel.MODO_NUEVO)

    _manejar(pan, conceptos, galeria, tecla=_ENTER)

    assert pan.modo == panel.MODO_NUEVO
    assert pan.aviso


def test_un_nombre_repetido_lo_dice_en_la_pantalla(taller) -> None:
    conceptos, galeria = taller
    pan = _estado(conceptos, galeria, modo=panel.MODO_NUEVO, escribiendo="aprobada")

    _manejar(pan, conceptos, galeria, tecla=_ENTER)

    assert "ya hay" in (pan.aviso or "")
    assert pan.modo == panel.MODO_NUEVO


def test_el_retroceso_borra_la_ultima_letra(taller) -> None:
    conceptos, galeria = taller
    pan = _estado(conceptos, galeria, modo=panel.MODO_NUEVO, escribiendo="filoo")

    _manejar(pan, conceptos, galeria, tecla=8)

    assert pan.escribiendo == "filo"


def test_escapar_deja_el_nombre_a_medias_sin_crear_nada(taller) -> None:
    conceptos, galeria = taller
    pan = _estado(conceptos, galeria, modo=panel.MODO_NUEVO, escribiendo="a medio escribir")

    sigue, _ = _manejar(pan, conceptos, galeria, tecla=_ESC)

    assert sigue is True and pan.modo == panel.MODO_LISTA
    assert pan.escribiendo == ""
    assert "a_medio_escribir" not in conceptos.nombres()


def test_los_numeros_eligen_un_concepto(taller) -> None:
    conceptos, galeria = taller
    pan = _estado(conceptos, galeria)

    _manejar(pan, conceptos, galeria, tecla=ord("3"))

    assert pan.elegido == 2


def test_guardar_aca_archiva_en_el_elegido_y_cierra(taller) -> None:
    conceptos, galeria = taller
    pan = _estado(conceptos, galeria, elegido=2)
    archivadas: list[str] = []

    sigue, mensaje = _manejar(
        pan, conceptos, galeria, accion="guardar_aqui", archivadas=archivadas
    )

    assert sigue is False
    assert archivadas == [pan.conceptos[2].nombre]
    assert "GUARDADA" in (mensaje or "")


def test_borrar_un_concepto_pide_confirmacion_primero(taller) -> None:
    conceptos, galeria = taller
    pan = _estado(conceptos, galeria, elegido=1)

    _manejar(pan, conceptos, galeria, tecla=ord("b"))

    assert pan.modo == panel.MODO_BORRAR
    assert len(conceptos.nombres()) == 2  # todavia no se borro nada


def test_decir_que_no_deja_todo_como_estaba(taller) -> None:
    conceptos, galeria = taller
    pan = _estado(conceptos, galeria, elegido=1, modo=panel.MODO_BORRAR)

    _manejar(pan, conceptos, galeria, tecla=ord("n"))

    assert pan.modo == panel.MODO_LISTA
    assert len(conceptos.nombres()) == 2


def test_confirmar_borra_el_concepto_elegido(taller) -> None:
    conceptos, galeria = taller
    pan = _estado(conceptos, galeria, elegido=1, modo=panel.MODO_BORRAR)
    nombre = pan.conceptos[1].nombre

    sigue, mensaje = _manejar(pan, conceptos, galeria, accion="confirmar_borrado")

    assert sigue is True
    assert nombre not in conceptos.nombres()
    assert "papelera" in (mensaje or "")


def test_la_galeria_de_modelos_no_se_puede_borrar(taller) -> None:
    """No es un concepto: es la foto contra la que se compara."""
    conceptos, galeria = taller
    pan = _estado(conceptos, galeria, elegido=0)
    assert pan.conceptos[0].especial

    _manejar(pan, conceptos, galeria, tecla=ord("b"))

    assert pan.modo == panel.MODO_LISTA
    assert pan.aviso


def test_ver_fotos_lista_las_del_concepto_elegido(taller) -> None:
    conceptos, galeria = taller
    pan = _estado(conceptos, galeria, elegido=1)

    _manejar(pan, conceptos, galeria, tecla=ord("v"))

    assert pan.modo == panel.MODO_FOTOS
    assert {p.name for p in pan.fotos} == {"a1.jpg", "a2.jpg"}


def test_ver_fotos_del_destino_de_modelos(taller) -> None:
    conceptos, galeria = taller
    pan = _estado(conceptos, galeria, elegido=0)

    _manejar(pan, conceptos, galeria, tecla=ord("v"))

    assert [p.name for p in pan.fotos] == ["modelo.jpg"]


def test_borrar_una_foto_la_saca_de_la_lista(taller) -> None:
    conceptos, galeria = taller
    pan = _estado(conceptos, galeria, elegido=1, modo=panel.MODO_FOTOS)
    pan.fotos = conceptos.fotos("aprobada")
    pan.foto_elegida = 0
    borrada = pan.fotos[0]

    sigue, mensaje = _manejar(pan, conceptos, galeria, accion="borrar_foto")

    assert sigue is True
    assert not borrada.exists()
    assert len(pan.fotos) == 1
    assert "PAPELERA" in (mensaje or "")


def test_borrar_la_ultima_foto_no_deja_un_indice_colgado(taller) -> None:
    conceptos, galeria = taller
    pan = _estado(conceptos, galeria, elegido=2, modo=panel.MODO_FOTOS)
    pan.fotos = conceptos.fotos("rechazada")
    pan.foto_elegida = 0

    _manejar(pan, conceptos, galeria, accion="borrar_foto")

    assert pan.fotos == []
    assert pan.foto_elegida == 0
    # Y la pantalla se sigue dibujando con la carpeta vacia.
    assert panel.componer(pan, _ALTO, panel.Miniaturas()).imagen.shape[0] == _ALTO


def test_volver_de_las_fotos_no_cierra_el_panel(taller) -> None:
    conceptos, galeria = taller
    pan = _estado(conceptos, galeria, elegido=1, modo=panel.MODO_FOTOS)

    sigue, _ = _manejar(pan, conceptos, galeria, tecla=_ESC)

    assert sigue is True and pan.modo == panel.MODO_LISTA


# ---------------------------------------------------------------------------
# Lo que se arreglo despues de la primera version
# ---------------------------------------------------------------------------


def _muchos(cantidad: int) -> list[Concepto]:
    return [
        Concepto(nombre=f"concepto_{i:02d}", carpeta=Path("."), fotos=i)
        for i in range(cantidad)
    ]


def test_con_mas_conceptos_que_filas_el_elegido_igual_se_ve() -> None:
    """Antes se dibujaban los que entraban y el resto se anunciaba con '...y 12 mas': los
    que no se veian no se podian elegir ni con el mouse ni con las teclas."""
    estado = panel.EstadoPanel(conceptos=_muchos(25), elegido=24)

    pantalla = panel.componer(estado, _ALTO)

    assert "elegir:24" in pantalla.botones
    # Y el primero quedo fuera de la vista, que es justamente para lo que se corrio.
    assert "elegir:0" not in pantalla.botones


def test_al_principio_de_la_lista_no_se_corre_nada() -> None:
    estado = panel.EstadoPanel(conceptos=_muchos(25), elegido=0)

    pantalla = panel.componer(estado, _ALTO)

    assert "elegir:0" in pantalla.botones


def test_moverse_con_s_llega_al_ultimo(taller) -> None:
    conceptos, galeria = taller
    pan = _estado(conceptos, galeria)
    pan.conceptos = _muchos(25)

    for _ in range(24):
        _manejar(pan, conceptos, galeria, tecla=ord("s"))

    assert pan.elegido == 24
    assert "elegir:24" in panel.componer(pan, _ALTO).botones


def test_enter_no_confirma_el_borrado_de_un_concepto(taller) -> None:
    """ENTER es 'guardar aca' en la lista. Si tambien confirmara aca, un 'b' de mas
    seguido del ENTER de siempre se llevaria puesto un concepto entero."""
    conceptos, galeria = taller
    pan = _estado(conceptos, galeria, elegido=1, modo=panel.MODO_BORRAR)
    nombre = pan.conceptos[1].nombre

    _manejar(pan, conceptos, galeria, tecla=_ENTER)

    assert nombre in conceptos.nombres()
    assert pan.modo == panel.MODO_BORRAR


def test_la_s_si_confirma_el_borrado(taller) -> None:
    conceptos, galeria = taller
    pan = _estado(conceptos, galeria, elegido=1, modo=panel.MODO_BORRAR)
    nombre = pan.conceptos[1].nombre

    _manejar(pan, conceptos, galeria, tecla=ord("s"))

    assert nombre not in conceptos.nombres()


def test_el_punto_de_color_marca_si_el_concepto_es_para_tirar(taller) -> None:
    """Sin esto, todo concepto que creara el taller nacia 'sano' y un diente astillado
    salia en VERDE: la pantalla pinta segun esta marca."""
    conceptos, galeria = taller
    pan = _estado(conceptos, galeria, elegido=1)   # 0 es la galeria de modelos
    nombre = pan.conceptos[1].nombre
    assert not pan.conceptos[1].defecto

    sigue, mensaje = _manejar(pan, conceptos, galeria, tecla=ord("d"))

    assert sigue is True
    assert "TIRAR" in (mensaje or "")
    assert nombre in conceptos.defectos()
    # Y se recuerda para la proxima vez que se abra el programa.
    assert nombre in Conceptos(str(conceptos.carpeta)).defectos()


def test_la_marca_se_puede_sacar(taller) -> None:
    conceptos, galeria = taller
    conceptos.marcar_defecto("aprobada", True)
    pan = _estado(conceptos, galeria, elegido=1)

    _, mensaje = _manejar(pan, conceptos, galeria, tecla=ord("d"))

    assert "BUENA" in (mensaje or "")
    assert conceptos.defectos() == []


def test_el_punto_es_un_boton(taller) -> None:
    conceptos, galeria = taller
    estado = _estado(conceptos, galeria)

    pantalla = panel.componer(estado, _ALTO)

    # La galeria de modelos no lo tiene: no es ni bueno ni malo.
    assert "defecto:0" not in pantalla.botones
    assert "defecto:1" in pantalla.botones
    # Y apretarlo marca, no elige.
    x1, y1, x2, y2 = pantalla.botones["defecto:1"]
    assert pantalla.boton_en((x1 + x2) // 2, (y1 + y2) // 2) == "defecto:1"


def test_la_galeria_de_modelos_no_se_puede_marcar(taller) -> None:
    conceptos, galeria = taller
    pan = _estado(conceptos, galeria, elegido=0)

    _manejar(pan, conceptos, galeria, tecla=ord("d"))

    assert pan.aviso
    assert conceptos.defectos() == []


# ---------------------------------------------------------------------------
# La orden de afilado: cargarla a mano y contar diente por diente
# ---------------------------------------------------------------------------


def test_la_pantalla_de_la_orden_pide_los_datos(taller) -> None:
    conceptos, galeria = taller
    estado = _estado(conceptos, galeria, modo=panel.MODO_ORDEN, cargando_orden=True)

    pantalla = panel.componer(estado, _ALTO)

    assert {"campo:0", "campo:4", "crear_orden", "volver_orden"} <= set(pantalla.botones)


def test_cargar_una_orden_a_mano(taller, tmp_path: Path) -> None:
    conceptos, galeria = taller
    ordenes = Ordenes(str(tmp_path / "ordenes"))
    pan = _estado(conceptos, galeria, modo=panel.MODO_ORDEN, cargando_orden=True)

    for texto, campo in (("1234", 0), ("NP-998", 1), ("Carpinteria", 2),
                         ("3", 3), ("24", 4)):
        pan.campo = campo
        for letra in texto:
            _manejar(pan, conceptos, galeria, tecla=ord(letra), ordenes=ordenes)
    sigue, mensaje = _manejar(pan, conceptos, galeria, accion="crear_orden", ordenes=ordenes)

    assert sigue is True
    assert "1234" in (mensaje or "")
    assert ordenes.abierta is not None
    assert len(ordenes.abierta.sierras) == 3
    assert ordenes.abierta.sierras[0].dientes == 24
    assert ordenes.abierta.nota_pedido == "NP-998"


def test_en_cuantas_sierras_no_entran_letras(taller, tmp_path: Path) -> None:
    conceptos, galeria = taller
    ordenes = Ordenes(str(tmp_path / "ordenes"))
    pan = _estado(conceptos, galeria, modo=panel.MODO_ORDEN, cargando_orden=True, campo=3)

    _manejar(pan, conceptos, galeria, tecla=ord("x"), ordenes=ordenes)

    assert pan.valores[3] == ""
    assert "numero" in (pan.aviso or "")


def test_una_orden_sin_numero_no_se_crea(taller, tmp_path: Path) -> None:
    conceptos, galeria = taller
    ordenes = Ordenes(str(tmp_path / "ordenes"))
    pan = _estado(conceptos, galeria, modo=panel.MODO_ORDEN, cargando_orden=True)
    pan.valores = ["", "", "", "2", "10"]

    _manejar(pan, conceptos, galeria, accion="crear_orden", ordenes=ordenes)

    assert ordenes.abierta is None
    assert pan.aviso


def test_no_se_pisa_una_orden_que_ya_existe(taller, tmp_path: Path) -> None:
    """El numero de orden es la identidad del informe: si se pisa, se pierde el anterior."""
    conceptos, galeria = taller
    ordenes = Ordenes(str(tmp_path / "ordenes"))
    ordenes.abrir(Orden.nueva("1234", 1, 5))
    ordenes.cerrar()
    pan = _estado(conceptos, galeria, modo=panel.MODO_ORDEN, cargando_orden=True)
    pan.valores = ["1234", "", "", "2", "10"]

    _manejar(pan, conceptos, galeria, accion="crear_orden", ordenes=ordenes)

    assert "ya hay una orden" in (pan.aviso or "")
    assert ordenes.abierta is None


def test_la_pantalla_de_la_orden_abierta_muestra_el_avance(taller, tmp_path: Path) -> None:
    conceptos, galeria = taller
    orden = Orden.nueva("1234", 2, 6)
    orden.anotar(estado="afilado", aprobado=True)
    estado = _estado(conceptos, galeria, modo=panel.MODO_ORDEN, orden=orden)

    pantalla = panel.componer(estado, _ALTO)

    assert {"siguiente_sierra", "saltear_diente", "informe", "cerrar_orden"} <= set(
        pantalla.botones
    )


def test_saltear_un_diente_desde_la_pantalla(taller, tmp_path: Path) -> None:
    conceptos, galeria = taller
    ordenes = Ordenes(str(tmp_path / "ordenes"))
    orden = ordenes.abrir(Orden.nueva("1234", 1, 4))
    pan = _estado(conceptos, galeria, modo=panel.MODO_ORDEN, orden=orden)

    _, mensaje = _manejar(pan, conceptos, galeria, tecla=ord("x"), ordenes=ordenes)

    assert "DIENTE 1" in (mensaje or "")
    assert orden.diente().numero == 2
    # Y quedo guardado en disco, no solo en memoria. SALTEADO, no revisado: el conteo
    # avanza al diente 2 pero el informe no puede decir que se reviso uno, porque nadie lo
    # miro. Esta assertion decia `total_revisados == 1` y era el defecto escrito como
    # prueba: salteando los 24 dientes, el informe daba la sierra por revisada entera.
    guardada = Ordenes(str(tmp_path / "ordenes")).abierta
    assert guardada.total_revisados == 0
    assert guardada.total_salteados == 1


def test_el_informe_se_guarda_desde_la_pantalla(taller, tmp_path: Path) -> None:
    conceptos, galeria = taller
    ordenes = Ordenes(str(tmp_path / "ordenes"))
    orden = ordenes.abrir(Orden.nueva("1234", 1, 2))
    orden.anotar(estado="roto", aprobado=False)
    pan = _estado(conceptos, galeria, modo=panel.MODO_ORDEN, orden=orden)

    _, mensaje = _manejar(pan, conceptos, galeria, accion="informe", ordenes=ordenes)

    assert "informe_1234" in (mensaje or "")
    assert (tmp_path / "ordenes" / "informe_1234.txt").is_file()
    assert (tmp_path / "ordenes" / "informe_1234.csv").is_file()


def test_cerrar_la_orden_deja_el_informe_escrito(taller, tmp_path: Path) -> None:
    conceptos, galeria = taller
    ordenes = Ordenes(str(tmp_path / "ordenes"))
    orden = ordenes.abrir(Orden.nueva("1234", 1, 2))
    pan = _estado(conceptos, galeria, modo=panel.MODO_ORDEN, orden=orden)

    _manejar(pan, conceptos, galeria, accion="cerrar_orden", ordenes=ordenes)

    assert ordenes.abierta is None
    assert (tmp_path / "ordenes" / "informe_1234.txt").is_file()


def test_de_la_orden_se_vuelve_al_HISTORIAL_sin_cerrar_el_panel(taller, tmp_path) -> None:
    """La jerarquia es taller -> ORDENES -> esta orden, asi que subir un piso desde una
    orden es la lista de ordenes. Antes caia en los CONCEPTOS --que no tienen nada que ver
    con lo que se estaba haciendo-- y de ahi no habia forma de volver a la orden de al
    lado."""
    conceptos, galeria = taller
    ordenes = Ordenes(str(tmp_path / "ordenes"))
    orden = ordenes.abrir(Orden.nueva("1234", 1, 2))
    pan = _estado(conceptos, galeria, modo=panel.MODO_ORDEN, orden=orden)

    sigue, _ = _manejar(pan, conceptos, galeria, tecla=_ESC, ordenes=ordenes)

    assert sigue is True and pan.modo == panel.MODO_ORDENES
    # Y el historial queda leido: la lista de antes mostraria el avance viejo.
    assert [fila.numero for fila in pan.historial] == ["1234"]


def test_el_formulario_avisa_que_el_numero_de_dientes_es_para_todas_las_sierras() -> None:
    """Orden.nueva pone ese numero en TODAS las sierras de la orden, asi que una orden con
    una sierra de 24 y otra de 36 no se puede cargar: hay que partirla en dos ordenes. Sin
    decirlo, el operario pone un numero y el informe declara "SIERRA 2: 24 dientes, 24
    revisados" de una sierra de 36."""
    etiquetas = dict(panel.CAMPOS_ORDEN)

    assert "Dientes por sierra (el mismo)" in etiquetas


# ---------------------------------------------------------------------------
# EL HISTORIAL DE ORDENES. Las teclas, no el dibujo.
#
# Lo que mas se cuida: que la 'e' y la 'b' trabajen sobre la fila que el operario VE
# elegida, y que un borrado no deje el indice apuntando a una orden que ya no esta. Es el
# mismo error que costaba un concepto entero en la lista de conceptos.
# ---------------------------------------------------------------------------


def _con_ordenes(tmp_path: Path, *numeros: str) -> Ordenes:
    """Un taller con varias ordenes guardadas. La ultima queda como la abierta."""
    ordenes = Ordenes(str(tmp_path / "ordenes"))
    for numero in numeros:
        ordenes.abrir(Orden.nueva(numero, 1, 2))
    return ordenes


def _historial(conceptos, galeria, ordenes, numero: str = "") -> panel.EstadoPanel:
    """El panel ya parado en el historial, con una fila elegida a proposito.

    La fila se busca por numero y no se escribe el indice a mano: dos ordenes creadas en el
    mismo segundo tienen la misma fecha, asi que el orden entre ellas no esta garantizado y
    una prueba que diga 'la fila 0' pasaria o fallaria segun el reloj.
    """
    pan = _estado(conceptos, galeria)
    run_live._abrir_historial(pan, ordenes)
    if numero:
        pan.orden_elegida = next(
            indice for indice, fila in enumerate(pan.historial)
            if numero in (fila.numero, fila.archivo)
        )
    return pan


def test_el_historial_se_abre_con_la_orden_en_curso_ya_elegida(taller, tmp_path) -> None:
    """El que aprieta 'o' con una orden en curso quiere esa. Si cayera parado en la
    primera fila --la mas nueva, que puede ser otra-- un ENTER de reflejo retomaria la
    que no era."""
    conceptos, galeria = taller
    ordenes = _con_ordenes(tmp_path, "111", "222")
    pan = _estado(conceptos, galeria)

    run_live._abrir_historial(pan, ordenes)

    assert pan.modo == panel.MODO_ORDENES
    assert len(pan.historial) == 2
    assert pan.historial[pan.orden_elegida].numero == "222"


def test_el_enter_del_historial_retoma_la_orden_elegida(taller, tmp_path) -> None:
    conceptos, galeria = taller
    ordenes = _con_ordenes(tmp_path, "111", "222")
    pan = _historial(conceptos, galeria, ordenes, "111")

    sigue, mensaje = _manejar(pan, conceptos, galeria, tecla=_ENTER, ordenes=ordenes)

    assert sigue is True and pan.modo == panel.MODO_ORDEN
    assert ordenes.abierta is not None and ordenes.abierta.numero == "111"
    assert "111" in (mensaje or "")
    # La fila de atras tiene que decir que ahora ESTA es la abierta.
    abiertas = [fila.numero for fila in pan.historial if fila.esta_abierta]
    assert abiertas == ["111"]


def test_una_orden_cerrada_no_se_retoma_y_se_dice_por_que(taller, tmp_path) -> None:
    """Su informe ya se imprimio y se entrego: anotarle dientes encima deja el papel del
    cliente diciendo menos de lo que el archivo guarda. Y el aviso no puede culpar al
    disco, que no tiene nada que ver."""
    conceptos, galeria = taller
    ordenes = _con_ordenes(tmp_path, "111")
    ordenes.cerrar()
    pan = _historial(conceptos, galeria, ordenes, "111")

    sigue, _ = _manejar(pan, conceptos, galeria, tecla=_ENTER, ordenes=ordenes)

    assert sigue is True and pan.modo == panel.MODO_ORDENES
    assert ordenes.abierta is None
    assert "cerro" in (pan.aviso or "") and "disco" not in (pan.aviso or "")


def test_la_e_exporta_la_orden_elegida_en_un_solo_archivo(taller, tmp_path) -> None:
    conceptos, galeria = taller
    ordenes = _con_ordenes(tmp_path, "111", "222")
    pan = _historial(conceptos, galeria, ordenes, "111")

    sigue, mensaje = _manejar(pan, conceptos, galeria, tecla=ord("e"), ordenes=ordenes)

    zips = list((tmp_path / "ordenes" / "exportado").glob("*.zip"))
    assert sigue is True and pan.modo == panel.MODO_ORDENES
    assert len(zips) == 1 and "111" in zips[0].name
    assert zips[0].name in (mensaje or "")


def test_la_b_del_historial_no_borra_sola(taller, tmp_path) -> None:
    """En el historial ENTER es 'retomar esta orden', la tecla de todos los dias. Si la
    confirmacion aceptara ENTER, una 'b' de mas seguida del ENTER de siempre se llevaria
    la orden y sus pruebas."""
    conceptos, galeria = taller
    ordenes = _con_ordenes(tmp_path, "111")
    pan = _historial(conceptos, galeria, ordenes, "111")

    _manejar(pan, conceptos, galeria, tecla=ord("b"), ordenes=ordenes)
    assert pan.modo == panel.MODO_BORRAR_ORDEN

    _manejar(pan, conceptos, galeria, tecla=_ENTER, ordenes=ordenes)
    assert (tmp_path / "ordenes" / "111.json").is_file()


def test_la_s_manda_la_orden_a_la_papelera_y_la_saca_de_la_lista(taller, tmp_path) -> None:
    conceptos, galeria = taller
    ordenes = _con_ordenes(tmp_path, "111", "222")
    pan = _historial(conceptos, galeria, ordenes, "111")
    _manejar(pan, conceptos, galeria, tecla=ord("b"), ordenes=ordenes)

    sigue, mensaje = _manejar(pan, conceptos, galeria, tecla=ord("s"), ordenes=ordenes)

    assert sigue is True and pan.modo == panel.MODO_ORDENES
    assert not (tmp_path / "ordenes" / "111.json").is_file()
    assert list((tmp_path / "ordenes" / "_papelera").glob("*/111.json"))
    assert [fila.numero for fila in pan.historial] == ["222"]
    # El cartel dice que NO se perdio: la orden es lo que contesta un reclamo.
    assert "PAPELERA" in (mensaje or "")


def test_borrar_la_orden_abierta_deja_el_programa_sin_orden(taller, tmp_path) -> None:
    """Si el .abierta siguiera apuntando al archivo que se fue, el programa arrancaria la
    proxima vez buscando una orden que no esta."""
    conceptos, galeria = taller
    ordenes = _con_ordenes(tmp_path, "111")
    pan = _historial(conceptos, galeria, ordenes, "111")
    _manejar(pan, conceptos, galeria, tecla=ord("b"), ordenes=ordenes)

    _manejar(pan, conceptos, galeria, tecla=ord("s"), ordenes=ordenes)

    assert ordenes.abierta is None and pan.orden is None
    assert pan.historial == []


def test_borrar_la_ultima_orden_no_deja_un_indice_colgado(taller, tmp_path) -> None:
    conceptos, galeria = taller
    ordenes = _con_ordenes(tmp_path, "111")
    pan = _historial(conceptos, galeria, ordenes, "111")
    _manejar(pan, conceptos, galeria, tecla=ord("b"), ordenes=ordenes)

    _manejar(pan, conceptos, galeria, tecla=ord("s"), ordenes=ordenes)

    assert pan.orden_elegida == 0 and pan.pagina_ordenes == 0
    # Y las teclas que trabajan sobre la fila elegida no explotan con la lista vacia.
    assert _manejar(pan, conceptos, galeria, tecla=ord("e"), ordenes=ordenes) == (True, None)
    assert _manejar(pan, conceptos, galeria, tecla=ord("b"), ordenes=ordenes) == (True, None)


def test_cancelar_el_borrado_deja_la_orden_donde_estaba(taller, tmp_path) -> None:
    conceptos, galeria = taller
    ordenes = _con_ordenes(tmp_path, "111")
    pan = _historial(conceptos, galeria, ordenes, "111")
    _manejar(pan, conceptos, galeria, tecla=ord("b"), ordenes=ordenes)

    _manejar(pan, conceptos, galeria, tecla=ord("n"), ordenes=ordenes)

    assert pan.modo == panel.MODO_ORDENES
    assert (tmp_path / "ordenes" / "111.json").is_file()


def test_la_n_del_historial_abre_el_formulario_de_orden_nueva(taller, tmp_path) -> None:
    conceptos, galeria = taller
    ordenes = _con_ordenes(tmp_path, "111")
    pan = _historial(conceptos, galeria, ordenes, "111")

    _manejar(pan, conceptos, galeria, tecla=ord("n"), ordenes=ordenes)

    assert pan.modo == panel.MODO_ORDEN and pan.cargando_orden is True
    assert pan.valores == ["" for _ in panel.CAMPOS_ORDEN] and pan.campo == 0
    # El 'retomar' de la vez anterior no puede quedar puesto: el primer ENTER de este
    # formulario retomaria una orden sin preguntar nada.
    assert pan.retomar == ""


def test_el_esc_del_historial_cierra_el_panel(taller, tmp_path) -> None:
    """El historial es el primer piso: no hay nada mas arriba a donde subir."""
    conceptos, galeria = taller
    ordenes = _con_ordenes(tmp_path, "111")
    pan = _historial(conceptos, galeria, ordenes, "111")

    assert _manejar(pan, conceptos, galeria, tecla=_ESC, ordenes=ordenes) == (False, None)


def test_una_orden_ilegible_aparece_pero_no_se_abre_ni_se_exporta(taller, tmp_path) -> None:
    """Es justo la que hay que arreglar a mano, asi que tiene fila. Lo que no se hace es
    mostrarla como prueba: no se exporta como reclamo algo que el programa no pudo leer."""
    conceptos, galeria = taller
    ordenes = _con_ordenes(tmp_path, "111")
    (tmp_path / "ordenes" / "777.json").write_text("{esto no es", encoding="utf-8")
    pan = _historial(conceptos, galeria, ordenes, "777")

    _manejar(pan, conceptos, galeria, tecla=_ENTER, ordenes=ordenes)
    assert pan.modo == panel.MODO_ORDENES and "no se puede leer" in (pan.aviso or "")

    _manejar(pan, conceptos, galeria, tecla=ord("e"), ordenes=ordenes)
    assert not (tmp_path / "ordenes" / "exportado").exists()

    # Pero SI se puede sacar del medio.
    _manejar(pan, conceptos, galeria, tecla=ord("b"), ordenes=ordenes)
    _manejar(pan, conceptos, galeria, tecla=ord("s"), ordenes=ordenes)
    assert not (tmp_path / "ordenes" / "777.json").is_file()


def test_los_digitos_solo_eligen_las_filas_de_esta_pagina(taller, tmp_path) -> None:
    """En pantalla entran ocho filas: medido, el panel se compone a 672 px de alto y cada
    fila ocupa 56. Un 9 elegiria una orden que no se ve, y la 'b' siguiente la mandaria a
    la papelera a ciegas."""
    conceptos, galeria = taller
    ordenes = _con_ordenes(tmp_path, *[f"{numero}" for numero in range(101, 112)])
    pan = _historial(conceptos, galeria, ordenes)
    pan.orden_elegida, pan.pagina_ordenes = 0, 0

    _manejar(pan, conceptos, galeria, tecla=ord("9"), ordenes=ordenes)
    assert pan.orden_elegida == 0

    _manejar(pan, conceptos, galeria, tecla=ord("3"), ordenes=ordenes)
    assert pan.orden_elegida == 2


def test_bajar_desde_la_ultima_fila_de_la_pagina_cambia_de_pagina(taller, tmp_path) -> None:
    """Sin esto la marca desaparecia de la pantalla y la 'e' o la 'b' siguientes
    trabajaban sobre una fila que el operario no veia."""
    conceptos, galeria = taller
    ordenes = _con_ordenes(tmp_path, *[f"{numero}" for numero in range(101, 112)])
    pan = _historial(conceptos, galeria, ordenes)
    pan.orden_elegida, pan.pagina_ordenes = 7, 0

    _manejar(pan, conceptos, galeria, tecla=ord("s"), ordenes=ordenes)

    assert pan.orden_elegida == 8 and pan.pagina_ordenes == 1


def test_el_espacio_pasa_a_la_pagina_siguiente(taller, tmp_path) -> None:
    conceptos, galeria = taller
    ordenes = _con_ordenes(tmp_path, *[f"{numero}" for numero in range(101, 112)])
    pan = _historial(conceptos, galeria, ordenes)
    pan.orden_elegida, pan.pagina_ordenes = 0, 0

    _manejar(pan, conceptos, galeria, tecla=ord(" "), ordenes=ordenes)
    assert pan.pagina_ordenes == 1 and pan.orden_elegida == 8

    # Y de la ultima vuelve a la primera: con dos paginas y once ordenes, no se queda
    # trabado abajo.
    _manejar(pan, conceptos, galeria, tecla=ord(" "), ordenes=ordenes)
    assert pan.pagina_ordenes == 0 and pan.orden_elegida == 0


def test_el_clic_en_una_fila_la_elige(taller, tmp_path) -> None:
    conceptos, galeria = taller
    ordenes = _con_ordenes(tmp_path, "111", "222")
    pan = _historial(conceptos, galeria, ordenes)

    _manejar(pan, conceptos, galeria, accion="elegir_orden:1", ordenes=ordenes)

    assert pan.orden_elegida == 1


def test_la_w_y_la_s_caminan_el_historial_fila_por_fila(taller, tmp_path) -> None:
    """Son las dos unicas teclas para llegar a una orden que no tiene digito en pantalla.
    Si una de las dos no anduviera --o saltara de a dos filas-- a la orden del reclamo solo
    se llegaria con el mouse, y el operario tiene la sierra en la mano."""
    conceptos, galeria = taller
    ordenes = _con_ordenes(tmp_path, "101", "102", "103")
    pan = _historial(conceptos, galeria, ordenes)
    # Las tres se crean en el mismo segundo, asi que el orden entre ellas no esta
    # garantizado: se camina por indice y se compara contra la lista que quedo dibujada.
    numeros = [fila.numero for fila in pan.historial]
    pan.orden_elegida, pan.pagina_ordenes = 0, 0

    for esperado in (1, 2):
        _manejar(pan, conceptos, galeria, tecla=ord("s"), ordenes=ordenes)
        assert pan.orden_elegida == esperado
        assert pan.historial[pan.orden_elegida].numero == numeros[esperado]

    for esperado in (1, 0):
        _manejar(pan, conceptos, galeria, tecla=ord("w"), ordenes=ordenes)
        assert pan.orden_elegida == esperado
        assert pan.historial[pan.orden_elegida].numero == numeros[esperado]

    # Y desde la primera, la 'w' da la vuelta a la ultima en vez de quedarse trabada.
    _manejar(pan, conceptos, galeria, tecla=ord("w"), ordenes=ordenes)
    assert pan.orden_elegida == 2


def test_moverse_sin_ordenes_o_con_una_sola_no_se_sale_de_la_lista(taller, tmp_path) -> None:
    """La carpeta vacia es el dia que se instala el programa, y una sola orden es el taller
    de todos los dias. Una fila elegida fuera de la lista no se ve: se nota recien cuando la
    'e' o la 'b' siguientes buscan esa fila y el programa se cae con la sierra en el torno."""
    conceptos, galeria = taller
    vacias = Ordenes(str(tmp_path / "sin_ordenes"))
    pan = _historial(conceptos, galeria, vacias)
    assert pan.historial == []

    for tecla in (ord("s"), ord("w"), ord("s")):
        sigue, mensaje = _manejar(pan, conceptos, galeria, tecla=tecla, ordenes=vacias)
        assert (sigue, mensaje) == (True, None)
        assert pan.orden_elegida == 0 and pan.pagina_ordenes == 0
    # Y las teclas que trabajan sobre la fila elegida no buscan una fila que no hay.
    assert _manejar(pan, conceptos, galeria, tecla=_ENTER, ordenes=vacias) == (True, None)
    assert pan.modo == panel.MODO_ORDENES

    una = _con_ordenes(tmp_path, "111")
    pan = _historial(conceptos, galeria, una)
    for tecla in (ord("s"), ord("s"), ord("w")):
        _manejar(pan, conceptos, galeria, tecla=tecla, ordenes=una)
        assert pan.orden_elegida == 0
        assert pan.historial[pan.orden_elegida].numero == "111"


def test_el_esc_de_la_confirmacion_de_borrado_no_se_lleva_la_orden(taller, tmp_path) -> None:
    """ESC es la tecla con la que el operario sale de todas las pantallas del programa, asi
    que aca tiene que ser 'no'. Si confirmara, la salida de siempre borraria la orden --y su
    prueba-- justo cuando el que entro por error quiere irse sin tocar nada."""
    conceptos, galeria = taller
    ordenes = _con_ordenes(tmp_path, "111")
    pan = _historial(conceptos, galeria, ordenes, "111")
    _manejar(pan, conceptos, galeria, tecla=ord("b"), ordenes=ordenes)
    assert pan.modo == panel.MODO_BORRAR_ORDEN

    sigue, mensaje = _manejar(pan, conceptos, galeria, tecla=_ESC, ordenes=ordenes)

    # Vuelve al historial sin cerrar el panel: ESC sube un piso, no dos.
    assert (sigue, mensaje) == (True, None)
    assert pan.modo == panel.MODO_ORDENES
    assert (tmp_path / "ordenes" / "111.json").is_file()
    assert not (tmp_path / "ordenes" / "_papelera").exists()
    assert ordenes.abierta is not None and ordenes.abierta.numero == "111"


# ---------------------------------------------------------------------------
# EL HISTORIAL DE ORDENES. El dibujo, no las teclas.
#
# Las de arriba miran lo que cambia en el disco. Estas miran lo que LEE el operario, que es
# donde vive la regla de no mostrar como prueba algo que no es prueba: una orden vieja que
# solo guarda el nombre pelado de una foto --medido, las 14 que nombran las ordenes 123 y
# 1234 estan todas en la papelera de la coleccion y ninguna viva-- no tiene prueba
# guardada, y la fila tiene que decirlo con esas palabras y no contarla.
#
# El texto se le pide a las funciones de dibujo y no a la imagen: componer() devuelve la
# imagen y los botones, y leer de vuelta las letras de los pixeles seria adivinar.
# ---------------------------------------------------------------------------

# 672 es el alto al que se compone el panel en la PC del taller (pantalla de 1360x720).
# Los otros dos son una ventana baja y la pantalla entera.
_ALTO_TALLER = 672
_ALTOS_DE_VENTANA = (492, _ALTO_TALLER, 720)

_DIBUJOS = {
    panel.MODO_LISTA: panel._dibujar_lista,
    panel.MODO_ORDENES: panel._dibujar_ordenes,
    panel.MODO_BORRAR_ORDEN: panel._dibujar_borrar_orden,
}


def _fila(numero: str = "5113546", **cambios) -> Resumen:
    """Una fila del historial armada a mano, con los numeros de una orden de verdad.

    A mano y no leyendo el disco porque lo que se prueba aca es el dibujo: armar una orden
    con 15 fotos de prueba guardadas para ver si la fila dice "15 CON PRUEBA" seria
    escribir 15 jpg para leer un renglon.
    """
    datos = dict(
        numero=numero, archivo=numero, cliente="Aserradero del Litoral",
        nota_pedido="", abierta="16/11/2026 09:34:20", cerrada="16/11/2026 12:11:25",
        sierras=4, total_dientes=80, revisados=15, rechazados=15,
        con_prueba=0, con_nombre_suelto=0, esta_abierta=False, ilegible=False,
        clave="2026-11-16 09:34:20",
    )
    datos.update(cambios)
    return Resumen(**datos)


def _pantalla_de_ordenes(*filas: Resumen, elegida: int = 0,
                         modo: str = panel.MODO_ORDENES) -> panel.EstadoPanel:
    """El panel parado en el historial (o en la confirmacion) con esas filas."""
    return panel.EstadoPanel(modo=modo, historial=list(filas), orden_elegida=elegida)


def _dibujar(estado: panel.EstadoPanel, alto: int = _ALTO_TALLER):
    """(renglones, carteles de los botones, botones) de la pantalla que pide el estado.

    Las funciones de dibujo devuelven los renglones ANTES de que el pincel los pase a
    pixeles, asi que el texto se les pregunta a ellas.
    """
    lienzo = np.full((alto, panel.ANCHO_LIENZO, 3), panel._FONDO, np.uint8)
    botones: dict = {}
    etiquetas: list = []
    lineas = _DIBUJOS[estado.modo](lienzo, estado, botones, etiquetas)
    return lineas, etiquetas, botones


def _textos(estado: panel.EstadoPanel, alto: int = _ALTO_TALLER) -> list[str]:
    """Todo lo que se lee en esa pantalla: los renglones y los carteles de los botones."""
    lineas, etiquetas, _ = _dibujar(estado, alto)
    return [linea[0] for linea in lineas] + [etiqueta[0] for etiqueta in etiquetas]


def _dice(estado: panel.EstadoPanel, pedazo: str, alto: int = _ALTO_TALLER) -> bool:
    return any(pedazo in texto for texto in _textos(estado, alto))


@pytest.mark.parametrize("alto", _ALTOS_DE_VENTANA)
@pytest.mark.parametrize("cuantas", [0, 1, 25])
def test_el_historial_se_dibuja_con_ninguna_con_una_y_con_muchas_ordenes(
    cuantas: int, alto: int
) -> None:
    """Es la puerta de entrada a las ordenes. Con cero se cae el taller recien instalado,
    que no podria cargar ni la primera; con muchas se cae despues de meses de andar, cuando
    ya no se puede volver atras. Y la ventana no siempre mide lo mismo."""
    estado = _pantalla_de_ordenes(*[_fila(str(100 + i)) for i in range(cuantas)])

    pantalla = panel.componer(estado, alto)

    assert pantalla.imagen.shape == (alto, panel.ANCHO_LIENZO, 3)
    # Pase lo que pase se puede salir, y se puede cargar una orden nueva.
    assert {"nueva_orden_historial", "volver_ordenes"} <= set(pantalla.botones)


def test_los_botones_que_promete_el_historial_estan_todos() -> None:
    """run_live lee estas claves tal cual. Una que falte deja una tecla sin boton, y en el
    taller el mouse es lo que se usa cuando la tecla no se acuerda."""
    estado = _pantalla_de_ordenes(*[_fila(str(100 + i)) for i in range(11)])

    pantalla = panel.componer(estado, _ALTO_TALLER)

    assert {"abrir_orden", "exportar_orden", "borrar_orden",
            "nueva_orden_historial", "volver_ordenes"} <= set(pantalla.botones)
    # Una fila por orden A LA VISTA y ni una mas: medido, a 672 px entran 8 filas de 56 px
    # y sobran 8. La novena esta en la pagina siguiente, y un boton puesto sobre una fila
    # que no se dibujo seria un clic a ciegas.
    assert {f"elegir_orden:{indice}" for indice in range(8)} <= set(pantalla.botones)
    assert "elegir_orden:8" not in pantalla.botones


def test_con_mas_ordenes_que_filas_la_elegida_igual_se_dibuja(taller) -> None:
    """Es el mismo agujero que costaba un concepto entero en la lista de conceptos: con la
    marca fuera de la pantalla, la 'e' exporta y la 'b' manda a la papelera una orden que
    el operario no esta viendo."""
    conceptos, galeria = taller
    pan = _estado(conceptos, galeria, modo=panel.MODO_ORDENES)
    pan.historial = [_fila(str(100 + indice)) for indice in range(25)]

    # Se llega como se llega de verdad: con la 'w', que desde la primera fila da la vuelta
    # hasta la ultima. La pagina la mueve run_live y el panel dibuja la que le dejaron.
    _manejar(pan, conceptos, galeria, tecla=ord("w"))

    assert pan.orden_elegida == 24
    pantalla = panel.componer(pan, _ALTO_TALLER)
    assert "elegir_orden:24" in pantalla.botones
    # Y la primera quedo fuera de la vista, que es justamente para lo que se pagina.
    assert "elegir_orden:0" not in pantalla.botones


def test_la_fila_de_una_orden_sin_prueba_dice_que_no_tiene_ninguna() -> None:
    """Con este renglon el operario decide si puede contestar un reclamo con la foto en la
    mano. Si la pantalla le dice que la tiene cuando no la tiene, llega al telefono sin
    nada y recien ahi se entera."""
    estado = _pantalla_de_ordenes(_fila(con_prueba=0, con_nombre_suelto=0))

    textos = _textos(estado)

    assert any("SIN PRUEBA GUARDADA" in texto for texto in textos)
    assert not any("CON PRUEBA" in texto for texto in textos)


def test_una_orden_con_solo_nombres_de_foto_sueltos_no_cuenta_como_prueba() -> None:
    """Medido en el taller: las 14 fotos que nombran las ordenes 123 y 1234 estan todas en
    data/coleccion/_papelera y ninguna viva en la coleccion, asi que ese nombre pelado hoy
    no abre ningun archivo. Contarlo en la columna de la prueba es prometer una foto que no
    esta, que es lo mismo que inventar un diente."""
    estado = _pantalla_de_ordenes(_fila("1234", con_prueba=0, con_nombre_suelto=7))

    textos = _textos(estado)

    assert any("SIN PRUEBA GUARDADA" in texto for texto in textos)
    # Ni "7 CON PRUEBA" ni ningun otro numero pegado a esa palabra: pruebas de verdad hay
    # cero, y los 7 nombres no suman ahi.
    assert not any("CON PRUEBA" in texto for texto in textos)
    # Pero se nombran, con las palabras que dicen lo que son.
    assert any("7 nombres de foto sueltos, no son prueba" in texto for texto in textos)


def test_la_orden_con_pruebas_guardadas_las_cuenta_en_la_fila() -> None:
    """La contracara de las dos de arriba. Si la columna dijera siempre que no hay prueba,
    las dos pasarian igual y no estarian probando nada."""
    estado = _pantalla_de_ordenes(_fila(con_prueba=15, con_nombre_suelto=0))

    textos = _textos(estado)

    assert any("15 CON PRUEBA" in texto for texto in textos)
    assert not any("SIN PRUEBA GUARDADA" in texto for texto in textos)


def test_la_orden_abierta_se_distingue_de_las_cerradas_de_un_vistazo() -> None:
    """El operario entra aca para retomar la que esta andando. Si la fila de la abierta se
    leyera igual que las cerradas, un ENTER de reflejo cae en la que no era; y una cerrada
    no se retoma, asi que el programa le dice que no y el no sabe por que."""
    abierta = _fila("5113546", cerrada="", esta_abierta=True)
    estado = _pantalla_de_ordenes(abierta, _fila("213132"))

    textos = _textos(estado)

    assert "ABIERTA" in textos                                      # al lado del numero
    assert any("SIGUE ABIERTA" in texto for texto in textos)        # donde va el cierre
    assert any("cerrada 16/11/2026 12:11:25" in texto for texto in textos)

    # Y la marca de la izquierda, que es lo unico que se ve de lejos sin leer: circulo
    # lleno verde en la abierta, hueco en la cerrada. El centro del circulo de la primera
    # fila cae en (44, 115) y cada fila baja _ALTO_FILA px.
    imagen = panel.componer(estado, _ALTO_TALLER).imagen
    assert tuple(imagen[115, 44]) == tuple(panel._VERDE_OK)
    assert tuple(imagen[115 + panel._ALTO_FILA, 44]) != tuple(panel._VERDE_OK)


def test_la_fila_de_un_archivo_que_no_se_pudo_leer_lo_dice_con_todas_las_letras() -> None:
    """Esconderla seria esconder justo la que hay que arreglar a mano. Y tampoco se la
    puede mostrar como una orden mas: de un .json que no se pudo abrir no se sabe nada, asi
    que esa fila no ofrece retomar ni exportar y no cuenta dientes ni pruebas."""
    estado = _pantalla_de_ordenes(
        _fila("777", ilegible=True, cliente="", total_dientes=0, revisados=0,
              rechazados=0, abierta="", cerrada="")
    )

    textos = _textos(estado)
    pantalla = panel.componer(estado, _ALTO_TALLER)

    assert any("NO SE PUDO LEER" in texto for texto in textos)
    assert any("no se pudo leer" in texto and "Explorador" in texto for texto in textos)
    assert not any("CON PRUEBA" in texto for texto in textos)
    assert not any("revisados" in texto for texto in textos)
    # Sacarla del medio es lo unico que se puede hacer con ella.
    assert "borrar_orden" in pantalla.botones
    assert "abrir_orden" not in pantalla.botones
    assert "exportar_orden" not in pantalla.botones


def test_el_historial_vacio_no_nombra_teclas_ni_botones_que_no_hacen_nada() -> None:
    """Una tecla escrita que no hace nada es lo mismo que un boton pintado que no se puede
    apretar: la proxima vez el operario no le cree a ninguno de los dos."""
    estado = _pantalla_de_ordenes()

    textos = _textos(estado)
    pantalla = panel.componer(estado, _ALTO_TALLER)

    assert set(pantalla.botones) == {"nueva_orden_historial", "volver_ordenes"}
    assert not any("ENTER retomar" in texto for texto in textos)
    # Y se dice como se carga la primera, que es lo unico que hay para hacer aca.
    assert any("Todavia no hay ninguna orden guardada." in texto for texto in textos)


def test_un_cliente_con_nombre_larguisimo_no_se_sale_de_la_fila() -> None:
    """Sin cortarlo pisa la columna de la prueba --el dato por el que se abre esta
    pantalla-- y sigue hasta el borde del lienzo, donde se corta solo: lo que queda a la
    vista puede ser el nombre de otro cliente, y nada avisa que esta cortado."""
    largo = "Aserradero y Carpinteria Hermanos " + "W" * 60
    estado = _pantalla_de_ordenes(_fila(cliente=largo, con_prueba=15))

    lineas, _etiquetas, _botones = _dibujar(estado)

    assert any(".." in texto for texto, *_resto in lineas)
    for texto, (x, _y), _color, tamano, negrita in lineas:
        ancho, _alto_texto = panel.tipografia.medir(texto, tamano, negrita=negrita)
        assert x + ancho <= panel.ANCHO_LIENZO, f"se sale del lienzo: {texto!r}"


# ---------------------------------------------------------------------------
# La confirmacion de mandar una orden a la papelera
# ---------------------------------------------------------------------------


def test_la_pantalla_de_borrado_nombra_la_orden_y_las_fotos_que_se_lleva() -> None:
    """Los dientes se pueden volver a revisar; la foto del diente de la semana pasada no.
    El que aprieta esto tiene que leer ANTES cuanta prueba se va con la orden, y de que
    orden se trata: el indice de la fila se mueve solo cuando el historial se relee, lo que
    vale es el numero escrito aca."""
    estado = _pantalla_de_ordenes(_fila("5113546", con_prueba=15),
                                  modo=panel.MODO_BORRAR_ORDEN)

    textos = _textos(estado)
    pantalla = panel.componer(estado, _ALTO_TALLER)

    assert any("5113546" in texto for texto in textos)
    assert any("Aserradero del Litoral" in texto for texto in textos)
    assert any("15 fotos de prueba" in texto for texto in textos)
    # Y que no se pierde: va a la papelera y de ahi se saca con el Explorador.
    assert any("papelera" in texto for texto in textos)
    assert set(pantalla.botones) == {"confirmar_borrado_orden", "cancelar_borrado_orden"}


def test_al_borrar_una_orden_sin_pruebas_no_se_prometen_fotos_que_no_estan() -> None:
    """Si esta pantalla contara los nombres sueltos como fotos, el operario creeria que
    esta tirando una prueba que en realidad nunca se guardo --y al contrario, el dia que SI
    haya fotos tiene que saber que esas se van."""
    estado = _pantalla_de_ordenes(_fila("1234", con_prueba=0, con_nombre_suelto=7),
                                  modo=panel.MODO_BORRAR_ORDEN)

    textos = _textos(estado)

    assert any("No tiene ninguna foto de prueba guardada" in texto for texto in textos)
    assert not any("7 fotos de prueba" in texto for texto in textos)
    assert any("solo tienen anotado el nombre" in texto for texto in textos)


def test_borrar_la_orden_abierta_avisa_que_el_programa_queda_sin_ninguna() -> None:
    """Al irse la orden en curso, los dientes que se revisen despues no se anotan en ningun
    lado hasta que se abra otra. Eso hay que saberlo antes de apretar, no despues."""
    abierta = _pantalla_de_ordenes(_fila("5113546", cerrada="", esta_abierta=True),
                                   modo=panel.MODO_BORRAR_ORDEN)
    cerrada = _pantalla_de_ordenes(_fila("213132"), modo=panel.MODO_BORRAR_ORDEN)

    assert _dice(abierta, "ES LA ORDEN ABIERTA")
    # Y con una cerrada no se dice, que seria asustar al operario por nada.
    assert not _dice(cerrada, "ES LA ORDEN ABIERTA")


@pytest.mark.parametrize("alto", _ALTOS_DE_VENTANA)
def test_la_pantalla_de_borrado_se_dibuja_sin_ninguna_orden_elegida(alto: int) -> None:
    """Se puede llegar con el historial vacio detras --la ultima orden se fue y la lista se
    releyo-- y ahi no hay numero que nombrar. Antes de caerse, lo dice."""
    estado = _pantalla_de_ordenes(modo=panel.MODO_BORRAR_ORDEN)

    pantalla = panel.componer(estado, alto)

    assert pantalla.imagen.shape == (alto, panel.ANCHO_LIENZO, 3)
    assert _dice(estado, "No hay ninguna orden elegida.", alto)


# ---------------------------------------------------------------------------
# La orden sale de la lista de conceptos
# ---------------------------------------------------------------------------


def test_la_lista_de_conceptos_ya_no_entra_a_la_orden_de_afilado(taller) -> None:
    """Ordenes y conceptos son hermanas --una dice contra que se compara el diente, la otra
    para quien se esta trabajando-- y la orden colgaba de los conceptos. Ademas la fila de
    botones ya estaba llena: medido, median 1.249 px de los 1.280 y no entraba uno mas."""
    conceptos, galeria = taller
    estado = _estado(conceptos, galeria)

    pantalla = panel.componer(estado, _ALTO_TALLER)
    textos = _textos(estado)

    assert "orden" not in pantalla.botones
    assert not any("ORDEN DE AFILADO" in texto for texto in textos)
    # Y la 'o' sale de la linea de teclas: una tecla que no esta no se nombra.
    teclas = next(texto for texto in textos if texto.startswith("teclas:"))
    assert "o orden" not in teclas
    assert "v ver fotos" in teclas and "b borrar" in teclas


def test_los_textos_del_historial_y_del_borrado_no_llevan_acentos() -> None:
    """La prueba global mira los archivos de src. Esta mira los renglones ya ARMADOS, que
    es lo que de verdad llega a la pantalla: una fecha, un numero y un nombre pegados con
    un f-string y cortados por _acortar. cv2.putText dibuja cada tilde como un '?'."""
    casos = [
        _pantalla_de_ordenes(),
        _pantalla_de_ordenes(_fila("5113546", cerrada="", esta_abierta=True)),
        _pantalla_de_ordenes(_fila("OA/123", archivo="OA_123", con_prueba=3)),
        _pantalla_de_ordenes(_fila("1234", con_nombre_suelto=7)),
        _pantalla_de_ordenes(_fila("777", ilegible=True, cliente="", abierta="")),
        _pantalla_de_ordenes(_fila(cerrada="", abierta="")),
        # La prueba que el archivo nombra y no esta, en sus tres formas, y la fecha que
        # sale del ARCHIVO y no de la orden: textos nuevos, armados con f-strings y
        # cortados por _acortar, que es donde se cuela una tilde sin que nadie la vea.
        _pantalla_de_ordenes(_fila(con_prueba=0, prueba_perdida=3)),
        _pantalla_de_ordenes(_fila(con_prueba=0, prueba_perdida=1)),
        _pantalla_de_ordenes(_fila(con_prueba=160, prueba_perdida=160)),
        _pantalla_de_ordenes(_fila(abierta="", clave_estimada=True)),
        _pantalla_de_ordenes(_fila(abierta="el martes", clave_estimada=True)),
        _pantalla_de_ordenes(modo=panel.MODO_BORRAR_ORDEN),
        _pantalla_de_ordenes(_fila(con_prueba=15), modo=panel.MODO_BORRAR_ORDEN),
        _pantalla_de_ordenes(_fila(con_nombre_suelto=7), modo=panel.MODO_BORRAR_ORDEN),
        _pantalla_de_ordenes(_fila(con_prueba=0, prueba_perdida=3),
                             modo=panel.MODO_BORRAR_ORDEN),
        _pantalla_de_ordenes(_fila(con_prueba=160, prueba_perdida=160),
                             modo=panel.MODO_BORRAR_ORDEN),
        _pantalla_de_ordenes(_fila(cerrada="", esta_abierta=True),
                             modo=panel.MODO_BORRAR_ORDEN),
    ]

    for estado in casos:
        for texto in _textos(estado):
            raros = sorted({letra for letra in texto if ord(letra) > 126})
            assert not raros, f"{texto!r} tiene {raros}"


# ---------------------------------------------------------------------------
# EL ARCHIVO DE LA FILA ELEGIDA, Y NO OTRO
#
# El historial lista ARCHIVOS y cada archivo lleva un numero adentro. Los dos textos no
# son el mismo, y el dia que no coinciden todo lo que esta pantalla hace --exportar,
# retomar, borrar-- cae sobre otra orden. No es un caso de laboratorio: "123 - copia.json"
# es lo que deja Windows al copiar una orden, y es justo lo que la pantalla de borrado le
# sugiere al operario que haga antes de tocar nada.
# ---------------------------------------------------------------------------


def _escribir_orden(carpeta: Path, archivo: str, numero: str, cliente: str,
                    abierta: str = "") -> Path:
    """Un .json de orden escrito a mano, con el nombre de archivo y el numero que se pidan.

    A mano porque es justo lo que el programa NO puede hacer: Ordenes.abrir() nombra el
    archivo con el numero saneado, asi que para tener dos archivos distintos con el mismo
    numero adentro hay que escribirlos como los escribe el Explorador de Windows. La fecha
    se pone a proposito para que el historial los ordene siempre igual: dos ordenes
    guardadas en el mismo segundo tienen la misma clave y el orden entre ellas lo decide el
    nombre del archivo.
    """
    orden = Orden.nueva(numero, 1, 2, cliente=cliente)
    if abierta:
        orden.abierta = abierta
    carpeta.mkdir(parents=True, exist_ok=True)
    ruta = carpeta / f"{archivo}.json"
    ruta.write_text(json.dumps(orden.para_json(), ensure_ascii=False), encoding="utf-8")
    return ruta


def _dos_tocayas(carpeta: Path) -> Ordenes:
    """La orden del cliente y una copia vieja, las dos con el numero 123 adentro."""
    _escribir_orden(carpeta, "123", "123", "EL CLIENTE BUENO", "16/11/2026 09:34:20")
    _escribir_orden(carpeta, "copia_vieja", "123", "UNA COPIA VIEJA", "10/11/2026 08:00:00")
    return Ordenes(str(carpeta))


def test_el_historial_lista_los_dos_archivos_que_llevan_el_mismo_numero(
    taller, tmp_path
) -> None:
    """Primero lo que el operario VE, porque es lo que hace al resto peligroso: dos filas
    que dicen 123 y se diferencian solo por el archivo. Si el historial mostrara una sola
    --la que el numero alcanza-- la otra no existiria para el programa y nadie iria a
    buscarla; mostrando las dos, cada tecla tiene que saber sobre cual cae."""
    conceptos, galeria = taller
    ordenes = _dos_tocayas(tmp_path / "ordenes")

    pan = _historial(conceptos, galeria, ordenes)

    assert [fila.numero for fila in pan.historial] == ["123", "123"]
    assert sorted(fila.archivo for fila in pan.historial) == ["123", "copia_vieja"]
    # Y la fila nombra su archivo cuando no se llama como el numero, que es lo unico que
    # distingue una fila de la otra en pantalla.
    assert _dice(pan, "archivo copia_vieja.json")


def test_exportar_arma_el_zip_de_la_fila_elegida_y_no_del_tocayo(taller, tmp_path) -> None:
    """QUE SE ROMPIA: el zip es lo que se le manda al cliente cuando reclama. Exportando la
    fila de la copia vieja salia el paquete de la orden de al lado --otro trabajo, otros
    dientes, otras fotos-- con el numero del reclamo encima, y nada en la pantalla decia
    que el contenido era de otra. El taller contesta el reclamo con la orden equivocada y
    se entera cuando el cliente la abre."""
    conceptos, galeria = taller
    carpeta = tmp_path / "ordenes"
    ordenes = _dos_tocayas(carpeta)
    pan = _historial(conceptos, galeria, ordenes, "copia_vieja")

    _manejar(pan, conceptos, galeria, tecla=ord("e"), ordenes=ordenes)

    zips = list((carpeta / "exportado").glob("*.zip"))
    assert len(zips) == 1
    with zipfile.ZipFile(zips[0]) as paquete:
        adentro = json.loads(paquete.read("orden.json"))
        informe = paquete.read("informe.txt").decode("utf-8")
    assert adentro["cliente"] == "UNA COPIA VIEJA"
    # Y el informe que viaja adentro es el de la MISMA orden que el .json, no el de la
    # tocaya: es el papel que se lee del otro lado.
    assert "UNA COPIA VIEJA" in informe and "EL CLIENTE BUENO" not in informe
    # El nombre del zip tambien lleva el archivo, para que el que lo busca con el
    # Explorador sepa cual de las dos es.
    assert "copia_vieja" in zips[0].name


def test_borrar_se_lleva_el_archivo_de_la_fila_elegida_y_no_el_del_cliente(
    taller, tmp_path
) -> None:
    """QUE SE ROMPIA: esto es lo peor de todo. Eligiendo la fila de la copia vieja y
    apretando b y s, la que se iba a la papelera era la orden de verdad del cliente y la
    copia quedaba en su lugar. El operario ve que el historial sigue mostrando una orden
    123 y se queda tranquilo: lo que perdio --el trabajo del dia, con sus fotos-- recien
    aparece el dia del reclamo, y hay que ir a buscarlo a _papelera con el Explorador."""
    conceptos, galeria = taller
    carpeta = tmp_path / "ordenes"
    ordenes = _dos_tocayas(carpeta)
    pan = _historial(conceptos, galeria, ordenes, "copia_vieja")

    _manejar(pan, conceptos, galeria, tecla=ord("b"), ordenes=ordenes)
    # La pantalla de confirmacion tiene que dejar ver CUAL de las dos se va. Las dos dicen
    # 123, asi que lo unico que las separa es el cliente: si la confirmacion no lo dijera,
    # el operario estaria apretando "si, borrala" sin saber a cual le dice si.
    assert _dice(pan, "UNA COPIA VIEJA")
    assert not _dice(pan, "EL CLIENTE BUENO")

    _manejar(pan, conceptos, galeria, tecla=ord("s"), ordenes=ordenes)

    assert not (carpeta / "copia_vieja.json").is_file()
    assert list(carpeta.glob("_papelera/*/copia_vieja.json"))
    # La del cliente quedo donde estaba Y con su contenido: que el archivo exista no
    # alcanza, porque un borrado mal direccionado lo deja existiendo y vacio.
    assert (carpeta / "123.json").is_file()
    assert json.loads((carpeta / "123.json").read_text(encoding="utf-8"))["cliente"] == (
        "EL CLIENTE BUENO"
    )
    assert [fila.archivo for fila in pan.historial] == ["123"]


def test_la_copia_que_deja_windows_se_exporta_y_se_borra_por_su_nombre(
    taller, tmp_path
) -> None:
    """EL CASO "123 - copia", que es el que de verdad va a pasar en el taller: es el nombre
    que pone Windows al copiar y lo que la pantalla de borrado sugiere hacer antes de tocar
    nada. QUE SE ROMPIA, y por dos caminos distintos: pasando el numero, las dos teclas
    caian sobre 123.json; y pasando el nombre del archivo por donde va el numero tampoco
    alcanzaba, porque sanear_numero("123 - copia") da "123_-_copia" y eso no es ningun
    archivo. Medido: sanear_numero("123 - copia") == "123_-_copia"."""
    conceptos, galeria = taller
    carpeta = tmp_path / "ordenes"
    _escribir_orden(carpeta, "123", "123", "EL CLIENTE BUENO", "16/11/2026 09:34:20")
    _escribir_orden(carpeta, "123 - copia", "123", "LA COPIA DE WINDOWS",
                    "15/11/2026 09:34:20")
    ordenes = Ordenes(str(carpeta))
    pan = _historial(conceptos, galeria, ordenes, "123 - copia")

    _manejar(pan, conceptos, galeria, tecla=ord("e"), ordenes=ordenes)
    zips = list((carpeta / "exportado").glob("*.zip"))
    assert len(zips) == 1
    with zipfile.ZipFile(zips[0]) as paquete:
        assert json.loads(paquete.read("orden.json"))["cliente"] == "LA COPIA DE WINDOWS"

    _manejar(pan, conceptos, galeria, tecla=ord("b"), ordenes=ordenes)
    _manejar(pan, conceptos, galeria, tecla=ord("s"), ordenes=ordenes)

    assert not (carpeta / "123 - copia.json").is_file()
    assert list(carpeta.glob("_papelera/*/123 - copia.json"))
    assert (carpeta / "123.json").is_file()
    assert json.loads((carpeta / "123.json").read_text(encoding="utf-8"))["cliente"] == (
        "EL CLIENTE BUENO"
    )


def test_la_orden_cuyo_archivo_no_se_llama_como_su_numero_no_se_retoma(
    taller, tmp_path
) -> None:
    """QUE SE ROMPIA: retomar deja la orden abierta, y de ahi en adelante cada diente se
    escribe en el archivo que le toca al NUMERO. O sea que retomando "123 - copia" se leia
    de un archivo y se escribia en otro: el operario se pasaba la tarde anotando dientes
    encima de la orden del cliente, que es la que estaba bien. Perder una tarde de anotar
    es menos que pisar la orden de al lado, asi que no se retoma y se dice por que.

    Se prueban las DOS filas en la misma pantalla: la copia no se retoma y la que se llama
    como su numero se retoma igual que siempre. Sin la segunda mitad, una guardia que no
    retomara NINGUNA orden pasaria esta prueba, y el programa quedaria sin forma de seguir
    una orden a medias --que es lo que se hace todos los dias--."""
    conceptos, galeria = taller
    carpeta = tmp_path / "ordenes"
    _escribir_orden(carpeta, "123", "123", "EL CLIENTE BUENO", "16/11/2026 09:34:20")
    _escribir_orden(carpeta, "123 - copia", "123", "LA COPIA DE WINDOWS",
                    "15/11/2026 09:34:20")
    ordenes = Ordenes(str(carpeta))
    pan = _historial(conceptos, galeria, ordenes, "123 - copia")

    sigue, mensaje = _manejar(pan, conceptos, galeria, tecla=_ENTER, ordenes=ordenes)

    # No se retomo NADA: ni la copia ni, sobre todo, la del cliente.
    assert (sigue, mensaje) == (True, None)
    assert pan.modo == panel.MODO_ORDENES and pan.orden is None
    assert ordenes.abierta is None
    assert json.loads((carpeta / "123.json").read_text(encoding="utf-8"))["cliente"] == (
        "EL CLIENTE BUENO"
    )
    # Y el aviso nombra LOS DOS archivos --el que se eligio y el que se escribiria-- y
    # manda a las dos teclas que SI andan sobre esa fila. Un "no se pudo" pelado deja al
    # operario creyendo que el disco esta roto.
    aviso = pan.aviso or ""
    assert "123 - copia" in aviso and "123.json" in aviso
    assert "'e'" in aviso and "'b'" in aviso

    # LA OTRA FILA, la que se llama como su numero: esa se retoma como siempre. Son las 4
    # ordenes de verdad del taller, asi que es el camino de todos los dias.
    pan = _historial(conceptos, galeria, ordenes, "123")

    _manejar(pan, conceptos, galeria, tecla=_ENTER, ordenes=ordenes)

    assert pan.modo == panel.MODO_ORDEN and pan.aviso is None
    assert ordenes.abierta is not None
    assert ordenes.abierta.cliente == "EL CLIENTE BUENO"


# ---------------------------------------------------------------------------
# UNA SOLA CUENTA DE FILAS POR PAGINA
#
# El alto del lienzo lo decide la camara, asi que CAMBIA de PC en PC y hasta mientras el
# programa anda --con una orden abierta aparece la barra de dientes y el recuadro de la
# imagen se achica--. Durante un tiempo hubo dos cuentas: el que dibujaba medidaba el
# lienzo y el que leia las teclas usaba un 8 fijo. Lo que se prueba aca es que haya UNA.
# ---------------------------------------------------------------------------

# MEDIDO AQUI, componiendo el historial de verdad y contando los botones "elegir_orden:"
# que quedaron (30 ordenes cargadas, para que sobren filas en todos los casos):
#
#   420 px -> 3 filas      600 px -> 6 filas      664 px -> 8 filas
#   492 px -> 4 filas      656 px -> 7 filas      672 px -> 8 filas
#   520 px -> 5 filas      663 px -> 7 filas      720 px -> 8 filas
#
# 420 es el piso: componer() no dibuja mas chico que eso --medido, pidiendole 300 px
# devuelve un lienzo de 420-- y filas_de_ordenes() corta en el mismo numero, asi que por
# abajo las dos cuentas siguen de acuerdo.
#
# De donde salen esos altos NO se midio aca: lo dice la medicion que quedo escrita en
# panel.filas_de_ordenes(), que compuso la pantalla del taller con tres camaras --640x480
# da un lienzo de 672 px, 1280x720 y 1920x1080 dan 656--. O sea que 672 es esta PC y 656
# es cualquier otra, que es el caso que rompia. 663 y 664 estan para clavar el escalon: un
# cambio que corra la cuenta un pixel se ve aca.
_FILAS_POR_ALTO = {420: 3, 492: 4, 520: 5, 600: 6,
                   656: 7, 663: 7, 664: 8, 672: 8, 720: 8}


@pytest.mark.parametrize("alto, filas", sorted(_FILAS_POR_ALTO.items()))
def test_las_filas_que_se_dibujan_son_las_que_dice_la_unica_cuenta(
    alto: int, filas: int
) -> None:
    """QUE SE ROMPIA: con dos cuentas distintas, el programa paginaba de ocho en ocho
    mientras dibujaba siete filas. La octava tecla elegia una orden que no estaba en
    pantalla, y la 'b' siguiente la mandaba a la papelera a ciegas."""
    estado = _pantalla_de_ordenes(*[_fila(str(100 + indice)) for indice in range(30)])

    pantalla = panel.componer(estado, alto)

    dibujadas = {clave for clave in pantalla.botones if clave.startswith("elegir_orden:")}
    assert len(dibujadas) == filas
    # La funcion publica dice lo mismo que el dibujo, que es lo que le permite a run_live
    # saber cuantas filas van a entrar ANTES de dibujar el primer cuadro.
    assert panel.filas_de_ordenes(alto) == filas
    # Y el que dibuja deja anotado cuanto le entro de verdad: es de aca de donde salen los
    # digitos y las paginas.
    assert estado.filas_por_pagina == filas


@pytest.mark.parametrize("alto, filas", sorted(_FILAS_POR_ALTO.items()))
def test_ningun_digito_elige_una_fila_que_no_se_dibujo(
    taller, tmp_path, alto: int, filas: int
) -> None:
    """QUE SE ROMPIA: la tecla 8 con siete filas en pantalla dejaba elegida la fila que no
    se veia, sin que nada cambiara a la vista. La 'b' y la 's' siguientes --las teclas de
    todos los dias-- se llevaban a la papelera una orden que el operario nunca vio."""
    conceptos, galeria = taller
    ordenes = _con_ordenes(tmp_path, *[str(numero) for numero in range(101, 121)])
    pan = _historial(conceptos, galeria, ordenes)
    pan.orden_elegida, pan.pagina_ordenes = 0, 0
    # Se dibuja primero, igual que en el programa: el que dibuja es el que mide el lienzo y
    # deja anotadas las filas que entraron.
    pantalla = panel.componer(pan, alto)
    dibujadas = {int(clave.split(":")[1]) for clave in pantalla.botones
                 if clave.startswith("elegir_orden:")}
    assert len(dibujadas) == filas

    for digito in range(1, 10):
        pan.orden_elegida = 0
        _manejar(pan, conceptos, galeria, tecla=ord(str(digito)), ordenes=ordenes)
        assert pan.orden_elegida in dibujadas, (
            f"con el lienzo en {alto} px se dibujaron {filas} filas y la tecla {digito} "
            f"dejo elegida la fila {pan.orden_elegida}"
        )

    # Y el ultimo digito que SI tiene fila la elige. Sin esto la prueba pasaria igual con
    # todos los digitos muertos, que seria el otro error y tambien dejaria al operario sin
    # poder elegir una orden desde el teclado.
    pan.orden_elegida = 0
    _manejar(pan, conceptos, galeria, tecla=ord(str(filas)), ordenes=ordenes)
    assert pan.orden_elegida == filas - 1


@pytest.mark.parametrize("alto, filas", sorted(_FILAS_POR_ALTO.items()))
def test_la_fila_elegida_se_dibuja_con_cualquier_lienzo(
    taller, tmp_path, alto: int, filas: int
) -> None:
    """Se camina el historial entero con la 's', que es como se llega a una orden que no
    tiene digito en pantalla, y en cada paso se mira que la fila marcada este dibujada.

    QUE SE ROMPIA: la pagina se calculaba con 8 y se dibujaba con 7 (o con 5, segun la
    camara), asi que a partir de la fila 8 la marca caia en una pagina y el dibujo en otra.
    La marca desaparecia de la pantalla sin aviso y la 'e' y la 'b' seguian trabajando
    sobre ella."""
    conceptos, galeria = taller
    ordenes = _con_ordenes(tmp_path, *[str(numero) for numero in range(101, 121)])
    pan = _historial(conceptos, galeria, ordenes)
    pan.orden_elegida, pan.pagina_ordenes = 0, 0
    panel.componer(pan, alto)

    for paso in range(len(pan.historial)):
        pantalla = panel.componer(pan, alto)
        assert f"elegir_orden:{pan.orden_elegida}" in pantalla.botones, (
            f"con el lienzo en {alto} px ({filas} filas), al paso {paso} la fila "
            f"{pan.orden_elegida} quedo elegida y no dibujada"
        )
        _manejar(pan, conceptos, galeria, tecla=ord("s"), ordenes=ordenes)


def test_el_espacio_cae_en_la_primera_fila_de_la_pagina_que_se_dibujo(
    taller, tmp_path
) -> None:
    """El espacio pasa de pagina y deja elegida la primera fila de la nueva. QUE SE
    ROMPIA: saltaba de a ocho mientras la pagina se dibujaba de a cinco, asi que caia en el
    medio de la pagina --o afuera-- y el operario perdia de vista cual estaba marcada.

    Se usa un lienzo de 520 px, que es el de una ventana baja: ahi entran 5 filas y la
    diferencia con el 8 fijo se ve de una."""
    conceptos, galeria = taller
    ordenes = _con_ordenes(tmp_path, *[str(numero) for numero in range(101, 112)])
    pan = _historial(conceptos, galeria, ordenes)
    pan.orden_elegida, pan.pagina_ordenes = 0, 0
    panel.componer(pan, 520)
    assert pan.filas_por_pagina == 5

    _manejar(pan, conceptos, galeria, tecla=ord(" "), ordenes=ordenes)

    pantalla = panel.componer(pan, 520)
    dibujadas = sorted(int(clave.split(":")[1]) for clave in pantalla.botones
                       if clave.startswith("elegir_orden:"))
    assert dibujadas == [5, 6, 7, 8, 9]
    assert pan.orden_elegida == dibujadas[0]


def test_el_historial_se_abre_en_la_pagina_donde_de_verdad_esta_la_orden_en_curso(
    taller, tmp_path
) -> None:
    """El que aprieta 'o' con una orden en curso cae parado en ella. QUE SE ROMPIA: la
    pagina se buscaba con el 8 fijo, asi que con una ventana donde entran 5 filas la orden
    en curso quedaba elegida en una pagina que no se estaba dibujando. El operario veia el
    historial empezado desde arriba, con la marca en ninguna parte, y el ENTER de reflejo
    --que es lo que se aprieta para volver a la orden-- retomaba la que no era."""
    conceptos, galeria = taller
    carpeta = tmp_path / "ordenes"
    for indice in range(11):
        # Fechas distintas y a mano: once ordenes abiertas en el mismo segundo tienen la
        # misma clave y el orden entre ellas lo decide el nombre del archivo.
        _escribir_orden(carpeta, f"90{indice:02d}", f"90{indice:02d}",
                        f"Cliente {indice}", f"{11 - indice:02d}/11/2026 09:00:00")
    ordenes = Ordenes(str(carpeta))
    # La septima de la lista queda abierta: no entra en una pagina de 5 ni en una de 8 en
    # el mismo lugar, que es justo donde las dos cuentas se separan.
    assert ordenes.retomar("9006") is not None
    # El panel se arma como lo arma run_live: con las filas que midio ANTES de dibujar.
    pan = _estado(conceptos, galeria, filas_por_pagina=panel.filas_de_ordenes(520))

    run_live._abrir_historial(pan, ordenes)

    assert pan.historial[pan.orden_elegida].archivo == "9006"
    pantalla = panel.componer(pan, 520)
    assert f"elegir_orden:{pan.orden_elegida}" in pantalla.botones


# ---------------------------------------------------------------------------
# LA PRUEBA QUE EL ARCHIVO NOMBRA Y NO ESTA EN EL DISCO
#
# Son TRES cosas distintas y hasta ahora dos de ellas se leian igual:
#   - la orden que nunca tuvo prueba: no hay nada que buscar;
#   - la que tiene la foto guardada: el reclamo se contesta con ella en la mano;
#   - la que el .json NOMBRA y el disco ya no tiene: hay que ir a buscarla a otra parte
#     --la papelera de la coleccion, la PC de la que se copio la carpeta, el respaldo--
#     antes de contestar nada.
# La primera y la tercera decian las dos "SIN PRUEBA GUARDADA" en ambar.
# ---------------------------------------------------------------------------


def test_la_fila_dice_las_pruebas_que_el_archivo_nombra_y_no_estan() -> None:
    """QUE SE ROMPIA: esta fila se leia IGUAL que la de una orden que nunca tuvo prueba, y
    el zip del reclamo salia sin las fotos sin que nada avisara. El operario manda el
    paquete creyendo que no habia nada que mandar, cuando en realidad habia tres fotos que
    alguien movio y que todavia pueden estar en la papelera de la coleccion."""
    estado = _pantalla_de_ordenes(_fila(con_prueba=0, prueba_perdida=3))

    textos = _textos(estado)

    assert any("3 PRUEBAS QUE NO ESTAN" in texto for texto in textos)
    # Y NO se dice lo mismo que cuando nunca hubo prueba: son dos cosas distintas y lo que
    # el operario hace con cada una es distinto.
    assert not any("SIN PRUEBA GUARDADA" in texto for texto in textos)


def test_la_prueba_perdida_se_dice_en_rojo_y_la_que_nunca_hubo_en_ambar() -> None:
    """El color es lo unico que se lee de lejos, con la sierra en la mano. El rojo dice
    "aca falta algo que estaba"; el ambar, "aca no hay nada que buscar"."""
    perdida = _fila(con_prueba=0, prueba_perdida=3)
    nunca = _fila(con_prueba=0, prueba_perdida=0)

    assert panel._texto_prueba(perdida)[1] == panel._ROJO_NO
    assert panel._texto_prueba(nunca)[1] == panel._AMBAR


def test_una_sola_prueba_perdida_se_dice_en_singular() -> None:
    """"1 PRUEBAS QUE NO ESTAN" lo lee un operario, no un programador. El renglon tiene que
    poder leerse en voz alta por telefono mientras el cliente espera."""
    estado = _pantalla_de_ordenes(_fila(con_prueba=0, prueba_perdida=1))

    assert _dice(estado, "1 PRUEBA QUE NO ESTA")
    assert not _dice(estado, "1 PRUEBAS QUE NO ESTAN")


def test_con_media_prueba_en_el_disco_la_fila_dice_las_dos_cosas() -> None:
    """QUE SE ROMPIA: una fila que dijera solo "160 CON PRUEBA" con 160 fotos faltando esta
    contando como prueba lo que no esta, que es lo mismo que inventar un diente. El que lee
    esa fila arma el reclamo creyendo que tiene la sierra entera documentada."""
    estado = _pantalla_de_ordenes(_fila(con_prueba=160, prueba_perdida=160))

    textos = _textos(estado)

    assert any("160 CON PRUEBA, 160 NO ESTAN" in texto for texto in textos)
    # En ambar y no en verde: media prueba no es prueba completa.
    assert panel._texto_prueba(_fila(con_prueba=160, prueba_perdida=160))[1] == panel._AMBAR


def test_la_pantalla_de_borrado_declara_la_prueba_que_ya_no_esta() -> None:
    """QUE SE ROMPIA: la confirmacion decia "No tiene ninguna foto de prueba guardada",
    que es verdad y enganosa al mismo tiempo: el archivo nombra fotos, y el que las vaya a
    buscar manana tiene que saber que existieron. Borrando sin saberlo se va tambien el
    unico papel que dice que esas fotos se sacaron alguna vez."""
    perdida = _pantalla_de_ordenes(_fila("1234", con_prueba=0, prueba_perdida=3),
                                   modo=panel.MODO_BORRAR_ORDEN)
    nunca = _pantalla_de_ordenes(_fila("1234", con_prueba=0, prueba_perdida=0),
                                 modo=panel.MODO_BORRAR_ORDEN)

    assert _dice(perdida, "nombra 3")
    assert _dice(perdida, "ya no estan")
    assert not _dice(perdida, "No tiene ninguna foto de prueba guardada.")
    # Y la que nunca tuvo sigue diciendo lo de siempre, que es lo correcto para ella.
    assert _dice(nunca, "No tiene ninguna foto de prueba guardada.")


def test_la_pantalla_de_borrado_no_calla_la_prueba_perdida_cuando_hay_de_las_dos() -> None:
    """Decir solo "se lleva las 160 fotos que tiene" deja creyendo que la prueba esta
    completa justo cuando la mitad ya no esta. Son dos cosas y se dicen las dos: lo que se
    lleva la papelera y lo que ya no estaba antes de apretar nada."""
    estado = _pantalla_de_ordenes(_fila(con_prueba=160, prueba_perdida=160),
                                  modo=panel.MODO_BORRAR_ORDEN)

    textos = _textos(estado)

    assert any("160 fotos de prueba" in texto for texto in textos)
    assert any("160 que el archivo nombra" in texto and "ya no estan" in texto
               for texto in textos)


def test_la_prueba_perdida_sale_del_DISCO_y_no_de_lo_que_dice_el_archivo(tmp_path) -> None:
    """La de arriba arma la fila a mano. Esta la arma leyendo ordenes de verdad del disco,
    que es de donde sale en el taller: se guardan tres pruebas, se borra una foto como la
    borraria alguien limpiando la carpeta, y la fila tiene que darse cuenta SOLA.

    Es la regla del taller: no se muestra como prueba algo que no es prueba. Si la cuenta
    saliera de lo que el .json dice --que es lo facil-- las 14 fotos que nombran las
    ordenes 123 y 1234 contarian como prueba, y medido estan todas en la papelera de la
    coleccion y ninguna viva."""
    ordenes = Ordenes(str(tmp_path / "ordenes"))
    ordenes.abrir(Orden.nueva("1234", 1, 4, cliente="Aserradero"))
    foto = np.full((60, 80, 3), 128, np.uint8)
    for diente in (1, 2, 3):
        ruta = ordenes.guardar_prueba("1234", 1, diente, foto)
        assert ruta
        ordenes.abierta.anotar(estado="rechazada", aprobado=False,
                               origen="lo anoto el operario", prueba=ruta)
    assert ordenes.guardar() is not None

    con_las_tres = ordenes.historial()[0]
    assert (con_las_tres.con_prueba, con_las_tres.prueba_perdida) == (3, 0)
    assert _dice(_pantalla_de_ordenes(con_las_tres), "3 CON PRUEBA")

    # Alguien limpia la carpeta y se lleva una foto. El .json no cambia.
    deposito = ordenes.carpeta_pruebas("1234")
    sorted(deposito.iterdir())[0].unlink()

    falta_una = ordenes.historial()[0]
    assert (falta_una.con_prueba, falta_una.prueba_perdida) == (2, 1)
    assert _dice(_pantalla_de_ordenes(falta_una), "2 CON PRUEBA, 1 NO ESTAN")

    # Y se lleva las otras dos.
    for sobrante in sorted(deposito.iterdir()):
        sobrante.unlink()

    sin_ninguna = ordenes.historial()[0]
    assert (sin_ninguna.con_prueba, sin_ninguna.prueba_perdida) == (0, 3)
    pantalla = _pantalla_de_ordenes(sin_ninguna)
    assert _dice(pantalla, "3 PRUEBAS QUE NO ESTAN")
    # Lo que NO puede decir es lo mismo que una orden que nunca tuvo nada: aca hay tres
    # fotos que alguien tiene que ir a buscar.
    assert not _dice(pantalla, "SIN PRUEBA GUARDADA")


# ---------------------------------------------------------------------------
# LO QUE LA PANTALLA DE BORRADO PROMETE ES LO QUE EL MOVIMIENTO SE LLEVA
#
# Las de arriba arman la fila a mano y miran el renglon. Estas arman la orden EN EL DISCO,
# aprietan las teclas de verdad y despues van a contar los .jpg que llegaron a la papelera:
# lo que se compara es la promesa contra el movimiento, que son justo las dos cosas que
# hasta ahora podian no coincidir --la fila se contaba contra la carpeta del NUMERO y el
# borrado movia la carpeta del ARCHIVO--.
#
# MEDIDO AQUI, con una orden de 3 pruebas guardadas con guardar_prueba() en un taller de
# mentira (tmp_path), apretando 'b' y 's' en el manejador de teclas de verdad:
#
#   el .json se llama como su numero    -> promete 3 y se mueven 3 .jpg
#   renombrado a mano a "<numero> viejo"-> promete 0 y se mueven 0, y las 3 fotos siguen
#                                          en data/pruebas/<numero>
#   el .json ilegible con su carpeta    -> no cuenta ninguna, dice que la carpeta se va
#                                          con la orden, y se mueven las 2 que hay
#
# Esos tres son los que miden las dos pruebas de abajo. HAY UN CUARTO CASO, medido aparte en
# una copia y no aca, en el que los dos numeros NO son iguales: una orden con 3 pruebas y 2
# fotos huerfanas al lado --las que ningun diente nombra, de un guardado que fallo-- promete
# 3 y mueve 5, porque lo que se mueve es la carpeta entera. O sea que la cuenta nunca promete
# DE MAS, que es el lado que importa: una foto que ningun diente nombra no se cuenta como
# prueba --contarla seria mostrar como prueba algo que no es prueba-- y por eso la igualdad
# se mide en los casos sin huerfanas.
#
# COMPROBADO QUE FALLAN SIN EL ARREGLO. Se volvio el codigo a lo de antes, de a una cosa por
# vez, y se corrieron las dos pruebas de esta seccion:
#
#   la regla vieja --_es_prueba_de() resolviendo la ruta del .json contra data y aceptando
#   cualquier cosa--      la pantalla prometia 3 y el movimiento se llevaba 0:
#                         "assert promete == 0" fallando con "assert 3 == 0"
#   el renglon sin la carpeta ("...que tiene guardadas." sin el pruebas/<archivo>)
#                         falla el assert que pide el nombre de la carpeta
#   una pantalla que no promete nunca ninguna foto
#                         falla la mitad de arriba: "assert promete == 3" con "assert 0 == 3"
#   el renglon viejo del archivo ilegible ("No tiene ninguna foto de prueba guardada.")
#                         falla el assert de "No se pudo contar su prueba"
# ---------------------------------------------------------------------------


def _con_pruebas(carpeta: Path, numero: str, cuantas: int = 3,
                 cliente: str = "Aserradero del Litoral") -> Ordenes:
    """Una orden guardada con `cuantas` fotos de prueba de verdad en su deposito.

    Pasa por guardar_prueba() y anotar(), no escribe los .jpg a mano: lo que estas pruebas
    miran es que la ruta que queda adentro del .json y el archivo que queda en el disco sean
    el mismo, y escribirlos aparte seria medir contra una ruta inventada por la prueba.
    """
    ordenes = Ordenes(str(carpeta))
    ordenes.abrir(Orden.nueva(numero, 1, 10, cliente=cliente))
    foto = np.full((60, 80, 3), 128, np.uint8)
    for diente in range(1, cuantas + 1):
        ruta = ordenes.guardar_prueba(numero, 1, diente, foto)
        assert ruta, "no se pudo guardar la foto de prueba"
        ordenes.abierta.anotar(estado="rechazada", aprobado=False,
                               origen="lo anoto el operario", prueba=ruta)
    assert ordenes.guardar() is not None
    return ordenes


def _fotos_que_promete(estado: panel.EstadoPanel) -> int:
    """Cuantas fotos de prueba dice la pantalla de borrado que se va a llevar.

    Se LEE de los renglones que la pantalla arma, no se le pregunta al Resumen: lo que hay
    que comparar contra el movimiento es lo que el operario lee antes de apretar. Con la
    orden ilegible esta funcion revienta a proposito --esa pantalla no cuenta ninguna
    prueba, dice que no se pudo contar-- y ese caso tiene su propia prueba.
    """
    for texto in _textos(estado):
        cuantas = re.search(r"Se lleva las (\d+) fotos de prueba", texto)
        if cuantas:
            return int(cuantas.group(1))
        if ("No se lleva ninguna foto" in texto
                or "No tiene ninguna foto de prueba" in texto):
            return 0
    raise AssertionError(
        f"la pantalla de borrado no dice que prueba se lleva: {_textos(estado)}"
    )


def _borrar_con_las_teclas(taller, ordenes: Ordenes,
                           buscar: str) -> tuple[int, int, list[str]]:
    """Aprieta 'b' y 's' sobre esa fila. Devuelve (lo que prometio, los .jpg que se
    movieron, los renglones de la confirmacion).

    Las teclas pasan por el manejador de verdad de run_live y los .jpg se cuentan en el
    disco, adentro de la papelera: es el unico par de numeros que puede desmentir a la
    pantalla.
    """
    conceptos, galeria = taller
    pan = _historial(conceptos, galeria, ordenes, buscar)

    _manejar(pan, conceptos, galeria, tecla=ord("b"), ordenes=ordenes)

    assert pan.modo == panel.MODO_BORRAR_ORDEN
    renglones = _textos(pan)
    promete = _fotos_que_promete(pan)

    _manejar(pan, conceptos, galeria, tecla=ord("s"), ordenes=ordenes)

    papelera = ordenes.carpeta / "_papelera"
    movidas = len(list(papelera.rglob("*.jpg"))) if papelera.is_dir() else 0
    return promete, movidas, renglones


def test_la_pantalla_de_borrado_promete_las_fotos_que_el_movimiento_se_lleva(
    taller, tmp_path: Path
) -> None:
    """QUE SE ROMPIA EN EL TALLER: la pantalla prometia llevarse las 3 fotos de prueba de la
    orden y el movimiento se llevaba CERO. Pasaba con cualquier .json renombrado o copiado a
    mano --"1234 viejo.json", que es lo que queda despues de tocar la carpeta con el
    Explorador-- porque la fila contaba las fotos de la carpeta del NUMERO y el borrado
    movia la carpeta del ARCHIVO. El que contesta un reclamo lee "se lleva las 3 fotos de
    prueba", aprieta, y despues va a la papelera a rescatarlas y no hay ninguna: las 3
    siguen en data/pruebas/1234, donde ya nadie las va a buscar porque la pantalla dijo que
    se habian ido con la orden.

    SE MIDEN LOS DOS CASOS EN LA MISMA PRUEBA. Con uno solo no probaria nada: una pantalla
    que no prometiera nunca ninguna foto pasaria la mitad de abajo, y el operario se
    quedaria sin saber cuanta prueba se esta llevando el dia que de verdad se lleva.
    """
    # EL CASO DE TODOS LOS DIAS: el .json se llama como su numero, como las 4 ordenes de
    # verdad del taller. Promete 3 y se lleva 3.
    carpeta = tmp_path / "ordenes"
    ordenes = _con_pruebas(carpeta, "1234")
    deposito = ordenes.carpeta_pruebas("1234")
    assert len(list(deposito.glob("*.jpg"))) == 3

    promete, movidas, renglones = _borrar_con_las_teclas(taller, ordenes, "1234")

    assert promete == 3
    assert movidas == promete, (
        f"la pantalla prometio {promete} fotos y se movieron {movidas}"
    )
    # Y SE NOMBRA LA CARPETA, que es lo que hace la promesa comprobable antes de apretar:
    # con el nombre escrito el operario puede ir a mirarla con el Explorador.
    assert any("pruebas/1234" in texto for texto in renglones), renglones
    # La carpeta se fue de verdad: si siguiera ahi, "se lleva" seria falso al otro lado.
    assert not deposito.is_dir()

    # EL CASO DEL ARCHIVO TOCADO A MANO. Las fotos viven en data/pruebas/5000 y el .json se
    # llama "5000 viejo", asi que esas fotos no son la prueba de ESTE archivo: la pantalla
    # no promete ninguna, y el movimiento no se lleva ninguna.
    otra = tmp_path / "tocada_a_mano"
    _con_pruebas(otra, "5000")
    (otra / "5000.json").rename(otra / "5000 viejo.json")
    renombrada = Ordenes(str(otra))
    quedaron = sorted((otra.parent / "pruebas" / "5000").glob("*.jpg"))
    assert len(quedaron) == 3

    promete, movidas, renglones = _borrar_con_las_teclas(taller, renombrada, "5000 viejo")

    assert promete == 0
    assert movidas == promete, (
        f"la pantalla prometio {promete} fotos y se movieron {movidas}"
    )
    # Se nombra la carpeta DE ESTE ARCHIVO --la que el borrado hubiera movido-- y no la del
    # numero: nombrar pruebas/5000 seria nombrar una carpeta que el movimiento no toca.
    assert any("pruebas/5000 viejo" in texto for texto in renglones), renglones
    # Y las 3 fotos siguen donde estaban. No se prometieron y no se movieron: son dato del
    # taller y el que las necesite las va a encontrar ahi.
    assert sorted((otra.parent / "pruebas" / "5000").glob("*.jpg")) == quedaron


def test_la_pantalla_de_un_archivo_ilegible_no_dice_que_no_tiene_prueba(
    taller, tmp_path: Path
) -> None:
    """QUE SE ROMPIA: de un .json que no se pudo abrir la pantalla decia "No tiene ninguna
    foto de prueba guardada", que es afirmar lo que nadie pudo contar --y encima era falso:
    medido, con dos fotos en data/pruebas/777 el borrado se las llevo igual, porque para
    mover la carpeta no hace falta leer el .json--. El operario aprieta creyendo que no se
    va ninguna prueba y la prueba se va.

    Ahora dice las dos cosas que SI se saben: que no se pudo contar, y que si hay carpeta se
    va con la orden. Las dos se comprueban abajo: el movimiento se lleva las 2 fotos.
    """
    conceptos, galeria = taller
    carpeta = tmp_path / "ordenes"
    carpeta.mkdir(parents=True)
    # Un .json que esta y no se puede leer, que es lo que queda de un corte de luz a mitad
    # de un guardado.
    (carpeta / "777.json").write_text("{esto no es un json", encoding="utf-8")
    _foto(tmp_path / "pruebas" / "777", "s01_d001.jpg")
    _foto(tmp_path / "pruebas" / "777", "s01_d002.jpg")
    ordenes = Ordenes(str(carpeta))
    assert ordenes.historial()[0].ilegible

    pan = _historial(conceptos, galeria, ordenes, "777")
    _manejar(pan, conceptos, galeria, tecla=ord("b"), ordenes=ordenes)
    renglones = _textos(pan)

    assert any("No se pudo contar su prueba" in texto for texto in renglones), renglones
    assert any("pruebas/777" in texto for texto in renglones), renglones
    # Lo que ya no se afirma: que no tiene prueba. Es lo que nadie midio.
    assert not any("No tiene ninguna foto de prueba" in texto for texto in renglones)

    _manejar(pan, conceptos, galeria, tecla=ord("s"), ordenes=ordenes)

    # Y lo que dijo es cierto: la carpeta se fue con la orden.
    movidas = sorted(p.name for p in (carpeta / "_papelera").rglob("*.jpg"))
    assert movidas == ["s01_d001.jpg", "s01_d002.jpg"]
    assert not (tmp_path / "pruebas" / "777").is_dir()


# ---------------------------------------------------------------------------
# LA ORDEN Y SU COPIA DE WINDOWS NO PUEDEN LEERSE IGUAL
#
# "1234 - copia.json" es lo que deja el Explorador al copiar una orden, y el .json que
# queda adentro es el MISMO TEXTO: mismo numero, mismo cliente, misma fecha. O sea que todo
# lo que la pantalla de confirmacion sabe decir de una lo dice igual de la otra, menos una
# cosa: como se llama el archivo. Y el borrado va por el archivo.
#
# COMPROBADO QUE FALLAN SIN EL ARREGLO: con hay_que_nombrar_el_archivo() devolviendo False
# --que es como estaba antes, el archivo no se nombraba en ninguna parte-- las dos
# confirmaciones salen con la MISMA lista de renglones y los dos borrados devuelven el mismo
# cartel palabra por palabra, "ORDEN 1234 A LA PAPELERA: <el mismo sello>".
# ---------------------------------------------------------------------------


def _la_orden_y_su_copia(carpeta: Path, archivo_copia: str = "1234 - copia") -> Ordenes:
    """La orden 1234 y la copia que deja Windows, con el .json identico palabra por palabra.

    Identico a proposito: copiar un archivo con el Explorador no le cambia nada adentro, asi
    que las dos filas traen el mismo numero, el mismo cliente y la misma fecha. Es lo que
    hace que las dos pantallas de confirmacion salgan iguales si no se nombra el archivo.
    """
    _escribir_orden(carpeta, "1234", "1234", "Aserradero del Litoral",
                    "16/11/2026 09:34:20")
    shutil.copy2(carpeta / "1234.json", carpeta / f"{archivo_copia}.json")
    assert (carpeta / "1234.json").read_text(encoding="utf-8") == (
        (carpeta / f"{archivo_copia}.json").read_text(encoding="utf-8")
    )
    return Ordenes(str(carpeta))


def test_las_dos_confirmaciones_de_la_misma_orden_no_salen_identicas(tmp_path: Path) -> None:
    """QUE SE ROMPIA: con la orden 1234 y su copia de Windows en el historial, las dos
    pantallas de confirmacion salian PALABRA POR PALABRA iguales --mismo numero, mismo
    cliente, misma fecha, mismos dientes-- y ninguna de las dos nombraba el archivo. El
    borrado va por el nombre del archivo, asi que el operario estaba apretando "si, borrala"
    sin tener con que saber a cual de las dos le estaba diciendo si: una es el trabajo del
    dia y la otra una copia que no importa, y por pantalla eran la misma cosa.

    Las dos mitades hacen falta. Si la fila que SI se llama como su numero tambien nombrara
    el archivo, serian dos pantallas distintas igual --la prueba pasaria-- pero el renglon
    de mas estaria en las 4 ordenes de verdad del taller, que son justamente las que no lo
    necesitan: un renglon que repite el numero con otras palabras es un renglon que el
    operario deja de leer.
    """
    ordenes = _la_orden_y_su_copia(tmp_path / "ordenes")
    filas = ordenes.historial()
    assert sorted(fila.archivo for fila in filas) == ["1234", "1234 - copia"]
    assert [fila.numero for fila in filas] == ["1234", "1234"]
    cual = {fila.archivo: indice for indice, fila in enumerate(filas)}

    de_la_orden = _pantalla_de_ordenes(*filas, elegida=cual["1234"],
                                       modo=panel.MODO_BORRAR_ORDEN)
    de_la_copia = _pantalla_de_ordenes(*filas, elegida=cual["1234 - copia"],
                                       modo=panel.MODO_BORRAR_ORDEN)

    assert _textos(de_la_orden) != _textos(de_la_copia), (
        "las dos confirmaciones se leen igual y el borrado va por el archivo"
    )
    # LA COPIA NOMBRA SU ARCHIVO, y dice por que se lo esta nombrando.
    assert _dice(de_la_copia, "1234 - copia.json")
    assert _dice(de_la_copia, "no se llama como su numero")
    # LA QUE SE LLAMA COMO SU NUMERO NO LO NOMBRA: es el caso de las 4 ordenes del taller.
    assert not _dice(de_la_orden, "El archivo que se va")
    # Las dos siguen diciendo de que orden se trata, que es lo primero que se lee.
    for pantalla in (de_la_orden, de_la_copia):
        assert _dice(pantalla, "Sacar la orden 1234 del historial")
        assert _dice(pantalla, "Aserradero del Litoral")


def test_el_cartel_del_borrado_nombra_la_misma_orden_que_nombro_la_pantalla(
    taller, tmp_path: Path
) -> None:
    """El cartel es lo unico que queda despues de apretar: la pantalla ya se fue y el
    historial tiene una fila menos.

    QUE SE ROMPIA: los dos borrados devolvian el mismo cartel, "ORDEN 1234 A LA PAPELERA",
    asi que despues de apretar 's' el operario no tenia con que saber cual de las dos filas
    se habia ido. Y es el momento en que mas importa: si se fue la que no era, hay que ir a
    buscarla a _papelera con el Explorador ANTES de que se borre otra, porque las dos
    carpetas de papelera se llaman con la fecha y la hora y adentro el .json se llama igual.

    Los dos carteles se sacan de dos borrados de verdad, cada uno en su carpeta: el cartel
    sale del manejador de teclas, no de una funcion de texto aparte.
    """
    conceptos, galeria = taller
    carteles: dict[str, str] = {}
    for indice, cual in enumerate(("1234", "1234 - copia")):
        carpeta = tmp_path / f"taller_{indice}" / "ordenes"
        ordenes = _la_orden_y_su_copia(carpeta)
        pan = _historial(conceptos, galeria, ordenes)
        # La fila se elige por el ARCHIVO y no por el numero: las dos dicen 1234 adentro, y
        # buscar por numero caeria en la primera de las dos --que puede ser cualquiera de
        # las dos, porque con la misma fecha el orden lo decide el nombre del archivo--.
        pan.orden_elegida = next(donde for donde, fila in enumerate(pan.historial)
                                 if fila.archivo == cual)

        _manejar(pan, conceptos, galeria, tecla=ord("b"), ordenes=ordenes)
        _sigue, cartel = _manejar(pan, conceptos, galeria, tecla=ord("s"), ordenes=ordenes)

        assert cartel, "el borrado no dejo ningun cartel"
        carteles[cual] = cartel
        # Y se fue la que el cartel nombra, no la de al lado.
        assert not (carpeta / f"{cual}.json").is_file()
        assert list(carpeta.glob("_papelera/*/" + f"{cual}.json"))

    assert carteles["1234"] != carteles["1234 - copia"]
    assert carteles["1234 - copia"].startswith(
        "ORDEN 1234 del archivo 1234 - copia.json A LA PAPELERA"
    )
    # La de todos los dias no arrastra el archivo: su numero ya dice cual es.
    assert carteles["1234"].startswith("ORDEN 1234 A LA PAPELERA")
    assert "del archivo" not in carteles["1234"]


# ---------------------------------------------------------------------------
# EL AVISO DE UN BORRADO A MEDIAS SOBREVIVE AL REINTENTO
#
# Un borrado se traba cuando el operario tiene una foto abierta en el visor de Windows.
# Ahi queda media orden en una carpeta de papelera, y el NOMBRE de esa carpeta se dice en
# un solo lugar: el renglon del aviso del historial. La tecla con la que se reintenta es la
# 'b', y era justo la que lo borraba.
# ---------------------------------------------------------------------------


def test_el_aviso_del_borrado_a_medias_sobrevive_a_la_b_del_reintento(
    taller, tmp_path: Path, monkeypatch
) -> None:
    """QUE SE ROMPIA: la 'b' del reintento se llevaba el unico cartel que decia donde quedo
    la mitad de la orden. El operario lee "quedo A MEDIAS en _papelera/<fecha y hora>",
    aprieta 'b' para volver a intentarlo, y el renglon desaparece antes de que haya podido
    anotar el nombre de la carpeta: a partir de ahi tiene que adivinar a cual de las
    carpetas de papelera --que se llaman todas con la fecha y la hora-- entrar a mirar.

    COMPROBADO QUE FALLA SIN EL ARREGLO: con _limpiar_aviso() puesto de vuelta en
    "pan.aviso = None" a secas, el primer assert de la 'b' dice
    "assert None == 'el borrado de 1234.json quedo A MEDIAS en _papelera/...: falto
    pruebas/1234'". Y con _decir() sin apagar el pegado falla la de abajo.

    El movimiento se traba igual que en el taller: shutil.move de una carpeta copia archivo
    por archivo cuando no la puede renombrar, asi que deja una foto del otro lado y se
    planta en la siguiente.

    Las cuatro mitades son una sola cosa: el cartel aparece, la 'b' no se lo lleva, el 'no,
    dejala' tampoco, y cuando el reintento SALE el cartel se va --porque lo que decia ya no
    es cierto--. Sin la ultima, un aviso pegado para siempre seria el otro error: un cartel
    viejo que ninguna tecla saca y que el operario deja de leer.
    """
    conceptos, galeria = taller
    carpeta = tmp_path / "ordenes"
    ordenes = _con_pruebas(carpeta, "1234")
    pan = _historial(conceptos, galeria, ordenes, "1234")
    mover = shutil.move

    def copia_a_medias(origen, destino):
        """Lo que hace shutil.move con una carpeta que no puede renombrar: copia la primera
        foto y se traba en la que sigue."""
        fuente = Path(origen)
        if fuente.is_dir():
            Path(destino).mkdir(parents=True, exist_ok=True)
            primera = sorted(fuente.iterdir())[0]
            shutil.copy2(primera, Path(destino) / primera.name)
            raise OSError(13, "Permission denied")
        return mover(origen, destino)

    monkeypatch.setattr(shutil, "move", copia_a_medias)

    _manejar(pan, conceptos, galeria, tecla=ord("b"), ordenes=ordenes)
    _manejar(pan, conceptos, galeria, tecla=ord("s"), ordenes=ordenes)

    a_medias = pan.aviso or ""
    assert "A MEDIAS" in a_medias, a_medias
    assert "1234.json" in a_medias and "pruebas/1234" in a_medias
    # LA CARPETA QUE NOMBRA EXISTE. Un aviso que mandara a mirar una carpeta inventada seria
    # peor que no decir nada: el operario va, no encuentra nada y deja de creerle al cartel.
    nombrada = re.search(r"_papelera/(\S+):", a_medias)
    assert nombrada, a_medias
    assert (carpeta / "_papelera" / nombrada.group(1)).is_dir()

    # LA 'b' DEL REINTENTO NO SE LO LLEVA.
    _manejar(pan, conceptos, galeria, tecla=ord("b"), ordenes=ordenes)
    assert pan.aviso == a_medias
    # Ni el "no, dejala": el que cancela es justamente el que va a ir a mirar esa carpeta.
    _manejar(pan, conceptos, galeria, tecla=_ESC, ordenes=ordenes)
    assert pan.aviso == a_medias

    # Y CUANDO EL REINTENTO SALE, EL CARTEL SE VA: la orden ya no quedo a medias.
    monkeypatch.setattr(shutil, "move", mover)
    _manejar(pan, conceptos, galeria, tecla=ord("b"), ordenes=ordenes)
    _manejar(pan, conceptos, galeria, tecla=ord("s"), ordenes=ordenes)

    assert pan.aviso is None
    assert not (carpeta / "1234.json").is_file()


def test_un_aviso_nuevo_apaga_el_pegado_y_la_tecla_siguiente_lo_limpia(
    taller, tmp_path: Path
) -> None:
    """PEGADO QUIERE DECIR "ESTE TEXTO NO SE LIMPIA SOLO", no "este renglon de la pantalla
    no se limpia nunca". Si el aviso del borrado a medias dejara pegado al que viene
    despues, el operario arrastraria un cartel viejo que ninguna tecla saca: la pantalla
    diria "no se pudo exportar" para siempre, tambien cuando la exportacion siguiente salio
    bien, y eso es volver a mentirle pero al otro lado.
    """
    conceptos, galeria = taller
    ordenes = _con_pruebas(tmp_path / "ordenes", "1234")
    pan = _historial(conceptos, galeria, ordenes, "1234")
    pan.aviso, pan.aviso_pegado = "el borrado quedo A MEDIAS en _papelera/20261116_164709", True

    run_live._decir(pan, "no se pudo exportar; revise el disco")

    assert pan.aviso == "no se pudo exportar; revise el disco"
    assert pan.aviso_pegado is False
    # Y la tecla siguiente --un digito, que es lo mas inocente que hay-- se lo lleva.
    _manejar(pan, conceptos, galeria, tecla=ord("1"), ordenes=ordenes)
    assert pan.aviso is None


# ---------------------------------------------------------------------------
# LOS NOMBRES SUELTOS SE SIGUEN CONTANDO AUNQUE YA HAYA UNA PRUEBA
#
# Un nombre suelto es Diente.foto: el nombre pelado de la foto que entro a la coleccion, sin
# carpeta. No es la prueba de nada y la fila tiene que declararlo como lo que es. Lo que se
# arreglo es que dejaba de declararlo en cuanto la orden conseguia UNA prueba de verdad.
#
# MEDIDO AQUI, leyendo el taller de verdad sin tocarlo (C:/Users/WoodTools-02/Desktop/
# Afilado/data): la orden 123 tiene 7 nombres sueltos y la 1234 otros 7, y de esas 14 fotos
# NINGUNA esta viva en data/coleccion --las 14 estan en data/coleccion/_papelera-- asi que
# hoy ese nombre no abre ningun archivo. Las 4 ordenes del taller tienen 0 dientes con
# prueba de verdad.
# ---------------------------------------------------------------------------


def test_la_fila_sigue_contando_los_nombres_sueltos_con_una_prueba_encima() -> None:
    """QUE SE ROMPIA: la PRIMERA prueba de verdad que conseguia una orden vieja hacia
    desaparecer el aviso de todos sus nombres sueltos, y de paso pintaba la fila de verde.
    El que lee esa fila de lejos ve verde, arma el reclamo creyendo que la sierra esta
    documentada, y lo que tiene es una foto de ochenta dientes. Es el caso de la orden 1234
    del taller: 7 dientes con el nombre pelado de una foto --las 7 en la papelera de la
    coleccion, medido-- el dia que le entre la primera prueba de verdad.

    COMPROBADO QUE FALLA SIN EL ARREGLO, volviendo el codigo a lo de antes, las dos
    mitades por separado: con la guarda vieja (con_prueba == 0 and con_nombre_suelto > 0) el
    renglon de abajo no dice los 7 --queda "1 CON PRUEBA" y nada mas-- y con el verde
    incondicional la columna sale (88, 170, 70) en vez del ambar (20, 160, 235).

    El numero de la columna no cambia --sigue siendo el de las pruebas que estan-- lo que
    cambia es el color, que es lo unico que se lee con la sierra en la mano.
    """
    con_una_prueba = _fila(con_prueba=1, con_nombre_suelto=7)

    textos = _textos(_pantalla_de_ordenes(con_una_prueba))

    # La prueba que esta se cuenta, y los 7 sueltos se siguen declarando.
    assert any("1 CON PRUEBA" in texto for texto in textos)
    assert any("7 nombres de foto sueltos, no son prueba" in texto for texto in textos), textos
    # AMBAR Y NO VERDE: el verde es "esta orden tiene su prueba", no "tiene algo de prueba".
    assert panel._texto_prueba(con_una_prueba)[1] == panel._AMBAR

    # Y LA CONTRACARA: la orden que tiene su prueba y ningun nombre suelto sigue en verde.
    # Sin esta mitad, una fila que fuera siempre ambar pasaria la prueba de arriba y el
    # color dejaria de decir nada.
    entera = _fila(con_prueba=15, con_nombre_suelto=0)
    assert panel._texto_prueba(entera)[1] == panel._VERDE_OK
    assert not any("nombres de foto sueltos" in texto
                   for texto in _textos(_pantalla_de_ordenes(entera)))


# ---------------------------------------------------------------------------
# EL RENGLON DE LOS .exe DE RESPALDO, UNA SOLA VEZ
#
# Al lado de los datos del taller hay cuatro .exe de respaldo, que son el plan B si este
# programa falla.
#
# MEDIDO AQUI: el ordenes.py del commit 1cc3c2b --el codigo con el que se armaron esos
# .exe-- arma cada diente con "Diente(**d)" y la palabra "prueba" no aparece NI UNA VEZ en
# todo el archivo (git show 1cc3c2b:src/afilado/ordenes.py), asi que un diente con prueba le
# es un argumento inesperado y le vuelve ilegible la orden entera. Y hoy son 0 de 4: leidas
# las 4 ordenes del taller, ninguna tiene todavia un diente con prueba, asi que este renglon
# no se dibuja. Aparece con la primera.
#
# COMPROBADO QUE FALLA SIN EL ARREGLO, las dos mitades por separado: dibujando el renglon una
# vez por fila con prueba salen 3 renglones iguales ("assert 3 == 1"), y dibujandolo siempre
# --sin preguntar si hay alguna orden con prueba-- aparece con el historial sin ninguna.
# ---------------------------------------------------------------------------


def test_el_renglon_de_los_exe_de_respaldo_aparece_una_sola_vez() -> None:
    """Es un dato de como son las cosas, no algo que acaba de pasar, asi que va UNA vez y no
    pegado a cada fila: lo que se repite en ocho renglones se deja de leer, y ocho renglones
    iguales tapan el que cuenta algo nuevo.

    Y con ninguna orden con prueba NO SE DICE NADA: hoy las 4 ordenes del taller dan
    con_prueba=0, y un renglon que avisara igual estaria afirmando algo que todavia no pasa
    --que los .exe de respaldo ya no abren las ordenes-- cuando la verdad es que las abren
    todas. Las dos mitades van juntas: con una sola, un renglon escrito a lo loco en cada
    pantalla pasaria la mitad de arriba.
    """
    con_prueba = [_fila(str(100 + indice), archivo=str(100 + indice),
                        con_prueba=1 if indice < 3 else 0) for indice in range(8)]

    textos = _textos(_pantalla_de_ordenes(*con_prueba))

    cuantos = [texto for texto in textos if "respaldo del taller" in texto]
    assert len(cuantos) == 1, cuantos
    assert "3 ordenes ya tienen prueba guardada" in cuantos[0]

    # NINGUNA CON PRUEBA: no se dice nada.
    ninguna = [_fila(str(100 + indice), archivo=str(100 + indice)) for indice in range(8)]
    assert not any("respaldo del taller" in texto
                   for texto in _textos(_pantalla_de_ordenes(*ninguna)))

    # UNA SOLA: en singular. "1 ordenes" no es castellano de ninguna parte, y este renglon
    # lo lee un operario.
    una = _textos(_pantalla_de_ordenes(_fila(con_prueba=1)))
    sola = [texto for texto in una if "respaldo del taller" in texto]
    assert len(sola) == 1 and "1 orden ya tiene prueba guardada" in sola[0], sola


def test_la_fila_dice_cuantas_fotos_le_faltan_al_concepto_para_decidir() -> None:
    """ESTA PANTALLA ES DONDE SE DESTRABA EL "NO ESTOY SEGURO" DE TODOS LOS DIENTES.

    Un concepto que no llego al minimo de fotos no puede ser un veredicto, asi que la
    memoria no lo mide y no aparece en `aciertos`: su fila quedaba mostrando "2 fotos" y
    nada mas, igual que si estuviera todo bien. El 16/11/2026 habia dos conceptos asi
    --Diente_sin_placa y Sierra_Nueva_TP, con 2 fotos de 3-- y por eso el programa no se
    animaba con ningun diente. Alcanzaba con una foto de cada uno, y no habia forma de
    saberlo mirando la pantalla.
    """
    flaco = Concepto(nombre="Sierra_Nueva_TP", carpeta=Path("."), fotos=2)
    lleno = Concepto(nombre="Dientes_rotos", carpeta=Path("."), fotos=8)
    estado = panel.EstadoPanel(
        conceptos=[flaco, lleno],
        aciertos={"Dientes_rotos": (8, 8)},
        minimo_fotos=3,
    )

    assert "falta 1 foto mas" in panel._texto_aciertos(estado, flaco)
    assert "acierta 8 de 8" in panel._texto_aciertos(estado, lleno)
    # Sin minimo --no se sabe cual es-- la fila no inventa nada.
    sin_minimo = panel.EstadoPanel(conceptos=[flaco], aciertos={})
    assert panel._texto_aciertos(sin_minimo, flaco) == "2 fotos"
