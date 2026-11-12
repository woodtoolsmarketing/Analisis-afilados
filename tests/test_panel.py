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

from pathlib import Path

import cv2
import numpy as np
import pytest

from afilado import panel
from afilado.cli import run_live
from afilado.conceptos import Concepto, Conceptos
from afilado.ordenes import Orden, Ordenes

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
    # Y quedo guardado en disco, no solo en memoria.
    assert Ordenes(str(tmp_path / "ordenes")).abierta.total_revisados == 1


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


def test_de_la_orden_se_vuelve_a_los_conceptos_sin_cerrar_el_panel(taller, tmp_path) -> None:
    conceptos, galeria = taller
    ordenes = Ordenes(str(tmp_path / "ordenes"))
    orden = ordenes.abrir(Orden.nueva("1234", 1, 2))
    pan = _estado(conceptos, galeria, modo=panel.MODO_ORDEN, orden=orden)

    sigue, _ = _manejar(pan, conceptos, galeria, tecla=_ESC, ordenes=ordenes)

    assert sigue is True and pan.modo == panel.MODO_LISTA
