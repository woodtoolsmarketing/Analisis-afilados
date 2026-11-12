"""Pruebas de la orden de afilado y su informe.

LO QUE ESTE INFORME TIENE QUE PODER CONTESTAR es un reclamo: "esta sierra volvio mal
afilada". Si el informe pierde un diente, o dice que se reviso uno que no se reviso, no
sirve para eso y es peor que no tenerlo. Por eso lo que mas se prueba aca es que nada se
pierda: los dientes sin revisar aparecen igual, cada anotacion se guarda en el acto, y lo
que se escribio a mano llega entero al papel.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from afilado.ordenes import Orden, Ordenes, Sierra, sanear_numero


def _orden(numero: str = "1234", sierras: int = 2, dientes: int = 3) -> Orden:
    return Orden.nueva(
        numero, sierras, dientes, nota_pedido="NP-998", cliente="Carpinteria del Sur"
    )


# ---------------------------------------------------------------------------
# Armar la orden
# ---------------------------------------------------------------------------


def test_una_orden_nace_con_todos_sus_dientes() -> None:
    orden = _orden(sierras=3, dientes=24)

    assert len(orden.sierras) == 3
    assert orden.total_dientes == 72
    assert orden.total_revisados == 0
    # Y los dientes existen desde el principio, aunque no se haya revisado ninguno: el
    # informe tiene que poder mostrar los que faltan.
    assert [d.numero for d in orden.sierras[0].revisados] == list(range(1, 25))


def test_una_orden_sin_numero_no_se_crea() -> None:
    with pytest.raises(ValueError):
        Orden.nueva("   ", 1, 10)


@pytest.mark.parametrize("sierras,dientes", [(500, 10), (1, 5000)])
def test_un_cero_de_mas_se_frena(sierras: int, dientes: int) -> None:
    """Tecleando a mano, 240 dientes y 2400 se escriben casi igual."""
    with pytest.raises(ValueError):
        Orden.nueva("1234", sierras, dientes)


def test_el_numero_de_orden_queda_usable_como_archivo() -> None:
    assert sanear_numero("OA 1234/25") == "OA_1234_25"
    assert sanear_numero("////") == ""


# ---------------------------------------------------------------------------
# Anotar diente por diente
# ---------------------------------------------------------------------------


def test_anotar_avanza_al_diente_siguiente() -> None:
    orden = _orden(sierras=1, dientes=3)

    assert orden.diente().numero == 1
    orden.anotar(estado="afilado", parecido=0.91, aprobado=True, origen="la memoria")
    assert orden.diente().numero == 2
    assert orden.total_revisados == 1


def test_el_diente_guarda_lo_que_se_vio() -> None:
    orden = _orden(sierras=1, dientes=2)

    diente = orden.anotar(
        estado="mal_afilado", parecido=0.78, aprobado=False,
        origen="la memoria", foto="mal_afilado_001.jpg",
    )

    assert diente.numero == 1 and diente.estado == "mal_afilado"
    assert diente.aprobado is False and diente.parecido == pytest.approx(0.78)
    assert diente.foto == "mal_afilado_001.jpg"
    assert diente.instante    # queda la fecha y la hora


def test_cuando_se_terminan_los_dientes_no_se_anota_de_mas() -> None:
    """Una sierra de 3 dientes no puede terminar con 4 anotaciones."""
    orden = _orden(sierras=1, dientes=3)
    for _ in range(3):
        orden.anotar(estado="afilado", aprobado=True)

    assert orden.anotar(estado="afilado", aprobado=True) is None
    assert orden.sierras[0].cuantos_revisados == 3
    assert orden.sierras[0].completa


def test_saltear_deja_el_hueco_explicado() -> None:
    """Un diente que no se pudo mirar tiene que figurar como tal, no desaparecer."""
    orden = _orden(sierras=1, dientes=2)

    orden.saltear("no se pudo apoyar")

    diente = orden.sierras[0].revisados[0]
    assert diente.revisado and diente.aprobado is None
    assert "no se pudo apoyar" in diente.nota
    assert orden.diente().numero == 2


def test_se_cuentan_los_rechazados_y_los_que_no_se_pudo_decidir() -> None:
    orden = _orden(sierras=1, dientes=4)
    orden.anotar(estado="afilado", aprobado=True)
    orden.anotar(estado="roto", aprobado=False)
    orden.anotar(estado="mal_afilado", aprobado=False)
    orden.anotar(estado=None, aprobado=None, origen="la memoria no se animo")

    assert orden.total_rechazados == 2
    assert orden.sierras[0].sin_decidir == 1
    assert orden.completa


def test_pasar_a_la_sierra_siguiente_y_dar_la_vuelta() -> None:
    orden = _orden(sierras=3, dientes=2)

    assert orden.siguiente_sierra() == 2
    assert orden.diente().numero == 1          # arranca de cero en la sierra nueva
    orden.anotar(estado="afilado", aprobado=True)
    assert orden.siguiente_sierra() == 3
    assert orden.siguiente_sierra() == 1
    assert orden.sierras[1].cuantos_revisados == 1


def test_el_resumen_dice_donde_esta_parado() -> None:
    orden = _orden(sierras=2, dientes=10)
    orden.anotar(estado="afilado", aprobado=True)

    resumen = orden.resumen()

    assert "1234" in resumen and "sierra 1 de 2" in resumen and "diente 2 de 10" in resumen


# ---------------------------------------------------------------------------
# El informe
# ---------------------------------------------------------------------------


def test_el_informe_lleva_lo_que_se_cargo_a_mano() -> None:
    orden = _orden()

    informe = orden.informe()

    assert "1234" in informe and "NP-998" in informe and "Carpinteria del Sur" in informe


def test_el_informe_muestra_los_dientes_sin_revisar() -> None:
    """Que falte un diente es justamente lo que un reclamo va a preguntar."""
    orden = _orden(sierras=1, dientes=3)
    orden.anotar(estado="afilado", aprobado=True)

    informe = orden.informe()

    assert informe.count("(sin revisar)") == 2


def test_el_informe_distingue_aprobado_rechazado_y_sin_decidir() -> None:
    orden = _orden(sierras=1, dientes=3)
    orden.anotar(estado="afilado", parecido=0.95, aprobado=True)
    orden.anotar(estado="roto", parecido=0.88, aprobado=False)
    orden.anotar(estado=None, parecido=None, aprobado=None)

    informe = orden.informe()

    assert "APROBADO" in informe and "RECHAZADO" in informe and "sin decidir" in informe
    assert "95%" in informe and "88%" in informe


def test_la_planilla_tiene_una_fila_por_diente() -> None:
    orden = _orden(sierras=2, dientes=5)
    orden.anotar(estado="afilado", aprobado=True)

    filas = orden.filas_csv()

    assert len(filas) == 1 + 10              # encabezado + 2 sierras x 5 dientes
    assert filas[0][0] == "orden"
    assert filas[1][5] == "1" and filas[1][7] == "APROBADO"
    assert filas[2][7] == "sin revisar"


# ---------------------------------------------------------------------------
# Guardar y retomar
# ---------------------------------------------------------------------------


def test_la_orden_se_guarda_con_cada_diente(tmp_path: Path) -> None:
    """Un corte de luz en la mitad de una sierra no puede borrar dos horas de trabajo."""
    ordenes = Ordenes(str(tmp_path / "ordenes"))
    orden = ordenes.abrir(_orden(sierras=1, dientes=5))

    orden.anotar(estado="afilado", aprobado=True)
    ordenes.guardar()

    datos = json.loads(ordenes.ruta("1234").read_text(encoding="utf-8"))
    assert datos["sierras"][0]["revisados"][0]["estado"] == "afilado"


def test_al_abrir_el_programa_se_retoma_la_orden(tmp_path: Path) -> None:
    carpeta = str(tmp_path / "ordenes")
    primeras = Ordenes(carpeta)
    orden = primeras.abrir(_orden(sierras=2, dientes=4))
    orden.anotar(estado="afilado", aprobado=True)
    orden.siguiente_sierra()
    primeras.guardar()

    otras = Ordenes(carpeta)

    assert otras.abierta is not None
    assert otras.abierta.numero == "1234"
    assert otras.abierta.sierra_actual == 2
    assert otras.abierta.total_revisados == 1


def test_cerrar_una_orden_la_deja_guardada(tmp_path: Path) -> None:
    carpeta = str(tmp_path / "ordenes")
    ordenes = Ordenes(carpeta)
    ordenes.abrir(_orden())

    ordenes.cerrar()

    assert ordenes.abierta is None
    assert Ordenes(carpeta).abierta is None
    # Pero la orden NO se borra: el informe se puede pedir despues.
    guardada = ordenes.cargar("1234")
    assert guardada is not None and guardada.cerrada


def test_el_informe_queda_en_dos_archivos(tmp_path: Path) -> None:
    ordenes = Ordenes(str(tmp_path / "ordenes"))
    orden = ordenes.abrir(_orden(sierras=1, dientes=2))
    orden.anotar(estado="roto", parecido=0.9, aprobado=False)

    texto, planilla = ordenes.escribir_informe()

    assert texto is not None and planilla is not None
    assert "RECHAZADO" in texto.read_text(encoding="utf-8")
    assert "orden;nota_pedido" in planilla.read_text(encoding="utf-8-sig")


def test_una_orden_rota_en_disco_no_frena_el_programa(tmp_path: Path) -> None:
    carpeta = tmp_path / "ordenes"
    carpeta.mkdir()
    (carpeta / "1234.json").write_text("{ esto no es json", encoding="utf-8")
    (carpeta / ".abierta").write_text("1234", encoding="utf-8")

    ordenes = Ordenes(str(carpeta))

    assert ordenes.abierta is None
    assert ordenes.cargar("1234") is None


def test_lo_anotado_sobrevive_a_la_vuelta_completa(tmp_path: Path) -> None:
    """Guardar y volver a leer no puede cambiar ni un dato del informe."""
    carpeta = str(tmp_path / "ordenes")
    ordenes = Ordenes(carpeta)
    orden = ordenes.abrir(_orden(sierras=2, dientes=3))
    orden.anotar(estado="mal_afilado", parecido=0.81, aprobado=False, origen="la memoria",
                 foto="f.jpg", nota="el filo esta redondeado")
    orden.saltear("no se pudo apoyar")
    ordenes.guardar()

    recuperada = Ordenes(carpeta).abierta

    assert recuperada.informe() == orden.informe()
    assert recuperada.filas_csv() == orden.filas_csv()


def test_listar_muestra_las_ordenes_guardadas(tmp_path: Path) -> None:
    carpeta = str(tmp_path / "ordenes")
    ordenes = Ordenes(carpeta)
    ordenes.abrir(_orden("1111"))
    ordenes.cerrar()
    ordenes.abrir(_orden("2222"))

    assert Ordenes(carpeta).listar() == ["1111", "2222"]


def test_una_sierra_que_viene_con_dientes_cargados_no_se_duplica() -> None:
    """Al releer del disco, los dientes ya estan: __post_init__ no puede agregarlos de nuevo."""
    sierra = Sierra(numero=1, dientes=3, revisados=[])
    assert len(sierra.revisados) == 3

    otra = Sierra(numero=1, dientes=3, revisados=list(sierra.revisados))
    assert len(otra.revisados) == 3
