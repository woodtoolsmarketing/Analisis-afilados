"""Pruebas de la orden de afilado y su informe.

LO QUE ESTE INFORME TIENE QUE PODER CONTESTAR es un reclamo: "esta sierra volvio mal
afilada". Si el informe pierde un diente, o dice que se reviso uno que no se reviso, no
sirve para eso y es peor que no tenerlo. Por eso lo que mas se prueba aca es que nada se
pierda: los dientes sin revisar aparecen igual, cada anotacion se guarda en el acto, y lo
que se escribio a mano llega entero al papel.
"""

from __future__ import annotations

import json
import re
import shutil
import zipfile
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import pytest

from afilado.ordenes import (
    VERSION_FORMATO,
    Orden,
    Ordenes,
    Sierra,
    clave_de_fecha,
    nombre_de_archivo_sano,
    sanear_numero,
)


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
    assert diente.anotado and diente.aprobado is None
    assert "no se pudo apoyar" in diente.nota
    assert orden.diente().numero == 2


def test_un_diente_salteado_no_cuenta_como_revisado() -> None:
    """Lo peor que podia decir el informe: apretar saltear en los 24 dientes dejaba la
    sierra 'completa' con '24 revisados' sin que nadie hubiera mirado uno."""
    orden = _orden(sierras=1, dientes=3)

    for _ in range(3):
        orden.saltear("no se pudo apoyar")

    sierra = orden.sierras[0]
    assert sierra.cuantos_revisados == 0
    assert sierra.cuantos_salteados == 3
    assert sierra.sin_decidir == 0          # tampoco son dientes "sin decidir"
    assert orden.total_revisados == 0
    # Pero la sierra SI esta terminada: no queda nada por tocar, y el programa no puede
    # quedarse pidiendo un diente que el operario ya decidio pasar de largo.
    assert sierra.completa and orden.completa
    assert orden.diente() is None


def test_el_informe_distingue_salteado_de_sin_revisar() -> None:
    orden = _orden(sierras=1, dientes=3)
    orden.anotar(estado="afilado", aprobado=True)
    orden.saltear("el diente estaba tapado")

    informe = orden.informe()

    assert "(salteado: el diente estaba tapado)" in informe
    assert "(sin revisar)" in informe        # el tercero, que nadie toco
    assert "1 revisados" in informe and "1 salteados" in informe
    assert ["salteado"] == [f[7] for f in orden.filas_csv()[1:] if f[7] == "salteado"]


def test_una_orden_vieja_sigue_distinguiendo_los_salteados(tmp_path: Path) -> None:
    """Las ordenes guardadas antes de que el salteo tuviera campo propio se reconocen por
    el origen. Sin esto, al releerlas los salteados volvian a contar como revisados."""
    from afilado.ordenes import Diente

    viejo = Diente(numero=1, origen="sin revisar", nota="salteado", instante="2026-01-01 10:00")

    assert viejo.salteado and not viejo.revisado and viejo.anotado


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


def test_si_no_se_puede_guardar_se_dice_y_no_queda_basura(tmp_path: Path, monkeypatch) -> None:
    """Reproducido con el disco lleno: guardar() devolvia None, nadie lo miraba, y el
    programa seguia anotando dientes toda la tarde contra una orden que no existia en
    ningun archivo. Ademas dejaba un .json.tmp a medio escribir por cada intento."""
    from afilado.ordenes import Ordenes

    ordenes = Ordenes(str(tmp_path / "ordenes"))
    orden = ordenes.abrir(_orden(sierras=1, dientes=4))
    assert not ordenes.fallo_al_guardar

    def _lleno(*_args, **_kwargs):
        raise OSError(28, "No space left on device")

    monkeypatch.setattr("os.replace", _lleno)
    orden.anotar(estado="afilado", aprobado=True)

    assert ordenes.guardar() is None
    assert ordenes.fallo_al_guardar
    assert list((tmp_path / "ordenes").glob("*.tmp")) == []


def test_abrir_avisa_cuando_la_orden_no_llego_al_disco(tmp_path: Path, monkeypatch) -> None:
    """Y NO LA DEJA ABIERTA. Dejandola, el operario veia el cartel de la orden arriba, se
    pasaba la tarde anotando dientes contra ella, y al cerrar el programa no quedaba nada
    porque nunca hubo archivo."""
    from afilado.ordenes import Ordenes

    ordenes = Ordenes(str(tmp_path / "ordenes"))
    monkeypatch.setattr("os.replace", lambda *a, **k: (_ for _ in ()).throw(OSError(28, "lleno")))

    ordenes.abrir(_orden(sierras=1, dientes=4))

    assert ordenes.fallo_al_guardar
    assert ordenes.abierta is None
    assert not (tmp_path / "ordenes" / "1234.json").exists()


def test_si_no_se_puede_abrir_la_nueva_sigue_valiendo_la_anterior(tmp_path: Path, monkeypatch) -> None:
    """El operario tiene una orden a medio revisar y abre otra que no se puede guardar: lo
    que no puede pasar es que pierda la que estaba trabajando."""
    from afilado.ordenes import Ordenes

    ordenes = Ordenes(str(tmp_path / "ordenes"))
    ordenes.abrir(_orden("1111", sierras=1, dientes=4))
    ordenes.abierta.anotar(estado="afilado", aprobado=True)
    monkeypatch.setattr("os.replace", lambda *a, **k: (_ for _ in ()).throw(OSError(28, "lleno")))

    ordenes.abrir(_orden("2222", sierras=1, dientes=4))

    assert ordenes.abierta is not None
    assert ordenes.abierta.numero == "1111"
    assert ordenes.abierta.total_revisados == 1


def test_un_json_danado_no_deja_escribirle_encima(tmp_path: Path) -> None:
    """El control de duplicados usaba cargar(), que devuelve None tanto si el archivo no
    esta como si esta pero no se puede leer. Con un .json cortado --un corte de luz en la
    mitad de un guardado-- la orden 1234 pasaba por 'nueva' y se escribia encima de la
    unica copia del trabajo de ese dia."""
    from afilado.ordenes import Ordenes

    carpeta = tmp_path / "ordenes"
    carpeta.mkdir(parents=True)
    (carpeta / "1234.json").write_text('{"numero": "1234", "sier', encoding="utf-8")
    ordenes = Ordenes(str(carpeta))

    assert ordenes.cargar("1234") is None       # no se puede leer...
    assert ordenes.existe("1234")               # ...pero esta, y eso es lo que importa


def test_el_informe_de_una_orden_cerrada_dice_cuando_se_cerro(tmp_path: Path) -> None:
    """El papel que se le entrega al cliente llevaba impreso 'Cerrada: sigue abierta'."""
    from afilado.ordenes import Ordenes

    ordenes = Ordenes(str(tmp_path / "ordenes"))
    ordenes.abrir(_orden(sierras=1, dientes=2))
    ordenes.abierta.anotar(estado="afilado", aprobado=True)

    cerrada = ordenes.cerrar()
    texto, _planilla = ordenes.escribir_informe(cerrada)

    informe = texto.read_text(encoding="utf-8")
    assert "sigue abierta" not in informe
    assert cerrada.cerrada and cerrada.cerrada in informe


def test_un_parecido_que_no_es_numero_no_voltea_el_informe() -> None:
    """Viene de un .json que puede estar editado a mano o escrito por una version vieja.
    Con un texto adentro, el informe reventaba al formatearlo, y con el .exe sin consola
    eso es el programa cerrandose solo justo cuando se imprime el papel de un reclamo."""
    from afilado.ordenes import Orden

    orden = Orden.desde_datos({
        "numero": "1234",
        "sierras": [{"numero": 1, "dientes": 2, "revisados": [
            {"numero": 1, "estado": "afilado", "parecido": "altisimo",
             "aprobado": True, "instante": "12/11/2026 10:00"},
        ]}],
    })

    assert orden.sierras[0].revisados[0].parecido is None
    assert "APROBADO" in orden.informe()
    assert orden.filas_csv()


def test_una_sierra_de_cero_dientes_no_nace_terminada() -> None:
    """Por aca va a entrar la app de afilados. Un 'dientes': 0 daba una sierra COMPLETA
    sin revisar nada: el informe decia la orden terminada y no se podia anotar un diente."""
    from afilado.ordenes import Orden

    orden = Orden.desde_datos({"numero": "1234", "sierras": [{"numero": 1, "dientes": 0}]})

    sierra = orden.sierras[0]
    assert sierra.dientes >= 1
    assert not sierra.completa
    assert orden.diente() is not None


def test_el_csv_no_deja_que_excel_ejecute_lo_que_vino_del_mostrador() -> None:
    """El cliente y la nota de pedido los teclea el operario con lo que le pasan, y Excel
    ejecuta cualquier celda que arranque con = + - o @."""
    from afilado.ordenes import Orden

    orden = Orden.nueva("1234", 1, 1, nota_pedido="=1+1", cliente="@SUM(A1)")
    orden.anotar(estado="afilado", aprobado=True, origen="-cmd")

    fila = orden.filas_csv()[1]

    assert fila[1] == "'=1+1" and fila[2] == "'@SUM(A1)"
    assert fila[9] == "'-cmd"


def test_una_orden_a_medio_revisar_no_pierde_dientes_ni_repite() -> None:
    """Rellenando por POSICION, con 6 dientes y los numeros 5 y 6 ya anotados la lista
    quedaba [5, 6, 3, 4, 5, 6]: el informe imprimia el 5 y el 6 dos veces, el 1 y el 2 en
    ninguna parte, y el total cerrando igual. Hoy no se llega aca con las fotos y las
    teclas del taller --guardar() serializa siempre la lista completa-- pero es la puerta
    por donde va a entrar la app de afilados."""
    orden = Orden.desde_datos({
        "numero": "9",
        "sierras": [{
            "numero": 1, "dientes": 6,
            "revisados": [
                {"numero": 5, "estado": "afilado", "aprobado": True,
                 "instante": "12/11/2026 10:00:00"},
                # "6" de texto: lo que puede mandar otra app. Sin reconocerlo como ya
                # anotado, el informe sacaba dos renglones para el mismo diente.
                {"numero": "6", "estado": "roto", "aprobado": False,
                 "instante": "12/11/2026 10:01:00"},
            ],
        }],
    })
    sierra = orden.sierras[0]

    assert [d.numero for d in sierra.revisados] == [1, 2, 3, 4, 5, 6]
    assert sierra.cuantos_revisados == 2
    assert sierra.proximo().numero == 1
    assert orden.informe().count("(sin revisar)") == 4


@pytest.mark.parametrize("payload", [
    {"sierras": {"1": {"dientes": 4}}},
    {"sierras": ["sierra 1"]},
    ["una", "lista"],
    7,
])
def test_un_json_con_otra_forma_no_impide_abrir_el_programa(tmp_path: Path, payload) -> None:
    """Reproducido: los cuatro daban AttributeError --'str'/'list'/'int' object has no
    attribute 'get'-- desde Ordenes.__init__, que corre antes de que haya pantalla, asi que
    el programa no abria; y con el .abierta apuntando ahi, no abria mas."""
    carpeta = tmp_path / "ordenes"
    carpeta.mkdir()
    (carpeta / "1234.json").write_text(json.dumps(payload), encoding="utf-8")
    (carpeta / ".abierta").write_text("1234", encoding="utf-8")

    ordenes = Ordenes(str(carpeta))

    assert ordenes.abierta is None
    assert ordenes.cargar("1234") is None


def test_el_informe_imprime_el_numero_que_tecleo_el_operario(tmp_path: Path) -> None:
    """Guardando el numero saneado, el papel de un reclamo salia con un numero que nadie
    escribio: Orden.nueva("OA/123") imprimia "Orden de afilado : OA_123". El saneado es
    solo para el nombre del archivo."""
    carpeta = str(tmp_path / "ordenes")
    ordenes = Ordenes(carpeta)
    orden = ordenes.abrir(Orden.nueva("OA/123", 1, 2))

    assert orden.numero == "OA/123"
    assert "Orden de afilado : OA/123" in orden.informe()
    assert orden.filas_csv()[1][0] == "OA/123"
    assert ordenes.ruta(orden.numero).name == "OA_123.json"
    texto, planilla = ordenes.escribir_informe()
    assert (texto.name, planilla.name) == ("informe_OA_123.txt", "informe_OA_123.csv")
    assert Ordenes(carpeta).abierta.numero == "OA/123"
    # Y sigue haciendo falta que quede algo usable como nombre de archivo.
    with pytest.raises(ValueError):
        Orden.nueva("////", 1, 2)


def test_la_orden_se_fuerza_al_disco_antes_del_renombre(tmp_path: Path, monkeypatch) -> None:
    """guardar() promete que un corte de luz no borra las dos horas anteriores. Sin forzar
    los datos, el renombre podia asentarse antes que ellos y el .json quedaba en bytes
    nulos: reproducido ese estado, Ordenes(...).abierta vuelve None y los 40 dientes
    anotados no se recuperan de ninguna parte, porque se reescribe siempre el mismo archivo
    y no hay copia generacional. MEDIDO: el fsync cuesta 0.4 ms por diente."""
    import os

    pasos: list[str] = []
    fsync_real, replace_real = os.fsync, os.replace
    monkeypatch.setattr(os, "fsync", lambda d: (pasos.append("fsync"), fsync_real(d))[1])
    monkeypatch.setattr(os, "replace", lambda a, b: (pasos.append("replace"), replace_real(a, b))[1])

    ordenes = Ordenes(str(tmp_path / "ordenes"))
    ordenes.abrir(_orden(sierras=1, dientes=2))

    assert pasos == ["fsync", "replace"]


def test_el_informe_no_hace_pasar_el_numero_de_sierra_por_una_identificacion() -> None:
    """Hoy nada escribe Sierra.identificacion --no hay campo en el formulario ni tecla, y en
    los tres informes de verdad del taller la columna salio vacia-- asi que "SIERRA 2 de 3"
    es solo el orden en que se pasaron por el microscopio. El informe lo tiene que decir:
    si no, un reclamo sobre UNA sierra concreta se lee sobre ese numero."""
    orden = _orden(sierras=2, dientes=2)
    orden.sierras[1].identificacion = "CIRC-450-07"

    papel = orden.informe()

    assert "SIERRA 1 de 2   (sin identificar: es el orden en que se pasaron)" in papel
    assert "SIERRA 2 de 2   (CIRC-450-07)" in papel
    assert "El numero de SIERRA es el orden en que se pasaron por el microscopio" in papel


def test_una_orden_cerrada_no_se_retoma_al_abrir_el_programa(tmp_path: Path) -> None:
    """Si el borrado del marcador .abierta falla --o el programa se corta entre el guardar()
    y el _recordar_abierta(None) de cerrar()-- el programa retomaba una orden ya informada y
    entregada y seguia anotando dientes encima: reproducido, el papel decia "3 en total, 1
    revisados" y el .json termino con 2."""
    carpeta = tmp_path / "ordenes"
    ordenes = Ordenes(str(carpeta))
    ordenes.abrir(_orden("1234", sierras=1, dientes=3))
    ordenes.abierta.anotar(estado="afilado", aprobado=True, origen="la memoria")
    ordenes.guardar()
    ordenes.escribir_informe()
    ordenes.cerrar()
    (carpeta / ".abierta").write_text("1234", encoding="utf-8")      # el borrado fallo

    de_nuevo = Ordenes(str(carpeta))

    assert de_nuevo.abierta is None
    assert (carpeta / ".abierta").read_text(encoding="utf-8") == ""
    assert de_nuevo.cargar("1234").total_revisados == 1              # sigue en disco


def test_la_cabecera_del_informe_no_se_parte_con_un_salto_de_linea() -> None:
    """desde_datos() es la puerta de la app de afilados y no limpiaba nada: con un salto de
    linea en la nota de pedido el papel imprimia "segunda linea" sola y sin etiqueta, y la
    linea del cliente con la columna corrida. El .strip() de nueva() tampoco frena un salto
    en el MEDIO."""
    orden = Orden.desde_datos({
        "numero": "1234",
        "nota_pedido": 'NP-1; "urgente"\nsegunda linea',
        "cliente": "   Aserradero\tSur  ",
        "sierras": [{"numero": 1, "dientes": 1}],
    })

    papel = orden.informe().splitlines()

    assert 'Nota de pedido   : NP-1; "urgente" segunda linea' in papel
    assert "Cliente          : Aserradero Sur" in papel
    # Y lo mismo por el camino de hoy, el formulario.
    a_mano = Orden.nueva("1234", 1, 1, nota_pedido="NP-1\nsegunda", cliente=" Aserradero ")
    assert (a_mano.nota_pedido, a_mano.cliente) == ("NP-1 segunda", "Aserradero")


def test_un_informe_que_falla_no_destruye_el_que_ya_estaba(tmp_path: Path) -> None:
    """Reproducido con el disco lleno a la mitad del .csv: el informe_1234.csv bueno (5411
    bytes, 41 filas) quedaba en 241 bytes --la cabecera y una fila-- porque se escribia
    encima. Y la funcion devolvia (None, None) aunque el .txt si hubiera salido, asi que la
    pantalla no podia decir que mitad falto."""
    ordenes = Ordenes(str(tmp_path / "ordenes"))
    orden = ordenes.abrir(_orden(sierras=1, dientes=3))
    orden.anotar(estado="roto", parecido=0.3, aprobado=False, origen="la memoria")
    texto, planilla = ordenes.escribir_informe()
    bueno_txt = texto.read_text(encoding="utf-8")
    bueno_csv = planilla.read_text(encoding="utf-8-sig")

    def se_llena_el_disco():
        raise OSError(28, "No space left on device")

    orden.filas_csv = se_llena_el_disco          # type: ignore[method-assign]

    assert ordenes.escribir_informe() == (texto, None)
    assert planilla.read_text(encoding="utf-8-sig") == bueno_csv
    assert texto.read_text(encoding="utf-8") == bueno_txt
    assert not list((tmp_path / "ordenes").glob("*.tmp"))


def test_la_tecla_s_recorre_todas_las_sierras_aunque_repitan_numero() -> None:
    """Buscando por NUMERO, con [1, 1, 2] seis 's' devolvian siempre la primera y los
    dientes de la segunda no se podian anotar nunca. Y con sierra_actual fuera de rango la
    sierra 1 quedaba sin visitar. sierra_actual es la POSICION, 1..N."""
    orden = Orden(numero="9", sierras=[
        Sierra(numero=1, dientes=2, identificacion="A"),
        Sierra(numero=1, dientes=2, identificacion="B"),
        Sierra(numero=2, dientes=2, identificacion="C"),
    ])

    recorrido = []
    for _ in range(4):
        recorrido.append(orden.sierra().identificacion)
        orden.siguiente_sierra()

    assert recorrido == ["A", "B", "C", "A"]

    fuera = Orden(numero="9", sierra_actual=99, sierras=[
        Sierra(numero=1, dientes=2, identificacion="A"),
        Sierra(numero=2, dientes=2, identificacion="B"),
    ])
    assert fuera.sierra().identificacion == "B"      # se acota al final, no cae en la A
    assert fuera.siguiente_sierra() == 1             # y la 's' da la vuelta
    assert fuera.sierra().identificacion == "A"


def test_una_orden_que_dejo_de_ser_la_abierta_se_puede_retomar(tmp_path: Path) -> None:
    """Solo se podia trabajar la orden marcada en .abierta: listar() no se llamaba desde
    ninguna parte del programa y cargar() solo se usaba para rechazar numeros repetidos.
    Abrir una segunda orden dejaba la primera sin forma de seguirla ni de volver a
    informarla."""
    carpeta = str(tmp_path / "ordenes")
    ordenes = Ordenes(carpeta)
    primera = ordenes.abrir(_orden("1111", sierras=1, dientes=3))
    primera.anotar(estado="afilado", aprobado=True, origen="la memoria")
    ordenes.guardar()
    ordenes.abrir(_orden("2222", sierras=1, dientes=3))

    retomada = ordenes.retomar("1111")

    assert retomada is not None
    assert ordenes.abierta is retomada
    assert retomada.total_revisados == 1
    assert retomada.diente().numero == 2                # sigue donde habia quedado
    assert Ordenes(carpeta).abierta.numero == "1111"    # y se retoma al abrir el programa
    # Una orden CERRADA no se retoma: su informe ya se imprimio y se entrego.
    ordenes.cerrar()
    assert ordenes.retomar("1111") is None
    assert ordenes.retomar("9999") is None              # ni una que no existe


# ---------------------------------------------------------------------------
# El historial, la papelera, el zip del reclamo y la foto de prueba
# ---------------------------------------------------------------------------


def _en_disco(carpeta: Path, orden: Orden) -> Path:
    """Deja el .json de una orden en la carpeta sin pasar por abrir().

    Hace falta para poder elegir la fecha: abrir() guarda la orden con la fecha de hoy y
    estas pruebas necesitan justamente fechas de distintos meses y anios.
    """
    carpeta.mkdir(parents=True, exist_ok=True)
    archivo = carpeta / f"{sanear_numero(orden.numero)}.json"
    archivo.write_text(
        json.dumps(asdict(orden), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return archivo


def _con_fecha(numero: str, abierta: str, cerrada: str = "") -> Orden:
    orden = _orden(numero, sierras=1, dientes=3)
    orden.abierta = abierta
    orden.cerrada = cerrada
    return orden


# El .json de la orden 123 tal como esta hoy en el taller: 80 dientes, 7 anotados, los 7
# con el nombre pelado de la foto que entro a la coleccion y NINGUNO con prueba. Medido en
# C:/Users/WoodTools-02/Desktop/Afilado/data/ordenes/123.json (19.234 bytes): no tiene
# campo "version" --o sea que es la version 1-- ni campo "prueba" en ningun diente.
_FOTOS_DE_LA_123 = [
    "afilado_20261112_161511_098.jpg",
    "afilado_20261112_161529_715.jpg",
    "afilado_20261112_161544_128.jpg",
    "afilado_20261112_161605_184.jpg",
    "afilado_20261112_161616_684.jpg",
    "afilado_20261112_161626_660.jpg",
    "afilado_20261112_161644_580.jpg",
]


def _la_123_como_esta_en_el_taller() -> dict:
    return {
        "numero": "123",
        "nota_pedido": "126",
        "cliente": "5564",
        "abierta": "12/11/2026 16:14:21",
        "cerrada": "12/11/2026 16:16:52",
        "sierra_actual": 1,
        "sierras": [{
            "numero": 1,
            "dientes": 80,
            "identificacion": "",
            "revisados": [
                {
                    "numero": i + 1,
                    "estado": "afilado",
                    "parecido": None,
                    "aprobado": True,
                    "origen": "lo archivo el operario",
                    "instante": f"12/11/2026 16:15:{11 + i:02d}",
                    "foto": nombre,
                    "nota": "",
                }
                for i, nombre in enumerate(_FOTOS_DE_LA_123)
            ],
        }],
    }


def test_la_orden_de_enero_queda_arriba_aunque_el_texto_de_la_fecha_diga_lo_contrario(
    tmp_path: Path,
) -> None:
    """LA PRUEBA QUE MAS IMPORTA DEL HISTORIAL: el operario busca "la de ayer" arriba.

    Las dos formas en que esta pantalla ya salio mal estan medidas: listar() devuelve los
    nombres de archivo en orden alfabetico --['123', '1234', '213132', '5113546'], que no
    tiene nada que ver con cuando se trabajo cada orden-- y la fecha se guarda
    dia/mes/anio, asi que ordenada como texto "03/01/2027 10:00:00" queda ANTES de
    "12/11/2026 15:27:44". Las tres ordenes de aca abajo estan elegidas para que el orden
    alfabetico y el de la fecha como texto den los dos resultados equivocados.
    """
    carpeta = tmp_path / "ordenes"
    _en_disco(carpeta, _con_fecha("4001", "03/01/2027 10:00:00"))
    _en_disco(carpeta, _con_fecha("4002", "12/11/2026 15:27:44"))
    _en_disco(carpeta, _con_fecha("4003", "16/11/2026 09:34:20"))

    historial = Ordenes(str(carpeta)).historial()

    assert [fila.archivo for fila in historial] == ["4001", "4003", "4002"]
    assert historial[0].clave == "2027-01-03 10:00:00"
    # Las dos trampas, dichas en la prueba para que no vuelvan: por nombre de archivo la
    # 4002 saldria antes que la 4003, y por la fecha como texto la de enero saldria ultima.
    assert Ordenes(str(carpeta)).listar() == ["4001", "4002", "4003"]
    como_texto = sorted(
        (fila.abierta for fila in historial), reverse=True
    )
    assert como_texto[0] == "16/11/2026 09:34:20"       # ...y no la de 2027


def test_la_fecha_del_taller_se_ordena_bien_una_vez_dada_vuelta(tmp_path: Path) -> None:
    """Es la funcion que sostiene el orden del historial, y la que manana hace mecanica la
    migracion a la base de datos: si se equivoca, toda la pantalla se equivoca."""
    assert clave_de_fecha("16/11/2026 09:34:20") == "2026-11-16 09:34:20"
    assert clave_de_fecha("03/01/2027 10:00:00") > clave_de_fecha("12/11/2026 15:27:44")

    # Lo que no se entiende se ubica con el mtime del archivo, que no es la fecha de la
    # orden pero es lo unico medible que hay.
    archivo = tmp_path / "1234.json"
    archivo.write_text("{}", encoding="utf-8")
    cuando = archivo.stat().st_mtime
    assert clave_de_fecha("sin fecha", cuando) == datetime.fromtimestamp(
        cuando
    ).strftime("%Y-%m-%d %H:%M:%S")

    # Y si no hay ni eso se dice que no se sabe --"", que ordena primero y por lo tanto
    # queda ultima en el historial-- en vez de inventar un dia.
    assert clave_de_fecha("") == ""
    assert clave_de_fecha("12-11-2026") == ""


def test_una_orden_que_no_se_puede_leer_sale_igual_en_el_historial(tmp_path: Path) -> None:
    """Esconderla seria esconder justo la que hay que arreglar a mano, y es la pantalla
    desde donde se la manda a la papelera. Ademas el historial no puede caerse por un
    .json cortado: pasa con un corte de luz en la mitad de un guardado."""
    carpeta = tmp_path / "ordenes"
    _en_disco(carpeta, _con_fecha("4001", "16/11/2026 09:34:20"))
    (carpeta / "4002.json").write_text('{"numero": "4002", "sier', encoding="utf-8")

    historial = Ordenes(str(carpeta)).historial()

    assert len(historial) == 2
    rota = [fila for fila in historial if fila.archivo == "4002"][0]
    assert rota.ilegible
    # Los contadores en cero, no inventados: de ese archivo no se pudo leer nada.
    assert (rota.total_dientes, rota.revisados, rota.con_prueba) == (0, 0, 0)
    assert rota.numero == "4002"            # lo unico que se sabe es como se llama
    assert not [fila for fila in historial if fila.archivo == "4001"][0].ilegible


def test_el_historial_de_una_carpeta_sin_ordenes_no_frena_el_programa(tmp_path: Path) -> None:
    """La primera vez que se abre el programa en una PC nueva no hay ninguna orden, y esa
    pantalla tiene que poder abrirse igual para cargar la primera."""
    assert Ordenes(str(tmp_path / "ordenes")).historial() == []


def test_una_orden_vieja_tiene_nombres_de_foto_sueltos_y_ninguna_prueba_guardada(
    tmp_path: Path,
) -> None:
    """ESTA ES LA REGLA DE QUE INVENTAR UN DIENTE ES PEOR QUE PERDERLO, aplicada a la
    prueba: una orden de las que ya estan en el taller tiene siete dientes con el nombre
    pelado de una foto y NINGUNA prueba guardada. Medido en la 123 y en la 1234: las 14
    fotos que nombran estan todas en la papelera de la coleccion --el operario borro ese
    concepto el 13/11-- y ninguna viva. Mostrar esos nombres como prueba seria mostrar como
    prueba algo que no existe.
    """
    carpeta = tmp_path / "ordenes"
    carpeta.mkdir(parents=True)
    (carpeta / "123.json").write_text(
        json.dumps(_la_123_como_esta_en_el_taller()), encoding="utf-8"
    )

    fila = Ordenes(str(carpeta)).historial()[0]

    assert fila.revisados == 7
    assert fila.con_prueba == 0
    assert fila.con_nombre_suelto == 7
    # Y la orden se declara vieja: no promete pruebas que no puede tener.
    assert Ordenes(str(carpeta)).cargar("123").version == 1


def test_un_renglon_que_nombra_una_foto_que_ya_no_esta_no_cuenta_como_prueba(
    tmp_path: Path,
) -> None:
    """Un .json puede nombrar una prueba que despues alguien movio o borro de la carpeta.
    Contarla igual es exactamente lo que ya paso con las 14 fotos de las ordenes 123 y
    1234: el informe entregado las nombra y hoy no estan. La prueba se mira en el disco."""
    carpeta = tmp_path / "ordenes"
    orden = _orden("4001", sierras=1, dientes=2)
    orden.anotar(estado="afilado", aprobado=True, origen="la memoria",
                 prueba="pruebas/4001/s01_d001_20261116_093432_481.jpg")
    orden.anotar(estado="roto", aprobado=False, origen="la memoria",
                 prueba="pruebas/4001/s01_d002_20261116_093440_112.jpg")
    _en_disco(carpeta, orden)
    # Solo una de las dos fotos esta de verdad en el deposito.
    deposito = tmp_path / "pruebas" / "4001"
    deposito.mkdir(parents=True)
    (deposito / "s01_d001_20261116_093432_481.jpg").write_bytes(b"una foto")

    fila = Ordenes(str(carpeta)).historial()[0]

    assert fila.revisados == 2
    assert fila.con_prueba == 1              # la que esta, no las dos que se nombran
    assert fila.con_nombre_suelto == 0       # y ninguna de las dos es un nombre suelto


def test_borrar_la_orden_abierta_suelta_el_marcador_y_el_programa_vuelve_a_arrancar(
    tmp_path: Path,
) -> None:
    """Si el .abierta queda apuntando a un archivo que ya no esta, la proxima vez que el
    operario abre el programa arranca apuntando a una orden que no existe. Esto se prueba
    construyendo Ordenes() de nuevo, que es lo que hace el arranque de verdad."""
    carpeta = tmp_path / "ordenes"
    ordenes = Ordenes(str(carpeta))
    ordenes.abrir(_orden("4001", sierras=1, dientes=3))
    ordenes.abierta.anotar(estado="afilado", aprobado=True, origen="la memoria")
    ordenes.guardar()
    assert (carpeta / ".abierta").read_text(encoding="utf-8").strip() == "4001"

    quedo = ordenes.borrar("4001")

    assert quedo is not None and quedo.completo and quedo.destino.is_dir()
    assert ordenes.abierta is None
    assert not ordenes.existe("4001")
    de_nuevo = Ordenes(str(carpeta))          # el arranque del programa
    assert de_nuevo.abierta is None
    assert de_nuevo.historial() == []


def test_la_orden_que_se_mando_a_la_papelera_se_puede_sacar_de_vuelta_con_todo(
    tmp_path: Path,
) -> None:
    """NO SE BORRA DE VERDAD. Una orden es lo que contesta un reclamo: si el operario se
    equivoca de fila, tiene que haber de donde sacarla. Y se tiene que ir con los informes
    y con las fotos de prueba, que son lo que respalda cada diente."""
    import numpy as np

    carpeta = tmp_path / "ordenes"
    ordenes = Ordenes(str(carpeta))
    orden = ordenes.abrir(_orden("4001", sierras=1, dientes=3))
    prueba = ordenes.guardar_prueba("4001", 1, 1, np.zeros((480, 640, 3), np.uint8))
    orden.anotar(estado="afilado", aprobado=True, origen="la memoria", prueba=prueba)
    ordenes.guardar()
    ordenes.escribir_informe()
    guardado = (carpeta / "4001.json").read_text(encoding="utf-8")

    quedo = ordenes.borrar("4001")

    # De la carpeta de ordenes no quedo nada de esa orden...
    assert not (carpeta / "4001.json").exists()
    assert not (carpeta / "informe_4001.txt").exists()
    assert not (carpeta / "informe_4001.csv").exists()
    assert not ordenes.carpeta_pruebas("4001").exists()
    # ...pero esta todo en la papelera, con el mismo nombre y el mismo contenido.
    assert quedo is not None and quedo.completo
    assert (quedo.destino / "4001.json").read_text(encoding="utf-8") == guardado
    assert (quedo.destino / "informe_4001.txt").is_file()
    assert (quedo.destino / "informe_4001.csv").is_file()
    # Y la foto, en la MISMA ruta que el .json le da --"pruebas/4001/..."--, no en una
    # carpeta pruebas pelada: asi el rescate es copiar esa carpeta de vuelta a data y nada
    # mas. Tiene su prueba aparte, mas abajo.
    assert (quedo.destino / prueba).is_file()
    assert quedo.movidos == [
        f"pruebas/{Path(prueba).parent.name}", "informe_4001.txt", "informe_4001.csv",
        "4001.json",
    ]
    # Y recuperarla es copiar el .json de vuelta: la orden vuelve entera.
    (carpeta / "4001.json").write_text(guardado, encoding="utf-8")
    recuperada = Ordenes(str(carpeta)).cargar("4001")
    assert recuperada is not None and recuperada.total_revisados == 1


def test_el_zip_del_reclamo_lleva_el_json_el_informe_y_las_fotos(tmp_path: Path) -> None:
    """Es UN SOLO archivo porque es lo que el operario copia a un pendrive y manda por mail
    cuando entra un reclamo: si va a medias --el informe sin las fotos, o las fotos sin el
    informe-- del otro lado no se puede contestar nada."""
    import numpy as np

    carpeta = tmp_path / "ordenes"
    ordenes = Ordenes(str(carpeta))
    orden = ordenes.abrir(_orden("4001", sierras=1, dientes=2))
    prueba = ordenes.guardar_prueba("4001", 1, 1, np.zeros((480, 640, 3), np.uint8))
    orden.anotar(estado="roto", parecido=0.4, aprobado=False, origen="la memoria",
                 prueba=prueba)
    ordenes.guardar()

    zip_final = ordenes.exportar("4001")

    assert zip_final is not None and zip_final.is_file()
    with zipfile.ZipFile(zip_final) as paquete:
        adentro = sorted(paquete.namelist())
        # La foto va en la ruta que el .json le da, tal cual: "pruebas/<orden>/<foto>".
        assert adentro == sorted(["orden.json", "informe.txt", "informe.csv", prueba])
        # El informe se arma al vuelo desde el .json, asi que dice lo ultimo que se anoto.
        informe = paquete.read("informe.txt").decode("utf-8")
        assert "RECHAZADO" in informe and "4001" in informe
        # Y la foto viaja entera, no como un renglon que la nombra.
        assert len(paquete.read(prueba)) == (tmp_path / prueba).stat().st_size
        assert json.loads(paquete.read("orden.json"))["numero"] == "4001"


def test_no_se_exporta_una_orden_que_no_se_pudo_leer(tmp_path: Path) -> None:
    """No se muestra como prueba algo que el programa no pudo leer: un zip armado a partir
    de un .json cortado seria un reclamo contestado con la mitad de los dientes."""
    carpeta = tmp_path / "ordenes"
    carpeta.mkdir(parents=True)
    (carpeta / "4001.json").write_text('{"numero": "4001", "sier', encoding="utf-8")
    ordenes = Ordenes(str(carpeta))

    assert ordenes.exportar("4001") is None
    assert ordenes.exportar("9999") is None          # ni una que no existe
    # Y no queda un zip a medio armar que alguien pueda mandar creyendo que es el bueno.
    assert not list(carpeta.rglob("*.zip"))
    assert not list(carpeta.rglob("*.tmp"))
    # Pero la fila sigue estando en el historial: de ahi se la manda a la papelera.
    assert ordenes.historial()[0].ilegible


def test_la_foto_de_prueba_queda_en_el_disco_donde_dice_el_renglon(tmp_path: Path) -> None:
    """Un renglon que nombra una foto que no esta es una prueba que no existe. Medido en la
    orden de verdad 5113546: 15 dientes anotados y CERO con foto, los 15 por el camino de
    la captura automatica."""
    import numpy as np

    ordenes = Ordenes(str(tmp_path / "ordenes"))
    cuadro = np.zeros((480, 640, 3), np.uint8)      # el tamano que da el microscopio

    ruta = ordenes.guardar_prueba("5113546", 1, 7, cuadro)

    assert ruta is not None
    # Relativa y con barras normales: la carpeta Afilado se copia de una PC a otra.
    assert not Path(ruta).is_absolute() and "\\" not in ruta
    assert ruta.startswith("pruebas/5113546/")
    assert re.fullmatch(r"s01_d007_\d{8}_\d{6}_\d{3}\.jpg", Path(ruta).name)
    # Y la foto ESTA, leida desde la carpeta data igual que la va a leer el informe.
    archivo = tmp_path / ruta
    assert archivo.is_file() and archivo.stat().st_size > 0
    # La prueba del diente de al lado no pisa a la anterior: las dos son prueba de algo.
    otra = ordenes.guardar_prueba("5113546", 1, 8, cuadro)
    assert otra != ruta and (tmp_path / otra).is_file()
    # Sin imagen no se inventa un archivo vacio: se dice que no se pudo.
    assert ordenes.guardar_prueba("5113546", 1, 9, None) is None


def test_el_deposito_de_pruebas_no_cae_adentro_de_la_coleccion(tmp_path: Path) -> None:
    """Medido antes de decidirlo: si la prueba entrara a la coleccion, archivar() --que
    llama a refrescar()-- pasa de 209 ms con 15 fotos a 3.153 ms con 320, y 320 fotos es
    UNA orden de 4 sierras de 80 dientes, contra 1.200 ms que el operario apoya el diente.
    Por eso la prueba va a un deposito propio, al lado de las ordenes."""
    import numpy as np

    datos = tmp_path / "data"
    ordenes = Ordenes(str(datos / "ordenes"))

    deposito = ordenes.carpeta_pruebas("5113546")

    assert deposito == datos / "pruebas" / "5113546"
    assert ordenes.carpeta not in deposito.parents     # ni adentro de la carpeta ordenes
    assert "coleccion" not in str(deposito)
    # El numero se sanea para el nombre de la carpeta, igual que para el del .json.
    assert ordenes.carpeta_pruebas("OA/123").name == "OA_123"
    # Y lo que se guarda no deja ningun .jpg en la carpeta que mira la autoevaluacion.
    ordenes.guardar_prueba("5113546", 1, 1, np.zeros((480, 640, 3), np.uint8))
    assert not list(ordenes.carpeta.rglob("*.jpg"))
    assert len(list(deposito.glob("*.jpg"))) == 1


def test_una_orden_con_barra_en_el_numero_se_exporta_y_se_borra_igual(tmp_path: Path) -> None:
    """El numero que se teclea y el nombre del archivo no son lo mismo: "OA/123" se guarda
    en OA_123.json. El historial muestra el numero tecleado y el nombre del archivo, y los
    botones pueden llegar con cualquiera de los dos: con uno solo, el operario aprieta
    EXPORTAR y no pasa nada."""
    carpeta = tmp_path / "ordenes"
    ordenes = Ordenes(str(carpeta))
    ordenes.abrir(_orden("OA/123", sierras=1, dientes=2))

    fila = ordenes.historial()[0]
    assert (fila.numero, fila.archivo) == ("OA/123", "OA_123")

    como_lo_tecleo = ordenes.exportar("OA/123")
    como_se_llama_el_archivo = ordenes.exportar("OA_123")
    assert como_lo_tecleo is not None and como_se_llama_el_archivo is not None
    # Los dos caen en el mismo archivo: el zip se nombra con el numero saneado y un sello
    # de un segundo, asi que exportar dos veces la misma orden dentro del mismo segundo
    # reescribe el mismo zip. Es la misma orden y el mismo contenido, pero queda dicho.
    assert como_lo_tecleo.name.startswith("orden_OA_123_")
    assert como_se_llama_el_archivo.is_file()
    with zipfile.ZipFile(como_lo_tecleo) as paquete:
        # Adentro va el numero tal cual se tecleo: es el que el cliente tiene en el papel.
        assert json.loads(paquete.read("orden.json"))["numero"] == "OA/123"

    assert ordenes.borrar("OA_123") is not None
    assert not ordenes.existe("OA/123")
    # Y por el otro nombre tambien, con la orden vuelta a abrir.
    ordenes.abrir(_orden("OA/123", sierras=1, dientes=2))
    assert ordenes.borrar("OA/123") is not None
    assert not (carpeta / "OA_123.json").exists()


# ---------------------------------------------------------------------------
# El guardado: el borrado a medias, la ruta de la foto, la planilla y la version
# ---------------------------------------------------------------------------


def _con_pruebas(
    ordenes: Ordenes, numero: str, cuantas: int, dientes: int = 3
) -> tuple[Orden, list[str]]:
    """Una orden abierta con `cuantas` fotos de prueba guardadas DE VERDAD en el deposito.

    Pasa por guardar_prueba() y no escribe los .jpg a mano a proposito: lo que estas pruebas
    miran es justamente que la ruta que queda en el .json y el archivo que queda en el disco
    sean el mismo, y escribirlos aparte seria probar contra una ruta inventada por la prueba.
    """
    import numpy as np

    orden = ordenes.abrir(_orden(numero, sierras=1, dientes=dientes))
    rutas: list[str] = []
    for diente in range(1, cuantas + 1):
        # 480x640 es el cuadro que da el microscopio del taller.
        ruta = ordenes.guardar_prueba(numero, 1, diente, np.zeros((480, 640, 3), np.uint8))
        assert ruta is not None
        orden.anotar(estado="roto", parecido=0.4, aprobado=False, origen="la memoria",
                     prueba=ruta)
        rutas.append(ruta)
    ordenes.guardar()
    return orden, rutas


def test_un_borrado_trabado_en_la_foto_deja_la_orden_entera_y_visible(
    tmp_path: Path, monkeypatch
) -> None:
    """SIN ESTO, LA ORDEN QUE NO SE PUDO BORRAR ES LA QUE DESAPARECE DEL TALLER.

    El .json se movia primero --"es el unico que no se puede volver a generar"-- y si
    despues se trababa una foto, que es lo que pasa cuando el operario la tiene abierta en
    el visor de Windows, quedaba lo peor de los dos mundos: la orden ya no figuraba en el
    historial --la unica pantalla desde donde se la exporta-- y el cartel le decia al
    operario que no se habia borrado. La orden del cliente, y el operario buscandola donde
    ya no estaba.

    Ahora el .json es lo ULTIMO que se mueve, asi que un borrado que se traba deja la orden
    entera y en el historial, y ahi el None es la verdad: no se movio nada.
    """
    carpeta = tmp_path / "ordenes"
    ordenes = Ordenes(str(carpeta))
    _orden_viva, rutas = _con_pruebas(ordenes, "4001", 1)
    ordenes.escribir_informe()
    mover = shutil.move

    def se_traba_la_foto(origen, destino):
        """La carpeta de fotos no se deja mover: la foto esta abierta en el visor."""
        if Path(origen).is_dir():
            raise OSError(13, "Permission denied")
        return mover(origen, destino)

    monkeypatch.setattr(shutil, "move", se_traba_la_foto)

    assert ordenes.borrar("4001") is None
    # La orden sigue ENTERA donde estaba, y sobre todo sigue a la vista y exportable.
    assert (carpeta / "4001.json").is_file()
    assert (carpeta / "informe_4001.txt").is_file()
    assert (carpeta / "informe_4001.csv").is_file()
    assert (tmp_path / rutas[0]).is_file()
    assert [fila.archivo for fila in ordenes.historial()] == ["4001"]
    assert ordenes.exportar("4001") is not None
    # Y el operario no perdio la orden que estaba trabajando por un borrado que no paso.
    assert ordenes.abierta is not None and ordenes.abierta.numero == "4001"
    # Tampoco queda una carpeta de papelera vacia por cada intento: si no, en la carpeta
    # del taller se van juntando y el que entra a mirar que paso no entiende cual es cual.
    assert list((carpeta / "_papelera").glob("*")) == []


def test_un_borrado_que_dejo_algo_en_la_papelera_no_dice_que_no_se_borro(
    tmp_path: Path, monkeypatch
) -> None:
    """DEVOLVER None ES DECIR "NO SE BORRO NADA", Y ESO HAY QUE PREGUNTARLO AL DISCO.

    Mover una carpeta no es un solo paso: cuando no se la puede renombrar, shutil.move copia
    archivo por archivo, asi que puede dejar unas fotos del otro lado y trabarse en la que
    sigue. Si ahi se da por hecho que no se movio nada, al operario se le dice que la orden
    sigue completa mientras parte de sus fotos ya estan en _papelera, y nadie le dice donde
    mirar. Por eso, antes de decidir, se le pregunta al disco si llego algo.
    """
    carpeta = tmp_path / "ordenes"
    ordenes = Ordenes(str(carpeta))
    _orden_viva, rutas = _con_pruebas(ordenes, "4001", 2)
    mover = shutil.move

    def copia_a_medias(origen, destino):
        """Lo que hace shutil.move cuando no puede renombrar la carpeta: copia la primera
        foto y se traba en la siguiente."""
        fuente = Path(origen)
        if fuente.is_dir():
            Path(destino).mkdir(parents=True, exist_ok=True)
            primera = sorted(fuente.iterdir())[0]
            shutil.copy2(primera, Path(destino) / primera.name)
            raise OSError(13, "Permission denied")
        return mover(origen, destino)

    monkeypatch.setattr(shutil, "move", copia_a_medias)

    quedo = ordenes.borrar("4001")

    assert quedo is not None                 # algo se fue: decir que no seria mentira
    assert not quedo.completo
    assert quedo.falto == "pruebas/4001"     # lo que se trabo, con nombre
    assert quedo.movidos == []               # ni una pieza COMPLETA, y aun asi hay algo ahi
    # Lo que llego, en la carpeta que el aviso nombra: es donde el operario lo va a buscar.
    assert len(list((quedo.destino / "pruebas" / "4001").glob("*.jpg"))) == 1
    # Y el .json ni se intento mover, asi que la orden sigue a la vista y se puede exportar.
    assert (carpeta / "4001.json").is_file()
    assert [fila.archivo for fila in ordenes.historial()] == ["4001"]


def test_el_json_se_va_ultimo_y_el_marcador_de_la_orden_abierta_con_el(
    tmp_path: Path, monkeypatch
) -> None:
    """EL ORDEN DE LOS MOVIMIENTOS ES EL ARREGLO, y son dos cosas en el mismo renglon.

    Primero las fotos, despues los informes, y el .json al final: lo que se puede volver a
    generar se va antes que lo que no, para que un borrado trabado nunca deje la orden fuera
    del historial. Y el marcador .abierta se suelta JUSTO ANTES del .json: soltandolo al
    principio, un borrado que se traba le saca al operario la orden que estaba anotando
    --deja de poder cargar dientes-- por una orden que al final sigue en el disco.
    """
    carpeta = tmp_path / "ordenes"
    ordenes = Ordenes(str(carpeta))
    _orden_viva, _rutas = _con_pruebas(ordenes, "4001", 1)
    ordenes.escribir_informe()
    mover = shutil.move

    def se_traba_la_planilla(origen, destino):
        if Path(origen).name == "informe_4001.csv":
            raise OSError(13, "Permission denied")
        return mover(origen, destino)

    monkeypatch.setattr(shutil, "move", se_traba_la_planilla)

    quedo = ordenes.borrar("4001")

    assert quedo is not None and not quedo.completo
    # Se trabo en la planilla, o sea DESPUES de las fotos y del .txt y ANTES del .json.
    assert quedo.movidos == ["pruebas/4001", "informe_4001.txt"]
    assert quedo.falto == "informe_4001.csv"
    assert (carpeta / "4001.json").is_file()
    assert [fila.archivo for fila in ordenes.historial()] == ["4001"]
    # Y la orden abierta sigue abierta, en memoria y en el marcador del disco.
    assert ordenes.abierta is not None and ordenes.abierta.numero == "4001"
    assert (carpeta / ".abierta").read_text(encoding="utf-8").strip() == "4001"


def test_la_papelera_guarda_las_fotos_con_la_carpeta_de_la_orden_adentro(
    tmp_path: Path,
) -> None:
    """EL RESCATE TIENE QUE SALIR CON EL EXPLORADOR DE WINDOWS, que es lo que hay en el
    taller: copiar la carpeta de vuelta a data y listo.

    Sin el nivel pruebas/<orden>/, las fotos caen un escalon arriba de donde el .json las
    nombra, y despues del rescate la prueba esta en el disco y no se encuentra: ninguna de
    las rutas del .json resuelve y nada avisa que falta un nivel de carpeta. Un reclamo
    contestado sin las fotos que estaban ahi al lado.
    """
    carpeta = tmp_path / "ordenes"
    ordenes = Ordenes(str(carpeta))
    _orden_viva, rutas = _con_pruebas(ordenes, "4001", 2)

    quedo = ordenes.borrar("4001")

    assert quedo is not None and quedo.completo
    # En la papelera, cada foto esta en la MISMA ruta relativa que el .json le da.
    for ruta in rutas:
        assert (quedo.destino / ruta).is_file()
    # El rescate: la carpeta pruebas de la papelera, copiada de vuelta a data, y el .json
    # de vuelta al lado de las ordenes. dirs_exist_ok porque data/pruebas sigue estando:
    # lo que se movio fue la carpeta de la orden, no el deposito entero.
    shutil.copytree(quedo.destino / "pruebas", tmp_path / "pruebas", dirs_exist_ok=True)
    (carpeta / "4001.json").write_text(
        (quedo.destino / "4001.json").read_text(encoding="utf-8"), encoding="utf-8"
    )

    de_nuevo = Ordenes(str(carpeta))
    recuperada = de_nuevo.cargar("4001")
    assert recuperada is not None
    assert de_nuevo.pruebas_en_disco(recuperada) == set(rutas)
    fila = de_nuevo.historial()[0]
    assert (fila.con_prueba, fila.prueba_perdida) == (2, 0)


def test_el_zip_pone_cada_foto_en_la_ruta_que_el_json_le_da(tmp_path: Path) -> None:
    """EL PAQUETE TIENE QUE CERRAR SOLO, porque es lo unico que el otro lado va a tener.

    Cada renglon del informe nombra "pruebas/<orden>/<foto>.jpg", que es lo que dice el
    .json. Antes las fotos viajaban en una carpeta "pruebas/" pelada, asi que ninguna de
    esas rutas resolvia adentro del paquete: el que recibe el reclamo busca la foto del
    diente donde el papel dice, no la encuentra, y concluye que las pruebas no vinieron.
    Medido con una orden de 3 pruebas: 3 dientes diciendo NO ESTA y 0 de 3 rutas resolviendo,
    con las 3 fotos adentro del zip.
    """
    carpeta = tmp_path / "ordenes"
    ordenes = Ordenes(str(carpeta))
    _orden_viva, rutas = _con_pruebas(ordenes, "4001", 2)

    zip_final = ordenes.exportar("4001")

    assert zip_final is not None
    with zipfile.ZipFile(zip_final) as paquete:
        adentro = set(paquete.namelist())
        informe = paquete.read("informe.txt").decode("utf-8")
        guardado = json.loads(paquete.read("orden.json"))
    # Las rutas del .json y las del paquete son las mismas, no dos listas parecidas.
    nombradas = [
        d["prueba"] for d in guardado["sierras"][0]["revisados"] if d.get("prueba")
    ]
    assert nombradas == rutas
    assert set(nombradas) <= adentro
    # Y el informe de adentro las cuenta como encontradas: ningun renglon de diente queda
    # con el aviso de que falta la foto.
    assert "Fotos de prueba  : 2 nombradas, 2 encontradas" in informe
    assert [l for l in informe.splitlines() if "(NO ESTA)" in l and "prueba:" in l] == []


def test_el_informe_del_zip_cuenta_las_fotos_que_entraron_al_paquete(tmp_path: Path) -> None:
    """EL INFORME QUE VIAJA CUENTA CONTRA EL PAQUETE, NO CONTRA EL DISCO DE ESTA PC.

    No son lo mismo, y el caso no es raro: "4001 - copia.json" es lo que deja Windows al
    copiar una orden, y esa copia nombra las fotos de la orden original, que estan en el
    disco de aca pero NO en el paquete de la copia. Contando contra el disco, el informe del
    reclamo dice "1 encontradas" de una foto que el que abre el zip no tiene en ninguna
    parte: eso es mostrar como prueba algo que no esta.
    """
    carpeta = tmp_path / "ordenes"
    ordenes = Ordenes(str(carpeta))
    _orden_viva, rutas = _con_pruebas(ordenes, "4001", 1)
    shutil.copy(carpeta / "4001.json", carpeta / "4001 - copia.json")

    paquete_copia = ordenes.exportar("4001", archivo="4001 - copia")

    assert paquete_copia is not None
    with zipfile.ZipFile(paquete_copia) as paquete:
        assert [n for n in paquete.namelist() if n.startswith("pruebas/")] == []
        informe = paquete.read("informe.txt").decode("utf-8")
    assert "Fotos de prueba  : 1 nombradas, 0 encontradas   -   FALTAN 1" in informe
    assert f"prueba: {rutas[0]}   (NO ESTA)" in informe
    # Y la foto SIGUE en el disco de esta PC: es justo lo que hacia mentir al informe.
    assert (tmp_path / rutas[0]).is_file()


def test_la_planilla_no_hace_pasar_la_foto_de_entrenamiento_por_la_prueba() -> None:
    """LA COLUMNA SE LLAMABA "foto" Y NO HABIA NINGUNA CON LA PRUEBA.

    El que abre el .csv de un reclamo leia esos nombres como la foto del diente, y no lo
    son: es la foto que entro a la coleccion para que el programa aprenda, sin carpeta --ni
    el programa la encuentra-- y puede estar borrada. Medido en la orden de verdad 123: 7
    filas con foto=afilado_20261112_161511_098.jpg y parecidas, y las 7 estan en
    coleccion/_papelera desde que el operario borro ese concepto el 13/11. Y al reves: en
    una orden con pruebas de verdad guardadas, la unica columna que hablaba de fotos salia
    VACIA en todas las filas.
    """
    orden = _orden(sierras=1, dientes=2)
    orden.anotar(estado="afilado", aprobado=True, origen="lo archivo el operario",
                 foto="afilado_20261112_161511_098.jpg")
    orden.anotar(estado="roto", aprobado=False, origen="la memoria",
                 prueba="pruebas/1234/s01_d002_20261116_093440_112.jpg",
                 nota="el filo esta redondeado")

    cabecera, entrenamiento, con_prueba = orden.filas_csv()

    assert cabecera == [
        "orden", "nota_pedido", "cliente", "sierra", "identificacion", "diente",
        "estado", "resultado", "parecido", "origen", "cuando",
        "foto_de_entrenamiento", "prueba", "nota",
    ]
    # Las dos columnas dicen cosas distintas, y cada diente llena la suya.
    assert entrenamiento[11] == "afilado_20261112_161511_098.jpg"
    assert entrenamiento[12] == ""
    assert con_prueba[11] == ""
    assert con_prueba[12] == "pruebas/1234/s01_d002_20261116_093440_112.jpg"
    # Las columnas de antes no se corrieron: la que se movio es la nota, del 12 al 13.
    assert cabecera[:11] == [
        "orden", "nota_pedido", "cliente", "sierra", "identificacion", "diente",
        "estado", "resultado", "parecido", "origen", "cuando",
    ]
    assert con_prueba[13] == "el filo esta redondeado"


def test_el_informe_dice_cuantas_pruebas_nombra_y_cuantas_se_encontraron() -> None:
    """LOS DOS NUMEROS, SIEMPRE, porque no son el mismo.

    Un .json nombra fotos y las fotos pueden no estar: es exactamente lo que ya paso con las
    ordenes 123 y 1234 del taller, que entre las dos nombran 14 y hoy no hay ninguna viva.
    Un informe que diga "2 fotos de prueba" sin decir que una no esta es el papel de un
    reclamo prometiendo una prueba que no se puede mostrar. Y donde falta, se dice en el
    renglon del diente, que es donde alguien va a buscarla.
    """
    orden = _orden(sierras=1, dientes=2)
    orden.anotar(estado="roto", aprobado=False, origen="la memoria",
                 prueba="pruebas/1234/s01_d001_20261116_093432_481.jpg")
    orden.anotar(estado="roto", aprobado=False, origen="la memoria",
                 prueba="pruebas/1234/s01_d002_20261116_093440_112.jpg")

    papel = orden.informe({"pruebas/1234/s01_d001_20261116_093432_481.jpg"})

    assert "Fotos de prueba  : 2 nombradas, 1 encontradas   -   FALTAN 1" in papel
    # La que esta se nombra pelada; la que no esta, con el aviso al lado del diente.
    assert "prueba: pruebas/1234/s01_d001_20261116_093432_481.jpg" in papel
    assert "s01_d001_20261116_093432_481.jpg   (NO ESTA)" not in papel
    assert "prueba: pruebas/1234/s01_d002_20261116_093440_112.jpg   (NO ESTA)" in papel
    # Al pie de la sierra otra vez, porque un reclamo entra por UNA sierra.
    assert "fotos de prueba de esta sierra: 2 nombradas, 1 encontradas   -   FALTAN 1" in papel
    # Y si nadie fue a buscarlas, se dice ESO: no se las da por presentes, pero tampoco se
    # declara que falten --nadie las fue a ver-- asi que ningun diente lleva el aviso.
    sin_buscar = orden.informe()
    assert "Fotos de prueba  : 2 nombradas, no se fue a buscar si estan" in sin_buscar
    assert [l for l in sin_buscar.splitlines() if "(NO ESTA)" in l and "prueba:" in l] == []


def test_el_informe_de_una_orden_sin_pruebas_lo_dice_en_renglon_aparte(tmp_path: Path) -> None:
    """ESTA ES LA REGLA DE QUE INVENTAR UN DIENTE ES PEOR QUE PERDERLO, en el papel.

    La 123 tal como esta en el taller tiene 7 dientes con el nombre pelado de una foto de
    entrenamiento y NINGUNA prueba. El informe que salia nombraba esos 7 archivos y no decia
    en ninguna parte que no eran la prueba ni que no estaban: 7 renglones que parecen
    respaldados abajo de dientes que el papel da por revisados. Ahora lo dice en renglon
    aparte, antes de que se lea un solo veredicto, y el bloque de como leer el informe
    aclara que la columna de la planilla tampoco es la prueba.
    """
    carpeta = tmp_path / "ordenes"
    carpeta.mkdir(parents=True)
    (carpeta / "123.json").write_text(
        json.dumps(_la_123_como_esta_en_el_taller()), encoding="utf-8"
    )
    ordenes = Ordenes(str(carpeta))
    orden = ordenes.cargar("123")

    papel = orden.informe(ordenes.pruebas_en_disco(orden))

    assert "ESTA ORDEN NO TIENE NINGUNA FOTO DE PRUEBA" in papel
    assert "Fotos de prueba  : ninguna" in papel
    # Y ninguno de los 7 nombres de entrenamiento figura como prueba de nada: ningun
    # renglon de diente nombra una prueba, y los 7 archivos no aparecen en el papel.
    assert [l for l in papel.splitlines() if l.strip().startswith("prueba:")] == []
    for nombre in _FOTOS_DE_LA_123:
        assert nombre not in papel
    assert "'foto_de_entrenamiento' DE LA PLANILLA: eso NO es la" in papel


def test_al_guardar_queda_escrita_la_version_que_escribio_el_archivo(tmp_path: Path) -> None:
    """LA VERSION LA SUBE EL QUE ESCRIBE, y antes no la subia nadie.

    Medido con la orden de verdad 123: no trae campo version --o sea que es la 1-- y despues
    de anotarle un diente con prueba de verdad el .json quedaba diciendo version 1 Y con la
    prueba adentro. El dia que la migracion a la base de datos use la version para saber si
    una orden puede tener pruebas, con esa mentira adentro se le pasan de largo justo las
    pruebas de las ordenes viejas que se siguieron trabajando.

    Y version_origen NO se toca nunca: es lo unico que queda de donde salio la orden. Las 4
    ordenes del taller no traen ninguno de los dos campos, asi que nacen en 1 y en 0.
    """
    import numpy as np

    carpeta = tmp_path / "ordenes"
    carpeta.mkdir(parents=True)
    del_taller = _la_123_como_esta_en_el_taller()
    del_taller["cerrada"] = ""                  # como la 5113546, que sigue abierta
    assert "version" not in del_taller and "version_origen" not in del_taller
    (carpeta / "123.json").write_text(json.dumps(del_taller), encoding="utf-8")
    ordenes = Ordenes(str(carpeta))
    leida = ordenes.cargar("123")
    assert (leida.version, leida.version_origen) == (1, 0)

    # El operario la retoma y le anota el diente 8, el primero sin anotar de los 80, con su
    # foto de prueba: el camino por el que una orden vieja se vuelve a escribir.
    retomada = ordenes.retomar("123")
    ruta = ordenes.guardar_prueba("123", 1, 8, np.zeros((480, 640, 3), np.uint8))
    retomada.anotar(estado="roto", aprobado=False, origen="la memoria", prueba=ruta)
    ordenes.guardar()

    en_disco = json.loads((carpeta / "123.json").read_text(encoding="utf-8"))
    assert en_disco["version"] == VERSION_FORMATO
    assert en_disco["version_origen"] == 0      # sigue siendo una orden de antes
    # Y la prueba esta adentro: la version que se escribio es la que puede guardarla.
    assert any(
        d.get("prueba") == ruta for d in en_disco["sierras"][0]["revisados"]
    )
    # Una orden nacida hoy declara las dos cosas: con que se escribio y de donde viene.
    nueva = Orden.nueva("9001", 1, 2)
    assert (nueva.version, nueva.version_origen) == (VERSION_FORMATO, VERSION_FORMATO)


def test_la_prueba_que_se_perdio_no_se_cuenta_igual_que_la_que_nunca_existio(
    tmp_path: Path,
) -> None:
    """ERAN DOS FILAS IDENTICAS EN EL HISTORIAL Y SON DOS COSAS DISTINTAS.

    Una orden que nunca tuvo prueba y una orden cuyos dientes nombran fotos que ya no estan
    daban las dos con_prueba=0 y con_nombre_suelto=0: la prueba perdida no caia en ninguna
    columna, y de ahi salia un zip sin fotos sin que nada lo avisara. Medido con 3 fotos de
    verdad del microscopio guardadas como prueba (291.502 bytes en el deposito): con las
    fotos la fila dice con_prueba=3, y borrando el deposito --una limpieza de disco, o la
    carpeta Afilado copiada a otra PC sin los MiB de fotos-- decia exactamente lo mismo que
    la orden que nunca tuvo ninguna.
    """
    carpeta = tmp_path / "ordenes"
    ordenes = Ordenes(str(carpeta))
    ordenes.abrir(_orden("9002", sierras=1, dientes=3))
    ordenes.abierta.anotar(estado="afilado", aprobado=True, origen="la memoria")
    ordenes.guardar()
    _orden_viva, rutas = _con_pruebas(ordenes, "9001", 3)

    por_archivo = {fila.archivo: fila for fila in ordenes.historial()}
    assert (por_archivo["9001"].con_prueba, por_archivo["9001"].prueba_perdida) == (3, 0)
    assert (por_archivo["9002"].con_prueba, por_archivo["9002"].prueba_perdida) == (0, 0)

    # La carpeta Afilado copiada a otra PC sin las fotos, o una limpieza de disco.
    shutil.rmtree(ordenes.carpeta_pruebas("9001"))

    por_archivo = {fila.archivo: fila for fila in Ordenes(str(carpeta)).historial()}
    perdida, sin_prueba = por_archivo["9001"], por_archivo["9002"]
    assert (perdida.con_prueba, perdida.prueba_perdida) == (0, 3)
    assert (sin_prueba.con_prueba, sin_prueba.prueba_perdida) == (0, 0)
    # Lo que importa: las dos filas se pueden distinguir, y los dientes siguen contados.
    assert perdida.prueba_perdida != sin_prueba.prueba_perdida
    assert (perdida.revisados, sin_prueba.revisados) == (3, 1)
    # Un nombre de foto de entrenamiento no es ninguna de las dos cosas y no se mezcla.
    assert perdida.con_nombre_suelto == 0


def test_la_fecha_que_salio_del_archivo_se_declara_estimada(tmp_path: Path) -> None:
    """LA FECHA DEL ARCHIVO NO ES LA FECHA DE LA ORDEN: se corre sola cada vez que alguien
    copia la carpeta Afilado a otra PC.

    Medido con la 123 del taller a la que se le vacio el campo abierta: su clave paso a ser
    la del archivo y la fila se fue ARRIBA del historial, haciendo pasar el mtime por la
    orden mas nueva del taller. Mezclar las dos en la misma columna sin decirlo es mostrar
    una fecha que no es la fecha, y la pantalla no tiene con que aclararlo si la fila no lo
    trae.
    """
    carpeta = tmp_path / "ordenes"
    _en_disco(carpeta, _con_fecha("4001", "16/11/2026 09:34:20"))
    _en_disco(carpeta, _con_fecha("4002", ""))              # sin fecha de orden
    (carpeta / "4003.json").write_text('{"numero": "4003", "sier', encoding="utf-8")

    por_archivo = {fila.archivo: fila for fila in Ordenes(str(carpeta)).historial()}

    assert por_archivo["4001"].clave == "2026-11-16 09:34:20"
    assert not por_archivo["4001"].clave_estimada
    # La que no tiene fecha se ubica con el mtime, que es lo unico medible, y lo declara.
    assert por_archivo["4002"].clave and por_archivo["4002"].clave_estimada
    # Y la ilegible tambien: de un .json cortado no sale ninguna fecha de orden.
    assert por_archivo["4003"].ilegible and por_archivo["4003"].clave_estimada


def test_borrar_la_copia_hecha_a_mano_no_se_lleva_la_orden_del_cliente(tmp_path: Path) -> None:
    """"123 - copia.json" ES LO QUE DEJA WINDOWS AL COPIAR UNA ORDEN, y esa copia la hace el
    operario con el Explorador, no porque el programa se lo pida: leida la pantalla de
    borrado (panel.py, _dibujar_borrar_orden y _renglones_de_la_prueba), lo que dice es de
    que cliente es la orden, cuando se abrio, cuantos dientes tiene, que prueba se lleva, si
    es la orden abierta, y que no se pierde porque va a data/ordenes/_papelera y de ahi se
    saca con el Explorador. NO le sugiere copiar nada --antes este comentario decia que si,
    y nadie lo habia ido a mirar--.

    El historial manda a borrar el numero que esta ADENTRO del .json --"123"-- y aca se lo
    volvia a sanear, lo que apunta a 123.json: la orden del cliente. Medido con la copia al
    lado de la orden: la fila de la copia dice archivo='123 - copia' y numero='123', asi que
    borrar esa fila se llevaba la orden de verdad con sus fotos y dejaba la copia intacta.
    """
    carpeta = tmp_path / "ordenes"
    ordenes = Ordenes(str(carpeta))
    _orden_viva, rutas = _con_pruebas(ordenes, "123", 1)
    ordenes.escribir_informe()
    shutil.copy(carpeta / "123.json", carpeta / "123 - copia.json")

    copia = [fila for fila in ordenes.historial() if fila.archivo == "123 - copia"][0]
    assert (copia.numero, copia.archivo) == ("123", "123 - copia")

    quedo = ordenes.borrar(copia.numero, archivo=copia.archivo)

    assert quedo is not None and quedo.completo
    assert quedo.movidos == ["123 - copia.json"]
    assert not (carpeta / "123 - copia.json").exists()
    # La orden del cliente quedo entera: el .json, el informe y su foto de prueba.
    assert (carpeta / "123.json").is_file()
    assert (carpeta / "informe_123.txt").is_file()
    assert (tmp_path / rutas[0]).is_file()
    assert [fila.archivo for fila in ordenes.historial()] == ["123"]
    assert ordenes.cargar("123").total_revisados == 1
    # Y cada una se lee por su archivo, no por el numero que las dos comparten.
    shutil.copy(carpeta / "123.json", carpeta / "123 - copia.json")
    assert ordenes.cargar("123", archivo="123 - copia") is not None


def test_una_ruta_disfrazada_de_nombre_de_archivo_no_borra_ni_exporta_nada(
    tmp_path: Path,
) -> None:
    """`archivo` NO SE SANEA --el nombre salio del disco, no de un teclado-- Y POR ESO SE
    MIRA QUE SEA UN NOMBRE.

    Un texto que no se sanea entra tal cual al armar la ruta, y por ahi no puede entrar una
    ruta: reproducido aca, con un 4001.json un nivel arriba de la carpeta de ordenes,
    borrar(archivo="../4001") lo mandaba a la papelera. Lo que se borra o se exporta tiene
    que estar en la carpeta de ordenes y en ninguna otra parte.
    """
    carpeta = tmp_path / "ordenes"
    ordenes = Ordenes(str(carpeta))
    _con_pruebas(ordenes, "4001", 1)
    # Un archivo de otra cosa, afuera de la carpeta de ordenes.
    (tmp_path / "4001.json").write_text('{"numero": "el de afuera"}', encoding="utf-8")

    assert nombre_de_archivo_sano("4001")
    assert nombre_de_archivo_sano("123 - copia")            # ese si es un nombre
    for disfraz in ("../4001", "..", "x/y", "C:/otra/parte/4001", r"..\4001", ""):
        assert not nombre_de_archivo_sano(disfraz)

    assert ordenes.borrar("4001", archivo="../4001") is None
    assert ordenes.exportar("4001", archivo=r"..\4001") is None
    assert ordenes.cargar("4001", archivo="C:/otra/parte/4001") is None
    assert ordenes.carpeta_pruebas("4001", archivo="../4001") is None

    # Ni se toco lo de afuera ni se toco la orden de adentro.
    assert (tmp_path / "4001.json").is_file()
    assert (carpeta / "4001.json").is_file()
    assert not (carpeta / "_papelera").exists()
    assert not list(carpeta.rglob("*.zip"))


def test_la_carpeta_de_pruebas_se_nombra_por_el_archivo_cuando_el_archivo_viene(
    tmp_path: Path,
) -> None:
    """SI EL DEPOSITO SE NOMBRARA SIEMPRE POR EL NUMERO, la orden y su copia compartirian
    las fotos: borrar la copia se llevaria a la papelera la prueba que respalda la orden del
    cliente, y exportar la copia mandaria en su zip las fotos de la otra.

    Para una orden que creo el programa los dos textos son el mismo --medido en las 4 ordenes
    del taller: sanear_numero(numero) es igual al stem del .json en las cuatro-- asi que no
    hay ninguna foto que migrar ni ninguna orden vieja que cambie de deposito.
    """
    ordenes = Ordenes(str(tmp_path / "data" / "ordenes"))

    assert ordenes.carpeta_pruebas("123").name == "123"
    assert ordenes.carpeta_pruebas("123", "123").name == "123"
    assert ordenes.carpeta_pruebas("123", "123 - copia").name == "123 - copia"
    # El numero tecleado se sanea igual que para el .json; el nombre del archivo, no.
    assert ordenes.carpeta_pruebas("OA/123").name == "OA_123"
    assert ordenes.carpeta_pruebas("123", "../afuera") is None


def test_una_orden_sin_pruebas_se_guarda_con_la_forma_que_lee_el_exe_viejo(
    tmp_path: Path,
) -> None:
    """AL LADO DE LOS DATOS DEL TALLER HAY TRES .exe DE RESPALDO, y los tres arman cada
    diente con Diente(**d): UNA clave de mas adentro de un diente no les cuesta el campo,
    les cuesta la ORDEN ENTERA.

    MEDIDO con las 4 ordenes de verdad copiadas aparte y leidas con el modulo del commit con
    que se armaron esos .exe: guardadas con el asdict() pelado --que escribe "prueba": "" en
    los 336 renglones de diente-- quedaban 0 de 4 legibles (TypeError: Diente.__init__() got
    an unexpected keyword argument 'prueba'); omitiendo la prueba vacia, 4 de 4. Y eso
    pasaba a CADA diente que se anota, porque se guarda a cada diente.
    """
    import numpy as np

    # Los campos que conoce el Diente de esos .exe. No es una lista inventada: es la del
    # modulo de ese commit, que no tiene campo prueba.
    del_exe_viejo = {
        "numero", "estado", "parecido", "aprobado", "origen", "instante", "foto", "nota",
        "salteado",
    }
    carpeta = tmp_path / "ordenes"
    ordenes = Ordenes(str(carpeta))
    orden = ordenes.abrir(_orden("4001", sierras=1, dientes=3))
    orden.anotar(estado="afilado", aprobado=True, origen="la memoria")
    ordenes.guardar()

    dientes = json.loads(
        (carpeta / "4001.json").read_text(encoding="utf-8")
    )["sierras"][0]["revisados"]
    assert len(dientes) == 3
    for diente in dientes:
        assert set(diente) <= del_exe_viejo

    # Y cuando la prueba EXISTE, la clave esta: ahi es el dato y no se puede omitir. Lo que
    # no se puede evitar es que esa orden no la lea el .exe viejo; lo que si, que una orden
    # sin pruebas deje de leerse por un campo vacio.
    ruta = ordenes.guardar_prueba("4001", 1, 2, np.zeros((480, 640, 3), np.uint8))
    orden.anotar(estado="roto", aprobado=False, origen="la memoria", prueba=ruta)
    ordenes.guardar()
    dientes = json.loads(
        (carpeta / "4001.json").read_text(encoding="utf-8")
    )["sierras"][0]["revisados"]
    assert [d["numero"] for d in dientes if "prueba" in d] == [2]
    assert dientes[1]["prueba"] == ruta


def test_una_clave_de_diente_desconocida_no_cuesta_la_orden_entera(tmp_path: Path) -> None:
    """ES LA MISMA CORTESIA QUE ESTE PROGRAMA LE PIDE AL .exe VIEJO, DEL OTRO LADO.

    El dia que la app de afilados --o una version mas nueva de este programa-- escriba un
    campo mas adentro de un diente, el Diente(**d) pelado tira TypeError desde adentro de
    Orden.desde_datos(), que envuelve la orden entera: la orden sale ILEGIBLE y no se puede
    ni abrir ni exportar, que es de donde sale el zip de un reclamo. Reproducido con la orden
    de verdad 123 y una clave de mas en UN diente: cargar() devolvia None y el historial
    mostraba 0 dientes de 80.
    """
    from afilado.ordenes import Diente

    carpeta = tmp_path / "ordenes"
    carpeta.mkdir(parents=True)
    datos = _la_123_como_esta_en_el_taller()
    datos["sierras"][0]["revisados"][0]["temperatura_del_taller"] = 21
    (carpeta / "123.json").write_text(json.dumps(datos), encoding="utf-8")
    ordenes = Ordenes(str(carpeta))

    orden = ordenes.cargar("123")

    assert orden is not None
    assert (orden.total_dientes, orden.total_revisados) == (80, 7)
    fila = ordenes.historial()[0]
    assert not fila.ilegible and fila.revisados == 7
    assert ordenes.exportar("123") is not None      # el zip del reclamo sigue saliendo
    # El mecanismo, a la vista: lo que la version vieja hace con cada diente, y lo que se
    # hace ahora. El campo que no se conoce se ignora, pero el diente no se pierde.
    with pytest.raises(TypeError):
        Diente(**{"numero": 1, "temperatura_del_taller": 21})
    assert Diente.desde_datos({"numero": 1, "temperatura_del_taller": 21}).numero == 1


# ---------------------------------------------------------------------------
# UNA SOLA IDEA DE DONDE ESTA LA PRUEBA: lo que se cuenta, lo que viaja en el
# zip del reclamo y lo que se mueve a la papelera
# ---------------------------------------------------------------------------


def _apuntar_la_prueba(archivo: Path, diente: int, ruta: str) -> None:
    """Deja el .json nombrando `ruta` como prueba de ese diente, SIN tocar ninguna foto.

    Es lo que queda cuando alguien edita un .json con el Bloc de notas, o cuando la carpeta
    Afilado viene de otra PC: el renglon del diente nombra una ruta y las fotos estan donde
    estan. Que la foto no se mueva es todo el punto de estas pruebas: lo que se mira es la
    diferencia entre "esa foto no esta" y "esa foto no es la prueba de esta orden".
    """
    datos = json.loads(archivo.read_text(encoding="utf-8"))
    datos["sierras"][0]["revisados"][diente - 1]["prueba"] = ruta
    archivo.write_text(json.dumps(datos, indent=2, ensure_ascii=False), encoding="utf-8")


def test_una_prueba_que_apunta_afuera_de_la_carpeta_de_la_orden_no_es_prueba(
    tmp_path: Path, caplog
) -> None:
    """SIN ESTO, EL PAPEL DE UN RECLAMO PRESENTA COMO FOTO DEL DIENTE CUALQUIER ARCHIVO DEL
    DISCO, Y LA PANTALLA LO DICE EN VERDE.

    La ruta que el .json nombra se resolvia contra data/ y con eso alcanzaba, asi que valia
    cualquier cosa. Los cuatro renglones de aca abajo son las cuatro formas que se pueden
    escribir en un .json: salir de la carpeta con "..", nombrar un archivo con su letra de
    unidad, ponerle una barra adelante, y nombrar la carpeta de OTRA orden.
    MEDIDO VOLVIENDO EL CODIGO A COMO ERA, en esta misma prueba: la fila decia con_prueba=3
    y prueba_perdida=1, el informe "4 nombradas, 3 encontradas   -   FALTAN 1", y adentro del
    zip del reclamo viajaban el .ini de configuracion --como "Users/.../configuracion.ini",
    porque zipfile le saca la letra de unidad-- la foto de la papelera de la coleccion --como
    "coleccion/_papelera/..."-- y la foto de la orden 4002. Tres archivos que no son la
    prueba de esta orden, presentados como la prueba de tres dientes: a los dos primeros
    zipfile les cambio el nombre adentro del paquete, asi que ni el renglon que los nombra
    resuelve, y el tercero es la foto de OTRA orden metida en el paquete de esta.
    LOS CUATRO ARCHIVOS EXISTEN, y eso es lo que separa este arreglo del de la foto que
    falta: no se cuentan como perdidas porque no esten, se cuentan como perdidas porque no
    son la prueba de esta orden. La de la barra adelante es la unica que el codigo de antes
    tampoco contaba --el join la mandaba a C:/pruebas/..., que no existe-- y es la foto de
    verdad de este mismo diente: tampoco cuenta, y no por donde cae un join sino porque
    medido, zipfile le saca la barra al guardarla --queda "pruebas/4001/..." adentro del
    paquete-- asi que el renglon del informe que la nombra con la barra no resuelve del otro
    lado.
    """
    carpeta = tmp_path / "ordenes"
    ordenes = Ordenes(str(carpeta))
    _orden_viva, rutas = _con_pruebas(ordenes, "4001", 4, dientes=4)
    _otra_viva, de_la_otra = _con_pruebas(ordenes, "4002", 1)

    # Una foto que SI esta, en la papelera de la coleccion, alcanzada saliendo de la carpeta:
    # es el caso de verdad, las 14 fotos de las ordenes 123 y 1234 del taller estan ahi.
    en_la_coleccion = (
        tmp_path / "coleccion" / "_papelera" / "afilado_20261113_161954"
        / "afilado_20261112_161511_098.jpg"
    )
    en_la_coleccion.parent.mkdir(parents=True)
    en_la_coleccion.write_bytes((tmp_path / rutas[0]).read_bytes())
    # Un archivo que no es ninguna foto, nombrado con su letra de unidad.
    de_configuracion = tmp_path / "configuracion.ini"
    de_configuracion.write_text("[taller]\ncamara=1\n", encoding="utf-8")

    afuera = {
        1: "pruebas/../coleccion/_papelera/afilado_20261113_161954/"
           "afilado_20261112_161511_098.jpg",
        2: de_configuracion.as_posix(),                 # "C:/.../configuracion.ini"
        3: "/" + rutas[2],                              # su propia foto, con barra adelante
        4: de_la_otra[0],                               # "pruebas/4002/..."
    }
    for diente, ruta in afuera.items():
        _apuntar_la_prueba(carpeta / "4001.json", diente, ruta)
    # Los cuatro archivos estan en el disco: ninguno de los cuatro casos es "falta la foto".
    assert en_la_coleccion.is_file() and de_configuracion.is_file()
    assert (tmp_path / rutas[2]).is_file() and (tmp_path / de_la_otra[0]).is_file()

    taller = Ordenes(str(carpeta))          # el programa abierto de nuevo, leyendo el disco
    por_archivo = {fila.archivo: fila for fila in taller.historial()}

    de_la_4001 = por_archivo["4001"]
    assert (de_la_4001.con_prueba, de_la_4001.prueba_perdida) == (0, 4)
    # Y la orden que SI tiene su prueba en su carpeta sigue contandola: la regla no se lleva
    # puesto el caso normal, que es el de las 4 ordenes del taller.
    assert (por_archivo["4002"].con_prueba, por_archivo["4002"].prueba_perdida) == (1, 0)

    orden = taller.cargar("4001")
    assert taller.pruebas_en_disco(orden) == set()
    papel = orden.informe(taller.pruebas_en_disco(orden))
    assert "Fotos de prueba  : 4 nombradas, 0 encontradas   -   FALTAN 4" in papel
    for ruta in afuera.values():
        assert f"prueba: {ruta}   (NO ESTA)" in papel

    # El zip del reclamo no lleva NINGUNA de las cuatro: la misma cuenta que la pantalla.
    paquete_final = taller.exportar("4001")
    assert paquete_final is not None
    with zipfile.ZipFile(paquete_final) as paquete:
        assert sorted(paquete.namelist()) == ["informe.csv", "informe.txt", "orden.json"]

    # Y queda escrito CUAL era la ruta, con el nombre del .json: es lo unico con lo que el
    # que pueda abrir ese archivo puede arreglarlo.
    assert "que esta afuera de su carpeta de pruebas" in caplog.text
    for ruta in afuera.values():
        assert ruta in caplog.text


def test_el_zip_del_reclamo_no_lleva_la_foto_que_ningun_diente_nombra(
    tmp_path: Path,
) -> None:
    """UNA FOTO QUE NINGUN RENGLON NOMBRA PARECE LA PRUEBA DE UN DIENTE Y NO ES LA DE
    NINGUNO, Y ESO ES JUSTO LO QUE NO SE PUEDE MANDAR EN UN RECLAMO.

    En el deposito quedan fotos que el .json no nombra: la de un guardado que escribio la
    foto y no llego a escribir el .json --la foto va antes, a proposito-- y los .jpg.tmp de
    un corte de luz. Llevan el numero de sierra y de diente en el nombre, igual que las
    buenas. El zip recorria la carpeta con iterdir(), asi que viajaban: medido volviendo el
    codigo a como era, en esta misma prueba, el paquete salia con 4 archivos adentro de
    pruebas/ --las 2 buenas, la huerfana y el .jpg.tmp-- y el informe nombrando 2. El que
    abre el reclamo del otro lado ve cuatro fotos de dientes y dos renglones que las
    expliquen.
    Y no se borran: las saco el microscopio y pueden ser la unica foto de ese diente.
    """
    carpeta = tmp_path / "ordenes"
    ordenes = Ordenes(str(carpeta))
    _orden_viva, rutas = _con_pruebas(ordenes, "4001", 2)
    deposito = ordenes.carpeta_pruebas("4001")
    # Con el mismo nombre que les pone guardar_prueba(): es lo que las hace pasar por prueba.
    huerfana = deposito / "s01_d003_20261116_161039_481.jpg"
    huerfana.write_bytes((tmp_path / rutas[0]).read_bytes())
    del_corte_de_luz = deposito / "s01_d004_20261116_161040_112.jpg.tmp"
    del_corte_de_luz.write_bytes((tmp_path / rutas[0]).read_bytes())

    paquete_final = ordenes.exportar("4001")

    assert paquete_final is not None
    with zipfile.ZipFile(paquete_final) as paquete:
        fotos = sorted(n for n in paquete.namelist() if n.startswith("pruebas/"))
        informe = paquete.read("informe.txt").decode("utf-8")
    # Adentro del paquete esta exactamente lo que los dientes nombran, ni una mas.
    assert fotos == sorted(rutas)
    assert "Fotos de prueba  : 2 nombradas, 2 encontradas" in informe
    # Y las dos que no viajaron siguen en la carpeta del taller: no viajar no es borrarse.
    assert huerfana.is_file() and del_corte_de_luz.is_file()


def test_el_informe_dice_cuantas_fotos_quedaron_en_la_carpeta_que_nadie_nombra(
    tmp_path: Path,
) -> None:
    """QUE NO VIAJEN NO PUEDE SER CALLARLAS: el que arma el reclamo tiene que saber que en
    esa carpeta quedo algo que el papel no nombra.

    Si no se dice, la foto huerfana --que puede ser la unica que quedo de ese diente, de un
    guardado que se corto-- no existe para nadie: no esta en el paquete y ningun renglon
    habla de ella. El papel es lo unico que el operario lee antes de mandar el mail.
    Lo dicen LOS DOS informes, el que viaja en el zip y el que queda al lado de la orden,
    porque son el mismo papel leido en dos lados y tienen que decir lo mismo.
    """
    carpeta = tmp_path / "ordenes"
    ordenes = Ordenes(str(carpeta))
    orden_viva, rutas = _con_pruebas(ordenes, "4001", 2)
    deposito = ordenes.carpeta_pruebas("4001")
    for nombre in ("s01_d003_20261116_161039_481.jpg",
                   "s01_d004_20261116_161040_112.jpg.tmp"):
        (deposito / nombre).write_bytes((tmp_path / rutas[0]).read_bytes())

    assert ordenes.pruebas_huerfanas(orden_viva) == 2

    los_dos_renglones = [
        "                   en la carpeta hay 2 fotos que ningun diente de esta",
        "                   orden nombra: no viajan en el paquete ni se borran",
    ]
    texto, _planilla = ordenes.escribir_informe(orden_viva)
    al_lado_de_la_orden = texto.read_text(encoding="utf-8").splitlines()
    paquete_final = ordenes.exportar("4001")
    with zipfile.ZipFile(paquete_final) as paquete:
        del_reclamo = paquete.read("informe.txt").decode("utf-8").splitlines()

    for renglon in los_dos_renglones:
        assert renglon in al_lado_de_la_orden
        assert renglon in del_reclamo
        # DOS RENGLONES Y NO UNO: el informe se imprime a 72 columnas --es el ancho de las
        # reglas de "=" que lo encabezan-- y en una sola linea esto no entra, asi que en el
        # papel se partiria solo y donde cayera. Los numeros se miden aca abajo, no se creen.
        assert len(renglon) <= 72
    en_una_sola_linea = los_dos_renglones[0] + " " + los_dos_renglones[1].strip()
    assert (len(en_una_sola_linea), len(en_una_sola_linea.strip())) == (121, 102)
    # Y la orden mas grande que sale del taller son 4 sierras de 80 dientes, o sea que el
    # numero puede llegar a tres digitos: ahi el primer renglon mide justo 72 y entra igual.
    assert len(los_dos_renglones[0].replace(" 2 fotos", " 320 fotos")) == 72

    # Y con el deposito limpio no se dice nada: no hay nada que avisar.
    for suelta in deposito.iterdir():
        if suelta.name not in [Path(r).name for r in rutas]:
            suelta.unlink()
    assert ordenes.pruebas_huerfanas(orden_viva) == 0
    assert "en la carpeta hay" not in orden_viva.informe(set(rutas), 0)


def test_una_carpeta_de_pruebas_que_no_se_deja_mirar_no_se_declara_vacia(
    tmp_path: Path, monkeypatch
) -> None:
    """"NO SE PUDO MIRAR" NO ES "NO HAY NINGUNA", y el informe no puede afirmar lo que nadie
    conto.

    Si la carpeta no se deja leer --permisos, la unidad de red del taller caida, el antivirus
    con el archivo tomado-- devolver 0 seria escribir en el papel de un reclamo que ahi no
    quedo ninguna foto suelta sin que nadie haya ido a ver. Con None el informe no dice nada,
    que es lo unico cierto.
    """
    carpeta = tmp_path / "ordenes"
    ordenes = Ordenes(str(carpeta))
    orden_viva, _rutas = _con_pruebas(ordenes, "4001", 1)
    deposito = ordenes.carpeta_pruebas("4001")
    mirar = Path.iterdir

    def no_se_deja_mirar(self):
        if self == deposito:
            raise OSError(13, "Permission denied")
        return mirar(self)

    monkeypatch.setattr(Path, "iterdir", no_se_deja_mirar)

    assert ordenes.pruebas_huerfanas(orden_viva) is None
    # Y una carpeta que no esta si es 0: no hay archivos que nadie nombre.
    assert ordenes.pruebas_huerfanas(_orden("9999", sierras=1, dientes=1)) == 0
    assert "en la carpeta hay" not in orden_viva.informe(set(), None)


def test_la_fila_y_el_zip_contestan_la_misma_pregunta_de_si_la_prueba_esta(
    tmp_path: Path, monkeypatch
) -> None:
    """DOS PAPELES DEL MISMO RECLAMO, SOBRE EL MISMO DISCO, NO PUEDEN DECIR LO CONTRARIO.

    "Esta la foto?" se contestaba de dos maneras: la fila del historial listaba la carpeta y
    comparaba el nombre exacto, y el zip preguntaba is_file(). Mientras las dos den lo mismo
    no se nota; cuando no, la pantalla dice una cosa y el paquete que sale por mail dice la
    otra. Los dos casos de aca abajo pasan sin que nadie toque un .json a mano.

    UNO: LA CARPETA QUE NO SE DEJA LISTAR --un permiso mal puesto, la unidad de red caida,
    una carpeta que vino de otra PC--. El OSError del iterdir se volvia una lista vacia, o
    sea "la prueba no esta", que es afirmar lo que nadie pudo medir. MEDIDO volviendo el
    codigo a como era, con icacls /deny :(RD) sobre el deposito de una orden de 3 pruebas:
    la fila salia "3 PRUEBAS QUE NO ESTAN" en rojo, la pantalla de borrado decia que no se
    llevaba ninguna foto, y el zip salia CON LAS 3 y su informe diciendo "3 nombradas, 3
    encontradas". La foto se abre por su nombre aunque la carpeta no se deje recorrer
    --verificado, is_file() da True ahi-- asi que las tres estan y las tres se cuentan.

    DOS: LA MISMA FOTO NOMBRADA DE OTRA FORMA. Windows abre igual el archivo con otras
    mayusculas, con la extension en mayuscula o con un punto o un espacio al final, pero el
    nombre exacto de la lista no coincide. MEDIDO: de 5 formas, 4 daban fila en rojo con la
    foto adentro del zip. Aca no se afirma cual es la respuesta --depende del disco, y en
    uno que distinga mayusculas la respuesta correcta es que no esta-- se afirma que la fila,
    el informe y el paquete dan LA MISMA.
    """
    carpeta = tmp_path / "ordenes"
    ordenes = Ordenes(str(carpeta))
    _orden_viva, rutas = _con_pruebas(ordenes, "4001", 3)
    deposito = ordenes.carpeta_pruebas("4001")
    assert deposito is not None
    mirar = Path.iterdir

    def no_se_deja_listar(self):
        if self == deposito:
            raise OSError(13, "Permission denied")
        return mirar(self)

    monkeypatch.setattr(Path, "iterdir", no_se_deja_listar)

    fila = [f for f in ordenes.historial() if f.archivo == "4001"][0]
    paquete_final = ordenes.exportar("4001")
    assert paquete_final is not None
    with zipfile.ZipFile(paquete_final) as paquete:
        fotos = sorted(n for n in paquete.namelist() if n.startswith("pruebas/"))
    # Las tres estan: no se pudo RECORRER la carpeta, que es otra cosa.
    assert fila.con_prueba == 3 and fila.prueba_perdida == 0
    assert fotos == sorted(rutas)

    monkeypatch.undo()

    # Y el mismo archivo nombrado de otra forma: lo que importa es que no se contradigan.
    otra = tmp_path / "otra"
    otras_ordenes = Ordenes(str(otra))
    _otra_viva, otras_rutas = _con_pruebas(otras_ordenes, "4002", 1)
    archivo = otra / "4002.json"
    _apuntar_la_prueba(archivo, 1, otras_rutas[0].upper())

    fila = [f for f in otras_ordenes.historial() if f.archivo == "4002"][0]
    paquete_final = otras_ordenes.exportar("4002")
    assert paquete_final is not None
    with zipfile.ZipFile(paquete_final) as paquete:
        cuantas = sum(1 for n in paquete.namelist() if n.startswith("PRUEBAS/"))
        informe = paquete.read("informe.txt").decode("utf-8")
    assert fila.con_prueba == cuantas
    assert f"1 nombradas, {cuantas} encontradas" in informe


def test_el_marcador_de_la_orden_abierta_no_se_suelta_si_el_json_no_se_movio(
    tmp_path: Path, monkeypatch
) -> None:
    """EL .json ES EL UNICO DE LA FILA QUE EL OPERARIO PUEDE TENER ABIERTO EN EL BLOC DE
    NOTAS, Y ES JUSTO EL ULTIMO QUE SE MUEVE.

    El marcador se soltaba en el renglon de ANTES del movimiento, asi que con el .json
    trabado pasaba lo peor: el borrado devolvia completo=False, el .json seguia en el disco
    --o sea que la orden no se borro-- y aun asi el programa quedaba sin orden abierta. El
    operario dejaba de poder anotar dientes, en medio de una sierra, por una orden que al
    final sigue ahi. Medido volviendo el codigo a como era: abierta=None y el marcador del
    disco en "", con 4001.json ahi mismo y el log diciendo que el borrado quedo a medias.
    Y HACE FALTA ESTA PRUEBA APARTE: la otra del orden de los movimientos traba la planilla,
    que se mueve ANTES del .json, asi que ese renglon no se toca. Comprobado con el codigo
    viejo puesto de vuelta: esa prueba pasa igual, porque soltando el marcador antes o
    despues, cuando lo que se traba es la planilla la orden abierta queda igual. Esta es la
    unica que lo agarra.
    """
    carpeta = tmp_path / "ordenes"
    ordenes = Ordenes(str(carpeta))
    _orden_viva, rutas = _con_pruebas(ordenes, "4001", 1)
    ordenes.escribir_informe()
    mover = shutil.move

    def se_traba_el_json(origen, destino):
        """El .json abierto en el Bloc de notas: todo lo demas se mueve, el .json no."""
        if Path(origen).name == "4001.json":
            raise OSError(13, "Permission denied")
        return mover(origen, destino)

    monkeypatch.setattr(shutil, "move", se_traba_el_json)

    quedo = ordenes.borrar("4001")

    assert quedo is not None and not quedo.completo
    assert quedo.falto == "4001.json"
    assert quedo.movidos == ["pruebas/4001", "informe_4001.txt", "informe_4001.csv"]
    assert (carpeta / "4001.json").is_file()        # la orden NO se borro
    # Asi que el operario sigue con su orden abierta, en memoria y en el marcador del disco.
    assert ordenes.abierta is not None and ordenes.abierta.numero == "4001"
    assert (carpeta / ".abierta").read_text(encoding="utf-8").strip() == "4001"
    # Y al abrir el programa de nuevo la retoma, que es lo que el operario espera encontrar.
    de_nuevo = Ordenes(str(carpeta))
    assert de_nuevo.abierta is not None and de_nuevo.abierta.numero == "4001"
    assert de_nuevo.abierta.total_revisados == 1
    assert (tmp_path / rutas[0]).exists() is False  # la prueba si se fue a la papelera


def test_un_reintento_de_borrado_no_parte_la_orden_en_dos_carpetas_de_papelera(
    tmp_path: Path, monkeypatch
) -> None:
    """DOS CARPETAS DE PAPELERA CON LA MISMA ORDEN ADENTRO Y NADIE PUEDE SABER CUAL ES LA
    BUENA.

    Mover una carpeta no es un solo paso: cuando no se la puede renombrar, shutil.move copia
    foto por foto, asi que un borrado trabado deja unas fotos del otro lado Y las deja en su
    lugar. El reintento --que es lo que hace cualquiera: apretar otra vez-- abria OTRA
    carpeta con sello nuevo y volvia a copiar: medido volviendo el codigo a como era, la
    orden quedaba repartida entre _papelera/20261116_173906, con una foto, y
    _papelera/20261116_173906_1, con el .json y las tres. La misma orden en dos carpetas, con
    una foto repetida, y nadie puede saber mirando cual es la buena.
    Y EL REINTENTO TIENE QUE METER LAS FOTOS ADENTRO DE LA CARPETA QUE YA ESTA, de a una:
    shutil.move contra un destino que ya existe como carpeta no la reemplaza, la mete
    adentro. Medido sacando ese paso: las tres fotos terminaban en pruebas/4001/4001/ --un
    escalon mas abajo de donde el .json las nombra-- con la copia del primer intento sola en
    pruebas/4001/. Rescatadas con el Explorador, que es como se rescata en el taller, la
    prueba vuelve al disco y no se encuentra.
    """
    carpeta = tmp_path / "ordenes"
    ordenes = Ordenes(str(carpeta))
    _orden_viva, rutas = _con_pruebas(ordenes, "4001", 3)
    mover = shutil.move

    def copia_a_medias(origen, destino):
        """Lo que hace shutil.move cuando no puede renombrar la carpeta: copia la primera
        foto y se traba en la siguiente."""
        fuente = Path(origen)
        if fuente.is_dir():
            Path(destino).mkdir(parents=True, exist_ok=True)
            primera = sorted(fuente.iterdir())[0]
            shutil.copy2(primera, Path(destino) / primera.name)
            raise OSError(13, "Permission denied")
        return mover(origen, destino)

    monkeypatch.setattr(shutil, "move", copia_a_medias)
    primero = ordenes.borrar("4001")
    assert primero is not None and not primero.completo
    assert primero.falto == "pruebas/4001"
    # La foto que alcanzo a pasar quedo en la papelera Y sigue en el deposito: por eso el
    # reintento no puede empezar una carpeta nueva.
    assert len(list((primero.destino / "pruebas" / "4001").glob("*.jpg"))) == 1
    assert len(list(ordenes.carpeta_pruebas("4001").glob("*.jpg"))) == 3

    monkeypatch.undo()                              # el visor de Windows, cerrado
    segundo = ordenes.borrar("4001")

    assert segundo is not None and segundo.completo
    assert segundo.destino == primero.destino
    # UNA SOLA carpeta de papelera para esta orden, no una por intento.
    assert [p.name for p in (carpeta / "_papelera").iterdir()] == [primero.destino.name]
    # Y cada foto en la MISMA ruta relativa que el .json le da, no un escalon mas abajo.
    for ruta in rutas:
        assert (segundo.destino / ruta).is_file()
    assert not (segundo.destino / "pruebas" / "4001" / "4001").exists()
    assert (segundo.destino / "4001.json").is_file()
    assert not ordenes.carpeta_pruebas("4001").exists()
    assert ordenes.historial() == []


def test_dos_exportaciones_en_el_mismo_segundo_no_se_pisan(
    tmp_path: Path, monkeypatch
) -> None:
    """EL ZIP ES LO QUE SE MANDA POR MAIL: PISAR EL ANTERIOR SIN DECIRLO ES PERDER EL
    PAQUETE QUE EL OPERARIO CREE TENER.

    El nombre lleva un sello de fecha y hora con resolucion de UN SEGUNDO, y exportar()
    tarda mucho menos que eso: medido en este disco con una orden de 3 pruebas, mediana 5,8
    ms (de 5,2 a 6,6 en seis corridas), y las seis salieron con el MISMO sello
    20261116_173223. O sea que dos 'e' seguidas en el historial caen en el mismo segundo sin
    ninguna dificultad, y ahi el segundo zip caia encima del primero: medido volviendo el
    codigo a como era, las dos exportaciones devolvieron la misma ruta
    --orden_4001_20261116_161040.zip-- y quedaba un solo paquete en la carpeta. Y la pantalla
    no tiene como avisarlo: leido el manejador de la 'e' (run_live.py), lo que dibuja es
    "EXPORTADA EN exportado/<nombre del zip> - <cuanto pesa>" con el nombre que exportar() le
    devuelve, asi que las dos veces decia lo mismo y las dos veces parecia que habia salido.
    El reloj se clava a proposito, para que la prueba no dependa de donde cae el borde del
    segundo: sin clavarlo, una corrida de cada tantas se parte en dos segundos y la prueba
    pasa sin probar nada.
    """
    from afilado import ordenes as modulo

    carpeta = tmp_path / "ordenes"
    ordenes = Ordenes(str(carpeta))
    _orden_viva, rutas = _con_pruebas(ordenes, "4001", 2)

    class _el_mismo_segundo(datetime):
        """El reloj del taller clavado: las dos exportaciones en el mismo segundo."""

        @classmethod
        def now(cls, tz=None):
            return datetime(2026, 11, 16, 16, 10, 40)

    monkeypatch.setattr(modulo, "datetime", _el_mismo_segundo)

    primero = ordenes.exportar("4001")
    segundo = ordenes.exportar("4001")

    assert primero is not None and segundo is not None
    assert primero != segundo
    assert primero.name == "orden_4001_20261116_161040.zip"
    assert segundo.name == "orden_4001_20261116_161040_1.zip"
    # Y los dos son paquetes enteros, no uno bueno y un pedazo: el primero tiene que seguir
    # sirviendo para el reclamo que ya se mando.
    for paquete_final in (primero, segundo):
        with zipfile.ZipFile(paquete_final) as paquete:
            assert sorted(n for n in paquete.namelist() if n.startswith("pruebas/")) == (
                sorted(rutas)
            )
            assert "INFORME DE AFILADO" in paquete.read("informe.txt").decode("utf-8")

    # Cuando el nombre exacto lo elige el que llama, se respeta tal cual: ahi la decision es
    # suya y no la nuestra, y el programa no le cambia el nombre al archivo que pidio.
    elegido = tmp_path / "para_el_mail.zip"
    assert ordenes.exportar("4001", destino=elegido) == elegido
    assert ordenes.exportar("4001", destino=elegido) == elegido
    assert [p.name for p in tmp_path.glob("*.zip")] == ["para_el_mail.zip"]


def test_el_papel_que_explica_de_donde_salieron_las_fotos_viaja_con_el_reclamo(
    tmp_path: Path,
) -> None:
    """SIETE FOTOS SIN UNA LINEA QUE DIGA QUE SON ES PRESENTARLAS COMO LO QUE NO SON.

    Las ordenes 123 y 1234 se revisaron cuando el programa no tenia deposito de pruebas, y
    sus 14 fotos se rescataron de la papelera de la coleccion cuatro dias despues. Al lado
    de ellas quedo procedencia.txt, que dice eso y trae el sha256 de cada una. El que
    recibe el reclamo del otro lado solo tiene el paquete: si ese papel no viaja, abre
    siete fotos sin ninguna manera de saber que no se sacaron en el momento de juzgar.

    Es la UNICA excepcion a "solo viaja lo que un diente nombra", y no la rompe: lo que esa
    regla impide es mandar una FOTO que parezca la prueba de un diente sin serlo, y este
    archivo no puede confundirse con la prueba de ninguno.
    MEDIDO en el taller el 17/11/2026, despues de mudar las fotos: el zip de la orden 123
    salia con 10 archivos y sin el papel, y el informe decia "en la carpeta hay 1 fotos que
    ningun diente de esta orden nombra" --llamando foto a un .txt-- y que no viajaba.
    """
    carpeta = tmp_path / "ordenes"
    ordenes = Ordenes(str(carpeta))
    orden_viva, rutas = _con_pruebas(ordenes, "4001", 2)
    deposito = ordenes.carpeta_pruebas("4001")
    assert deposito is not None
    (deposito / "procedencia.txt").write_text(
        "DE DONDE SALIERON LAS 2 FOTOS DE ESTA CARPETA\n", encoding="utf-8"
    )

    paquete_final = ordenes.exportar("4001")

    assert paquete_final is not None
    with zipfile.ZipFile(paquete_final) as paquete:
        adentro = sorted(n for n in paquete.namelist() if n.startswith("pruebas/"))
    assert adentro == sorted(rutas + ["pruebas/4001/procedencia.txt"])
    # Y NO SE CUENTA COMO FOTO SUELTA: no es una foto, y no es cierto que no viaje.
    assert ordenes.pruebas_huerfanas(orden_viva) == 0
    assert "en la carpeta hay" not in orden_viva.informe(set(rutas), 0)


def test_una_sola_foto_suelta_se_dice_en_singular(tmp_path: Path) -> None:
    """Esto lo lee el taller en el papel de un reclamo: "hay 1 fotos" no se escribe."""
    carpeta = tmp_path / "ordenes"
    ordenes = Ordenes(str(carpeta))
    orden_viva, rutas = _con_pruebas(ordenes, "4001", 1)

    texto = orden_viva.informe(set(rutas), 1)

    assert "hay 1 foto que ningun diente" in texto
    assert "hay 1 fotos" not in texto
    assert "hay 2 fotos que ningun diente" in orden_viva.informe(set(rutas), 2)
