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
