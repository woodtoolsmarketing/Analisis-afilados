"""El bucle entero, de punta a punta, con una camara de mentira y teclas escritas a mano.

POR QUE HACE FALTA ESTO ADEMAS DE LAS PRUEBAS DE CADA MODULO
-------------------------------------------------------------
Los modulos estan probados de a uno, pero los errores que llegaron al taller no fueron de
un modulo: fueron de como se enganchan. Un cierre que lee una variable que despues se
reasigna, una lambda que captura la foto equivocada, una funcion que cambio de parametros
y su unico llamador quedo viejo. Nada de eso lo ve una prueba de unidad, y todo eso lo
pagaba el operario abriendo el programa.

Aca se corre _ejecutar_revision de verdad --el mismo bucle que corre en la PC del
taller-- con la camara, la ventana y el teclado de mentira, y se mira que quedo en disco.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import pytest

from afilado import config as config_mod
from afilado.cli import run_live

_ESC = 27
_ENTER = 13
_ESPACIO = 32


def _diente(semilla: int = 0, lado: int = 240) -> np.ndarray:
    """Una imagen con estructura: el ruido puro no se parece a lo que da el microscopio."""
    img = np.full((lado, lado, 3), 60, np.uint8)
    cv2.fillPoly(img, [np.array([[10, lado - 20], [10, 120 + semilla * 4],
                                 [lado - 60, 40], [lado - 20, 70],
                                 [lado - 20, lado - 20]])], (190, 190, 190))
    cv2.line(img, (10, 120 + semilla * 4), (lado - 60, 40), (235, 235, 235), 6)
    for i in range(0, lado, 7):
        cv2.line(img, (i, 0), (i + 40 + semilla * 9, lado), (0, 0, 0), 1)
    return cv2.GaussianBlur(img, (5, 5), 0)


class _CamaraFalsa:
    """Entrega una foto distinta cada vez: si entregara siempre la misma, el guardado se
    frenaria solo por duplicada y la prueba no probaria nada."""

    congelada = False
    descripcion = "camara de prueba"

    def __init__(self, *_args, **_kwargs) -> None:
        self.leidos = 0
        self.cerrada = False

    def abrir(self) -> None:
        pass

    def leer(self) -> Optional[np.ndarray]:
        self.leidos += 1
        return _diente(self.leidos % 7)

    def cerrar(self) -> None:
        self.cerrada = True


@pytest.fixture
def taller(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Un taller de mentira: todas las rutas relativas caen adentro de tmp_path."""
    monkeypatch.setattr(config_mod, "raiz_repo", lambda: tmp_path)
    monkeypatch.setattr(run_live, "Camara", _CamaraFalsa)
    monkeypatch.setattr(run_live, "crear_clasificador", lambda cfg: None)

    # La ventana, el teclado y el mouse, de mentira.
    monkeypatch.setattr(cv2, "namedWindow", lambda *a, **k: None)
    monkeypatch.setattr(cv2, "setMouseCallback", lambda *a, **k: None)
    monkeypatch.setattr(cv2, "imshow", lambda *a, **k: None)
    monkeypatch.setattr(cv2, "resizeWindow", lambda *a, **k: None)
    monkeypatch.setattr(cv2, "destroyAllWindows", lambda: None)
    return tmp_path


def _correr(
    teclas: list[int], monkeypatch: pytest.MonkeyPatch, recuadros: bool = False
) -> int:
    """Corre el bucle mandandole esas teclas. Al quedarse sin teclas, sale.

    `recuadros` enciende interfaz.recuadros_de_diente, que de fabrica va apagado porque
    con la luz a contraluz del taller el detector marca el cuadro entero. Las pruebas que
    cuidan el analisis POR DIENTE lo encienden a mano: la funcion sigue existiendo y tiene
    que seguir andando para el dia que el detector sepa trabajar con esa luz.
    """
    pendientes = list(teclas)

    def _waitKey(_espera: int = 0) -> int:
        # Al terminar el guion se manda ESC, que SIEMPRE escala un nivel --de un
        # formulario a su pantalla, de una pantalla al panel, del panel al programa-- y
        # termina cerrando. Con 'q' no alcanzaba: adentro del formulario de la orden la
        # 'q' es una letra, asi que la prueba se quedaba tecleando "qqqq..." para siempre
        # y lo unico que se veia era que pytest se habia colgado. Ya paso dos veces.
        return pendientes.pop(0) if pendientes else _ESC

    monkeypatch.setattr(cv2, "waitKey", _waitKey)

    cfg = config_mod.AppConfig(clases=["aprobada", "rechazada"], clases_defecto=["rechazada"])
    cfg.interfaz.modo = "simple"
    cfg.interfaz.evaluar_en_vivo = True
    cfg.interfaz.nitidez_minima = 1.0
    cfg.interfaz.recuadros_de_diente = recuadros
    cfg.detector.tarea = "classify"
    cfg.memoria.minimo_por_concepto = 1
    return run_live._ejecutar_revision(cfg, argparse.Namespace(grabar=None))


def _fotos(base: Path, concepto: str) -> list[Path]:
    carpeta = base / "data" / "coleccion" / concepto
    return sorted(carpeta.glob("*.jpg")) if carpeta.is_dir() else []


# ---------------------------------------------------------------------------


def test_arranca_y_sale_limpio(taller: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    assert _correr([ord("q")], monkeypatch) == 0


def test_guardar_una_foto_la_deja_en_su_carpeta(taller: Path, monkeypatch) -> None:
    _correr([_ESPACIO, ord("a"), ord("q")], monkeypatch)

    assert len(_fotos(taller, "aprobada")) == 1


def test_la_memoria_se_arma_sola_con_lo_que_se_va_guardando(taller: Path, monkeypatch) -> None:
    """Esto es lo que pidio el taller: que aprenda de cada foto, sin entrenar nada."""
    _correr(
        [_ESPACIO, ord("a"), ord(" "), ord("a"), ord(" "), ord("r"), ord("q")],
        monkeypatch,
    )

    cache = taller / "data" / "memoria" / "memoria.npz"
    assert cache.is_file()
    # Las fotos quedaron repartidas en sus dos conceptos.
    assert len(_fotos(taller, "aprobada")) >= 1
    assert len(_fotos(taller, "rechazada")) >= 1


def test_crear_un_concepto_desde_el_panel_y_archivar_ahi(taller: Path, monkeypatch) -> None:
    """El camino completo del pedido: abrir el panel, crear un concepto y mandar la foto."""
    teclas = [_ESPACIO, ord("c"), ord("n")]
    teclas += [ord(c) for c in "astillado"]
    teclas += [_ENTER, _ENTER, ord("q")]

    _correr(teclas, monkeypatch)

    assert (taller / "data" / "coleccion" / "astillado").is_dir()
    assert len(_fotos(taller, "astillado")) == 1


def test_la_q_adentro_del_nombre_no_cierra_el_programa(taller: Path, monkeypatch) -> None:
    """Si la 'q' se escapara al manejador de siempre, el programa se cerraria en la mitad
    de la palabra y el operario no entenderia por que."""
    teclas = [ord("c"), ord("n")]
    teclas += [ord(c) for c in "quemado"]
    teclas += [_ENTER, _ESC, ord("q")]

    _correr(teclas, monkeypatch)

    assert (taller / "data" / "coleccion" / "quemado").is_dir()


def test_borrar_un_concepto_desde_el_panel(taller: Path, monkeypatch) -> None:
    _correr([_ESPACIO, ord("a"), ord("q")], monkeypatch)
    assert len(_fotos(taller, "aprobada")) == 1

    # 2 elige 'aprobada' (1 es la galeria de modelos), b pide borrar, s confirma.
    _correr([ord("c"), ord("2"), ord("b"), ord("s"), ord("q")], monkeypatch)

    assert not (taller / "data" / "coleccion" / "aprobada").exists()
    assert list((taller / "data" / "coleccion" / "_papelera").rglob("*.jpg"))


def test_el_concepto_nuevo_se_puede_usar_sin_reiniciar(taller: Path, monkeypatch) -> None:
    """El almacen valida contra su lista de clases: si no se entera del concepto nuevo,
    guardar ahi falla con 'clase desconocida' hasta cerrar y volver a abrir."""
    teclas = [ord("c"), ord("n")] + [ord(c) for c in "rebaba"] + [_ENTER, _ESC]
    # Ya fuera del panel: congelar y guardar en el concepto nuevo desde el panel otra vez.
    teclas += [_ESPACIO, ord("c"), ord("4"), _ENTER, ord("q")]

    _correr(teclas, monkeypatch)

    assert len(_fotos(taller, "rebaba")) == 1


def test_borrar_una_foto_desde_el_panel(taller: Path, monkeypatch) -> None:
    _correr([_ESPACIO, ord("a"), ord("q")], monkeypatch)
    guardada = _fotos(taller, "aprobada")[0]

    # c abre, 2 elige aprobada, v ve sus fotos, b borra la elegida.
    _correr([ord("c"), ord("2"), ord("v"), ord("b"), ord("q")], monkeypatch)

    assert not guardada.exists()
    assert _fotos(taller, "aprobada") == []


def test_la_foto_de_modelo_tambien_se_elige_desde_el_panel(taller: Path, monkeypatch) -> None:
    """Es el destino que el taller mas usa: tiene que estar en la misma lista."""
    _correr([_ESPACIO, ord("c"), ord("1"), _ENTER, ord("q")], monkeypatch)

    modelos = list((taller / "data" / "referencia" / "microscopio").glob("*.jpg"))
    assert len(modelos) == 1


def test_no_se_archiva_dos_veces_la_misma_foto_congelada(taller: Path, monkeypatch) -> None:
    """El error que dejo 50 copias por clase: con la imagen congelada, cada pulsacion de
    GUARDAR reescribia el MISMO frame."""
    _correr([_ESPACIO, ord("a"), ord("a"), ord("a"), ord("q")], monkeypatch)

    assert len(_fotos(taller, "aprobada")) == 1


def test_con_la_camara_congelada_no_entra_ni_una_foto_de_modelo(
    taller: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Con el microscopio desenchufado el programa cae en DroidCam, que muestra un cartel
    fijo. Ese cartel ya entro una vez al dataset como sierra APROBADA; si entrara como
    foto de MODELO seria peor todavia, porque todo lo que venga despues se compara contra
    el."""
    class _Congelada(_CamaraFalsa):
        congelada = True

    monkeypatch.setattr(run_live, "Camara", _Congelada)

    _correr([ord("f"), _ESPACIO, ord("a"), ord("q")], monkeypatch)

    assert not list((taller / "data" / "referencia").rglob("*.jpg"))
    assert _fotos(taller, "aprobada") == []


# ---------------------------------------------------------------------------
# La orden de afilado, de punta a punta
# ---------------------------------------------------------------------------


def _ordenes(base: Path):
    from afilado.ordenes import Ordenes
    return Ordenes(str(base / "data" / "ordenes"))


def test_archivar_un_diente_lo_anota_en_la_orden(taller: Path, monkeypatch) -> None:
    """Es lo que convierte una pila de fotos sueltas en 'la sierra 1, diente 3 de 24'."""
    from afilado.ordenes import Orden, Ordenes

    ordenes = Ordenes(str(taller / "data" / "ordenes"))
    ordenes.abrir(Orden.nueva("1234", 1, 4, nota_pedido="NP-1", cliente="Taller"))

    _correr([_ESPACIO, ord("a"), ord(" "), ord("a"), ord("q")], monkeypatch)

    guardada = _ordenes(taller).abierta
    assert guardada is not None
    assert guardada.total_revisados == 2
    primero = guardada.sierras[0].revisados[0]
    assert primero.estado == "aprobada" and primero.aprobado is True
    assert primero.foto.endswith(".jpg")
    assert "operario" in primero.origen


def test_el_informe_sale_con_lo_que_se_archivo(taller: Path, monkeypatch) -> None:
    from afilado.ordenes import Orden, Ordenes

    ordenes = Ordenes(str(taller / "data" / "ordenes"))
    ordenes.abrir(Orden.nueva("1234", 1, 3))

    # Se archiva un diente bueno y uno malo, y despues se pide el informe desde el panel.
    _correr([_ESPACIO, ord("a"), ord(" "), ord("r"), ord("o"), ord("i"), ord("q")],
            monkeypatch)

    informe = (taller / "data" / "ordenes" / "informe_1234.txt").read_text(encoding="utf-8")
    assert "APROBADO" in informe and "RECHAZADO" in informe
    assert "(sin revisar)" in informe          # el tercero no se reviso y se dice
    assert (taller / "data" / "ordenes" / "informe_1234.csv").is_file()


def test_la_orden_se_carga_a_mano_desde_el_programa(taller: Path, monkeypatch) -> None:
    teclas = [ord("o")]
    teclas += [ord(c) for c in "77"] + [_ENTER]        # orden
    teclas += [ord(c) for c in "NP9"] + [_ENTER]       # nota de pedido
    teclas += [_ENTER]                                  # cliente vacio
    teclas += [ord("2"), _ENTER]                        # sierras
    teclas += [ord("5"), _ENTER]                        # dientes (ultimo ENTER = crear)
    teclas += [ord("q")]

    _correr(teclas, monkeypatch)

    abierta = _ordenes(taller).abierta
    assert abierta is not None
    assert abierta.numero == "77" and abierta.nota_pedido == "NP9"
    assert len(abierta.sierras) == 2 and abierta.sierras[0].dientes == 5


def test_al_abrir_el_programa_se_retoma_la_orden(taller: Path, monkeypatch) -> None:
    """El taller corta a mitad de una sierra para atender el mostrador."""
    from afilado.ordenes import Orden, Ordenes

    ordenes = Ordenes(str(taller / "data" / "ordenes"))
    ordenes.abrir(Orden.nueva("1234", 1, 4))
    _correr([_ESPACIO, ord("a"), ord("q")], monkeypatch)

    _correr([_ESPACIO, ord("a"), ord("q")], monkeypatch)

    assert _ordenes(taller).abierta.total_revisados == 2


def test_sin_orden_abierta_todo_sigue_funcionando(taller: Path, monkeypatch) -> None:
    """La orden es opcional: el que solo quiere mirar una sierra no tiene que cargar nada."""
    _correr([_ESPACIO, ord("a"), ord("q")], monkeypatch)

    assert len(_fotos(taller, "aprobada")) == 1
    assert _ordenes(taller).abierta is None


# ---------------------------------------------------------------------------
# El analisis POR DIENTE, de punta a punta
# ---------------------------------------------------------------------------


def _dos_dientes(lado: int = 240) -> np.ndarray:
    """Dos dientes uno al lado del otro, con la sombra del hueco en el medio."""
    izq, der = _diente(0, lado), _diente(4, lado)
    sombra = np.full((lado, 40, 3), 15, np.uint8)
    return np.hstack([izq, sombra, der])


def test_con_dos_dientes_se_archiva_uno_solo(taller: Path, monkeypatch) -> None:
    """Marcar los dos con el mismo estado de un saque etiquetaria mal el que no se miro."""
    class _Dos(_CamaraFalsa):
        def leer(self):
            self.leidos += 1
            return _dos_dientes()

    monkeypatch.setattr(run_live, "Camara", _Dos)

    _correr([_ESPACIO, ord("a"), ord("q")], monkeypatch)

    assert len(_fotos(taller, "aprobada")) == 1


def test_lo_que_se_archiva_es_el_RECORTE_del_diente(taller: Path, monkeypatch) -> None:
    """La memoria juzga el recorte, asi que lo que se guarda tiene que ser lo mismo: si no,
    lo que el programa aprende y lo que compara serian dos cosas distintas."""
    class _Dos(_CamaraFalsa):
        def leer(self):
            self.leidos += 1
            return _dos_dientes()

    monkeypatch.setattr(run_live, "Camara", _Dos)
    entera = _dos_dientes()

    _correr([_ESPACIO, ord("a"), ord("q")], monkeypatch, recuadros=True)

    guardada = cv2.imread(str(_fotos(taller, "aprobada")[0]))
    assert guardada.shape[1] < entera.shape[1] * 0.75


def test_la_tecla_2_cambia_cual_diente_se_guarda(taller: Path, monkeypatch) -> None:
    class _Dos(_CamaraFalsa):
        def leer(self):
            self.leidos += 1
            return _dos_dientes()

    monkeypatch.setattr(run_live, "Camara", _Dos)

    _correr([_ESPACIO, ord("1"), ord("a"), ord("q")], monkeypatch, recuadros=True)
    primero = cv2.imread(str(_fotos(taller, "aprobada")[0]))
    _correr([_ESPACIO, ord("2"), ord("r"), ord("q")], monkeypatch, recuadros=True)
    segundo = cv2.imread(str(_fotos(taller, "rechazada")[0]))

    assert primero.shape != segundo.shape or not np.array_equal(primero, segundo)


def test_con_dos_dientes_la_orden_anota_de_a_uno(taller: Path, monkeypatch) -> None:
    from afilado.ordenes import Orden, Ordenes

    class _Dos(_CamaraFalsa):
        def leer(self):
            self.leidos += 1
            return _dos_dientes()

    monkeypatch.setattr(run_live, "Camara", _Dos)
    Ordenes(str(taller / "data" / "ordenes")).abrir(Orden.nueva("1234", 1, 6))

    _correr([_ESPACIO, ord("1"), ord("a"), ord("2"), ord("r"), ord("q")], monkeypatch, recuadros=True)

    guardada = Ordenes(str(taller / "data" / "ordenes")).abierta
    assert guardada.total_revisados == 2
    assert guardada.sierras[0].revisados[0].estado == "aprobada"
    assert guardada.sierras[0].revisados[1].estado == "rechazada"


def test_con_un_diente_todo_sigue_como_antes(taller: Path, monkeypatch) -> None:
    _correr([_ESPACIO, ord("a"), ord("q")], monkeypatch)

    assert len(_fotos(taller, "aprobada")) == 1


# ---------------------------------------------------------------------------
# La captura automatica dentro del bucle
# ---------------------------------------------------------------------------


def _correr_automatico(
    taller: Path, monkeypatch, camara, teclas, dientes: int = 6, nitidez: float = 1.0
):
    """Arranca una orden, pone esa camara y deja correr el bucle."""
    from afilado.ordenes import Orden, Ordenes

    Ordenes(str(taller / "data" / "ordenes")).abrir(Orden.nueva("1234", 1, dientes))
    # Una foto de modelo, para que haya contra que comparar y exista un veredicto.
    modelo = taller / "data" / "referencia" / "microscopio" / "modelo.jpg"
    modelo.parent.mkdir(parents=True, exist_ok=True)
    assert cv2.imwrite(str(modelo), _diente(0))

    monkeypatch.setattr(run_live, "Camara", camara)
    pendientes = list(teclas)

    def _waitKey(_e: int = 0) -> int:
        return pendientes.pop(0) if pendientes else _ESC

    monkeypatch.setattr(cv2, "waitKey", _waitKey)
    cfg = config_mod.AppConfig(clases=["aprobada", "rechazada"], clases_defecto=["rechazada"])
    cfg.interfaz.modo = "simple"
    cfg.interfaz.evaluar_en_vivo = True
    cfg.interfaz.nitidez_minima = nitidez
    cfg.interfaz.segundos_entre_fotos = 0.05      # para no esperar cinco segundos reales
    cfg.detector.tarea = "classify"
    cfg.memoria.minimo_por_concepto = 1
    run_live._ejecutar_revision(cfg, argparse.Namespace(grabar=None))
    return Ordenes(str(taller / "data" / "ordenes")).abierta


class _Quieta(_CamaraFalsa):
    """La misma pieza, quieta: solo el temblor del sensor."""

    def leer(self):
        self.leidos += 1
        base = _diente(0)
        ruido = np.random.default_rng(self.leidos).integers(-2, 3, base.shape, dtype=np.int16)
        return np.clip(base.astype(np.int16) + ruido, 0, 255).astype(np.uint8)


def test_con_la_orden_abierta_el_diente_se_anota_solo(taller: Path, monkeypatch) -> None:
    """El operario pasa la sierra y no toca una tecla."""
    orden = _correr_automatico(taller, monkeypatch, _Quieta, [255] * 400 + [ord("q")])

    assert orden is not None
    assert orden.total_revisados >= 1
    primero = orden.sierras[0].revisados[0]
    assert "anotado solo" in primero.origen


def test_una_sierra_quieta_no_llena_el_informe_sola(taller: Path, monkeypatch) -> None:
    """La prueba que mas importa: sin mover la pieza, se anota UN diente y nada mas.

    Sin esto, una sierra olvidada sobre la mesa diria en el informe que se reviso entera
    diente por diente, y ese informe es justo el que contesta un reclamo."""
    orden = _correr_automatico(taller, monkeypatch, _Quieta, [255] * 1200 + [ord("q")])

    assert orden.total_revisados == 1


def test_sin_orden_abierta_no_se_anota_nada(taller: Path, monkeypatch) -> None:
    """La captura automatica es de la orden: sin orden, el programa no cuenta dientes."""
    monkeypatch.setattr(run_live, "Camara", _Quieta)
    _correr([255] * 400 + [ord("q")], monkeypatch)

    assert _ordenes(taller).abierta is None


def test_el_cartel_y_el_recuadro_dicen_lo_mismo(taller: Path, monkeypatch) -> None:
    """Paso en el taller: 84% en verde en el cartel y 40% en rojo sobre el diente, los dos
    de la misma pieza. El recuadro comparaba el RECORTE del diente contra la foto ENTERA
    del modelo, que es comparar dos cosas distintas."""
    estados: list = []
    original = run_live.revision.componer_simple

    def espiar(vivo, referencia, cfg, estado, alto=None):
        if estado.congelado and estado.dientes:
            estados.append((estado.parecido, [d.parecido for d in estado.dientes]))
        return original(vivo, referencia, cfg, estado, alto)

    monkeypatch.setattr(run_live.revision, "componer_simple", espiar)
    modelo = taller / "data" / "referencia" / "microscopio" / "modelo.jpg"
    modelo.parent.mkdir(parents=True, exist_ok=True)
    assert cv2.imwrite(str(modelo), _diente(0))

    _correr([_ESPACIO] + [255] * 5 + [ord("q")], monkeypatch)

    assert estados, "no se llego a dibujar la pantalla congelada"
    del_cartel, de_los_dientes = estados[-1]
    assert len(de_los_dientes) == 1
    assert del_cartel == pytest.approx(de_los_dientes[0])


class _OtroDiente(_CamaraFalsa):
    """La sierra avanza un diente DE VERDAD entre una tecla y la otra.

    _CamaraFalsa cambia la foto en cada lectura, pero MEDIDO con la firma de
    automatico.huella(), dos cuadros seguidos de _diente(n%7) distan 0.0207-0.0250: menos
    que los 0.04 de "es la misma foto", asi que para el programa siguen siendo el mismo
    diente. Dos dientes consecutivos de verdad del taller distan 0.1048 como minimo, y
    _diente(0) contra _diente(6) da 0.1231, que es lo que hace falta para que el freno de
    la 'g' los deje pasar como dos dientes.
    """

    def leer(self):
        self.leidos += 1
        return _diente(0) if self.leidos < 2 else _diente(6)


def test_la_tecla_g_anota_el_diente_sin_archivar_la_foto(taller: Path, monkeypatch) -> None:
    """Son dos cosas distintas: archivar le ensena al programa y mete la foto en el
    dataset; anotar dice 'este diente ya lo mire' y llena el informe."""
    from afilado.ordenes import Orden, Ordenes

    Ordenes(str(taller / "data" / "ordenes")).abrir(Orden.nueva("1234", 1, 4))
    monkeypatch.setattr(run_live, "Camara", _OtroDiente)

    _correr([ord("g"), ord("g"), ord("q")], monkeypatch)

    guardada = _ordenes(taller).abierta
    assert guardada.total_revisados == 2
    assert "lo anoto el operario" in guardada.sierras[0].revisados[0].origen
    # Y NO se archivo ninguna foto: el dataset no se toca.
    assert _fotos(taller, "aprobada") == []


def test_sin_orden_abierta_la_g_lo_dice(taller: Path, monkeypatch) -> None:
    _correr([ord("g"), ord("q")], monkeypatch)

    assert _ordenes(taller).abierta is None


def test_el_boton_de_anotar_aparece_con_la_orden_abierta() -> None:
    from afilado import revision
    from afilado.config import AppConfig

    cfg = AppConfig(clases=["afilado", "mal_afilado"], clases_defecto=["mal_afilado"])
    imagen = np.full((480, 640, 3), 120, np.uint8)

    sin_orden = revision.componer_simple(imagen, imagen, cfg, revision.EstadoRevision(), 672)
    con_orden = revision.componer_simple(
        imagen, imagen, cfg,
        revision.EstadoRevision(orden_resumen="ORDEN 1234 - sierra 1 de 1 - diente 3 de 24"),
        672,
    )

    assert "anotar" not in sin_orden.botones
    assert "anotar" in con_orden.botones


# ---------------------------------------------------------------------------
# LO QUE PASA DELANTE DEL MICROSCOPIO MIENTRAS LA SIERRA ESTA QUIETA.
# test_una_sierra_quieta_no_llena_el_informe_sola usa una pieza quieta en una
# escena donde no pasa nada, y en ese mundo el freno nunca se ponia a prueba.
# En el taller cruza una mano, parpadea el tubo fluorescente y el operario tapa
# el microscopio con la palma para acomodar la pieza. Cada una de esas tres
# cosas alcanzaba para que la MISMA pieza quieta se anotara de nuevo.
# ---------------------------------------------------------------------------


class _QuietaConMano(_Quieta):
    """La pieza quieta, y cada tanto una mano cruzando por delante.

    MEDIDO: con el freno de antes, esto dejaba anotados los SEIS dientes de la sierra en
    esta misma prueba, todos con el mismo parecido.
    """

    def leer(self):
        imagen = super().leer()
        if self.leidos % 17 == 0:
            imagen[:, 40:160] = 20
        return imagen


class _QuietaConParpadeo(_Quieta):
    """El tubo fluorescente del taller, que no es continuo.

    MEDIDO: este caso ya pasaba antes del arreglo. La huella se toma normalizada, asi que
    un cambio de exposicion no la mueve. Queda como guardia de esa propiedad.
    """

    def leer(self):
        imagen = super().leer()
        if self.leidos % 13 == 0:
            imagen = np.clip(imagen.astype(np.float32) * 0.55, 0, 255).astype(np.uint8)
        return imagen


class _QuietaYTapada(_Quieta):
    """La pieza quieta, y a partir del cuadro 120 el microscopio tapado con la mano."""

    def leer(self):
        imagen = super().leer()
        if self.leidos > 120:
            return cv2.GaussianBlur(np.full_like(imagen, 28), (31, 31), 0)
        return imagen


@pytest.mark.parametrize("camara", [_QuietaConMano, _QuietaConParpadeo])
def test_lo_que_cruza_por_delante_no_llena_el_informe(taller: Path, monkeypatch, camara) -> None:
    orden = _correr_automatico(taller, monkeypatch, camara, [255] * 400 + [ord("q")])

    assert orden.total_revisados == 1, "la misma pieza quieta se anoto varias veces"


def test_con_el_microscopio_tapado_no_se_anota_nada(taller: Path, monkeypatch) -> None:
    """Sin esto, el analisis se saltea por falta de foco pero el veredicto de antes queda
    pegado en pantalla, y la captura automatica lo anotaba como si fuera de ahora: el
    informe terminaba con el parecido de la foto de antes de tapar el microscopio."""
    orden = _correr_automatico(
        taller, monkeypatch, _QuietaYTapada, [255] * 400 + [ord("q")], nitidez=5.0
    )

    assert orden.sierras[0].revisados[0].origen.endswith("anotado solo")
    assert orden.total_revisados == 1


@pytest.mark.parametrize("tecla", [ord("a"), ord("g")])
def test_anotar_a_mano_no_deja_el_diente_contado_dos_veces(
    taller: Path, monkeypatch, tecla
) -> None:
    """'a' archiva y anota, 'g' solo anota. En los dos casos ese diente ya esta contado, y
    la captura automatica tiene que enterarse: con la pieza quieta, una sola pulsacion
    dejaba DOS renglones para el mismo diente --el del operario y el del programa un
    segundo despues-- y el informe decia que se habian revisado dos."""
    orden = _correr_automatico(taller, monkeypatch, _Quieta, [tecla] + [255] * 400 + [ord("q")])

    assert orden.total_revisados == 1
    assert "el operario" in orden.sierras[0].revisados[0].origen


def test_con_la_imagen_congelada_la_g_no_gasta_un_diente_por_pulsacion(
    taller: Path, monkeypatch
) -> None:
    """REPRODUCIDO: congelada UNA foto y diez 'g', el informe decia diez dientes revisados
    --los diez con el mismo parecido y el mismo instante-- y la foto era una. OpenCV
    repite el evento si se deja la tecla apretada, asi que no hace falta ni quererlo."""
    from afilado.ordenes import Orden, Ordenes

    Ordenes(str(taller / "data" / "ordenes")).abrir(Orden.nueva("1234", 1, 24))

    _correr([_ESPACIO] + [ord("g")] * 10 + [ord("q")], monkeypatch)

    assert _ordenes(taller).abierta.total_revisados == 1


def test_con_mas_de_24_modelos_las_fotos_no_se_releen_en_cada_analisis(
    taller: Path, monkeypatch
) -> None:
    """La cache comparaba sus 24 nombres contra TODOS los de la carpeta, asi que arriba de
    24 nunca validaba y las 24 fotos volvian del disco en cada analisis. Medido con 60
    cuadros de 640x480 y clasificar_cada_n_frames=5: 289 cv2.imread y 21.6 ms por cuadro
    con 25 fotos, contra 25 imread y 13.5 ms."""
    carpeta = taller / "data" / "referencia" / "microscopio"
    carpeta.mkdir(parents=True, exist_ok=True)
    for i in range(25):
        assert cv2.imwrite(str(carpeta / f"mod{i:02d}.jpg"), _diente(i % 7))

    leida_de_verdad = cv2.imread
    leidas: list = []

    def contar(ruta, *resto):
        leidas.append(str(ruta))
        return leida_de_verdad(ruta, *resto)

    monkeypatch.setattr(cv2, "imread", contar)

    _correr([255] * 30 + [ord("q")], monkeypatch)

    de_modelo = [r for r in leidas if "microscopio" in r]
    assert len(de_modelo) <= 26, (
        f"las fotos de modelo se releyeron en cada analisis: {len(de_modelo)} lecturas"
    )


def test_congelado_cambiar_de_modelo_mueve_el_porcentaje(taller: Path, monkeypatch) -> None:
    """Congelado, la 'm' cambiaba la foto y el nombre de la cabecera pero no el numero: la
    pantalla mostraba dos cosas que no eran de la misma pieza. Medido con dos dientes a la
    vista y tres modelos, los parecidos quedaban clavados en (0.6823, 0.6173) mientras la
    cabecera iba de mod0 a mod2."""
    class _Dos(_CamaraFalsa):
        def leer(self):
            self.leidos += 1
            return _dos_dientes()

    monkeypatch.setattr(run_live, "Camara", _Dos)
    carpeta = taller / "data" / "referencia" / "microscopio"
    carpeta.mkdir(parents=True, exist_ok=True)
    for i in range(3):
        assert cv2.imwrite(str(carpeta / f"mod{i}.jpg"), _diente(i * 3))

    vistos: list = []
    original = run_live.revision.componer_simple

    def espiar(vivo, referencia, cfg, estado, alto=None):
        if estado.congelado and estado.dientes:
            vistos.append(tuple(d.parecido for d in estado.dientes))
        return original(vivo, referencia, cfg, estado, alto)

    monkeypatch.setattr(run_live.revision, "componer_simple", espiar)

    _correr([_ESPACIO, 255, ord("m"), 255, ord("m"), 255, ord("q")], monkeypatch, recuadros=True)

    assert len(vistos) > 2, "no se llego a dibujar la pantalla congelada"
    assert len(set(vistos)) > 1, "se cambio de modelo y el porcentaje no se movio"


def test_el_recorte_que_se_archiva_sale_del_cuadro_que_se_midio(
    taller: Path, monkeypatch
) -> None:
    """El recuadro se mide uno de cada cada_n_frames cuadros, y recortarlo de la foto de
    AHORA lo deja corrido lo que se movio la mano. Reproducido con la pieza corriendo
    12 px por cuadro: el recuadro era (87,158,302,342) y la pieza ya estaba en
    (135,158,350,342), 48 px a la derecha."""
    pieza = _diente(0, 240)
    entregados: list = []

    def cuadro(n: int):
        fondo = np.full((480, 640, 3), 12, np.uint8)
        x = 20 + (n * 12) % 340
        fondo[120:360, x:x + 240] = pieza
        return fondo

    class _Movil(_CamaraFalsa):
        def leer(self):
            self.leidos += 1
            foto = cuadro(self.leidos)
            entregados.append(foto)
            return foto

    monkeypatch.setattr(run_live, "Camara", _Movil)
    carpeta = taller / "data" / "referencia" / "microscopio"
    carpeta.mkdir(parents=True, exist_ok=True)
    assert cv2.imwrite(str(carpeta / "mod0.jpg"), pieza)

    cajas: list = []
    original = run_live.revision.componer_simple

    def espiar(vivo, referencia, cfg, estado, alto=None):
        cajas.append((len(entregados), [d.caja for d in estado.dientes if d.elegido]))
        return original(vivo, referencia, cfg, estado, alto)

    monkeypatch.setattr(run_live.revision, "componer_simple", espiar)

    # El config del TALLER: se analiza uno de cada cinco cuadros. La 'a' cae en el cuadro
    # 9, cuatro cuadros despues del ultimo analisis, que fue el del cuadro 5.
    pendientes = [255] * 8 + [ord("a")] + [ord("q")]
    monkeypatch.setattr(
        cv2, "waitKey", lambda *_a: pendientes.pop(0) if pendientes else _ESC
    )
    cfg = config_mod.AppConfig(clases=["aprobada", "rechazada"], clases_defecto=["rechazada"])
    cfg.interfaz.modo = "simple"
    cfg.interfaz.evaluar_en_vivo = True
    cfg.interfaz.nitidez_minima = 1.0
    cfg.detector.tarea = "classify"
    cfg.detector.clasificar_cada_n_frames = 5
    cfg.memoria.minimo_por_concepto = 1
    run_live._ejecutar_revision(cfg, argparse.Namespace(grabar=None))

    archivadas = _fotos(taller, "aprobada")
    assert len(archivadas) == 1
    guardada = cv2.imread(str(archivadas[0]))
    caja = next(c for n, c in cajas if n == 9 and c)[0]
    del_analisis = run_live.dientes.Diente(caja=caja, area=0.0).recortar(entregados[4])
    del_momento = run_live.dientes.Diente(caja=caja, area=0.0).recortar(entregados[8])
    dif_analisis = float(np.abs(guardada.astype(int) - del_analisis.astype(int)).mean())
    dif_momento = float(np.abs(guardada.astype(int) - del_momento.astype(int)).mean())
    assert dif_analisis < dif_momento, (
        "lo archivado salio del cuadro de la tecla y el recuadro del cuadro anterior"
    )


class _CamaraCongelada(_CamaraFalsa):
    """El microscopio desenchufado: DroidCam deja un cartel fijo y la camara lo sabe.

    El control de foco NO lo atrapa --el cartel es texto blanco nitido sobre negro-- asi
    que hay que preguntarle a la camara, que es lo que ya hace motivo_para_no_archivar.
    """

    congelada = True

    def leer(self):
        self.leidos += 1
        imagen = np.zeros((480, 640, 3), np.uint8)
        cv2.putText(imagen, "Start DroidCam", (40, 240), cv2.FONT_HERSHEY_SIMPLEX, 1.6,
                    (255, 255, 255), 3, cv2.LINE_AA)
        return imagen


def test_con_la_camara_congelada_la_captura_automatica_no_anota(
    taller: Path, monkeypatch
) -> None:
    """REPRODUCIDO con el taller de hoy: con el microscopio desenchufado la captura
    automatica anotaba un diente de la orden del cliente --'afilado', 32%, RECHAZADO-- que
    era el cartel de DroidCam. El camino de a mano ya lo frenaba ('CAMARA CONGELADA -
    conecte el microscopio'); este preguntaba por la imagen y no por la camara."""
    orden = _correr_automatico(
        taller, monkeypatch, _CamaraCongelada, [255] * 400 + [ord("q")]
    )

    assert orden.total_revisados == 0


def test_la_foto_de_modelo_se_recorta_con_la_misma_regla_que_la_de_la_camara(
    taller: Path, monkeypatch
) -> None:
    """Con maximo=1 el recuadro que queda es el de MASCARA mas grande, y en las fotos del
    taller esa es la del hueco: medido sobre afilado_20261112_161626_660, maximo=1
    devuelve (84,300,261,447) --el hueco-- y la pantalla elige (237,63,323,308) --la cara
    de widia, rayado 43.6 contra 30.4--. Si la foto de modelo se recorta por el hueco,
    todo lo que se compare contra ella se mide contra un hueco."""
    pedidos: list[int] = []
    original = run_live.dientes.detectar

    def espiar(frame, maximo=4):
        pedidos.append(maximo)
        return original(frame, maximo)

    monkeypatch.setattr(run_live.dientes, "detectar", espiar)
    modelo = taller / "data" / "referencia" / "microscopio" / "modelo.jpg"
    modelo.parent.mkdir(parents=True, exist_ok=True)
    assert cv2.imwrite(str(modelo), _diente(0))

    _correr([_ESPACIO] + [255] * 3 + [ord("q")], monkeypatch)

    assert pedidos, "no se llamo al detector ni una vez"
    assert 1 not in pedidos, (
        "la foto de modelo se recorta con maximo=1 y la de la camara con maximo=4: "
        "dos reglas distintas para la misma pieza"
    )


def test_dos_modelos_con_el_mismo_nombre_no_comparten_el_recorte(
    taller: Path, monkeypatch
) -> None:
    """galeria.nombre_actual viene SIN extension, asi que modelo.jpg y modelo.png
    compartian el recorte de la cache. Medido con dos dientes a la vista: la 'm' dejaba
    los parecidos clavados en (0.6823, 0.6173) y la cabecera decia "modelo" las dos
    veces, asi que el operario no tenia como notarlo."""
    class _Dos(_CamaraFalsa):
        def leer(self):
            self.leidos += 1
            return _dos_dientes()

    monkeypatch.setattr(run_live, "Camara", _Dos)
    carpeta = taller / "data" / "referencia" / "microscopio"
    carpeta.mkdir(parents=True, exist_ok=True)
    assert cv2.imwrite(str(carpeta / "modelo.jpg"), _diente(0, 240))
    assert cv2.imwrite(str(carpeta / "modelo.png"), _dos_dientes(240))

    vistos: list = []
    original = run_live.revision.componer_simple

    def espiar(vivo, referencia, cfg, estado, alto=None):
        if estado.dientes:
            vistos.append(tuple(d.parecido for d in estado.dientes))
        return original(vivo, referencia, cfg, estado, alto)

    monkeypatch.setattr(run_live.revision, "componer_simple", espiar)

    _correr([255, ord("m"), 255, 255, ord("q")], monkeypatch, recuadros=True)

    assert len(set(vistos)) > 1, (
        "se paso a la otra foto de modelo y el parecido no se movio: comparten el recorte"
    )


def test_al_archivar_el_aviso_se_pinta_antes_de_medirse(taller: Path, monkeypatch) -> None:
    """Medirse cuesta --118 ms con 15 fotos del microscopio, 575 ms con 41 y 1554 ms con
    121-- y corre adentro del mismo cuadro que archiva: mientras tanto la ventana se queda
    con el cuadro anterior y SIN aviso, y el operario que no ve nada vuelve a apretar."""
    from afilado.memoria import Memoria

    secuencia: list = []
    medirse = Memoria.autoevaluacion

    def medirse_espiada(self):
        secuencia.append("se mide")
        return medirse(self)

    monkeypatch.setattr(Memoria, "autoevaluacion", medirse_espiada)

    original = run_live.revision.componer_simple

    def espiar(vivo, referencia, cfg, estado, alto=None):
        if estado.mensaje and "GUARDADA" in estado.mensaje:
            secuencia.append("se pinta el aviso")
        return original(vivo, referencia, cfg, estado, alto)

    monkeypatch.setattr(run_live.revision, "componer_simple", espiar)

    _correr([ord("a"), 255, ord("q")], monkeypatch)

    assert "se pinta el aviso" in secuencia, "nunca se pinto el aviso de GUARDADA"
    # Despues del aviso TIENE que quedar una medicion: comparar contra la PRIMERA
    # 'se mide' hace fallar la prueba cuando la coleccion ya tiene fotos, porque ahi
    # memoria.resumen() del arranque ya se midio una vez.
    assert "se mide" in secuencia[secuencia.index("se pinta el aviso"):], (
        f"la ventana se queda sin aviso mientras la memoria se mide: {secuencia}"
    )


def test_al_archivar_queda_escrito_lo_que_mostraba_la_pantalla(
    taller: Path, monkeypatch
) -> None:
    """La columna 'parecido' es cuanto se parecio AL ESTADO que puso el operario, asi que
    por este camino queda vacia y esta bien. Pero lo que mostraba la pantalla --el
    concepto y el porcentaje con los que el operario decidio-- no puede perderse: es la
    otra mitad de lo que un reclamo necesita ver."""
    from afilado.ordenes import Orden, Ordenes

    Ordenes(str(taller / "data" / "ordenes")).abrir(Orden.nueva("1234", 1, 4))

    # La primera 'a' deja una foto en la memoria; para la segunda ya hay con que comparar.
    _correr([_ESPACIO, ord("a"), ord(" "), ord("a"), ord("q")], monkeypatch)

    segundo = _ordenes(taller).abierta.sierras[0].revisados[1]
    assert segundo.parecido is None
    assert "en pantalla decia" in segundo.origen and "%" in segundo.origen


def test_el_renglon_que_se_contradice_viene_explicado() -> None:
    """"afilado RECHAZADO 42%" en el mismo renglon: el estado es el concepto de la foto
    archivada que mas se parecio y el APROBADO/RECHAZADO sale de si ese parecido llega al
    umbral, dos criterios distintos. Reproducido en el informe de verdad con el taller de
    hoy. Ningun numero es falso; lo que falla es que el papel de un reclamo se contradice
    y no dice por que."""
    from afilado import revision
    from afilado.config import AppConfig

    cfg = AppConfig(clases=["afilado", "mal_afilado"], clases_defecto=["mal_afilado"])
    cfg.interfaz.umbral_parecido = 0.70
    estado = revision.EstadoRevision(dientes=[revision.DienteEnPantalla(
        caja=(0, 0, 100, 100), parecido=0.42, aprobado=False, concepto="afilado",
        elegido=True,
    )])

    resultado = run_live._resultado_del_diente(estado, cfg)

    assert resultado.estado == "afilado" and resultado.aprobado is False
    assert "42%" in resultado.origen and "70% para aprobar" in resultado.origen


def test_la_tecla_g_anota_de_donde_salio_el_veredicto(taller: Path, monkeypatch) -> None:
    """El origen es "de donde salio el veredicto", no quien apreto la tecla: asi esta
    documentado en ordenes.Diente.origen. Decia "lo anoto el operario" al lado de un
    APROBADO que habia puesto el programa, y en el .csv eso se lee igual que el renglon de
    la tecla 'a', donde el estado SI lo eligio el operario."""
    orden = _correr_automatico(taller, monkeypatch, _Quieta, [ord("g"), ord("q")])

    diente = orden.sierras[0].revisados[0]
    assert diente.aprobado is not None, "sin veredicto en pantalla no prueba nada"
    assert "lo anoto el operario" in diente.origen
    assert diente.origen != "lo anoto el operario", "eso dice quien apreto la tecla"


def test_la_g_sin_veredicto_deja_dicho_que_no_habia_veredicto(
    taller: Path, monkeypatch
) -> None:
    """Sin modelo y sin memoria el renglon se iba en blanco --el informe lo imprime como
    'sin decidir'-- contado como revisado y sin ninguna nota, que es como se ve un diente
    que el programa no se animo a juzgar. No es lo mismo: no habia nada que juzgar."""
    from afilado.ordenes import Orden, Ordenes

    Ordenes(str(taller / "data" / "ordenes")).abrir(Orden.nueva("1234", 1, 4))

    _correr([ord("g"), ord("q")], monkeypatch)

    diente = _ordenes(taller).abierta.sierras[0].revisados[0]
    assert diente.estado is None and diente.aprobado is None
    assert diente.nota != "", "renglon contado como revisado, en blanco y sin explicacion"


# ---------------------------------------------------------------------------
# LOS RECUADROS, APAGADOS. El taller lo dijo asi: "capta cualquier imagen".
# ---------------------------------------------------------------------------


class _ContraluzFalsa(_CamaraFalsa):
    """Como ilumina el taller desde el 13/11: el diente sale NEGRO contra un fondo
    blanco. El detector busca lo MAS CLARO, asi que se queda con el fondo y el recuadro
    abarca el cuadro entero: medido sobre las dos fotos reales de ese dia, caja
    (0,0,640,480) en las dos."""

    def leer(self):
        self.leidos += 1
        img = np.full((480, 640, 3), 235, np.uint8)
        cv2.rectangle(img, (240, 60), (400, 420), (25, 25, 25), -1)
        cv2.rectangle(img, (40, 180), (150, 440), (30, 30, 30), -1)
        return cv2.GaussianBlur(img, (5, 5), 0)


def test_con_la_luz_de_hoy_el_detector_marca_el_cuadro_entero() -> None:
    """Por que se apagaron: el detector busca lo MAS CLARO y a contraluz el diente es lo
    mas oscuro, asi que se queda con el fondo. Medido sobre las dos fotos reales que saco
    el taller el 13/11 con esta luz: caja (0,0,640,480) en las dos, el cuadro entero
    marcado como si fuera un diente."""
    from afilado import dientes as D

    imagen = _ContraluzFalsa().leer()
    alto, ancho = imagen.shape[:2]

    encontrados = D.detectar(imagen)

    assert len(encontrados) == 1
    x1, y1, x2, y2 = encontrados[0].caja
    assert (x2 - x1) * (y2 - y1) >= 0.95 * alto * ancho, (
        f"el recuadro no abarca todo: {encontrados[0].caja}"
    )


def test_apagados_lo_que_se_archiva_es_el_cuadro_entero(taller: Path, monkeypatch) -> None:
    """Y esto es lo que empareja la coleccion. Mientras hubo recuadros, lo archivado salia
    del recorte del diente elegido, y cuando el detector no separaba nada ese "recorte"
    era el cuadro completo: en la coleccion del taller convivian fotos de 640x446 y de
    253x257. Medido, la memoria separaba esos dos grupos 14 de 14 al 100% de confianza
    sin mirar el filo ni una vez.

    Se usa la camara de siempre --un diente CLARO sobre fondo oscuro-- porque esa es la
    que el detector si recorta: con la de contraluz el recuadro es el cuadro entero y
    archivaria lo mismo con los recuadros encendidos o apagados, o sea que no probaria
    nada."""
    _correr([_ESPACIO, ord("a"), ord("q")], monkeypatch)

    fotos = _fotos(taller, "aprobada")
    assert len(fotos) == 1
    guardada = cv2.imread(str(fotos[0]))
    entera = _diente(0)
    assert guardada.shape[:2] == entera.shape[:2], (
        f"se archivo un recorte de {guardada.shape[1]}x{guardada.shape[0]} en vez del "
        f"cuadro entero de {entera.shape[1]}x{entera.shape[0]}"
    )
