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


def _correr(teclas: list[int], monkeypatch: pytest.MonkeyPatch) -> int:
    """Corre el bucle mandandole esas teclas. Al quedarse sin teclas, sale."""
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

    _correr([_ESPACIO, ord("a"), ord("q")], monkeypatch)

    guardada = cv2.imread(str(_fotos(taller, "aprobada")[0]))
    assert guardada.shape[1] < entera.shape[1] * 0.75


def test_la_tecla_2_cambia_cual_diente_se_guarda(taller: Path, monkeypatch) -> None:
    class _Dos(_CamaraFalsa):
        def leer(self):
            self.leidos += 1
            return _dos_dientes()

    monkeypatch.setattr(run_live, "Camara", _Dos)

    _correr([_ESPACIO, ord("1"), ord("a"), ord("q")], monkeypatch)
    primero = cv2.imread(str(_fotos(taller, "aprobada")[0]))
    _correr([_ESPACIO, ord("2"), ord("r"), ord("q")], monkeypatch)
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

    _correr([_ESPACIO, ord("1"), ord("a"), ord("2"), ord("r"), ord("q")], monkeypatch)

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


def _correr_automatico(taller: Path, monkeypatch, camara, teclas, dientes: int = 6):
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
    cfg.interfaz.nitidez_minima = 1.0
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


def test_la_tecla_g_anota_el_diente_sin_archivar_la_foto(taller: Path, monkeypatch) -> None:
    """Son dos cosas distintas: archivar le ensena al programa y mete la foto en el
    dataset; anotar dice 'este diente ya lo mire' y llena el informe."""
    from afilado.ordenes import Orden, Ordenes

    Ordenes(str(taller / "data" / "ordenes")).abrir(Orden.nueva("1234", 1, 4))

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
