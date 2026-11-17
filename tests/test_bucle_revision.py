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
import os
import re
import shutil
import time
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
    # La 'o' abre el HISTORIAL y el ENTER entra a la orden en curso --que es la fila que
    # viene elegida-- recien ahi la 'i' saca el informe.
    _correr([_ESPACIO, ord("a"), ord(" "), ord("r"),
             ord("o"), _ENTER, ord("i"), ord("q")],
            monkeypatch)

    informe = (taller / "data" / "ordenes" / "informe_1234.txt").read_text(encoding="utf-8")
    assert "APROBADO" in informe and "RECHAZADO" in informe
    assert "(sin revisar)" in informe          # el tercero no se reviso y se dice
    assert (taller / "data" / "ordenes" / "informe_1234.csv").is_file()


def test_la_orden_se_carga_a_mano_desde_el_programa(taller: Path, monkeypatch) -> None:
    # La 'o' abre el historial y la 'n' el formulario: la orden nueva se carga desde ahi.
    teclas = [ord("o"), ord("n")]
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


# ---------------------------------------------------------------------------
# La 'o' del taller: por donde se entra a las ordenes
# ---------------------------------------------------------------------------


def test_la_o_del_taller_abre_el_historial_y_no_el_formulario(taller: Path, monkeypatch) -> None:
    """Sin orden en curso, la 'o' caia derecho en "cargar una orden nueva", asi que las
    ordenes ya guardadas no existian para el programa: la del reclamo que entra por
    telefono habia que buscarla con el Explorador. Ahora la 'o' muestra lo que hay, y la
    orden cerrada de la semana pasada se exporta con una tecla sin reabrirla."""
    from afilado import panel
    from afilado.ordenes import Orden, Ordenes

    ordenes = Ordenes(str(taller / "data" / "ordenes"))
    ordenes.abrir(Orden.nueva("99", 1, 4, cliente="Aserradero"))
    ordenes.cerrar()                       # no queda ninguna abierta, como en el taller hoy
    assert ordenes.abierta is None

    # Se anota que pantalla se dibujo en cada vuelta: la prueba es que la 'o' lleve al
    # historial, no que haya quedado un archivo de casualidad.
    vistos: list[str] = []
    componer_de_verdad = panel.componer

    def espiar(estado, alto, miniaturas=None):
        vistos.append(estado.modo)
        return componer_de_verdad(estado, alto, miniaturas)

    monkeypatch.setattr(panel, "componer", espiar)

    # 'o' abre el historial con la unica orden ya elegida, y la 'e' la exporta.
    _correr([ord("o"), ord("e"), _ESC, ord("q")], monkeypatch)

    assert panel.MODO_ORDENES in vistos
    assert panel.MODO_ORDEN not in vistos      # el formulario no se abrio solo
    zips = list((taller / "data" / "ordenes" / "exportado").glob("*.zip"))
    assert len(zips) == 1 and "99" in zips[0].name
    # Y no se creo ninguna orden nueva: la 'e' no fue una letra tecleada en un formulario.
    assert sorted(p.stem for p in (taller / "data" / "ordenes").glob("*.json")) == ["99"]


def test_la_pantalla_del_taller_tiene_un_boton_para_entrar_a_las_ordenes() -> None:
    """La tecla sola no alcanza: el que no sabe que la 'o' existe no la aprieta nunca.

    QUE SE ROMPIA: la orden de afilado se entraba desde el panel de CONCEPTOS, como si una
    orden fuera una cosa de los conceptos. Desde la pantalla del taller --la unica que el
    operario ve todo el dia-- no habia nada que dijera que el programa guarda ordenes, asi
    que la del reclamo que entra por telefono se buscaba con el Explorador de Windows.
    """
    from afilado import revision
    from afilado.config import AppConfig

    cfg = AppConfig(clases=["afilado", "mal_afilado"], clases_defecto=["mal_afilado"])
    imagen = np.full((480, 640, 3), 120, np.uint8)

    en_vivo = revision.componer_simple(imagen, imagen, cfg, revision.EstadoRevision(), 672)
    congelada = revision.componer_simple(
        imagen, imagen, cfg, revision.EstadoRevision(congelado=True), 672
    )

    assert "ordenes" in en_vivo.botones
    # Y tambien con la imagen congelada, que es donde el operario se queda mirando el
    # veredicto: es ahi donde se acuerda del reclamo de la manana.
    assert "ordenes" in congelada.botones


def test_el_clic_en_el_boton_ORDENES_abre_el_historial(taller: Path, monkeypatch) -> None:
    """El boton tiene que estar Y tiene que llevar a alguna parte. Un boton dibujado que no
    hace nada es peor que no tenerlo: la proxima vez el operario no le cree a ninguno.

    El clic entra por el callback de verdad de OpenCV --se encola y se atiende en el cuadro
    siguiente, igual que en el taller-- y el rectangulo sale de la pantalla que el bucle
    ACABA de dibujar, no de una composicion aparte: asi el clic cae donde se lo ve.
    """
    from afilado import panel

    vistos: list[str] = []
    componer_panel = panel.componer

    def espiar_panel(estado, alto, miniaturas=None):
        vistos.append(estado.modo)
        return componer_panel(estado, alto, miniaturas)

    monkeypatch.setattr(panel, "componer", espiar_panel)

    callback: list = []
    monkeypatch.setattr(cv2, "setMouseCallback", lambda ventana, cb: callback.append(cb))

    apretado: list = []
    componer_taller = run_live.revision.componer_simple

    def espiar_taller(vivo, referencia, cfg, estado, alto=None):
        pantalla = componer_taller(vivo, referencia, cfg, estado, alto)
        if callback and not apretado and "ordenes" in pantalla.botones:
            x1, y1, x2, y2 = pantalla.botones["ordenes"]
            callback[0](cv2.EVENT_LBUTTONDOWN, (x1 + x2) // 2, (y1 + y2) // 2, 0, None)
            apretado.append(True)
        return pantalla

    monkeypatch.setattr(run_live.revision, "componer_simple", espiar_taller)

    _correr([255, 255, ord("q")], monkeypatch)

    assert apretado, "el boton ORDENES no estaba en la pantalla del taller"
    assert panel.MODO_ORDENES in vistos
    # Y lleva al HISTORIAL, no al formulario de orden nueva: lo que se busca con ese boton
    # es una orden que ya existe.
    assert panel.MODO_ORDEN not in vistos


def _renglones_escritos(cfg, estado, monkeypatch) -> list[str]:
    """Todo lo que la pantalla del taller ESCRIBE, preguntandole al pincel.

    Los botones vuelven en Pantalla.botones, pero el renglon de ayuda --el que nombra con
    su tecla el boton que no entro-- se dibuja derecho con el pincel y no vuelve en ninguna
    lista. Se le pregunta al pincel, que es el que de verdad lo pone en pantalla, en vez de
    leer las letras de los pixeles.
    """
    from afilado import revision, tipografia

    escritos: list[str] = []
    original = tipografia.Pincel.texto

    def espiar(self, texto, org, color, tamano, negrita=False):
        escritos.append(texto)
        return original(self, texto, org, color, tamano, negrita)

    monkeypatch.setattr(tipografia.Pincel, "texto", espiar)
    imagen = np.full((480, 640, 3), 120, np.uint8)
    revision.componer_simple(imagen, imagen, cfg, estado, 672)
    return escritos


def test_el_boton_ORDENES_que_no_entra_queda_nombrado_con_su_tecla(monkeypatch) -> None:
    """EL CASO DEL TALLER: con los nombres de concepto de verdad --GUARDAR SIERRA_NUEVA_TP
    y GUARDAR DIENTES_ROTOS-- la barra de abajo se llena y el boton ORDENES NO ENTRA.
    Medido componiendo esta pantalla: la izquierda termina en x=866, despues de CONCEPTOS
    quedan 89 px y el boton pide 140.

    QUE SE ROMPIA: el renglon de ayuda decia solo "tecla c: conceptos", asi que en esa
    pantalla --la congelada, la que el operario mira para decidir si la sierra se tira--
    las ordenes no existian y no habia como llegar a ellas sin volver primero a la de en
    vivo. La tecla queda escrita, que es el unico camino que queda.
    """
    from afilado import revision
    from afilado.config import AppConfig

    cfg = AppConfig(clases=["sierra_nueva_tp", "dientes_rotos"],
                    clases_defecto=["dientes_rotos"])
    del_taller = [("sierra_nueva_tp", "GUARDAR SIERRA_NUEVA_TP", False),
                  ("dientes_rotos", "GUARDAR DIENTES_ROTOS", True)]
    estado = revision.EstadoRevision(congelado=True, botones_guardar=del_taller)
    # Primero, que de verdad no entro: si entrara, esta prueba no estaria probando nada.
    imagen = np.full((480, 640, 3), 120, np.uint8)
    pantalla = revision.componer_simple(imagen, imagen, cfg, estado, 672)
    assert "ordenes" not in pantalla.botones

    escritos = _renglones_escritos(cfg, estado, monkeypatch)

    ayuda = [texto for texto in escritos if texto.startswith("teclas")]
    assert ayuda == ["teclas:  o ordenes"]


def test_con_la_barra_libre_no_se_nombra_una_tecla_de_un_boton_que_esta(
    monkeypatch
) -> None:
    """La contracara, y la que evita el otro error: nombrar teclas de botones que SI estan
    llena el renglon de arriba de letras que nadie necesita, y con el renglon lleno el
    titulo SIERRA queda tapado. Solo se nombra lo que no entro."""
    from afilado.config import AppConfig
    from afilado import revision

    cfg = AppConfig(clases=["aprobada", "rechazada"], clases_defecto=["rechazada"])

    escritos = _renglones_escritos(cfg, revision.EstadoRevision(), monkeypatch)

    assert not [texto for texto in escritos if texto.startswith("teclas")]


@pytest.mark.parametrize("camara, hasta_completarla", [
    # ARCHIVAR LA FOTO: el espacio congela, la 'a' la guarda en el primer concepto sano y
    # anota el diente. La segunda 'a' ya no tiene diente donde anotar y sale el cartel.
    (_CamaraFalsa, [_ESPACIO, ord("a"), _ESPACIO, ord("a")]),
    # ANOTAR A MANO: la 'g' no archiva nada, solo llena el informe. Hace falta que la
    # sierra avance DE VERDAD entre las dos 'g' --por eso la camara que cambia de diente--
    # o el freno de la foto repetida las junta en un solo diente y la sierra no se completa.
    (_OtroDiente, [ord("g"), ord("g")]),
])
def test_el_cartel_de_la_sierra_completa_nombra_las_teclas_que_pasan_de_sierra(
    taller: Path, monkeypatch, camara, hasta_completarla
) -> None:
    """Los dos caminos por los que se completa una sierra a mano dan este cartel, y los dos
    lo decian mal. Se prueban los dos: son dos pedazos de codigo distintos y el dia que
    alguien arregle uno, el otro sigue mintiendo.

    QUE SE ROMPIA: el cartel decia "pase a la siguiente con 'o'", y la 'o' NO pasa de
    sierra: abre el historial. El operario con la sierra 2 en la mano apretaba 'o', le
    aparecia una lista de ordenes que no habia pedido, y la sierra siguiente se le anotaba
    entera encima de la sierra 1 --todos los dientes en la misma columna del informe--
    hasta que alguien se diera cuenta.

    Aca se lee el cartel y DESPUES se aprietan las teclas que nombra, con el bucle de
    verdad: si el cartel dijera otra cosa, o si esas teclas hicieran otra cosa, falla. Y se
    prueban las dos mitades --la 'o' sola y las tres juntas-- porque la primera es la que
    hace que el cartel viejo fuera mentira y la segunda la que hace que el nuevo sea
    cierto; de a una no alcanza ninguna.
    """
    from afilado.ordenes import Orden, Ordenes

    carteles: list[str] = []
    original = run_live.revision.componer_simple

    def espiar(vivo, referencia, cfg, estado, alto=None):
        if estado.mensaje:
            carteles.append(estado.mensaje)
        return original(vivo, referencia, cfg, estado, alto)

    monkeypatch.setattr(run_live.revision, "componer_simple", espiar)
    monkeypatch.setattr(run_live, "Camara", camara)
    ordenes = Ordenes(str(taller / "data" / "ordenes"))
    ordenes.abrir(Orden.nueva("7001", 3, 1))

    # PRIMERA VUELTA. Sierras de UN diente: el primer anotado la completa y el siguiente ya
    # no tiene donde, asi que sale el cartel. Y despues la 'o' SOLA, que es lo que decia el
    # cartel viejo.
    _correr(hasta_completarla + [ord("o"), ord("q")], monkeypatch)

    completa = [texto for texto in carteles if "COMPLETA" in texto]
    assert completa, f"nunca salio el cartel de la sierra completa: {carteles}"
    # El cartel nombra las TRES teclas, en el orden en que se aprietan.
    assert "'o', ENTER y 's'" in completa[0]
    # Y la 'o' sola NO cambio de sierra: por eso el cartel viejo era mentira.
    assert Ordenes(str(taller / "data" / "ordenes")).abierta.sierra_actual == 1

    # SEGUNDA VUELTA: las tres teclas que el cartel nombra, en ese orden.
    _correr([ord("o"), _ENTER, ord("s"), ord("q"), ord("q")], monkeypatch)

    assert any("SIERRA 2 de 3" in texto for texto in carteles)
    assert Ordenes(str(taller / "data" / "ordenes")).abierta.sierra_actual == 2


def test_el_cartel_sin_orden_abierta_manda_a_las_teclas_que_de_verdad_sirven(
    taller: Path, monkeypatch
) -> None:
    """QUE SE ROMPIA: el cartel decia "NO HAY ORDEN ABIERTA - abrila con 'o'", y la 'o' no
    abre ninguna orden: abre el historial, y ahi hay dos caminos --ENTER retoma la elegida,
    'n' carga una nueva--. El que apretaba 'o' esperando el formulario se encontraba con
    una lista, y con el historial vacio --el dia que se instala el programa-- con una lista
    vacia y ninguna pista de que hacer."""
    from afilado import panel
    from afilado.ordenes import Ordenes

    carteles: list[str] = []
    original = run_live.revision.componer_simple

    def espiar(vivo, referencia, cfg, estado, alto=None):
        if estado.mensaje:
            carteles.append(estado.mensaje)
        return original(vivo, referencia, cfg, estado, alto)

    monkeypatch.setattr(run_live.revision, "componer_simple", espiar)

    vistos: list[str] = []
    componer_panel = panel.componer

    def espiar_panel(estado, alto, miniaturas=None):
        vistos.append(estado.modo)
        return componer_panel(estado, alto, miniaturas)

    monkeypatch.setattr(panel, "componer", espiar_panel)

    # La 'g' es anotar el diente a mano, que es lo que se aprieta sin orden abierta.
    _correr([ord("g"), 255, ord("o"), ord("q")], monkeypatch)

    sin_orden = [texto for texto in carteles if "NO HAY ORDEN ABIERTA" in texto]
    assert sin_orden, f"nunca salio el cartel de sin orden: {carteles}"
    # Nombra las dos teclas, y no promete que la 'o' abra una orden.
    assert "'o'" in sin_orden[0] and "'n'" in sin_orden[0]
    # Y la 'o' llevo al historial, que es lo que el cartel dice.
    assert panel.MODO_ORDENES in vistos
    # Sin tocar nada mas, no se abrio ninguna orden: la 'o' muestra, no crea.
    assert Ordenes(str(taller / "data" / "ordenes")).abierta is None
    assert not list((taller / "data" / "ordenes").glob("*.json"))


# ---------------------------------------------------------------------------
# LA FOTO DE PRUEBA DE CADA DIENTE.
#
# Esto es lo que contesta un reclamo. Medido en la orden de verdad 5113546 antes
# de este trabajo: 15 dientes anotados, los 15 con origen "la memoria, anotado
# solo" y CERO con foto. Un informe que dice "diente 7: RECHAZADO" y no tiene con
# que mostrarlo es la palabra del taller contra la del cliente.
#
# La prueba NO es Diente.foto. `foto` es el nombre pelado de la foto que entro a
# la coleccion de entrenamiento --sin carpeta, asi que ni el programa la
# encuentra-- y es lo unico que tienen las ordenes viejas: las 14 que nombran las
# ordenes 123 y 1234 estan todas en la papelera de la coleccion. `prueba` es la
# ruta de la foto que respalda el veredicto, en el deposito propio.
#
# Se mira EN EL DISCO. Que el .json nombre una foto no prueba que la foto este:
# eso es exactamente lo que ya paso una vez.
#
# COMPROBADO QUE NO PASAN SIN EL ARREGLO. Las siete se corrieron contra una copia
# del repo con el run_live.py de antes de este trabajo --el del ultimo commit, que
# no llama a guardar_prueba() en ninguno de los tres caminos-- y la ordenes.py de
# hoy. Fallaron las siete, cada una donde tenia que fallar:
#
#     se anota solo         7 dientes anotados y 0 fotos de prueba en el deposito
#     boton ANOTAR DIENTE   el diente se anoto a mano y se quedo sin foto de prueba
#     archivar a mano       la prueba quedo en ''
#     no cae en la coleccion   el archivo de la prueba no existe
#     sierra completa       el unico diente de la sierra se anoto sin foto
#     el disco que falla    la nota no dice que falto la prueba (las dos)
# ---------------------------------------------------------------------------


def _pruebas(base: Path, numero: str = "1234") -> list[Path]:
    """Las fotos de prueba que quedaron en el deposito de esa orden, por nombre.

    Ordenadas por nombre quedan en orden de diente: el nombre arranca con
    s<sierra>_d<diente>, asi que d001 va antes que d002.
    """
    carpeta = base / "data" / "pruebas" / numero
    return sorted(carpeta.glob("*.jpg")) if carpeta.is_dir() else []


def _donde_quedo(base: Path, diente) -> Path:
    """La foto de prueba de ese diente, en el disco. Diente.prueba es relativa a data."""
    return base / "data" / diente.prueba


def _anotados(orden) -> list:
    return [d for d in orden.sierras[0].revisados if d.anotado]


class _SierraQuePasa(_CamaraFalsa):
    """La sierra pasando diente por diente delante del microscopio, sola.

    Apoya un diente 0.40 s y pasa al siguiente, que es lo que hace el operario.
    Los dos dientes que alterna no son dos semillas cualesquiera: con la firma de
    automatico.huella(), _diente(0) contra _diente(6) da 0.1226-0.1232 --bien
    arriba del 0.07 que el programa pide para creer que paso al diente
    siguiente-- y el mismo diente con el temblor del sensor da 0.0011-0.0015,
    bien abajo del 0.04 que pide para contarlo en la racha. Con dos semillas
    consecutivas no alcanzaria: _diente(0) contra _diente(1) da 0.0250 y para el
    programa serian el mismo diente temblando.

    El tiempo se mira con el reloj y no con los cuadros leidos a proposito: la
    racha avanza una foto cada interfaz.segundos_entre_fotos, asi que lo que
    manda es el reloj y no la velocidad a la que esta PC dibuje la ventana.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._arranque = time.monotonic()

    def leer(self):
        self.leidos += 1
        cual = 0 if int((time.monotonic() - self._arranque) / 0.40) % 2 == 0 else 6
        base = _diente(cual)
        # El temblor del sensor, para que no sean dos fotos identicas: sin esto el
        # guardado se frenaria solo por duplicada y la prueba no probaria nada.
        ruido = np.random.default_rng(self.leidos).integers(-2, 3, base.shape, dtype=np.int16)
        return np.clip(base.astype(np.int16) + ruido, 0, 255).astype(np.uint8)


def test_cada_diente_que_se_anota_solo_deja_su_foto_en_el_disco(
    taller: Path, monkeypatch
) -> None:
    """Es el camino que dejo la orden 5113546 con 15 dientes y ninguna foto, y es el
    que mas usa el taller: el operario pasa la sierra y no toca una tecla, asi que
    si la foto no se guarda aca no se guarda en ninguna parte."""
    # MEDIDO con 120 cuadros y una sierra de 8 dientes: anota 6 en tres corridas
    # seguidas (6, 6, 6) y tarda 2,4 a 2,7 s. Con 200 cuadros anota los 8 y tarda
    # 4,4 a 5,4 s. Se piden 3 y no 6 porque el que cuenta es el reloj de la PC: lo
    # que esta prueba cuida es que haya UNA foto por cada diente anotado, cuantos
    # sean es del caso de arriba.
    orden = _correr_automatico(
        taller, monkeypatch, _SierraQuePasa, [255] * 120 + [ord("q")], dientes=8
    )

    anotados = _anotados(orden)
    assert len(anotados) >= 3, f"se anotaron {len(anotados)} dientes: no alcanza para probar"
    fotos = _pruebas(taller)
    assert len(fotos) == len(anotados), (
        f"{len(anotados)} dientes anotados y {len(fotos)} fotos de prueba en el deposito"
    )
    for diente in anotados:
        assert diente.prueba, f"el diente {diente.numero} quedo anotado sin foto de prueba"
        quedo = _donde_quedo(taller, diente)
        assert quedo.is_file(), f"el diente {diente.numero} nombra una foto que no esta: {quedo}"
        # Que el archivo este no alcanza: media foto tambien esta. Tiene que abrirse.
        assert cv2.imread(str(quedo)) is not None, f"la prueba del diente {diente.numero} no se abre"
    # Y la foto de cada diente lleva SU numero: una prueba pegada al diente de al
    # lado seria peor que no tenerla.
    assert [f.name[:8] for f in fotos] == [f"s01_d{d.numero:03d}" for d in anotados]


def test_el_boton_de_anotar_diente_tambien_deja_su_foto(taller: Path, monkeypatch) -> None:
    """La tecla 'g' no archiva nada en la coleccion --no le ensena nada al programa--
    y justo por eso era la que podia quedarse sin ninguna foto: el diente entraba al
    informe y no habia con que respaldarlo."""
    orden = _correr_automatico(taller, monkeypatch, _Quieta, [ord("g"), ord("q")])

    diente = orden.sierras[0].revisados[0]
    assert "lo anoto el operario" in diente.origen
    assert diente.prueba, "el diente se anoto a mano y se quedo sin foto de prueba"
    assert _donde_quedo(taller, diente).is_file()
    assert len(_pruebas(taller)) == 1
    # La 'g' sigue sin tocar la coleccion: lo que se guardo es prueba, no entrenamiento.
    assert _fotos(taller, "aprobada") == []


def test_al_archivar_a_mano_quedan_las_dos_fotos_y_no_se_confunden(
    taller: Path, monkeypatch
) -> None:
    """Son dos cosas distintas con dos trabajos distintos: la de la coleccion le
    ensena al programa y la de prueba contesta el reclamo. Si quedara una sola, o se
    pierde lo que el programa aprende o se pierde con que mostrar el veredicto."""
    from afilado.ordenes import Orden, Ordenes

    Ordenes(str(taller / "data" / "ordenes")).abrir(Orden.nueva("1234", 1, 4))

    _correr([_ESPACIO, ord("a"), ord("q")], monkeypatch)

    diente = _ordenes(taller).abierta.sierras[0].revisados[0]
    # La de entrenamiento: el nombre pelado, como se guardo siempre.
    assert diente.foto.endswith(".jpg") and "/" not in diente.foto
    assert diente.foto in {f.name for f in _fotos(taller, "aprobada")}
    # Y la de prueba, aparte, con carpeta y todo.
    assert diente.prueba.startswith("pruebas/"), f"la prueba quedo en {diente.prueba!r}"
    assert _donde_quedo(taller, diente).is_file()
    assert Path(diente.prueba).name != diente.foto, "las dos fotos son el mismo archivo"


def test_la_foto_de_prueba_no_cae_adentro_de_la_coleccion(taller: Path, monkeypatch) -> None:
    """Si cayera ahi el programa se arrastraria: archivar() llama a refrescar(), que
    se mide contra toda la coleccion, y eso va de 209 ms con 15 fotos a 3.153 ms con
    320. Una orden de 4 sierras de 80 dientes son esas 320 fotos, o sea que el
    programa quedaria inusable --3,1 s por diente contra 1,2 s de apoyarlo-- antes de
    terminar la primera orden. Por eso el deposito es propio y refrescar() no lo mira."""
    orden = _correr_automatico(
        taller, monkeypatch, _SierraQuePasa, [255] * 120 + [ord("q")], dientes=8
    )

    anotados = _anotados(orden)
    assert len(anotados) >= 3
    coleccion = (taller / "data" / "coleccion").resolve()
    for diente in anotados:
        quedo = _donde_quedo(taller, diente).resolve()
        assert quedo.is_file()
        assert coleccion not in quedo.parents, f"la prueba entro a la coleccion: {quedo}"
    # Y la coleccion no engordo ni una foto: por este camino el operario no eligio
    # ningun concepto, asi que no hay nada que ensenarle al programa.
    assert list(coleccion.rglob("*.jpg")) == [], "la captura automatica dejo fotos en la coleccion"


def test_con_la_sierra_completa_no_queda_ninguna_foto_huerfana(
    taller: Path, monkeypatch
) -> None:
    """Una foto en el deposito que no es la prueba de ningun diente es basura que
    manana alguien va a mostrar como prueba de algo. Con la sierra completa anotar()
    devuelve None --no hay diente que respaldar-- asi que no se escribe nada."""
    from afilado.ordenes import Orden, Ordenes

    Ordenes(str(taller / "data" / "ordenes")).abrir(Orden.nueva("1234", 1, 1))
    monkeypatch.setattr(run_live, "Camara", _OtroDiente)

    _correr([ord("g"), ord("q")], monkeypatch)
    antes = _pruebas(taller)
    assert len(antes) == 1, "el unico diente de la sierra se anoto sin foto"

    # La sierra ya esta completa y el operario insiste con la tecla.
    _correr([ord("g"), ord("g"), ord("q")], monkeypatch)

    assert _pruebas(taller) == antes, "quedo una foto de prueba de un diente que no existe"
    assert _ordenes(taller).abierta.total_revisados == 1


def test_si_la_foto_de_prueba_no_se_puede_escribir_el_diente_se_anota_igual(
    taller: Path, monkeypatch
) -> None:
    """Perder la foto es malo; perder el diente es peor. El diente que no esta en el
    informe es el diente que nadie reviso, y de ese no hay como defenderse. Asi que el
    renglon entra igual, con la prueba vacia y la nota que dice que falto: un renglon
    que dice que no tiene prueba no inventa nada."""
    from afilado.ordenes import Orden, Ordenes

    Ordenes(str(taller / "data" / "ordenes")).abrir(Orden.nueva("1234", 1, 4))
    # Una foto de modelo, para que el diente SI tenga veredicto: lo que falla es la
    # escritura de la prueba y nada mas.
    modelo = taller / "data" / "referencia" / "microscopio" / "modelo.jpg"
    modelo.parent.mkdir(parents=True, exist_ok=True)
    assert cv2.imwrite(str(modelo), _diente(0))

    # El disco que se niega a escribir, en el mismo lugar donde se niega de verdad: el
    # renombre final de guardar_prueba(). Asi se recorre el camino de error completo
    # --el log, el borrado del provisorio y el None-- y no una version de mentira.
    renombrar = os.replace

    def falla_solo_en_las_pruebas(origen, destino, *resto):
        if "pruebas" in str(destino):
            raise OSError("el disco dijo basta")
        return renombrar(origen, destino, *resto)

    monkeypatch.setattr(os, "replace", falla_solo_en_las_pruebas)
    monkeypatch.setattr(run_live, "Camara", _OtroDiente)

    codigo = _correr([ord("g"), ord("g"), ord("q")], monkeypatch)

    assert codigo == 0, "el programa se cerro con error porque no pudo guardar una foto"
    orden = _ordenes(taller).abierta
    assert orden.total_revisados == 2, "se perdieron dientes por una foto que no se pudo escribir"
    for diente in orden.sierras[0].revisados[:2]:
        assert diente.prueba == "", f"el diente {diente.numero} nombra una prueba que no se escribio"
        assert "no se pudo guardar la foto de prueba" in diente.nota
        assert diente.aprobado is not None, "se perdio tambien el veredicto"
    assert _pruebas(taller) == [], "quedo una foto a medias en el deposito"


def test_cuando_el_disco_falla_la_captura_automatica_sigue_anotando(
    taller: Path, monkeypatch
) -> None:
    """Es el camino en el que el operario no mira la pantalla: tiene las dos manos en
    la sierra. Si una foto que no se puede escribir frenara la cuenta de dientes, el
    informe saldria corto y nadie se enteraria hasta el reclamo."""
    renombrar = os.replace

    def falla_solo_en_las_pruebas(origen, destino, *resto):
        if "pruebas" in str(destino):
            raise OSError("el disco dijo basta")
        return renombrar(origen, destino, *resto)

    monkeypatch.setattr(os, "replace", falla_solo_en_las_pruebas)

    orden = _correr_automatico(
        taller, monkeypatch, _SierraQuePasa, [255] * 120 + [ord("q")], dientes=8
    )

    anotados = _anotados(orden)
    assert len(anotados) >= 3, f"se anotaron {len(anotados)} dientes: la cuenta se freno"
    for diente in anotados:
        assert diente.prueba == ""
        assert diente.nota == "no se pudo guardar la foto de prueba"
    assert _pruebas(taller) == []


# ---------------------------------------------------------------------------
# BORRAR UNA ORDEN DESDE EL BUCLE: LO QUE LA PANTALLA PROMETE Y LO QUE SE MUEVE
#
# Las pruebas del panel miran la pantalla y las de ordenes miran el disco. Aca se miran las
# dos a la vez y con el programa entero andando: se entra con la 'o', se aprieta 'b' y 's'
# como en el taller, se lee lo que la pantalla de confirmacion ESCRIBIO en ese cuadro y
# despues se cuentan los .jpg que llegaron a la papelera.
#
# Por que hace falta aca y no alcanza con las de arriba: la promesa la arma el panel, la
# cuenta la hace ordenes y el que las junta es el manejador de teclas. Las tres estan
# probadas de a una; lo que rompio en el taller fue el enganche --la pantalla contaba las
# fotos de una carpeta y el borrado movia otra-- y eso solo se ve corriendo las tres juntas.
#
# COMPROBADO QUE LAS DOS FALLAN SIN EL ARREGLO, volviendo el codigo a lo de antes de a una
# cosa por vez:
#
#   la regla vieja --_es_prueba_de() resolviendo la ruta contra data y aceptando cualquier
#   cosa--                  la pantalla prometia 3 fotos y el movimiento se llevaba 0:
#                           "assert promete == 0" fallando con "assert 3 == 0"
#   una pantalla que no promete nunca ninguna foto
#                           "assert promete == 3" fallando con "assert 0 == 3"
#   _limpiar_aviso() de antes, "pan.aviso = None" a secas
#                           el cuadro del reintento se dibujo sin ningun aviso:
#                           "assert 'A MEDIAS' in ''"
# ---------------------------------------------------------------------------


def _orden_cerrada_con_pruebas(base: Path, numero: str, cuantas: int = 3):
    """Una orden CERRADA con `cuantas` fotos de prueba de verdad en su deposito.

    Cerrada a proposito: con una orden abierta, la captura automatica le anota dientes y le
    guarda fotos mientras el bucle anda, y entonces la cuenta de fotos de prueba la estaria
    moviendo la camara en el medio de la medicion. Asi el unico que toca el deposito es el
    borrado, que es lo que se esta midiendo.
    """
    from afilado.ordenes import Orden, Ordenes

    ordenes = Ordenes(str(base / "data" / "ordenes"))
    ordenes.abrir(Orden.nueva(numero, 1, 10, cliente="Aserradero del Litoral"))
    foto = np.full((60, 80, 3), 128, np.uint8)
    for diente in range(1, cuantas + 1):
        ruta = ordenes.guardar_prueba(numero, 1, diente, foto)
        assert ruta, "no se pudo guardar la foto de prueba"
        ordenes.abierta.anotar(estado="rechazada", aprobado=False,
                               origen="lo anoto el operario", prueba=ruta)
    assert ordenes.guardar() is not None
    ordenes.cerrar()
    return ordenes


def _fotos_prometidas(renglones: list[str]) -> int:
    """Cuantas fotos dicen esos renglones que se van a llevar. Es lo que el operario lee."""
    for texto in renglones:
        cuantas = re.search(r"Se lleva las (\d+) fotos de prueba", texto)
        if cuantas:
            return int(cuantas.group(1))
        if ("No se lleva ninguna foto" in texto
                or "No tiene ninguna foto de prueba" in texto):
            return 0
    raise AssertionError(f"la confirmacion no dice que prueba se lleva: {renglones}")


def _borrar_desde_el_bucle(base: Path, monkeypatch) -> tuple[int, int]:
    """Entra con la 'o', borra la fila elegida con 'b' y 's' y devuelve
    (las fotos que la pantalla prometio, los .jpg que llegaron a la papelera).

    Los renglones se le piden a la funcion que los dibuja, espiandola: es el texto que se
    convirtio en pixeles en ese cuadro, no una composicion aparte hecha por la prueba.
    """
    from afilado import panel

    renglones: list[str] = []
    dibujar = panel._dibujar_borrar_orden

    def espiar(lienzo, estado, botones, etiquetas):
        lineas = dibujar(lienzo, estado, botones, etiquetas)
        renglones.extend(texto for texto, *_resto in lineas)
        return lineas

    monkeypatch.setattr(panel, "_dibujar_borrar_orden", espiar)

    papelera = base / "data" / "ordenes" / "_papelera"
    antes = len(list(papelera.rglob("*.jpg"))) if papelera.is_dir() else 0

    _correr([ord("o"), 255, ord("b"), 255, ord("s"), 255, ord("q")], monkeypatch)

    assert renglones, "la pantalla de confirmacion no se dibujo ni una vez"
    despues = len(list(papelera.rglob("*.jpg"))) if papelera.is_dir() else 0
    return _fotos_prometidas(renglones), despues - antes


def test_el_borrado_del_bucle_se_lleva_las_fotos_que_la_pantalla_prometio(
    taller: Path, monkeypatch
) -> None:
    """QUE SE ROMPIA EN EL TALLER: con la orden renombrada a mano --"5000 viejo.json", que
    es lo que queda despues de tocar la carpeta con el Explorador-- la pantalla prometia
    llevarse las 3 fotos de prueba y el movimiento se llevaba CERO. El que contesta un
    reclamo aprieta, despues va a _papelera a rescatar las fotos y no hay ninguna: las 3
    siguen en data/pruebas/5000, donde ya nadie las busca porque la pantalla dijo que se
    habian ido con la orden.

    Las dos mitades van juntas. La de abajo sola la pasaria una pantalla que no prometiera
    nunca ninguna foto, y entonces el operario no sabria nunca cuanta prueba esta tirando;
    la de arriba sola la pasaba el programa viejo.
    """
    # EL CAMINO DE TODOS LOS DIAS: el .json se llama como su numero, igual que las 4 ordenes
    # de verdad del taller. Promete 3 y se lleva 3.
    _orden_cerrada_con_pruebas(taller, "1234")
    deposito = taller / "data" / "pruebas" / "1234"
    assert len(list(deposito.glob("*.jpg"))) == 3

    promete, movidas = _borrar_desde_el_bucle(taller, monkeypatch)

    assert promete == 3
    assert movidas == promete, f"la pantalla prometio {promete} fotos y se movieron {movidas}"
    assert not deposito.is_dir()                 # se fue de verdad, no quedo una copia
    assert not (taller / "data" / "ordenes" / "1234.json").is_file()

    # EL ARCHIVO TOCADO A MANO. Las fotos viven en data/pruebas/5000 y el .json se llama
    # "5000 viejo": esas fotos no son la prueba de ESTE archivo, asi que no se prometen y no
    # se mueven. Lo que la pantalla dice de ellas --que el archivo las nombra y no estan en
    # su carpeta-- es lo unico que se midio.
    _orden_cerrada_con_pruebas(taller, "5000")
    ordenes = taller / "data" / "ordenes"
    (ordenes / "5000.json").rename(ordenes / "5000 viejo.json")
    quedaron = sorted((taller / "data" / "pruebas" / "5000").glob("*.jpg"))
    assert len(quedaron) == 3

    promete, movidas = _borrar_desde_el_bucle(taller, monkeypatch)

    assert promete == 0
    assert movidas == promete, f"la pantalla prometio {promete} fotos y se movieron {movidas}"
    assert not (ordenes / "5000 viejo.json").is_file()
    # Y las 3 fotos siguen donde estaban: no se prometieron y no se movieron.
    assert sorted((taller / "data" / "pruebas" / "5000").glob("*.jpg")) == quedaron


def test_el_aviso_del_borrado_a_medias_sigue_dibujado_despues_de_la_b(
    taller: Path, monkeypatch
) -> None:
    """QUE SE ROMPIA: la 'b' del reintento se llevaba el unico renglon que decia en que
    carpeta de papelera quedo la mitad de la orden. El operario lee "quedo A MEDIAS en
    _papelera/<fecha y hora>", aprieta 'b' para volver a intentarlo --que es lo que
    cualquiera hace-- y el renglon desaparece antes de que haya podido anotar el nombre: de
    ahi en mas tiene que adivinar a cual de las carpetas de papelera entrar, y se llaman
    todas con la fecha y la hora.

    Se mira EN LOS CUADROS QUE EL BUCLE DIBUJO, uno por uno, y no en el estado al final: lo
    que importa es que el renglon haya estado en la pantalla mientras el operario la tenia
    delante. El movimiento se traba como se traba en el taller --una foto abierta en el
    visor de Windows-- y shutil.move, que cuando no puede renombrar la carpeta copia archivo
    por archivo, deja una foto del otro lado y se planta en la siguiente.
    """
    from afilado import panel

    _orden_cerrada_con_pruebas(taller, "1234")
    mover = shutil.move

    def copia_a_medias(origen, destino):
        fuente = Path(origen)
        if fuente.is_dir():
            Path(destino).mkdir(parents=True, exist_ok=True)
            primera = sorted(fuente.iterdir())[0]
            shutil.copy2(primera, Path(destino) / primera.name)
            raise OSError(13, "Permission denied")
        return mover(origen, destino)

    monkeypatch.setattr(shutil, "move", copia_a_medias)

    vistos: list[tuple[str, Optional[str]]] = []
    componer = panel.componer

    def espiar(estado, alto, miniaturas=None):
        vistos.append((estado.modo, estado.aviso))
        return componer(estado, alto, miniaturas)

    monkeypatch.setattr(panel, "componer", espiar)

    # o = entrar a las ordenes, b = borrar, s = si, y la 'b' DEL REINTENTO.
    _correr([ord("o"), 255, ord("b"), 255, ord("s"), 255, ord("b"), 255, ord("q")],
            monkeypatch)

    confirmaciones = [aviso for modo, aviso in vistos
                      if modo == panel.MODO_BORRAR_ORDEN]
    assert len(confirmaciones) >= 2, (
        f"la confirmacion se dibujo {len(confirmaciones)} vez/veces: no hubo reintento"
    )
    # La primera vez no habia nada que avisar; la del reintento tiene que decir donde quedo
    # la mitad de la orden.
    assert confirmaciones[0] is None
    del_reintento = confirmaciones[-1] or ""
    assert "A MEDIAS" in del_reintento, del_reintento
    assert "1234.json" in del_reintento and "pruebas/1234" in del_reintento
    # LA CARPETA QUE NOMBRA EXISTE Y TIENE ALGO ADENTRO. Mandar a mirar una carpeta vacia
    # --o inventada-- seria hacerle perder el viaje al que va con el Explorador.
    nombrada = re.search(r"_papelera/(\S+):", del_reintento)
    assert nombrada, del_reintento
    quedo = taller / "data" / "ordenes" / "_papelera" / nombrada.group(1)
    assert quedo.is_dir() and list(quedo.rglob("*.jpg"))
    # Y la orden sigue entera en el disco: el .json es lo ultimo que se mueve, asi que un
    # borrado trabado no la saca del historial. Es lo que hace que el reintento tenga
    # sentido.
    assert (taller / "data" / "ordenes" / "1234.json").is_file()
