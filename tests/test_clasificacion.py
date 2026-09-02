"""Tests del modo clasificacion (aprobada/rechazada): config, almacen y overlay.

No dependen de torch/ultralytics: cubren la parte del sistema que funciona sin modelo
(validacion de config, archivado por clase, banda de veredicto).
"""
from __future__ import annotations

import numpy as np
import pytest

from afilado.almacen import AlmacenClasificacion
from afilado.config import AppConfig, DetectorConfig, cargar_config, guardar_config
from afilado.detector import (
    _a_probabilidades,
    _clases_de_metadatos,
    _lado_de_metadatos,
    _leer_metadatos,
    crear_clasificador,
    preparar_entrada_clasificacion,
)
from afilado import overlay
from afilado.tipos import Veredicto


def _frame() -> np.ndarray:
    return np.full((120, 160, 3), 127, np.uint8)


def test_config_acepta_tarea_classify(tmp_path):
    cfg = AppConfig(detector=DetectorConfig(tarea="classify"))
    ruta = tmp_path / "c.yaml"
    guardar_config(cfg, ruta)
    recargado = cargar_config(ruta)
    assert recargado.detector.tarea == "classify"


def test_config_rechaza_tarea_invalida(tmp_path):
    # Se escribe el YAML a mano: guardar_config tambien valida y lanzaria antes de cargar.
    ruta = tmp_path / "c.yaml"
    ruta.write_text("detector:\n  tarea: disparate\n", encoding="utf-8")
    with pytest.raises(ValueError):
        cargar_config(ruta)


def test_el_config_del_repo_entrena_e_infiere_al_mismo_tamano():
    """El config que se distribuye y el imgsz por defecto del entrenamiento deben coincidir.

    Si alguien cambia uno y se olvida del otro, el modelo se entrena a un tamano y se infiere
    a otro. Eso no lanza ningun error: degrada el veredicto en silencio, que es exactamente
    el fallo que un taller no puede detectar.
    """
    from afilado.cli.train import _IMGSZ_CLASIFICACION, _IMGSZ_DETECCION

    cfg = cargar_config(None)
    esperado = _IMGSZ_CLASIFICACION if cfg.detector.tarea == "classify" else _IMGSZ_DETECCION

    assert cfg.detector.imgsz == esperado


def test_almacen_guarda_por_clase_sin_mutar(tmp_path):
    alm = AlmacenClasificacion(str(tmp_path), ["aprobada", "rechazada"])
    frame = _frame()
    copia = frame.copy()
    ruta = alm.guardar(frame, "rechazada")
    assert ruta.parent.name == "rechazada"
    assert ruta.exists()
    assert alm.total_guardados == 1
    assert np.array_equal(frame, copia)  # no muto el frame de entrada


def test_almacen_rechaza_clase_desconocida(tmp_path):
    alm = AlmacenClasificacion(str(tmp_path), ["aprobada", "rechazada"])
    with pytest.raises(ValueError):
        alm.guardar(_frame(), "oxido")


def test_almacen_nombres_unicos(tmp_path):
    alm = AlmacenClasificacion(str(tmp_path), ["aprobada"])
    rutas = {alm.guardar(_frame(), "aprobada") for _ in range(5)}
    assert len(rutas) == 5  # ninguna se sobrescribe


def test_crear_clasificador_sin_modelo_devuelve_none(tmp_path):
    # Pesos inexistentes => None (modo recoleccion), sin necesitar ultralytics.
    cfg = AppConfig(detector=DetectorConfig(pesos=str(tmp_path / "no_existe.pt")))
    assert crear_clasificador(cfg) is None


def test_overlay_veredicto_copia_y_no_muta():
    cfg = AppConfig()
    frame = _frame()
    copia = frame.copy()
    v = Veredicto(clase="rechazada", clase_id=1, confianza=0.9,
                  probabilidades={"aprobada": 0.1, "rechazada": 0.9})
    anotado = overlay.dibujar_veredicto(frame, v, cfg)
    assert anotado is not frame
    assert anotado.shape == frame.shape
    assert np.array_equal(frame, copia)


def test_overlay_veredicto_sin_modelo_no_explota():
    cfg = AppConfig()
    frame = _frame()
    anotado = overlay.dibujar_veredicto(frame, None, cfg)  # modo recoleccion
    assert anotado.shape == frame.shape


# ---------------------------------------------------------------------------
# Ruta ONNX: la que corre en la PC del taller, sin torch ni ultralytics.
# Todo lo de aca abajo se ejecuta con opencv y numpy solos, igual que alla.
# ---------------------------------------------------------------------------


def test_preparar_entrada_forma_rango_y_tipo():
    entrada = preparar_entrada_clasificacion(_frame(), 224)

    assert entrada.shape == (1, 3, 224, 224)
    assert entrada.dtype == np.float32
    assert float(entrada.min()) >= 0.0
    assert float(entrada.max()) <= 1.0


def test_preparar_entrada_recorta_el_centro_y_no_deforma():
    """Tiene que quedarse con la banda central, no achatar la imagen entera.

    Con un resize plano el modelo devuelve OTRA clase: la geometria de este paso es parte
    del contrato con los pesos entrenados, no un detalle de implementacion.

    Imagen de 300x100 con una banda blanca en las columnas 100..200. Reducir el lado corto
    a 50 la deja en 150x50; el recorte centrado de 50 px toma las columnas 50..100 del
    reducido, que son exactamente la banda blanca original.
    """
    img = np.zeros((100, 300, 3), np.uint8)
    img[:, 100:200] = 255

    entrada = preparar_entrada_clasificacion(img, 50)

    assert float(entrada.min()) == 1.0


def test_preparar_entrada_convierte_bgr_a_rgb():
    """OpenCV entrega BGR y la red espera RGB. Confundirlos no rompe nada visible:
    simplemente empeora el veredicto sin que nadie se entere."""
    azul_en_bgr = np.zeros((64, 64, 3), np.uint8)
    azul_en_bgr[:, :, 0] = 255

    entrada = preparar_entrada_clasificacion(azul_en_bgr, 32)

    assert float(entrada[0, 0].max()) == 0.0  # canal R, vacio
    assert float(entrada[0, 2].min()) == 1.0  # canal B, lleno


@pytest.mark.parametrize(
    "forma", [(80, 200, 3), (200, 80, 3), (80, 200), (150, 150, 3), (20, 30, 3)]
)
def test_preparar_entrada_soporta_gris_vertical_cuadrado_y_ampliacion(forma):
    """Tambien tiene que andar con frames mas chicos que el lado pedido (ahi amplia)."""
    entrada = preparar_entrada_clasificacion(np.zeros(forma, np.uint8), 64)

    assert entrada.shape == (1, 3, 64, 64)


@pytest.mark.parametrize(
    "frame, lado",
    [
        (np.zeros((0, 0, 3), np.uint8), 224),
        (np.zeros((10, 10, 3), np.uint8), 0),
        (np.zeros((10, 10, 3), np.uint8), -5),
    ],
)
def test_preparar_entrada_rechaza_entradas_imposibles(frame, lado):
    with pytest.raises(ValueError):
        preparar_entrada_clasificacion(frame, lado)


def test_probabilidades_deja_pasar_lo_que_ya_viene_normalizado():
    """La cabeza de ultralytics ya aplica softmax; no hay que aplicarlo dos veces."""
    salida = np.array([[0.7, 0.3]], np.float32)

    assert np.allclose(_a_probabilidades(salida), [0.7, 0.3])


def test_probabilidades_normaliza_logits_crudos():
    """Si un export con otro opset devolviera logits, el overlay mostraria 'confianza 12.4'."""
    probabilidades = _a_probabilidades(np.array([[2.0, 1.0, -1.0]], np.float32))

    assert float(probabilidades.sum()) == pytest.approx(1.0)
    assert int(np.argmax(probabilidades)) == 0


def test_metadatos_se_leen_del_json_hermano(tmp_path):
    modelo = tmp_path / "m.onnx"
    modelo.write_bytes(b"x")
    (tmp_path / "m.onnx.json").write_text(
        '{"clases": ["aprobada", "rechazada"], "imgsz": 224}', encoding="utf-8"
    )

    datos = _leer_metadatos(modelo)

    assert datos["clases"] == ["aprobada", "rechazada"]
    assert datos["imgsz"] == 224


@pytest.mark.parametrize("contenido", ["{ esto no es json", "[1, 2, 3]"])
def test_metadatos_rotos_no_tumban_el_arranque(tmp_path, contenido):
    """Sin metadatos legibles se cae a lo que diga el config, pero el programa arranca."""
    modelo = tmp_path / "m.onnx"
    modelo.write_bytes(b"x")
    (tmp_path / "m.onnx.json").write_text(contenido, encoding="utf-8")

    assert _leer_metadatos(modelo) == {}


def test_metadatos_ausentes_devuelven_vacio(tmp_path):
    modelo = tmp_path / "m.onnx"
    modelo.write_bytes(b"x")

    assert _leer_metadatos(modelo) == {}


def test_crear_clasificador_con_onnx_corrupto_cae_a_recoleccion(tmp_path):
    """Un .onnx ilegible tiene que degradar a modo recoleccion, no tumbar el programa.

    En el taller un archivo copiado a medias desde un pendrive es un escenario real.
    """
    falso = tmp_path / "modelo.onnx"
    falso.write_bytes(b"esto no es un modelo onnx")
    cfg = AppConfig(detector=DetectorConfig(pesos=str(falso), tarea="classify"))

    assert crear_clasificador(cfg) is None


def test_crear_clasificador_elige_por_extension(tmp_path, monkeypatch):
    """.onnx va al runtime sin torch y cualquier otra cosa al de ultralytics."""
    from afilado import detector as modulo

    elegidos = []

    class Espia:
        def __init__(self, cfg, clases):
            elegidos.append(type(self).__name__)

        disponible = True
        descripcion = "espia"

        def clasificar(self, frame):
            return None

    class EspiaOnnx(Espia):
        pass

    class EspiaPt(Espia):
        pass

    monkeypatch.setattr(modulo, "DetectorClasificadorOnnx", EspiaOnnx)
    monkeypatch.setattr(modulo, "DetectorClasificador", EspiaPt)

    for nombre, esperado in (("m.onnx", "EspiaOnnx"), ("m.pt", "EspiaPt"),
                             ("m.ONNX", "EspiaOnnx")):
        ruta = tmp_path / nombre
        ruta.write_bytes(b"x")
        elegidos.clear()
        modulo.crear_clasificador(
            AppConfig(detector=DetectorConfig(pesos=str(ruta), tarea="classify"))
        )
        assert elegidos == [esperado], f"{nombre} fue a parar a {elegidos}"


# ---------------------------------------------------------------------------
# Lectura de los metadatos del .onnx. Un JSON mal escrito es un escenario real
# (se copia a mano, se edita, se genera con otra herramienta) y NUNCA puede
# tumbar el arranque en el taller: ahi nadie puede leer un traceback.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "metadatos, esperado",
    [
        ({"imgsz": 320}, 320),
        ({"imgsz": "320"}, 320),
        ({"imgsz": [320, 320]}, 320),  # asi lo guarda el propio ultralytics
        ({"imgsz": (320, 320)}, 320),
        ({"imgsz": [320]}, 320),
    ],
)
def test_lado_de_metadatos_acepta_los_formatos_reales(metadatos, esperado):
    assert _lado_de_metadatos(metadatos, 224) == esperado


@pytest.mark.parametrize(
    "metadatos",
    [
        {},                              # sin metadatos
        {"imgsz": None},
        {"imgsz": "no es un numero"},
        {"imgsz": [320, 640]},           # entrada no cuadrada
        {"imgsz": []},
        {"imgsz": [[320], [320]]},       # anidado
        {"imgsz": ["320", 320]},         # mezcla de tipos
        {"imgsz": 0},
        {"imgsz": -5},
        {"imgsz": {"alto": 320}},
    ],
)
def test_lado_de_metadatos_cae_al_config_ante_cualquier_disparate(metadatos):
    """Nunca lanza: devuelve el valor del config y sigue. Un imgsz raro no puede
    inutilizar un modelo que esta bien."""
    assert _lado_de_metadatos(metadatos, 224) == 224


def test_clases_de_metadatos_usa_las_del_modelo():
    clases = _clases_de_metadatos(
        {"clases": ["buena", "mala"]}, ["aprobada", "rechazada"], "m.onnx"
    )

    assert clases == ["buena", "mala"]


@pytest.mark.parametrize(
    "metadatos",
    [
        {},
        {"clases": []},          # lista vacia: dejaria al overlay mostrando "clase_0"
        {"clases": [0, 1]},
        {"clases": "aprobada"},
        {"clases": ["ok", 3]},
        {"clases": None},
    ],
)
def test_clases_de_metadatos_cae_al_config_si_no_sirven(metadatos):
    porDefecto = ["aprobada", "rechazada"]

    assert _clases_de_metadatos(metadatos, porDefecto, "m.onnx") == porDefecto


def test_clases_y_lado_se_leen_por_separado():
    """Son datos independientes: un JSON puede traer uno bien y el otro mal.

    Antes estaban atados y un JSON sin clases hacia que se ignorara tambien el imgsz.
    """
    metadatos = {"imgsz": 320}

    assert _lado_de_metadatos(metadatos, 224) == 320
    assert _clases_de_metadatos(metadatos, ["aprobada", "rechazada"], "m.onnx") == [
        "aprobada",
        "rechazada",
    ]
