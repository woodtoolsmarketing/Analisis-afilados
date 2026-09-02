"""Tests del modo clasificacion (aprobada/rechazada): config, almacen y overlay.

No dependen de torch/ultralytics: cubren la parte del sistema que funciona sin modelo
(validacion de config, archivado por clase, banda de veredicto).
"""
from __future__ import annotations

import numpy as np
import pytest

from afilado.almacen import AlmacenClasificacion
from afilado.config import AppConfig, DetectorConfig, cargar_config, guardar_config
from afilado.detector import crear_clasificador
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


def test_la_ayuda_informa_donde_termina_su_bloque():
    """_dibujar_ayuda tiene que decir hasta donde subio, o nadie puede apilar encima.

    texto_con_fondo recorta contra el borde del frame, asi que un texto que cae debajo de
    la ayuda no se ve "cortado": se ve ilegible, y nadie se entera de que se solapan.
    """
    img = np.zeros((200, 400, 3), np.uint8)

    tope = overlay._dibujar_ayuda(img, ("primera linea", "segunda linea"))

    assert 0 < tope < 200
    assert img[:tope].max() == 0, "dibujo por encima del tope que informa"
    assert img[tope:].max() > 0, "no dibujo nada debajo del tope"


def test_el_contador_de_guardadas_se_dibuja_encima_de_la_ayuda():
    """El contador es la unica confirmacion PERMANENTE de que la foto se archivo.

    El cartel verde de "GUARDADA" dura 1.5 segundos y se va; si el contador queda debajo
    de la ayuda (que esta visible por defecto), el operario no tiene forma de saber si
    apreto bien la tecla.
    """
    cfg = AppConfig(clases=["aprobada", "rechazada"], clases_defecto=["rechazada"])
    veredicto = Veredicto(
        clase="aprobada",
        clase_id=0,
        confianza=0.9,
        probabilidades={"aprobada": 0.9, "rechazada": 0.1},
    )
    frame = np.full((480, 640, 3), 127, np.uint8)

    con_cero = overlay.dibujar_veredicto(
        frame, veredicto, cfg, mostrar_ayuda=True, mensaje=None, capturas=0
    )
    con_siete = overlay.dibujar_veredicto(
        frame, veredicto, cfg, mostrar_ayuda=True, mensaje=None, capturas=7
    )

    tope_ayuda = overlay._dibujar_ayuda(np.zeros_like(frame), overlay._AYUDA_CLASIFICACION)
    filas_distintas = np.argwhere(np.any(con_cero != con_siete, axis=2))

    assert filas_distintas.size > 0, "cambiar el contador no cambio la imagen: quedo tapado"
    assert int(filas_distintas[:, 0].max()) < tope_ayuda, (
        "el contador invade la franja de la ayuda y queda ilegible"
    )


def test_el_contador_tambien_aparece_con_la_ayuda_oculta():
    """Con la tecla h se esconde la ayuda; el contador tiene que seguir estando."""
    cfg = AppConfig(clases=["aprobada", "rechazada"], clases_defecto=["rechazada"])
    frame = np.full((480, 640, 3), 127, np.uint8)

    con_cero = overlay.dibujar_veredicto(
        frame, None, cfg, mostrar_ayuda=False, mensaje=None, capturas=0
    )
    con_siete = overlay.dibujar_veredicto(
        frame, None, cfg, mostrar_ayuda=False, mensaje=None, capturas=7
    )

    assert not np.array_equal(con_cero, con_siete)
