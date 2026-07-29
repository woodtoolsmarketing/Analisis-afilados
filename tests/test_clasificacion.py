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
