"""Pruebas de la carga, el guardado y la validacion de la configuracion.

Cada validacion tiene su test porque un config invalido que se acepta en silencio es
peor que un error: el operario creeria estar midiendo con un umbral que en realidad
nunca se aplico.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import pytest
import yaml

from afilado.config import (
    CLASES_DEFECTO_POR_DEFECTO,
    CLASES_POR_DEFECTO,
    AppConfig,
    ArucoConfig,
    CamaraConfig,
    DetectorConfig,
    FeedbackConfig,
    FiltrosConfig,
    RoiConfig,
    cargar_config,
    guardar_config,
    raiz_repo,
    ruta_absoluta,
)


@pytest.fixture
def escribir_yaml(tmp_path: Path) -> Callable[[dict[str, Any]], Path]:
    """Devuelve una funcion que vuelca un dict a un config.yaml temporal."""

    def _escribir(datos: dict[str, Any]) -> Path:
        ruta = tmp_path / "config.yaml"
        ruta.write_text(
            yaml.safe_dump(datos, allow_unicode=True, sort_keys=False), encoding="utf-8"
        )
        return ruta

    return _escribir


def test_config_por_defecto_es_valida() -> None:
    cfg = AppConfig()

    assert cfg.clases == CLASES_POR_DEFECTO
    assert cfg.clases_defecto == CLASES_DEFECTO_POR_DEFECTO
    assert set(cfg.clases_defecto) < set(cfg.clases)
    assert "ok" in cfg.clases and "ok" not in cfg.clases_defecto
    assert cfg.detector.confianza == 0.80
    assert cfg.detector.tarea == "segment"


def test_archivo_inexistente_devuelve_defaults(tmp_path: Path) -> None:
    cfg = cargar_config(tmp_path / "no_existe.yaml")

    assert cfg == AppConfig()


def test_yaml_vacio_devuelve_defaults(tmp_path: Path) -> None:
    ruta = tmp_path / "vacio.yaml"
    ruta.write_text("", encoding="utf-8")

    assert cargar_config(ruta) == AppConfig()


def test_roundtrip_guardar_y_cargar(tmp_path: Path) -> None:
    """Lo que se guarda es exactamente lo que se vuelve a leer."""
    original = AppConfig(
        camara=CamaraConfig(fuente=2, ancho=1920, alto=1080, fps=60, enfoque=35),
        aruco=ArucoConfig(lado_mm=25.4, id_referencia=7, suavizado=0.5),
        roi=RoiConfig(x=0.2, y=0.1, w=0.6, h=0.8),
        detector=DetectorConfig(pesos="models/otro.pt", tarea="detect", confianza=0.65),
        filtros=FiltrosConfig(area_minima_mm2=5.0, kernel_morfologico=5),
        feedback=FeedbackConfig(directorio="data/otro", calidad_jpg=100),
        clases=["ok", "fisura"],
        clases_defecto=["fisura"],
    )
    ruta = tmp_path / "sub" / "config.yaml"

    guardar_config(original, ruta)

    assert ruta.is_file()
    assert cargar_config(ruta) == original


def test_el_yaml_guardado_usa_claves_en_espanol(tmp_path: Path) -> None:
    ruta = tmp_path / "config.yaml"

    guardar_config(AppConfig(), ruta)
    datos = yaml.safe_load(ruta.read_text(encoding="utf-8"))

    assert set(datos) == {
        "camara",
        "aruco",
        "roi",
        "detector",
        "filtros",
        "feedback",
        "clases",
        "clases_defecto",
    }
    assert datos["aruco"]["lado_mm"] == 30.0
    assert datos["camara"]["voltear_horizontal"] is False


def test_secciones_ausentes_toman_sus_defaults(
    escribir_yaml: Callable[[dict[str, Any]], Path]
) -> None:
    """Solo hace falta declarar lo que se cambia."""
    ruta = escribir_yaml({"aruco": {"lado_mm": 50.0}})

    cfg = cargar_config(ruta)

    assert cfg.aruco.lado_mm == 50.0
    assert cfg.aruco.diccionario == "DICT_4X4_50"  # default intacto
    assert cfg.camara == CamaraConfig()


def test_clave_desconocida_en_una_seccion(
    escribir_yaml: Callable[[dict[str, Any]], Path]
) -> None:
    """Un typo no puede pasar inadvertido: el umbral quedaria en su valor por defecto."""
    ruta = escribir_yaml({"detector": {"confianzza": 0.5}})

    with pytest.raises(ValueError, match="confianzza"):
        cargar_config(ruta)


def test_clave_desconocida_en_la_raiz(
    escribir_yaml: Callable[[dict[str, Any]], Path]
) -> None:
    ruta = escribir_yaml({"camara_web": {"fuente": 1}})

    with pytest.raises(ValueError, match="camara_web"):
        cargar_config(ruta)


def test_seccion_que_no_es_un_mapa(
    escribir_yaml: Callable[[dict[str, Any]], Path]
) -> None:
    ruta = escribir_yaml({"roi": [0.1, 0.1, 0.5, 0.5]})

    with pytest.raises(ValueError, match="roi"):
        cargar_config(ruta)


@pytest.mark.parametrize(
    "roi",
    [
        {"x": -0.1},
        {"y": 1.5},
        {"w": 2.0},
        {"h": -3.0},
    ],
)
def test_roi_fuera_de_0_a_1(
    escribir_yaml: Callable[[dict[str, Any]], Path], roi: dict[str, float]
) -> None:
    ruta = escribir_yaml({"roi": roi})

    with pytest.raises(ValueError, match="normalizado"):
        cargar_config(ruta)


def test_roi_que_se_sale_por_la_derecha(
    escribir_yaml: Callable[[dict[str, Any]], Path]
) -> None:
    ruta = escribir_yaml({"roi": {"x": 0.6, "w": 0.7}})

    with pytest.raises(ValueError, match="roi.x"):
        cargar_config(ruta)


def test_roi_que_se_sale_por_abajo(
    escribir_yaml: Callable[[dict[str, Any]], Path]
) -> None:
    ruta = escribir_yaml({"roi": {"y": 0.5, "h": 0.8}})

    with pytest.raises(ValueError, match="roi.y"):
        cargar_config(ruta)


def test_roi_que_ocupa_el_frame_entero_es_valido(
    escribir_yaml: Callable[[dict[str, Any]], Path]
) -> None:
    """x+w==1 es el borde exacto, no un desborde."""
    ruta = escribir_yaml({"roi": {"x": 0.0, "y": 0.0, "w": 1.0, "h": 1.0}})

    cfg = cargar_config(ruta)

    assert (cfg.roi.w, cfg.roi.h) == (1.0, 1.0)


@pytest.mark.parametrize("confianza", [-0.01, 1.01, 80.0])
def test_confianza_fuera_de_rango(
    escribir_yaml: Callable[[dict[str, Any]], Path], confianza: float
) -> None:
    ruta = escribir_yaml({"detector": {"confianza": confianza}})

    with pytest.raises(ValueError, match="confianza"):
        cargar_config(ruta)


def test_tarea_invalida(escribir_yaml: Callable[[dict[str, Any]], Path]) -> None:
    ruta = escribir_yaml({"detector": {"tarea": "clasificar"}})

    with pytest.raises(ValueError, match="tarea"):
        cargar_config(ruta)


@pytest.mark.parametrize("tarea", ["detect", "segment"])
def test_tareas_validas(
    escribir_yaml: Callable[[dict[str, Any]], Path], tarea: str
) -> None:
    ruta = escribir_yaml({"detector": {"tarea": tarea}})

    assert cargar_config(ruta).detector.tarea == tarea


@pytest.mark.parametrize("lado_mm", [0.0, -30.0])
def test_lado_mm_no_positivo(
    escribir_yaml: Callable[[dict[str, Any]], Path], lado_mm: float
) -> None:
    ruta = escribir_yaml({"aruco": {"lado_mm": lado_mm}})

    with pytest.raises(ValueError, match="lado_mm"):
        cargar_config(ruta)


@pytest.mark.parametrize("suavizado", [-0.1, 1.1])
def test_suavizado_fuera_de_0_a_1(
    escribir_yaml: Callable[[dict[str, Any]], Path], suavizado: float
) -> None:
    ruta = escribir_yaml({"aruco": {"suavizado": suavizado}})

    with pytest.raises(ValueError, match="suavizado"):
        cargar_config(ruta)


@pytest.mark.parametrize("kernel", [2, 4, 0, -3])
def test_kernel_morfologico_par_o_menor_que_1(
    escribir_yaml: Callable[[dict[str, Any]], Path], kernel: int
) -> None:
    """Un kernel par no tiene pixel central: desplazaria la mascara."""
    ruta = escribir_yaml({"filtros": {"kernel_morfologico": kernel}})

    with pytest.raises(ValueError, match="impar"):
        cargar_config(ruta)


@pytest.mark.parametrize("kernel", [1, 3, 5, 7])
def test_kernel_morfologico_impar_es_valido(
    escribir_yaml: Callable[[dict[str, Any]], Path], kernel: int
) -> None:
    ruta = escribir_yaml({"filtros": {"kernel_morfologico": kernel}})

    assert cargar_config(ruta).filtros.kernel_morfologico == kernel


def test_clases_vacia(escribir_yaml: Callable[[dict[str, Any]], Path]) -> None:
    ruta = escribir_yaml({"clases": [], "clases_defecto": []})

    with pytest.raises(ValueError, match="clases"):
        cargar_config(ruta)


def test_clase_defecto_que_no_esta_en_clases(
    escribir_yaml: Callable[[dict[str, Any]], Path]
) -> None:
    """El overlay no sabria de que color pintar una clase de defecto inexistente."""
    ruta = escribir_yaml({"clases": ["ok", "fisura"], "clases_defecto": ["fisura", "quemado"]})

    with pytest.raises(ValueError, match="quemado"):
        cargar_config(ruta)


def test_clases_debe_ser_lista_de_textos(
    escribir_yaml: Callable[[dict[str, Any]], Path]
) -> None:
    ruta = escribir_yaml({"clases": {"0": "ok"}})

    with pytest.raises(ValueError, match="clases"):
        cargar_config(ruta)


def test_clases_defecto_vacia_es_valido(
    escribir_yaml: Callable[[dict[str, Any]], Path]
) -> None:
    """Un dataset donde todo es 'ok' todavia no tiene defectos que pintar en rojo."""
    ruta = escribir_yaml({"clases": ["ok"], "clases_defecto": []})

    cfg = cargar_config(ruta)

    assert cfg.clases_defecto == []


def test_guardar_config_valida_antes_de_escribir(tmp_path: Path) -> None:
    """Un config invalido no llega al disco."""
    invalida = AppConfig(detector=DetectorConfig(confianza=5.0))
    ruta = tmp_path / "config.yaml"

    with pytest.raises(ValueError, match="confianza"):
        guardar_config(invalida, ruta)

    assert not ruta.exists()


def test_guardar_config_rechaza_lo_que_no_es_config(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        guardar_config({"camara": {}}, tmp_path / "config.yaml")  # type: ignore[arg-type]


def test_raiz_repo_contiene_configs() -> None:
    raiz = raiz_repo()

    assert raiz.is_dir()
    assert (raiz / "configs").is_dir()
    assert (raiz / "src" / "afilado").is_dir()


def test_ruta_absoluta_resuelve_relativas_contra_la_raiz() -> None:
    resuelta = ruta_absoluta("models/afilado_best.pt")

    assert resuelta.is_absolute()
    assert resuelta == raiz_repo() / "models" / "afilado_best.pt"


def test_ruta_absoluta_respeta_las_absolutas(tmp_path: Path) -> None:
    destino = tmp_path / "pesos.pt"

    assert ruta_absoluta(destino) == destino


def test_config_del_repo_es_valida() -> None:
    """El configs/config.yaml versionado tiene que cargar sin errores."""
    if not (raiz_repo() / "configs" / "config.yaml").is_file():
        pytest.skip("todavia no hay configs/config.yaml en el repo")

    cfg = cargar_config()

    assert cfg.clases
    assert set(cfg.clases_defecto) <= set(cfg.clases)
