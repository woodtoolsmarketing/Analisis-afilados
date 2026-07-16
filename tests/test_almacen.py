"""Pruebas del almacen de feedback.

Dos invariantes mandan y tienen aqui sus tests mas importantes:

1) La imagen guardada es la LIMPIA. Comparar bit a bit contra el frame de entrada no
   sirve porque el JPEG es con perdida: en su lugar se verifica (a) que el array de
   entrada no se toco y (b) que en la imagen releida no hay rastro del overlay. El
   frame limpio es gris uniforme y el anotado lleva un rectangulo rojo saturado: si el
   almacen guardara el anotado por error, el rojo aparaceria y el test lo cazaria,
   sobreviva lo que sobreviva a la compresion.

2) El gemelo de la imagen es el pre-etiquetado YOLO, no el informe: si el informe en
   castellano viviera junto a la imagen, Roboflow lo leeria como etiquetas.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from afilado.almacen import AlmacenFeedback
from afilado.config import FeedbackConfig
from afilado.tipos import Deteccion, Escala, Medida, ResultadoFrame

ALTO, ANCHO = 240, 320
GRIS_FONDO = 40
CLASES = ["ok", "desgastado", "fisura", "astillado", "oxido"]
PATRON_DIA = re.compile(r"^\d{4}-\d{2}-\d{2}$")


@pytest.fixture
def cfg(tmp_path: Path) -> FeedbackConfig:
    return FeedbackConfig(
        habilitado=True,
        directorio=str(tmp_path / "feedback"),
        guardar_json=True,
        guardar_anotada=True,
        calidad_jpg=100,
    )


@pytest.fixture
def almacen(cfg: FeedbackConfig) -> AlmacenFeedback:
    return AlmacenFeedback(cfg, CLASES)


@pytest.fixture
def frame_limpio() -> np.ndarray:
    """Frame gris uniforme: cualquier pixel de color delataria un overlay."""
    return np.full((ALTO, ANCHO, 3), GRIS_FONDO, dtype=np.uint8)


@pytest.fixture
def frame_anotado(frame_limpio: np.ndarray) -> np.ndarray:
    """Copia con overlay rojo saturado, como la que ve el operario en pantalla."""
    anotado = frame_limpio.copy()
    cv2.rectangle(anotado, (60, 60), (260, 180), (0, 0, 255), 4)
    cv2.putText(anotado, "desgastado 93%", (60, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    return anotado


def _escala_valida() -> Escala:
    return Escala(
        mm_por_px=0.2, px_por_mm=5.0, valida=True, fuente="aruco", id_marcador=0, error_lados=0.01
    )


def _medida(escala: Escala) -> Medida:
    medida = Medida(
        largo_px=100.0,
        ancho_px=50.0,
        area_px=5000.0,
        angulo_deg=30.0,
        caja_rotada_px=np.zeros((4, 2), dtype=np.int32),
        centro_px=(160.0, 120.0),
    )
    if escala.valida and escala.mm_por_px:
        medida.largo_mm = 20.0
        medida.ancho_mm = 10.0
        medida.area_mm2 = 200.0
        medida.fiable = True
    return medida


def _mascara() -> np.ndarray:
    mascara = np.zeros((ALTO, ANCHO), dtype=np.uint8)
    cv2.rectangle(mascara, (110, 95), (210, 145), 255, -1)
    return mascara


def _deteccion(
    clase_id: int = 1,
    clase: str = "desgastado",
    con_mascara: bool = False,
    escala: Escala | None = None,
) -> Deteccion:
    escala = escala if escala is not None else _escala_valida()
    return Deteccion(
        clase_id=clase_id,
        clase=clase,
        confianza=0.93,
        xyxy=(110.0, 95.0, 210.0, 145.0),
        mascara=_mascara() if con_mascara else None,
        medida=_medida(escala),
    )


def _resultado(
    detecciones: list[Deteccion] | None = None,
    descartadas: list[Deteccion] | None = None,
    escala: Escala | None = None,
) -> ResultadoFrame:
    return ResultadoFrame(
        detecciones=detecciones if detecciones is not None else [_deteccion()],
        descartadas=descartadas if descartadas is not None else [],
        escala=escala if escala is not None else _escala_valida(),
        roi=(48, 36, 272, 204),
        fps=27.5,
        forma_frame=(ALTO, ANCHO),
        modelo="YOLO11n-seg (cpu)",
    )


def _carpetas(ruta_imagen: Path) -> tuple[Path, Path, Path, Path]:
    """A partir de la ruta de la imagen devuelve (dia, imagenes, etiquetas, reportes)."""
    dir_imagenes = ruta_imagen.parent
    dia = dir_imagenes.parent
    return dia, dir_imagenes, dia / "etiquetas", dia / "reportes"


def test_estructura_de_carpetas_por_dia(
    almacen: AlmacenFeedback, frame_limpio: np.ndarray
) -> None:
    ruta = almacen.guardar(frame_limpio, _resultado(), etiqueta="fallo")

    dia, dir_imagenes, dir_etiquetas, dir_reportes = _carpetas(ruta)

    assert ruta.is_file()
    assert ruta.suffix == ".jpg"
    assert dir_imagenes.name == "imagenes"
    assert PATRON_DIA.match(dia.name)
    assert dir_etiquetas.is_dir()
    assert dir_reportes.is_dir()
    assert (dir_etiquetas / f"{ruta.stem}.txt").is_file()
    assert (dir_reportes / f"{ruta.stem}.txt").is_file()
    assert (dir_reportes / f"{ruta.stem}.json").is_file()


def test_el_nombre_lleva_la_etiqueta_y_un_sello_con_milisegundos(
    almacen: AlmacenFeedback, frame_limpio: np.ndarray
) -> None:
    ruta = almacen.guardar(frame_limpio, _resultado(), etiqueta="bueno")

    assert re.match(r"^bueno_\d{8}_\d{6}_\d{3}$", ruta.stem)


def test_el_informe_no_comparte_carpeta_con_la_imagen(
    almacen: AlmacenFeedback, frame_limpio: np.ndarray
) -> None:
    """El .txt gemelo de la imagen es el pre-etiquetado YOLO, no el informe legible.

    Si el informe en castellano viviera en imagenes/, YOLO y Roboflow lo tomarian por
    el archivo de etiquetas de la foto y lo importarian como basura.
    """
    ruta = almacen.guardar(frame_limpio, _resultado(), etiqueta="fallo")
    _, dir_imagenes, dir_etiquetas, dir_reportes = _carpetas(ruta)

    assert not (dir_imagenes / f"{ruta.stem}.txt").exists()
    assert list(dir_imagenes.iterdir()) == [ruta]

    gemelo = (dir_etiquetas / f"{ruta.stem}.txt").read_text(encoding="utf-8")
    informe = (dir_reportes / f"{ruta.stem}.txt").read_text(encoding="utf-8")

    # El gemelo es puro numero; el informe es prosa.
    assert re.fullmatch(r"[0-9.\s]+", gemelo)
    assert "INFORME DE CAPTURA" in informe


def test_no_se_escribe_sobre_el_frame_de_entrada(
    almacen: AlmacenFeedback, frame_limpio: np.ndarray, frame_anotado: np.ndarray
) -> None:
    copia = frame_limpio.copy()

    almacen.guardar(frame_limpio, _resultado(), etiqueta="fallo", frame_anotado=frame_anotado)

    assert np.array_equal(frame_limpio, copia)


def test_la_imagen_guardada_no_tiene_ni_un_pixel_del_overlay(
    almacen: AlmacenFeedback, frame_limpio: np.ndarray, frame_anotado: np.ndarray
) -> None:
    """Guardar el frame anotado ensenaria a la red a detectar rectangulos rojos."""
    ruta = almacen.guardar(
        frame_limpio, _resultado(), etiqueta="fallo", frame_anotado=frame_anotado
    )

    guardada = cv2.imread(str(ruta))

    assert guardada is not None
    assert guardada.shape == frame_limpio.shape
    # El fondo limpio es gris uniforme: el JPEG lo mueve unas unidades, nunca al rojo.
    assert int(guardada.max()) - int(guardada.min()) < 20
    assert int(guardada[:, :, 2].max()) < GRIS_FONDO + 30


def test_la_copia_anotada_va_aparte_y_si_lleva_el_overlay(
    almacen: AlmacenFeedback, frame_limpio: np.ndarray, frame_anotado: np.ndarray
) -> None:
    ruta = almacen.guardar(
        frame_limpio, _resultado(), etiqueta="fallo", frame_anotado=frame_anotado
    )
    dia, _, _, _ = _carpetas(ruta)

    revision = dia / "revision" / f"{ruta.stem}_anotada.jpg"

    assert revision.is_file()
    anotada = cv2.imread(str(revision))
    assert anotada is not None
    assert int(anotada[:, :, 2].max()) > 180  # el rojo del overlay sigue ahi


def test_sin_guardar_anotada_no_se_crea_revision(
    cfg: FeedbackConfig, frame_limpio: np.ndarray, frame_anotado: np.ndarray
) -> None:
    cfg.guardar_anotada = False
    almacen = AlmacenFeedback(cfg, CLASES)

    ruta = almacen.guardar(
        frame_limpio, _resultado(), etiqueta="fallo", frame_anotado=frame_anotado
    )
    dia, _, _, _ = _carpetas(ruta)

    assert not (dia / "revision").exists()


def test_sin_frame_anotado_no_se_crea_revision(
    almacen: AlmacenFeedback, frame_limpio: np.ndarray
) -> None:
    ruta = almacen.guardar(frame_limpio, _resultado(), etiqueta="fallo")
    dia, _, _, _ = _carpetas(ruta)

    assert not (dia / "revision").exists()


def test_pre_etiquetado_de_caja_en_formato_yolo(
    almacen: AlmacenFeedback, frame_limpio: np.ndarray
) -> None:
    """Sin mascara: clase cx cy w h, todos normalizados a [0,1]."""
    deteccion = _deteccion(clase_id=1, con_mascara=False)
    ruta = almacen.guardar(frame_limpio, _resultado([deteccion]), etiqueta="fallo")
    _, _, dir_etiquetas, _ = _carpetas(ruta)

    lineas = (dir_etiquetas / f"{ruta.stem}.txt").read_text(encoding="utf-8").splitlines()

    assert len(lineas) == 1
    partes = lineas[0].split()
    assert len(partes) == 5
    assert partes[0] == "1"
    valores = [float(p) for p in partes[1:]]
    assert all(0.0 <= v <= 1.0 for v in valores)
    # xyxy (110,95,210,145) sobre un frame de 320x240.
    assert valores == pytest.approx([160 / 320, 120 / 240, 100 / 320, 50 / 240], abs=1e-5)


def test_pre_etiquetado_de_segmentacion_en_formato_yolo(
    almacen: AlmacenFeedback, frame_limpio: np.ndarray
) -> None:
    """Con mascara: clase x1 y1 x2 y2 ... normalizados; poligono de al menos 3 puntos."""
    deteccion = _deteccion(clase_id=2, clase="fisura", con_mascara=True)
    ruta = almacen.guardar(frame_limpio, _resultado([deteccion]), etiqueta="fallo")
    _, _, dir_etiquetas, _ = _carpetas(ruta)

    partes = (dir_etiquetas / f"{ruta.stem}.txt").read_text(encoding="utf-8").split()

    assert partes[0] == "2"
    coordenadas = [float(p) for p in partes[1:]]
    assert len(coordenadas) % 2 == 0
    assert len(coordenadas) >= 6
    assert all(0.0 <= v <= 1.0 for v in coordenadas)


def test_pre_etiquetado_de_varias_detecciones_una_por_linea(
    almacen: AlmacenFeedback, frame_limpio: np.ndarray
) -> None:
    detecciones = [
        _deteccion(clase_id=0, clase="ok"),
        _deteccion(clase_id=3, clase="astillado"),
    ]
    ruta = almacen.guardar(frame_limpio, _resultado(detecciones), etiqueta="fallo")
    _, _, dir_etiquetas, _ = _carpetas(ruta)

    lineas = (dir_etiquetas / f"{ruta.stem}.txt").read_text(encoding="utf-8").splitlines()

    assert [linea.split()[0] for linea in lineas] == ["0", "3"]


def test_sin_detecciones_el_pre_etiquetado_queda_vacio(
    almacen: AlmacenFeedback, frame_limpio: np.ndarray
) -> None:
    """Un fondo sin piezas es un negativo legitimo para el dataset: etiqueta vacia."""
    ruta = almacen.guardar(frame_limpio, _resultado([]), etiqueta="fallo")
    _, _, dir_etiquetas, dir_reportes = _carpetas(ruta)

    assert (dir_etiquetas / f"{ruta.stem}.txt").read_text(encoding="utf-8") == ""
    assert "Ninguno supero los filtros." in (
        dir_reportes / f"{ruta.stem}.txt"
    ).read_text(encoding="utf-8")


def test_solo_se_pre_etiquetan_las_detecciones_aceptadas(
    almacen: AlmacenFeedback, frame_limpio: np.ndarray
) -> None:
    descartada = _deteccion(clase_id=4, clase="oxido")
    descartada.descartada_por = "roi"
    ruta = almacen.guardar(
        frame_limpio, _resultado([_deteccion(clase_id=1)], [descartada]), etiqueta="fallo"
    )
    _, _, dir_etiquetas, _ = _carpetas(ruta)

    lineas = (dir_etiquetas / f"{ruta.stem}.txt").read_text(encoding="utf-8").splitlines()

    assert [linea.split()[0] for linea in lineas] == ["1"]


def test_deteccion_geometrica_va_a_clase_0_y_se_avisa_en_el_informe(
    almacen: AlmacenFeedback, frame_limpio: np.ndarray
) -> None:
    """El detector geometrico no clasifica: la clase del pre-etiquetado es provisional."""
    geometrica = _deteccion(clase_id=-1, clase="sin_clasificar")
    geometrica.confianza = 0.0
    ruta = almacen.guardar(frame_limpio, _resultado([geometrica]), etiqueta="captura")
    _, _, dir_etiquetas, dir_reportes = _carpetas(ruta)

    linea = (dir_etiquetas / f"{ruta.stem}.txt").read_text(encoding="utf-8").split()
    informe = (dir_reportes / f"{ruta.stem}.txt").read_text(encoding="utf-8")
    datos = json.loads((dir_reportes / f"{ruta.stem}.json").read_text(encoding="utf-8"))

    assert linea[0] == "0"
    assert "provisional" in informe
    assert datos["detecciones"][0]["clase_provisional"] is True


def test_clase_id_fuera_del_listado_cae_a_0(
    almacen: AlmacenFeedback, frame_limpio: np.ndarray
) -> None:
    """Un modelo con mas clases que el config no puede escribir un indice invalido."""
    ajena = _deteccion(clase_id=99, clase="clase_de_otro_modelo")
    ruta = almacen.guardar(frame_limpio, _resultado([ajena]), etiqueta="fallo")
    _, _, dir_etiquetas, _ = _carpetas(ruta)

    assert (dir_etiquetas / f"{ruta.stem}.txt").read_text(encoding="utf-8").split()[0] == "0"


def test_el_informe_explica_lo_que_penso_la_ia(
    almacen: AlmacenFeedback, frame_limpio: np.ndarray
) -> None:
    descartada = _deteccion(clase_id=4, clase="oxido")
    descartada.descartada_por = "area_minima"
    ruta = almacen.guardar(
        frame_limpio,
        _resultado([_deteccion(clase_id=1, clase="desgastado")], [descartada]),
        etiqueta="fallo",
    )
    _, _, _, dir_reportes = _carpetas(ruta)

    informe = (dir_reportes / f"{ruta.stem}.txt").read_text(encoding="utf-8")

    assert "fallo" in informe
    assert "YOLO11n-seg (cpu)" in informe
    assert "desgastado" in informe
    assert "93.0 %" in informe
    assert "20.00 x 10.00 mm" in informe
    assert "OBJETOS DETECTADOS: 1" in informe
    assert "OBJETOS DESCARTADOS: 1" in informe
    # El motivo del descarte se traduce a algo que el operario entiende.
    assert "aserrin" in informe or "area minima" in informe
    assert "1 px = 0.2000 mm" in informe


def test_el_informe_avisa_cuando_no_hay_escala(
    almacen: AlmacenFeedback, frame_limpio: np.ndarray
) -> None:
    sin_escala = Escala()
    ruta = almacen.guardar(
        frame_limpio,
        _resultado([_deteccion(escala=sin_escala)], escala=sin_escala),
        etiqueta="fallo",
    )
    _, _, _, dir_reportes = _carpetas(ruta)

    informe = (dir_reportes / f"{ruta.stem}.txt").read_text(encoding="utf-8")

    assert "NO HAY ESCALA VALIDA" in informe
    assert "SIN ESCALA" in informe


def test_el_informe_avisa_de_la_escala_extrapolada(
    almacen: AlmacenFeedback, frame_limpio: np.ndarray
) -> None:
    extrapolada = _escala_valida()
    extrapolada.fuente = "aruco_extrapolado"
    extrapolada.frames_sin_marcador = 12
    ruta = almacen.guardar(frame_limpio, _resultado(escala=extrapolada), etiqueta="fallo")
    _, _, _, dir_reportes = _carpetas(ruta)

    informe = (dir_reportes / f"{ruta.stem}.txt").read_text(encoding="utf-8")

    assert "12 frames" in informe
    assert "ultima escala conocida" in informe


def test_el_json_es_serializable_con_tipos_de_numpy(
    almacen: AlmacenFeedback, frame_limpio: np.ndarray
) -> None:
    """El detector entrega np.float32: json.dumps reventaria sin el conversor."""
    deteccion = _deteccion(con_mascara=True)
    deteccion.confianza = np.float32(0.87)
    deteccion.clase_id = np.int64(1)
    deteccion.xyxy = tuple(np.float32(v) for v in deteccion.xyxy)

    ruta = almacen.guardar(frame_limpio, _resultado([deteccion]), etiqueta="fallo")
    _, _, _, dir_reportes = _carpetas(ruta)

    datos = json.loads((dir_reportes / f"{ruta.stem}.json").read_text(encoding="utf-8"))

    assert datos["detecciones"][0]["confianza"] == pytest.approx(0.87, abs=1e-4)
    assert datos["detecciones"][0]["tiene_mascara"] is True


def test_el_json_guarda_escala_medidas_y_descartes(
    almacen: AlmacenFeedback, frame_limpio: np.ndarray
) -> None:
    descartada = _deteccion(clase_id=4, clase="oxido")
    descartada.descartada_por = "roi"
    ruta = almacen.guardar(
        frame_limpio, _resultado([_deteccion()], [descartada]), etiqueta="fallo"
    )
    _, _, _, dir_reportes = _carpetas(ruta)

    datos = json.loads((dir_reportes / f"{ruta.stem}.json").read_text(encoding="utf-8"))

    assert datos["etiqueta_operario"] == "fallo"
    assert datos["modelo"] == "YOLO11n-seg (cpu)"
    assert datos["forma_frame"] == {"alto": ALTO, "ancho": ANCHO}
    assert datos["roi"] == [48, 36, 272, 204]
    assert datos["clases_configuradas"] == CLASES
    assert datos["escala"]["valida"] is True
    assert datos["escala"]["mm_por_px"] == pytest.approx(0.2)
    assert datos["detecciones"][0]["medida"]["largo_mm"] == pytest.approx(20.0)
    assert datos["descartadas"][0]["descartada_por"] == "roi"


def test_sin_json_no_se_escribe_el_json(
    cfg: FeedbackConfig, frame_limpio: np.ndarray
) -> None:
    cfg.guardar_json = False
    almacen = AlmacenFeedback(cfg, CLASES)

    ruta = almacen.guardar(frame_limpio, _resultado(), etiqueta="fallo")
    _, _, _, dir_reportes = _carpetas(ruta)

    assert not (dir_reportes / f"{ruta.stem}.json").exists()
    assert (dir_reportes / f"{ruta.stem}.txt").is_file()  # el informe legible sigue


def test_total_guardados_cuenta_las_capturas(
    almacen: AlmacenFeedback, frame_limpio: np.ndarray
) -> None:
    assert almacen.total_guardados == 0

    almacen.guardar(frame_limpio, _resultado(), etiqueta="fallo")
    almacen.guardar(frame_limpio, _resultado(), etiqueta="bueno")

    assert almacen.total_guardados == 2


def test_dos_capturas_seguidas_no_se_pisan(
    almacen: AlmacenFeedback, frame_limpio: np.ndarray
) -> None:
    """Dos 'e' seguidas caen en el mismo milisegundo: ninguna puede perderse."""
    rutas = [almacen.guardar(frame_limpio, _resultado(), etiqueta="fallo") for _ in range(5)]

    assert len({r.name for r in rutas}) == 5
    assert all(r.is_file() for r in rutas)
    _, dir_imagenes, dir_etiquetas, _ = _carpetas(rutas[0])
    assert len(list(dir_imagenes.iterdir())) == 5
    assert len(list(dir_etiquetas.iterdir())) == 5


def test_etiqueta_con_caracteres_peligrosos_se_sanea(
    almacen: AlmacenFeedback, frame_limpio: np.ndarray
) -> None:
    """Una etiqueta con barras crearia subcarpetas o romperia la ruta en Windows."""
    ruta = almacen.guardar(frame_limpio, _resultado(), etiqueta="../fallo raro:*?")

    assert ruta.parent.name == "imagenes"
    assert not set(ruta.stem) & set('/\\:*?"<>|. ')


def test_etiquetas_distintas_conviven_el_mismo_dia(
    almacen: AlmacenFeedback, frame_limpio: np.ndarray
) -> None:
    fallo = almacen.guardar(frame_limpio, _resultado(), etiqueta="fallo")
    bueno = almacen.guardar(frame_limpio, _resultado(), etiqueta="bueno")

    assert fallo.parent == bueno.parent
    assert fallo.name.startswith("fallo_")
    assert bueno.name.startswith("bueno_")


def test_frame_en_gris_se_guarda_sin_error(almacen: AlmacenFeedback) -> None:
    gris = np.full((ALTO, ANCHO), GRIS_FONDO, dtype=np.uint8)

    ruta = almacen.guardar(gris, _resultado(), etiqueta="fallo")

    releida = cv2.imread(str(ruta), cv2.IMREAD_GRAYSCALE)
    assert releida is not None
    assert releida.shape == (ALTO, ANCHO)
