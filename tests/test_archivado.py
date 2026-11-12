"""Pruebas del filtro que decide si una foto puede entrar al dataset.

POR QUE EXISTE ESTO
-------------------
En la colección del taller aparecieron fotos que no eran sierras. La peor: el cartel fijo
"Start DroidCam", archivado como sierra APROBADA. Entró porque el microscopio estaba
desenchufado, el programa cayó en la cámara virtual --avisándolo en el registro, que nadie
mira-- y la tecla 'a' guarda lo que haya en pantalla sin preguntar nada.

Una foto mal etiquetada no se nota: entra, se entrena con ella y el modelo sale raro sin
que nadie pueda decir por qué.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from afilado.cli import run_live
from afilado.config import AppConfig


class _CamaraFalsa:
    def __init__(self, congelada: bool = False) -> None:
        self.congelada = congelada


@pytest.fixture
def cfg() -> AppConfig:
    c = AppConfig(clases=["aprobada", "rechazada"], clases_defecto=["rechazada"])
    c.interfaz.nitidez_minima = 30.0
    return c


def _nitida() -> np.ndarray:
    return np.random.default_rng(5).integers(0, 255, (240, 320, 3), dtype=np.uint8)


def _movida() -> np.ndarray:
    return cv2.GaussianBlur(_nitida(), (31, 31), 0)


def test_una_foto_nitida_de_una_camara_viva_se_archiva(cfg: AppConfig) -> None:
    assert run_live.motivo_para_no_archivar(_CamaraFalsa(), _nitida(), cfg) is None


def test_no_se_archiva_con_la_camara_congelada(cfg: AppConfig) -> None:
    """El caso DroidCam. Lo que se ve no es la sierra, sea cual sea la tecla que se apriete."""
    motivo = run_live.motivo_para_no_archivar(_CamaraFalsa(congelada=True), _nitida(), cfg)

    assert motivo is not None
    assert "CONGELADA" in motivo


def test_el_control_de_foco_NO_alcanza_para_la_camara_congelada(cfg: AppConfig) -> None:
    """Esta es la razon por la que se le pregunta a la camara y no a la imagen.

    El cartel de DroidCam es texto blanco nitido sobre negro: medido sobre la foto real que
    quedo en la coleccion, da nitidez 642 --la mas alta de todo el dataset, mas que
    cualquier diente-- asi que pasa cualquier umbral de foco con comodidad.
    """
    cartel = np.zeros((240, 320, 3), np.uint8)
    cv2.putText(cartel, "Start DroidCam", (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                (255, 255, 255), 2)

    assert run_live.revision.nitidez(cartel) > cfg.interfaz.nitidez_minima, (
        "si esto falla, el cartel dejo de ser nitido y el test perdio sentido"
    )
    assert run_live.motivo_para_no_archivar(_CamaraFalsa(congelada=True), cartel, cfg)


def test_no_se_archiva_una_imagen_movida(cfg: AppConfig) -> None:
    motivo = run_live.motivo_para_no_archivar(_CamaraFalsa(), _movida(), cfg)

    assert motivo is not None
    assert "MOVIDA" in motivo


def test_el_motivo_dice_que_hacer_no_solo_que_fallo(cfg: AppConfig) -> None:
    """El operario tiene que poder arreglarlo sin llamar a nadie."""
    congelada = run_live.motivo_para_no_archivar(_CamaraFalsa(congelada=True), _nitida(), cfg)
    movida = run_live.motivo_para_no_archivar(_CamaraFalsa(), _movida(), cfg)

    assert "conecte" in congelada.lower()
    assert "enfoque" in movida.lower()


def test_el_motivo_de_la_imagen_movida_trae_el_numero(cfg: AppConfig) -> None:
    """Sin el numero no se sabe si falto poco o esta totalmente fuera de foco."""
    motivo = run_live.motivo_para_no_archivar(_CamaraFalsa(), _movida(), cfg)

    assert any(c.isdigit() for c in motivo)


def test_un_video_no_se_considera_congelado(cfg: AppConfig) -> None:
    """Las fuentes que no son camara no tienen el atributo; no pueden bloquear el archivado."""

    class _SinAtributo:
        pass

    assert run_live.motivo_para_no_archivar(_SinAtributo(), _nitida(), cfg) is None


def test_el_umbral_sale_del_config(cfg: AppConfig) -> None:
    """Un taller con otra iluminacion puede necesitar otro umbral."""
    cfg.interfaz.nitidez_minima = 1e9

    assert run_live.motivo_para_no_archivar(_CamaraFalsa(), _nitida(), cfg) is not None


# ---------------------------------------------------------------------------
# EL CONTEO ACUMULADO. total_guardados solo cuenta la sesion actual, que no
# sirve para saber si ya alcanza para entrenar: el dataset se junta de a ratos.
# ---------------------------------------------------------------------------


def test_cuenta_las_fotos_que_ya_estaban_en_disco(tmp_path) -> None:
    from afilado.almacen import AlmacenClasificacion

    for clase, cuantas in (("aprobada", 3), ("rechazada", 2)):
        carpeta = tmp_path / clase
        carpeta.mkdir()
        for i in range(cuantas):
            cv2.imwrite(str(carpeta / f"{clase}_{i}.jpg"), np.zeros((8, 8, 3), np.uint8))

    almacen = AlmacenClasificacion(str(tmp_path), ["aprobada", "rechazada"])

    assert almacen.total_guardados == 0, "esta sesion no guardo nada"
    assert almacen.conteo_por_clase() == {"aprobada": 3, "rechazada": 2}


def test_una_clase_sin_carpeta_cuenta_cero(tmp_path) -> None:
    from afilado.almacen import AlmacenClasificacion

    almacen = AlmacenClasificacion(str(tmp_path), ["aprobada", "rechazada"])

    assert almacen.conteo_por_clase() == {"aprobada": 0, "rechazada": 0}


def test_el_conteo_sube_al_guardar(tmp_path) -> None:
    from afilado.almacen import AlmacenClasificacion

    almacen = AlmacenClasificacion(str(tmp_path), ["aprobada", "rechazada"])
    almacen.guardar(np.zeros((8, 8, 3), np.uint8), "aprobada")

    assert almacen.conteo_por_clase()["aprobada"] == 1


def test_no_cuenta_archivos_que_no_son_fotos(tmp_path) -> None:
    """Un .txt o un .ultima tirado en la carpeta no es una muestra de entrenamiento."""
    from afilado.almacen import AlmacenClasificacion

    carpeta = tmp_path / "aprobada"
    carpeta.mkdir()
    cv2.imwrite(str(carpeta / "buena.jpg"), np.zeros((8, 8, 3), np.uint8))
    (carpeta / "notas.txt").write_text("no soy una foto", encoding="utf-8")

    almacen = AlmacenClasificacion(str(tmp_path), ["aprobada"])

    assert almacen.conteo_por_clase() == {"aprobada": 1}


# ---------------------------------------------------------------------------
# NO GUARDAR DOS VECES LA MISMA FOTO. Lo que arruino el dataset del taller:
# con la imagen congelada, cada pulsacion de GUARDAR escribia otra vez el MISMO
# frame. Quedaron 50 archivos por clase que eran 2 fotos, y una de ellas estaba
# archivada como buena Y como mala, byte por byte el mismo archivo.
# ---------------------------------------------------------------------------


def _almacen(tmp_path):
    from afilado.almacen import AlmacenClasificacion

    return AlmacenClasificacion(str(tmp_path), ["aprobada", "rechazada"])


def _foto(semilla: int = 1) -> np.ndarray:
    rng = np.random.default_rng(semilla)
    return rng.integers(0, 255, (120, 160, 3), dtype=np.uint8)


def test_la_misma_foto_no_entra_dos_veces(tmp_path) -> None:
    almacen = _almacen(tmp_path)
    foto = _foto()
    almacen.guardar(foto, "aprobada")

    assert almacen.ya_guardada(foto) == "aprobada"


def test_la_misma_foto_no_puede_ser_buena_Y_mala(tmp_path) -> None:
    """El caso exacto que aparecio en el dataset: un archivo identico en las dos clases."""
    almacen = _almacen(tmp_path)
    foto = _foto()
    almacen.guardar(foto, "aprobada")

    assert almacen.ya_guardada(foto) == "aprobada", "al ir a guardarla como mala, tiene que frenar"


def test_una_foto_nueva_entra_sin_problema(tmp_path) -> None:
    almacen = _almacen(tmp_path)
    almacen.guardar(_foto(1), "aprobada")

    assert almacen.ya_guardada(_foto(999)) is None


def _diente(semilla: int = 0, lado: int = 240) -> np.ndarray:
    """Una imagen CON ESTRUCTURA, parecida a lo que entrega el microscopio.

    El ruido aleatorio no sirve para probar esto: al achicarlo a 32x32 queda gris
    uniforme, y como la huella divide por el desvio, amplifica el residuo y cualquier
    cambio minimo se ve enorme. Medido, el mismo ruido apenas aclarado daba 0.08, contra
    0.0063 de dos fotos reales del taller con distinta exposicion.
    """
    img = np.full((lado, lado, 3), 60, np.uint8)
    cv2.fillPoly(img, [np.array([[10, lado - 20], [10, 120], [lado - 60, 40],
                                 [lado - 20, 70], [lado - 20, lado - 20]])], (190, 190, 190))
    cv2.line(img, (10, 120), (lado - 60, 40), (235, 235, 235), 6)
    for i in range(0, lado, 7):
        cv2.line(img, (i, 0), (i + 40 + semilla * 3, lado), (0, 0, 0), 1)
    return cv2.GaussianBlur(img, (5, 5), 0)


def test_un_cambio_suave_de_luz_no_la_convierte_en_foto_nueva(tmp_path) -> None:
    """Medido sobre fotos reales: el mismo frame con otra exposicion daba 0.0063-0.0074.

    Importa porque la lampara del microscopio no es estable. Si cada parpadeo contara como
    foto nueva, se colarian cincuenta copias igual.
    """
    almacen = _almacen(tmp_path)
    foto = _diente()
    almacen.guardar(foto, "aprobada")
    apenas_mas_oscura = np.clip(foto.astype(np.float32) * 0.9 - 10, 0, 255).astype(np.uint8)

    assert almacen.ya_guardada(apenas_mas_oscura) == "aprobada"


def test_mover_la_pieza_SI_cuenta_como_foto_nueva(tmp_path) -> None:
    """El limite del otro lado: si frenara esto, no se podria juntar el dataset."""
    almacen = _almacen(tmp_path)
    foto = _diente()
    almacen.guardar(foto, "aprobada")
    M = np.float32([[1, 0, 12], [0, 1, -9]])
    corrida = cv2.warpAffine(foto, M, foto.shape[1::-1], borderMode=cv2.BORDER_REPLICATE)

    assert almacen.ya_guardada(corrida) is None


def test_el_bucle_frena_la_repetida(tmp_path) -> None:
    from afilado.config import AppConfig as _AppConfig

    c = _AppConfig(clases=["aprobada", "rechazada"], clases_defecto=["rechazada"])
    c.interfaz.nitidez_minima = 1.0
    almacen = _almacen(tmp_path)
    foto = _foto()

    assert run_live.motivo_para_no_archivar(_CamaraFalsa(), foto, c, almacen) is None
    almacen.guardar(foto, "aprobada")
    motivo = run_live.motivo_para_no_archivar(_CamaraFalsa(), foto, c, almacen)

    assert motivo is not None and "YA LA GUARDASTE" in motivo
    assert "aprobada" in motivo, "tiene que decir en que clase quedo"


def test_sin_almacen_el_filtro_sigue_andando(tmp_path) -> None:
    """La firma vieja no puede romperse: hay otro bucle que llama sin almacen."""
    from afilado.config import AppConfig as _AppConfig

    c = _AppConfig(clases=["aprobada"], clases_defecto=[])
    c.interfaz.nitidez_minima = 1.0

    assert run_live.motivo_para_no_archivar(_CamaraFalsa(), _foto(), c) is None
