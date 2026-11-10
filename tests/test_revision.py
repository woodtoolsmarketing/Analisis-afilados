"""Pruebas de la pantalla de revision (la del microscopio).

No abren ninguna ventana: `componer` devuelve la imagen y el mapa de botones, y todo lo que
importa se puede comprobar sobre eso. Lo que se prueba es lo que romperia la pantalla en el
taller sin que nadie lo note al mirarla de reojo: botones que se pisan entre si, botones que
se salen del lienzo, y el clic que cae en el boton equivocado.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from afilado import revision
from afilado.config import AppConfig
from afilado.tipos import Veredicto

_RAIZ = Path(__file__).resolve().parents[1]


@pytest.fixture
def cfg() -> AppConfig:
    return AppConfig(clases=["aprobada", "rechazada"], clases_defecto=["rechazada"])


def _imagen(ancho: int = 800, alto: int = 600) -> np.ndarray:
    return np.full((alto, ancho, 3), 120, np.uint8)


def _rects_se_solapan(a, b) -> bool:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    return ax1 < bx2 and bx1 < ax2 and ay1 < by2 and by1 < ay2


@pytest.mark.parametrize("congelado", [False, True])
def test_los_botones_no_se_pisan_entre_si(cfg: AppConfig, congelado: bool) -> None:
    """Dos botones superpuestos significan que un clic puede guardar la clase equivocada."""
    estado = revision.EstadoRevision(congelado=congelado, capturas=12)

    pantalla = revision.componer(_imagen(), _imagen(), cfg, estado)

    rects = list(pantalla.botones.values())
    for i, primero in enumerate(rects):
        for segundo in rects[i + 1 :]:
            assert not _rects_se_solapan(primero, segundo), pantalla.botones


@pytest.mark.parametrize("congelado", [False, True])
def test_los_botones_entran_en_el_lienzo(cfg: AppConfig, congelado: bool) -> None:
    """Un boton dibujado fuera del lienzo no se ve, pero igual acepta clics: peor todavia."""
    estado = revision.EstadoRevision(congelado=congelado, capturas=3)

    pantalla = revision.componer(_imagen(), _imagen(), cfg, estado)
    alto, ancho = pantalla.imagen.shape[:2]

    for nombre, (x1, y1, x2, y2) in pantalla.botones.items():
        assert 0 <= x1 < x2 <= ancho, f"{nombre} se sale a lo ancho"
        assert 0 <= y1 < y2 <= alto, f"{nombre} se sale a lo alto"


def test_el_clic_cae_en_el_boton_que_se_ve(cfg: AppConfig) -> None:
    estado = revision.EstadoRevision(congelado=True)

    pantalla = revision.componer(_imagen(), _imagen(), cfg, estado)

    for nombre, (x1, y1, x2, y2) in pantalla.botones.items():
        centro = ((x1 + x2) // 2, (y1 + y2) // 2)
        assert pantalla.boton_en(*centro) == nombre
    assert pantalla.boton_en(0, 0) is None


def test_en_vivo_solo_se_puede_revisar_o_salir(cfg: AppConfig) -> None:
    """Antes de congelar no hay nada que guardar: ofrecerlo invitaria a archivar una foto
    movida, que es basura para el dataset."""
    pantalla = revision.componer(_imagen(), _imagen(), cfg, revision.EstadoRevision())

    assert set(pantalla.botones) == {"revisar", "salir"}


def test_congelado_aparecen_los_de_guardar(cfg: AppConfig) -> None:
    estado = revision.EstadoRevision(congelado=True)

    pantalla = revision.componer(_imagen(), _imagen(), cfg, estado)

    assert set(pantalla.botones) == {"seguir", "aprobada", "rechazada", "salir"}


def test_sin_referencia_no_explota_y_avisa(cfg: AppConfig) -> None:
    """El dia 1 no hay foto de referencia y el programa tiene que arrancar igual."""
    estado = revision.EstadoRevision(referencia_faltante=True)

    pantalla = revision.componer(_imagen(), None, cfg, estado)

    assert pantalla.imagen.shape[1] == revision.ANCHO_LIENZO
    assert "revisar" in pantalla.botones


@pytest.mark.parametrize(
    "forma", [(600, 800, 3), (480, 640, 3), (1080, 1920, 3), (600, 800), (240, 320, 3)]
)
def test_anda_con_cualquier_resolucion_de_camara(cfg: AppConfig, forma) -> None:
    """El microscopio da 800x600, una webcam 640x480, otra 1920x1080. Todas tienen que
    entrar en el mismo lienzo sin deformarse ni desbordar."""
    pantalla = revision.componer(np.zeros(forma, np.uint8), _imagen(), cfg, revision.EstadoRevision())

    assert pantalla.imagen.shape[1] == revision.ANCHO_LIENZO
    assert pantalla.imagen.shape[0] > 0


def test_no_se_toca_la_imagen_que_entra(cfg: AppConfig) -> None:
    """La foto que se archiva tiene que ser la de la camara, sin un pixel dibujado encima."""
    vivo = _imagen()
    copia = vivo.copy()

    revision.componer(vivo, _imagen(), cfg, revision.EstadoRevision(congelado=True))

    assert np.array_equal(vivo, copia)


def test_el_encaje_respeta_la_proporcion() -> None:
    """Estirar la imagen para llenar el panel haria comparar un diente deformado contra una
    referencia sin deformar."""
    ancha = np.zeros((100, 400, 3), np.uint8)
    ancha[:, :] = 200

    panel = revision._encajar(ancha, 200, 200)

    assert panel.shape == (200, 200, 3)
    # con 4:1 metido en un cuadrado, arriba y abajo tienen que quedar bandas negras
    assert int(panel[0].max()) == 0
    assert int(panel[-1].max()) == 0
    assert int(panel[100].max()) > 0


def test_los_textos_de_pantalla_no_llevan_acentos() -> None:
    """cv2.putText usa fuentes Hershey, que NO tienen acentos: cada tilde sale como '?'.

    Es un error facil de cometer y dificil de ver en una revision de codigo, porque el
    codigo se lee perfecto y solo se rompe en la pantalla del taller.
    """
    culpables: dict[str, list[str]] = {}
    for ruta in sorted((_RAIZ / "src" / "afilado").rglob("*.py")):
        raros = sorted({c for c in ruta.read_text(encoding="utf-8") if ord(c) > 127})
        if raros:
            culpables[ruta.name] = raros

    assert not culpables, (
        "estos modulos tienen caracteres que cv2.putText dibuja como '?' y que la consola "
        f"de Windows tambien rompe: {culpables}"
    )


# ---------------------------------------------------------------------------
# MODO SIMPLE: el cartel RESULTADO con APROBADO / DESAPROBADO.
#
# Lo que se prueba es el umbral y el COLOR, porque el color es lo unico que el
# operario mira de lejos. Un verde cuando tendria que ser rojo deja pasar una
# sierra que hay que tirar.
# ---------------------------------------------------------------------------


def _veredicto(aprobada: float) -> Veredicto:
    """Un veredicto binario con la probabilidad dada para 'aprobada'."""
    gana_aprobada = aprobada >= 0.5
    return Veredicto(
        clase="aprobada" if gana_aprobada else "rechazada",
        clase_id=0 if gana_aprobada else 1,
        confianza=aprobada if gana_aprobada else 1.0 - aprobada,
        probabilidades={"aprobada": aprobada, "rechazada": 1.0 - aprobada},
    )


@pytest.mark.parametrize("probabilidad", [0.0, 0.3, 0.7, 0.95, 1.0])
def test_la_coincidencia_es_la_probabilidad_de_las_clases_sanas(
    cfg: AppConfig, probabilidad: float
) -> None:
    assert revision.coincidencia(_veredicto(probabilidad), cfg) == pytest.approx(probabilidad)


def test_sin_veredicto_no_hay_coincidencia(cfg: AppConfig) -> None:
    """Sin modelo no se inventa un numero: el cartel va a decir SIN MODELO."""
    assert revision.coincidencia(None, cfg) is None


def test_la_coincidencia_suma_todas_las_clases_sanas() -> None:
    """Si manana el esquema pasa de aprobada/rechazada a varios defectos con nombre."""
    cfg = AppConfig(
        clases=["ok", "aceptable", "fisura", "oxido"],
        clases_defecto=["fisura", "oxido"],
    )
    veredicto = Veredicto(
        clase="ok",
        clase_id=0,
        confianza=0.5,
        probabilidades={"ok": 0.5, "aceptable": 0.3, "fisura": 0.15, "oxido": 0.05},
    )

    assert revision.coincidencia(veredicto, cfg) == pytest.approx(0.8)


def test_un_veredicto_sin_desglose_igual_dice_algo(cfg: AppConfig) -> None:
    """Sin probabilidades por clase, la confianza de la clase ganadora alcanza."""
    sana = Veredicto(clase="aprobada", clase_id=0, confianza=0.9, probabilidades={})
    defectuosa = Veredicto(clase="rechazada", clase_id=1, confianza=0.9, probabilidades={})

    assert revision.coincidencia(sana, cfg) == pytest.approx(0.9)
    assert revision.coincidencia(defectuosa, cfg) == pytest.approx(0.1)


def _color_del_cartel(pantalla) -> tuple[int, int, int]:
    """Lee el color de fondo del cartel RESULTADO, abajo al centro."""
    alto, ancho = pantalla.imagen.shape[:2]
    # el cartel ocupa la franja de abajo, arriba de los botones
    fila = alto - 62 - revision._ALTO_RESULTADO // 2
    return tuple(int(v) for v in pantalla.imagen[fila, 40])


@pytest.mark.parametrize(
    "probabilidad, espera_verde",
    [(1.0, True), (0.85, True), (0.70, True), (0.6999, False), (0.5, False), (0.0, False)],
)
def test_el_umbral_decide_el_color_del_cartel(
    cfg: AppConfig, probabilidad: float, espera_verde: bool
) -> None:
    """70% justo cuenta como APROBADO: el umbral es "al menos", no "mas de"."""
    cfg.interfaz.umbral_aprobacion = 0.70
    estado = revision.EstadoRevision(congelado=True, veredicto=_veredicto(probabilidad))

    pantalla = revision.componer_simple(_imagen(), _imagen(), cfg, estado)
    azul, verde, rojo = _color_del_cartel(pantalla)

    if espera_verde:
        assert verde > 120 and rojo < 80, f"deberia ser verde y dio {(azul, verde, rojo)}"
    else:
        assert rojo > 120 and verde < 80, f"deberia ser rojo y dio {(azul, verde, rojo)}"


def test_el_umbral_se_puede_mover(cfg: AppConfig) -> None:
    """Subir el umbral tiene que cambiar el veredicto de la misma pieza."""
    estado = revision.EstadoRevision(congelado=True, veredicto=_veredicto(0.75))

    cfg.interfaz.umbral_aprobacion = 0.70
    verde = _color_del_cartel(revision.componer_simple(_imagen(), _imagen(), cfg, estado))
    cfg.interfaz.umbral_aprobacion = 0.90
    rojo = _color_del_cartel(revision.componer_simple(_imagen(), _imagen(), cfg, estado))

    assert verde[1] > 120, "con umbral 70 una pieza al 75% tiene que aprobar"
    assert rojo[2] > 120, "con umbral 90 la misma pieza tiene que desaprobar"


def test_antes_de_revisar_no_dice_ni_aprobado_ni_desaprobado(cfg: AppConfig) -> None:
    """En vivo el cartel esta en gris: dar un veredicto sin que se lo pidan seria juzgar
    una imagen que el operario todavia no encuadro."""
    pantalla = revision.componer_simple(_imagen(), _imagen(), cfg, revision.EstadoRevision())
    azul, verde, rojo = _color_del_cartel(pantalla)

    assert verde < 120 and rojo < 120
    assert set(pantalla.botones) == {"revisar", "salir"}


@pytest.mark.parametrize("congelado", [False, True])
def test_los_botones_del_modo_simple_no_se_pisan(cfg: AppConfig, congelado: bool) -> None:
    estado = revision.EstadoRevision(congelado=congelado, veredicto=_veredicto(0.9))

    pantalla = revision.componer_simple(_imagen(), _imagen(), cfg, estado)

    rects = list(pantalla.botones.values())
    alto, ancho = pantalla.imagen.shape[:2]
    for i, primero in enumerate(rects):
        assert 0 <= primero[0] < primero[2] <= ancho
        assert 0 <= primero[1] < primero[3] <= alto
        for segundo in rects[i + 1 :]:
            assert not _rects_se_solapan(primero, segundo), pantalla.botones


@pytest.mark.parametrize("forma", [(600, 800, 3), (480, 640, 3), (1080, 1920, 3)])
def test_el_modo_simple_anda_con_cualquier_camara(cfg: AppConfig, forma) -> None:
    pantalla = revision.componer_simple(
        np.zeros(forma, np.uint8), _imagen(), cfg, revision.EstadoRevision(congelado=True)
    )

    assert pantalla.imagen.shape[1] == revision.ANCHO_LIENZO


@pytest.mark.parametrize("alto_maximo", [900, 672, 600, 520])
def test_el_lienzo_entra_en_la_pantalla_que_se_le_diga(cfg: AppConfig, alto_maximo: int) -> None:
    """El lienzo se arma para ENTRAR, en vez de dejar que la ventana lo escale.

    Si no entra, OpenCV estira la imagen al ancho de la ventana y la fila de botones queda
    cortada abajo. Un boton que no se ve pero igual recibe clics es peor que no tenerlo.
    """
    estado = revision.EstadoRevision(congelado=True, veredicto=_veredicto(0.9))

    pantalla = revision.componer_simple(_imagen(), _imagen(), cfg, estado, alto_maximo)

    alto = pantalla.imagen.shape[0]
    assert alto <= alto_maximo, f"el lienzo mide {alto} y el maximo era {alto_maximo}"
    mas_bajo = max(rect[3] for rect in pantalla.botones.values())
    assert mas_bajo <= alto, "hay un boton dibujado fuera del lienzo"


@pytest.mark.parametrize("alto_maximo", [400, 200, 1])
def test_por_debajo_del_piso_la_pantalla_no_se_achica_mas(
    cfg: AppConfig, alto_maximo: int
) -> None:
    """Hay un piso: el cartel y los botones se respetan enteros aunque el monitor sea chico.

    Achicarlos para entrar en cualquier pantalla dejaria la herramienta sin lo unico que
    justifica esta pantalla: un resultado legible de lejos y botones que se puedan apretar.
    """
    estado = revision.EstadoRevision(congelado=True, veredicto=_veredicto(0.9))

    pantalla = revision.componer_simple(_imagen(), _imagen(), cfg, estado, alto_maximo)

    assert pantalla.imagen.shape[0] == revision.ALTO_MINIMO_LIENZO
    mas_bajo = max(rect[3] for rect in pantalla.botones.values())
    assert mas_bajo <= pantalla.imagen.shape[0]


def test_al_achicar_se_sacrifican_los_recuadros_no_el_resultado(cfg: AppConfig) -> None:
    """Lo que se achica es la imagen de la camara; el cartel y los botones no se tocan."""
    estado = revision.EstadoRevision(congelado=True, veredicto=_veredicto(0.9))

    grande = revision.componer_simple(_imagen(), _imagen(), cfg, estado, 900)
    chica = revision.componer_simple(_imagen(), _imagen(), cfg, estado, 560)

    alto_boton_grande = grande.botones["seguir"][3] - grande.botones["seguir"][1]
    alto_boton_chica = chica.botones["seguir"][3] - chica.botones["seguir"][1]

    assert alto_boton_grande == alto_boton_chica, "se achicaron los botones"
    assert chica.imagen.shape[0] < grande.imagen.shape[0], "no se achico nada"
    # la diferencia entera salio de los recuadros, no de las franjas fijas
    assert (grande.imagen.shape[0] - chica.imagen.shape[0]) == pytest.approx(
        900 - 560, abs=revision._ALTO_FIJO_SIMPLE
    )
