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
from afilado.memoria import Juicio, Vecino
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

    # La pantalla clasica no lleva el boton de CONCEPTOS: el panel se abre igual con la
    # tecla 'c'. La del taller es la simple, y ahi si esta el boton.
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


def _color_pintado(pantalla, color) -> bool:
    """True si ese color aparece de verdad en la franja del resultado.

    Se busca en toda la franja en vez de leer un pixel fijo: asi el test sigue valiendo
    cuando cambia el diseno, que es justo lo que paso. Lo que importa no es DONDE esta el
    verde sino que el verde este.
    """
    # SOLO la tarjeta de resultado: la fila de botones de abajo tiene un boton verde y uno
    # rojo fijos, asi que incluirla daria siempre positivo para los dos colores.
    desde = -(revision._ALTO_BOTONES + revision._MARGEN_SIMPLE + revision._ALTO_RESULTADO)
    hasta = -(revision._ALTO_BOTONES + revision._MARGEN_SIMPLE)
    tarjeta = pantalla.imagen[desde:hasta]
    return bool(np.any(np.all(np.abs(tarjeta.astype(int) - np.array(color)) < 14, axis=2)))


@pytest.mark.parametrize(
    "probabilidad, espera_aprobado",
    [(1.0, True), (0.85, True), (0.70, True), (0.6999, False), (0.5, False), (0.0, False)],
)
def test_el_umbral_decide_el_veredicto(
    cfg: AppConfig, probabilidad: float, espera_aprobado: bool
) -> None:
    """70% justo cuenta como APROBADO: el umbral es "al menos", no "mas de"."""
    cfg.interfaz.umbral_aprobacion = 0.70
    estado = revision.EstadoRevision(congelado=True, veredicto=_veredicto(probabilidad))

    titulo, _, color, aprobado = revision.estado_resultado(estado, cfg)

    assert aprobado is espera_aprobado
    assert titulo == ("APROBADO" if espera_aprobado else "DESAPROBADO")
    assert color == (revision._VERDE_OK if espera_aprobado else revision._ROJO_NO)


@pytest.mark.parametrize("probabilidad, espera_aprobado", [(0.95, True), (0.2, False)])
def test_el_color_del_veredicto_se_pinta_de_verdad(
    cfg: AppConfig, probabilidad: float, espera_aprobado: bool
) -> None:
    """El color es lo unico que el operario mira de lejos: tiene que estar en la imagen."""
    estado = revision.EstadoRevision(congelado=True, veredicto=_veredicto(probabilidad))
    _, _, color, _ = revision.estado_resultado(estado, cfg)

    pantalla = revision.componer_simple(_imagen(), _imagen(), cfg, estado)

    assert _color_pintado(pantalla, color)
    contrario = revision._ROJO_NO if espera_aprobado else revision._VERDE_OK
    assert not _color_pintado(pantalla, contrario), "se pinto tambien el color opuesto"


def test_el_umbral_se_puede_mover(cfg: AppConfig) -> None:
    """Subir el umbral tiene que cambiar el veredicto de la misma pieza."""
    estado = revision.EstadoRevision(congelado=True, veredicto=_veredicto(0.75))

    cfg.interfaz.umbral_aprobacion = 0.70
    assert revision.estado_resultado(estado, cfg)[3] is True
    cfg.interfaz.umbral_aprobacion = 0.90
    assert revision.estado_resultado(estado, cfg)[3] is False


def test_antes_de_revisar_no_dice_ni_aprobado_ni_desaprobado(cfg: AppConfig) -> None:
    """En vivo el cartel esta en gris: dar un veredicto sin que se lo pidan seria juzgar
    una imagen que el operario todavia no encuadro."""
    estado = revision.EstadoRevision()
    pantalla = revision.componer_simple(_imagen(), _imagen(), cfg, estado)

    titulo, _, _, aprobado = revision.estado_resultado(estado, cfg)
    assert titulo == "ESPERANDO" and aprobado is None
    assert not _color_pintado(pantalla, revision._VERDE_OK)
    assert not _color_pintado(pantalla, revision._ROJO_NO)
    assert set(pantalla.botones) == {"revisar", "conceptos", "salir"}


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


# ---------------------------------------------------------------------------
# EVALUACION EN VIVO. El veredicto se actualiza solo, sin apretar nada, y eso
# trae dos riesgos que estos tests cubren: opinar sobre una imagen movida, y
# parpadear entre aprobado y rechazado con cada temblor de la mano.
# ---------------------------------------------------------------------------


def test_la_nitidez_cae_con_el_desenfoque() -> None:
    """Es la medida con la que se decide si vale la pena opinar.

    Medido sobre una foto real del microscopio: enfocada 94.6, con un desenfoque de 3x3
    baja a 7.1. Aca se reproduce esa caida sobre una imagen sintetica con textura.
    """
    rng = np.random.default_rng(7)
    textura = rng.integers(0, 255, (200, 200, 3), dtype=np.uint8)

    enfocada = revision.nitidez(textura)
    borrosa = revision.nitidez(cv2.GaussianBlur(textura, (9, 9), 0))

    assert enfocada > borrosa * 5, f"enfocada {enfocada:.0f}, borrosa {borrosa:.0f}"


def test_una_imagen_plana_no_tiene_nitidez() -> None:
    """Una pared lisa o la tapa puesta: nada que enfocar, nada sobre que opinar."""
    assert revision.nitidez(np.full((100, 100, 3), 128, np.uint8)) < 1.0


def test_enfocando_no_da_veredicto(cfg: AppConfig) -> None:
    """Mientras la imagen no sirve, el cartel no puede decir aprobado ni desaprobado.

    Mostrar un veredicto sobre una foto movida es peor que no mostrar nada: el operario
    no tiene forma de saber que lo que vio no valia.
    """
    estado = revision.EstadoRevision(esperando_foco=True, nitidez=8.0)

    titulo, detalle, color, aprobado = revision.estado_resultado(estado, cfg)

    assert titulo == "ENFOCANDO"
    assert aprobado is None
    assert color == revision._AMBAR
    assert "8" in detalle and "30" in detalle, "tiene que mostrar el numero medido y el minimo"


def test_enfocando_le_gana_a_un_veredicto_viejo(cfg: AppConfig) -> None:
    """Si la pieza se movio, el veredicto anterior ya no describe lo que se ve."""
    estado = revision.EstadoRevision(
        esperando_foco=True, nitidez=5.0, veredicto=_veredicto(0.99)
    )

    titulo, _, _, aprobado = revision.estado_resultado(estado, cfg)

    assert titulo == "ENFOCANDO" and aprobado is None


def test_en_vivo_con_foco_si_da_veredicto(cfg: AppConfig) -> None:
    estado = revision.EstadoRevision(
        esperando_foco=False, nitidez=95.0, veredicto=_veredicto(0.9)
    )

    titulo, _, _, aprobado = revision.estado_resultado(estado, cfg)

    assert titulo == "APROBADO" and aprobado is True


def test_el_cartel_de_enfocando_se_dibuja(cfg: AppConfig) -> None:
    estado = revision.EstadoRevision(esperando_foco=True, nitidez=12.0)

    pantalla = revision.componer_simple(_imagen(), _imagen(), cfg, estado)

    assert _color_pintado(pantalla, revision._AMBAR)
    assert not _color_pintado(pantalla, revision._VERDE_OK)
    assert not _color_pintado(pantalla, revision._ROJO_NO)


def test_sin_evaluar_en_vivo_sigue_diciendo_esperando(cfg: AppConfig) -> None:
    """El modo a pedido no cambia: sin nitidez medida, el cartel espera el boton."""
    titulo, detalle, _, _ = revision.estado_resultado(revision.EstadoRevision(), cfg)

    assert titulo == "ESPERANDO" and "REVISAR" in detalle


# ---------------------------------------------------------------------------
# MODO RECOLECCION: sin modelo entrenado. El programa no juzga, pero sigue
# siendo util --se compara a ojo contra la foto de modelo y se archiva con
# a/r-- y el medidor de foco importa MAS que con modelo: una foto movida que
# entra al dataset queda etiquetada como buena o mala sin que se vea nada.
# ---------------------------------------------------------------------------


def test_sin_modelo_no_se_inventa_un_veredicto(cfg: AppConfig) -> None:
    """Un veredicto fabricado es peor que ninguno: el operario no puede saber que no valia.

    Paso de verdad en el taller. El modelo que viajaba en el paquete estaba entrenado con
    fotos cenitales de la sierra entera, y sobre primeros planos de microscopio devolvia
    numeros al azar: 62%, 58%, 49%, 38%, 25% sobre cinco fotos de la misma pieza bien
    puesta. La pantalla mostraba DESAPROBADO 36% con las dos imagenes practicamente
    iguales.
    """
    estado = revision.EstadoRevision(congelado=True, veredicto=None)

    titulo, _, _, aprobado = revision.estado_resultado(estado, cfg)

    assert titulo == "SIN MODELO"
    assert aprobado is None, "sin modelo no se aprueba ni se desaprueba"


def test_sin_modelo_pero_enfocado_avisa_que_falta_el_modelo(cfg: AppConfig) -> None:
    """Con la nitidez ya medida en vivo, el cartel no puede volver a 'apriete REVISAR'."""
    estado = revision.EstadoRevision(esperando_foco=False, nitidez=95.0, veredicto=None)

    titulo, _, _, aprobado = revision.estado_resultado(estado, cfg)

    assert (titulo, aprobado) == ("SIN MODELO", None)


def test_sin_modelo_el_medidor_de_foco_sigue_avisando(cfg: AppConfig) -> None:
    """Es el unico aviso que queda para no archivar una foto movida."""
    estado = revision.EstadoRevision(esperando_foco=True, nitidez=9.0, veredicto=None)

    titulo, detalle, color, aprobado = revision.estado_resultado(estado, cfg)

    assert titulo == "ENFOCANDO" and aprobado is None
    assert "9" in detalle, "tiene que mostrar la nitidez medida"


def test_sin_modelo_la_pantalla_no_pinta_verde_ni_rojo(cfg: AppConfig) -> None:
    estado = revision.EstadoRevision(congelado=True, veredicto=None)

    pantalla = revision.componer_simple(_imagen(), _imagen(), cfg, estado)

    assert not _color_pintado(pantalla, revision._VERDE_OK)
    assert not _color_pintado(pantalla, revision._ROJO_NO)


# ---------------------------------------------------------------------------
# EL CARTEL DE "SIN MODELO" COMO ESTADO DE LA MEMORIA. Antes mostraba una meta
# --"aprobada 12/50"--; el taller pidio que no haya cupo que llenar, sino que el
# programa aprenda de cada foto. Lo que se informa ahora es que sabe, no cuanto
# falta para un numero inventado.
# ---------------------------------------------------------------------------


def test_sin_modelo_muestra_cuantas_fotos_van(cfg: AppConfig) -> None:
    estado = revision.EstadoRevision(
        congelado=True, conteos={"aprobada": 12, "rechazada": 8}
    )

    titulo, detalle, _, _ = revision.estado_resultado(estado, cfg)

    assert titulo == "SIN MODELO"
    assert "12" in detalle and "8" in detalle


def test_el_cartel_ya_no_muestra_ninguna_meta(cfg: AppConfig) -> None:
    """Pedido del taller: nada de "12/50". Ni con la clave vieja escrita en el config."""
    cfg.interfaz.fotos_por_clase_objetivo = 50
    estado = revision.EstadoRevision(
        congelado=True, conteos={"aprobada": 12, "rechazada": 8}
    )

    _, detalle, _, _ = revision.estado_resultado(estado, cfg)

    assert "/50" not in detalle
    assert "50" not in detalle


def test_sin_fotos_el_cartel_dice_que_hacer(cfg: AppConfig) -> None:
    """Arrancando de cero, un conteo vacio no explica de donde salen esas fotos."""
    estado = revision.EstadoRevision(congelado=True, conteos=None)

    _, detalle, _, _ = revision.estado_resultado(estado, cfg)

    assert "APROBADA" in detalle and "RECHAZADA" in detalle


def test_manda_el_resumen_de_la_memoria_si_lo_hay(cfg: AppConfig) -> None:
    """La memoria sabe algo que el conteo de carpetas no: cuanto acierta."""
    estado = revision.EstadoRevision(
        congelado=True,
        conteos={"aprobada": 12, "rechazada": 8},
        resumen_memoria="memoria: 20 fotos - acierta 17 de 20 que tiene guardadas",
    )

    _, detalle, _, _ = revision.estado_resultado(estado, cfg)

    assert "acierta 17 de 20" in detalle


def test_el_avance_no_pinta_verde_ni_rojo(cfg: AppConfig) -> None:
    """Sigue sin haber veredicto: el color tiene que ser el de espera."""
    estado = revision.EstadoRevision(congelado=True, conteos={"aprobada": 50, "rechazada": 50})

    _, _, color, aprobado = revision.estado_resultado(estado, cfg)

    assert color == revision._AMBAR and aprobado is None


# ---------------------------------------------------------------------------
# EL VEREDICTO QUE SALE DE LA MEMORIA. Mientras no haya modelo entrenado, quien
# opina es la memoria de fotos archivadas: compara contra TODAS las que el taller
# junto, no contra la unica foto de modelo que este elegida. Y cuando no le da
# para opinar, lo dice: un APROBADO salido de un empate es peor que no decir nada.
# ---------------------------------------------------------------------------


def _juicio(**kwargs) -> Juicio:
    datos = dict(
        concepto="aprobada",
        parecido=0.86,
        margen=0.12,
        confianza=0.9,
        fotos_en_memoria=24,
        vecinos=[Vecino(concepto="aprobada", archivo="a.jpg", parecido=0.86,
                        alineado=0.9, bordes=0.82)],
    )
    datos.update(kwargs)
    return Juicio(**datos)


def test_la_memoria_aprueba_y_dice_de_donde_sale(cfg: AppConfig) -> None:
    estado = revision.EstadoRevision(congelado=True, juicio=_juicio())

    titulo, detalle, color, aprobado = revision.estado_resultado(estado, cfg)

    assert titulo == "APROBADO" and aprobado is True and color == revision._VERDE_OK
    assert "aprobada" in detalle and "24 fotos guardadas" in detalle


def test_la_memoria_desaprueba_si_el_concepto_es_de_rechazo(cfg: AppConfig) -> None:
    estado = revision.EstadoRevision(congelado=True, juicio=_juicio(concepto="rechazada"))

    titulo, _, color, aprobado = revision.estado_resultado(estado, cfg)

    assert titulo == "DESAPROBADO" and aprobado is False and color == revision._ROJO_NO


def test_un_concepto_nuevo_del_taller_tambien_decide(cfg: AppConfig) -> None:
    """El taller puede crear 'diente_astillado' desde el panel; si lo marco como defecto,
    la pantalla tiene que pintarse de rojo igual que con 'rechazada'."""
    cfg.clases = ["aprobada", "rechazada", "diente_astillado"]
    cfg.clases_defecto = ["rechazada", "diente_astillado"]
    estado = revision.EstadoRevision(
        congelado=True, juicio=_juicio(concepto="diente_astillado")
    )

    titulo, detalle, _, aprobado = revision.estado_resultado(estado, cfg)

    assert titulo == "DESAPROBADO" and aprobado is False
    assert "diente_astillado" in detalle


def test_si_la_memoria_no_se_anima_no_se_inventa_un_veredicto(cfg: AppConfig) -> None:
    estado = revision.EstadoRevision(
        congelado=True,
        juicio=_juicio(concepto=None, motivo="esta muy parejo: aprobada y rechazada"),
    )

    titulo, detalle, color, aprobado = revision.estado_resultado(estado, cfg)

    assert titulo == "NO ESTOY SEGURO"
    assert aprobado is None and color == revision._AMBAR
    assert "muy parejo" in detalle


def test_sin_vecinos_el_cartel_dice_que_falta_memoria(cfg: AppConfig) -> None:
    estado = revision.EstadoRevision(
        congelado=True,
        juicio=_juicio(concepto=None, vecinos=[], motivo="la memoria todavia no tiene fotos"),
    )

    titulo, _, _, aprobado = revision.estado_resultado(estado, cfg)

    assert titulo == "SIN MEMORIA" and aprobado is None


def test_la_memoria_le_gana_al_parecido_contra_una_sola_foto(cfg: AppConfig) -> None:
    """Treinta fotos archivadas dicen mas que la unica foto de modelo que este elegida."""
    estado = revision.EstadoRevision(
        congelado=True, juicio=_juicio(concepto="rechazada"), parecido=0.95
    )

    titulo, detalle, _, _ = revision.estado_resultado(estado, cfg)

    assert titulo == "DESAPROBADO"
    assert "foto de modelo" not in detalle


def test_el_modelo_entrenado_le_gana_a_la_memoria(cfg: AppConfig) -> None:
    """El orden es: modelo entrenado, memoria, parecido. La pantalla dice cual fue."""
    estado = revision.EstadoRevision(
        congelado=True,
        veredicto=Veredicto(clase="aprobada", clase_id=0, confianza=0.9,
                            probabilidades={"aprobada": 0.9, "rechazada": 0.1}),
        juicio=_juicio(concepto="rechazada"),
    )

    titulo, detalle, _, _ = revision.estado_resultado(estado, cfg)

    assert titulo == "APROBADO"
    assert "modelo entrenado" in detalle


def test_la_segunda_linea_dice_cuanto_acierta_y_donde_mirar(cfg: AppConfig) -> None:
    estado = revision.EstadoRevision(
        congelado=True, juicio=_juicio(confianza=0.83, zona="arriba a la derecha")
    )

    extra = revision.detalle_extra(estado, cfg)

    assert "83%" in extra and "arriba a la derecha" in extra


def test_sin_con_que_medir_la_segunda_linea_lo_dice(cfg: AppConfig) -> None:
    """No se muestra un porcentaje de confianza inventado cuando hay pocas fotos."""
    estado = revision.EstadoRevision(congelado=True, juicio=_juicio(confianza=None, zona=None))

    extra = revision.detalle_extra(estado, cfg)

    assert "pocas fotos" in extra


def test_sin_veredicto_de_la_memoria_no_hay_segunda_linea(cfg: AppConfig) -> None:
    estado = revision.EstadoRevision(congelado=True, parecido=0.8)

    assert revision.detalle_extra(estado, cfg) is None


def test_la_pantalla_entera_se_arma_con_el_veredicto_de_la_memoria(cfg: AppConfig) -> None:
    estado = revision.EstadoRevision(
        congelado=True, juicio=_juicio(confianza=0.8, zona="en el centro")
    )

    pantalla = revision.componer_simple(_imagen(), _imagen(), cfg, estado, 672)

    assert pantalla.imagen.shape[0] == 672
    assert _color_pintado(pantalla, revision._VERDE_OK)


# ---------------------------------------------------------------------------
# EL ANALISIS POR DIENTE. Con dos dientes en la imagen no hay UN veredicto: hay
# uno por diente, cada uno sobre su recuadro, y el cartel grande pasa a ser el
# resumen. Un APROBADO enorme sobre una foto con un diente bueno y uno roto
# seria mentira.
# ---------------------------------------------------------------------------


def _en_pantalla(**kwargs) -> revision.DienteEnPantalla:
    datos = dict(caja=(10, 10, 100, 100), parecido=0.9, aprobado=True, concepto="afilado")
    datos.update(kwargs)
    return revision.DienteEnPantalla(**datos)


def test_con_un_solo_diente_el_cartel_es_el_de_siempre(cfg: AppConfig) -> None:
    estado = revision.EstadoRevision(
        congelado=True, dientes=[_en_pantalla()],
        juicio=Juicio(concepto="aprobada", parecido=0.9, fotos_en_memoria=8,
                      vecinos=[Vecino(concepto="aprobada", archivo="a.jpg", parecido=0.9,
                                      alineado=0.9, bordes=0.9)]),
    )

    titulo, _, _, aprobado = revision.estado_resultado(estado, cfg)

    assert titulo == "APROBADO" and aprobado is True


def test_con_dos_dientes_el_cartel_resume(cfg: AppConfig) -> None:
    estado = revision.EstadoRevision(congelado=True, dientes=[
        _en_pantalla(aprobado=True),
        _en_pantalla(caja=(200, 10, 300, 100), aprobado=False, concepto="roto"),
    ])

    titulo, detalle, color, aprobado = revision.estado_resultado(estado, cfg)

    assert titulo == "2 DIENTES"
    assert "1 bien" in detalle and "1 para rechazar" in detalle
    # Alcanza uno malo para que la sierra no pase: el color manda al rojo.
    assert color == revision._ROJO_NO and aprobado is False


def test_dos_dientes_buenos_pintan_verde(cfg: AppConfig) -> None:
    estado = revision.EstadoRevision(congelado=True, dientes=[
        _en_pantalla(), _en_pantalla(caja=(200, 10, 300, 100)),
    ])

    _, _, color, aprobado = revision.estado_resultado(estado, cfg)

    assert color == revision._VERDE_OK and aprobado is True


def test_un_diente_sin_decidir_no_deja_pasar_la_sierra(cfg: AppConfig) -> None:
    estado = revision.EstadoRevision(congelado=True, dientes=[
        _en_pantalla(), _en_pantalla(caja=(200, 10, 300, 100), aprobado=None, concepto=None),
    ])

    _, detalle, color, aprobado = revision.estado_resultado(estado, cfg)

    assert "1 sin decidir" in detalle
    assert color == revision._AMBAR and aprobado is None


def test_la_segunda_linea_dice_cual_se_guarda(cfg: AppConfig) -> None:
    estado = revision.EstadoRevision(congelado=True, dientes=[
        _en_pantalla(elegido=False),
        _en_pantalla(caja=(200, 10, 300, 100), elegido=True),
    ])

    extra = revision.detalle_extra(estado, cfg)

    assert "diente 2" in extra and "1 2" in extra


def test_el_recuadro_de_cada_diente_se_dibuja(cfg: AppConfig) -> None:
    """Dos cajas chicas sobre la imagen: tienen que aparecer el verde y el rojo."""
    estado = revision.EstadoRevision(congelado=True, dientes=[
        _en_pantalla(caja=(20, 20, 200, 300), aprobado=True),
        _en_pantalla(caja=(380, 20, 600, 300), aprobado=False, concepto="roto"),
    ])

    pantalla = revision.componer_simple(_imagen(), _imagen(), cfg, estado, 672)

    # _color_pintado mira solo la tarjeta del resultado; los recuadros van sobre la foto.
    for color in (revision._VERDE_OK, revision._ROJO_NO):
        pintados = (pantalla.imagen == np.array(color, np.uint8)).all(axis=2).sum()
        assert pintados > 500, f"no se dibujo el recuadro {color}"


def test_un_diente_que_ocupa_toda_la_foto_no_lleva_recuadro(cfg: AppConfig) -> None:
    """Seria el borde del panel: no agrega nada y tapa la imagen."""
    imagen = _imagen()
    alto, ancho = imagen.shape[:2]
    estado = revision.EstadoRevision(
        congelado=True, dientes=[_en_pantalla(caja=(0, 0, ancho, alto))]
    )

    pantalla = revision.componer_simple(imagen, imagen, cfg, estado, 672)

    assert pantalla.imagen.shape[0] == 672


def test_con_dos_dientes_la_derecha_muestra_un_ejemplo_de_cada_uno(cfg: AppConfig) -> None:
    ejemplo_a = np.full((80, 100, 3), 200, np.uint8)
    ejemplo_b = np.full((80, 100, 3), 60, np.uint8)
    estado = revision.EstadoRevision(congelado=True, dientes=[
        _en_pantalla(ejemplo=ejemplo_a, concepto="afilado"),
        _en_pantalla(caja=(380, 20, 600, 300), ejemplo=ejemplo_b, concepto="roto",
                     aprobado=False),
    ])

    pantalla = revision.componer_simple(_imagen(), _imagen(), cfg, estado, 672)

    # Las dos fotos de ejemplo tienen que estar en la mitad derecha del lienzo.
    derecha = pantalla.imagen[:, revision.ANCHO_LIENZO // 2:]
    assert (np.abs(derecha.astype(int) - 200) < 12).any()
    assert (np.abs(derecha.astype(int) - 60) < 12).any()
