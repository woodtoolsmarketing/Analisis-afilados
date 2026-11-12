"""Pruebas de la memoria: aprender de cada foto archivada, sin entrenar nada.

LO QUE MAS SE CUIDA ACA
-----------------------
Que la memoria NO MIENTA sobre si misma. Es la pieza que le dice al operario cuanta
precision tiene el programa, y ese numero sale de esconder cada foto guardada y
preguntarle a las demas. Si esa cuenta estuviera mal --si una foto pudiera votarse a si
misma-- el acierto daria casi 100% siempre y el taller confiaria en un numero vacio. Por
eso la primera prueba es la que TIENE que salir mal: con etiquetas al azar, el acierto
tiene que derrumbarse.

Lo segundo es que no haya cupo: el taller pidio que aprenda con cada foto que se le carga,
sin un limite que llenar.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from afilado import memoria as M


# ---------------------------------------------------------------------------
# Imagenes de prueba CON ESTRUCTURA: el ruido puro no se parece en nada a lo que
# entrega un microscopio, y las medidas sobre ruido no dicen nada del taller.
# ---------------------------------------------------------------------------


def _diente(semilla: int = 0, lado: int = 240) -> np.ndarray:
    img = np.full((lado, lado, 3), 60, np.uint8)
    cv2.fillPoly(img, [np.array([[10, lado - 20], [10, 120 + semilla * 4],
                                 [lado - 60, 40], [lado - 20, 70],
                                 [lado - 20, lado - 20]])], (190, 190, 190))
    cv2.line(img, (10, 120 + semilla * 4), (lado - 60, 40), (235, 235, 235), 6)
    for i in range(0, lado, 7):
        cv2.line(img, (i, 0), (i + 40 + semilla * 9, lado), (0, 0, 0), 1)
    return cv2.GaussianBlur(img, (5, 5), 0)


def _astillado(semilla: int = 0, lado: int = 240) -> np.ndarray:
    """Otra familia de imagenes, bien distinta: el filo arriba y el metal abajo."""
    img = np.full((lado, lado, 3), 200, np.uint8)
    cv2.fillPoly(img, [np.array([[0, 0], [lado, 0], [lado, 90 + semilla * 5],
                                 [0, 60 + semilla * 5]])], (40, 40, 40))
    for i in range(0, lado, 11):
        cv2.circle(img, (i, 150 + (i * semilla) % 40), 6, (90, 90, 90), -1)
    return cv2.GaussianBlur(img, (5, 5), 0)


def _mas_luz(imagen: np.ndarray, factor: float = 1.25, suma: float = 18.0) -> np.ndarray:
    return np.clip(imagen.astype(np.float32) * factor + suma, 0, 255).astype(np.uint8)


def _corrida(imagen: np.ndarray, dx: int = 14, dy: int = -9) -> np.ndarray:
    M_ = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(imagen, M_, imagen.shape[1::-1], borderMode=cv2.BORDER_REPLICATE)


def _guardar(carpeta: Path, nombre: str, imagen: np.ndarray) -> Path:
    carpeta.mkdir(parents=True, exist_ok=True)
    ruta = carpeta / nombre
    assert cv2.imwrite(str(ruta), imagen)
    return ruta


def _memoria(tmp_path: Path, **kwargs) -> M.Memoria:
    return M.Memoria(str(tmp_path / "memoria"), **kwargs)


def _taller(tmp_path: Path, buenas: int = 4, malas: int = 4) -> dict[str, list[Path]]:
    """Un taller chico pero coherente: dos familias de imagenes bien distintas."""
    base = tmp_path / "coleccion"
    mapa = {"aprobada": [], "rechazada": []}
    for i in range(buenas):
        mapa["aprobada"].append(_guardar(base / "aprobada", f"a{i}.jpg", _diente(i)))
    for i in range(malas):
        mapa["rechazada"].append(_guardar(base / "rechazada", f"r{i}.jpg", _astillado(i)))
    return mapa


# ---------------------------------------------------------------------------
# Lo que mide el parecido
# ---------------------------------------------------------------------------


def test_la_luz_no_cambia_el_parecido() -> None:
    """La lampara del microscopio no es estable; el filo no cambia porque cambie la luz."""
    _, _, forma = M.rasgos(_diente())
    _, _, con_luz = M.rasgos(_mas_luz(_diente()))

    assert M.alineado(forma, con_luz) > 0.95


def test_correr_la_pieza_no_derrumba_el_parecido() -> None:
    """La pieza se apoya a mano y nunca cae en el mismo lugar."""
    _, _, forma = M.rasgos(_diente())
    _, _, movida = M.rasgos(_corrida(_diente()))

    assert M.alineado(forma, movida) > 0.90


def test_dos_piezas_distintas_dan_menos_que_la_misma() -> None:
    _, _, una = M.rasgos(_diente())
    _, _, igual = M.rasgos(_mas_luz(_diente()))
    _, _, otra = M.rasgos(_astillado())

    assert M.alineado(una, otra) < M.alineado(una, igual)


def test_la_zona_distinta_dice_donde_mirar() -> None:
    """Un porcentaje dice cuanto se parece; esto dice en que parte no."""
    base = _diente()
    manchada = base.copy()
    cv2.circle(manchada, (210, 30), 18, (0, 0, 0), -1)
    _, _, a = M.rasgos(base)
    _, _, b = M.rasgos(manchada)

    zona = M.zona_distinta(b, a) or ""
    assert "arriba" in zona and "derecha" in zona


def test_una_zona_que_no_se_destaca_no_se_nombra() -> None:
    """Siempre hay una zona que es la peor. Nombrarla cuando todas se parecen seria
    mandar al operario a mirar al azar; medido, dos fotos del mismo diente con otra
    exposicion reparten la diferencia pareja."""
    _, _, a = M.rasgos(_diente())
    _, _, igual = M.rasgos(_diente())
    _, _, con_luz = M.rasgos(_mas_luz(_diente(), factor=1.1, suma=6))

    assert M.zona_distinta(a, igual) is None
    assert M.zona_distinta(a, con_luz) is None


# ---------------------------------------------------------------------------
# Aprender de cada foto, sin cupo
# ---------------------------------------------------------------------------


def test_cada_foto_nueva_entra_a_la_memoria_al_instante(tmp_path: Path) -> None:
    mapa = _taller(tmp_path, buenas=2, malas=2)
    mem = _memoria(tmp_path)
    mem.sincronizar(mapa)
    assert mem.cantidad == 4

    mapa["aprobada"].append(
        _guardar(tmp_path / "coleccion" / "aprobada", "nueva.jpg", _diente(7))
    )
    agregadas, sacadas = mem.sincronizar(mapa)

    assert (agregadas, sacadas) == (1, 0)
    assert mem.cantidad == 5


def test_no_hay_tope_de_fotos(tmp_path: Path) -> None:
    """El taller pidio justamente esto: que no haya un cupo que llenar."""
    mapa = _taller(tmp_path, buenas=30, malas=30)
    mem = _memoria(tmp_path)

    mem.sincronizar(mapa)

    assert mem.cantidad == 60
    assert mem.conteo_por_concepto() == {"aprobada": 30, "rechazada": 30}


def test_una_foto_borrada_sale_de_la_memoria(tmp_path: Path) -> None:
    mapa = _taller(tmp_path, buenas=3, malas=3)
    mem = _memoria(tmp_path)
    mem.sincronizar(mapa)

    mapa["aprobada"].pop()
    agregadas, sacadas = mem.sincronizar(mapa)

    assert (agregadas, sacadas) == (0, 1)
    assert mem.cantidad == 5


# ---------------------------------------------------------------------------
# El veredicto
# ---------------------------------------------------------------------------


def test_reconoce_la_familia_de_la_pieza(tmp_path: Path) -> None:
    mem = _memoria(tmp_path)
    mem.sincronizar(_taller(tmp_path))

    juicio = mem.juzgar(_mas_luz(_astillado(1)), minimo_por_concepto=2)

    assert juicio.concepto == "rechazada"
    assert juicio.parecido > 0.8
    assert juicio.vecinos[0].concepto == "rechazada"


def test_con_un_solo_concepto_no_opina(tmp_path: Path) -> None:
    """Con una sola bolsa, todo cae en esa bolsa: no es un veredicto, es una tautologia."""
    mapa = _taller(tmp_path)
    mem = _memoria(tmp_path)
    mem.sincronizar({"aprobada": mapa["aprobada"]})

    juicio = mem.juzgar(_diente(), minimo_por_concepto=2)

    assert juicio.concepto is None
    assert "dos conceptos" in (juicio.motivo or "")


def test_sin_fotos_no_opina(tmp_path: Path) -> None:
    mem = _memoria(tmp_path)

    juicio = mem.juzgar(_diente())

    assert juicio.concepto is None and juicio.motivo


def test_lo_que_no_se_parece_a_nada_no_se_fuerza(tmp_path: Path) -> None:
    """La foto del cartel de DroidCam entro al dataset una vez. Esto es para que no vuelva
    a pasar en el veredicto: si no se parece a nada guardado, se dice."""
    mem = _memoria(tmp_path)
    mem.sincronizar(_taller(tmp_path))

    juicio = mem.juzgar(np.full((240, 240, 3), 20, np.uint8), minimo_por_concepto=2)

    assert juicio.concepto is None
    assert "no se parece" in (juicio.motivo or "")


def test_el_veredicto_no_depende_de_cuantas_fotos_tenga_cada_concepto(tmp_path: Path) -> None:
    """Con 20 buenas y 3 malas, el concepto grande no puede ganar por mayoria: cada uno
    compite con sus mejores vecinos, no con su cantidad."""
    base = tmp_path / "coleccion"
    mapa = {
        "aprobada": [_guardar(base / "aprobada", f"a{i}.jpg", _diente(i)) for i in range(20)],
        "rechazada": [_guardar(base / "rechazada", f"r{i}.jpg", _astillado(i)) for i in range(3)],
    }
    mem = _memoria(tmp_path)
    mem.sincronizar(mapa)

    juicio = mem.juzgar(_astillado(1), minimo_por_concepto=2)

    assert juicio.concepto == "rechazada"


# ---------------------------------------------------------------------------
# LA PRUEBA QUE TIENE QUE SALIR MAL: la memoria no puede mentir sobre si misma
# ---------------------------------------------------------------------------


def test_con_etiquetas_al_azar_el_acierto_se_derrumba(tmp_path: Path) -> None:
    """Dientes y astillados repartidos a ciegas entre dos conceptos: no hay nada que
    aprender, y el numero que muestra la pantalla tiene que decirlo."""
    base = tmp_path / "coleccion"
    todas = [_diente(i) for i in range(6)] + [_astillado(i) for i in range(6)]
    mapa = {"uno": [], "otro": []}
    for i, imagen in enumerate(todas):
        concepto = "uno" if i % 2 == 0 else "otro"   # mezcla las dos familias a proposito
        mapa[concepto].append(_guardar(base / concepto, f"{i}.jpg", imagen))

    mem = _memoria(tmp_path)
    mem.sincronizar(mapa)
    auto = mem.autoevaluacion()

    assert auto.total == 12
    assert auto.acierto is not None and auto.acierto < 0.75


def test_con_conceptos_de_verdad_el_acierto_es_alto(tmp_path: Path) -> None:
    """El otro lado del mismo control: si los conceptos separan, el numero sube."""
    mem = _memoria(tmp_path)
    mem.sincronizar(_taller(tmp_path, buenas=5, malas=5))

    auto = mem.autoevaluacion()

    assert auto.total == 10
    assert auto.acierto == 1.0
    assert auto.por_concepto["aprobada"] == (5, 5)


def test_ninguna_foto_se_vota_a_si_misma(tmp_path: Path) -> None:
    """Si pudiera, el acierto daria 100% siempre: cada foto es identica a si misma."""
    base = tmp_path / "coleccion"
    # Una foto archivada en el concepto EQUIVOCADO: un diente guardado como rechazada.
    mapa = {
        "aprobada": [_guardar(base / "aprobada", f"a{i}.jpg", _diente(i)) for i in range(4)],
        "rechazada": [_guardar(base / "rechazada", f"r{i}.jpg", _astillado(i)) for i in range(3)]
        + [_guardar(base / "rechazada", "colada.jpg", _diente(9))],
    }
    mem = _memoria(tmp_path)
    mem.sincronizar(mapa)

    auto = mem.autoevaluacion()

    assert auto.aciertos == auto.total - 1
    assert auto.por_concepto["rechazada"] == (3, 4)


def test_la_confianza_sale_de_los_casos_igual_de_ajustados() -> None:
    auto = M.Autoevaluacion(
        total=10,
        aciertos=5,
        casos=[(0.01, False), (0.02, False), (0.03, True), (0.04, False),
               (0.05, True), (0.20, True), (0.25, True), (0.30, True),
               (0.35, True), (0.40, False)],
    )

    # Un margen comodo mira todos los casos: 6 de 10.
    assert auto.confianza(1.0) == pytest.approx(0.6)
    # Uno ajustado mira solo los ajustados, y ahi el panorama es peor.
    assert auto.confianza(0.05, minimo_casos=5) == pytest.approx(0.4)
    # Sin casos suficientes no se inventa un numero.
    assert auto.confianza(0.02, minimo_casos=5) is None


def test_la_autoevaluacion_dice_si_se_midio_sobre_una_parte(tmp_path: Path) -> None:
    """Recortar en silencio seria mentir con un numero que parece el de toda la memoria."""
    mem = _memoria(tmp_path, maximo_autoevaluacion=6)
    mem.sincronizar(_taller(tmp_path, buenas=5, malas=5))

    auto = mem.autoevaluacion()

    assert auto.recortada is True
    assert auto.total == 6
    assert "ultimas" in mem.resumen()


# ---------------------------------------------------------------------------
# La cache: abrir el programa con cientos de fotos no puede costar leerlas todas
# ---------------------------------------------------------------------------


def test_el_segundo_arranque_no_vuelve_a_leer_las_fotos(tmp_path: Path) -> None:
    mapa = _taller(tmp_path)
    _memoria(tmp_path).sincronizar(mapa)

    otra = _memoria(tmp_path)
    agregadas, sacadas = otra.sincronizar(mapa)

    assert (agregadas, sacadas) == (0, 0)
    assert otra.cantidad == 8


def test_una_foto_cambiada_se_vuelve_a_analizar(tmp_path: Path) -> None:
    mapa = _taller(tmp_path, buenas=2, malas=2)
    _memoria(tmp_path).sincronizar(mapa)

    ruta = mapa["aprobada"][0]
    cv2.imwrite(str(ruta), _astillado(3))

    otra = _memoria(tmp_path)
    agregadas, _ = otra.sincronizar(mapa)

    assert agregadas == 1


def test_una_cache_rota_no_rompe_el_programa(tmp_path: Path) -> None:
    mapa = _taller(tmp_path, buenas=2, malas=2)
    mem = _memoria(tmp_path)
    mem.sincronizar(mapa)
    mem.archivo_cache.write_bytes(b"esto no es un npz")

    otra = _memoria(tmp_path)
    agregadas, _ = otra.sincronizar(mapa)

    assert agregadas == 4
    assert otra.cantidad == 4


def test_la_misma_foto_en_otro_concepto_se_reetiqueta(tmp_path: Path) -> None:
    """El operario mueve una foto de carpeta con el Explorador: la memoria tiene que
    seguirla, no quedarse con la etiqueta vieja."""
    base = tmp_path / "coleccion"
    mapa = _taller(tmp_path, buenas=3, malas=3)
    mem = _memoria(tmp_path)
    mem.sincronizar(mapa)

    movida = mapa["aprobada"].pop()
    destino = base / "rechazada" / movida.name
    movida.replace(destino)
    mapa["rechazada"].append(destino)
    mem.sincronizar(mapa)

    assert mem.conteo_por_concepto() == {"aprobada": 2, "rechazada": 4}


def test_una_foto_ilegible_no_frena_el_arranque(tmp_path: Path) -> None:
    mapa = _taller(tmp_path, buenas=2, malas=2)
    rota = tmp_path / "coleccion" / "aprobada" / "rota.jpg"
    rota.write_bytes(b"no soy una imagen")
    mapa["aprobada"].append(rota)

    mem = _memoria(tmp_path)
    mem.sincronizar(mapa)

    assert mem.cantidad == 4


def test_el_resumen_cuenta_lo_que_sabe(tmp_path: Path) -> None:
    mem = _memoria(tmp_path)
    mem.sincronizar(_taller(tmp_path, buenas=3, malas=3))

    resumen = mem.resumen()

    assert "6 fotos" in resumen and "aprobada 3" in resumen and "acierta" in resumen


def test_sin_fotos_el_resumen_invita_a_sacarlas(tmp_path: Path) -> None:
    assert "vacia" in _memoria(tmp_path).resumen()
