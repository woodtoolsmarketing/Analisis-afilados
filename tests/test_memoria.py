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


def _casos_tipicos() -> list[tuple[float, bool]]:
    """Doce decisiones ajustadas que salieron casi todas mal, y doce comodas que salieron
    casi todas bien. Es la forma que tiene de verdad una memoria que anda: cuando los dos
    conceptos quedan a un pelo, se equivoca; cuando uno gana por lejos, acierta."""
    ajustadas = [(0.01 + i * 0.004, i % 6 == 0) for i in range(12)]    # 2 de 12 bien
    comodas = [(0.40 + i * 0.01, i % 6 != 0) for i in range(12)]       # 10 de 12 bien
    return ajustadas + comodas


def test_la_confianza_sale_de_las_decisiones_parecidas_a_esta() -> None:
    auto = M.Autoevaluacion(total=24, aciertos=12, casos=_casos_tipicos())

    ajustada = auto.confianza(0.02)
    comoda = auto.confianza(0.45)

    assert ajustada is not None and comoda is not None
    assert ajustada < 0.5 < comoda


def test_una_decision_comoda_no_arrastra_el_acierto_de_los_empates() -> None:
    """El error que tenia esto: filtraba los casos con margen MENOR O IGUAL al actual, asi
    que para un margen comodo el conjunto era toda la memoria y la confianza caia al
    acierto global. La decision mas clara del dia se rechazaba por culpa de los empates."""
    auto = M.Autoevaluacion(total=24, aciertos=12, casos=_casos_tipicos())

    acierto_global = auto.acierto
    comoda = auto.confianza(0.45)

    assert acierto_global == pytest.approx(0.5)
    assert comoda > acierto_global


def test_un_empate_de_verdad_no_se_escapa_sin_control() -> None:
    """El otro lado del mismo error: para un margen chiquito quedaban dos o tres casos,
    menos que el minimo, devolvia None y el empate pasaba sin que nadie lo controlara."""
    auto = M.Autoevaluacion(total=24, aciertos=12, casos=_casos_tipicos())

    assert auto.confianza(0.001) is not None


def test_sin_casos_suficientes_no_se_inventa_una_confianza() -> None:
    auto = M.Autoevaluacion(total=3, aciertos=2, casos=[(0.1, True), (0.2, True), (0.3, False)])

    assert auto.confianza(0.2) is None


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


# ---------------------------------------------------------------------------
# Lo que encontro la revision adversarial. Cada una de estas pruebas es un error
# que el programa cometia de verdad y que se reprodujo antes de arreglarlo.
# ---------------------------------------------------------------------------


def test_el_porcentaje_que_se_muestra_es_del_concepto_que_se_nombra(tmp_path: Path) -> None:
    """Llego a decir "se parece a X (100%)" cuando ese 100% era de una foto archivada en
    OTRO concepto: el numero salia del mejor vecino de toda la memoria y la pantalla lo
    escribia pegado al nombre del que gano. Es el tipo de numero que ya costo caro aca."""
    base = tmp_path / "coleccion"
    mapa = {
        # Una foto clavada con la consulta, y dos que no tienen nada que ver.
        "aprobada": [_guardar(base / "aprobada", "a0.jpg", _diente(0)),
                     _guardar(base / "aprobada", "a1.jpg", _astillado(1)),
                     _guardar(base / "aprobada", "a2.jpg", _astillado(2))],
        # Tres parecidas entre si: gana el concepto, pero ninguna es LA mas parecida.
        "rechazada": [_guardar(base / "rechazada", f"r{i}.jpg", _diente(i)) for i in (4, 5, 6)],
    }
    mem = _memoria(tmp_path)
    mem.sincronizar(mapa)

    juicio = mem.juzgar(_mas_luz(_diente(0)), minimo_por_concepto=1, confianza_minima=0.0)

    assert juicio.concepto == "rechazada"
    assert juicio.vecinos[0].concepto == "aprobada"      # el mejor de TODA la memoria
    mejor_del_ganador = max(v.parecido for v in juicio.vecinos if v.concepto == "rechazada")
    assert juicio.parecido == pytest.approx(mejor_del_ganador)
    assert juicio.parecido < juicio.vecinos[0].parecido


def test_archivar_la_primera_foto_de_un_concepto_no_da_vuelta_el_veredicto(
    tmp_path: Path,
) -> None:
    """Cada concepto promediaba SUS mejores tres, o menos si no tenia tres, asi que el de
    una sola foto puntuaba por su maximo y el de cuatro por un promedio. Medido: el mismo
    concepto pasaba de 0.9997 con una foto a 0.6328 con tres. Entonces crear un concepto
    nuevo y archivarle la primera foto le daba vuelta el veredicto a la pieza que se
    estaba mirando."""
    base = tmp_path / "coleccion"
    mapa = _taller(tmp_path, buenas=4, malas=4)
    consulta = _mas_luz(_diente(1))
    mem = _memoria(tmp_path)
    mem.sincronizar(mapa)
    antes = mem.juzgar(consulta, minimo_por_concepto=1, confianza_minima=0.0).concepto
    assert antes == "aprobada"

    # El operario crea un concepto nuevo y le archiva UNA foto, que no es la mas parecida.
    mapa["filo_quemado"] = [
        _guardar(base / "filo_quemado", "q0.jpg", _reapoyar_girado(_diente(1)))
    ]
    mem.sincronizar(mapa)

    assert mem.juzgar(consulta, minimo_por_concepto=1, confianza_minima=0.0).concepto == antes


def _reapoyar_girado(imagen: np.ndarray, grados: float = 10.0) -> np.ndarray:
    alto, ancho = imagen.shape[:2]
    M_ = cv2.getRotationMatrix2D((ancho / 2, alto / 2), grados, 1.0)
    return cv2.warpAffine(imagen, M_, (ancho, alto), borderMode=cv2.BORDER_REPLICATE)


def test_una_foto_de_un_solo_color_no_se_parece_a_nada() -> None:
    """Al normalizarla queda en ceros, y matchTemplate con una plantilla sin varianza
    devuelve el maximo contra CUALQUIER imagen. Una foto negra archivada le daba a su
    concepto un piso de 0.500 contra todo lo que entrara."""
    _, _, plana = M.rasgos(np.full((240, 240, 3), 17, np.uint8))
    _, _, diente = M.rasgos(_diente())

    assert M.alineado(diente, plana) == 0.0
    assert M.alineado(plana, plana) == 0.0


def test_una_foto_negra_archivada_no_se_roba_el_veredicto(tmp_path: Path) -> None:
    base = tmp_path / "coleccion"
    mapa = _taller(tmp_path, buenas=5, malas=5)
    mapa["quemado"] = [_guardar(base / "quemado", "negra.jpg",
                                np.zeros((240, 240, 3), np.uint8))]
    mem = _memoria(tmp_path)
    mem.sincronizar(mapa)

    for imagen, esperado in ((_diente(2), "aprobada"), (_astillado(2), "rechazada")):
        juicio = mem.juzgar(_mas_luz(imagen), minimo_por_concepto=1, confianza_minima=0.0)
        assert juicio.concepto == esperado


def test_el_recorte_de_la_autoevaluacion_mide_las_mas_nuevas_de_verdad(
    tmp_path: Path,
) -> None:
    """Las filas vienen agrupadas por concepto, asi que cortar el final de la lista se
    quedaba con las fotos mas VIEJAS del ULTIMO concepto: el acierto que la pantalla
    presenta como el de toda la memoria podia salir de un solo concepto."""
    import os

    base = tmp_path / "coleccion"
    mapa: dict[str, list[Path]] = {"aprobada": [], "rechazada": []}
    # Se alternan en el tiempo: las 6 mas nuevas son 3 de cada concepto.
    for i in range(6):
        for concepto, imagen in (("aprobada", _diente(i)), ("rechazada", _astillado(i))):
            ruta = _guardar(base / concepto, f"{i}.jpg", imagen)
            os.utime(ruta, (1_700_000_000 + i * 60, 1_700_000_000 + i * 60))
            mapa[concepto].append(ruta)

    mem = _memoria(tmp_path, maximo_autoevaluacion=6)
    mem.sincronizar(mapa)
    auto = mem.autoevaluacion()

    assert auto.recortada is True
    assert auto.total == 6
    # Las dos clases tienen que estar representadas: si midiera el final de la lista,
    # las 6 serian todas de 'rechazada'.
    assert set(auto.por_concepto) == {"aprobada", "rechazada"}


def test_la_cache_converge_aunque_dos_conceptos_tengan_fotos_con_el_mismo_nombre(
    tmp_path: Path,
) -> None:
    """Sin la carpeta en la clave, cada arranque informaba una foto nueva y una que ya no
    estaba, y volvia a analizar las dos para siempre."""
    base = tmp_path / "coleccion"
    mapa = {
        "aprobada": [_guardar(base / "aprobada", "pieza.jpg", _diente(1))],
        "rechazada": [_guardar(base / "rechazada", "pieza.jpg", _astillado(1))],
    }
    _memoria(tmp_path).sincronizar(mapa)

    otra = _memoria(tmp_path)

    assert otra.sincronizar(mapa) == (0, 0)


def test_una_cache_cortada_a_la_mitad_no_deja_al_programa_sin_arrancar(
    tmp_path: Path,
) -> None:
    """Un corte de luz en la mitad de la escritura. La cache es un atajo, no un dato."""
    mapa = _taller(tmp_path, buenas=2, malas=2)
    mem = _memoria(tmp_path)
    mem.sincronizar(mapa)
    datos = mem.archivo_cache.read_bytes()
    mem.archivo_cache.write_bytes(datos[: len(datos) // 2])

    otra = _memoria(tmp_path)

    assert otra.sincronizar(mapa) == (4, 0)


# ---------------------------------------------------------------------------
# Quien puede ser un veredicto, y con que regla se mide la confianza
#
# EL DIA QUE ESTO SALIO MAL: 16/11/2026. El programa dijo "NO ESTOY SEGURO" en TODOS los
# dientes de la jornada. La coleccion tenia 15 fotos en 4 conceptos, y dos de esos
# conceptos tenian 2 fotos de las 3 que el propio programa exige para animarse a decidir.
# Esos conceptos flacos competian igual: podian ganar la votacion, y sus fotos entraban en
# la autoevaluacion que decide si confiar en el veredicto. Diente_sin_placa, con 2 fotos,
# acertaba 0 de 2, y esos dos errores dejaban la confianza en 62% --abajo del minimo de
# 70%-- para CUALQUIER margen por debajo de 11 puntos.
# ---------------------------------------------------------------------------


def _mancha(semilla: int = 0, lado: int = 240) -> np.ndarray:
    """Una tercera familia, bien distinta de las otras dos: un circulo claro al medio."""
    img = np.full((lado, lado, 3), 30, np.uint8)
    cv2.circle(img, (lado // 2, lado // 2), 70 + semilla * 6, (220, 220, 220), -1)
    cv2.circle(img, (lado // 2, lado // 2), 30, (60, 60, 60), -1)
    return cv2.GaussianBlur(img, (7, 7), 0)


def test_un_concepto_al_que_le_faltan_fotos_no_puede_ser_el_veredicto(
    tmp_path: Path,
) -> None:
    """Y TAMPOCO SE CONTESTA OTRA COSA EN SU LUGAR: se dice a que se parece y que falta.

    El programa ya exigia 3 fotos por concepto para animarse a juzgar, pero despues dejaba
    competir a uno de 2: podia ganar y ser el veredicto. O sea que el mismo numero servia
    para declarar un concepto "todavia no esta listo" y para dejarlo decidir.
    Contestar con el segundo seria peor todavia: el operario veria el nombre de otra cosa.
    """
    mapa = _taller(tmp_path, buenas=4, malas=4)
    base = tmp_path / "coleccion"
    mapa["nueva"] = [_guardar(base / "nueva", f"n{i}.jpg", _mancha(i)) for i in range(2)]
    mem = _memoria(tmp_path)
    mem.sincronizar(mapa)

    juicio = mem.juzgar(_mancha(5), minimo_por_concepto=3)

    assert juicio.concepto is None, "nombro un concepto de 2 fotos"
    assert juicio.falta_fotos == "nueva"
    # El cartel tiene que decir cuantas tiene y cuantas hacen falta: es lo unico que el
    # operario puede arreglar, y se arregla sacando una foto.
    assert "2 de las 3" in str(juicio.motivo)
    assert "nueva" in str(juicio.motivo)


def test_la_autoevaluacion_mide_con_la_misma_regla_con_la_que_se_decide(
    tmp_path: Path,
) -> None:
    """SI UN CONCEPTO NO PUEDE SER VEREDICTO, SUS FOTOS NO MIDEN DECISIONES QUE EL PROGRAMA
    VAYA A TOMAR.

    Es lo que dejaba la confianza en 62% el 16/11: las 2 fotos de un concepto de 2 fotos
    entraban como errores de decisiones que, con la regla puesta, el programa ni siquiera
    toma --dice que faltan fotos--. Lo que NO se puede hacer es saltear todo lo del
    concepto flaco: una foto suya sobre la que el programa SI contesta otro nombre es un
    error de verdad y se cuenta, porque ese diente va a llegar al microscopio igual.
    """
    mapa = _taller(tmp_path, buenas=4, malas=4)
    base = tmp_path / "coleccion"
    mapa["nueva"] = [_guardar(base / "nueva", f"n{i}.jpg", _mancha(i)) for i in range(2)]
    mem = _memoria(tmp_path)
    mem.sincronizar(mapa)

    auto = mem.autoevaluacion(minimo_por_concepto=3)

    # Ningun caso medido puede tener como respuesta correcta un concepto que no puede ser
    # veredicto: de esos, o el programa no decide, o decide mal y cuenta como error.
    assert "nueva" not in auto.por_concepto
    # Y las que no se midieron no desaparecen: quedan contadas como lo que son.
    assert auto.sin_decidir >= 1
    assert auto.total + auto.sin_decidir <= mem.cantidad
    # La regla con la que se midio queda escrita, para poder tirar la medicion si cambia.
    assert auto.minimo == 3


def test_una_medicion_hecha_con_otra_regla_no_se_reusa(tmp_path: Path) -> None:
    """La autoevaluacion se guarda en el .npz y vuelve a cargarse al abrir el programa.

    Sin esto, el taller abriria el programa con la medicion vieja --hecha cuando los
    conceptos flacos votaban-- y seguiria frenado hasta que alguien archive una foto nueva.
    """
    mapa = _taller(tmp_path, buenas=4, malas=4)
    base = tmp_path / "coleccion"
    mapa["nueva"] = [_guardar(base / "nueva", f"n{i}.jpg", _mancha(i)) for i in range(2)]
    mem = _memoria(tmp_path)
    mem.sincronizar(mapa)

    con_tres = mem.autoevaluacion(minimo_por_concepto=3)
    con_uno = mem.autoevaluacion(minimo_por_concepto=1)

    assert con_tres.minimo == 3 and con_uno.minimo == 1
    # Con minimo 1 vuelve a entrar el concepto flaco: es otra medicion, no la guardada.
    assert "nueva" in con_uno.por_concepto
    assert con_uno.sin_decidir == 0


def test_con_pocas_fotos_la_confianza_dice_el_acierto_global_y_no_finge_una_banda() -> None:
    """EL NUMERO QUE FRENO EL TALLER ENTERO.

    La ventana de "decisiones asi de ajustadas" es el 25% de los casos con piso de 8. Con
    15 casos el piso manda y la ventana son 8, el 53% de la memoria: los margenes que
    entran van de 1,8 a 17,9 puntos. Eso no es una banda alrededor de este margen, es casi
    toda la memoria con otro nombre, y daba el MISMO 62% para cualquier margen por debajo
    de 11 puntos. El programa no dudaba de ese diente: dudaba de todos.
    """
    # 15 casos: 12 bien, 3 mal, con los errores en margenes chicos --como la coleccion de
    # verdad del taller el 16/11--.
    casos = [(0.018, False), (0.038, True), (0.046, True), (0.163, False),
             (0.181, False), (0.189, True), (0.239, True), (0.245, True),
             (0.251, True), (0.268, True), (0.281, True), (0.317, True),
             (0.332, True), (0.350, True), (0.360, True)]
    auto = M.Autoevaluacion(total=15, aciertos=12, casos=casos)

    assert auto.mide_por_banda() is False
    # Todos los margenes dan lo mismo, porque es el acierto global: no hay banda.
    assert auto.confianza(0.0) == pytest.approx(12 / 15)
    assert auto.confianza(0.30) == pytest.approx(12 / 15)

    # Con 16 casos la ventana de 8 ya es la mitad justa y vuelve a ser una banda.
    con_mas = M.Autoevaluacion(total=16, aciertos=13, casos=casos + [(0.37, True)])
    assert con_mas.mide_por_banda() is True
    assert con_mas.confianza(0.0) != con_mas.confianza(0.30)
