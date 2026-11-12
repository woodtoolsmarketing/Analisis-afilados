"""La MEMORIA: el programa aprende de cada foto que se le carga, sin entrenar nada.

QUE PROBLEMA RESUELVE
---------------------
Entrenar una red pide juntar cientos de fotos, parar, correr el entrenamiento en otra PC y
recien ahi tener un modelo. Hasta ese dia el programa no sabe nada, y cada foto que el
operario guarda no sirve para nada HOY. La memoria da vuelta eso: cada foto que entra se
suma al instante a lo que el programa compara, y el veredicto de la foto siguiente ya la
tiene en cuenta. No hay cupo que llenar ni boton de entrenar.

No reemplaza al modelo entrenado --cuando haya uno, manda el modelo-- pero cubre todo el
tiempo en que no lo hay, que hasta ahora era tiempo perdido.

COMO DECIDE
-----------
Se le pregunta a las fotos guardadas: cuales se parecen mas a la que estoy mirando, y en
que concepto estan archivadas esas. Es comparacion de fotos, no una red; eso tiene dos
consecuencias que conviene tener claras: la pantalla puede decir A QUE CONCEPTO se parecio
y cuanto --el detalle de cada vecino queda en Juicio.vecinos, para quien lo necesite-- y no
inventa nada que no este guardado.

EN DOS ETAPAS, PARA QUE PUEDA CRECER SIN LIMITE
-----------------------------------------------
Comparar bien dos fotos cuesta 0.47 ms, que con mil fotos en memoria serian 470 ms por
veredicto. Por eso primero se descarta barato y despues se mide caro:

    etapa barata  un producto escalar por foto: 0.1 microsegundos cada una. Mil fotos
                  son 0.1 ms. Sirve para quedarse con las candidatas de cada concepto.
    etapa exacta  matchTemplate, que ALINEA las dos fotos antes de compararlas, sobre las
                  12 candidatas de cada concepto. Con cinco conceptos son 60 comparaciones,
                  28 ms, cueste lo que cueste la memoria entera.

Lo que se pierde por descartar barato, medido sobre las 19 fotos de microscopio del taller
(todas en una bolsa, que es el caso mas exigente): de los tres vecinos que habria elegido la
comparacion exacta, con 6 candidatas sobreviven el 84.2% y con 12 el 98.2%. Por eso van 12:
el veredicto se calcula una vez por pieza, no en cada cuadro, asi que los 5 ms de diferencia
no se notan y perder uno de cada seis vecinos si. Ese 98.2% sale de 19 fotos; cuando la
memoria tenga cientos hay que volver a medirlo, y por eso es configurable.

QUE MIDE EL PARECIDO (y por que estas dos cosas y no una)
----------------------------------------------------------
    alineado  correlacion de las dos imagenes en gris, normalizadas, quedandose con el
              mejor calce. Saca de la cuenta la luz --que cambia entre foto y foto-- y el
              encuadre --la pieza se apoya a mano y nunca cae igual--.
    bordes    hacia donde apuntan los bordes en cada uno de 16 bloques. Es la geometria
              del filo, no su brillo.

Las dos son ciegas a la luz: medido sobre tres fotos del MISMO diente con exposiciones
distintas, las dos dan 0.999-1.000. Y no dicen lo mismo: entre si correlacionan 0.63, o
sea que cada una ve algo que la otra no. Se promedian mitad y mitad porque no hay con que
justificar inclinar la balanza: ninguna medicion de este taller dice todavia cual de las
dos separa mejor un filo sano de uno gastado.

LO QUE NO HACE: CORREGIR ROTACION
---------------------------------
Si la pieza se apoya girada, el parecido baja aunque sea el mismo diente. Buscar tambien el
angulo multiplicaria el costo por la cantidad de angulos probados. Esta medido sobre las 19
fotos del taller, volviendo a juzgar cada una despues de perturbarla como pasa de verdad al
reapoyar la pieza (el veredicto se mantiene en):

    quieta (control)      19 de 19      mas luz (+25%)        19 de 19
    corrida 10 px         19 de 19      menos luz (-25%)      19 de 19
    corrida 25 px         19 de 19      girada 3 grados       17 de 19
    corrida 50 px         19 de 19      girada 7 grados       16 de 19
                                        girada 15 grados      10 de 19

    reapoyada de verdad (14 px + 4 grados + 12% de luz):      17 de 19

O sea: correrla y la luz NO mueven el veredicto; girarla SI. La recomendacion practica para
el taller --apoyar la pieza siempre con la misma orientacion-- sale de aca, y no de una
suposicion. Si algun dia hace falta, el lugar donde agregar busqueda de angulo es
alineado(), y tests/test_estabilidad.py es la prueba que tendria que cambiar.

LA MEMORIA SE MIDE A SI MISMA
-----------------------------
autoevaluacion() esconde cada foto guardada y le pregunta a las demas en que concepto va.
Eso da un porcentaje de acierto sobre las fotos del taller, no sobre un folleto, y se
actualiza con cada foto nueva. Es el unico numero honesto sobre "cuanta precision tiene
esto", y de ahi sale tambien la confianza: de las veces que la memoria estuvo igual de
ajustada que ahora, cuantas acerto.

LO QUE ESE NUMERO TIENE DE OPTIMISTA, y conviene saberlo: si la misma pieza quedo
fotografiada dos veces en el mismo concepto, al esconder una la otra sigue ahi y la ubica
sin esfuerzo. O sea que el acierto sube cuando se sacan varias tomas de la misma pieza, y
sube sin que el programa haya aprendido nada nuevo. Para que el numero signifique lo que
parece, las fotos de un concepto tienen que ser de PIEZAS distintas.
"""

from __future__ import annotations

import json
import logging
import os
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from .config import ruta_absoluta

_log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Medidas del descriptor. Cambiar cualquiera invalida la cache: va en _VERSION.
# ---------------------------------------------------------------------------

# A 128 px el parecido da lo mismo que a 256 --medido sobre los 171 pares de las fotos de
# microscopio: diferencia media 0.011, maxima 0.052, concordancia de orden 0.997-- y cuesta
# 4.7 veces menos (0.47 ms contra 2.23 ms). Ademas la copia chica que la memoria guarda de
# cada foto pasa de 64 KB a 16 KB: mil fotos entran en 16 MB de RAM.
LADO_FORMA = 128
LADO_MINI = 32
_BLOQUES = 4
_ANGULOS = 9
_RECORTE = 5  # la plantilla es el centro: se recorta 1/5 por lado

DIM_MINI = LADO_MINI * LADO_MINI
DIM_BORDES = _BLOQUES * _BLOQUES * _ANGULOS

# Sube cuando cambia como se calcula o se guarda algo: una cache de otra version se
# descarta y se rehace sola. La 2 cambio la clave de cada foto para incluir su carpeta.
_VERSION = 2
_ARCHIVO = "memoria.npz"

_SIN_FOTOS = "la memoria todavia no tiene fotos"


@dataclass
class Vecino:
    """Una foto guardada que se parecio a la que se esta mirando."""

    concepto: str
    archivo: str
    parecido: float      # 0..1, el promedio de las dos medidas
    alineado: float
    bordes: float


@dataclass
class Juicio:
    """Lo que la memoria tiene para decir de una foto."""

    concepto: Optional[str] = None          # None = no se anima a decidir
    parecido: float = 0.0                   # con el vecino mas parecido
    margen: float = 0.0                     # cuanto le saca el 1o al 2o concepto
    confianza: Optional[float] = None       # medida, de la autoevaluacion. None = sin medir
    motivo: Optional[str] = None            # por que no se anima, en castellano
    vecinos: list[Vecino] = field(default_factory=list)
    puntajes: dict[str, float] = field(default_factory=dict)
    zona: Optional[str] = None              # donde mas se diferencia del mejor vecino
    fotos_en_memoria: int = 0

    @property
    def decidio(self) -> bool:
        return self.concepto is not None


@dataclass
class Autoevaluacion:
    """Cuanto acierta la memoria sobre sus PROPIAS fotos, escondiendolas de a una."""

    total: int = 0
    aciertos: int = 0
    por_concepto: dict[str, tuple[int, int]] = field(default_factory=dict)
    # (margen, si acerto) de cada foto probada: con esto se calibra la confianza.
    casos: list[tuple[float, bool]] = field(default_factory=list)
    medidas_de: int = 0   # cuantas fotos habia cuando se midio
    recortada: bool = False  # True si se midio sobre las mas nuevas, no sobre todas

    @property
    def acierto(self) -> Optional[float]:
        return self.aciertos / self.total if self.total else None

    def confianza(self, margen: float, minimo_casos: int = 8) -> Optional[float]:
        """Que tan seguido acerto la memoria cuando estuvo IGUAL DE AJUSTADA que ahora.

        Un margen grande entre el primero y el segundo concepto es una decision comoda; uno
        chico es una decision al filo. En vez de inventar un umbral de margen, se mira que
        paso de verdad en las decisiones PARECIDAS A ESTA: si de las doce veces que estuvo
        asi de ajustada acerto seis, la confianza es 50% y el programa lo dice.

        SE MIRAN LOS MARGENES MAS CERCANOS, no "los menores o iguales". Al principio esto
        filtraba con m <= margen, y hacia exactamente lo contrario de lo que dice: para un
        margen comodo el conjunto era TODA la memoria --todos los casos son menores o
        iguales que uno comodo-- asi que la confianza caia al acierto global y la decision
        mas clara del dia se rechazaba; y para un empate de verdad quedaban dos o tres
        casos, menos que el minimo, devolvia None y el empate pasaba sin control. Medido
        sobre una memoria real: una consulta con 50.8 puntos de ventaja salia "NO ESTOY
        SEGURO - esta muy parejo", y dos empates de 31 y 21 puntos salian APROBADO sin una
        palabra de duda.

        None si no hay casos suficientes para que el numero signifique algo.
        """
        if len(self.casos) < minimo_casos:
            return None
        ventana = max(minimo_casos, round(len(self.casos) * 0.25))
        cercanos = sorted(self.casos, key=lambda caso: abs(caso[0] - margen))[:ventana]
        return sum(1 for _, acerto in cercanos if acerto) / len(cercanos)


# ---------------------------------------------------------------------------
# Descriptores
# ---------------------------------------------------------------------------


def _gris(imagen: np.ndarray, lado: int) -> np.ndarray:
    if imagen.ndim == 3:
        imagen = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    return cv2.resize(imagen, (lado, lado), interpolation=cv2.INTER_AREA)


def _normalizar(gris: np.ndarray) -> np.ndarray:
    """Saca brillo y contraste propios: lo que queda es la forma, no la luz."""
    g = gris.astype(np.float32)
    return (g - g.mean()) / (g.std() + 1e-6)


def _firma_bordes(gris: np.ndarray) -> np.ndarray:
    """Hacia donde apuntan los bordes en cada bloque. Vector de largo 1.

    Se parte en 4x4 y no se mira la imagen entera de una: dos piezas con las mismas
    estriaciones en distinto lugar darian identicas si se mezclara todo. Partiendo en
    bloques se conserva DONDE esta cada cosa, que es lo que distingue un filo entero de
    uno con la punta redondeada.
    """
    gx = cv2.Sobel(gris, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gris, cv2.CV_32F, 0, 1, ksize=3)
    magnitud = cv2.magnitude(gx, gy)
    # El angulo va modulo pi: un borde no tiene "sentido", solo direccion.
    angulo = (np.arctan2(gy, gx) % np.pi) / np.pi
    paso = gris.shape[0] // _BLOQUES
    trozos = []
    for fila in range(_BLOQUES):
        for columna in range(_BLOQUES):
            corte = (slice(fila * paso, (fila + 1) * paso),
                     slice(columna * paso, (columna + 1) * paso))
            histograma, _ = np.histogram(
                angulo[corte], bins=_ANGULOS, range=(0.0, 1.0), weights=magnitud[corte]
            )
            trozos.append(histograma / (histograma.sum() + 1e-6))
    vector = np.concatenate(trozos).astype(np.float32)
    return vector / (float(np.linalg.norm(vector)) + 1e-9)


def rasgos(imagen: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(mini, bordes, forma) de una foto.

    mini   vector de 1024 con la miniatura normalizada, dividido por su largo, para que el
           producto escalar entre dos sea directamente la correlacion (-1 a 1).
    bordes vector de 144 con la firma de bordes, de largo 1: su producto escalar es el
           coseno (0 a 1).
    forma  la foto en gris a 128x128, en uint8, para la comparacion alineada.
    """
    forma = _gris(imagen, LADO_FORMA)
    mini = _normalizar(_gris(imagen, LADO_MINI)).ravel() / float(LADO_MINI)
    return mini.astype(np.float32), _firma_bordes(forma), forma


def alineado(forma_a: np.ndarray, forma_b: np.ndarray) -> float:
    """Parecido de dos formas quedandose con el mejor calce. 0 a 1.

    Es la misma cuenta que revision.parecido(), a 128 px. Se recorta el centro de B y se lo
    busca sobre A: asi la pieza puede estar corrida y el numero no se derrumba.
    """
    margen = LADO_FORMA // _RECORTE
    plantilla = _normalizar(forma_b)[margen:-margen, margen:-margen]
    if plantilla.size == 0:
        return 0.0
    # UNA FOTO DE UN SOLO COLOR SE PARECE A TODO. Al normalizarla queda en ceros, y
    # matchTemplate con una plantilla sin varianza devuelve el maximo contra cualquier
    # imagen: medido, una foto negra archivada daba alineado 1.000 contra un diente, contra
    # un astillado y contra cualquier cosa, y su concepto se quedaba con un piso de 0.500
    # para toda consulta. Una foto asi no se parece a nada: no tiene con que parecerse.
    if float(plantilla.std()) < 1e-6:
        return 0.0
    respuesta = cv2.matchTemplate(_normalizar(forma_a), plantilla, cv2.TM_CCOEFF_NORMED)
    return float(np.clip(respuesta.max(), 0.0, 1.0))


_NOMBRES_ZONA = (
    "arriba a la izquierda", "arriba", "arriba a la derecha",
    "a la izquierda", "en el centro", "a la derecha",
    "abajo a la izquierda", "abajo", "abajo a la derecha",
)


# Cuanto tiene que destacarse una zona sobre la mediana de las nueve para nombrarla.
# SIEMPRE hay una zona que es la peor; decirla cuando no se destaca es mandar al operario
# a mirar al azar. Medido sobre las fotos de microscopio del taller:
#
#     el mismo diente con otra exposicion              1.03 - 1.04
#     dos tomas realmente distintas                    1.47 - 2.83  (mediana 1.89)
#     una foto con una mancha agregada en una esquina  3.58
#
# El corte va en 3.0: arriba quedan las diferencias que estan EN UN LUGAR, abajo las que
# estan repartidas por toda la imagen, que no sirven para mandar a mirar a ningun lado.
_ZONA_SE_DESTACA = 3.0


def zona_distinta(forma_a: np.ndarray, forma_b: np.ndarray) -> Optional[str]:
    """En que parte de la imagen se diferencian mas, ya alineadas.

    Un porcentaje dice CUANTO se parece; esto dice DONDE no. Para el que afila es la
    diferencia entre "no se parece" y "fijate la punta".

    Devuelve None cuando no se puede medir o cuando ninguna zona se destaca de las demas.
    """
    margen = LADO_FORMA // _RECORTE
    a = _normalizar(forma_a)
    b = _normalizar(forma_b)
    plantilla = b[margen:-margen, margen:-margen]
    if plantilla.size == 0:
        return None
    respuesta = cv2.matchTemplate(a, plantilla, cv2.TM_CCOEFF_NORMED)
    _, _, _, mejor = cv2.minMaxLoc(respuesta)

    # El calce de la plantilla da el desplazamiento entre las dos fotos; con el se compara
    # TODO lo que se superpone, no solo el centro recortado. Si no, una diferencia en el
    # borde --justo donde suele estar la punta del diente-- quedaria afuera.
    dx, dy = mejor[0] - margen, mejor[1] - margen
    ancho, alto = LADO_FORMA - abs(dx), LADO_FORMA - abs(dy)
    if ancho < 30 or alto < 30:
        return None
    recorte_a = a[max(0, dy):max(0, dy) + alto, max(0, dx):max(0, dx) + ancho]
    recorte_b = b[max(0, -dy):max(0, -dy) + alto, max(0, -dx):max(0, -dx) + ancho]
    diferencia = np.abs(recorte_a - recorte_b)

    paso_f, paso_c = alto // 3, ancho // 3
    if paso_f == 0 or paso_c == 0:
        return None
    valores = [
        float(diferencia[fila * paso_f:(fila + 1) * paso_f,
                         columna * paso_c:(columna + 1) * paso_c].mean())
        for fila in range(3)
        for columna in range(3)
    ]
    peor = max(valores)
    mediana = float(np.median(valores))
    if mediana <= 1e-6 or peor < mediana * _ZONA_SE_DESTACA:
        return None
    return _NOMBRES_ZONA[valores.index(peor)]


# ---------------------------------------------------------------------------
# La memoria
# ---------------------------------------------------------------------------


class Memoria:
    """Todas las fotos archivadas, listas para comparar contra lo que ve la camara."""

    def __init__(
        self,
        directorio: str = "data/memoria",
        candidatas_por_concepto: int = 6,
        vecinos_por_concepto: int = 3,
        maximo_autoevaluacion: int = 120,
    ) -> None:
        self._base = ruta_absoluta(directorio)
        self._candidatas = max(1, int(candidatas_por_concepto))
        self._vecinos = max(1, int(vecinos_por_concepto))
        self._tope_auto = max(0, int(maximo_autoevaluacion))

        self._claves: list[str] = []        # archivo|mtime|tamano, para saber que cambio
        self._archivos: list[str] = []
        self._conceptos: list[str] = []
        self._conceptos_np = np.zeros(0, dtype=object)
        self._mini = np.zeros((0, DIM_MINI), np.float32)
        self._bordes = np.zeros((0, DIM_BORDES), np.float32)
        self._formas = np.zeros((0, LADO_FORMA, LADO_FORMA), np.uint8)
        self._auto: Optional[Autoevaluacion] = None
        self._cargar()

    # ------------------------------------------------------------------ lectura
    @property
    def cantidad(self) -> int:
        return len(self._archivos)

    @property
    def archivo_cache(self) -> Path:
        return self._base / _ARCHIVO

    def conteo_por_concepto(self) -> dict[str, int]:
        conteo: dict[str, int] = {}
        for concepto in self._conceptos:
            conteo[concepto] = conteo.get(concepto, 0) + 1
        return conteo

    def conceptos_con_fotos(self) -> list[str]:
        return sorted(self.conteo_por_concepto())

    def lista_para_juzgar(self, minimo_por_concepto: int = 3) -> bool:
        """True si hay al menos dos conceptos con fotos suficientes para comparar.

        Con un solo concepto no hay decision posible: todo se parece a lo unico que hay.
        """
        conteo = self.conteo_por_concepto()
        return sum(1 for n in conteo.values() if n >= max(1, minimo_por_concepto)) >= 2
    # ------------------------------------------------------------------ carga

    def sincronizar(self, mapa: dict[str, list[Path]]) -> tuple[int, int]:
        """Pone la memoria al dia con las fotos que hay en disco.

        Devuelve (agregadas, sacadas). Solo lee del disco las fotos nuevas o cambiadas: el
        resto sale de la cache, asi que abrir el programa con cientos de fotos cuesta lo
        mismo que abrirlo con una.
        """
        por_clave = {clave: i for i, clave in enumerate(self._claves)}
        claves_nuevas: list[str] = []
        archivos: list[str] = []
        conceptos: list[str] = []
        filas_mini: list[np.ndarray] = []
        filas_bordes: list[np.ndarray] = []
        filas_formas: list[np.ndarray] = []
        agregadas = 0

        for concepto, rutas in mapa.items():
            for ruta in rutas:
                clave = _clave(ruta)
                if clave is None:
                    continue
                indice = por_clave.get(clave)
                if indice is not None and self._conceptos[indice] == concepto:
                    filas_mini.append(self._mini[indice])
                    filas_bordes.append(self._bordes[indice])
                    filas_formas.append(self._formas[indice])
                else:
                    imagen = cv2.imread(str(ruta))
                    if imagen is None:
                        _log.warning("no se pudo leer la foto %s; se la saltea", ruta)
                        continue
                    try:
                        mini, bordes, forma = rasgos(imagen)
                    except cv2.error as error:
                        _log.warning("no se pudo analizar %s: %s", ruta, error)
                        continue
                    filas_mini.append(mini)
                    filas_bordes.append(bordes)
                    filas_formas.append(forma)
                    agregadas += 1
                claves_nuevas.append(clave)
                archivos.append(str(ruta))
                conceptos.append(concepto)

        sacadas = max(0, len(self._claves) - (len(claves_nuevas) - agregadas))
        cambio = agregadas > 0 or sacadas > 0 or len(claves_nuevas) != len(self._claves)

        self._claves = claves_nuevas
        self._archivos = archivos
        self._conceptos = conceptos
        self._mini = (np.stack(filas_mini) if filas_mini
                      else np.zeros((0, DIM_MINI), np.float32))
        self._bordes = (np.stack(filas_bordes) if filas_bordes
                        else np.zeros((0, DIM_BORDES), np.float32))
        self._formas = (np.stack(filas_formas) if filas_formas
                        else np.zeros((0, LADO_FORMA, LADO_FORMA), np.uint8))

        self._reindexar()

        if cambio:
            self._auto = None
            self._guardar()
            _log.info(
                "memoria: %d fotos (%d nuevas, %d que ya no estan)",
                self.cantidad, agregadas, sacadas,
            )
        return agregadas, sacadas

    # ------------------------------------------------------------------ juicio
    def juzgar(
        self,
        imagen: np.ndarray,
        parecido_minimo: float = 0.35,
        confianza_minima: float = 0.70,
        minimo_por_concepto: int = 3,
    ) -> Juicio:
        """En que concepto cae esta foto, segun las que ya estan guardadas."""
        juicio = Juicio(fotos_en_memoria=self.cantidad)
        if self.cantidad == 0:
            juicio.motivo = _SIN_FOTOS
            return juicio
        if not self.lista_para_juzgar(minimo_por_concepto):
            minimo = max(1, int(minimo_por_concepto))
            cuantas = f"{minimo} fotos" if minimo > 1 else "una foto"
            listos = sum(1 for n in self.conteo_por_concepto().values() if n >= minimo)
            juicio.motivo = (
                f"para comparar hacen falta dos conceptos con {cuantas} cada uno; "
                f"por ahora hay {listos}"
            )
            return juicio
        try:
            mini, bordes, forma = rasgos(imagen)
        except cv2.error:
            juicio.motivo = "no se pudo analizar la imagen"
            return juicio

        vecinos, puntajes = self._vecinos_y_puntajes(mini, bordes, forma, excluir=None)
        if not vecinos:
            juicio.motivo = _SIN_FOTOS
            return juicio

        juicio.vecinos = vecinos
        juicio.puntajes = puntajes
        juicio.parecido = vecinos[0].parecido

        ordenados = sorted(puntajes.items(), key=lambda kv: kv[1], reverse=True)
        mejor, valor = ordenados[0]
        segundo = ordenados[1][1] if len(ordenados) > 1 else 0.0
        juicio.margen = float(valor - segundo)

        if juicio.parecido < float(parecido_minimo):
            juicio.motivo = (
                f"esta foto no se parece a ninguna de las {self.cantidad} guardadas "
                f"(la que mas, {juicio.parecido * 100:.0f}%)"
            )
            return juicio

        auto = self.autoevaluacion()
        juicio.confianza = auto.confianza(juicio.margen)
        if juicio.confianza is not None and juicio.confianza < float(confianza_minima):
            juicio.motivo = (
                f"{mejor} y {ordenados[1][0]} quedaron a {juicio.margen * 100:.1f} puntos, "
                f"y en decisiones asi de ajustadas la memoria acierta "
                f"{juicio.confianza * 100:.0f}% de las veces"
            )
            return juicio

        juicio.concepto = mejor
        # EL PORCENTAJE QUE SE MUESTRA TIENE QUE SER DEL CONCEPTO QUE SE NOMBRA. Hasta aca
        # juicio.parecido es el mejor vecino de TODA la memoria --que es lo correcto para
        # decidir si la foto se parece a algo-- pero la pantalla lo escribe pegado al
        # nombre del concepto que gano: medido, llego a decir "se parece a filo_quemado
        # (100%)" cuando ese 100% era de una foto archivada en aprobada. Es exactamente el
        # tipo de numero que ya costo caro en este proyecto.
        propio = max(
            (v for v in vecinos if v.concepto == mejor),
            key=lambda v: v.parecido,
            default=vecinos[0],
        )
        juicio.parecido = propio.parecido
        try:
            juicio.zona = zona_distinta(forma, self._formas[self._indice_de(propio)])
        except (cv2.error, ValueError):
            juicio.zona = None
        return juicio

    def _indice_de(self, vecino: Vecino) -> int:
        return self._archivos.index(vecino.archivo)

    def _fecha_de(self, indice: int) -> int:
        """La fecha de la foto, sacada de su clave. 0 si la clave no tiene forma."""
        try:
            return int(self._claves[indice].split("|")[-2])
        except (IndexError, ValueError):
            return 0

    def _vecinos_y_puntajes(
        self,
        mini: np.ndarray,
        bordes: np.ndarray,
        forma: np.ndarray,
        excluir: Optional[int],
    ) -> tuple[list[Vecino], dict[str, float]]:
        """El corazon: etapa barata por concepto, etapa exacta sobre las candidatas."""
        barato = 0.5 * (self._mini @ mini) + 0.5 * (self._bordes @ bordes)
        if excluir is not None:
            barato[excluir] = -np.inf

        vecinos: list[Vecino] = []
        por_concepto: dict[str, list[Vecino]] = {}
        for concepto in self.conteo_por_concepto():
            filas = np.flatnonzero(self._conceptos_np == concepto)
            if excluir is not None:
                filas = filas[filas != excluir]
            if filas.size == 0:
                continue
            orden = filas[np.argsort(-barato[filas])][: self._candidatas]
            del_concepto: list[Vecino] = []
            for fila in orden:
                valor_alineado = alineado(forma, self._formas[fila])
                valor_bordes = float(np.clip(self._bordes[fila] @ bordes, 0.0, 1.0))
                del_concepto.append(
                    Vecino(
                        concepto=concepto,
                        archivo=self._archivos[fila],
                        parecido=0.5 * (valor_alineado + valor_bordes),
                        alineado=valor_alineado,
                        bordes=valor_bordes,
                    )
                )
            del_concepto.sort(key=lambda v: v.parecido, reverse=True)
            por_concepto[concepto] = del_concepto
            vecinos.extend(del_concepto)

        # TODOS LOS CONCEPTOS SE PUNTUAN CON LA MISMA CANTIDAD DE FOTOS. Antes cada uno
        # promediaba sus mejores tres, o menos si no tenia tres, asi que el de UNA foto
        # puntuaba por su maximo y el de cuatro por un promedio: medido, el mismo concepto
        # pasaba de 0.9997 con una foto a 0.6328 con tres, y entonces archivar la primera
        # foto de un concepto nuevo le daba vuelta el veredicto a la pieza que se estaba
        # mirando. Se usa el minimo comun --el concepto con menos candidatas manda-- para
        # que la comparacion sea entre iguales y ninguno quede afuera por chico.
        comun = min(
            [self._vecinos] + [len(v) for v in por_concepto.values()]
        ) if por_concepto else self._vecinos
        comun = max(1, comun)
        puntajes = {
            concepto: float(sum(v.parecido for v in lista[:comun]) / comun)
            for concepto, lista in por_concepto.items()
        }

        vecinos.sort(key=lambda v: v.parecido, reverse=True)
        return vecinos, puntajes

    # ------------------------------------------------------------------ medirse
    def autoevaluacion(self) -> Autoevaluacion:
        """Esconde cada foto y le pregunta a las demas. Se calcula una vez y se guarda."""
        if self._auto is not None:
            return self._auto

        total = self.cantidad
        auto = Autoevaluacion(medidas_de=total)
        if total < 2:
            self._auto = auto
            return auto

        # Con muchas fotos esto es O(n^2) y hay que acotarlo: se mide sobre las mas nuevas
        # y se dice en pantalla que se midio sobre esas. Recortar en silencio seria mentir
        # con un numero que parece el acierto de toda la memoria.
        #
        # LAS MAS NUEVAS POR FECHA, no por posicion en la lista. Las filas vienen agrupadas
        # por concepto --se arman recorriendo el mapa de carpetas-- asi que cortar el final
        # de la lista se quedaba con las fotos mas VIEJAS del ULTIMO concepto, y el acierto
        # que la pantalla presenta como el de toda la memoria podia salir de un solo
        # concepto. La fecha ya esta guardada en la clave de cada foto.
        indices = list(range(total))
        if self._tope_auto and total > self._tope_auto:
            indices.sort(key=self._fecha_de)
            indices = indices[-self._tope_auto:]
            auto.recortada = True

        for i in indices:
            _, puntajes = self._vecinos_y_puntajes(
                self._mini[i], self._bordes[i], self._formas[i], excluir=i
            )
            if len(puntajes) < 2:
                continue
            ordenados = sorted(puntajes.items(), key=lambda kv: kv[1], reverse=True)
            predicho = ordenados[0][0]
            margen = float(ordenados[0][1] - ordenados[1][1])
            verdadero = self._conceptos[i]
            acerto = predicho == verdadero
            auto.total += 1
            auto.aciertos += int(acerto)
            bien, cuantas = auto.por_concepto.get(verdadero, (0, 0))
            auto.por_concepto[verdadero] = (bien + int(acerto), cuantas + 1)
            auto.casos.append((margen, acerto))

        self._auto = auto
        self._guardar()
        return auto

    def resumen(self) -> str:
        """Una linea para la pantalla: cuanto sabe y cuanto acierta."""
        if self.cantidad == 0:
            return "memoria vacia: cada foto que guardes se suma aca"
        conteo = self.conteo_por_concepto()
        partes = "  ".join(f"{nombre} {cantidad}" for nombre, cantidad in sorted(conteo.items()))
        auto = self.autoevaluacion()
        if auto.total < 2:
            return f"memoria: {self.cantidad} fotos ({partes})"
        sobre = " de las ultimas" if auto.recortada else " de"
        return (
            f"memoria: {self.cantidad} fotos ({partes})  -  acierta "
            f"{auto.aciertos}{sobre} {auto.total} que tiene guardadas"
        )

    # ------------------------------------------------------------------ disco
    def _cargar(self) -> None:
        archivo = self.archivo_cache
        if not archivo.is_file():
            return
        try:
            with np.load(archivo, allow_pickle=False) as datos:
                if int(datos["version"][0]) != _VERSION:
                    _log.info("la memoria guardada es de otra version; se rehace")
                    return
                self._claves = [str(v) for v in datos["claves"]]
                self._archivos = [str(v) for v in datos["archivos"]]
                self._conceptos = [str(v) for v in datos["conceptos"]]
                self._mini = datos["mini"].astype(np.float32)
                self._bordes = datos["bordes"].astype(np.float32)
                self._formas = datos["formas"].astype(np.uint8)
                crudo = str(datos["auto"][0]) if "auto" in datos else ""
        except (OSError, ValueError, KeyError, EOFError, zipfile.BadZipFile) as error:
            # Un npz a medio escribir --corte de luz, disco lleno-- no puede dejar al
            # taller sin programa: la cache no es un dato, es un atajo, y se rehace sola
            # leyendo las fotos.
            _log.warning("no se pudo leer la memoria guardada (%s); se rehace", error)
            self._vaciar()
            return

        if not (len(self._claves) == len(self._archivos) == len(self._conceptos)
                == len(self._mini) == len(self._bordes) == len(self._formas)):
            _log.warning("la memoria guardada esta incompleta; se rehace")
            self._vaciar()
            return

        self._auto = _auto_desde_json(crudo)
        self._reindexar()

    def _reindexar(self) -> None:
        """La lista de conceptos tambien como arreglo, que es con lo que se filtra."""
        self._conceptos_np = np.array(self._conceptos, dtype=object)

    def _vaciar(self) -> None:
        self._claves, self._archivos, self._conceptos = [], [], []
        self._mini = np.zeros((0, DIM_MINI), np.float32)
        self._bordes = np.zeros((0, DIM_BORDES), np.float32)
        self._formas = np.zeros((0, LADO_FORMA, LADO_FORMA), np.uint8)
        self._auto = None
        self._reindexar()

    def _guardar(self) -> None:
        """Escribe la cache. Que falle solo cuesta tiempo en el proximo arranque.

        Se escribe en un archivo aparte y recien al final se lo pone en su lugar de un
        golpe. Asi una escritura cortada a la mitad --el operario apaga la PC con la llave
        general, que en un taller pasa-- no llega nunca a ser la cache viva.
        """
        provisorio = self.archivo_cache.with_suffix(".npz.tmp")
        try:
            self._base.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                provisorio,
                version=np.array([_VERSION], np.int32),
                claves=_textos(self._claves),
                archivos=_textos(self._archivos),
                conceptos=_textos(self._conceptos),
                mini=self._mini,
                bordes=self._bordes,
                formas=self._formas,
                auto=np.array([_auto_a_json(self._auto)], dtype="U"),
            )
            # np.savez le agrega .npz al nombre si no lo termina en .npz.
            escrito = provisorio if provisorio.is_file() else Path(str(provisorio) + ".npz")
            os.replace(escrito, self.archivo_cache)
        except (OSError, ValueError) as error:
            _log.warning("no se pudo guardar la memoria: %s", error)


def _textos(lista: list[str]) -> np.ndarray:
    """Lista de textos como arreglo de numpy, soportando la lista vacia."""
    if not lista:
        return np.zeros(0, dtype="U1")
    return np.array(lista, dtype="U")


def _clave(ruta: Path) -> Optional[str]:
    """Identifica una foto por carpeta, nombre, fecha y tamano.

    La CARPETA entra en la clave porque dos conceptos pueden tener una foto con el mismo
    nombre --pasa al copiar una tanda con el Explorador-- y sin ella la cache no convergia
    nunca: cada arranque informaba una foto nueva y una que ya no estaba, y volvia a
    analizar las dos.
    """
    try:
        est = ruta.stat()
    except OSError:
        return None
    return f"{ruta.parent.name}/{ruta.name}|{est.st_mtime_ns}|{est.st_size}"


def _auto_a_json(auto: Optional[Autoevaluacion]) -> str:
    if auto is None:
        return ""
    return json.dumps(
        {
            "total": auto.total,
            "aciertos": auto.aciertos,
            "por_concepto": {k: list(v) for k, v in auto.por_concepto.items()},
            "casos": [[round(m, 6), bool(a)] for m, a in auto.casos],
            "medidas_de": auto.medidas_de,
            "recortada": auto.recortada,
        }
    )


def _auto_desde_json(crudo: str) -> Optional[Autoevaluacion]:
    if not crudo:
        return None
    try:
        datos = json.loads(crudo)
        return Autoevaluacion(
            total=int(datos["total"]),
            aciertos=int(datos["aciertos"]),
            por_concepto={k: (int(v[0]), int(v[1])) for k, v in datos["por_concepto"].items()},
            casos=[(float(m), bool(a)) for m, a in datos["casos"]],
            medidas_de=int(datos.get("medidas_de", 0)),
            recortada=bool(datos.get("recortada", False)),
        )
    except (ValueError, KeyError, TypeError):
        return None
