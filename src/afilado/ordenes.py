"""La ORDEN DE AFILADO y su informe: de que sierra es cada diente que se reviso.

PARA QUE SIRVE ESTO
-------------------
Hasta ahora el programa miraba dientes sueltos: decia algo de la foto que tenia adelante y
se olvidaba. Cuando llega un reclamo --"esta sierra volvio mal afilada"-- no hay con que
contestar, porque no quedo registrado QUE sierra era, de que orden, ni que se vio en cada
diente.

Una orden de afilado agrupa eso: el numero de orden, la nota de pedido, el cliente, cuantas
sierras entraron y cuantos dientes tiene cada una. Cada diente que se revisa queda anotado
adentro con la fecha, la hora, en que estado quedo y de donde salio ese veredicto. De ahi
sale el informe que se imprime o se manda.

TODO SE CARGA A MANO, POR AHORA
-------------------------------
La idea es que estos datos vengan de la app de afilados, pero eso todavia no esta conectado
y conectarlo a medias seria peor que nada: un numero de orden equivocado en un informe de
reclamo es peor que no tener informe. Asi que el operario los escribe, y el dia que haya
conexion lo unico que cambia es de donde salen los mismos campos --Orden.desde_datos() esta
hecha para eso--.

QUE NO HACE ESTE MODULO
-----------------------
No mira imagenes ni decide nada: solo guarda lo que le dicen y lo escribe ordenado. Quien
decide es la memoria (memoria.py) o el modelo entrenado; aca llega el resultado ya tomado.
"""

from __future__ import annotations

import csv
import io
import json
import logging
import os
import shutil
import zipfile
from dataclasses import asdict, dataclass, field, fields
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from .config import ruta_absoluta

_log = logging.getLogger(__name__)

# EN QUE FORMATO SE ESCRIBIO EL ARCHIVO QUE SE ESTA LEYENDO. La 2 es la primera que puede
# guardar la foto de prueba de cada diente. Medido: las 4 ordenes que hay hoy en el taller
# NO tienen campo version, asi que las cuatro son la 1.
#
# ESTO NO SIRVE PARA SABER SI UNA ORDEN TIENE PRUEBAS, Y ANTES PROMETIA QUE SI. La version
# dice con que programa se escribio el archivo, no que haya adentro: una orden del taller
# que era version 1 y a la que se le anota UN diente con prueba queda escrita por esta
# version --o sea version 2-- con los otros dientes sin prueba, y una orden nacida hoy
# puede no tener ninguna. Medido antes de cambiarlo: se le anoto un diente con prueba de
# verdad a la 1234 del taller, se guardo, y el .json quedaba con version 1 Y con una prueba
# adentro. Asi que la migracion a la base de datos NO puede filtrar por version: tiene que
# mirar diente.prueba en TODAS las ordenes, una por una. Para saber de donde viene una
# orden esta version_origen, que no se toca al reescribirla.
VERSION_FORMATO = 2

_ARCHIVO_ABIERTA = ".abierta"
_PAPELERA = "_papelera"
# El deposito de pruebas cuelga de la carpeta data, AL LADO de las ordenes y FUERA de la
# coleccion de entrenamiento. Medido antes de decidirlo: si la prueba entrara a la
# coleccion, archivar() --que llama a refrescar()-- pasa de 209 ms con 15 fotos a 3.153 ms
# con 320, y 320 fotos es UNA orden de 4 sierras de 80 dientes. Contra los 1.200 ms que el
# operario apoya cada diente, eso es inusable antes de terminar la primera orden.
_CARPETA_PRUEBAS = "pruebas"
_CARPETA_EXPORTADO = "exportado"
# EL PAPEL QUE EXPLICA DE DONDE SALIERON LAS FOTOS DE UNA CARPETA DE PRUEBAS. No es una
# prueba ni una foto suelta: es el unico archivo del deposito cuyo trabajo es que lo lea el
# que recibe el reclamo, asi que viaja en el paquete y no se cuenta entre las fotos que
# ningun diente nombra.
# POR QUE EXISTE: las ordenes 123 y 1234 se revisaron cuando el programa todavia no tenia
# deposito de pruebas, y sus 14 fotos se rescataron de la papelera de la coleccion cuatro
# dias despues, el 16/11/2026. Esas fotos NO se guardaron en el momento de juzgar el diente,
# y mandarlas en un reclamo sin decirlo seria presentarlas como algo que no son. El papel lo
# dice, con el sha256 de cada una.
# Que no se cuente como huerfana tampoco es un favor: el informe decia "en la carpeta hay 1
# fotos que ningun diente de esta orden nombra" y ese archivo no es una foto.
_PAPEL_PROCEDENCIA = "procedencia.txt"
# La misma calidad que usa el resto del programa (config.calidad_jpg = 95). Medido sobre
# las 15 fotos de verdad del microscopio reescritas a 95: mediana 95.881 bytes (min 75.332,
# max 110.154), practicamente lo mismo que pesan las originales (mediana 95.648). O sea que
# guardar la prueba no la degrada ni la engorda.
_CALIDAD_PRUEBA = 95
# Lo que saltear() deja escrito en el origen. Se usa tambien para reconocer los salteados
# de las ordenes viejas, de cuando el diente no tenia un campo propio para esto.
_ORIGEN_SALTEADO = "sin revisar"
_CARACTERES_VALIDOS = set(
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_"
)

# Topes de sensatez para lo que se teclea. No son limites del negocio: son el corte entre
# "se equivoco al escribir" y "esto es un numero". Una sierra circular grande ronda los 100
# dientes; 500 ya es un cero de mas.
MAX_SIERRAS = 200
MAX_DIENTES = 500


def sanear_numero(texto: str) -> str:
    """Deja un numero de orden usable como nombre de archivo."""
    limpio = "".join(c if c in _CARACTERES_VALIDOS else "_" for c in str(texto))
    return limpio[:40].strip("_-")


def nombre_de_archivo_sano(archivo: str) -> bool:
    """Si `archivo` es un nombre de archivo PELADO y no una ruta disfrazada.

    POR QUE HACE FALTA. El historial lleva el nombre del .json tal cual, y ese nombre es lo
    que ahora se puede pasar a cargar(), borrar(), exportar() y carpeta_pruebas() para que
    no se lo vuelva a sanear --si se lo sanea otra vez, "123 - copia.json" apunta al 123 de
    verdad y borrarlo se lleva la orden del cliente--. Pero un nombre que no se sanea es un
    nombre que entra tal cual al armar una ruta, y por ahi NO puede entrar una ruta: ni
    "..\\..\\algo", ni "C:/otra/parte", ni una barra suelta. Lo que vino del disco no se
    discute, pero lo que llega por un parametro se mira.
    """
    texto = str(archivo)
    if not texto:
        return False
    if "/" in texto or "\\" in texto or ":" in texto or ".." in texto:
        return False
    # Y el cinturon por si Windows reconoce alguna forma mas de nombrar una carpeta de la
    # que no nos enteramos: si el nombre no es igual a su propio ultimo tramo, es una ruta.
    return texto == Path(texto).name


def clave_de_fecha(texto: str, respaldo: float = 0.0) -> str:
    """Pasa la fecha que se muestra --"16/11/2026 09:34:20"-- a una que SE ORDENA BIEN:
    "2026-11-16 09:34:20".

    HACE FALTA PORQUE LA FECHA GUARDADA NO SE PUEDE ORDENAR COMO TEXTO. _ahora() escribe
    dia/mes/anio, asi que ordenando esos textos "03/01/2027 10:00:00" queda ANTES de
    "12/11/2026 15:27:44" --verificado con las fechas de verdad-- y el historial mostraria
    arriba una orden de enero del anio que viene como si fuera la mas vieja. Con el anio
    adelante, ordenar texto y ordenar fechas es lo mismo.

    Si el texto no se entiende se usa 'respaldo', que es el mtime del archivo: no es la
    fecha de la orden, pero es lo unico medible que hay y ubica la fila en el lugar
    aproximado en vez de mandarla al fondo. Si no hay ni eso devuelve "", que ordena
    primero: una orden sin fecha legible queda ultima en el historial --que va al reves--
    y eso es correcto, no sabemos cuando fue.

    Y es la que manana hace mecanica la migracion a la base de datos: el dia que las
    ordenes vivan en una tabla, esta es la columna por la que se indexan.
    """
    crudo = " ".join(str(texto or "").split())
    if crudo:
        try:
            return datetime.strptime(crudo, "%d/%m/%Y %H:%M:%S").strftime(
                "%Y-%m-%d %H:%M:%S"
            )
        except ValueError:
            pass
    if respaldo:
        try:
            return datetime.fromtimestamp(float(respaldo)).strftime("%Y-%m-%d %H:%M:%S")
        except (OSError, OverflowError, TypeError, ValueError):
            return ""
    return ""


@dataclass
class Diente:
    """Lo que se vio en un diente. Es la unidad del informe."""

    numero: int
    estado: Optional[str] = None        # el concepto: afilado, mal_afilado, roto...
    parecido: Optional[float] = None    # 0..1, cuanto se parecio a ese concepto
    aprobado: Optional[bool] = None     # None = no se pudo decidir
    origen: str = ""                    # de donde salio el veredicto, en castellano
    instante: str = ""
    foto: str = ""                      # la foto archivada, si se archivo
    nota: str = ""                      # el motivo del salteo, o lo que haga falta aclarar
    salteado: bool = False              # se paso de largo: HAY renglon, pero no se miro
    # LA FOTO DE PRUEBA, en ruta relativa a la carpeta data y con barras normales:
    # "pruebas/5113546/s01_d007_20261116_093432_481.jpg". Relativa y no absoluta porque la
    # carpeta Afilado se copia de una PC a otra y una ruta con C:\Users\... adentro no
    # sirve del otro lado.
    #
    # NO ES LO MISMO QUE `foto`, Y MEZCLARLAS SERIA INVENTAR UNA PRUEBA. `foto` guarda el
    # nombre pelado de la foto que entro a la coleccion de entrenamiento --sin carpeta, asi
    # que ni el programa la encuentra-- y es lo unico que tienen las ordenes viejas. Medido:
    # las 14 fotos que nombraban las ordenes 123 y 1234 estaban TODAS en la papelera de la
    # coleccion y ninguna viva, porque el operario borro ese concepto el 13/11. Un informe
    # que muestre ese nombre como prueba esta mostrando algo que no existe.
    # El 17/11/2026 esas 14 se rescataron de la papelera y se copiaron al deposito de
    # pruebas de cada orden, asi que la 123 y la 1234 ya tienen `prueba` ademas de `foto`;
    # al lado de las fotos quedo un procedencia.txt que dice que no se sacaron en el
    # momento de juzgar el diente. Las demas ordenes viejas siguen con el nombre pelado.
    prueba: str = ""

    def __post_init__(self) -> None:
        # Ordenes guardadas antes de que existiera el campo: ahi un diente salteado se
        # reconocia solo por el origen que le ponia saltear(). Sin esto, al releer una de
        # esas ordenes los salteados volvian a contarse como revisados.
        if not self.salteado and self.origen == _ORIGEN_SALTEADO:
            self.salteado = True
        # LA PRUEBA TIENE QUE SER UN TEXTO CON BARRAS NORMALES. Una orden vieja no trae el
        # campo --y ahi no hay nada que arreglar--, pero una tocada a mano puede traer un
        # null o la ruta con barras de Windows, y las dos cosas se pagan despues: el null
        # revienta al armar la ruta del archivo y las barras invertidas no se encuentran
        # cuando la carpeta Afilado se abre en otra PC.
        if not isinstance(self.prueba, str):
            self.prueba = "" if self.prueba is None else str(self.prueba)
        if "\\" in self.prueba:
            self.prueba = self.prueba.replace("\\", "/")
        # EL PARECIDO TIENE QUE SER UN NUMERO O NADA. Viene de un .json que puede estar
        # editado a mano o escrito por una version vieja, y con un texto adentro el
        # informe reventaba al formatearlo: ValueError sin atrapar, y con el .exe sin
        # consola eso es el programa que se cierra solo en la cara del operario, justo
        # cuando esta imprimiendo el papel de un reclamo.
        if self.parecido is not None:
            try:
                self.parecido = float(self.parecido)
            except (TypeError, ValueError):
                _log.warning(
                    "el diente %s traia un parecido que no es numero (%r): se deja vacio",
                    self.numero, self.parecido,
                )
                self.parecido = None

    @classmethod
    def desde_datos(cls, datos: dict[str, Any]) -> "Diente":
        """Un diente desde un diccionario, IGNORANDO las claves que esta version no conoce.

        Antes era Diente(**datos) pelado, y ahi una clave de mas no costaba un campo:
        costaba la ORDEN ENTERA. TypeError sale de adentro de Orden.desde_datos(), que lo
        envuelve todo, asi que la orden salia ilegible y no se podia ni exportar.
        Reproducido con una de las ordenes de verdad del taller: se le agrego a un diente
        una clave que esta version no conoce y cargar() devolvio None, 0 dientes revisados
        de 40.

        Es la misma cortesia que el programa necesita de los .exe viejos --por eso el campo
        prueba no se escribe cuando esta vacio-- y lo que hace que el PROXIMO campo nuevo
        no vuelva a costar una orden.
        """
        return cls(**{k: v for k, v in datos.items() if k in _CAMPOS_DIENTE})

    @property
    def anotado(self) -> bool:
        """Si este diente ya tiene renglon: revisado O salteado. Es lo que hace avanzar
        el conteo al siguiente."""
        return bool(self.instante)

    @property
    def revisado(self) -> bool:
        """Si este diente se MIRO de verdad.

        Un salteado no cuenta, y la diferencia no es un detalle: con esto adentro, apretar
        la tecla de saltear veinticuatro veces dejaba el informe diciendo "24 revisados" y
        la sierra marcada como completa sin que nadie hubiera mirado un solo diente. Ese
        informe es el que contesta un reclamo.
        """
        return bool(self.instante) and not self.salteado


# LAS CLAVES QUE UN DIENTE CONOCE, calculadas UNA VEZ al importar el modulo y no por
# diente: una orden de 4 sierras de 80 dientes son 320 dientes que se leen de un tiron cada
# vez que se abre el historial.
_CAMPOS_DIENTE = frozenset(f.name for f in fields(Diente))


@dataclass
class Sierra:
    """Una sierra de la orden, con sus dientes."""

    numero: int
    dientes: int
    identificacion: str = ""
    revisados: list[Diente] = field(default_factory=list)

    def __post_init__(self) -> None:
        # La lista se crea entera desde el principio: un informe de reclamo tiene que poder
        # mostrar TAMBIEN los dientes que no se revisaron. Un hueco en el medio es un dato.
        #
        # Y SE RELLENA POR NUMERO FALTANTE, NO POR POSICION. Rellenando por posicion, una
        # orden que llega a medio revisar --de la app de afilados, de un parcial
        # recuperado, de un .json tocado a mano-- perdia dientes: medido con 6 dientes y
        # los numeros 5 y 6 ya anotados, la lista quedaba [5, 6, 3, 4, 5, 6] y el informe
        # imprimia el 5 y el 6 dos veces, el 1 y el 2 en ninguna parte, y el total
        # cerrando igual ("6 dientes - 2 revisados"). Ese papel es el que contesta un
        # reclamo. Con los dientes completos --el ida y vuelta del propio programa, que
        # serializa siempre la lista entera-- no cambia nada: no falta ninguno y el orden
        # ya venia 1..N.
        for diente in self.revisados:
            try:
                diente.numero = int(diente.numero)
            except (TypeError, ValueError):
                # No se descarta el renglon: un diente con el numero raro a la vista es
                # menos malo que un diente que desaparece del informe.
                diente.numero = 0
        ya = {d.numero for d in self.revisados}
        for numero in range(1, int(self.dientes) + 1):
            if numero not in ya:
                self.revisados.append(Diente(numero=numero))
        self.revisados.sort(key=lambda d: d.numero)

    @property
    def cuantos_revisados(self) -> int:
        return sum(1 for d in self.revisados if d.revisado)

    @property
    def cuantos_salteados(self) -> int:
        return sum(1 for d in self.revisados if d.salteado)

    @property
    def cuantas_pruebas(self) -> int:
        """Cuantos dientes de esta sierra NOMBRAN una foto de prueba.

        NOMBRAN, no "tienen": esto cuenta renglones del .json y no mira el disco, que
        Sierra no sabe donde esta. Que la foto este o no se pregunta aparte --lo hace
        Ordenes.pruebas_en_disco()-- y el informe imprime los DOS numeros, porque la
        diferencia entre los dos es justo lo que ya paso con las ordenes 123 y 1234: 14
        fotos nombradas y ninguna viva.
        """
        return sum(1 for d in self.revisados if d.prueba)

    @property
    def completa(self) -> bool:
        """Si no queda ningun diente por tocar. Un salteado ya esta tocado --tiene su
        renglon y su motivo-- aunque no cuente como revisado: si no fuera asi, una sierra
        con un diente salteado no se terminaria nunca y el programa pediria revisar un
        diente que el operario ya decidio pasar de largo."""
        return self.proximo() is None

    @property
    def rechazados(self) -> int:
        return sum(1 for d in self.revisados if d.revisado and d.aprobado is False)

    @property
    def sin_decidir(self) -> int:
        return sum(1 for d in self.revisados if d.revisado and d.aprobado is None)

    def proximo(self) -> Optional[Diente]:
        """El primer diente sin anotar. None si ya estan todos."""
        for diente in self.revisados:
            if not diente.anotado:
                return diente
        return None


@dataclass
class Orden:
    """Una orden de afilado completa."""

    numero: str
    nota_pedido: str = ""
    cliente: str = ""
    abierta: str = ""
    cerrada: str = ""
    sierras: list[Sierra] = field(default_factory=list)
    sierra_actual: int = 1
    # CON QUE VERSION SE ESCRIBIO EL ARCHIVO QUE SE LEYO. Arranca en 1 --no en
    # VERSION_FORMATO-- porque las ordenes que ya estan en el disco no tienen el campo y
    # son la 1: medido, las 4 del taller no lo traen.
    #
    # Y LA SUBE EL QUE ESCRIBE: guardar() la pone en VERSION_FORMATO antes de serializar,
    # porque el archivo que se esta escribiendo lo escribio esta version y eso es un hecho,
    # no una opinion. Antes no la subia nadie y quedaba mintiendo al reves: una orden del
    # taller con un diente anotado con prueba de verdad seguia diciendo version 1.
    version: int = 1
    # DE QUE VERSION VIENE LA ORDEN, que es lo que version ya no puede decir una vez que se
    # la reescribe. 0 es "nacio antes de que hubiera versiones": las 4 ordenes del taller
    # no traen ni este campo ni el otro. nueva() lo pone en VERSION_FORMATO y desde_datos()
    # lo lee y NO LO TOCA NUNCA, ni al guardar: es el unico dato que queda de donde salio.
    version_origen: int = 0

    # ------------------------------------------------------------------ armado
    @classmethod
    def nueva(
        cls,
        numero: str,
        cantidad_sierras: int,
        dientes_por_sierra: int,
        nota_pedido: str = "",
        cliente: str = "",
    ) -> "Orden":
        # EL NUMERO QUE SE IMPRIME NO ES EL NOMBRE DEL ARCHIVO. Guardando el saneado, el
        # papel de un reclamo salia con un numero que el operario nunca escribio:
        # Orden.nueva("OA/123") imprimia "Orden de afilado : OA_123". Ahora se guarda tal
        # cual se teclea y el saneado se usa solo donde hace falta --Ordenes.ruta() y los
        # informe_*.txt/.csv--. Lo que sigue valiendo es que tiene que QUEDAR algo usable
        # como nombre de archivo: "////" no es un numero de orden.
        limpio = " ".join(str(numero).split())
        if not sanear_numero(limpio):
            raise ValueError("la orden necesita un numero")
        sierras = max(1, int(cantidad_sierras))
        # UN SOLO NUMERO PARA TODAS LAS SIERRAS DE LA ORDEN. Sierra lleva su propio campo
        # dientes y desde_datos() lo respeta por sierra, pero la carga a mano --que hoy es
        # la unica-- tiene un campo y pone el mismo numero en todas. Con tres sierras de
        # 24, 36 y 60 hay que elegir uno, y de ahi el informe declara "SIERRA 2: 24
        # dientes, 24 revisados" de una sierra de 36: una sierra dada por completa a la
        # que le faltan doce dientes, en el papel que contesta el reclamo. Hasta que el
        # formulario acepte un numero por sierra, la pantalla lo avisa.
        dientes = max(1, int(dientes_por_sierra))
        if sierras > MAX_SIERRAS:
            raise ValueError(f"{sierras} sierras es demasiado; el tope es {MAX_SIERRAS}")
        if dientes > MAX_DIENTES:
            raise ValueError(f"{dientes} dientes es demasiado; el tope es {MAX_DIENTES}")
        return cls(
            numero=limpio,
            nota_pedido=_una_linea(nota_pedido),
            cliente=_una_linea(cliente),
            abierta=_ahora(),
            sierras=[Sierra(numero=i + 1, dientes=dientes) for i in range(sierras)],
            # Una orden nueva nace en el formato de hoy: es la que SI puede guardar prueba.
            version=VERSION_FORMATO,
            # Y nacio aca, no se heredo de ninguna version anterior. Esto no cambia nunca
            # mas, aunque la orden se reescriba mil veces.
            version_origen=VERSION_FORMATO,
        )

    @classmethod
    def desde_datos(cls, datos: dict[str, Any]) -> "Orden":
        """Arma una orden desde un diccionario, venga de donde venga.

        Es el punto por donde va a entrar la app de afilados el dia que se conecte: lo que
        hoy teclea el operario y lo que manana mande la otra app terminan los dos aca, asi
        que el resto del programa no se entera del cambio.
        """
        sierras = []
        # Las claves de diente que esta version no conoce, para decirlo UNA VEZ por orden y
        # no una por diente: en una orden de 320 dientes serian 320 renglones de log
        # iguales tapando todo lo demas.
        sobrantes: set[str] = set()
        for crudo in datos.get("sierras", []):
            revisados = []
            for d in crudo.get("revisados", []):
                sobrantes |= set(d) - _CAMPOS_DIENTE
                revisados.append(Diente.desde_datos(d))
            # UNA SIERRA TIENE AL MENOS UN DIENTE, y el tope es el mismo que para lo que
            # teclea el operario. Esta es la puerta por donde va a entrar la app de
            # afilados, y sin esto un "dientes": 0 --o un negativo-- daba una sierra que
            # nacia COMPLETA sin haberse revisado nada: el informe decia "0 de 0
            # revisados, orden terminada" y la sierra no se podia anotar nunca.
            cuantos = max(1, min(int(crudo.get("dientes", 0) or 1), MAX_DIENTES))
            sierras.append(
                Sierra(
                    numero=int(crudo.get("numero", len(sierras) + 1)),
                    dientes=cuantos,
                    identificacion=str(crudo.get("identificacion", "")),
                    revisados=revisados,
                )
            )
        # La version se lee con pinzas: un "version": "dos" o un null no pueden tirar abajo
        # la lectura de una orden entera, que es lo unico que contesta un reclamo. Si no se
        # entiende, se la toma por la vieja, que es lo prudente: la vieja no promete nada.
        try:
            version = int(datos.get("version", 1) or 1)
        except (TypeError, ValueError):
            _log.warning(
                "la orden %r traia una version que no es numero (%r): se toma por la 1",
                datos.get("numero", ""), datos.get("version"),
            )
            version = 1
        # La misma pinza para version_origen, y el mismo criterio: si no se entiende se la
        # toma por 0, que es "no se sabe de donde viene". Inventarle un numero seria
        # inventar la procedencia de la orden.
        try:
            version_origen = int(datos.get("version_origen", 0) or 0)
        except (TypeError, ValueError):
            _log.warning(
                "la orden %r traia una version_origen que no es numero (%r): se toma por 0",
                datos.get("numero", ""), datos.get("version_origen"),
            )
            version_origen = 0
        if sobrantes:
            # INFO y no DEBUG: el logger raiz arranca en INFO. Es el unico aviso de que el
            # .json trae algo que esta version no esta mirando --lo escribio un programa
            # mas nuevo, o alguien lo toco a mano-- y antes eso dejaba la orden ilegible.
            _log.info(
                "la orden %r trae campos de diente que esta version no conoce y se "
                "ignoran: %s",
                datos.get("numero", ""), ", ".join(sorted(sobrantes)),
            )
        return cls(
            # Sin sanear, igual que nueva(): el saneado es para el nombre del archivo.
            numero=" ".join(str(datos.get("numero", "")).split()),
            nota_pedido=_una_linea(datos.get("nota_pedido", "")),
            cliente=_una_linea(datos.get("cliente", "")),
            abierta=str(datos.get("abierta", "")),
            cerrada=str(datos.get("cerrada", "")),
            sierras=sierras,
            sierra_actual=int(datos.get("sierra_actual", 1)),
            version=version,
            version_origen=version_origen,
        )

    # ------------------------------------------------------------------ lectura
    @property
    def total_dientes(self) -> int:
        return sum(s.dientes for s in self.sierras)

    @property
    def total_revisados(self) -> int:
        return sum(s.cuantos_revisados for s in self.sierras)

    @property
    def total_rechazados(self) -> int:
        return sum(s.rechazados for s in self.sierras)

    @property
    def total_salteados(self) -> int:
        return sum(s.cuantos_salteados for s in self.sierras)

    @property
    def total_pruebas(self) -> int:
        """Cuantos dientes de la orden NOMBRAN una foto de prueba. No mira el disco."""
        return sum(s.cuantas_pruebas for s in self.sierras)

    @property
    def completa(self) -> bool:
        return all(s.completa for s in self.sierras)

    def pruebas_nombradas(self) -> list[str]:
        """Las rutas de prueba que nombra la orden, en el orden del informe.

        Es lo que hay que ir a buscar al disco para saber cuantas de esas fotos estan de
        verdad. Puede repetirse una ruta si el .json se toco a mano; no se deduplica, porque
        lo que el informe cuenta son RENGLONES de diente.
        """
        return [d.prueba for s in self.sierras for d in s.revisados if d.prueba]

    def para_json(self) -> dict[str, Any]:
        """La orden lista para escribir, SIN el campo prueba de los dientes que no la tienen.

        POR QUE SE OMITE. Los tres .exe de respaldo guardados al lado de los datos arman
        cada diente con Diente(**d), asi que UNA clave de mas adentro de un diente les hace
        ilegible la ORDEN ENTERA --no el campo: la orden--. Medido con las 4 ordenes de
        verdad del taller: hoy las 4 se leen con el formato viejo, y con asdict() pelado
        --que escribe "prueba": "" en los 160 dientes-- quedaban 0 de 4 legibles en cuanto
        este programa las volvia a guardar, que es a cada diente que se anota.
        Omitiendo la prueba vacia, una orden sin pruebas queda con exactamente la misma
        forma que antes y el .exe viejo la sigue abriendo.
        Orden y Sierra no necesitan esto: el formato viejo las lee campo por campo con
        .get(), asi que version y version_origen no le molestan --verificado contra el
        desde_datos() del commit con que se armaron esos .exe--.
        Lo que NO se puede evitar es que una orden CON pruebas de verdad no la lea el .exe
        viejo: ahi la clave tiene que estar, porque es el dato.
        """
        datos = asdict(self)
        for sierra in datos.get("sierras", []):
            for diente in sierra.get("revisados", []):
                if not diente.get("prueba"):
                    diente.pop("prueba", None)
        return datos

    def sierra(self) -> Optional[Sierra]:
        """La sierra en la que se esta trabajando. sierra_actual es la POSICION, 1..N.

        Buscando por NUMERO, dos sierras con el mismo numero --un .json de afuera, o la app
        de afilados-- dejaban la tecla 's' girando sobre la primera para siempre:
        reproducido con numeros [1, 1, 2], seis 's' devuelven [1, 1, 1, 1, 1, 1] y los
        dientes de la segunda no se podian anotar nunca. Y con un sierra_actual fuera de
        rango (99), sierra() caia a la primera pero sierra_actual seguia en 99, asi que la
        primera 's' saltaba a la 2 y la sierra 1 quedaba sin visitar. Para una orden
        cargada a mano numero == posicion, asi que el .json guardado no cambia.
        """
        if not self.sierras:
            return None
        return self.sierras[min(max(int(self.sierra_actual), 1), len(self.sierras)) - 1]

    def diente(self) -> Optional[Diente]:
        """El diente que toca revisar ahora."""
        sierra = self.sierra()
        return sierra.proximo() if sierra is not None else None

    def resumen(self) -> str:
        """Una linea para la pantalla: donde esta parado el operario."""
        sierra = self.sierra()
        if sierra is None:
            return f"ORDEN {self.numero}"
        diente = sierra.proximo()
        cual = diente.numero if diente is not None else sierra.dientes
        return (
            f"ORDEN {self.numero}   -   sierra {sierra.numero} de {len(self.sierras)}"
            f"   -   diente {cual} de {sierra.dientes}"
        )

    # ------------------------------------------------------------------ cambios
    def anotar(
        self,
        estado: Optional[str] = None,
        parecido: Optional[float] = None,
        aprobado: Optional[bool] = None,
        origen: str = "",
        foto: str = "",
        nota: str = "",
        salteado: bool = False,
        prueba: str = "",
    ) -> Optional[Diente]:
        """Deja anotado el diente que tocaba y pasa al siguiente. None si ya estaban todos."""
        sierra = self.sierra()
        if sierra is None:
            return None
        diente = sierra.proximo()
        if diente is None:
            return None
        diente.estado = estado
        diente.parecido = None if parecido is None else float(parecido)
        diente.aprobado = aprobado
        diente.origen = str(origen)
        diente.foto = str(foto)
        diente.nota = str(nota)
        diente.salteado = bool(salteado)
        # La prueba se escribe aca junto con todo lo demas, aunque quien la guarda en el
        # disco es Ordenes.guardar_prueba(): asi el renglon del diente y la foto que lo
        # respalda quedan de un solo acto y no hay un estado intermedio donde el diente ya
        # este anotado y la prueba todavia no figure. Y se asigna siempre, tambien vacia,
        # para que un diente reanotado no se quede con la prueba del veredicto anterior.
        diente.prueba = str(prueba)
        diente.instante = _ahora()
        return diente

    def saltear(self, motivo: str = "salteado") -> Optional[Diente]:
        """Pasa de largo un diente, dejando dicho por que, y sigue con el siguiente.

        El renglon queda --un hueco explicado es un dato-- pero NO cuenta como revisado.
        """
        return self.anotar(nota=motivo, origen=_ORIGEN_SALTEADO, salteado=True)

    def siguiente_sierra(self) -> int:
        """Pasa a la sierra siguiente, dando la vuelta. Devuelve la POSICION en que quedo
        --1..N--, que para una orden cargada a mano es el numero de la sierra."""
        if self.sierras:
            posicion = min(max(int(self.sierra_actual), 1), len(self.sierras))
            self.sierra_actual = (posicion % len(self.sierras)) + 1
        return self.sierra_actual

    def cerrar(self) -> None:
        self.cerrada = _ahora()

    # ------------------------------------------------------------------ informe
    def _cuenta_de_pruebas(
        self, nombradas: list[str], encontradas: Optional[set[str]]
    ) -> str:
        """Una linea diciendo cuantas pruebas se NOMBRAN y cuantas se ENCONTRARON.

        LOS DOS NUMEROS, siempre, porque no son el mismo y confundirlos es mostrar como
        prueba algo que no existe: las ordenes 123 y 1234 del taller nombran 14 fotos entre
        las dos y hoy no hay ninguna viva --estan en la papelera de la coleccion, el
        operario borro ese concepto el 13/11--. Un informe que diga "14 fotos" sin decir
        que no estan es el papel que contesta un reclamo con pruebas que no se pueden
        mostrar.

        Se dice "encontradas" y no "en el disco" porque este mismo informe se escribe en
        dos lados y en cada uno hay que buscarlas en otra parte: el que queda al lado de la
        orden las busca en el deposito y el que viaja adentro del .zip las busca EN EL ZIP.

        Y cuando nadie las fue a buscar se dice ESO, no se da el nombrado por bueno.
        """
        cuantas = len(nombradas)
        if not cuantas:
            return "ninguna"
        if encontradas is None:
            return f"{cuantas} nombradas, no se fue a buscar si estan"
        estan = sum(1 for ruta in nombradas if ruta in encontradas)
        texto = f"{cuantas} nombradas, {estan} encontradas"
        if estan < cuantas:
            texto += f"   -   FALTAN {cuantas - estan}"
        return texto

    def informe(
        self,
        encontradas: Optional[set[str]] = None,
        huerfanas: Optional[int] = None,
    ) -> str:
        """El informe en castellano llano, para imprimir o mandar ante un reclamo.

        `encontradas` son las rutas de prueba cuya foto se encontro DE VERDAD, que es lo
        unico que la Orden no puede averiguar sola: aca adentro no hay con que mirar ninguna
        carpeta ni ningun zip. Para el informe que queda en el disco lo arma
        Ordenes.pruebas_en_disco(); para el que viaja en el reclamo lo arma exportar() con
        lo que metio adentro del paquete, que es lo que el otro lado va a poder abrir.

        Si no viene, el informe dice cuantas pruebas NOMBRA la orden y aclara que no se fue
        a buscar si estan. Nunca las da por presentes sin que alguien las haya visto.

        `huerfanas` son los archivos que estan en la carpeta de pruebas y que NINGUN diente
        nombra: una foto de un guardado que no llego a escribir el .json, o un .jpg.tmp de
        un corte de luz. No viajan en el paquete del reclamo --una foto que ningun renglon
        nombra parece la prueba de un diente y no es la de ninguno-- pero tampoco se borran,
        asi que el papel dice cuantas quedaron ahi. Lo cuenta Ordenes.pruebas_huerfanas();
        si no viene, o viene 0, el informe no dice nada: nadie fue a contarlas.
        """
        lineas = [
            "INFORME DE AFILADO",
            "=" * 72,
            f"Orden de afilado : {self.numero}",
            f"Nota de pedido   : {self.nota_pedido or '-'}",
            f"Cliente          : {self.cliente or '-'}",
            f"Abierta          : {self.abierta or '-'}",
            f"Cerrada          : {self.cerrada or 'sigue abierta'}",
            "",
            f"Sierras          : {len(self.sierras)}",
            f"Dientes          : {self.total_dientes} en total, "
            f"{self.total_revisados} revisados, {self.total_rechazados} para rechazar"
            + (f", {self.total_salteados} salteados" if self.total_salteados else ""),
            # LAS FOTOS DE PRUEBA, EN EL PAPEL Y NO SOLO EN EL LOG. Esto estaba unicamente
            # en el log --"la orden X se exporto SIN fotos de prueba"--, que lo ve el que
            # abre la carpeta del programa y no el que manda el reclamo por mail. Medido en
            # la orden de verdad 123: el zip salia con 7 renglones nombrando archivos de
            # entrenamiento, 0 fotos adentro y el informe.txt sin decirlo en ninguna parte.
            f"Fotos de prueba  : "
            f"{self._cuenta_de_pruebas(self.pruebas_nombradas(), encontradas)}",
        ]
        if huerfanas:
            # LO QUE QUEDO EN LA CARPETA Y EL PAPEL NO NOMBRA. No viaja en el paquete,
            # porque una foto que ningun diente nombra no es la prueba de ningun diente y
            # mostrarla como si lo fuera es lo que la regla del taller prohibe. Pero esta
            # ahi, la saco el microscopio, y no decirlo seria esconderla: puede ser la
            # unica foto de un diente que quedo de un guardado que se corto. Reproducido:
            # una orden de 2 pruebas con una huerfana y un .jpg.tmp al lado mandaba 4
            # archivos adentro de pruebas/ y el informe nombraba 2.
            # EN DOS RENGLONES Y SANGRADOS BAJO LA ETIQUETA: el informe se imprime a 72
            # columnas --es el ancho de las reglas de "=" de arriba y abajo-- y en uno solo
            # esto medido daba 102 caracteres, o sea que en el papel se partia solo y donde
            # cayera.
            # "1 fotos" no: el taller lee esto en el papel de un reclamo.
            cuantas = "1 foto" if huerfanas == 1 else f"{huerfanas} fotos"
            lineas += [
                f"                   en la carpeta hay {cuantas} que ningun diente de esta",
                "                   orden nombra: no viajan en el paquete ni se borran",
            ]
        lineas.append("")
        if not self.total_pruebas:
            # EN RENGLON APARTE Y EN MAYUSCULAS: el que lee el reclamo tiene que tropezarse
            # con esto antes de leer un solo veredicto, porque abajo hay dientes
            # RECHAZADOS y ni una foto con que mostrarlos.
            lineas += ["ESTA ORDEN NO TIENE NINGUNA FOTO DE PRUEBA", ""]
        for sierra in self.sierras:
            lineas.append("-" * 72)
            titulo = f"SIERRA {sierra.numero} de {len(self.sierras)}"
            if sierra.identificacion:
                titulo += f"   ({sierra.identificacion})"
            else:
                # EL PROGRAMA NO TIENE CON QUE IDENTIFICAR LA SIERRA, y eso se dice. Hoy
                # nada escribe Sierra.identificacion --no hay campo en el formulario ni
                # tecla para cargarla, y en los tres informes de verdad del taller la
                # columna salio vacia-- asi que lo unico que el informe sabe de "la sierra
                # 2" es que fue la segunda que el operario paso por el microscopio. Sin
                # esta linea, el que lee un reclamo sobre UNA sierra concreta toma ese
                # numero por una identificacion de la pieza.
                titulo += "   (sin identificar: es el orden en que se pasaron)"
            lineas.append(titulo)
            resumen = (
                f"  {sierra.dientes} dientes   -   {sierra.cuantos_revisados} revisados   "
                f"-   {sierra.rechazados} para rechazar   -   "
                f"{sierra.sin_decidir} sin decidir"
            )
            if sierra.cuantos_salteados:
                resumen += f"   -   {sierra.cuantos_salteados} salteados"
            lineas.append(resumen)
            lineas.append("")
            lineas.append("   diente  estado          resultado     parecido  cuando")
            for diente in sierra.revisados:
                if diente.salteado:
                    # Salteado NO es lo mismo que sin revisar, y el que lee un reclamo
                    # tiene que poder distinguirlos: uno es un diente que nadie llego a
                    # mirar y el otro es uno que el operario decidio pasar de largo, con
                    # el motivo escrito al lado.
                    motivo = f": {diente.nota}" if diente.nota else ""
                    lineas.append(
                        f"   {diente.numero:>6}  (salteado{motivo})   {diente.instante}"
                    )
                    continue
                if not diente.revisado:
                    lineas.append(f"   {diente.numero:>6}  (sin revisar)")
                    continue
                resultado = (
                    "APROBADO" if diente.aprobado is True
                    else "RECHAZADO" if diente.aprobado is False
                    else "sin decidir"
                )
                parecido = "" if diente.parecido is None else f"{diente.parecido * 100:.0f}%"
                lineas.append(
                    f"   {diente.numero:>6}  {(diente.estado or '-'):<15} {resultado:<13} "
                    f"{parecido:>8}  {diente.instante}"
                )
                # DE DONDE SALIO ESE RENGLON. Estaba solo en el .csv, y el .txt es el que
                # se imprime y se le entrega a un cliente que reclama: ahi se leia
                # "afilado APROBADO 66%" sin ninguna forma de saber que el estado lo habia
                # puesto el operario a mano y que el programa opinaba otra cosa.
                if diente.origen:
                    lineas.append(f"           {diente.origen}")
                if diente.nota:
                    lineas.append(f"           nota: {diente.nota}")
                # LA PRUEBA, AL LADO DEL VEREDICTO QUE RESPALDA. Si no esta nombrada ahi,
                # el que recibe el zip no tiene como saber cual de las fotos de la carpeta
                # pruebas/ corresponde a este diente. Y si el .json la nombra pero la foto
                # no vino, se dice aca mismo en vez de dejar un renglon que parece
                # respaldado.
                if diente.prueba:
                    falta = (
                        "" if encontradas is None or diente.prueba in encontradas
                        else "   (NO ESTA)"
                    )
                    lineas.append(f"           prueba: {diente.prueba}{falta}")
            # AL PIE DE LA SIERRA, los dos numeros otra vez pero de esta sierra sola: un
            # reclamo entra por UNA sierra, no por la orden entera.
            de_la_sierra = [d.prueba for d in sierra.revisados if d.prueba]
            lineas.append(
                "  fotos de prueba de esta sierra: "
                + self._cuenta_de_pruebas(de_la_sierra, encontradas)
            )
            lineas.append("")

        lineas += [
            "=" * 72,
            "COMO LEER ESTE INFORME",
            "-" * 72,
            "La columna 'estado' es en que quedo cada diente. Puede venir de dos lados y",
            "debajo de cada renglon se dice de cual: de lo que el programa tenia guardado",
            "de otros dientes iguales, o del operario, que lo archivo el mismo. Cuando el",
            "operario decidio y el programa opinaba otra cosa, ahi tambien se dice que",
            "opinaba.",
            "La columna 'parecido' dice CUANTO se parecio al estado que se le puso: un 60%",
            "es una decision al filo y un 95% no. Donde dice 'sin decidir', el programa no",
            "se animo a opinar y lo dice en vez de inventar un veredicto. Y donde el",
            "parecido esta VACIO es porque no hay un numero honesto que poner ahi: nadie",
            "midio cuanto se parecia ese diente al estado que figura al lado.",
            "",
            "El numero de SIERRA es el orden en que se pasaron por el microscopio, no una",
            "identificacion de la pieza: el programa todavia no tiene donde anotarla. Si al",
            "lado del titulo hay un nombre entre parentesis, ese si vino cargado.",
            "",
            "Los dientes sin revisar aparecen igual, en blanco: que falte uno es un dato.",
            "Un diente SALTEADO es otra cosa: el operario lo paso de largo a proposito y",
            "el motivo esta escrito al lado. No cuenta como revisado en ningun total.",
            "",
            "LAS FOTOS DE PRUEBA se cuentan de dos maneras y las dos figuran: cuantas",
            "NOMBRA la orden y cuantas estan de verdad en el disco. No son lo mismo. Una",
            "foto nombrada que no esta no prueba nada, y donde eso pasa el renglon del",
            "diente lo dice con un (NO ESTA) al lado. Si abajo de un diente no hay ningun",
            "renglon que diga 'prueba:', ese diente no tiene foto: el veredicto esta, pero",
            "no hay con que mostrarlo.",
            "",
            "Y OJO CON LA COLUMNA 'foto_de_entrenamiento' DE LA PLANILLA: eso NO es la",
            "prueba. Es el nombre de la foto que se guardo para que el programa aprenda, sin",
            "carpeta, asi que ni el programa la encuentra; puede estar borrada. La prueba es",
            "la columna 'prueba' y las fotos que viajan en la carpeta pruebas/.",
            "",
        ]
        return "\n".join(lineas)

    def filas_csv(self) -> list[list[str]]:
        """Una fila por diente, para abrir en Excel.

        LA COLUMNA DE LA FOTO DE ENTRENAMIENTO SE LLAMA POR SU NOMBRE, y la prueba tiene la
        suya. Antes la columna se llamaba "foto" y no habia ninguna con la prueba: el que
        abria el .csv del reclamo leia esos nombres como la foto del diente. Medido en la
        orden de verdad 123 exportada: 7 filas con foto=afilado_20261112_161511_098.jpg y
        parecidos, de los cuales 0 son archivos vivos --estan todos en coleccion/_papelera--
        y 0 fotos adentro del zip. Y al reves, en una orden con 3 pruebas de verdad
        guardadas, la columna "foto" salia VACIA en los 3 dientes: la unica columna que
        hablaba de fotos estaba vacia justo cuando SI habia prueba.
        """
        filas = [[
            "orden", "nota_pedido", "cliente", "sierra", "identificacion", "diente",
            "estado", "resultado", "parecido", "origen", "cuando",
            "foto_de_entrenamiento", "prueba", "nota",
        ]]
        for sierra in self.sierras:
            for diente in sierra.revisados:
                resultado = (
                    "salteado" if diente.salteado
                    else "APROBADO" if diente.aprobado is True
                    else "RECHAZADO" if diente.aprobado is False
                    else ("sin decidir" if diente.revisado else "sin revisar")
                )
                filas.append([_celda(v) for v in (
                    self.numero, self.nota_pedido, self.cliente,
                    str(sierra.numero), sierra.identificacion, str(diente.numero),
                    diente.estado or "", resultado,
                    "" if diente.parecido is None else f"{diente.parecido:.4f}",
                    diente.origen, diente.instante,
                    diente.foto, diente.prueba, diente.nota,
                )])
        return filas


@dataclass
class Resumen:
    """UNA FILA DEL HISTORIAL: lo que el panel necesita de una orden sin abrirla.

    Es un resumen y no la Orden entera porque la pantalla muestra ocho filas y no puede
    cargar ocho ordenes completas para eso. Lo que importa de los campos:

    - `numero` es el que tecleo el operario y `archivo` el nombre del .json. NO SON LO
      MISMO: una orden cargada como "OA/123" se guarda en OA_123.json. El historial lleva
      los dos porque el que busca su orden busca el numero que escribio, y el que la va a
      borrar o exportar necesita el archivo.
    - `con_prueba` son los dientes que tienen la foto de prueba EN EL DISCO, y
      `con_nombre_suelto` los que solo tienen el nombre pelado de una foto de la coleccion.
      Estan separados porque un nombre suelto NO ES PRUEBA y la pantalla tiene que decirlo:
      medido, la orden abierta 5113546 tiene 15 dientes anotados, los 15 con origen "la
      memoria, anotado solo" y CERO con foto guardada.
    - `ilegible` es el .json que esta pero no se pudo leer. Tiene fila igual: esconderla
      seria esconder justo la que hay que arreglar a mano.
    - `prueba_perdida` son los dientes que NOMBRAN una prueba cuya foto no aparecio en el
      disco. Es una tercera cosa, distinta de las dos de arriba, y antes no se contaba en
      ninguna columna: la fila quedaba IGUAL que la de una orden que nunca tuvo prueba
      --con_prueba=0, con_nombre_suelto=0-- y el zip salia sin las fotos sin avisar.
      Reproducido borrando data/pruebas/9001, que es lo que pasa en una limpieza de disco o
      cuando la carpeta Afilado se copia a otra PC sin los 29 MiB de fotos: con las 3 fotos
      en el disco la fila decia con_prueba=3, y sin ellas decia exactamente lo mismo que
      una orden sin pruebas.
    - `clave_estimada` es True cuando `clave` NO salio de la fecha de la orden sino del
      mtime del archivo. La fecha del archivo no es la fecha de la orden --se corre sola
      cada vez que alguien copia la carpeta-- y una columna de fechas que mezcla las dos sin
      decirlo esta mostrando una fecha que no es la fecha. Medido: a la 123 del taller se le
      vacio el campo abierta y su clave paso a ser la del archivo, que la mandaba ARRIBA en
      el historial como si fuera la orden mas nueva.
    """

    numero: str
    archivo: str
    cliente: str
    nota_pedido: str
    abierta: str
    cerrada: str
    sierras: int
    total_dientes: int
    revisados: int
    rechazados: int
    con_prueba: int
    con_nombre_suelto: int
    esta_abierta: bool
    ilegible: bool
    clave: str
    # LOS DOS CON DEFAULT, Y EL DEFAULT ES EL QUE NO AFIRMA NADA: una fila armada a mano
    # --las de las pruebas de pantalla-- no declara pruebas perdidas que no conto ni una
    # fecha estimada que no estimo. Quien las llena de verdad es _resumir(), que las mide.
    prueba_perdida: int = 0
    clave_estimada: bool = False


@dataclass
class Borrado:
    """COMO QUEDO UN BORRADO. Es lo que la pantalla necesita para decirle la verdad al
    operario, y una ruta pelada no alcanzaba.

    Antes borrar() devolvia la carpeta de la papelera, o None. Con None la pantalla decia
    "no se pudo borrar la orden" y eso era falso justo cuando mas importaba: medido con la
    orden 9002 y una foto trabada --lo que pasa si el operario la tiene abierta en el visor
    de Windows-- el aviso decia que no se habia borrado mientras el .json y los dos informes
    YA estaban en la papelera, el historial habia quedado vacio y exportar("9002") devolvia
    None. O sea: la orden desaparecida de la unica pantalla desde donde se la podia
    exportar, y el operario buscandola donde ya no estaba.

    - `destino` es la carpeta de la papelera, para poder nombrarla en pantalla.
    - `completo` es True solo si se movio TODO lo que habia.
    - `movidos` son los nombres de lo que se fue COMPLETO, para que el que mira el aviso
      sepa que esperar adentro de esa carpeta.
    - `falto` es lo que no se pudo mover, vacio cuando esta completo. Con esto la pantalla
      puede decir "quedo a medias, mire _papelera/<sello>" en vez de mentir.
    """

    destino: Path
    completo: bool
    movidos: list[str]
    falto: str


class Ordenes:
    """Las ordenes guardadas en disco, y cual es la que esta abierta."""

    def __init__(self, directorio: str = "data/ordenes") -> None:
        self._base = ruta_absoluta(directorio)
        self._abierta: Optional[Orden] = None
        self._fallo_al_guardar = False
        self._cargar_abierta()

    # ------------------------------------------------------------------ lectura
    @property
    def carpeta(self) -> Path:
        return self._base

    @property
    def abierta(self) -> Optional[Orden]:
        return self._abierta

    def listar(self) -> list[str]:
        if not self._base.is_dir():
            return []
        return sorted(p.stem for p in self._base.glob("*.json"))

    def ruta(self, numero: str) -> Path:
        return self._base / f"{sanear_numero(numero)}.json"

    def existe(self, numero: str) -> bool:
        """Si ya hay un archivo para esa orden, se pueda leer o no.

        Es distinto de `cargar(numero) is not None`, y la diferencia costaba una orden
        entera: cargar() devuelve None tanto cuando el archivo NO esta como cuando esta
        pero no se puede leer --un .json cortado por un corte de luz, por ejemplo-- y el
        control de duplicados, que usaba cargar(), daba via libre en el segundo caso. El
        programa abria la orden 1234 "nueva" y la escribia encima de la vieja, que era
        justo la que habia quedado rota y la unica copia del trabajo de ese dia.
        """
        return self.ruta(numero).is_file()

    def _stem(self, numero: str, archivo: str) -> Optional[str]:
        """COMO SE DIRECCIONA UNA ORDEN: por el nombre del archivo si vino, y si no por el
        numero saneado, igual que siempre. None si lo que vino no es un nombre de archivo.

        EL PROBLEMA QUE ARREGLA. El historial manda a borrar, exportar y retomar el NUMERO
        que esta adentro del .json, y aca adentro se le volvia a aplicar sanear_numero().
        Con un archivo que no se llama como su numero eso APUNTA A OTRO ARCHIVO. Y ese caso
        no es raro: "123 - copia.json" es lo que deja Windows al copiar una orden.
        ESA COPIA LA HACE EL OPERARIO CON EL EXPLORADOR, no porque el programa se lo pida:
        leida la pantalla de borrado, lo que dice es de que cliente es la orden, cuantos
        dientes tiene, que prueba se lleva, si es la orden abierta y que no se pierde
        porque va a data/ordenes/_papelera. NO le sugiere copiar nada --antes este
        comentario decia que si, y nadie lo habia ido a mirar--.
        Medido con las 4 ordenes del taller mas esa copia: la fila de la copia dice
        archivo='123 - copia' y numero='123', asi que borrar por el numero se llevaba
        123.json --la orden de verdad del cliente-- y dejaba la copia intacta.

        Con `archivo`, ese texto ES el stem del .json, tal cual, SIN sanear: el nombre salio
        del disco, no de un teclado. Lo que se mira es que sea un nombre y no una ruta.
        """
        if not archivo:
            return sanear_numero(numero)
        if not nombre_de_archivo_sano(archivo):
            # ERROR y no warning: alguien le paso una ruta a algo que esperaba un nombre de
            # archivo, y lo que sigue es borrar o exportar. Se corta aca.
            _log.error(
                "no se usa %r como nombre de archivo de la orden %s: una ruta no puede "
                "entrar por ahi",
                archivo, numero,
            )
            return None
        return archivo

    def cargar(self, numero: str, archivo: str = "") -> Optional[Orden]:
        """La orden guardada, o None si no esta o no se pudo leer.

        `archivo` es el nombre del .json sin la extension --Resumen.archivo-- y cuando viene
        manda sobre `numero`: ver _stem(). Vacio, todo sigue como siempre.
        """
        stem = self._stem(numero, archivo)
        if stem is None:
            return None
        archivo_json = self._base / f"{stem}.json"
        if not archivo_json.is_file():
            return None
        try:
            return Orden.desde_datos(
                json.loads(archivo_json.read_text(encoding="utf-8"))
            )
        except (OSError, ValueError, KeyError, TypeError, AttributeError) as error:
            # ATTRIBUTEERROR TAMBIEN. Un .json con forma valida pero distinta --"sierras"
            # como diccionario {"1": {...}}, una lista de textos, o el json entero siendo
            # un numero-- hacia que desde_datos() llamara .get() sobre un str, un list o un
            # int. Esa excepcion salia de Ordenes.__init__, que corre antes de que haya
            # pantalla, asi que el .exe no abria; y como el .abierta seguia apuntando a ese
            # archivo, no abria mas. Reproducido con los cuatro payloads.
            _log.warning("no se pudo leer la orden %s: %s", archivo_json, error)
            return None

    def historial(self) -> list[Resumen]:
        """Todas las ordenes de la carpeta, LA MAS NUEVA PRIMERO.

        Se ordena por `clave` --la fecha dada vuelta-- y no por el nombre del archivo, que
        es lo que devuelve listar(): medido, listar() da ['123', '1234', '213132',
        '5113546'], que es orden alfabetico y no tiene nada que ver con cuando se trabajo
        cada orden. El operario que busca "la de ayer" la busca arriba.

        NO EXPLOTA NUNCA. Una carpeta sin ordenes devuelve [], y una orden que no se puede
        leer aparece igual con ilegible=True y los contadores en cero: esta pantalla es
        desde donde se la va a exportar o mandar a la papelera, asi que es la ultima que
        puede permitirse esconderla.

        MEDIDO con las 4 ordenes de verdad del taller copiadas aparte: mediana 2,18 ms
        (p95 2,46 ms, max 3,04 ms) para las cuatro filas, contando los dientes con prueba y
        mirando el deposito. Escalado copiando esas mismas ordenes: 25 ordenes 12,5 ms, 100
        ordenes 54,8 ms. Esas tres vuelven a dar lo mismo hoy --2,6 ms las cuatro filas,
        15,4 ms las 25, 59,9 ms las 100-- porque ninguna de esas ordenes nombra prueba y la
        regla no corre.

        EL CASO GORDO SI SE PUSO MAS CARO Y ES POR LA REGLA DE DONDE ESTA LA PRUEBA: 5
        ordenes, una de 4 sierras de 80 dientes con las 320 pruebas en el disco, medido 34,6
        ms antes y 64,5 ms con la regla recien puesta. Lo que se agrego son 320 chequeos de
        que la foto este adentro de la carpeta de pruebas DE ESA ORDEN. Lo caro de eso era
        resolve(): hecho por foto daba 188,9 ms --medido: 74,7 ms los resolve() de las
        rutas y 65,2 ms los de la carpeta, contra 0,1 ms todos los chequeos de texto
        juntos-- asi que se hace por CARPETA, con un resolve() cacheado y un is_symlink()
        por foto. Los 30 ms de diferencia son eso, y son lo que impide que el historial
        pinte en verde "320 CON PRUEBA" por fotos que estan en otra parte.
        Y NO SUBIO al dejar de listar la carpeta y preguntar con el mismo is_file() que el
        zip --ver _hay_prueba()--: el mismo caso gordo en el mismo banco da 57,4 / 58,2 /
        57,8 ms contra 62,0 / 61,6 / 62,2 ms listando, o sea 4 ms menos. Listar no era mas
        barato porque iterdir() hace un stat por archivo igual.

        Esto se arma cuando el operario entra a la pantalla del historial, no en el medio de
        la revision: a 60 ms no se nota, y el dia que haya 400 ordenes conviene paginar la
        lectura, no adivinar.
        """
        if not self._base.is_dir():
            return []
        filas = [self._resumir(archivo) for archivo in sorted(self._base.glob("*.json"))]
        # Al reves: la clave ordena de vieja a nueva y la pantalla las quiere al contrario.
        filas.sort(key=lambda fila: fila.clave, reverse=True)
        return filas

    def carpeta_pruebas(self, numero: str, archivo: str = "") -> Optional[Path]:
        """Donde van las fotos de prueba de esa orden. AFUERA de la coleccion.

        Afuera porque la coleccion es lo que el programa mira para aprender y para
        autoevaluarse, y cada foto que entra ahi le cuesta tiempo a cada diente: medido,
        archivar() con 15 fotos tarda 209 ms y con 320 tarda 3.153 ms, contra 1.200 ms que
        el operario apoya el diente. El deposito de pruebas no lo mira nadie mas que el
        informe y el reclamo, asi que puede crecer todo lo que haga falta.

        SE NOMBRA POR EL ARCHIVO CUANDO EL ARCHIVO VIENE, igual que el .json. Para una orden
        que creo el programa los dos textos son el mismo --verificado en las 4 ordenes del
        taller: sanear_numero(numero) == el stem del .json en las 4, asi que no hay nada
        que migrar y no se mueve ninguna foto-- pero para una copia hecha a mano son
        distintos, y con la regla de antes borrar "123 - copia" se llevaba a la papelera las
        fotos de la orden 123, que es la del cliente.

        Devuelve None solo si `archivo` no es un nombre de archivo: ver _stem().

        Y ES LA UNICA CARPETA QUE CUENTA COMO PRUEBA DE ESTA ORDEN: _es_prueba_de() mide
        contra esta misma, asi que lo que la pantalla cuenta, lo que el zip lleva y lo que
        el borrado mueve salen todos de aca y no pueden discrepar.
        """
        stem = self._stem(numero, archivo)
        if stem is None:
            return None
        return self._deposito(stem)

    def _deposito(self, stem: str) -> Path:
        """La carpeta de pruebas de un stem ya mirado. Un solo lugar donde se arma el
        nombre, porque carpeta_pruebas(), _es_prueba_de(), el zip y el borrado tienen que
        hablar todos de la MISMA carpeta. Con el stem vacio queda "sin_numero", igual que
        antes: una orden sin numero usable tiene deposito, no se le pierde la foto."""
        return self._datos() / _CARPETA_PRUEBAS / (stem or "sin_numero")

    # ------------------------------------------------------------------ cambios
    def abrir(self, orden: Orden) -> Orden:
        """Deja esta orden como la que se esta trabajando y la guarda.

        SI NO LLEGA AL DISCO, NO SE ABRE. Antes se dejaba igual en memoria: el operario
        veia el cartel de la orden arriba, se pasaba la tarde anotando dientes contra ella
        y al cerrar el programa no quedaba nada, porque nunca hubo archivo. Ahora se vuelve
        a la que habia --que puede ser ninguna-- y quien llamo se entera por
        `fallo_al_guardar`, que es lo que mira el formulario para avisar en pantalla.
        """
        anterior = self._abierta
        self._abierta = orden
        if self.guardar() is None:
            self._abierta = anterior
            _log.error(
                "no se abrio la orden %s: no se pudo escribir en el disco", orden.numero
            )
            return orden
        _log.info(
            "orden %s abierta: %d sierras, %d dientes en total",
            orden.numero, len(orden.sierras), orden.total_dientes,
        )
        return orden

    def cerrar(self) -> Optional[Orden]:
        """Cierra la orden abierta --sin borrarla-- y deja de anotar en ella."""
        orden = self._abierta
        if orden is not None:
            orden.cerrar()
            self.guardar()
            _log.info("orden %s cerrada", orden.numero)
        self._abierta = None
        self._recordar_abierta(None)
        return orden

    def retomar(self, numero: str) -> Optional[Orden]:
        """Vuelve a dejar abierta una orden guardada que no se cerro. None si no se pudo.

        Hasta ahora solo se podia trabajar la orden marcada en .abierta: cargar() se usaba
        unicamente para rechazar numeros repetidos y listar() no se llamaba desde ninguna
        parte del programa. Una orden que dejaba de ser la abierta --porque se abrio otra--
        no se podia seguir anotando ni volver a informar.

        Una orden CERRADA no se retoma: su informe ya se imprimio y se entrego, y seguir
        anotando dientes encima deja el papel entregado diciendo menos de lo que el .json
        tiene.
        """
        orden = self.cargar(numero)
        if orden is None or orden.cerrada:
            return None
        self._abierta = orden
        self.guardar()
        _log.info(
            "orden %s retomada: %d de %d dientes",
            orden.numero, orden.total_revisados, orden.total_dientes,
        )
        return orden

    def guardar(self) -> Optional[Path]:
        """Escribe la orden abierta. Se llama despues de CADA diente anotado.

        Guardar de a un diente y no al final es a proposito: lo que este informe tiene que
        poder contestar es un reclamo, y un corte de luz en la mitad de una sierra no puede
        borrar las dos horas anteriores.

        HASTA DONDE LLEGA ESA PROMESA: se escribe un provisorio, se fuerzan sus datos al
        disco (flush + fsync) y recien ahi se renombra, asi que lo que queda es el .json
        anterior entero o el nuevo entero, nunca uno a medias. Lo que NO se puede forzar
        en Windows es el asiento del renombre en si --fsync de un directorio da
        PermissionError, probado-- asi que un corte en el instante justo del renombre
        todavia puede costar el ultimo diente. Los de antes estan.
        """
        orden = self._abierta
        if orden is None:
            return None
        # LA VERSION LA SUBE EL QUE ESCRIBE, antes de serializar. El archivo que esta por
        # quedar en el disco lo escribio ESTA version: es un hecho, no una promesa. Antes no
        # la subia nadie, y una orden del taller a la que se le anotaba un diente con prueba
        # de verdad quedaba en el disco diciendo version 1 Y con la prueba adentro.
        # version_origen NO se toca: es de donde vino la orden y eso no cambia al
        # reescribirla.
        orden.version = VERSION_FORMATO
        self._fallo_al_guardar = True       # hasta que el os.replace diga lo contrario
        archivo = self.ruta(orden.numero)
        provisorio = archivo.with_suffix(".json.tmp")
        try:
            self._base.mkdir(parents=True, exist_ok=True)
            with open(provisorio, "w", encoding="utf-8") as manejador:
                # para_json() y no asdict() pelado: omite el campo prueba de los dientes
                # que no la tienen, que es lo que deja la orden legible para los .exe de
                # respaldo. Medido: 4 de 4 ordenes del taller legibles con el formato viejo
                # antes y despues, contra 0 de 4 con asdict() pelado.
                manejador.write(
                    json.dumps(orden.para_json(), indent=2, ensure_ascii=False)
                )
                # LOS DATOS AL DISCO ANTES DEL RENOMBRE. Sin esto el renombre podia
                # asentarse antes que los datos y el 1234.json quedaba en bytes nulos:
                # reproducido ese estado, Ordenes(...).abierta vuelve None,
                # escribir_informe() devuelve (None, None) y los 40 dientes de las dos
                # horas anteriores no se recuperan de ninguna parte, porque se reescribe
                # siempre el mismo archivo y no hay copia generacional.
                # MEDIDO en el disco del taller, 30 escrituras del .json de una orden de
                # 2x40 dientes (26453 bytes): mediana 0.8 ms sin fsync, 1.2 ms con fsync.
                # Son 0.4 ms por diente anotado, contra los 59 ms que dura un cuadro a 17
                # fps: no se nota.
                manejador.flush()
                os.fsync(manejador.fileno())
            os.replace(provisorio, archivo)
        except OSError as error:
            _log.error("no se pudo guardar la orden %s: %s", orden.numero, error)
            # Y se limpia el provisorio. Si queda, en la carpeta de ordenes se van juntando
            # .json.tmp huerfanos --uno por cada intento fallido-- y el dia que alguien
            # entra a mirar que paso, lo que encuentra son archivos a medio escribir.
            try:
                provisorio.unlink(missing_ok=True)
            except OSError:
                pass
            return None
        self._fallo_al_guardar = False
        self._recordar_abierta(orden.numero)
        return archivo

    @property
    def fallo_al_guardar(self) -> bool:
        """Si el ultimo intento de guardar no llego al disco.

        Hace falta porque guardar() se llama desde adentro de abrir() y de cerrar(), que
        devuelven la orden y no la ruta: sin esto, el programa anunciaba "ORDEN 5678: 1
        sierra de 20 dientes" con el disco lleno y nada escrito.
        """
        return self._fallo_al_guardar

    def pruebas_en_disco(self, orden: Orden, archivo: str = "") -> set[str]:
        """De las pruebas que la orden NOMBRA, las que SON prueba de esta orden y estan.

        Es el dato que la Orden no puede averiguar sola --no sabe donde esta la carpeta
        data-- y sin el, el informe solo puede contar renglones. Las rutas que devuelve son
        las mismas que dice el .json, asi que se comparan tal cual.

        `archivo` es el nombre del .json sin extension, igual que en cargar() y borrar():
        es lo que dice de QUE carpeta de pruebas se trata. Vacio --que es como lo llama
        escribir_informe(), que recibe la orden y no el nombre del archivo-- la carpeta es
        la del numero saneado, que es donde guardar_prueba() escribe mientras se anota.

        Se pregunta por cada foto con el MISMO is_file() que usa el zip --ver
        _hay_prueba()-- asi que este informe, la fila del historial y el paquete del
        reclamo cuentan lo mismo. Medido en el historial completo con las 4 ordenes de
        verdad del taller, que es el mismo recorrido: mediana 2,18 ms las cuatro filas.
        """
        stem = self._stem(orden.numero, archivo)
        if stem is None:
            return set()
        return self._pruebas_con_el_stem(orden, stem)

    def pruebas_huerfanas(self, orden: Orden, archivo: str = "") -> Optional[int]:
        """Cuantos archivos hay en el deposito de esa orden que NINGUN diente nombra.
        None si no se pudo mirar la carpeta.

        QUE SON Y POR QUE SE CUENTAN. Una foto huerfana es una que guardar_prueba() ya
        escribio y cuyo .json despues no se pudo guardar --la foto va antes que el json, a
        proposito-- o un .jpg.tmp que quedo de un corte de luz. Ningun renglon del informe
        habla de ella, asi que adentro del paquete de un reclamo es una foto que parece
        prueba de un diente y no es la prueba de ninguno: justo lo que la regla del taller
        prohibe. Por eso NO VIAJA en el zip.
        Pero tampoco se borra: es dato del taller, la saco el microscopio y puede ser la
        unica foto de un diente que alguien tenga. Lo que se hace es DECIR cuantas hay, en
        el informe, para que el que arme el reclamo sepa que en esa carpeta quedo algo que
        el papel no nombra.

        Medido reproduciendo el caso: una orden de 2 pruebas de verdad con una huerfana y
        un .jpg.tmp al lado daba un zip con 4 archivos adentro de pruebas/ y 2 renglones
        "prueba:" en el informe. El .jpg.tmp viajaba y se contaba como foto.

        None y no 0 cuando la carpeta no se deja leer: 0 seria afirmar que no hay ninguna,
        y eso no se midio. Con la carpeta ausente si es 0: no hay archivos que nadie nombre.
        """
        stem = self._stem(orden.numero, archivo)
        if stem is None:
            return None
        carpeta = self._deposito(stem)
        if not carpeta.is_dir():
            return 0
        resueltas: dict[Path, Optional[Path]] = {}
        # Se comparan RUTAS ENTERAS y no nombres: si alguna vez una prueba cuelga de una
        # subcarpeta del deposito, dos archivos pueden llamarse igual y por nombre uno
        # taparia al otro. Las dos salen de self._datos(), asi que son comparables tal cual.
        nombradas = {
            ruta for ruta in (
                self._es_prueba_de(nombrada, stem, resueltas)
                for nombrada in orden.pruebas_nombradas()
            ) if ruta is not None
        }
        try:
            return sum(
                1 for hay in carpeta.iterdir()
                if hay.is_file()
                and hay not in nombradas
                and hay.name != _PAPEL_PROCEDENCIA
            )
        except OSError as error:
            _log.warning(
                "no se pudo mirar la carpeta de pruebas %s: %s", carpeta, error
            )
            return None

    def escribir_informe(self, orden: Optional[Orden] = None) -> tuple[Optional[Path], Optional[Path]]:
        """Deja el informe en texto y la planilla. Devuelve las dos rutas, cada una por
        separado: None en una de las dos es "esa mitad no salio".

        SE ESCRIBE A PROVISORIO Y SE RENOMBRA, igual que guardar(). Escribiendo directo
        encima, pedir el informe una segunda vez y que falle DESTRUIA el que ya estaba:
        reproducido con el disco lleno a la mitad del .csv --informe_1234.csv quedo en 241
        bytes, la cabecera y una fila, donde tenia 5411 y 41 filas-- y con el .csv
        bloqueado --el open(...,"w") ya habia truncado y quedo en 0 bytes--. Y devolviendo
        (None, None) cuando solo fallo la planilla, el programa no podia decir que el .txt
        si habia salido.
        """
        orden = orden or self._abierta
        if orden is None:
            return None, None
        # El numero se sanea PARA EL NOMBRE DEL ARCHIVO, igual que en ruta(): lo que se
        # imprime adentro del informe es el numero tal cual lo escribio el operario, y ese
        # puede traer una barra o un espacio.
        nombre = sanear_numero(orden.numero) or "sin_numero"
        texto: Optional[Path] = self._base / f"informe_{nombre}.txt"
        planilla: Optional[Path] = self._base / f"informe_{nombre}.csv"
        try:
            self._base.mkdir(parents=True, exist_ok=True)
            provisorio = texto.with_suffix(".txt.tmp")
            # Con las pruebas MIRADAS en el disco, no contadas del .json: el informe que
            # queda al lado de la orden tiene que decir lo mismo que el que viaja en el zip.
            # Y con las huerfanas contadas por la misma razon: si el deposito tiene fotos
            # que ningun diente nombra, el zip no se las lleva --no son prueba de nadie-- y
            # el papel que queda al lado de la orden tiene que decir que ahi quedo algo.
            provisorio.write_text(
                orden.informe(
                    self.pruebas_en_disco(orden), self.pruebas_huerfanas(orden)
                ),
                encoding="utf-8",
            )
            os.replace(provisorio, texto)
        except OSError as error:
            _log.error("no se pudo escribir el informe de %s: %s", orden.numero, error)
            try:
                texto.with_suffix(".txt.tmp").unlink(missing_ok=True)
            except OSError:
                pass
            texto = None
        try:
            self._base.mkdir(parents=True, exist_ok=True)
            provisorio = planilla.with_suffix(".csv.tmp")
            with open(provisorio, "w", encoding="utf-8-sig", newline="") as manejador:
                # utf-8-sig y punto y coma: es lo que abre bien el Excel en castellano.
                csv.writer(manejador, delimiter=";").writerows(orden.filas_csv())
            os.replace(provisorio, planilla)
        except OSError as error:
            _log.error("no se pudo escribir la planilla de %s: %s", orden.numero, error)
            try:
                planilla.with_suffix(".csv.tmp").unlink(missing_ok=True)
            except OSError:
                pass
            planilla = None
        if texto is not None:
            _log.info("informe de la orden %s en %s", orden.numero, texto)
        return texto, planilla

    def guardar_prueba(
        self, numero: str, sierra: int, diente: int, imagen: Any
    ) -> Optional[str]:
        """Guarda la foto que respalda a UN diente y devuelve la ruta que va en
        Diente.prueba, o None si no se pudo escribir.

        POR QUE SE GUARDA. Medido en la orden abierta de verdad 5113546: 15 dientes
        anotados y CERO con foto. Un informe de reclamo que dice "diente 7: RECHAZADO" y no
        tiene con que mostrarlo es la palabra del taller contra la del cliente.

        LA FOTO VA ANTES QUE EL JSON, y por eso esto devuelve la ruta en vez de escribirla:
        quien anota llama primero aca y recien despues guarda la orden. Un json que nombra
        una foto que no esta es una prueba que no existe --es lo que paso con las 14 fotos
        de las ordenes 123 y 1234--. Al reves no molesta: una foto de mas en el deposito no
        le miente a nadie.

        NO SE LE APLICA NI EL FILTRO DE NITIDEZ NI EL DE FOTO REPETIDA. La prueba es el
        cuadro que el programa JUZGO, salga lindo o salga movido: si estaba movido, eso es
        un dato del reclamo --explica el veredicto-- y no un motivo para quedarse sin
        prueba. Y el aviso de "ESTA FOTO YA LA GUARDASTE" es de la coleccion, donde una
        foto repetida ensucia el entrenamiento; aca dos dientes parecidos son dos pruebas
        distintas.

        MEDIDO en este disco, 320 pruebas seguidas con una foto de verdad del microscopio
        (640x480): mediana 2,75 ms, p95 5,09 ms, max 13,69 ms, comprimir incluido
        (cv2.imencode a calidad 95: mediana 0,82 ms). Contra los 1.200 ms que el operario
        apoya cada diente es el 0,23 % del tiempo: la prueba sale gratis. Lo que ocupa:
        esas 320 fotos pesaron 33,50 MiB en total --medido con la mas pesada de las 15
        reales, 109.768 bytes cada una, asi que es el peor caso; con la mediana de 95.648
        serian 29,2 MiB-- y una orden de 4 sierras de 80 dientes es exactamente eso. En el
        C: de esta PC hay 116,7 GiB libres.
        """
        if imagen is None:
            return None
        # cv2 se importa aca adentro: todo el resto del modulo es stdlib y se puede leer una
        # orden o armar un informe en una PC sin camara ni OpenCV instalado. Esto es lo
        # unico que necesita comprimir una imagen.
        import cv2

        try:
            cual_sierra = max(0, int(sierra))
            cual_diente = max(0, int(diente))
        except (TypeError, ValueError):
            # Se guarda igual con 0: la foto es la prueba y perderla por un numero raro en
            # el nombre del archivo seria cambiar el dato importante por el adorno.
            _log.warning(
                "la prueba de la orden %s venia con sierra %r y diente %r: se nombra con 0",
                numero, sierra, diente,
            )
            cual_sierra, cual_diente = 0, 0
        try:
            hecho, crudo = cv2.imencode(
                ".jpg", imagen, [int(cv2.IMWRITE_JPEG_QUALITY), _CALIDAD_PRUEBA]
            )
        except cv2.error as error:
            _log.error("no se pudo comprimir la prueba de la orden %s: %s", numero, error)
            return None
        if not hecho:
            _log.error("no se pudo comprimir la prueba de la orden %s", numero)
            return None

        # Sin `archivo`: la prueba se guarda mientras se anota, y la orden que se esta
        # anotando se escribe en ruta(orden.numero), asi que la carpeta de pruebas tiene que
        # llevar ese mismo nombre saneado. Nunca es None con el archivo vacio, pero se mira:
        # devolver una ruta armada sobre un None seria perder la foto en una carpeta rara.
        carpeta = self.carpeta_pruebas(numero)
        if carpeta is None:
            _log.error("no se guarda la prueba de la orden %s: no hay carpeta", numero)
            return None
        ahora = datetime.now()
        sello = ahora.strftime("%Y%m%d_%H%M%S_") + f"{ahora.microsecond // 1000:03d}"
        nombre = f"s{cual_sierra:02d}_d{cual_diente:03d}_{sello}.jpg"
        destino = carpeta / nombre
        # Con milisegundos en el sello dos pruebas no deberian chocar, pero si chocan no se
        # pisa ninguna: las dos son la prueba de un diente.
        repeticion = 1
        while destino.exists():
            destino = carpeta / f"{destino.stem}_{repeticion}.jpg"
            repeticion += 1
        provisorio = destino.with_suffix(".jpg.tmp")
        try:
            carpeta.mkdir(parents=True, exist_ok=True)
            with open(provisorio, "wb") as manejador:
                manejador.write(crudo.tobytes())
                # EL MISMO BAILE QUE guardar(): los datos al disco (flush + fsync) y recien
                # ahi el renombre. Es por el corte de luz que ya costo un archivo: asi lo
                # que queda es la foto entera o ninguna foto, nunca media foto que el
                # informe nombra como prueba.
                manejador.flush()
                os.fsync(manejador.fileno())
            os.replace(provisorio, destino)
        except OSError as error:
            _log.error(
                "no se pudo guardar la prueba del diente %s de la orden %s: %s",
                cual_diente, numero, error,
            )
            try:
                provisorio.unlink(missing_ok=True)
            except OSError:
                pass
            return None
        # Relativa a la carpeta data y con barras normales, armada a mano y no con
        # os.path.relpath: la ruta se escribe en el .json y tiene que leerse igual en la PC
        # del taller y en la que abra la carpeta copiada.
        return f"{_CARPETA_PRUEBAS}/{carpeta.name}/{destino.name}"

    def borrar(self, numero: str, archivo: str = "") -> Optional[Borrado]:
        """Manda la orden a la papelera y devuelve COMO quedo, o None si no se movio nada.

        NO BORRA DE VERDAD, igual que cuando se elimina un concepto: mueve todo a
        _papelera/<fecha y hora>/. Lo que se va es el .json, los informes que esten al lado
        y la carpeta de pruebas. Una orden es lo que contesta un reclamo: si el operario se
        equivoca de fila, tiene que haber de donde sacarla de vuelta.

        `archivo` es el nombre del .json sin extension --Resumen.archivo-- y cuando viene
        manda sobre `numero`: ver _stem(). Con el numero pelado, borrar la fila de
        "123 - copia.json" se llevaba 123.json, la orden del cliente.

        EL ORDEN DE LOS MOVIMIENTOS ES AL REVES QUE ANTES, Y ES LO IMPORTANTE DE ESTA
        FUNCION. El .json iba primero "porque es el unico que no se puede volver a
        generar", y medido eso sale peor: si despues falla el movimiento de una foto, el
        .json ya esta en la papelera, la orden desaparecio del historial --la unica pantalla
        desde donde se la podia exportar-- y al operario se le dice "no se pudo borrar".
        Reproducido con la orden 9002 y 3 pruebas, trabando una foto con msvcrt.locking, que
        es lo que pasa si el operario la tiene abierta en el visor de Windows: aviso "no se
        pudo borrar", 0 filas en el historial, exportar("9002") devolviendo None, y el .json
        y los dos informes ya en _papelera/.

        Ahora van primero las pruebas, despues los informes, el marcador .abierta se suelta
        en el renglon de antes y el .json AL FINAL. Asi un borrado a medias deja la orden
        ENTERA Y VISIBLE en el historial: el .json es lo ultimo que se va, y mientras este,
        la orden se puede exportar. Lo que puede quedar partido es la prueba, y de eso avisa
        el informe, que cuenta las fotos que la orden nombra contra las que se encuentran.

        EL MARCADOR .abierta SE SUELTA JUSTO ANTES DEL .json y no al principio: si se
        suelta al principio y el borrado se traba, el operario se queda sin la orden abierta
        --dejo de poder anotar dientes-- por una orden que al final sigue en el disco.
        """
        stem = self._stem(numero, archivo)
        if stem is None:
            return None
        archivo_json = self._base / f"{stem}.json"
        if not archivo_json.is_file():
            # Tampoco se mira si se puede LEER: una orden ilegible es justamente una que el
            # operario puede querer sacar del medio.
            _log.error("no se borra la orden %s: no hay archivo %s", numero, archivo_json)
            return None
        # LA MISMA CARPETA DE UN INTENTO ANTERIOR SI HAY UNA, Y SI NO UNA NUEVA (ver
        # _papelera_de_un_intento_anterior): si no, el reintento parte la orden en dos.
        destino = self._papelera_de_un_intento_anterior(stem) or self._papelera_libre()
        pruebas = self.carpeta_pruebas(numero, archivo)
        movidos: list[str] = []
        yendo = ""
        try:
            destino.mkdir(parents=True, exist_ok=True)
            if pruebas is not None and pruebas.is_dir():
                # CON EL NIVEL DEL NUMERO DE ORDEN ADENTRO: pruebas/5113546/ y no pruebas/
                # pelada. Sin ese nivel, el rescate que haria cualquiera con el Explorador
                # --copiar la carpeta de vuelta a data-- deja las fotos un escalon arriba de
                # donde el .json las nombra. Medido: tras el rescate, con_prueba=0 de 3 y
                # las 3 rutas del .json dando existe=False. La prueba estaba y no se
                # encontraba, y nada avisaba que faltaba un nivel de carpeta.
                yendo = f"{_CARPETA_PRUEBAS}/{pruebas.name}"
                self._mover_deposito(pruebas, destino / _CARPETA_PRUEBAS / pruebas.name)
                movidos.append(yendo)
            for suelto in (
                self._base / f"informe_{stem}.txt",
                self._base / f"informe_{stem}.csv",
            ):
                if suelto.is_file():
                    yendo = suelto.name
                    shutil.move(str(suelto), str(destino / suelto.name))
                    movidos.append(yendo)
            yendo = archivo_json.name
            shutil.move(str(archivo_json), str(destino / archivo_json.name))
            movidos.append(yendo)
            # EL MARCADOR .abierta SE SUELTA DESPUES DE QUE EL .json SE MOVIO DE VERDAD, no
            # en el renglon de antes. El .json es el ultimo de la fila --a proposito-- y es
            # justo el unico que el operario puede tener abierto en el Bloc de notas, que
            # es lo que lo traba. Reproducido trabando 9200.json con msvcrt.locking: el
            # borrado devolvia completo=False con falto='9200.json', el .json SEGUIA en el
            # disco, y aun asi la orden abierta quedaba en None y el .abierta en "". O sea:
            # el operario perdia la orden en curso --dejaba de poder anotar dientes-- por
            # una orden que al final no se borro. Ahora se suelta recien cuando el
            # movimiento salio bien; si se traba, el shutil.move de arriba lanza y esto no
            # corre.
            if self._abierta is not None and sanear_numero(self._abierta.numero) == stem:
                self._abierta = None
                self._recordar_abierta(None)
        except OSError as error:
            # NO SE DESHACE NADA. Lo que ya se movio esta en la papelera y el log dice
            # donde: devolverlo a su lugar es otro movimiento que tambien puede fallar y
            # ahi si se perderia algo.
            _log.error(
                "el borrado de la orden %s quedo a medias: no se pudo mover %s (%s). Lo "
                "que si se fue a %s: %s",
                numero, yendo, error, destino, ", ".join(movidos) or "nada",
            )
            # SE MIRA LA PAPELERA, no se supone: shutil.move de una carpeta puede copiar
            # unos archivos y fallar en el siguiente --medido, 2 de 3 fotos quedaron del
            # otro lado-- asi que "no se movio nada" se pregunta al disco.
            if not any(p.is_file() for p in destino.rglob("*")):
                # Nada llego: se saca la carpeta de papelera vacia, que si no se van
                # juntando una por cada intento fallido, y se dice que no se borro. Por aca
                # pasa tambien el unico caso en que el mkdir de arriba es el que falla, y
                # por eso `falto` no sale nunca vacio con completo=False: si no se alcanzo a
                # intentar ningun movimiento, tampoco hay nada en la papelera.
                shutil.rmtree(destino, ignore_errors=True)
                return None
            return Borrado(
                destino=destino, completo=False, movidos=movidos, falto=yendo
            )
        _log.info(
            "orden %s mandada a la papelera: %s en %s",
            numero, ", ".join(movidos), destino,
        )
        return Borrado(destino=destino, completo=True, movidos=movidos, falto="")

    def exportar(
        self, numero: str, destino: Optional[Path] = None, archivo: str = ""
    ) -> Optional[Path]:
        """Arma UN SOLO .zip con todo lo de esa orden y devuelve donde quedo, o None.

        UN SOLO ARCHIVO porque es lo que el operario copia a un pendrive o manda por mail
        cuando entra un reclamo: adentro van la orden.json, el informe en texto, la
        planilla y la carpeta pruebas/ con las fotos. Tres archivos y una carpeta sueltos se
        mandan a medias. zipfile viene con Python, asi que esto no agrega nada que instalar.

        EL INFORME SE ARMA AL VUELO desde la orden que se acaba de leer, no se copia el
        informe_<numero>.txt del disco: ese puede ser de otra corrida --de antes de los
        ultimos diez dientes-- y entonces el papel del reclamo diria menos de lo que el
        .json guarda.

        SI LA ORDEN ES ILEGIBLE DEVUELVE None. No se exporta como prueba algo que el
        programa no pudo leer.

        `archivo` es el nombre del .json sin extension --Resumen.archivo-- y cuando viene
        manda sobre `numero`: ver _stem().

        LAS FOTOS VAN ADENTRO DEL ZIP EN LA MISMA RUTA QUE EL .json LES DA. Antes viajaban
        en "pruebas/" pelada mientras el .json decia "pruebas/<orden>/", asi que ninguna de
        las rutas del documento resolvia: medido con una orden de 3 pruebas, 3 dientes
        dando NO ESTA y 0 de 3 rutas resolviendo. Las fotos viajaban igual, pero el papel
        que las nombra apuntaba a una carpeta que no existe adentro del paquete, y el que
        recibe el reclamo concluye que las pruebas no vinieron.

        Y LAS QUE VIAJAN SON LAS QUE LOS DIENTES NOMBRAN, no lo que hay en la carpeta. Una
        foto que ningun diente nombra --una huerfana de un guardado cuyo .json no se pudo
        escribir, un .jpg.tmp de un corte de luz-- NO entra: ningun renglon del informe
        habla de ella, asi que del otro lado parece la prueba de un diente y no es la de
        ninguno. No se borra, es dato del taller; el informe dice cuantas quedaron.
        Se miden con la MISMA regla con la que el historial cuenta la columna de pruebas
        --_es_prueba_de()-- asi que la fila que el operario vio y lo que el paquete lleva
        son la misma cuenta: no hace falta ningun aviso que avise que no coinciden.

        EL NOMBRE DEL ZIP BUSCA UN LUGAR LIBRE cuando lo armamos nosotros, igual que la
        papelera: el sello tiene resolucion de un segundo y exportar() tarda milisegundos,
        asi que dos 'e' seguidas caian en el mismo nombre y la segunda pisaba el paquete de
        la primera mientras el cartel decia que se habia escrito uno nuevo. Si el que llama
        pasa el nombre exacto del .zip, se respeta tal cual: ahi la decision es suya.

        MEDIDO: la orden de verdad 5113546 --15 dientes anotados, ninguna prueba-- 17,2 ms
        y 2.864 bytes de zip. Una orden de 4 sierras de 80 dientes con las 320 fotos de
        prueba adentro: 1.220 ms y 33,51 MiB, o sea que el zip no comprime nada --el .jpg
        ya viene comprimido-- y lo que pesa es la prueba. Esos 1,2 s pasan con el operario
        esperando en la pantalla del historial, no en el medio de una sierra, y entran en un
        pendrive y en un mail.
        Vuelto a medir despues de pasar las fotos por la regla, en este disco: la 5113546
        sale en 5,0 ms y 3.275 bytes --los 2.864 de antes son de cuando el informe.txt tenia
        menos texto, no del zip-- y una orden de 4x80 con las 320 fotos en 960 ms y 29,76
        MiB; esas 320 son tres fotos del microscopio repetidas, asi que el tamano no es
        comparable con el de 320 distintas, pero el tiempo si: recorrer las rutas que los
        dientes nombran no sale mas caro que recorrer la carpeta.
        """
        stem = self._stem(numero, archivo)
        if stem is None:
            return None
        orden = self.cargar(numero, archivo)
        if orden is None:
            _log.error(
                "no se exporta la orden %s: el archivo no esta o no se pudo leer", numero
            )
            return None
        saneado = stem or sanear_numero(orden.numero) or "sin_numero"
        sello = datetime.now().strftime("%Y%m%d_%H%M%S")
        nuestro = True
        if destino is None:
            final = self._base / _CARPETA_EXPORTADO / f"orden_{saneado}_{sello}.zip"
        else:
            final = Path(destino)
            # Si lo que vino es una carpeta --existe como carpeta, o no termina en .zip--
            # se le pone adentro el nombre de siempre. Asi el que llama puede pasar
            # simplemente el pendrive.
            if final.is_dir() or final.suffix.lower() != ".zip":
                final = final / f"orden_{saneado}_{sello}.zip"
            else:
                # El nombre exacto lo eligio el que llama: se respeta tal cual, incluso si
                # pisa algo. Ahi la decision no es nuestra.
                nuestro = False
        if nuestro:
            # UN NOMBRE LIBRE, EL MISMO BUCLE QUE _papelera_libre() Y guardar_prueba(). El
            # sello tiene resolucion de un segundo y exportar() tarda mucho menos que eso:
            # medido en este disco, una orden de 3 pruebas sale en 11,9 ms y el segundo zip
            # en 12,7 ms, y la orden de verdad 5113546 --15 dientes, sin pruebas-- entre
            # 3,1 y 15,7 ms (mediana 10,3). O sea que dos 'e' seguidas caen en el MISMO
            # segundo sin ninguna dificultad. Reproducido: los dos zips salian con el
            # nombre orden_9400_20261116_161040.zip, quedaba uno solo en la carpeta y el
            # cartel decia igual que se habia escrito uno nuevo. El zip de un reclamo es lo
            # que se manda por mail: pisar el anterior sin decirlo es perder el paquete que
            # el operario creia tener.
            raiz = final.with_suffix("")
            repeticion = 1
            while final.exists():
                final = raiz.with_name(f"{raiz.name}_{repeticion}").with_suffix(".zip")
                repeticion += 1
        provisorio = final.with_suffix(".zip.tmp")
        # Las fotos van PRIMERO y el informe DESPUES, aunque adentro del zip queden en el
        # orden que sea. Es para que el informe pueda decir lo que el paquete lleva de
        # verdad y no lo que hay en el disco de esta PC: son dos cosas distintas --una copia
        # hecha a mano nombra las fotos de la orden original, que estan en el disco pero NO
        # en su paquete-- y el que abre el reclamo del otro lado solo tiene el paquete.
        adentro: set[str] = set()
        cuantas = 0
        try:
            final.parent.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(
                provisorio, "w", compression=zipfile.ZIP_DEFLATED
            ) as paquete:
                # El .json se copia tal cual esta en el disco: ES el documento. Volviendolo
                # a serializar desde la orden leida, lo que viaja en el zip es lo que este
                # programa entendio y no lo que el taller tiene guardado, y si alguna vez
                # difieren --un campo que esta version no conoce-- el que manda el reclamo
                # no tiene como darse cuenta.
                paquete.write(self._base / f"{stem}.json", "orden.json")
                # LAS FOTOS QUE LOS DIENTES NOMBRAN, NO LO QUE HAY EN LA CARPETA. Antes se
                # recorria pruebas.iterdir(), asi que viajaba todo lo que estuviera ahi:
                # reproducido con una orden de 2 pruebas de verdad, una foto huerfana --de
                # un guardado cuyo .json no se pudo escribir-- y un .jpg.tmp olvidado de un
                # corte de luz, el zip salia con 4 archivos adentro de pruebas/ y el informe
                # nombraba 2. Esas dos de mas llevan el numero de diente en el nombre y
                # ningun renglon del papel habla de ellas: del otro lado parecen la prueba
                # de un diente y no son la prueba de ninguno, que es justo lo que no se
                # puede hacer. No se borran --son dato del taller-- y el informe dice
                # cuantas quedaron en la carpeta.
                #
                # Y se pasan por la MISMA regla que cuenta la pantalla --_es_prueba_de()--
                # asi que lo que el zip lleva y lo que la fila dice son la misma cuenta, no
                # dos parecidas.
                resueltas: dict[Path, Optional[Path]] = {}
                for ruta in orden.pruebas_nombradas():
                    if ruta in adentro:
                        # El .json puede nombrar la misma foto en dos dientes si alguien lo
                        # toco a mano. El informe cuenta RENGLONES y ahi se repite; adentro
                        # del zip no se puede escribir dos veces el mismo nombre.
                        continue
                    foto = self._es_prueba_de(ruta, stem, resueltas)
                    if foto is None:
                        _log.warning(
                            "%s.json nombra como prueba %r, que esta afuera de su carpeta "
                            "de pruebas (%s): no entra al paquete del reclamo",
                            stem, ruta, self._deposito(stem),
                        )
                        continue
                    if not foto.is_file():
                        continue
                    # La ruta de adentro del zip es la MISMA que dice Diente.prueba, tal
                    # cual: "pruebas/<carpeta de la orden>/<foto>". Si el .json nombra una
                    # ruta y el paquete la pone en otra, el documento no sirve. Por eso la
                    # regla no deja pasar ni "\\" ni tramos "." --zipfile les cambiaria el
                    # nombre adentro del paquete y el renglon del informe no cerraria--.
                    paquete.write(foto, ruta)
                    adentro.add(ruta)
                    cuantas += 1
                # Y EL PAPEL QUE EXPLICA DE DONDE SALIERON ESAS FOTOS, si lo hay. Es la
                # unica excepcion a "solo viaja lo que un diente nombra", y no la rompe: lo
                # que esa regla impide es mandar una FOTO que parezca la prueba de un diente
                # sin serlo, y este archivo no puede confundirse con la prueba de ninguno.
                # Sin el, el que recibe el reclamo de la orden 123 abre siete fotos sin
                # ninguna manera de saber que no se sacaron en el momento de juzgar el
                # diente, que es justo lo que no se puede dejar pasar. Ver _PAPEL_PROCEDENCIA.
                papel = self._deposito(stem) / _PAPEL_PROCEDENCIA
                if papel.is_file():
                    paquete.write(
                        papel,
                        f"{_CARPETA_PRUEBAS}/{self._deposito(stem).name}/"
                        f"{_PAPEL_PROCEDENCIA}",
                    )
                # Y el informe cuenta contra lo que ENTRO al paquete, no contra el disco.
                # Las huerfanas se cuentan igual, porque quedaron del lado de aca.
                paquete.writestr(
                    "informe.txt",
                    orden.informe(adentro, self.pruebas_huerfanas(orden, archivo)),
                )
                planilla = io.StringIO(newline="")
                csv.writer(planilla, delimiter=";").writerows(orden.filas_csv())
                # utf-8-sig y punto y coma, igual que escribir_informe(): es lo que abre
                # bien el Excel en castellano.
                paquete.writestr("informe.csv", planilla.getvalue().encode("utf-8-sig"))
            os.replace(provisorio, final)
        except (OSError, ValueError, zipfile.BadZipFile) as error:
            _log.error("no se pudo exportar la orden %s: %s", numero, error)
            try:
                provisorio.unlink(missing_ok=True)
            except OSError:
                pass
            return None
        if not cuantas:
            # Se dice, no se esconde: un zip sin fotos sigue siendo el informe, pero el que
            # lo manda tiene que saber que no lleva con que mostrar ningun diente. Y ahora
            # tambien lo dice el informe.txt de adentro, que es lo que el otro lado abre.
            _log.info("la orden %s se exporto SIN fotos de prueba: no hay ninguna guardada",
                      numero)
        resuelven = sum(1 for ruta in orden.pruebas_nombradas() if ruta in adentro)
        if resuelven < orden.total_pruebas:
            # Que el .json nombre mas pruebas de las que el paquete lleva no se esconde, y
            # ya no queda solo aca: el informe.txt de adentro lo dice tambien, con el
            # (NO ESTA) al lado del diente. Esto es para el que mira el log del programa.
            _log.warning(
                "la orden %s nombra %d fotos de prueba y el paquete lleva %d de esas "
                "(%d fotos en total): el informe.txt de adentro lo dice",
                numero, orden.total_pruebas, resuelven, cuantas,
            )
        _log.info(
            "orden %s exportada en %s (%d fotos de prueba)", numero, final, cuantas
        )
        return final

    # ------------------------------------------------------------------ internos
    def _datos(self) -> Path:
        """La carpeta data, donde cuelgan las ordenes. El deposito de pruebas va ahi al
        lado, y por eso Diente.prueba se guarda relativo a esta: la carpeta Afilado entera
        se copia de una PC a otra y adentro del .json no puede haber una ruta con una letra
        de unidad."""
        return self._base.parent

    def _resumir(self, archivo: Path) -> Resumen:
        """Una fila del historial a partir de UN .json. No lanza: devuelve la fila
        ilegible."""
        try:
            cuando_el_archivo = archivo.stat().st_mtime
        except OSError:
            cuando_el_archivo = 0.0
        # Se lee el archivo que se encontro, no cargar(stem): cargar() vuelve a sanear el
        # nombre, y un .json que alguien copio a mano con un caracter raro apuntaria a otro
        # archivo --o a ninguno-- y la orden saldria ilegible sin serlo.
        orden: Optional[Orden] = None
        try:
            orden = Orden.desde_datos(json.loads(archivo.read_text(encoding="utf-8")))
        except (OSError, ValueError, KeyError, TypeError, AttributeError) as error:
            # La misma lista de excepciones que cargar(), y por lo mismo: un .json con
            # forma valida pero distinta hace que desde_datos() llame .get() sobre un texto
            # o un numero.
            _log.warning("la orden %s no se pudo leer para el historial: %s", archivo, error)
        if orden is None:
            # Sin fecha de adentro, la unica medible es la del archivo, y se declara como
            # lo que es: de un .json que no se pudo leer no sale ninguna fecha de orden.
            del_archivo = clave_de_fecha("", cuando_el_archivo)
            return Resumen(
                numero=archivo.stem, archivo=archivo.stem, cliente="", nota_pedido="",
                abierta="", cerrada="", sierras=0, total_dientes=0, revisados=0,
                rechazados=0, con_prueba=0, con_nombre_suelto=0, esta_abierta=False,
                ilegible=True,
                clave=del_archivo,
                prueba_perdida=0,
                clave_estimada=bool(del_archivo),
            )
        con_prueba = 0
        con_nombre_suelto = 0
        prueba_perdida = 0
        resueltas: dict[Path, Optional[Path]] = {}
        for sierra in orden.sierras:
            for diente in sierra.revisados:
                if diente.prueba:
                    # CON LA CARPETA DE ESTE ARCHIVO, no del numero de adentro: archivo.stem
                    # es el nombre del .json que se esta resumiendo. Por eso la fila de una
                    # copia hecha a mano ya no promete las fotos de la orden original.
                    # Medido con "123 - copia.json" al lado de "123.json" y las 3 fotos en
                    # data/pruebas/123: la fila de la copia decia con_prueba=3 y en verde,
                    # mientras su zip salia de 2.218 bytes con 0 fotos --el de la orden de
                    # verdad, 294.739-- y su borrado se llevaba 0 fotos diciendo que se
                    # llevaba las 3.
                    if self._hay_prueba(diente.prueba, archivo.stem, resueltas):
                        con_prueba += 1
                    else:
                        # LA PRUEBA QUE EL .json NOMBRA Y NO ESTA --o que esta afuera de la
                        # carpeta de esta orden, que para el reclamo es lo mismo: no es su
                        # prueba--. Antes no caia en ninguna columna: la fila quedaba igual
                        # que la de una orden que nunca tuvo prueba y el zip salia sin las
                        # fotos sin decirlo. Es una tercera cosa y se cuenta aparte; sale
                        # gratis, es el mismo recorrido.
                        prueba_perdida += 1
                elif diente.foto:
                    # SOLO EL NOMBRE PELADO: no es prueba y se cuenta aparte. Es lo que
                    # tienen las ordenes viejas y lo que la pantalla tiene que declarar como
                    # lo que es.
                    con_nombre_suelto += 1
        abierta_ahora = (
            self._abierta is not None
            and sanear_numero(self._abierta.numero) == archivo.stem
        )
        # DE DONDE SALIO LA FECHA DE LA FILA. Si el texto de la orden se entiende, la clave
        # es esa fecha; si no, cae al mtime del archivo, que NO es la fecha de la orden --se
        # corre sola cada vez que alguien copia la carpeta Afilado a otra PC-- y hasta ahora
        # nada lo decia. Medido con la 123 del taller a la que se le vacio el campo abierta:
        # la clave paso a ser la del archivo y la fila se fue ARRIBA del historial, haciendo
        # pasar el mtime por la orden mas nueva del taller.
        del_texto = clave_de_fecha(orden.abierta)
        clave = del_texto or clave_de_fecha("", cuando_el_archivo)
        return Resumen(
            numero=orden.numero or archivo.stem,
            archivo=archivo.stem,
            cliente=orden.cliente,
            nota_pedido=orden.nota_pedido,
            abierta=orden.abierta,
            cerrada=orden.cerrada,
            sierras=len(orden.sierras),
            total_dientes=orden.total_dientes,
            revisados=orden.total_revisados,
            rechazados=orden.total_rechazados,
            con_prueba=con_prueba,
            con_nombre_suelto=con_nombre_suelto,
            esta_abierta=abierta_ahora,
            ilegible=False,
            clave=clave,
            prueba_perdida=prueba_perdida,
            clave_estimada=not del_texto and bool(clave),
        )

    def _es_prueba_de(
        self,
        relativa: str,
        stem: str,
        resueltas: Optional[dict[Path, Optional[Path]]] = None,
    ) -> Optional[Path]:
        """DONDE ESTA LA PRUEBA, UNA SOLA IDEA: la prueba de un diente es un archivo que
        vive ADENTRO de la carpeta de pruebas DE ESE ARCHIVO DE ORDEN. Nada mas es prueba.

        Devuelve por donde hay que ir a buscar la foto, o None si lo que el .json nombra no
        es una prueba de esta orden. None no quiere decir "no esta": quiere decir "esto no
        es prueba", y quien llama lo cuenta como prueba perdida.

        POR QUE HACE FALTA MIRARLO. Antes esto resolvia la ruta del .json contra data y con
        eso alcanzaba, asi que valia cualquier cosa. Medido: un diente con
        prueba="pruebas/../coleccion/_papelera/afilado_20261113_161954/afilado_20261112_
        161511_098.jpg" --con la foto puesta ahi de verdad-- daba la fila en el historial
        con_prueba=1 y el informe al lado de la orden decia "1 nombradas, 1 encontradas",
        sin ningun (NO ESTA). Con prueba="C:/Windows/win.ini" pasaba lo mismo: el informe
        del reclamo presentaba un archivo de configuracion de Windows como la foto del
        diente. Esas dos cosas son exactamente lo que la regla del taller prohibe: mostrar
        como prueba algo que no es prueba.

        Y ES LA MISMA REGLA CON LA QUE BORRA Y EXPORTA, que es el otro arreglo. Contando
        con esta regla, lo que la pantalla cuenta es lo que el zip va a llevar y lo que el
        borrado va a mover: no pueden discrepar, porque es la misma cuenta.

        LO QUE SE MIRA, EN ESTE ORDEN:
        - Que no traiga ":" ni "\\": una ruta con letra de unidad no es relativa a data, y
          la barra invertida no se puede usar como nombre adentro de un .zip --el que abre
          el paquete del otro lado veria un archivo con otro nombre que el que el informe
          nombra--. guardar_prueba(), que es el unico que escribe Diente.prueba, siempre
          escribe barras normales.
        - Que ningun tramo sea "", "." ni "..": eso saca la barra inicial, la doble barra,
          el "pruebas/../otra/parte" y el "pruebas/123/./foto.jpg" --este ultimo no es un
          escape, pero zipfile le normaliza el nombre y entonces el renglon del informe
          apuntaria a una ruta que el paquete no tiene--.
        - Que arranque justo en "pruebas/<carpeta de esta orden>/" y que quede algo despues.
        - Y el cinturon, por si Windows reconoce alguna forma de nombrar una carpeta de la
          que no nos enteramos --un acceso directo, una union de las que hace "mklink /J",
          que no pide ni permiso de administrador--: resuelta de verdad con resolve(), la
          CARPETA donde cae la foto tiene que seguir estando adentro de la carpeta de
          pruebas de esta orden, y el ultimo tramo no puede ser un atajo.

        POR QUE EL CINTURON SE MIDE POR CARPETA Y NO POR FOTO. Es la misma garantia --una
        ruta es su carpeta mas un ultimo tramo-- y resolve() en Windows abre el archivo,
        asi que hacerlo por foto se paga caro. Medido con una orden de 4 sierras de 80
        dientes y las 320 pruebas en el disco, que es la orden mas grande que sale del
        taller: los 320 resolve() de las rutas costaban 74,7 ms y los 320 de la carpeta
        otros 65,2 ms, contra 0,1 ms de todos los chequeos de texto juntos. Por carpeta
        --un resolve() de la carpeta, cacheado, y un is_symlink() por foto-- la misma
        cuenta sale en 24 ms. Por eso `resueltas` es un cache que trae el que llama y vale
        para una sola pasada: dentro de un historial la carpeta no cambia, y entre dos
        pasadas no se da por sabido nada.
        """
        texto = str(relativa or "")
        if not texto:
            return None
        if ":" in texto or "\\" in texto:
            return None
        tramos = texto.split("/")
        if any(tramo in ("", ".", "..") for tramo in tramos):
            return None
        carpeta = self._deposito(stem)
        if len(tramos) < 3 or tramos[0] != _CARPETA_PRUEBAS or tramos[1] != carpeta.name:
            return None
        ruta = self._datos() / texto
        tope = self._carpeta_resuelta(carpeta, resueltas)
        padre = self._carpeta_resuelta(ruta.parent, resueltas)
        if tope is None or padre is None:
            return None
        if padre != tope and tope not in padre.parents:
            return None
        try:
            if ruta.is_symlink():
                return None
        except OSError:
            return None
        return ruta

    def _carpeta_resuelta(
        self, carpeta: Path, resueltas: Optional[dict[Path, Optional[Path]]]
    ) -> Optional[Path]:
        """La carpeta resuelta de verdad --siguiendo atajos y uniones-- o None si no se
        pudo. Con `resueltas`, una sola vez por carpeta y por pasada: el que llama es el
        dueno del cache, igual que con el cache de nombres de _hay_prueba()."""
        if resueltas is None:
            try:
                return carpeta.resolve()
            except OSError:
                return None
        if carpeta not in resueltas:
            try:
                resueltas[carpeta] = carpeta.resolve()
            except OSError:
                resueltas[carpeta] = None
        return resueltas[carpeta]

    def _hay_prueba(
        self,
        relativa: str,
        stem: str,
        resueltas: Optional[dict[Path, Optional[Path]]] = None,
    ) -> bool:
        """Si la foto que nombra el diente es prueba de esta orden Y ESTA de verdad.

        Se mira, no se cree. Un .json que nombra una foto que ya no esta no tiene prueba
        guardada, y contarla igual seria exactamente lo que paso con las ordenes 123 y
        1234: el informe entregado nombra 14 fotos que hoy solo existen en la papelera.

        SE PREGUNTA CON EL MISMO is_file() QUE USA exportar(). Es lo que hace que la fila
        de la pantalla y el paquete del reclamo no puedan discrepar: no dos preguntas
        parecidas, una sola pregunta hecha de una sola manera. Antes esto listaba la
        carpeta y comparaba el nombre exacto, y eso es OTRA pregunta: medido, da distinto
        en dos casos que pasan sin que nadie toque nada a mano.

        UNO, LA CARPETA QUE NO SE DEJA LISTAR --un permiso mal puesto, una unidad de red,
        una carpeta que vino de otra PC--. El OSError del iterdir se volvia una lista
        vacia, o sea "la prueba no esta", que es afirmar lo que nadie pudo medir.
        Reproducido con icacls /deny :(RD) sobre el deposito de una orden de 3 pruebas: la
        fila salia "3 PRUEBAS QUE NO ESTAN" en rojo, la pantalla de borrado decia que no
        se llevaba ninguna foto, y el zip salia CON LAS 3 y su informe diciendo "3
        nombradas, 3 encontradas". Dos papeles del mismo reclamo diciendo lo contrario.
        Con is_file() las tres se cuentan, que es la verdad: la foto se abre por su nombre
        aunque la carpeta no se deje recorrer --verificado, is_file() da True ahi--.
        El que no puede contestar es pruebas_huerfanas(), que necesita recorrerla, y por
        eso devuelve None: eso ya estaba bien y no cambia.

        DOS, LA MISMA FOTO NOMBRADA DE OTRA FORMA. Windows no distingue mayusculas ni le
        importa un punto o un espacio al final, asi que abre el archivo igual, pero el
        nombre exacto de la lista no coincide. Medido contra la foto
        s01_d001_20261116_120000.jpg, 4 de 5 formas fallaban --todo en MAYUSCULAS,
        extension .JPG, punto al final, espacio al final--: fila en rojo y la foto adentro
        del zip. Con la letra de verdad cambiada las dos dicen que no esta, que es lo
        correcto.

        Y NO SALE MAS CARO: listar la carpeta no era "un acceso al disco por orden" como
        decia aca, porque iterdir() hace un stat por archivo igual. Medido en el caso gordo
        --5 ordenes, una de 4 sierras de 80 dientes con las 320 pruebas en el disco, tres
        corridas de 7 vueltas-- historial() da 57,4 / 58,2 / 57,8 ms de mediana con
        is_file() contra 62,0 / 61,6 / 62,2 ms listando. Son 4 ms menos, no es el motivo
        del cambio: el motivo es que sea UNA pregunta. Pero conviene tenerlo escrito para
        que nadie vuelva a poner el listado creyendo que ahorra disco.
        """
        ruta = self._es_prueba_de(relativa, stem, resueltas)
        if ruta is None:
            # WARNING Y CON LA RUTA ESCRITA. No es un archivo que falta --eso ya se cuenta
            # y se dibuja-- es un .json que nombra como prueba algo que esta afuera del
            # deposito de la orden, y el unico que puede arreglarlo es el que pueda leer
            # que decia. Se cuenta como prueba perdida, que es lo que es: esta orden no
            # tiene esa prueba.
            # Se nombra el ARCHIVO y no el numero de orden: el numero puede repetirse
            # --una copia a mano lleva el mismo adentro-- y el que va a mirar que dice ese
            # .json necesita saber cual de los dos abrir.
            _log.warning(
                "%s.json nombra como prueba %r, que esta afuera de su carpeta de pruebas "
                "(%s): no se cuenta como prueba",
                stem, str(relativa), self._deposito(stem),
            )
            return False
        return ruta.is_file()

    def _pruebas_con_el_stem(self, orden: Orden, stem: str) -> set[str]:
        """Las rutas de prueba de la orden que SON prueba de `stem` y estan en el disco.

        El cache de carpetas resueltas se arma aca y vale para esta pasada nada mas: un
        resolve() por carpeta en lugar de uno por diente, que es lo caro. Ver
        _es_prueba_de() para la medicion.
        """
        resueltas: dict[Path, Optional[Path]] = {}
        return {
            ruta for ruta in orden.pruebas_nombradas()
            if self._hay_prueba(ruta, stem, resueltas)
        }

    def _mover_deposito(self, origen: Path, adentro: Path) -> None:
        """Mueve la carpeta de pruebas de la orden a la papelera.

        SI YA HAY UNA CARPETA DE UN INTENTO ANTERIOR, las fotos que quedaron se meten
        ADENTRO de esa, de a una. Hace falta porque shutil.move con un destino que ya existe
        como carpeta NO la reemplaza: mete la de origen ADENTRO, y el reintento dejaria las
        fotos en pruebas/<orden>/<orden>/, un nivel mas abajo de donde el .json las nombra.
        Rescatarlas con el Explorador --que es como se rescata en el taller-- dejaria la
        prueba en el disco y sin encontrarse.

        El rmdir del final es a proposito y no un rmtree: si quedo algo adentro que no se
        pudo mover, revienta y el borrado se declara a medias, que es la verdad. Borrar a
        ciegas la carpeta de origen seria perder una foto de prueba sin decirlo.
        """
        if adentro.is_dir():
            for pieza in sorted(origen.iterdir()):
                shutil.move(str(pieza), str(adentro / pieza.name))
            origen.rmdir()
            return
        adentro.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(origen), str(adentro))

    def _papelera_de_un_intento_anterior(self, stem: str) -> Optional[Path]:
        """La carpeta de papelera de un borrado de ESTA MISMA orden que quedo a medias, si
        hay una. None si no.

        EL PROBLEMA QUE ARREGLA: LA ORDEN PARTIDA EN DOS CARPETAS. Un borrado que se traba
        en la carpeta de pruebas deja una copia de unas fotos en la papelera Y las deja en
        su lugar --shutil.move copia archivo por archivo cuando no puede renombrar-- y el
        reintento abria OTRA carpeta de papelera con las mismas fotos. Reproducido con la
        orden 9300 y 3 pruebas, trabando una foto con msvcrt.locking, que es lo que pasa si
        el operario la tiene abierta en el visor de Windows: el primer intento dejo
        _papelera/20261116_161039 con 2 de las 3 fotos y las 3 seguian en el deposito; el
        segundo abrio _papelera/20261116_161040 con el .json y las 3. La misma orden en dos
        carpetas, con fotos repetidas, y nadie puede saber mirando cual es la buena.

        COMO SE RECONOCE UN INTENTO A MEDIAS, y por que se puede: el .json es lo ULTIMO que
        se mueve. Entonces una carpeta de papelera que tiene algo de esta orden y NO tiene
        su .json es, por construccion, un intento que no termino. Una que SI lo tiene es un
        borrado que se completo --o al menos llego hasta el .json-- y esa no se toca nunca:
        adentro esta la orden que alguien puede querer rescatar.

        Si hay mas de una se usa la ULTIMA: el sello del nombre es la fecha y la hora
        --%Y%m%d_%H%M%S, que ordena igual como texto que como fecha-- asi que la ultima es
        la del intento mas reciente, que es la que tiene mas cosas adentro.
        """
        papelera = self._base / _PAPELERA
        if not papelera.is_dir():
            return None
        deposito = self._deposito(stem)
        candidatas: list[Path] = []
        try:
            carpetas = sorted(p for p in papelera.iterdir() if p.is_dir())
        except OSError as error:
            _log.warning("no se pudo mirar la papelera %s: %s", papelera, error)
            return None
        for carpeta in carpetas:
            if (carpeta / f"{stem}.json").exists():
                # Ese borrado llego hasta el .json: es una orden guardada, no un intento.
                continue
            de_esta_orden = (
                (carpeta / _CARPETA_PRUEBAS / deposito.name).is_dir()
                or (carpeta / f"informe_{stem}.txt").is_file()
                or (carpeta / f"informe_{stem}.csv").is_file()
            )
            if de_esta_orden:
                candidatas.append(carpeta)
        if not candidatas:
            return None
        elegida = candidatas[-1]
        _log.info(
            "el borrado de la orden %s sigue en la carpeta de papelera %s, que quedo de un "
            "intento anterior: no se abre otra",
            stem, elegida,
        )
        return elegida

    def _papelera_libre(self) -> Path:
        """Una carpeta de papelera que todavia no exista.

        El sello tiene resolucion de un segundo y borrar dos ordenes dentro del mismo
        segundo pisaria la primera. La papelera existe justamente para que no se pierda
        nada. Adentro de la carpeta el .json se llama como se llamaba, asi que para saber
        que hay ahi no hace falta mas que entrar.
        """
        sello = datetime.now().strftime("%Y%m%d_%H%M%S")
        carpeta = self._base / _PAPELERA / sello
        repeticion = 1
        while carpeta.exists():
            carpeta = self._base / _PAPELERA / f"{sello}_{repeticion}"
            repeticion += 1
        return carpeta

    def _cargar_abierta(self) -> None:
        archivo = self._base / _ARCHIVO_ABIERTA
        if not archivo.is_file():
            return
        try:
            numero = archivo.read_text(encoding="utf-8").strip()
        except OSError:
            return
        if not numero:
            return
        orden = self.cargar(numero)
        if orden is not None and orden.cerrada:
            # UNA ORDEN CERRADA NO SE VUELVE A ABRIR SOLA. El marcador .abierta puede
            # quedar apuntando a una orden ya informada y entregada --si el borrado del
            # marcador falla, o si el programa se corta entre el guardar() y el
            # _recordar_abierta(None) de cerrar()-- y ahi el programa seguia anotando
            # dientes encima: reproducido, el papel entregado decia "3 en total, 1
            # revisados" y el .json termino con 2. Retomarla es una decision del operario,
            # no algo que pase solo al abrir el programa.
            _log.info("la orden %s ya estaba cerrada: no se retoma", orden.numero)
            self._recordar_abierta(None)
            return
        self._abierta = orden

    def _recordar_abierta(self, numero: Optional[str]) -> None:
        """Que orden retomar al abrir el programa. Que falle solo cuesta volver a elegirla."""
        try:
            self._base.mkdir(parents=True, exist_ok=True)
            (self._base / _ARCHIVO_ABIERTA).write_text(numero or "", encoding="utf-8")
        except OSError as error:
            # ERROR y no DEBUG: el logger raiz arranca en INFO, asi que en DEBUG esto no
            # quedaba escrito en ninguna parte. Y es el unico aviso que tiene el taller de
            # que el marcador quedo mal: al abrir el programa, la orden que se retoma no es
            # la que el operario dejo.
            _log.error("no se pudo recordar la orden abierta: %s", error)


def _celda(valor: str) -> str:
    """Un texto listo para meter en el .csv sin que Excel lo tome por una formula.

    El cliente y la nota de pedido los teclea el operario con lo que le pasa el mostrador,
    y Excel ejecuta cualquier celda que arranque con = + - o @. Una nota de pedido que
    diga `=cmd|...` se abre como un comando el dia que alguien mira el informe. Con la
    comilla adelante, Excel lo muestra como texto y no lo corre; el .txt del informe no
    pasa por aca porque nadie lo ejecuta.
    """
    texto = str(valor)
    if texto[:1] in ("=", "+", "-", "@", chr(9), chr(13)):
        return "'" + texto
    return texto


def _una_linea(valor: Any) -> str:
    """Un texto de UNA sola linea, para las etiquetas de ancho fijo del informe.

    El informe arma "Nota de pedido   : ..." con la etiqueta a la izquierda, asi que un
    salto de linea adentro del dato parte la cabecera: con una nota de pedido que traia un
    salto, el papel imprimia la primera mitad, despues "segunda linea" sola y SIN etiqueta,
    y despues "Cliente          :    Aserradero" con la columna corrida. Hoy no hay como
    teclear un salto --el formulario acepta 32..126-- pero si entra por desde_datos(), que
    es la puerta de la app de afilados y no limpiaba nada.
    """
    return " ".join(str(valor).split())


def _ahora() -> str:
    return datetime.now().strftime("%d/%m/%Y %H:%M:%S")
