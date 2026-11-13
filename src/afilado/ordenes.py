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
import json
import logging
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from .config import ruta_absoluta

_log = logging.getLogger(__name__)

_ARCHIVO_ABIERTA = ".abierta"
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

    def __post_init__(self) -> None:
        # Ordenes guardadas antes de que existiera el campo: ahi un diente salteado se
        # reconocia solo por el origen que le ponia saltear(). Sin esto, al releer una de
        # esas ordenes los salteados volvian a contarse como revisados.
        if not self.salteado and self.origen == _ORIGEN_SALTEADO:
            self.salteado = True
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
        )

    @classmethod
    def desde_datos(cls, datos: dict[str, Any]) -> "Orden":
        """Arma una orden desde un diccionario, venga de donde venga.

        Es el punto por donde va a entrar la app de afilados el dia que se conecte: lo que
        hoy teclea el operario y lo que manana mande la otra app terminan los dos aca, asi
        que el resto del programa no se entera del cambio.
        """
        sierras = []
        for crudo in datos.get("sierras", []):
            revisados = [Diente(**d) for d in crudo.get("revisados", [])]
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
        return cls(
            # Sin sanear, igual que nueva(): el saneado es para el nombre del archivo.
            numero=" ".join(str(datos.get("numero", "")).split()),
            nota_pedido=_una_linea(datos.get("nota_pedido", "")),
            cliente=_una_linea(datos.get("cliente", "")),
            abierta=str(datos.get("abierta", "")),
            cerrada=str(datos.get("cerrada", "")),
            sierras=sierras,
            sierra_actual=int(datos.get("sierra_actual", 1)),
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
    def completa(self) -> bool:
        return all(s.completa for s in self.sierras)

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
    def informe(self) -> str:
        """El informe en castellano llano, para imprimir o mandar ante un reclamo."""
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
            "",
        ]
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
        ]
        return "\n".join(lineas)

    def filas_csv(self) -> list[list[str]]:
        """Una fila por diente, para abrir en Excel."""
        filas = [[
            "orden", "nota_pedido", "cliente", "sierra", "identificacion", "diente",
            "estado", "resultado", "parecido", "origen", "cuando", "foto", "nota",
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
                    diente.origen, diente.instante, diente.foto, diente.nota,
                )])
        return filas


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

    def cargar(self, numero: str) -> Optional[Orden]:
        archivo = self.ruta(numero)
        if not archivo.is_file():
            return None
        try:
            return Orden.desde_datos(json.loads(archivo.read_text(encoding="utf-8")))
        except (OSError, ValueError, KeyError, TypeError, AttributeError) as error:
            # ATTRIBUTEERROR TAMBIEN. Un .json con forma valida pero distinta --"sierras"
            # como diccionario {"1": {...}}, una lista de textos, o el json entero siendo
            # un numero-- hacia que desde_datos() llamara .get() sobre un str, un list o un
            # int. Esa excepcion salia de Ordenes.__init__, que corre antes de que haya
            # pantalla, asi que el .exe no abria; y como el .abierta seguia apuntando a ese
            # archivo, no abria mas. Reproducido con los cuatro payloads.
            _log.warning("no se pudo leer la orden %s: %s", archivo, error)
            return None

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
        self._fallo_al_guardar = True       # hasta que el os.replace diga lo contrario
        archivo = self.ruta(orden.numero)
        provisorio = archivo.with_suffix(".json.tmp")
        try:
            self._base.mkdir(parents=True, exist_ok=True)
            with open(provisorio, "w", encoding="utf-8") as manejador:
                manejador.write(json.dumps(asdict(orden), indent=2, ensure_ascii=False))
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
            provisorio.write_text(orden.informe(), encoding="utf-8")
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


    # ------------------------------------------------------------------ internos
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
