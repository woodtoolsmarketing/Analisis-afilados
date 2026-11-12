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
    nota: str = ""                      # "salteado", o lo que haga falta aclarar

    @property
    def revisado(self) -> bool:
        return bool(self.instante)


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
        faltan = int(self.dientes) - len(self.revisados)
        for i in range(faltan):
            self.revisados.append(Diente(numero=len(self.revisados) + 1))

    @property
    def cuantos_revisados(self) -> int:
        return sum(1 for d in self.revisados if d.revisado)

    @property
    def completa(self) -> bool:
        return self.cuantos_revisados >= self.dientes

    @property
    def rechazados(self) -> int:
        return sum(1 for d in self.revisados if d.revisado and d.aprobado is False)

    @property
    def sin_decidir(self) -> int:
        return sum(1 for d in self.revisados if d.revisado and d.aprobado is None)

    def proximo(self) -> Optional[Diente]:
        """El primer diente sin revisar. None si ya estan todos."""
        for diente in self.revisados:
            if not diente.revisado:
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
        limpio = sanear_numero(numero)
        if not limpio:
            raise ValueError("la orden necesita un numero")
        sierras = max(1, int(cantidad_sierras))
        dientes = max(1, int(dientes_por_sierra))
        if sierras > MAX_SIERRAS:
            raise ValueError(f"{sierras} sierras es demasiado; el tope es {MAX_SIERRAS}")
        if dientes > MAX_DIENTES:
            raise ValueError(f"{dientes} dientes es demasiado; el tope es {MAX_DIENTES}")
        return cls(
            numero=limpio,
            nota_pedido=str(nota_pedido).strip(),
            cliente=str(cliente).strip(),
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
            sierras.append(
                Sierra(
                    numero=int(crudo.get("numero", len(sierras) + 1)),
                    dientes=int(crudo.get("dientes", 0)),
                    identificacion=str(crudo.get("identificacion", "")),
                    revisados=revisados,
                )
            )
        return cls(
            numero=sanear_numero(datos.get("numero", "")),
            nota_pedido=str(datos.get("nota_pedido", "")),
            cliente=str(datos.get("cliente", "")),
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
    def completa(self) -> bool:
        return all(s.completa for s in self.sierras)

    def sierra(self) -> Optional[Sierra]:
        """La sierra en la que se esta trabajando."""
        for s in self.sierras:
            if s.numero == self.sierra_actual:
                return s
        return self.sierras[0] if self.sierras else None

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
        diente.instante = _ahora()
        return diente

    def saltear(self, motivo: str = "salteado") -> Optional[Diente]:
        """Anota un diente como no revisado y sigue. Un hueco explicado es un dato."""
        return self.anotar(nota=motivo, origen="sin revisar")

    def siguiente_sierra(self) -> int:
        """Pasa a la sierra siguiente, dando la vuelta. Devuelve en cual quedo."""
        if self.sierras:
            indice = next(
                (i for i, s in enumerate(self.sierras) if s.numero == self.sierra_actual), 0
            )
            self.sierra_actual = self.sierras[(indice + 1) % len(self.sierras)].numero
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
            f"{self.total_revisados} revisados, {self.total_rechazados} para rechazar",
            "",
        ]
        for sierra in self.sierras:
            lineas.append("-" * 72)
            titulo = f"SIERRA {sierra.numero} de {len(self.sierras)}"
            if sierra.identificacion:
                titulo += f"   ({sierra.identificacion})"
            lineas.append(titulo)
            lineas.append(
                f"  {sierra.dientes} dientes   -   {sierra.cuantos_revisados} revisados   "
                f"-   {sierra.rechazados} para rechazar   -   "
                f"{sierra.sin_decidir} sin decidir"
            )
            lineas.append("")
            lineas.append("   diente  estado          resultado     parecido  cuando")
            for diente in sierra.revisados:
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
                if diente.nota:
                    lineas.append(f"           nota: {diente.nota}")
            lineas.append("")

        lineas += [
            "=" * 72,
            "COMO LEER ESTE INFORME",
            "-" * 72,
            "La columna 'estado' es en que quedo cada diente segun lo que el programa tenia",
            "guardado de otros dientes iguales, o segun el modelo entrenado si ya hay uno.",
            "La columna 'parecido' dice CUANTO se parecio al estado que se le puso: un 60%",
            "es una decision al filo y un 95% no. Donde dice 'sin decidir', el programa no",
            "se animo a opinar y lo dice en vez de inventar un veredicto.",
            "",
            "Los dientes sin revisar aparecen igual, en blanco: que falte uno es un dato.",
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
                    "APROBADO" if diente.aprobado is True
                    else "RECHAZADO" if diente.aprobado is False
                    else ("sin decidir" if diente.revisado else "sin revisar")
                )
                filas.append([
                    self.numero, self.nota_pedido, self.cliente,
                    str(sierra.numero), sierra.identificacion, str(diente.numero),
                    diente.estado or "", resultado,
                    "" if diente.parecido is None else f"{diente.parecido:.4f}",
                    diente.origen, diente.instante, diente.foto, diente.nota,
                ])
        return filas


class Ordenes:
    """Las ordenes guardadas en disco, y cual es la que esta abierta."""

    def __init__(self, directorio: str = "data/ordenes") -> None:
        self._base = ruta_absoluta(directorio)
        self._abierta: Optional[Orden] = None
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

    def cargar(self, numero: str) -> Optional[Orden]:
        archivo = self.ruta(numero)
        if not archivo.is_file():
            return None
        try:
            return Orden.desde_datos(json.loads(archivo.read_text(encoding="utf-8")))
        except (OSError, ValueError, KeyError, TypeError) as error:
            _log.warning("no se pudo leer la orden %s: %s", archivo, error)
            return None

    # ------------------------------------------------------------------ cambios
    def abrir(self, orden: Orden) -> Orden:
        """Deja esta orden como la que se esta trabajando y la guarda."""
        self._abierta = orden
        self.guardar()
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

    def guardar(self) -> Optional[Path]:
        """Escribe la orden abierta. Se llama despues de CADA diente anotado.

        Guardar de a un diente y no al final es a proposito: lo que este informe tiene que
        poder contestar es un reclamo, y un corte de luz en la mitad de una sierra no puede
        borrar las dos horas anteriores.
        """
        orden = self._abierta
        if orden is None:
            return None
        archivo = self.ruta(orden.numero)
        provisorio = archivo.with_suffix(".json.tmp")
        try:
            self._base.mkdir(parents=True, exist_ok=True)
            provisorio.write_text(
                json.dumps(asdict(orden), indent=2, ensure_ascii=False), encoding="utf-8"
            )
            os.replace(provisorio, archivo)
        except OSError as error:
            _log.error("no se pudo guardar la orden %s: %s", orden.numero, error)
            return None
        self._recordar_abierta(orden.numero)
        return archivo

    def escribir_informe(self, orden: Optional[Orden] = None) -> tuple[Optional[Path], Optional[Path]]:
        """Deja el informe en texto y la planilla. Devuelve las dos rutas."""
        orden = orden or self._abierta
        if orden is None:
            return None, None
        try:
            self._base.mkdir(parents=True, exist_ok=True)
            texto = self._base / f"informe_{orden.numero}.txt"
            texto.write_text(orden.informe(), encoding="utf-8")
            planilla = self._base / f"informe_{orden.numero}.csv"
            with open(planilla, "w", encoding="utf-8-sig", newline="") as manejador:
                # utf-8-sig y punto y coma: es lo que abre bien el Excel en castellano.
                csv.writer(manejador, delimiter=";").writerows(orden.filas_csv())
        except OSError as error:
            _log.error("no se pudo escribir el informe de %s: %s", orden.numero, error)
            return None, None
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
        if numero:
            self._abierta = self.cargar(numero)

    def _recordar_abierta(self, numero: Optional[str]) -> None:
        """Que orden retomar al abrir el programa. Que falle solo cuesta volver a elegirla."""
        try:
            self._base.mkdir(parents=True, exist_ok=True)
            (self._base / _ARCHIVO_ABIERTA).write_text(numero or "", encoding="utf-8")
        except OSError as error:
            _log.debug("no se pudo recordar la orden abierta: %s", error)


def _ahora() -> str:
    return datetime.now().strftime("%d/%m/%Y %H:%M:%S")
