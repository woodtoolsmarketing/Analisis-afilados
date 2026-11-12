"""Los CONCEPTOS: en que puede clasificarse una foto, y quien manda sobre esa lista.

QUE ES UN CONCEPTO
------------------
Una carpeta con fotos que significan lo mismo: "aprobada", "rechazada", "diente astillado",
"filo quemado". El operario decide cuales hay. Antes la lista vivia en configs/config.yaml
y cambiarla pedia editar un YAML a mano; ahora se maneja desde el programa.

LA VERDAD ESTA EN EL DISCO, NO EN UN ARCHIVO DE LISTA
-----------------------------------------------------
Los conceptos SON las carpetas de data/coleccion. El conceptos.json que esta al lado no
manda: solo agrega lo que una carpeta no puede decir por si sola --si ese concepto cuenta
como defecto, y en que orden se muestra--. Si alguien copia una carpeta nueva ahi con el
Explorador de Windows, al abrir el programa aparece como un concepto mas; si borra una, deja
de estar. Un indice que pretendiera ser la verdad se desincronizaria del disco en el primer
descuido, y entonces habria fotos en carpetas que el programa jura que no existen.

CARPETAS QUE EMPIEZAN CON "_" NO SON CONCEPTOS
-----------------------------------------------
Es la convencion para lo que vive ahi adentro sin ser una clase: _papelera, sobre todo.
Igual que models/_demo_no_usar.

BORRAR NO BORRA
---------------
Eliminar un concepto o una foto los MUEVE a _papelera. Un clic de mas no puede costar las
fotos que el taller tardo semanas en juntar; el que quiera el espacio vacia esa carpeta con
el Explorador, viendo lo que tira.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional

from .config import ruta_absoluta

_log = logging.getLogger(__name__)

EXTENSIONES = {".jpg", ".jpeg", ".png", ".bmp"}

_ARCHIVO_INDICE = "conceptos.json"
_PAPELERA = "_papelera"

# Caracteres que sobreviven a un nombre de carpeta en Windows, en Linux y como nombre de
# clase de YOLO el dia que se entrene. El resto se reemplaza por "_" a la vista del
# operario mientras escribe, para que no haya sorpresas entre lo que tecleo y lo que quedo.
_VALIDOS = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_ ")

# Un nombre mas largo que esto no entra en la pantalla ni en un boton.
LARGO_MAXIMO = 24


def caracter_valido(letra: str) -> bool:
    """Si esta letra se puede teclear en el nombre de un concepto.

    Se filtra al escribir, no al confirmar: asi el operario ve en el acto que una tecla no
    entra, en vez de descubrir despues que el nombre que escribio quedo en otra cosa.
    """
    return letra in _VALIDOS


def sanear(nombre: str) -> str:
    """Deja un nombre de concepto usable como carpeta. Devuelve "" si no queda nada.

    SE RECORTA ANTES DE LIMPIAR LOS BORDES, no al reves: al reves, un nombre largo que
    quedaba cortado justo en un guion bajo daba un resultado que volvia a cambiar si se lo
    saneaba de nuevo. Que sanear(sanear(x)) == sanear(x) importa porque hay dos caminos que
    lo aplican --el nombre que se teclea y el que se lee del indice-- y si no coincidieran,
    dos conceptos distintos podrian terminar apuntando a la misma carpeta.
    """
    limpio = "".join(c if c in _VALIDOS else "_" for c in nombre)
    limpio = "_".join(limpio.split())          # espacios -> un solo guion bajo
    return limpio[:LARGO_MAXIMO].strip("_-")


@dataclass
class Concepto:
    """Un concepto y su estado en disco."""

    nombre: str
    carpeta: Path
    fotos: int = 0
    defecto: bool = False
    # Un destino que NO es un concepto del dataset pero al que igual van fotos: hoy, la
    # galeria de fotos de modelo. Se lista junto a los demas porque la pregunta del
    # operario es "a donde mando esta foto" y esa carpeta es una respuesta posible, pero no
    # se lo cuenta como clase ni se lo puede borrar.
    especial: bool = False

    @property
    def tiene_fotos(self) -> bool:
        return self.fotos > 0


class Conceptos:
    """La lista de conceptos de una carpeta de coleccion, y como cambiarla."""

    def __init__(
        self,
        directorio: str = "data/coleccion",
        semilla: Optional[Iterable[str]] = None,
        defectos: Optional[Iterable[str]] = None,
    ) -> None:
        """`semilla` y `defectos` son la lista del config: se usan SOLO la primera vez.

        Sirven para que una instalacion que ya venia con aprobada/rechazada siga teniendo
        esos dos conceptos sin que nadie los vuelva a crear, y para saber cual de los dos
        contaba como defecto. Despues manda lo que haya en disco.
        """
        self._base = ruta_absoluta(directorio)
        self._semilla = [sanear(c) for c in (semilla or []) if sanear(c)]
        self._defectos_semilla = {sanear(c) for c in (defectos or []) if sanear(c)}
        self._meta: dict[str, dict] = {}
        self._lista: list[Concepto] = []
        self.recargar()

    # ------------------------------------------------------------------ lectura
    @property
    def carpeta(self) -> Path:
        return self._base

    @property
    def hubo_indice(self) -> bool:
        """Si ya existe el archivo de conceptos, o sea si esta carpeta se uso alguna vez.

        Sirve para distinguir "recien instalado" --donde la lista del config es la
        semilla-- de "el operario los borro a todos", donde la lista vacia es la verdad.
        """
        return (self._base / _ARCHIVO_INDICE).is_file()

    def listar(self) -> list[Concepto]:
        return list(self._lista)

    def nombres(self) -> list[str]:
        return [c.nombre for c in self._lista]

    def defectos(self) -> list[str]:
        return [c.nombre for c in self._lista if c.defecto]

    def _resolver(self, nombre: str) -> Optional[str]:
        """El nombre TAL COMO ESTA EN DISCO, o None si no hay ninguno asi.

        No alcanza con sanear() para encontrar la carpeta de un concepto: una carpeta que
        alguien copio con el Explorador puede llamarse "filo-" o tener una letra con
        acento, y sanear le saca el guion del final o le cambia la letra. Buscarla por el
        nombre saneado
        apuntaba entonces a una carpeta que no existe --y borrar con ese nombre llegaba a
        mover la carpeta equivocada--. Primero se busca tal cual, despues saneado, y por
        ultimo sin distinguir mayusculas, que es como las compara Windows.
        """
        actuales = {c.nombre for c in self._lista}
        if nombre in actuales:
            return nombre
        limpio = sanear(nombre)
        if limpio in actuales:
            return limpio
        plegados = {nombre.casefold(), limpio.casefold()}
        for actual in sorted(actuales):
            if actual.casefold() in plegados:
                return actual
        return None

    def existe(self, nombre: str) -> bool:
        return self._resolver(nombre) is not None

    def con_fotos(self) -> list[Concepto]:
        return [c for c in self._lista if c.tiene_fotos]

    def total_fotos(self) -> int:
        return sum(c.fotos for c in self._lista)

    def fotos(self, nombre: str) -> list[Path]:
        """Las fotos de un concepto, ordenadas de la mas nueva a la mas vieja.

        Mas nueva primero porque lo que el operario va a querer borrar casi siempre es lo
        que acaba de sacar: una foto movida, o una que archivo en la carpeta equivocada.
        """
        return self._fotos_de(self._resolver(nombre) or sanear(nombre))

    def _fotos_de(self, carpeta_nombre: str) -> list[Path]:
        """Las fotos de una carpeta por su nombre EXACTO. Es lo que usa recargar()."""
        carpeta = self._base / carpeta_nombre
        if not carpeta.is_dir():
            return []
        try:
            archivos = [
                p for p in carpeta.iterdir()
                if p.is_file() and p.suffix.lower() in EXTENSIONES
            ]
        except OSError as error:
            _log.warning("no se pudo leer %s: %s", carpeta, error)
            return []
        # Por FECHA, que es lo que quiere decir "la mas nueva". Ordenar por nombre coincide
        # mientras los archivos los ponga el programa --les pone la fecha adelante-- pero
        # no cuando alguien copia fotos con otro nombre desde el Explorador.
        try:
            return sorted(archivos, key=lambda p: p.stat().st_mtime, reverse=True)
        except OSError:
            return sorted(archivos, key=lambda p: p.name, reverse=True)

    def mapa(self) -> dict[str, list[Path]]:
        """{concepto: [fotos]} para que la memoria sepa que hay que recordar."""
        return {c.nombre: self.fotos(c.nombre) for c in self._lista}

    # ------------------------------------------------------------------ cambios
    def recargar(self) -> None:
        """Relee las carpetas del disco y las cruza con el indice.

        EL ORDEN EN QUE SE MUESTRAN es el del indice, o sea el orden en que se fueron
        creando, y los que aparecen por primera vez se agregan al final en orden
        alfabetico. Que no se reacomoden solos importa mas de lo que parece: el panel se
        maneja tambien con las teclas 1..9, y una lista que cambia de orden sola haria que
        el 3 sea un dia un concepto y al dia siguiente otro.
        """
        self._meta = self._leer_indice()
        en_disco: list[str] = []
        if self._base.is_dir():
            try:
                en_disco = sorted(
                    p.name for p in self._base.iterdir()
                    if p.is_dir() and not p.name.startswith(("_", "."))
                )
            except OSError as error:
                _log.warning("no se pudo leer %s: %s", self._base, error)

        # Primera vez: la lista del config siembra los conceptos, con su carpeta y todo.
        #
        # "Primera vez" es que NO EXISTA el indice, no que no haya carpetas. Si se mirara
        # solo las carpetas, el operario que borra el ultimo concepto desde el panel lo
        # veria reaparecer en el arranque siguiente, resucitado por el config, sin
        # entender por que.
        hubo_indice = (self._base / _ARCHIVO_INDICE).is_file()
        if not en_disco and not self._meta and self._semilla and not hubo_indice:
            for nombre in self._semilla:
                try:
                    (self._base / nombre).mkdir(parents=True, exist_ok=True)
                except OSError as error:
                    _log.warning("no se pudo crear la carpeta de '%s': %s", nombre, error)
                    continue
                en_disco.append(nombre)

        cambio = False
        for nombre in en_disco:
            if nombre not in self._meta:
                self._meta[nombre] = {"defecto": nombre in self._defectos_semilla}
                cambio = True
        # Una carpeta borrada con el Explorador deja de ser un concepto: manda el disco.
        for nombre in list(self._meta):
            if nombre not in en_disco:
                del self._meta[nombre]
                cambio = True
        if cambio:
            self._escribir_indice()

        self._lista = [
            Concepto(
                nombre=nombre,
                carpeta=self._base / nombre,
                fotos=len(self._fotos_de(nombre)),
                defecto=bool(datos.get("defecto", nombre in self._defectos_semilla)),
            )
            for nombre, datos in self._meta.items()
        ]

    def agregar(self, nombre: str, defecto: bool = False) -> Concepto:
        """Crea un concepto nuevo. Lanza ValueError si el nombre no sirve o ya existe."""
        limpio = sanear(nombre)
        if not limpio:
            raise ValueError("el nombre no puede quedar vacio")
        if limpio == _PAPELERA or limpio.startswith("_"):
            raise ValueError(f"'{limpio}' es un nombre reservado del programa")
        if self.existe(limpio):
            raise ValueError(f"ya hay un concepto que se llama '{limpio}'")

        (self._base / limpio).mkdir(parents=True, exist_ok=True)
        self._meta[limpio] = {"defecto": bool(defecto)}
        self._escribir_indice()
        self.recargar()
        _log.info("concepto nuevo: %s (defecto=%s)", limpio, defecto)
        for concepto in self._lista:
            if concepto.nombre == limpio:
                return concepto
        raise OSError(f"no se pudo crear la carpeta del concepto '{limpio}'")

    def eliminar(self, nombre: str) -> int:
        """Manda el concepto y sus fotos a _papelera. Devuelve cuantos archivos se movieron."""
        real = self._resolver(nombre)
        if real is None:
            raise ValueError(f"no hay ningun concepto que se llame '{sanear(nombre)}'")
        origen = self._base / real
        # Se cuenta TODO lo que se va, no solo las fotos sueltas de la carpeta: el numero
        # que devuelve esto es el que el panel le muestra al operario ("se fueron 12
        # fotos"), y tiene que corresponder con lo que de verdad se movio.
        cuantas = 0
        if origen.is_dir():
            cuantas = sum(
                1 for p in origen.rglob("*")
                if p.is_file() and p.suffix.lower() in EXTENSIONES
            )
            destino = self._destino_libre(self._papelera(), real, "")
            destino.parent.mkdir(parents=True, exist_ok=True)
            try:
                shutil.move(str(origen), str(destino))
            except OSError as error:
                # Si el movimiento quedo a medias --un archivo tomado por otro programa--
                # se limpia lo que llego a copiarse. Peor que fallar es dejar la mitad de
                # las fotos duplicadas y que nadie sepa cual es cual.
                shutil.rmtree(destino, ignore_errors=True)
                raise OSError(f"no se pudo mover '{real}' a la papelera: {error}") from error
            _log.info("concepto '%s' (%d fotos) movido a %s", real, cuantas, destino)
        self._meta.pop(real, None)
        self._escribir_indice()
        self.recargar()
        return cuantas

    def marcar_defecto(self, nombre: str, defecto: bool) -> None:
        """Dice si un concepto significa "pieza para rechazar"."""
        real = self._resolver(nombre)
        if real is None:
            raise ValueError(f"no hay ningun concepto que se llame '{sanear(nombre)}'")
        self._meta.setdefault(real, {})["defecto"] = bool(defecto)
        self._escribir_indice()
        self.recargar()

    def eliminar_foto(self, ruta: Path) -> Path:
        """Manda UNA foto a _papelera y devuelve donde quedo."""
        ruta = Path(ruta)
        if not ruta.is_file():
            raise ValueError(f"no existe la foto {ruta}")
        destino_dir = self._papelera() / ruta.parent.name
        destino_dir.mkdir(parents=True, exist_ok=True)
        destino = self._destino_libre(destino_dir, ruta.stem, ruta.suffix)
        try:
            shutil.move(str(ruta), str(destino))
        except OSError as error:
            # Todo o nada: si quedo una copia a medias en la papelera, se limpia. Dos
            # archivos iguales y nadie sabiendo cual es el bueno es peor que un error.
            try:
                destino.unlink(missing_ok=True)
            except OSError:
                pass
            raise OSError(f"no se pudo mover {ruta.name} a la papelera: {error}") from error
        _log.info("foto %s movida a %s", ruta.name, destino)
        self.recargar()
        return destino

    # ------------------------------------------------------------------ internos
    def _papelera(self) -> Path:
        return self._base / _PAPELERA

    def _destino_libre(self, carpeta: Path, base: str, sufijo: str) -> Path:
        """Un nombre que todavia no existe en la papelera.

        El sello tiene resolucion de un segundo, y borrar dos veces el mismo concepto --o
        dos fotos con el mismo nombre-- dentro del mismo segundo pisaba lo anterior sin
        decir nada. La papelera existe justamente para que nada se pierda.
        """
        sello = self._sello()
        destino = carpeta / f"{base}_{sello}{sufijo}"
        repeticion = 1
        while destino.exists():
            destino = carpeta / f"{base}_{sello}_{repeticion}{sufijo}"
            repeticion += 1
        return destino

    @staticmethod
    def _sello() -> str:
        return datetime.now().strftime("%Y%m%d_%H%M%S")

    def _leer_indice(self) -> dict[str, dict]:
        archivo = self._base / _ARCHIVO_INDICE
        if not archivo.is_file():
            return {}
        try:
            crudo = json.loads(archivo.read_text(encoding="utf-8"))
        except (OSError, ValueError) as error:
            # Un indice ilegible no puede dejar al taller sin conceptos: las carpetas
            # siguen estando y eso es lo que de verdad importa.
            _log.warning("no se pudo leer %s (%s); se usan las carpetas del disco",
                         archivo, error)
            return {}
        if not isinstance(crudo, dict):
            return {}
        conceptos = crudo.get("conceptos")
        if not isinstance(conceptos, list):
            return {}
        meta: dict[str, dict] = {}
        for entrada in conceptos:
            # El nombre se toma TAL CUAL se escribio: es el nombre de una carpeta que
            # existe. Sanearlo aca hacia que dos conceptos distintos en disco --"filo" y
            # "filo-"-- colapsaran en la misma clave del indice y uno se comiera al otro.
            if isinstance(entrada, str):
                nombre, datos = entrada.strip(), {}
            elif isinstance(entrada, dict):
                nombre = str(entrada.get("nombre", "")).strip()
                datos = {"defecto": bool(entrada.get("defecto", False))}
            else:
                continue
            if nombre and not nombre.startswith(("_", ".")) and "/" not in nombre \
                    and "\\" not in nombre:
                meta[nombre] = datos
        return meta

    def _escribir_indice(self) -> None:
        """Guarda el indice. Que falle no rompe nada: las carpetas siguen mandando.

        Se escribe aparte y se lo pone en su lugar de un golpe: un corte en la mitad de la
        escritura dejaria un JSON truncado, y lo unico que vive ahi --cual concepto cuenta
        como defecto-- se perderia en silencio, con la pantalla pintando de verde una
        pieza para tirar.
        """
        definitivo = self._base / _ARCHIVO_INDICE
        provisorio = definitivo.with_suffix(".json.tmp")
        try:
            self._base.mkdir(parents=True, exist_ok=True)
            provisorio.write_text(
                json.dumps(
                    {
                        "conceptos": [
                            {"nombre": nombre, "defecto": bool(datos.get("defecto", False))}
                            for nombre, datos in self._meta.items()
                        ]
                    },
                    indent=2,
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            os.replace(provisorio, definitivo)
        except OSError as error:
            _log.warning("no se pudo escribir el indice de conceptos: %s", error)
