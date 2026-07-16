"""Almacen del bucle de auto-entrenamiento: guarda en disco lo que la IA vio y lo que penso.

Dos decisiones de diseno mandan sobre todo este modulo:

1) LA IMAGEN SE GUARDA LIMPIA, sin un solo pixel dibujado encima.
   El overlay (cajas, textos, colores) es informacion para el operario, no para la red.
   Si se guardara el frame anotado, al reentrenar la red aprenderia a asociar "desgastado"
   con la presencia de un rectangulo rojo, no con el filo real de la herramienta: el modelo
   se volveria un detector de sus propios dibujos. La copia anotada, si se pide, vive
   aparte en revision/ y jamas entra al dataset.

2) EL ARCHIVO GEMELO DE LA IMAGEN ES EL PRE-ETIQUETADO YOLO, NO EL INFORME LEGIBLE.
   YOLO y Roboflow buscan las etiquetas por convencion de nombre: para "foto.jpg" esperan
   "foto.txt" con las cajas. Si ese .txt fuese el informe en castellano, el importador lo
   leeria como etiquetas y reventaria (o peor: importaria basura silenciosamente).
   Por eso el gemelo es el pre-etiquetado en formato YOLO, que ademas ahorra trabajo: en
   Roboflow solo hay que CORREGIR las cajas que la IA erro, no dibujarlas de cero.
   El informe legible se va a reportes/, fuera del alcance del importador.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np

from .config import FeedbackConfig, ruta_absoluta
from .medicion import contorno_principal
from .tipos import Deteccion, ResultadoFrame

_registro = logging.getLogger(__name__)

_CARACTERES_VALIDOS = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_")


def _sanear(nombre: str) -> str:
    """Deja solo caracteres seguros para un nombre de archivo multiplataforma."""
    limpio = "".join(c if c in _CARACTERES_VALIDOS else "_" for c in nombre).strip("_")
    return limpio or "captura"


def _json_por_defecto(valor: Any) -> Any:
    """Convierte tipos numpy a nativos: json.dumps revienta con np.float32/np.int64."""
    if isinstance(valor, np.integer):
        return int(valor)
    if isinstance(valor, np.floating):
        return float(valor)
    if isinstance(valor, np.ndarray):
        return valor.tolist()
    if isinstance(valor, datetime):
        return valor.isoformat(timespec="milliseconds")
    raise TypeError(f"tipo no serializable a JSON: {type(valor).__name__}")


def _acotar(valor: float) -> float:
    """Recorta a [0,1]: una caja puede salirse del frame y YOLO rechaza coordenadas fuera."""
    return min(1.0, max(0.0, float(valor)))


def _linea_caja(indice_clase: int, det: Deteccion, ancho: int, alto: int) -> str:
    """Pre-etiquetado YOLO de deteccion: clase cx cy w h, normalizados 0..1."""
    x1, y1, x2, y2 = (float(v) for v in det.xyxy)
    cx = _acotar((x1 + x2) / 2.0 / ancho)
    cy = _acotar((y1 + y2) / 2.0 / alto)
    an = _acotar(abs(x2 - x1) / ancho)
    al = _acotar(abs(y2 - y1) / alto)
    return f"{indice_clase} {cx:.6f} {cy:.6f} {an:.6f} {al:.6f}"


def _linea_segmento(indice_clase: int, det: Deteccion, ancho: int, alto: int) -> Optional[str]:
    """Pre-etiquetado YOLO de segmentacion: clase x1 y1 x2 y2 ..., normalizados 0..1.

    Devuelve None si la mascara no da un poligono de al menos 3 puntos; el llamador
    cae entonces al formato de caja.
    """
    if det.mascara is None:
        return None
    contorno = contorno_principal(det.mascara)
    if contorno is None:
        return None
    perimetro = cv2.arcLength(contorno, True)
    # Un contorno crudo trae cientos de puntos por el escalonado de la mascara; approxPolyDP
    # lo reduce a un poligono editable a mano en Roboflow sin perder la forma.
    aproximado = cv2.approxPolyDP(contorno, 0.002 * perimetro, True)
    puntos = aproximado.reshape(-1, 2)
    if len(puntos) < 3:
        return None
    partes = [str(indice_clase)]
    for x, y in puntos:
        partes.append(f"{_acotar(float(x) / ancho):.6f}")
        partes.append(f"{_acotar(float(y) / alto):.6f}")
    return " ".join(partes)


def _texto_medida(det: Deteccion) -> str:
    """Describe las medidas de una deteccion en castellano llano."""
    if det.medida is None:
        return "sin medida"
    med = det.medida
    if med.fiable and med.largo_mm is not None and med.ancho_mm is not None:
        base = f"{med.largo_mm:.2f} x {med.ancho_mm:.2f} mm"
        if med.area_mm2 is not None:
            base += f", area {med.area_mm2:.2f} mm2"
    else:
        base = f"{med.largo_px:.1f} x {med.ancho_px:.1f} px, area {med.area_px:.1f} px2 (SIN ESCALA)"
    return f"{base}, inclinada {med.angulo_deg:.1f} grados"


class AlmacenFeedback:
    """Persiste capturas del bucle de feedback listas para reentrenar.

    Estructura en disco, por dia:
        <directorio>/<AAAA-MM-DD>/imagenes/<etiqueta>_<sello>.jpg   imagen LIMPIA
        <directorio>/<AAAA-MM-DD>/etiquetas/<etiqueta>_<sello>.txt  pre-etiquetado YOLO
        <directorio>/<AAAA-MM-DD>/reportes/<etiqueta>_<sello>.json  lo que penso la IA
        <directorio>/<AAAA-MM-DD>/reportes/<etiqueta>_<sello>.txt   informe legible
        <directorio>/<AAAA-MM-DD>/revision/<etiqueta>_<sello>_anotada.jpg
    """

    def __init__(self, cfg: FeedbackConfig, clases: list[str]) -> None:
        self._cfg = cfg
        self._clases = list(clases)
        self._base = ruta_absoluta(cfg.directorio)
        self._total = 0

    @property
    def total_guardados(self) -> int:
        return self._total

    def guardar(
        self,
        frame_limpio: np.ndarray,
        resultado: ResultadoFrame,
        etiqueta: str = "fallo",
        frame_anotado: Optional[np.ndarray] = None,
    ) -> Path:
        """Guarda imagen limpia + pre-etiquetado + reportes. Devuelve la ruta de la imagen."""
        ahora = datetime.now()
        alto, ancho = frame_limpio.shape[:2]
        dia = self._base / ahora.strftime("%Y-%m-%d")
        dir_imagenes = dia / "imagenes"
        dir_etiquetas = dia / "etiquetas"
        dir_reportes = dia / "reportes"
        for carpeta in (dir_imagenes, dir_etiquetas, dir_reportes):
            carpeta.mkdir(parents=True, exist_ok=True)

        nombre = self._nombre_libre(dir_imagenes, _sanear(etiqueta), ahora)
        ruta_imagen = dir_imagenes / f"{nombre}.jpg"

        calidad = [int(cv2.IMWRITE_JPEG_QUALITY), int(self._cfg.calidad_jpg)]
        if not cv2.imwrite(str(ruta_imagen), frame_limpio, calidad):
            raise OSError(f"no se pudo escribir la imagen en {ruta_imagen}")

        (dir_etiquetas / f"{nombre}.txt").write_text(
            self._pre_etiquetado(resultado.detecciones, ancho, alto), encoding="utf-8"
        )
        (dir_reportes / f"{nombre}.txt").write_text(
            self._informe(resultado, etiqueta, ahora, ruta_imagen), encoding="utf-8"
        )
        if self._cfg.guardar_json:
            datos = self._datos(resultado, etiqueta, ahora, ancho, alto)
            (dir_reportes / f"{nombre}.json").write_text(
                json.dumps(datos, indent=2, ensure_ascii=False, default=_json_por_defecto),
                encoding="utf-8",
            )
        if self._cfg.guardar_anotada and frame_anotado is not None:
            dir_revision = dia / "revision"
            dir_revision.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(dir_revision / f"{nombre}_anotada.jpg"), frame_anotado, calidad)

        self._total += 1
        _registro.info("feedback guardado: %s (%d detecciones)", ruta_imagen, len(resultado.detecciones))
        return ruta_imagen

    def _nombre_libre(self, dir_imagenes: Path, etiqueta: str, ahora: datetime) -> str:
        """Sello con milisegundos; si dos capturas caen en el mismo ms, agrega sufijo."""
        sello = ahora.strftime("%Y%m%d_%H%M%S_") + f"{ahora.microsecond // 1000:03d}"
        base = f"{etiqueta}_{sello}"
        nombre = base
        repeticion = 1
        while (dir_imagenes / f"{nombre}.jpg").exists():
            nombre = f"{base}_{repeticion}"
            repeticion += 1
        return nombre

    def _indice_clase(self, det: Deteccion) -> int:
        """Indice de clase para el pre-etiquetado; el geometrico (clase_id<0) va a 0."""
        if 0 <= det.clase_id < len(self._clases):
            return det.clase_id
        return 0

    def _pre_etiquetado(self, detecciones: list[Deteccion], ancho: int, alto: int) -> str:
        """Arma el .txt en formato YOLO. Con mascara usa segmentacion; sin ella, caja.

        La presencia de mascara es la senal de que el modelo corrio en modo "segment":
        el detector solo la rellena en esa tarea.
        """
        lineas: list[str] = []
        for det in detecciones:
            indice = self._indice_clase(det)
            linea = _linea_segmento(indice, det, ancho, alto)
            if linea is None:
                linea = _linea_caja(indice, det, ancho, alto)
            lineas.append(linea)
        return "\n".join(lineas) + ("\n" if lineas else "")

    def _datos(
        self, resultado: ResultadoFrame, etiqueta: str, ahora: datetime, ancho: int, alto: int
    ) -> dict[str, Any]:
        esc = resultado.escala
        return {
            "instante": ahora.isoformat(timespec="milliseconds"),
            "etiqueta_operario": etiqueta,
            "modelo": resultado.modelo,
            "fps": round(float(resultado.fps), 2),
            "forma_frame": {"alto": int(alto), "ancho": int(ancho)},
            "roi": [int(v) for v in resultado.roi],
            "clases_configuradas": self._clases,
            "escala": {
                "valida": bool(esc.valida),
                "fuente": esc.fuente,
                "mm_por_px": None if esc.mm_por_px is None else float(esc.mm_por_px),
                "px_por_mm": None if esc.px_por_mm is None else float(esc.px_por_mm),
                "id_marcador": None if esc.id_marcador is None else int(esc.id_marcador),
                "error_lados": float(esc.error_lados),
                "frames_sin_marcador": int(esc.frames_sin_marcador),
            },
            "detecciones": [self._datos_deteccion(d) for d in resultado.detecciones],
            "descartadas": [self._datos_deteccion(d) for d in resultado.descartadas],
        }

    def _datos_deteccion(self, det: Deteccion) -> dict[str, Any]:
        datos: dict[str, Any] = {
            "clase": det.clase,
            "clase_id": int(det.clase_id),
            "clase_provisional": det.clase_id < 0,
            "confianza": round(float(det.confianza), 4),
            "xyxy": [round(float(v), 2) for v in det.xyxy],
            "tiene_mascara": det.mascara is not None,
            "descartada_por": det.descartada_por,
            "medida": None,
        }
        if det.medida is not None:
            med = det.medida
            datos["medida"] = {
                "largo_px": round(float(med.largo_px), 2),
                "ancho_px": round(float(med.ancho_px), 2),
                "area_px": round(float(med.area_px), 2),
                "angulo_deg": round(float(med.angulo_deg), 2),
                "centro_px": [round(float(v), 2) for v in med.centro_px],
                "largo_mm": None if med.largo_mm is None else round(float(med.largo_mm), 3),
                "ancho_mm": None if med.ancho_mm is None else round(float(med.ancho_mm), 3),
                "area_mm2": None if med.area_mm2 is None else round(float(med.area_mm2), 3),
                "fiable": bool(med.fiable),
            }
        return datos

    def _informe(
        self, resultado: ResultadoFrame, etiqueta: str, ahora: datetime, ruta_imagen: Path
    ) -> str:
        """Informe en castellano llano para que el operario entienda que penso la IA."""
        esc = resultado.escala
        alto, ancho = resultado.forma_frame
        lineas: list[str] = [
            "INFORME DE CAPTURA - Analisis de afilado",
            "=" * 60,
            f"Instante        : {ahora.strftime('%d/%m/%Y %H:%M:%S')}",
            f"Marcado como    : {etiqueta}",
            f"Imagen limpia   : {ruta_imagen.name}",
            f"Modelo          : {resultado.modelo}",
            f"Resolucion      : {ancho} x {alto} px    FPS: {resultado.fps:.1f}",
            f"Zona de interes : x{resultado.roi[0]}-{resultado.roi[2]}, y{resultado.roi[1]}-{resultado.roi[3]} px",
            "",
            "ESCALA",
            "-" * 60,
        ]
        if esc.valida and esc.mm_por_px is not None:
            lineas.append(
                f"Valida, origen '{esc.fuente}' (marcador id {esc.id_marcador}): "
                f"1 px = {esc.mm_por_px:.4f} mm"
            )
            if esc.frames_sin_marcador > 0:
                lineas.append(
                    f"Atencion: el marcador no se ve desde hace {esc.frames_sin_marcador} frames; "
                    "se esta reutilizando la ultima escala conocida."
                )
            if esc.error_lados > 0:
                lineas.append(
                    f"Desvio entre los lados del marcador: {esc.error_lados * 100:.1f} %. "
                    "Si es alto, la camara no esta perpendicular y las medidas se deforman."
                )
        else:
            lineas.append(
                "NO HAY ESCALA VALIDA: no se vio el marcador ArUco. Las medidas solo estan "
                "en pixeles y no se pueden convertir a milimetros."
            )
        lineas += [
            "",
            f"OBJETOS DETECTADOS: {len(resultado.detecciones)}",
            "-" * 60,
        ]
        if resultado.detecciones:
            for numero, det in enumerate(resultado.detecciones, start=1):
                provisional = " (clase provisional: detector geometrico, sin modelo)" if det.clase_id < 0 else ""
                lineas.append(
                    f"{numero}. {det.clase} con {det.confianza * 100:.1f} % de confianza{provisional}"
                )
                lineas.append(f"   Medidas: {_texto_medida(det)}")
        else:
            lineas.append("Ninguno supero los filtros.")
        lineas += [
            "",
            f"OBJETOS DESCARTADOS: {len(resultado.descartadas)}",
            "-" * 60,
        ]
        if resultado.descartadas:
            for numero, det in enumerate(resultado.descartadas, start=1):
                lineas.append(
                    f"{numero}. {det.clase} ({det.confianza * 100:.1f} %) descartado por: "
                    f"{_motivo(det.descartada_por)}"
                )
        else:
            lineas.append("Ninguno.")
        lineas += [
            "",
            "QUE HACER CON ESTA CAPTURA",
            "-" * 60,
            "La imagen esta limpia (sin dibujos) y lista para entrenar. Su archivo gemelo en",
            "etiquetas/ es el pre-etiquetado en formato YOLO: importalo en Roboflow y CORREGI",
            "solo lo que la IA erro, no hace falta dibujar las cajas de cero.",
            "",
        ]
        return "\n".join(lineas)


def _motivo(codigo: Optional[str]) -> str:
    """Traduce el codigo tecnico de descarte a una frase entendible."""
    motivos = {
        "roi": "su centro cayo fuera de la zona de interes",
        "area_minima": "es mas chico que el area minima admitida (posible aserrin o ruido)",
        "area_maxima": "es mas grande que el area maxima admitida (posible fondo o sombra)",
        "confianza": "la confianza quedo por debajo del umbral",
    }
    if codigo is None:
        return "motivo no informado"
    return motivos.get(codigo, codigo)
