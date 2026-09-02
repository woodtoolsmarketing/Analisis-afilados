"""Detectores y clasificadores de estado de herramientas de afilado.

Hay dos familias, segun lo que se le pregunta a la imagen:

DETECCION / SEGMENTACION (donde esta la pieza, cuanto mide) -- protocolo `Detector`:
  - `DetectorYolo`: envoltorio sobre un modelo YOLO de ultralytics.
  - `DetectorGeometrico`: fallback por umbralizacion de Otsu que NO necesita modelo entrenado.
    Permite medir piezas y, sobre todo, RECOLECTAR DATOS desde el primer dia: sin el, el bucle
    de auto-entrenamiento no podria arrancar (no hay modelo => no hay detecciones => no hay
    imagenes que guardar => nunca hay modelo).

CLASIFICACION (esta sierra sirve o no) -- protocolo `Clasificador`:
  - `DetectorClasificador`: corre el checkpoint .pt con ultralytics. Necesita torch.
  - `DetectorClasificadorOnnx`: corre el modelo exportado a .onnx con onnxruntime o, si no
    esta, con cv2.dnn. NO necesita torch ni ultralytics, y da el mismo veredicto. Es la ruta
    para la PC del taller: 80 MB de RAM en vez de 440 MB, y ~250 MB de entorno en vez de
    1.2 GB. `crear_clasificador` elige uno u otro segun la extension de detector.pesos.

`ultralytics` y `torch` se importan de forma PEREZOSA (dentro de los metodos, en try/except),
y `onnxruntime` tambien. El resto del paquete debe poder importarse en una maquina que no
tenga ninguno de los tres.

Ninguna implementacion de `Detector` asigna `medida` a las detecciones: medir es
responsabilidad del pipeline, que es quien conoce la escala del frame.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np

from .config import AppConfig, DetectorConfig, FiltrosConfig, ruta_absoluta
from .filtros import limpiar_mascara
from .tipos import Clasificador, Deteccion, Detector, Veredicto

logger = logging.getLogger(__name__)

_MENSAJE_SIN_ULTRALYTICS = (
    "No se pudo importar 'ultralytics'. Instalalo con: pip install ultralytics\n"
    "Mientras tanto el sistema sigue funcionando con el detector geometrico: podes medir "
    "y recolectar imagenes para entrenar, solo que sin clasificar el estado de la pieza."
)


def _resolver_dispositivo(dispositivo: str) -> str:
    """Traduce "auto" a "cuda:0" o "cpu" segun lo que haya disponible."""
    if dispositivo != "auto":
        return dispositivo
    try:
        import torch
    except ImportError:
        return "cpu"
    return "cuda:0" if torch.cuda.is_available() else "cpu"


class DetectorYolo:
    """Detector basado en un modelo YOLO entrenado (ultralytics)."""

    def __init__(self, cfg: DetectorConfig, clases: list[str]) -> None:
        self._cfg = cfg
        self._clases = list(clases)
        self._modelo: Optional[Any] = None
        self._dispositivo = "cpu"
        self._nombre_modelo = Path(cfg.pesos).stem or "yolo"
        self._nombres_modelo: dict[int, str] = {}
        self._aviso_clases_emitido = False

        try:
            from ultralytics import YOLO
        except ImportError:
            logger.error(_MENSAJE_SIN_ULTRALYTICS)
            return

        ruta_pesos = ruta_absoluta(cfg.pesos)
        if not ruta_pesos.is_file():
            logger.error("No existe el archivo de pesos: %s", ruta_pesos)
            return

        self._dispositivo = _resolver_dispositivo(cfg.dispositivo)
        try:
            self._modelo = YOLO(str(ruta_pesos))
        except Exception as exc:  # el .pt puede estar corrupto o ser de otra version
            logger.error("No se pudo cargar el modelo %s: %s", ruta_pesos, exc)
            return

        nombres = getattr(self._modelo, "names", None)
        if isinstance(nombres, dict):
            self._nombres_modelo = {int(k): str(v) for k, v in nombres.items()}
        elif isinstance(nombres, (list, tuple)):
            self._nombres_modelo = {i: str(v) for i, v in enumerate(nombres)}
        logger.info(
            "Modelo cargado: %s en %s (clases: %s)",
            ruta_pesos,
            self._dispositivo,
            ", ".join(self._nombres_modelo.values()) or "desconocidas",
        )

    @property
    def disponible(self) -> bool:
        return self._modelo is not None

    @property
    def descripcion(self) -> str:
        if self._modelo is None:
            return "YOLO no disponible"
        return f"{self._nombre_modelo} ({self._dispositivo})"

    def _nombre_de_clase(self, clase_id: int) -> str:
        """Prioriza los nombres del MODELO: el config puede estar desincronizado del .pt."""
        if clase_id in self._nombres_modelo:
            return self._nombres_modelo[clase_id]
        if 0 <= clase_id < len(self._clases):
            return self._clases[clase_id]
        return f"clase_{clase_id}"

    def _avisar_si_difieren_clases(self) -> None:
        if self._aviso_clases_emitido or not self._nombres_modelo:
            return
        self._aviso_clases_emitido = True
        del_modelo = [self._nombres_modelo[k] for k in sorted(self._nombres_modelo)]
        if del_modelo != self._clases:
            logger.warning(
                "Las clases del modelo %s no coinciden con las del config %s. "
                "Se usan las del modelo; revisa configs/config.yaml.",
                del_modelo,
                self._clases,
            )

    def _mascaras_al_frame(
        self, mascaras: np.ndarray, alto: int, ancho: int
    ) -> list[np.ndarray]:
        """Lleva las mascaras de la resolucion del modelo (imgsz) a la del frame.

        INTER_NEAREST y no INTER_LINEAR: interpolar valores de una mascara binaria inventa
        bordes intermedios que luego falsean el area medida.
        """
        salida: list[np.ndarray] = []
        for mascara in mascaras:
            binaria = (mascara > 0.5).astype(np.uint8) * 255
            if binaria.shape[:2] != (alto, ancho):
                binaria = cv2.resize(
                    binaria, (ancho, alto), interpolation=cv2.INTER_NEAREST
                )
            salida.append(binaria)
        return salida

    def predecir(self, frame: np.ndarray) -> list[Deteccion]:
        if self._modelo is None:
            return []

        try:
            resultados = self._modelo.predict(
                frame,
                conf=self._cfg.confianza,
                iou=self._cfg.iou,
                imgsz=self._cfg.imgsz,
                device=self._dispositivo,
                max_det=self._cfg.max_detecciones,
                verbose=False,
            )
        except Exception as exc:
            logger.error("Fallo la inferencia: %s", exc)
            return []

        if not resultados:
            return []
        resultado = resultados[0]
        self._avisar_si_difieren_clases()

        cajas = getattr(resultado, "boxes", None)
        if cajas is None or len(cajas) == 0:
            return []

        xyxy = cajas.xyxy.cpu().numpy()
        confianzas = cajas.conf.cpu().numpy()
        clases_id = cajas.cls.cpu().numpy().astype(int)

        alto, ancho = frame.shape[:2]
        mascaras: list[np.ndarray] = []
        objeto_mascaras = getattr(resultado, "masks", None)
        if objeto_mascaras is not None and objeto_mascaras.data is not None:
            mascaras = self._mascaras_al_frame(
                objeto_mascaras.data.cpu().numpy(), alto, ancho
            )

        detecciones: list[Deteccion] = []
        for indice in range(len(xyxy)):
            x1, y1, x2, y2 = (float(v) for v in xyxy[indice])
            clase_id = int(clases_id[indice])
            detecciones.append(
                Deteccion(
                    clase_id=clase_id,
                    clase=self._nombre_de_clase(clase_id),
                    confianza=float(confianzas[indice]),
                    xyxy=(x1, y1, x2, y2),
                    mascara=mascaras[indice] if indice < len(mascaras) else None,
                )
            )
        return detecciones


class DetectorGeometrico:
    """Fallback sin modelo: separa la pieza del fondo por umbral de Otsu.

    Asume PIEZA OSCURA SOBRE FONDO CLARO (THRESH_BINARY_INV): el metal a contraluz o sobre mesa
    blanca queda como silueta. Si en tu taller la pieza es clara sobre mesa oscura, la mascara
    sale invertida (detectara el fondo como pieza): hay que cambiar THRESH_BINARY_INV por
    THRESH_BINARY.
    """

    def __init__(self, cfg: FiltrosConfig) -> None:
        self._cfg = cfg

    @property
    def descripcion(self) -> str:
        return "geometrico (sin modelo entrenado)"

    def predecir(self, frame: np.ndarray) -> list[Deteccion]:
        if frame.ndim == 3:
            gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gris = frame
        # El blur previo evita que el ruido del sensor parta el histograma que usa Otsu.
        suavizado = cv2.GaussianBlur(gris, (5, 5), 0)
        _, binaria = cv2.threshold(
            suavizado, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU
        )
        binaria = limpiar_mascara(
            binaria, self._cfg.kernel_morfologico, self._cfg.iteraciones_morfologicas
        )

        contornos, _ = cv2.findContours(
            binaria, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        alto, ancho = binaria.shape[:2]
        detecciones: list[Deteccion] = []
        for contorno in contornos:
            if cv2.contourArea(contorno) < self._cfg.area_minima_px:
                continue
            mascara = np.zeros((alto, ancho), dtype=np.uint8)
            cv2.drawContours(mascara, [contorno], -1, 255, thickness=cv2.FILLED)
            x, y, w, h = cv2.boundingRect(contorno)
            detecciones.append(
                Deteccion(
                    clase_id=-1,
                    clase="sin_clasificar",
                    confianza=0.0,
                    xyxy=(float(x), float(y), float(x + w), float(y + h)),
                    mascara=mascara,
                )
            )
        return detecciones


def crear_detector(cfg: AppConfig) -> Detector:
    """Elige el mejor detector disponible: YOLO si hay pesos y ultralytics, si no el geometrico."""
    ruta_pesos = ruta_absoluta(cfg.detector.pesos)
    if not ruta_pesos.is_file():
        logger.warning(
            "No se encontraron los pesos en %s. Se usa el detector geometrico: mide y permite "
            "recolectar datos, pero no clasifica el estado.",
            ruta_pesos,
        )
        return DetectorGeometrico(cfg.filtros)

    detector = DetectorYolo(cfg.detector, cfg.clases)
    if not detector.disponible:
        logger.warning(
            "Hay pesos en %s pero el modelo no se pudo cargar (falta ultralytics/torch o el "
            "archivo no es valido). Se usa el detector geometrico.",
            ruta_pesos,
        )
        return DetectorGeometrico(cfg.filtros)
    return detector


class DetectorClasificador:
    """Envoltorio sobre un modelo YOLO de CLASIFICACION (aprobada/rechazada/...).

    Juzga la imagen entera y devuelve un unico Veredicto. No localiza cajas ni mide: para el
    veredicto "esta sierra sirve o no" no hace falta ubicar el defecto, solo mirar el conjunto.
    `ultralytics`/`torch` se importan de forma perezosa, igual que en DetectorYolo.
    """

    def __init__(self, cfg: DetectorConfig, clases: list[str]) -> None:
        self._cfg = cfg
        self._clases = list(clases)
        self._modelo: Optional[Any] = None
        self._nombre_modelo = Path(cfg.pesos).stem or "clasificador"
        self._nombres_modelo: dict[int, str] = {}

        try:
            from ultralytics import YOLO
        except ImportError:
            logger.error(_MENSAJE_SIN_ULTRALYTICS)
            return

        ruta_pesos = ruta_absoluta(cfg.pesos)
        if not ruta_pesos.is_file():
            logger.error("No existe el archivo de pesos: %s", ruta_pesos)
            return

        self._dispositivo = _resolver_dispositivo(cfg.dispositivo)
        try:
            self._modelo = YOLO(str(ruta_pesos))
        except Exception as exc:  # el .pt puede estar corrupto o ser de otra version
            logger.error("No se pudo cargar el clasificador %s: %s", ruta_pesos, exc)
            return

        nombres = getattr(self._modelo, "names", None)
        if isinstance(nombres, dict):
            self._nombres_modelo = {int(k): str(v) for k, v in nombres.items()}
        elif isinstance(nombres, (list, tuple)):
            self._nombres_modelo = {i: str(v) for i, v in enumerate(nombres)}

    @property
    def disponible(self) -> bool:
        return self._modelo is not None

    @property
    def descripcion(self) -> str:
        if self._modelo is None:
            return "clasificador no disponible"
        return f"clasificador {self._nombre_modelo} ({self._dispositivo})"

    def clasificar(self, frame: np.ndarray) -> Optional[Veredicto]:
        """Devuelve el Veredicto de la imagen, o None si el modelo no esta o la inferencia falla."""
        if self._modelo is None:
            return None
        try:
            resultados = self._modelo.predict(
                frame,
                imgsz=self._cfg.imgsz,
                device=self._dispositivo,
                verbose=False,
            )
        except Exception as exc:
            logger.error("Fallo la clasificacion: %s", exc)
            return None
        if not resultados:
            return None

        probs = getattr(resultados[0], "probs", None)
        if probs is None:
            logger.error(
                "El modelo %s no devolvio probabilidades: no parece de clasificacion. "
                "Entrena con 'train --datos <carpeta>' o revisa el .pt.",
                self._nombre_modelo,
            )
            return None

        datos = probs.data.cpu().numpy()
        clase_id = int(probs.top1)
        confianza = float(probs.top1conf)
        clase = self._nombres_modelo.get(clase_id, self._clase_por_indice(clase_id))
        probabilidades = {
            self._nombres_modelo.get(i, self._clase_por_indice(i)): float(p)
            for i, p in enumerate(datos)
        }
        return Veredicto(
            clase=clase,
            clase_id=clase_id,
            confianza=confianza,
            probabilidades=probabilidades,
        )

    def _clase_por_indice(self, indice: int) -> str:
        if 0 <= indice < len(self._clases):
            return self._clases[indice]
        return f"clase_{indice}"


_SUFIJO_METADATOS = ".json"


def _leer_metadatos(ruta_modelo: Path) -> dict[str, Any]:
    """Lee el JSON hermano que deja scripts/exportar_onnx.py (<modelo>.onnx.json).

    Ese archivo lleva las clases y el imgsz REALES con los que se exporto el modelo. Sin el
    habria que confiar en configs/config.yaml, que puede haber quedado desincronizado: una
    lista de clases corrida en un lugar convierte "aprobada" en "rechazada" sin avisar.
    """
    ruta = Path(str(ruta_modelo) + _SUFIJO_METADATOS)
    if not ruta.is_file():
        return {}
    try:
        with open(ruta, "r", encoding="utf-8") as manejador:
            datos = json.load(manejador)
    except (OSError, ValueError) as exc:
        logger.warning("No se pudo leer %s (%s); se usan los datos del config.", ruta, exc)
        return {}
    return datos if isinstance(datos, dict) else {}


def _lado_de_metadatos(metadatos: dict[str, Any], por_defecto: int) -> int:
    """Saca el lado de entrada del JSON del modelo, tolerando como venga escrito.

    ultralytics guarda su propio imgsz como [alto, ancho], asi que un JSON copiado de ahi
    trae una lista y no un numero. Cualquier valor que no se entienda cae al del config con
    un aviso: un metadato mal escrito no puede tumbar el arranque en el taller, que es
    justamente donde nadie puede leer un traceback.
    """
    valor = metadatos.get("imgsz")
    if valor is None:
        return por_defecto

    if isinstance(valor, (list, tuple)):
        numeros = [v for v in valor if isinstance(v, (int, float)) and not isinstance(v, bool)]
        if len(numeros) != len(valor) or not numeros or len(set(numeros)) != 1:
            logger.warning(
                "El JSON del modelo trae imgsz=%r, que no describe una entrada cuadrada. "
                "Se usa detector.imgsz=%d del config.",
                valor,
                por_defecto,
            )
            return por_defecto
        valor = numeros[0]

    try:
        lado = int(valor)
    except (TypeError, ValueError):
        logger.warning(
            "El JSON del modelo trae un imgsz ilegible (%r); se usa detector.imgsz=%d "
            "del config.",
            valor,
            por_defecto,
        )
        return por_defecto

    if lado < 1:
        logger.warning(
            "El JSON del modelo trae imgsz=%d, que es imposible; se usa detector.imgsz=%d "
            "del config.",
            lado,
            por_defecto,
        )
        return por_defecto
    return lado


def _clases_de_metadatos(
    metadatos: dict[str, Any], por_defecto: list[str], nombre_modelo: str
) -> list[str]:
    """Saca la lista de clases del JSON del modelo; si no sirve, usa la del config.

    Una lista VACIA no es una lista valida: dejaria al clasificador sin nombres y el operario
    veria "clase_0" y "clase_1" en la pantalla en vez de "aprobada" y "rechazada".
    """
    clases = metadatos.get("clases")
    if isinstance(clases, list) and clases and all(isinstance(c, str) for c in clases):
        if list(clases) != por_defecto:
            logger.warning(
                "Las clases del modelo %s no coinciden con las del config %s. "
                "Se usan las del modelo; revisa configs/config.yaml.",
                list(clases),
                por_defecto,
            )
        return list(clases)

    if not metadatos:
        logger.warning(
            "El modelo %s no trae su JSON de metadatos: se confia en las clases y el imgsz "
            "de configs/config.yaml. Reexportalo con scripts/exportar_onnx.py para que el "
            ".onnx viaje con su propia lista de clases.",
            nombre_modelo,
        )
    else:
        logger.warning(
            "El JSON de %s no trae una lista de clases utilizable (%r); se usan las del "
            "config: %s.",
            nombre_modelo,
            clases,
            por_defecto,
        )
    return list(por_defecto)


def _a_probabilidades(salida: np.ndarray) -> np.ndarray:
    """Normaliza la salida del modelo a probabilidades que sumen 1.

    La cabeza de clasificacion de ultralytics ya aplica softmax dentro del grafo, asi que
    normalmente esto no toca nada (verificado: la salida del .onnx suma 1.0000). Se deja
    porque un export con otro opset podria devolver logits crudos, y entonces el overlay
    mostraria "confianza 12.4" sin que nadie entienda por que.
    """
    valores = np.asarray(salida, dtype=np.float64).reshape(-1)
    if valores.size == 0:
        return valores
    if float(valores.min()) < 0.0 or abs(float(valores.sum()) - 1.0) > 1e-3:
        exponenciales = np.exp(valores - valores.max())
        return exponenciales / exponenciales.sum()
    return valores


def preparar_entrada_clasificacion(frame: np.ndarray, lado: int) -> np.ndarray:
    """Deja el frame como lo espera el modelo: tensor (1, 3, lado, lado) float32 en 0..1.

    Replica EXACTAMENTE lo que hace ultralytics antes de la red:

        Resize(lado corto -> lado, con antialias) -> CenterCrop(lado) -> BGR a RGB -> /255 -> CHW

    Cada paso importa, y el orden tambien. Deformar la imagen a lado x lado de una sola vez,
    en vez de reducir y recortar el centro, CAMBIA la clase que sale: verificado contra el
    modelo real. Y el antialias no es cosmetico: INTER_AREA es el equivalente de OpenCV al
    antialias de torchvision al REDUCIR, y es el unico interpolador que reproduce el .pt.
    Medido sobre el modelo binario del proyecto y 34 fotos del taller:

        INTER_AREA      34/34 mismo veredicto, diferencia media de probabilidad 0.0011
        INTER_LINEAR    34/34 mismo veredicto, diferencia media 0.0129  (11 veces peor)
        INTER_CUBIC     34/34 mismo veredicto, diferencia media 0.0180  (16 veces peor)

    Al AMPLIAR (un frame mas chico que el lado pedido) INTER_AREA degenera, asi que ahi se
    usa INTER_LINEAR, que es lo que hace torchvision cuando no hay nada que filtrar.
    """
    if frame.ndim == 2:
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    alto, ancho = frame.shape[:2]
    if alto == 0 or ancho == 0:
        raise ValueError("el frame esta vacio: no se puede clasificar")
    if lado < 1:
        raise ValueError(f"el lado de entrada debe ser >= 1, se recibio {lado}")

    escala = lado / min(alto, ancho)
    # El max() cubre el redondeo: para ciertas relaciones de aspecto round() puede devolver
    # lado-1, y entonces el recorte centrado saldria un pixel corto, con forma invalida.
    nuevo_ancho = max(lado, int(round(ancho * escala)))
    nuevo_alto = max(lado, int(round(alto * escala)))
    interpolacion = cv2.INTER_AREA if escala < 1.0 else cv2.INTER_LINEAR
    redimensionado = cv2.resize(frame, (nuevo_ancho, nuevo_alto), interpolation=interpolacion)

    x0 = (nuevo_ancho - lado) // 2
    y0 = (nuevo_alto - lado) // 2
    recorte = redimensionado[y0 : y0 + lado, x0 : x0 + lado]
    rgb = cv2.cvtColor(recorte, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return np.ascontiguousarray(rgb.transpose(2, 0, 1)[None])


def _abrir_onnx(ruta: Path) -> tuple[str, Any]:
    """Devuelve (nombre_del_backend, funcion_de_inferencia) para el .onnx indicado.

    Prefiere onnxruntime, que es unas tres veces mas rapido; si no esta instalado usa
    cv2.dnn, que NO agrega ninguna dependencia a la maquina (opencv ya es obligatorio en
    este proyecto por el modulo aruco). Los dos backends dan el mismo resultado
    --verificado: coinciden dentro de 1e-4 sobre las fotos del taller--, asi que cual haya
    en la maquina no cambia el veredicto, solo la velocidad.
    """
    try:
        import onnxruntime
    except ImportError:
        onnxruntime = None

    if onnxruntime is not None:
        sesion = onnxruntime.InferenceSession(str(ruta), providers=["CPUExecutionProvider"])
        nombre_entrada = sesion.get_inputs()[0].name

        def correr_con_onnxruntime(entrada: np.ndarray) -> np.ndarray:
            return sesion.run(None, {nombre_entrada: entrada})[0]

        return "onnxruntime", correr_con_onnxruntime

    red = cv2.dnn.readNetFromONNX(str(ruta))

    def correr_con_opencv(entrada: np.ndarray) -> np.ndarray:
        red.setInput(entrada)
        return red.forward()

    return "cv2.dnn", correr_con_opencv


class DetectorClasificadorOnnx:
    """Clasificador que corre el modelo exportado a ONNX, SIN torch ni ultralytics.

    Es la version pensada para la PC del taller. Medido en un i5-12400 sin GPU, contra la
    ruta .pt con ultralytics:

        RAM del proceso      440 MB  ->  80 MB de pico (48 MB en regimen)
        entorno en disco     1.2 GB  ->  ~250 MB   (torch solo pesa 533 MB)
        cargar el modelo     1.2 s   ->  0.05 s
        inferencia a 224 px  12 ms   ->  5.8 ms con cv2.dnn, 1.8 ms con onnxruntime

    El veredicto es el mismo que daria el .pt: verificado sobre el modelo binario del
    proyecto y 34 fotos del taller, 34 de 34 coinciden, con una diferencia media de
    probabilidad de 0.0011. Esa fidelidad depende enteramente del preprocesado; ver
    preparar_entrada_clasificacion.
    """

    def __init__(self, cfg: DetectorConfig, clases: list[str]) -> None:
        self._cfg = cfg
        self._clases = list(clases)
        self._correr: Optional[Any] = None
        self._backend = "ninguno"
        self._nombre_modelo = Path(cfg.pesos).stem or "clasificador"
        self._lado = int(cfg.imgsz)

        ruta = ruta_absoluta(cfg.pesos)
        if not ruta.is_file():
            logger.error("No existe el modelo ONNX: %s", ruta)
            return

        # Las clases y el tamano de entrada son datos INDEPENDIENTES: un JSON puede traer
        # uno bien y el otro mal, y cada uno cae por su cuenta a lo que diga el config.
        metadatos = _leer_metadatos(ruta)
        self._lado = _lado_de_metadatos(metadatos, int(cfg.imgsz))
        self._clases = _clases_de_metadatos(metadatos, self._clases, ruta.name)

        try:
            self._backend, self._correr = _abrir_onnx(ruta)
        except Exception as exc:  # el .onnx puede estar corrupto o traer ops no soportadas
            logger.error("No se pudo cargar el modelo ONNX %s: %s", ruta, exc)
            self._correr = None
            return

        # Una inferencia de prueba en el arranque. Si el lado no es el que espera el grafo,
        # es mejor enterarse ahora que en el primer frame, con el operario esperando.
        try:
            self._correr(np.zeros((1, 3, self._lado, self._lado), dtype=np.float32))
        except Exception as exc:
            logger.error(
                "El modelo %s no acepta entradas de %dx%d px (%s). Corregi detector.imgsz "
                "en el config o reexporta el .onnx con ese tamano.",
                ruta.name,
                self._lado,
                self._lado,
                exc,
            )
            self._correr = None
            return

        logger.info(
            "Modelo ONNX cargado: %s a %d px via %s (clases: %s)",
            ruta.name,
            self._lado,
            self._backend,
            ", ".join(self._clases),
        )

    @property
    def disponible(self) -> bool:
        return self._correr is not None

    @property
    def descripcion(self) -> str:
        if self._correr is None:
            return "clasificador ONNX no disponible"
        return f"clasificador {self._nombre_modelo}.onnx ({self._backend}, {self._lado} px)"

    def clasificar(self, frame: np.ndarray) -> Optional[Veredicto]:
        """Devuelve el Veredicto de la imagen, o None si el modelo no esta o algo fallo."""
        if self._correr is None:
            return None
        try:
            entrada = preparar_entrada_clasificacion(frame, self._lado)
            salida = self._correr(entrada)
        except Exception as exc:
            logger.error("Fallo la clasificacion ONNX: %s", exc)
            return None

        probabilidades = _a_probabilidades(salida)
        if probabilidades.size == 0:
            logger.error("El modelo %s devolvio una salida vacia.", self._nombre_modelo)
            return None

        clase_id = int(np.argmax(probabilidades))
        return Veredicto(
            clase=self._clase_por_indice(clase_id),
            clase_id=clase_id,
            confianza=float(probabilidades[clase_id]),
            probabilidades={
                self._clase_por_indice(indice): float(p)
                for indice, p in enumerate(probabilidades)
            },
        )

    def _clase_por_indice(self, indice: int) -> str:
        if 0 <= indice < len(self._clases):
            return self._clases[indice]
        return f"clase_{indice}"


def crear_clasificador(cfg: AppConfig) -> Optional[Clasificador]:
    """Devuelve el clasificador si hay modelo; si no, None (modo recoleccion).

    LA EXTENSION MANDA: un .onnx se corre sin torch (es la PC del taller) y cualquier otra
    cosa se trata como checkpoint de ultralytics (.pt, la PC de entrenamiento). No se
    adivina ni se prefiere uno sobre otro: se carga lo que diga detector.pesos en el config,
    para que nadie crea que actualizo el modelo cuando en realidad sigue corriendo el otro
    archivo.

    Que devuelva None NO es un error: significa que todavia no hay modelo entrenado y el
    programa en vivo entra en modo recoleccion (juntar fotos a mano con las teclas).
    """
    ruta_pesos = ruta_absoluta(cfg.detector.pesos)
    if not ruta_pesos.is_file():
        logger.warning(
            "No hay modelo de clasificacion en %s. Modo recoleccion: usa las teclas para "
            "guardar fotos por clase y despues entrena.",
            ruta_pesos,
        )
        return None

    es_onnx = ruta_pesos.suffix.lower() == ".onnx"
    clasificador: Clasificador = (
        DetectorClasificadorOnnx(cfg.detector, cfg.clases)
        if es_onnx
        else DetectorClasificador(cfg.detector, cfg.clases)
    )
    if not clasificador.disponible:
        motivo = (
            "el .onnx esta corrupto o usa operaciones que este runtime no soporta"
            if es_onnx
            else "falta ultralytics/torch o el archivo no es un checkpoint valido"
        )
        logger.warning(
            "Hay un modelo en %s pero no se pudo cargar (%s). Modo recoleccion.",
            ruta_pesos,
            motivo,
        )
        return None
    return clasificador
