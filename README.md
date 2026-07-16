# Analisis-afilado

Sistema de inspeccion visual de herramientas de afilado (dientes de widia, diamante PCD,
sierras circulares, fresas, insertos) con una camara web comun y una computadora.

---

## 1. Que hace el sistema

Una camara mirando la mesa desde arriba filma la pieza. El programa mira cada imagen que
entra, encuentra la herramienta, la mide en milimetros y dice en que estado esta: **ok**,
**desgastado**, **fisura**, **astillado** u **oxido**. Todo eso aparece en la pantalla en
vivo, con la pieza recuadrada, el estado escrito al lado y las medidas en milimetros.
Para saber cuanto mide un milimetro en la pantalla, el sistema usa un cuadradito impreso
en blanco y negro (un **marcador ArUco**) que se apoya al lado de la pieza: como el
programa sabe cuanto mide ese cuadrado de verdad, puede deducir el tamano de todo lo demas.

Lo importante es la segunda mitad: el sistema **aprende de los errores del operario que lo
corrige**. Cuando la maquina se equivoca —dice "ok" a un diente astillado, o marca fisura
donde solo hay una mancha de aceite— el operario aprieta una tecla y el sistema guarda esa
foto en el disco junto con un informe de lo que penso. Esas fotos se corrigen despues con
calma, se le vuelven a dar al programa para que estudie, y el sistema se vuelve mas
certero. Cuanto mas se lo usa y se lo corrige, mejor anda. No hace falta saber programar
para alimentarlo: alcanza con apretar una tecla cuando se manda una macana.

---

## 2. AVISO IMPORTANTE DE COMPATIBILIDAD

> ### Usar **Python 3.12**. Python 3.14 **NO sirve**.
>
> Las librerias de inteligencia artificial que usa el sistema (`torch` y `ultralytics`) todavia
> no publicaron version compatible con Python 3.14. Si se instala con 3.14, la instalacion
> falla o el programa arranca sin poder usar el modelo.
>
> Descargar Python 3.12 desde https://www.python.org/downloads/ y, durante la instalacion,
> tildar la casilla **"Add Python to PATH"**.

Para verificar que version esta activa, abrir PowerShell y escribir:

```powershell
python --version
```

Tiene que responder `Python 3.12.x`.

---

## 3. Instalacion

### 3.1 Instalacion normal (sin GPU)

Abrir PowerShell en la carpeta del proyecto y ejecutar:

```powershell
cd C:\Users\WoodTools-02\Desktop\vscode\Analisis-afilado
.\scripts\setup.ps1
```

El script crea el entorno virtual (`.venv`), instala todo lo que hace falta y deja el
proyecto listo. Puede tardar varios minutos: se baja bastante.

Si Windows se queja de que no puede ejecutar scripts, correr una sola vez:

```powershell
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
```

Cada vez que se abre una consola nueva para usar el sistema, hay que **activar el entorno**:

```powershell
.\.venv\Scripts\Activate.ps1
```

Se nota que quedo activo porque la linea de la consola empieza con `(.venv)`.

### 3.2 Si la maquina tiene placa NVIDIA (GPU)

Por defecto se instala la version de `torch` que trabaja con el procesador (CPU). Anda,
pero **entrenar en CPU es lentisimo** (dias en vez de horas). Si la maquina tiene una placa
NVIDIA, conviene instalar la version con CUDA.

Primero verificar que la placa se ve:

```powershell
nvidia-smi
```

Si responde con una tabla y un numero de version de CUDA, con el entorno activado:

```powershell
pip uninstall -y torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

Para confirmar que quedo bien:

```powershell
python -c "import torch; print(torch.cuda.is_available())"
```

Tiene que imprimir `True`. Si imprime `False`, se instalo la version de CPU o falta el
driver de la placa.

> El archivo `configs/config.yaml` tiene `dispositivo: auto`, asi que el sistema usa la
> placa sola si la encuentra. No hay que cambiar nada mas.

---

## 4. Puesta a punto FISICA (esto define si el sistema sirve o no)

Esta seccion vale mas que todo el software. Una camara mal puesta arruina las medidas y no
hay programa que lo arregle despues.

### 4.1 La camara

| Requisito | Por que |
|---|---|
| **A 90 grados exactos** sobre la mesa (cenital, perfectamente a plomo) | Cualquier inclinacion deforma la imagen: la pieza se ve como un trapecio y las medidas salen mal. Usar un nivel o una plomada. |
| **Firme, atornillada, que no se mueva** | Si se corre, cambia la escala y hay que recalibrar. |
| **Autofoco APAGADO** (ya viene asi en la configuracion) | Si el lente reenfoca solo, el tamano aparente del marcador cambia y todas las medidas en milimetros se corren. |
| Siempre a la misma altura | Cambiar la altura cambia la escala. |

El programa avisa en pantalla en rojo **"CAMARA INCLINADA"** si detecta que el marcador se
ve deformado. Si aparece ese cartel, la camara no esta a 90 grados: corregirla antes de
medir nada.

### 4.2 La luz

El metal pulido espeja y **ciega a la IA**: un reflejo blanco tapa el filo y el sistema no
ve nada, o ve una fisura donde solo hay un brillo.

- **Luz difusa**: domo de iluminacion o aro LED. Nunca una lampara puntual directa.
- **Filtro polarizador** en la camara para matar los brillos especulares.
- **Siempre la misma luz.** El sistema aprende con las fotos que se le dan; si se entrena
  con luz de domo y despues se trabaja con el tubo del techo, no reconoce nada.

### 4.3 El marcador ArUco

El cuadradito de referencia se genera asi:

```powershell
python .\scripts\make_aruco.py --diccionario DICT_4X4_50 --id 0 --lado-mm 30 --dpi 300 --salida marcador.png
```

Y despues:

| Regla | Por que |
|---|---|
| Imprimir **al 100%**, sin "ajustar a pagina" ni "escalar" | Si se imprime al 96%, el marcador mide 28.8 mm en vez de 30 y **todas** las medidas salen mal en ese mismo porcentaje. |
| **Verificar con calibre** el lado impreso | Es el patron de todo el sistema. Si mide 29.7, poner `lado_mm: 29.7` en `configs/config.yaml`. Medir, no confiar. |
| Papel **MATE** | El papel brillante espeja y el marcador no se detecta. |
| **NUNCA plastificar** | El plastico refleja y arruina la deteccion. Si se ensucia, se imprime otro. Es papel. |
| Pegarlo plano, sin arrugas ni ondas | Un marcador arrugado da una escala equivocada. |

### 4.4 LA REGLA DE ORO: el marcador va a la MISMA ALTURA que la cara que se mide

Esta es la limitacion principal del metodo y hay que entenderla.

La camara es un ojo solo: no sabe de profundidad. **Lo que esta mas cerca del lente se ve
mas grande.** Si el marcador esta apoyado en la mesa y la pieza tiene espesor, la cara de
arriba de la pieza esta mas cerca de la camara que el marcador, entonces se ve mas grande
de lo que es, y el sistema informa una medida **inflada**. Eso se llama **error de
paralaje**.

**Ejemplo del disco de 4 cm de espesor:**

```
        [ CAMARA ]
             |
             |                        <-- la cara superior del disco esta 4 cm
             |   ___________              MAS CERCA del lente que el marcador
             |  |           |   <-- cara que se mide (arriba del disco)
             |  |   DISCO   |  4 cm de espesor
   [ARUCO]___|__|___________|___
   ===========MESA=============    <-- el marcador esta aca abajo

   RESULTADO: el disco se ve mas grande de lo que es.
              Las medidas salen infladas. MAL.
```

**La solucion:** subir el marcador hasta la altura de la cara que se mide. Un taco de
madera, un bloque patron, un soporte hecho a medida: cualquier cosa que deje el marcador
en el mismo plano que la superficie que interesa medir.

```
        [ CAMARA ]
             |
             |   ___________
   [ARUCO]   |  |           |   <-- marcador y cara que se mide,
   |  taco|  |  |   DISCO   |       AMBOS al mismo nivel
   |______|__|__|___________|___
   ===========MESA=============

   RESULTADO: medidas correctas. BIEN.
```

> **Practica recomendada:** tener un juego de tacos de altura conocida (10, 20, 40 mm) con
> el marcador pegado arriba, y usar el que corresponda al espesor de la pieza del dia.

---

## 5. Uso diario

Con el entorno activado:

```powershell
python -m afilado.cli.run_live
```

Opciones que se pueden agregar:

| Opcion | Para que sirve |
|---|---|
| `--config <ruta>` | Usar otro archivo de configuracion. |
| `--fuente <n o ruta>` | Elegir otra camara (`--fuente 1`) o reproducir un video grabado. |
| `--pesos <ruta.pt>` | Probar otro modelo entrenado sin tocar la configuracion. |
| `--conf 0.9` | Subir o bajar el umbral de confianza para esta corrida. |
| `--sin-roi` | Analizar el frame completo, sin recortar la zona util. |
| `--grabar salida.mp4` | Grabar la sesion en video. |

### 5.1 Teclas

| Tecla | Que hace |
|---|---|
| `q` o `ESC` | Salir. |
| **`e`** | **Marcar ERROR de la IA.** Guarda la foto para reentrenar. **Es la tecla mas importante del sistema.** |
| `g` | Guardar un ejemplo BUENO (la IA acerto). Tambien sirve para el dataset. |
| `espacio` | Capturar sin dar juicio (simplemente guardar esta imagen). |
| `r` | Reiniciar la calibracion (si se movio el marcador o la camara). |
| `p` | Pausa. |
| `h` | Mostrar u ocultar la ayuda. |
| `c` | Ciclar el ROI (la zona util recortada). |

Cuando se guarda algo, aparece una confirmacion en pantalla durante un segundo y medio.

### 5.2 Que significa cada color

| Color | Significado |
|---|---|
| 🟩 **Verde** | Clase OK. La pieza esta bien. |
| 🟥 **Rojo** | Clase de defecto: desgastado, fisura, astillado u oxido. |
| 🟨 **Amarillo** | `sin_clasificar`. El sistema encontro un objeto y lo midio, pero todavia no hay modelo entrenado que diga que estado tiene. Es lo normal el dia 1. |

### 5.3 Carteles de aviso

| Cartel en rojo | Que hacer |
|---|---|
| **CAMARA INCLINADA** | La camara no esta a 90 grados. Corregirla. Las medidas no son confiables. |
| **SIN ESCALA - medidas no fiables** | No se ve el marcador ArUco. El sistema sigue detectando y midiendo en pixeles, pero **no informa milimetros**. Destapar el marcador. |

> El sistema **nunca inventa milimetros**. Si no ve el marcador, no da medidas en mm. Punto.
> Si el marcador queda tapado un rato (una mano, la pieza), sigue usando la ultima escala
> conocida durante unos 90 cuadros (unos 3 segundos) y despues avisa que no hay escala.

---

## 6. EL CICLO DE APRENDIZAJE (la parte mas importante)

El sistema no se entrena una vez y listo. **Es un circulo que se repite y cada vuelta lo
deja mejor.**

```
            ┌─────────────────────────────────────────────────┐
            │                                                 │
            v                                                 │
   ┌──────────────────┐                                       │
   │ 1. El sistema    │                                       │
   │    trabaja y     │                                       │
   │    SE EQUIVOCA   │                                       │
   └────────┬─────────┘                                       │
            v                                                 │
   ┌──────────────────┐                          ┌────────────┴──────────┐
   │ 2. El operario   │                          │ 7. El modelo NUEVO    │
   │    aprieta "e"   │                          │    reemplaza al viejo │
   └────────┬─────────┘                          └────────────▲──────────┘
            v                                                 │
   ┌──────────────────┐                          ┌────────────┴──────────┐
   │ 3. Se guarda la  │                          │ 6. train              │
   │    foto LIMPIA + │                          │    (la maquina        │
   │    pre-etiquetado│                          │     estudia)          │
   │    + informe     │                          └────────────▲──────────┘
   └────────┬─────────┘                                       │
            v                                                 │
   ┌──────────────────┐                          ┌────────────┴──────────┐
   │ 4. Fin de semana:│                          │ 5. prepare_dataset    │
   │    corregir las  │─────────────────────────>│    fusionar / dividir │
   │    cajas en      │                          └───────────────────────┘
   │    Roboflow      │
   └──────────────────┘
```

### Paso por paso

**1. El sistema trabaja y se equivoca.**
Es esperable. Ninguna IA acierta el 100%. Los errores son la materia prima del ciclo.

**2. El operario aprieta `e`.**
No hace falta explicar nada ni escribir nada. Se ve el error, se aprieta la tecla, se sigue
trabajando. Toma medio segundo.

**3. Se guarda la foto LIMPIA + el pre-etiquetado + el informe.**
Esto es clave y es una decision de diseno deliberada: la imagen que se guarda para
reentrenar es **la que salio de la camara, sin un solo pixel dibujado encima**. Ni cajas,
ni letras, ni colores.

> **Por que:** si se guardara la imagen con los recuadros verdes dibujados, al reentrenar la
> red aprenderia a **detectar rectangulos verdes** en vez de desgaste real. Aprenderia lo
> que dibujamos nosotros, no lo que hay en el metal.

Junto a la imagen se guarda un **pre-etiquetado**: las cajas que la IA creyo ver, ya
escritas en el formato que entiende Roboflow. Eso ahorra muchisimo trabajo: en Roboflow no
hay que dibujar las cajas de cero, solo **corregir las que la IA erro**. Ademas se guarda
un informe en castellano llano con lo que penso la maquina (que detecto, con cuanta
confianza, que medidas, que descarto y por que) y, aparte, una copia con las cajas
dibujadas para revisar comodo.

**4. Fin de semana: se corrigen las cajas en Roboflow.**
Con calma, en la computadora de la oficina. Se sube la carpeta de feedback a Roboflow, se
revisan las cajas pre-dibujadas, se corrigen las que estan mal, se ajusta la clase
(ok / desgastado / fisura / astillado / oxido) y se exporta.

**5. `prepare_dataset`: fusionar y dividir.**

```powershell
python -m afilado.cli.prepare_dataset fusionar --feedback data\feedback --salida data\dataset\crudo
python -m afilado.cli.prepare_dataset dividir --origen data\dataset\crudo --salida data\dataset --val 0.2
```

`fusionar` junta lo que se recolecto en el taller. `dividir` separa las imagenes en dos
montones: uno para que la maquina estudie (train) y otro para tomarle examen (val), y
genera el `data.yaml` que el entrenamiento necesita. Avisa si hay imagenes sin etiqueta o
etiquetas sueltas sin imagen.

**6. `train`: la maquina estudia.**

```powershell
python -m afilado.cli.train --datos data\dataset\data.yaml --modelo yolo11n-seg.pt --epocas 150
```

Puede tardar horas. Al terminar imprime las notas del examen (mAP50 y mAP50-95: cuanto mas
cerca de 1, mejor) y deja el modelo nuevo listo.

**7. El modelo nuevo reemplaza al viejo.**
El entrenamiento copia solo el mejor resultado a `models/afilado_best.pt`, que es
justamente el archivo que `run_live` levanta. La proxima vez que se abra el programa, ya
usa el modelo nuevo. **Y vuelve al paso 1**, ahora equivocandose menos.

### Se arranca SIN modelo, y esta bien

El dia 1 no hay ningun modelo entrenado —no existe `models/afilado_best.pt`— y **eso no
impide empezar**. El sistema detecta que no hay pesos y usa un **detector geometrico** de
respaldo: separa la pieza del fondo por contraste y la mide igual. En pantalla las piezas
salen en amarillo con la etiqueta `sin_clasificar`, porque todavia nadie le enseno a
distinguir estados.

Sirve para dos cosas desde el minuto cero:
1. **Medir** (que es lo que hace falta ya).
2. **Recolectar datos.** Se apoya la pieza, se aprieta `espacio` o `g`, y esa foto va a la
   pila que despues se etiqueta. **El dia 1 ya se esta construyendo el dataset.**

---

## 7. Cuantas imagenes hacen falta

| Concepto | Cantidad |
|---|---|
| Por clase (ok, desgastado, fisura, astillado, oxido) | **Cientos.** 200-300 como piso, 500+ para andar bien. |
| Total para un primer modelo usable | ~1.000 a 1.500 imagenes |
| Total para un modelo solido | ~3.000+ |

Reglas para que las fotos sirvan:

- **Todos los estados representados.** Si hay 900 fotos de "ok" y 30 de "fisura", el modelo
  va a decir "ok" a todo, porque acierta casi siempre haciendo eso. Hay que buscar
  activamente las piezas feas.
- **Misma luz y mismo angulo que la version final.** Las fotos de entrenamiento tienen que
  parecerse a lo que la camara va a ver todos los dias. Si se cambia la iluminacion o la
  altura de la camara despues de entrenar, hay que volver a juntar fotos.
- **Variedad dentro de cada estado.** Distintas piezas, distintas rotaciones, distintas
  posiciones en la mesa. Distintos grados de desgaste, no siempre el mismo diente.
- **Fotos nitidas.** Las borrosas se descartan solas en `prepare_dataset extraer`.

---

## 8. Estructura de carpetas

```
Analisis-afilado/
├── README.md                 <- este archivo
├── requirements.txt          <- lista de librerias que instala setup.ps1
├── configs/
│   └── config.yaml           <- TODOS los ajustes. Es lo unico que se toca a mano.
├── models/
│   └── afilado_best.pt       <- el modelo entrenado (aparece despues del primer train)
├── scripts/
│   ├── setup.ps1             <- instalador
│   └── make_aruco.py         <- genera el marcador para imprimir
├── src/afilado/              <- el programa (no tocar)
│   ├── config.py             <- lee configs/config.yaml
│   ├── camara.py             <- habla con la webcam
│   ├── calibracion.py        <- encuentra el ArUco y calcula la escala mm/pixel
│   ├── medicion.py           <- mide largo, ancho, area, angulo
│   ├── filtros.py            <- descarta polvo, aserrin y cosas fuera de la zona util
│   ├── detector.py           <- el modelo YOLO (y el respaldo geometrico)
│   ├── overlay.py            <- dibuja lo que se ve en pantalla
│   ├── almacen.py            <- guarda las capturas del ciclo de aprendizaje
│   ├── pipeline.py           <- ordena todos los pasos de cada imagen
│   └── cli/
│       ├── run_live.py       <- el programa del taller
│       ├── train.py          <- entrenamiento
│       └── prepare_dataset.py<- preparacion del dataset
├── tests/                    <- pruebas automaticas del codigo
└── data/                     <- todo lo que se genera trabajando
    ├── feedback/             <- lo que se guarda al apretar e / g / espacio
    │   └── 2026-07-16/       <- una carpeta por dia
    │       ├── imagenes/     <- fotos LIMPIAS (esto es lo que se sube a Roboflow)
    │       ├── etiquetas/    <- pre-etiquetado en formato YOLO
    │       ├── reportes/     <- .json y .txt con lo que penso la IA
    │       └── revision/     <- copia con las cajas dibujadas, para mirar
    └── dataset/              <- el dataset armado para entrenar
        ├── crudo/            <- imagenes + etiquetas juntas antes de dividir
        ├── images/{train,val}
        ├── labels/{train,val}
        └── data.yaml         <- le dice al entrenamiento donde esta todo
```

### Detalle que conviene saber

En `feedback/<fecha>/` la foto y su etiqueta **no viven en la misma carpeta**. Es a
proposito: YOLO y Roboflow interpretan que un archivo `.txt` con el mismo nombre que la
imagen **es el archivo de etiquetas**. Por eso el informe legible en castellano va aparte,
en `reportes/`, y lo que acompana a la imagen es el pre-etiquetado en formato YOLO.

---

## 9. Limitaciones honestas

Conviene tenerlas claras antes de prometer nada.

| Limitacion | Explicacion |
|---|---|
| **Paralaje / espesor** | **Es la limitacion principal.** Una camara sola no sabe de profundidad. Si el marcador y la cara que se mide no estan a la misma altura, la medida sale mal. No hay forma de corregirlo por software: hay que subir el marcador (ver seccion 4.4). |
| **Reflejos del metal** | El metal pulido espeja. Sin luz difusa y polarizador, el brillo tapa el filo y la IA no ve nada, o confunde un reflejo con una fisura. La iluminacion no es un lujo, es parte del instrumento. |
| **Una webcam 2D no mide profundidad** | No puede medir el desgaste **hacia adentro** (cuanto material se perdio en el eje vertical), ni la profundidad de una fisura, ni el radio del filo. Ve una silueta desde arriba: mide largo, ancho, area y angulo, nada mas. |
| **Depende de que nada se mueva** | Si se corre la camara, se cambia la altura o se cambia la lampara, hay que recalibrar y probablemente reentrenar. |
| **El modelo solo sabe lo que se le enseno** | Si aparece un tipo de herramienta que nunca vio, va a decir cualquier cosa. La respuesta es el ciclo: apretar `e` y sumarla al dataset. |

### Camino de upgrade

Cuando el metodo 2D toque su techo (tipicamente cuando haga falta medir profundidad o
eliminar el paralaje de raiz), el informe plantea este camino:

| Etapa | Equipo | Que resuelve |
|---|---|---|
| Hoy | Webcam 2D + ArUco | Medidas en el plano, clasificacion de estado, recoleccion de datos. |
| Upgrade de camara | **OAK-D** o **Intel RealSense** (camara con profundidad) | Mata el paralaje: la camara mide la distancia real a cada punto, asi que el espesor de la pieza deja de importar. Habilita medir desgaste en profundidad. |
| Upgrade de computo en planta | **NVIDIA Jetson Orin Nano** | Equipo chico, robusto y de bajo consumo para dejar fijo en la maquina, sin depender de una PC de escritorio. |

---

## 10. Hardware recomendado (para la PC de entrenamiento)

Esta es la maquina donde se entrena el modelo, no necesariamente la del taller.

| Componente | Recomendado | Por que |
|---|---|---|
| **Placa de video** | **RTX 3060 12 GB** o **RTX 4060 Ti 16 GB** | Lo que manda es la **memoria de video**, no la velocidad. 12 GB es el piso para entrenar comodo. Una placa de 8 GB obliga a bajar el batch y complica todo. |
| **Procesador** | **Intel i5-13400** o **AMD Ryzen 5 7600X** | Alimenta a la placa con imagenes. No hace falta gastar mas arriba. |
| **Memoria RAM** | **32 GB** | 16 GB se queda corto preparando datasets grandes. |
| **Disco** | **NVMe 1 TB** | Los datasets de imagenes son miles de archivos chicos; un disco mecanico frena el entrenamiento. |
| **Fuente** | **650-750 W, certificacion 80+ Gold** | La placa pega picos de consumo. Una fuente barata es el mejor camino a un equipo quemado. |
| **Gabinete** | Con **filtros antipolvo** y **presion positiva** (mas ventiladores entrando que saliendo) | **Estamos en una metalurgica.** El polvo y la viruta metalica conducen electricidad. La presion positiva hace que el aire salga por las rendijas en vez de chupar suciedad. Los filtros se limpian, la placa quemada se compra de nuevo. |

---

## 11. Preguntas frecuentes del taller

**Dice "SIN ESCALA" todo el tiempo.**
No ve el marcador. Revisar que este dentro del cuadro, bien iluminado, sin brillos, sin
arrugas y no tapado por la pieza o la mano.

**Las medidas dan mas grandes de lo que miden con el calibre.**
Casi seguro es paralaje: el marcador esta mas abajo que la cara que se mide (seccion 4.4).
Lo otro que hay que revisar es que el `lado_mm` de `configs/config.yaml` coincida con lo
que mide el marcador impreso **con calibre en la mano**.

**Todo sale en amarillo y dice `sin_clasificar`.**
No hay modelo entrenado todavia, esta usando el detector geometrico. Es normal al empezar.
Mide igual. Hay que juntar fotos y entrenar.

**Marca defectos donde no hay nada.**
Suele ser reflejo o aceite. Revisar iluminacion y polarizador. Se puede subir el umbral
(`--conf 0.9`). Y sobre todo: apretar `e` cada vez que pasa, para que aprenda.

**Se movio la camara.**
Apretar `r` para reiniciar la calibracion. Si el cambio fue grande (altura o angulo),
verificar los 90 grados y controlar unas piezas contra el calibre antes de confiar.

---

*Consultas: Valentin y Leandro, taller.*
