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

## 2. Compatibilidad

Hace falta **Python 3.10 o superior**. Probado sobre **Python 3.14.2** (esta PC), con OpenCV
4.13, numpy 2.5 y torch 2.13: funciona.

Para verificar que version esta activa, abrir PowerShell y escribir:

```powershell
python --version
```

Si no hay Python instalado, bajarlo de https://www.python.org/downloads/ y, durante la
instalacion, tildar la casilla **"Add Python to PATH"**.

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

Entre otras cosas registra la carpeta `src\` en el entorno, que es lo que permite escribir
`python -m afilado.cli.run_live` sin mas vueltas. **Si se salta este script** y se instalan
las dependencias a mano, ese comando falla con `ModuleNotFoundError: No module named
'afilado'`; la salida rapida es anteponer, en cada consola:

```powershell
$env:PYTHONPATH = "src"
```

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

### 3.3 La PC del taller: un solo .exe, sin instalar nada

La forma mas simple de llevar el sistema al taller es un ejecutable: **esa maquina no
instala nada**, ni siquiera Python. Se le copia un archivo y se hace doble clic.

En la maquina de desarrollo, desde un entorno liviano (ver 3.4):

```powershell
python scripts/construir_exe.py
```

Deja `dist\AnalisisAfilado.exe`, de unos **84 MB**. Adentro viajan el interprete de Python,
OpenCV, onnxruntime y el config; **no** viaja el modelo.

En el taller: copiar el .exe a una carpeta **con permiso de escritura** (el Escritorio
sirve; `Archivos de programa` **no**, porque ahi no va a poder guardar las fotos) y crear
el acceso directo:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\crear_acceso.ps1
```

Deja un **"Analisis de afilado"** en el Escritorio, con el icono del programa, apuntando al
.exe y con la carpeta de trabajo fijada a la del ejecutable. Doble clic y arranca en ~2
segundos. (Al .exe tambien se le puede dar doble clic directamente; el acceso es comodidad.)

**El .exe se arma SIN consola** (`--windowed`). Una ventana negra al lado de la ventana del
programa no le dice nada a nadie, y si alguien la cierra se lleva el programa puesto. Pero
eso tiene una consecuencia: sin consola, `sys.stdout` y `sys.stderr` son `None`, y todo lo
que el programa informa se perderia. Por eso **el registro va a un archivo**:

```
registro.txt                   <- lo que paso en esta sesion y en la anterior
```

Ese archivo es lo unico que queda cuando algo falla en el taller: **todos** los problemas de
camara que se arreglaron en este proyecto se diagnosticaron leyendolo. Rota solo, a 1 MB por
vuelta, asi que no crece sin limite. Si el programa no arranca, en vez de un parpadeo negro
aparece un cartel de Windows con la causa --sacada del propio registro-- y la ruta del
archivo. Para depurar con consola de verdad: `python scripts/construir_exe.py --con-consola`.

Lo que el programa escribe y lee queda **al lado del .exe**, no adentro:

```
AnalisisAfilado.exe
registro.txt                   <- el log de la sesion (mirá esto si algo falla)
data\coleccion\aprobada\      <- las fotos que archiva el operario
data\coleccion\rechazada\
data\referencia\microscopio\   <- las fotos de modelo contra las que se compara
models\afilado_best.onnx       <- el modelo, cuando llegue
models\afilado_best.onnx.json
configs\config.yaml            <- opcional: si existe, pisa al que viaja adentro
```

> Esto ultimo no es un detalle. En modo un-solo-archivo, el contenido del .exe se
> descomprime en una carpeta temporal que Windows **borra al cerrar el programa**. Si las
> fotos se guardaran ahi, cada sesion de recoleccion se perderia sin un solo mensaje de
> error. Por eso `config.raiz_repo()` apunta a la carpeta del ejecutable cuando detecta que
> esta empaquetado, y hay tests que lo verifican.

Sin modelo en `models\`, el .exe arranca igual en **modo recoleccion**: no da veredicto,
pero deja archivar fotos con `a` y `r`. Es exactamente lo que hace falta al principio.

#### Que fotos NO deja archivar

Una foto mal etiquetada no se nota: entra al dataset, se entrena con ella, y el modelo sale
raro sin que nadie pueda decir por que. Asi que `a` y `r` no guardan cualquier cosa. Se
frena, con el motivo en pantalla, si:

- **La camara esta congelada.** Con el microscopio desenchufado, el programa cae en la
  camara virtual que haya instalada (DroidCam y similares) y muestra su cartel fijo. Esto ya
  paso: ese cartel quedo archivado como sierra *aprobada*. Y **no lo atrapa un control de
  foco**: medido sobre la foto real, el cartel da nitidez **642** --la mas alta de todo el
  dataset, mas que cualquier diente-- porque es texto blanco nitido sobre negro. Por eso se
  le pregunta a la camara (`Camara.congelada`), no a la imagen.
- **La imagen esta movida.** El mismo diente enfocado mide 101 y con un desenfoque de 5x5
  cae a 3.7. El umbral sale de `interfaz.nitidez_minima` en el config.

### 3.4 La PC del taller: instalacion liviana (alternativa al .exe)

Si preferis tener el codigo en el taller en vez de un ejecutable --por ejemplo para poder
tocar el config comodo o actualizar con `git pull`--, esta la instalacion liviana.

Instalar todo (`requirements.txt`) deja un entorno de **1,2 GB**, de los cuales `torch` solo
son 533 MB. Eso hace falta para **entrenar**, no para **usar** el sistema. En la maquina del
taller se instala con:

```powershell
.\scripts\setup.ps1 -Taller
```

Eso instala `requirements-taller.txt` --opencv, numpy, pyyaml y onnxruntime-- y saltea todo
lo que solo tiene sentido en la maquina de entrenamiento. Medido instalando de cero:

| | Completa | `-Taller` |
|---|---|---|
| Entorno en disco | **1,2 GB** | **281 MB** |
| Paquetes | ~35 (torch, ultralytics, matplotlib, sympy...) | **7** |
| Puede entrenar | si | no |
| Puede dar veredictos | si | **si** |

Esa maquina no entrena ni exporta: recibe el modelo ya convertido. Si `onnxruntime` no tiene
wheel para su version de Python, el instalador reintenta sin el y avisa: el sistema sigue
funcionando con el `cv2.dnn` que ya trae OpenCV, solo que un poco mas lento.

El modelo entrenado se exporta una vez a formato ONNX y desde ahi corre con OpenCV, que este
proyecto ya necesita si o si por el modulo `aruco`. En la PC de entrenamiento:

```powershell
pip install onnx
python scripts/exportar_onnx.py
```

Eso deja `models/afilado_best.onnx` y su `models/afilado_best.onnx.json`. Se copian los dos
a la PC del taller (un pendrive alcanza: pesan unos 6 MB entre los dos) y ahi, en
`configs/config.yaml`:

```yaml
detector:
  pesos: models/afilado_best.onnx
```

Medido en un i5-12400 sin placa de video, corriendo el mismo bucle en vivo:

| | Con `.pt` (torch) | Con `.onnx` |
|---|---|---|
| Entorno en disco | 1,2 GB | ~280 MB |
| RAM del proceso | 440 MB | **48 MB** (pico 82 MB) |
| Arranque + carga del modelo | ~4 s | ~0,3 s |
| Inferencia a 224 px | 12 ms | 5,8 ms |
| Librerias necesarias | torch, ultralytics, opencv, numpy | **opencv, numpy** |

**El veredicto es el mismo.** No es una aproximacion: `scripts/exportar_onnx.py` corre el
`.pt` y el `.onnx` sobre las mismas fotos del dataset antes de dar el export por bueno, y si
algun veredicto difiere **borra el archivo y termina con error**. Verificado sobre 34 fotos
del taller: 34 de 34 coinciden, con una diferencia media de probabilidad de 0,0011.

> Opcional: instalar `onnxruntime` en la PC del taller la hace unas tres veces mas rapida
> que el `cv2.dnn` que ya trae OpenCV (1,8 ms contra 5,8 ms). Con
> `clasificar_cada_n_frames: 5` eso es menos de 1 ms por frame, asi que solo vale la pena en
> maquinas muy justas. El resultado es identico con cualquiera de los dos.

> El `.onnx.json` **no es opcional**. Ahi viajan las clases y el tamano de entrada reales del
> modelo. Sin ese archivo el programa se cae a lo que diga `configs/config.yaml`, y una lista
> de clases corrida convierte "aprobada" en "rechazada" sin avisar.

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

**Que camara usa.** Por defecto `camara.fuente` viene en **`auto`**: el programa recorre
los indices (0 a 3) y los backends de OpenCV hasta encontrar una que **entregue imagen**, y
avisa en el log cual eligio. No alcanza con que la camara "abra": hay drivers que abren y
despues no entregan un solo frame, y ahi el programa arrancaba para morirse treinta frames
mas tarde diciendo que revisaran el cable.

Si dice que no puede abrir la camara, lo primero:

```powershell
python -m afilado.cli.run_live --listar-camaras
```

Eso lista lo que responde y con que indice. Los avisos en rojo de OpenCV durante la busqueda
son normales: es el precio de preguntar por indices que no existen.

> El backend tambien se prueba solo. Antes se usaba unicamente DSHOW en Windows, y hay
> camaras --y cualquier sesion de Escritorio Remoto con la camara redirigida-- donde DSHOW
> no abre nada y MSMF anda perfecto. Esa era la causa mas comun de "no se pudo abrir la
> camara" con una camara sana enchufada.

Para fijar una camara concreta (por ejemplo si hay dos y siempre tiene que ser la USB),
poner el indice en vez de `auto`. Con un indice fijo el programa **no** cambia de camara por
su cuenta --podria ser otra-- pero igual prueba los otros backends, y si falla te dice en el
error que camaras si encontro y con que indice.

**Resolucion de captura** (`camara.ancho` / `camara.alto` en el config): viene en
**640x480**, que es lo que conviene para el modo clasificacion en una PC modesta. Es el
costo que se paga en TODOS los frames, se clasifique o no. Medido en un i5-12400, sumando
decodificar el MJPG + preparar la entrada de la red + dibujar el overlay:

| Resolucion | Costo fijo por frame | A 30 fps |
|---|---|---|
| 1280x720 | 6,45 ms | 19,4% de un nucleo |
| 800x600 | 4,04 ms | 12,1% |
| **640x480** | **3,07 ms** | **9,2%** |

Hay que subirla de nuevo a **1280x720** en dos casos:

- Cuando se pase a `tarea: detect` o `segment` con medicion por ArUco. Ahi los pixeles sobre
  el marcador **son** la precision en milimetros, y bajarlos es tirar exactitud a la basura.
- Si el operario necesita ver mejor la sierra en pantalla para decidir.

Tener en cuenta que las fotos que archiva el operario con `a`/`r` quedan a esta resolucion, y
son el dataset de todos los entrenamientos futuros. Para clasificar sobra (la red mira 224 px
igual), pero es una decision que despues no se puede deshacer.

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
| `--fuente <n o ruta>` | Elegir otra camara (`--fuente 1`), buscarla sola (`--fuente auto`) o reproducir un video grabado. |
| `--listar-camaras` | Mostrar que camaras responden en esta maquina y salir. Es lo primero que hay que correr si dice que no puede abrir la camara. |
| `--pesos <ruta.pt>` | Probar otro modelo entrenado sin tocar la configuracion. |
| `--conf 0.9` | Subir o bajar el umbral de confianza para esta corrida. |
| `--sin-roi` | Analizar el frame completo, sin recortar la zona util. |
| `--grabar salida.mp4` | Grabar la sesion en video. |

El programa arranca en el modo que diga `detector.tarea` en el config. Hoy esta en
**`classify`** (veredicto aprobada/rechazada). Los otros modos (`detect`, `segment`) sirven para
la etapa de medicion con ArUco, mas adelante.

### 5.1 Modo simple (el actual): dos recuadros y un cartel

Es la pantalla del taller, pensada para que la lea cualquiera de un vistazo y desde lejos.
Se activa con `interfaz.modo: simple`, que es como viene.

```
        SIERRA                      MODELO APROBADO
  +------------------+          +------------------+
  |  lo que ve la    |          |  la foto de una  |
  |  camara, en vivo |          |  pieza buena     |
  +------------------+          +------------------+

  RESULTADO
  +--------------------------------------------------+
  |                  A P R O B A D O                 |   verde
  |            coincidencia 87%  (minimo 70%)        |
  +--------------------------------------------------+

  [REVISAR]                                   [SALIR]
```

El cartel de abajo dice **APROBADO** en verde o **DESAPROBADO** en rojo. Nada mas: ni
porcentajes por clase, ni nombres de modelo, ni jerga.

#### Evaluacion en vivo

Por defecto (`interfaz.evaluar_en_vivo: true`) **el veredicto se actualiza solo** mientras
se mueve la pieza, sin apretar nada. REVISAR pasa a congelar la imagen con su veredicto,
para mirarla con calma o archivarla.

Juzgar cada frame tiene dos riesgos, y los dos estan atendidos:

**Opinar sobre una imagen que no sirve.** Antes de clasificar se mide el foco --la varianza
del laplaciano--. Si no llega a `interfaz.nitidez_minima`, el cartel dice **ENFOCANDO** en
ambar y **muestra la nitidez medida**, asi se ajusta el umbral mirando el numero en vez de a
ciegas. Medido sobre una foto real del microscopio: enfocada 94,6; con un desenfoque minimo
7,1; movida, entre 11 y 25. El valor depende de la camara, asi que con otra hay que volver a
mirar el numero y ajustar.

**Parpadear entre aprobado y rechazado.** El cartel solo cambia cuando
`interfaz.frames_para_confirmar` evaluaciones seguidas dan lo mismo. Verificado con un
clasificador que alterna a proposito en cada llamada: con 1 el cartel cambia 29 veces en 30
frames; con 3, ninguna. Con una imagen quieta y un veredicto estable, confirma en 3
evaluaciones.

Si la pieza se mueve, el veredicto anterior **se borra** en vez de quedar en pantalla:
mostrar un resultado que ya no describe lo que se ve es peor que no mostrar nada.

Con `evaluar_en_vivo: false` el programa no opina hasta que se aprieta REVISAR. Conviene si
la camara mira una zona donde pasa gente o herramientas.

#### De donde sale ese porcentaje

Hay **tres** fuentes posibles, y la pantalla **siempre dice cual fue**, abajo del cartel. Se
usa la primera que este disponible:

| Orden | Quien opina | Que dice el detalle en pantalla |
|---|---|---|
| 1 | El **modelo entrenado**, si existe `models/afilado_best.*` | `87% - segun el modelo entrenado` |
| 2 | La **memoria**: todas las fotos que el taller archivo | `se parece a rechazada (82%) - entre las 24 fotos guardadas` |
| 3 | El **parecido** contra la foto de modelo que este elegida | `79% - comparando con la foto de modelo` |
| 4 | Nadie todavia | `SIN MODELO` + lo que la memoria lleva juntado |

Que el origen este SIEMPRE escrito no es un adorno. En este proyecto aparecio un
**DESAPROBADO 36%** con las dos imagenes practicamente iguales: el numero venia de un modelo
entrenado con fotos de otra cosa, y mirando la pantalla no habia forma de saberlo. Un
porcentaje sin su origen es indistinguible de uno inventado.

**Que mide cada uno:**

- **Modelo entrenado:** cuanta probabilidad le da a que la pieza sea de las aprobadas. Es lo
  mas confiable, cuando existe. Umbral: `interfaz.umbral_aprobacion` (0,70 de fabrica).
- **Memoria:** cuanto se parece a las fotos ya archivadas, y en que concepto estan esas. Ver
  mas abajo.
- **Parecido contra la foto de modelo:** compara las dos imagenes que estan a la vista.
  Encendido por pedido del taller (`interfaz.veredicto_por_parecido`), con su propio umbral
  `interfaz.umbral_parecido`. **Ojo con lo que dice:** medido sobre las fotos del taller,
  dos tomas de la MISMA pieza dan 78% y dos de piezas DISTINTAS dan 79%. O sea que este
  APROBADO significa "se parece bastante a la foto de al lado", **no** "el filo esta sano".

El umbral de 0,70 es un punto de partida: si pasan piezas que no deberian, subirlo; si frena
piezas buenas, bajarlo. Conviene tocarlo recien cuando haya un modelo entrenado con fotos del
microscopio y se vea como se porta.

| Tecla | Boton | Que hace |
|---|---|---|
| `espacio` | **REVISAR** | Congela lo que se ve y lo juzga. |
| `espacio` | **OTRA PIEZA** | Vuelve al vivo. |
| `a` | **GUARDAR BUENA** | Archiva la foto en el primer concepto sano. |
| `r` | **GUARDAR MALA** | Archiva la foto en el primer concepto de rechazo. |
| `c` | **CONCEPTOS** | Abre el panel: elegir a que carpeta va la foto, crear conceptos nuevos, ver y borrar fotos. |
| `f` | — | **Agrega** lo que se ve como una foto de modelo nueva. |
| `m` | **OTRO MODELO** | Pasa a la siguiente foto de modelo. |
| `q` | **SALIR** | Cierra. |

#### La memoria: el programa aprende de cada foto que se archiva

**No hay cupo que llenar.** Cada foto que se guarda en un concepto se suma AL INSTANTE a lo
que el programa compara, y el veredicto de la foto siguiente ya la tiene en cuenta. No hay
que juntar cincuenta y entrenar: eso sigue existiendo, y es mejor, pero es para despues.

Lo que la memoria hace es preguntarle a las fotos guardadas: *cuales se parecen mas a esta, y
en que concepto estan archivadas esas*. Dos cosas importantes salen de que sea comparacion y
no una red: se puede explicar foto por foto, y no inventa nada que no este guardado.

**La memoria se mide a si misma.** Esconde cada foto guardada y le pregunta a las demas en
que concepto va. De ahi sale el `acierta 21 de 24` que aparece en pantalla y en el panel. Es
el unico numero honesto sobre cuanta precision tiene esto: sale de las fotos de ESTE taller y
sube (o baja) con cada foto que entra.

**Y cuando no esta segura, lo dice.** En vez de un umbral inventado, mira que paso de verdad
en los casos igual de ajustados: si de las doce veces que estuvo asi de parejo acerto seis,
la confianza es 50% y el cartel dice **NO ESTOY SEGURO** con el motivo. Un APROBADO salido de
un empate es peor que no decir nada, porque el operario no tiene como saber que fue un empate.

Tampoco fuerza un concepto si la foto no se parece a nada de lo guardado: ahi dice
`esta foto no se parece a ninguna de las N guardadas`.

Ademas del porcentaje, cuando la diferencia esta **en un lugar** y no repartida por toda la
imagen, lo dice: *la mayor diferencia esta arriba a la derecha*.

**Para que empiece a opinar** hacen falta `memoria.minimo_por_concepto` fotos (3 de fabrica)
en al menos **dos** conceptos distintos. Con uno solo no hay decision posible: todo se parece
a lo unico que hay.

**Se corrige sola, archivando.** Si la memoria dice "aprobada" y era mala, se aprieta GUARDAR
MALA: esa foto entra al concepto correcto y la proxima vez pesa. Eso es todo el
entrenamiento que hay que hacer en el dia a dia.

Cuesta poco: medido con 300 fotos en la PC del taller, el veredicto sale en **6,9 ms** --el
mismo tiempo que con 19 fotos, porque primero descarta barato y solo compara en serio las
mejores candidatas de cada concepto-- y abrir el programa con todas esas fotos ya analizadas
tarda 51 ms. La cache vive en `data/memoria/` y se puede borrar sin perder nada: se rehace
sola leyendo las fotos.

#### El panel de CONCEPTOS (tecla `c`)

Un **concepto** es una carpeta de fotos que significan lo mismo: `aprobada`, `rechazada`,
`diente_astillado`, `filo_quemado`. Los decide el taller, no el config.

```
  CONCEPTOS
  memoria: 24 fotos (aprobada 14  rechazada 10)  -  acierta 21 de 24

  [ ] 1  foto de modelo      19 fotos  -  es la foto con la que se compara, no un concepto
  (o) 2  aprobada            14 fotos  -  la memoria acierta 12 de 14
  (o) 3  rechazada           10 fotos  -  la memoria acierta 9 de 10
  (o) 4  filo_quemado        sin fotos todavia

  [GUARDAR ACA] [NUEVO CONCEPTO] [VER FOTOS] [BORRAR CONCEPTO]        [VOLVER]
```

- **A que carpeta va esta foto**: se elige la fila (con el mouse o con las teclas `1`..`9`) y
  se aprieta **GUARDAR ACA** o `ENTER`. A la derecha se ve la foto que se esta por archivar.
- **Crear un concepto**: **NUEVO CONCEPTO** o `n`, se escribe el nombre y `ENTER`. Queda
  creada la carpeta y se puede usar en el acto, sin cerrar el programa.
- **Ver y tirar fotos**: **VER FOTOS** o `v` muestra las del concepto elegido; `b` manda la
  elegida a la papelera. Sirve para sacar la foto movida o la que quedo en la carpeta
  equivocada.
- **Borrar un concepto**: **BORRAR CONCEPTO** o `b`, con confirmacion.

> **Borrar no borra.** Todo lo que se saca --una foto o un concepto entero-- va a
> `data/coleccion/_papelera/`. Un clic de mas no puede costar las fotos que el taller tardo
> semanas en juntar. El que quiera el espacio, vacia esa carpeta con el Explorador viendo lo
> que tira.

**Los conceptos SON las carpetas de `data/coleccion/`.** Una carpeta copiada ahi con el
Explorador aparece sola la proxima vez que se abre el programa; una borrada deja de estar. El
`conceptos.json` que esta al lado no manda: solo recuerda cual cuenta como defecto (los que
pintan el cartel de rojo) y en que orden se muestran.

La galeria de fotos de modelo aparece en la misma lista, aunque no sea un concepto: es otra
carpeta a la que van fotos, y la pregunta del operario es "a donde mando esta foto".

#### Las fotos de modelo

El panel derecho no es una sola foto: es una **carpeta** con una por cada modelo de sierra,
porque cada uno tiene su geometria de diente y comparar un LU3F-0300 contra un R01 no dice
nada. Arriba del panel se ve cual esta cargada y cuantas hay: `MODELO: LU3F-0300  (1 de 3)`.

- **Sacar una nueva**: poner la pieza buena bajo el microscopio y apretar `f`. **Agrega** una
  foto, no pisa las que ya estaban. Se guarda con la fecha y hora como nombre.
- **Ponerle nombre**: entrar a la carpeta y renombrar el archivo con el modelo de sierra.
  Ese nombre es el que aparece en pantalla.
- **Elegir entre varias**: tecla `m` o boton **OTRO MODELO**, que van rotando. El boton solo
  aparece si hay mas de una.
- **Ordenarlas**: se ciclan por orden alfabetico, asi que numerarlas (`01-LU3F.jpg`,
  `02-R01.jpg`) alcanza para fijar en que orden aparecen.
- **Copiarlas desde otra PC**: se pueden dejar directamente en la carpeta, incluso con el
  programa abierto; al apretar `m` se relee la carpeta y aparecen.

La carpeta la define `interfaz.referencia_aprobada` en el config. El programa recuerda cual
estaba elegida la ultima vez, asi que quien trabaja todo el dia con el mismo modelo no tiene
que buscarlo en cada arranque.

Los otros dos modos siguen disponibles en `interfaz.modo`: **`revision`** es la misma
pantalla partida pero con la banda de veredicto arriba y el detalle por clase, y
**`clasico`** es la pantalla anterior al microscopio, sin partir, juzgando todos los frames.

### 5.1.ter Modo revision: la variante con detalle por clase

Es la pantalla pensada para el **microscopio USB**. Se activa con `interfaz.modo: revision`
en el config, que es como viene.

La pantalla se parte en dos: a la izquierda lo que ve la camara **en vivo**, a la derecha la
foto de una pieza **aprobada** para comparar a ojo. Abajo, un boton **REVISAR**.

**El programa no juzga solo.** El operario acomoda el diente, lo enfoca, y recien cuando
esta conforme aprieta REVISAR (o la barra espaciadora, o hace clic). Ahi se congela ESE
frame y se analiza. Queda una foto quieta con su veredicto, que se puede mirar con calma.

> Esto es a proposito y es la diferencia con el modo clasico. Con el microscopio la imagen
> solo sirve cuando la pieza esta bien puesta: juzgar todos los frames seria juzgar sobre
> todo imagenes movidas y a medio encuadrar. Un veredicto que parpadea es un veredicto que
> el operario aprende a ignorar, y ahi se perdio la herramienta de control.

| Tecla | Boton | Que hace |
|---|---|---|
| `espacio` | **REVISAR** | Congela lo que se ve y lo analiza. |
| `espacio` | **OTRA PIEZA** | Descongela y vuelve al vivo, para la siguiente. |
| `a` | **GUARDAR APROBADA** | Archiva la foto congelada como aprobada. |
| `r` | **GUARDAR RECHAZADA** | Archiva la foto congelada como rechazada. |
| `f` | — | Toma lo que se ve **como la foto de referencia** y la deja fija a la derecha. |
| `q` o `ESC` | **SALIR** | Cierra el programa. |

Todo tiene tecla **y** boton: si en alguna maquina el mouse no responde bien dentro de la
ventana de OpenCV, el teclado sigue funcionando.

**El dia 1 no hay foto de referencia** y el panel derecho lo dice, con la instruccion:
poner una pieza aprobada bajo el microscopio y apretar `f`. Queda guardada en la ruta que
diga `interfaz.referencia_aprobada` y ya aparece siempre.

Sin modelo entrenado, la banda de arriba avisa que todavia no hay veredicto, pero `a` y `r`
archivan igual: es el modo recoleccion, y es lo que hay que hacer al principio.

### 5.1.bis Modo clasificacion: veredicto continuo sobre la sierra entera

La pantalla muestra una banda grande arriba con el veredicto: 🟩 **APROBADA** o 🟥 **RECHAZADA**,
con el porcentaje de confianza. Debajo, las probabilidades de cada clase (para ver si quedo al limite).

| Tecla | Que hace |
|---|---|
| `q` o `ESC` | Salir. |
| **`a`** | **Archivar la sierra que se ve como APROBADA** (buena). |
| **`r`** | **Archivar la sierra que se ve como RECHAZADA** (no afilable). |
| `p` | Pausa. |
| `h` | Mostrar u ocultar la ayuda. |

Las fotos se guardan **limpias** (sin la banda dibujada) en `data/coleccion/<clase>/`, que es
justo la carpeta que despues consume `construir_clasificacion.py`. Asi el operario junta el
dataset apuntando la camara y apretando una tecla, sin dibujar ni una caja.

**Sin modelo entrenado** la banda dice `SIN MODELO - MODO RECOLECCION`: no juzga, pero las teclas
`a`/`r` igual archivan las fotos por clase. Es el estado del dia 1: se junta, se entrena, y el
veredicto se enciende solo cuando aparece `models/afilado_best.pt`.

Ciclo para mejorar el modelo:

```bash
# tras varias sesiones juntando fotos, se reentrena con TODOS los conceptos que haya,
# con los nombres que el taller les puso desde el panel (_papelera queda afuera):
python scripts/construir_clasificacion.py --coleccion data/coleccion --forzar
python -m afilado.cli.train --datos data/clasificacion --epocas 100

# y si hay tandas sueltas en otro lado, se suman a mano (se pueden mezclar las dos formas):
python scripts/construir_clasificacion.py --coleccion data/coleccion --clase aprobada "ruta/tanda_buenas" --clase rechazada "ruta/tanda_malas" --forzar
```

> Conviene `--coleccion` antes que escribir los `--clase` a mano: el error que mas caro sale
> es entrenar sin una clase porque alguien se olvido de ponerla en la linea de comandos, y
> que el modelo nunca pueda decirla.

### 5.2 Modo deteccion/medicion (`tarea: detect` o `segment`)

Cuando se active la medicion con ArUco, el programa dibuja cajas/contornos y usa estas teclas:

| Tecla | Que hace |
|---|---|
| **`e`** | **Marcar ERROR de la IA.** Guarda la foto para reentrenar. |
| `g` | Guardar un ejemplo BUENO. |
| `espacio` | Capturar sin dar juicio. |
| `r` | Reiniciar la calibracion del ArUco. |
| `p` / `h` / `c` | Pausa / ayuda / ciclar ROI. |

Colores: 🟩 verde = clase sana · 🟥 rojo = defecto · 🟨 amarillo = `sin_clasificar` (sin modelo).
Avisos: **CAMARA INCLINADA** (no esta a 90°) y **SIN ESCALA** (no se ve el ArUco).

> El sistema **nunca inventa milimetros**. Si no ve el marcador, no da medidas en mm. Punto.

---

## 5.bis Estado de la primera prueba (datos reales del taller)

Ya se cargaron los dos primeros casos reales, que fijan los dos extremos del sistema:

| Caso | Sierra | Clase | Ficha |
|---|---|---|---|
| Aprobada (ideal) | Freud LU3F 0300 (Ø250, Z96, sana) | `aprobada` (0) | [data/referencia/LU3F-0300.md](data/referencia/LU3F-0300.md) |
| Rechazada (no afilable) | Sierra con diente faltante y oxido | `rechazada` (1) | [data/referencia/R01-no-afilable.md](data/referencia/R01-no-afilable.md) |

**Esquema de clases:** binario — `aprobada` / `rechazada`. La sierra completa se juzga sana o
descartada; todavia no se detalla *por que* se rechaza (eso es un refinamiento futuro).

**Como se entrena (clasificacion de imagenes):** juzgar la sierra entera como buena o
descartada es un problema de **clasificacion**, no de deteccion con cajas. Ventaja para el
taller: **no hay que dibujar ni una caja**, solo separar las fotos en carpetas por clase. El
flujo es:

```bash
# 1. Separas las fotos en dos carpetas (buenas / no afilables) y las convertis en dataset:
python scripts/construir_clasificacion.py \
    --clase aprobada  "ruta/a/fotos_sierras_buenas" \
    --clase rechazada "ruta/a/fotos_no_afilables" \
    --forzar
# 2. Entrenas (si --datos es una CARPETA, train.py entra en modo clasificacion solo):
python -m afilado.cli.train --datos data/clasificacion --epocas 100
```

El script `construir_clasificacion.py` aplica el mismo filtro de nitidez y deduplicacion que el
resto del sistema, y arma `data/clasificacion/{train,val}/<clase>/`.

**Lo que se validó con estas fotos reales:**
- El **filtro de nitidez** descartó solo las tomas movidas (de 51 fotos de la rechazada, ~35
  borrosas fuera; de 15 de la aprobada, 9 fuera). Funciona sobre fotos de taller, no de laboratorio.
- El **camino de clasificacion** (`construir_clasificacion` → `train`) corre de punta a punta y
  produce el `best.pt`. Quedaron 18 fotos de entrenamiento (6 aprobada + curadas de 12 rechazada).

**Lo que todavía NO funciona (y es esperado):**
- El modelo entrenado con ~22 fotos es **inservible**: da 100% en 4 fotos de validacion porque
  las memoriza, no porque haya aprendido. No es cautela: **faltan cientos de fotos por clase**.
  Por eso el `best.pt` de prueba se descartó y el sistema sigue en modo geometrico hasta juntar
  datos de verdad. Ver la sección 7.
- **El veredicto en vivo YA esta conectado** (`run_live` en modo `classify` muestra la banda
  aprobada/rechazada y archiva fotos por clase con `a`/`r`). Verificado con las sierras de
  referencia. Lo unico que falta para que sirva de verdad es el modelo entrenado con suficientes
  fotos: hasta entonces arranca en modo recoleccion. Ver la sección 5.1.

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
├── requirements.txt          <- librerias de la maquina de ENTRENAMIENTO
├── requirements-taller.txt   <- librerias de la PC del TALLER (sin torch, 281 MB)
├── configs/
│   └── config.yaml           <- TODOS los ajustes. Es lo unico que se toca a mano.
├── models/
│   ├── afilado_best.pt       <- el modelo entrenado (aparece despues del primer train)
│   ├── afilado_best.onnx     <- el mismo modelo para correr SIN torch (ver seccion 3.3)
│   └── afilado_best.onnx.json<- clases y tamano de entrada; viaja siempre con el .onnx
├── scripts/
│   ├── setup.ps1             <- instalador
│   ├── exportar_onnx.py      <- convierte el .pt en .onnx para la PC del taller
│   ├── construir_exe.py      <- arma AnalisisAfilado.exe para el taller
│   ├── crear_acceso.ps1     <- deja el acceso directo en el Escritorio
│   ├── lanzador_taller.py    <- punto de entrada de ese .exe
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

**Dice `ModuleNotFoundError: No module named 'afilado'`.**
El entorno no tiene registrada la carpeta `src\`. Se arregla corriendo `.\scripts\setup.ps1`
(es idempotente, se puede correr las veces que haga falta). Como parche momentaneo, en esa
misma consola: `$env:PYTHONPATH = "src"`.

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
