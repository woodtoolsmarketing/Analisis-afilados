# Analisis-afilado

Sistema de inspeccion visual de herramientas de afilado (dientes de widia, diamante PCD,
sierras circulares, fresas, insertos) con una camara web comun y una computadora.

---

## 1. Que hace el sistema

Una camara mirando la mesa desde arriba filma la pieza. **Hoy el programa juzga la foto
entera** --no recorta ni recuadra nada-- y contesta **APROBADO** o **DESAPROBADO**, con el
nombre del concepto que el taller archivo. Los conceptos son carpetas de fotos que el taller
crea desde el panel; hoy son cuatro: `Dientes_rotos`, `Sierra_Nueva_TP`, `Diente_sin_placa` y
`Sierra_Afilada`. O sea que la pantalla dice cosas como `DESAPROBADO - se parece a
Dientes_rotos (100%)`, y no un estado de una lista fija.

**La medida en milimetros y el detalle por tipo de defecto son la etapa siguiente, no la de
hoy.** El esquema de clases del config es **binario** --`aprobada` / `rechazada`
(`configs/config.yaml:434` y `config.yaml:439`)-- y el propio config deja escrito que refinar a
`diente_faltante, fisura, oxido, desgastado, astillado` es "cuando haya mas datos"
(`config.yaml:432`). El modo que corre es `detector.tarea: classify` (`config.yaml:377`); los
milimetros salen de `detect` o `segment`, que todavia no se usan (seccion 5.2). Tampoco hay
recuadro: `interfaz.recuadros_de_diente: false` (`config.yaml:259`) y `_analizar_dientes` toma
el cuadro entero (`src/afilado/cli/run_live.py:383`).

Cuando se encienda la medicion, para saber cuanto mide un milimetro en la pantalla el sistema
va a usar un cuadradito impreso en blanco y negro (un **marcador ArUco**) apoyado al lado de la
pieza: como el programa sabe cuanto mide ese cuadrado de verdad, puede deducir el tamano de
todo lo demas. **Sin el marcador a la vista no da milimetros**, ni ahora ni despues.

Lo importante es la segunda mitad: el sistema **aprende de los errores del operario que lo
corrige**. Cuando la maquina se equivoca --aprueba un diente roto, o lo manda a
`Dientes_rotos` cuando lo que hay es una mancha de aceite-- el operario aprieta una tecla y el
sistema guarda esa foto en el concepto que corresponde, junto con un informe de lo que penso. Esas fotos se corrigen despues con
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
registro.txt                   <- el log de la sesion (mira esto si algo falla)
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

- **La camara esta congelada.** Si el config pide un indice fijo y ese indice resulta ser
  una camara virtual (DroidCam y similares), se abre --lo pidieron-- pero no deja archivar.
  Esto ya paso: el cartel "Start DroidCam" quedo archivado como sierra *aprobada*. Y **no lo
  atrapa un control de foco**: medido sobre la foto real, el cartel da nitidez **642** --la
  mas alta de todo el dataset, mas que cualquier diente-- porque es texto blanco nitido
  sobre negro. Por eso se le pregunta a la camara (`Camara.congelada`), no a la imagen.

  > Buscando sola (`fuente: auto`, que es como viene), **una camara virtual ni siquiera se
  > abre**: se descarta y se sigue buscando, y si no hay ninguna de verdad el programa NO
  > arranca y dice que conecten el microscopio. Antes la abria igual "porque es mejor que
  > nada", y mostraba un cartel fijo que parecia una camara andando. No es mejor que nada.
- **La imagen esta movida.** El mismo diente enfocado mide 101 y con un desenfoque de 5x5
  cae a 3.7. El umbral sale de `interfaz.nitidez_minima` en el config.

**La tecla `f` es la excepcion, y a medias.** Agregar una foto de modelo **no** pasa por el
filtro de nitidez --el operario elige a ojo cual quiere de referencia, y una un poco movida
puede servirle-- pero **si** pasa por el de camara congelada. Con el microscopio
desenchufado, lo que se guardaria como "modelo" es el cartel de DroidCam, y entonces **todo
lo que venga despues se compara contra un cartel**.

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
| `--sin-roi` | Analizar el frame completo, sin recortar la zona util. **Solo sirve en `detect`/`segment`** |
| `--grabar salida.mp4` | Grabar la sesion en video. **Solo graba en `clasico` y en `detect`/`segment`** |

> **DOS OPCIONES QUE LA PANTALLA DEL TALLER NO MIRA.** Las dos se aceptan en la linea de
> comandos sin una queja y despues no hacen nada, asi que conviene saberlo antes de
> buscarle la vuelta:
>
> - **`--grabar` no graba en `simple` ni en `revision`** --los dos modos del microscopio--.
>   `args.grabar` se lee en dos lugares nada mas: el modo clasico
>   (`src/afilado/cli/run_live.py:1374`) y el bucle de deteccion (`run_live.py:2660`). La
>   pantalla del taller la arma `_ejecutar_revision` (`run_live.py:1594` a `run_live.py:2548`),
>   que no abre grabador en ninguna linea. No queda ningun archivo y no sale ningun aviso.
> - **`--sin-roi` no cambia nada en `simple` ni en `revision`.** Lo unico que hace es poner
>   `cfg.roi.habilitado` en false (`run_live.py:197`), y eso lo mira el Pipeline de
>   deteccion, que esas dos pantallas no usan. Ahi el cuadro entero se analiza **siempre**.

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

  +----------------------------------------------------------------------+
  | (X)  D E S A P R O B A D O                                           |   rojo
  |      se parece a Dientes_rotos (100%) - entre las 15 fotos guardadas |
  |      la memoria acierta 77% de las veces que decide                  |
  +----------------------------------------------------------------------+

  [REVISAR] [ANOTAR DIENTE]   [OTRO MODELO] [ORDENES] [CONCEPTOS] [SALIR]
```

De esa fila de botones, **ANOTAR DIENTE y OTRO MODELO estan ahi para mostrar donde caen**: el
primero necesita una orden abierta y el segundo mas de una foto de modelo, y hoy no se da
ninguna de las dos cosas. Medido en vivo con la coleccion del taller, lo que se dibuja es
`[REVISAR] [ORDENES] [CONCEPTOS] [SALIR]`.

**Arriba de esa tarjeta la pantalla no escribe nada.** Los unicos titulos de la pantalla son
los de las dos fotos de arriba: `SIERRA` y, a la derecha, `MODELO APROBADO` --que pasa a
`MODELO: <NOMBRE>` cuando la foto de modelo tiene nombre, y a `ASI SE VE CADA ESTADO` cuando
hay varios dientes a la vista-- (`revision.py:1359` a `revision.py:1367`). Al
veredicto se lo nombra por donde esta --**la tarjeta grande de abajo**-- y no por un rotulo,
porque rotulo no tiene. El dibujo de antes ponia ahi un `RESULTADO` y un renglon
`coincidencia 87% (minimo 70%)`: las dos cosas salen de `_cartel_resultado()`
(`src/afilado/revision.py:513`), que **no la llama nadie** --`grep -rn '_cartel_resultado'
--include=*.py .` devuelve una sola linea, la del `def`-- y por lo tanto nunca llegaron a una
pantalla. La pantalla del modo simple la dibuja `componer_simple`
(`src/afilado/revision.py:1146`).

**Ese dibujo es el de hoy, medido** componiendo `componer_simple` a 1280x672 con la coleccion
del taller (copiada al scratchpad; en `Desktop\Afilado` no se escribio nada) y la foto
`Dientes_rotos_20261116_084602_709.jpg` congelada. Lo que la tarjeta escribe son tres cosas, y
cada una sale de su propia funcion:

| Renglon | De donde sale | Que decia hoy, medido |
|---|---|---|
| la palabra grande | `estado_resultado()` (`revision.py:710`) | `DESAPROBADO`, en rojo, con el circulo de la cruz al lado |
| el renglon de abajo | el `detalle` de la misma funcion (`revision.py:1226`) | `se parece a Dientes_rotos (100%) - entre las 15 fotos guardadas` |
| el renglon mas chico | `detalle_extra()` (`revision.py:886`) | `la memoria acierta 77% de las veces que decide` |

**El `minimo 70%` existe, pero no va ahi.** Se dibuja **suelto, pegado al borde derecho** de la
tarjeta, sin parentesis y sin la palabra `coincidencia` (`revision.py:1450`), y **solo cuando
hay barra de coincidencia**, o sea solo cuando el que opina es el modelo entrenado. Con el
veredicto de la memoria --el de hoy-- no hay barra, no hay `minimo` escrito, y ese renglon lo
usa el de `detalle_extra`. El dia que haya modelo entrenado, el renglon de abajo pasa a decir
`NN% - segun el modelo entrenado` (ver la tabla de "De donde sale ese porcentaje"), aparece la
barra con la marca del umbral, aparece el `minimo 70%` contra el borde derecho y el renglon de
`detalle_extra` **deja de verse**: comparten el mismo lugar.

Y con la imagen **congelada**, los de la izquierda cambian. Asi queda hoy, **medido**
componiendo `componer_simple` a 1280x672 con los conceptos del taller:

```
  [OTRA PIEZA] [GUARDAR SIERRA_NUEVA_TP] [GUARDAR DIENTES_ROTOS] [CONCEPTOS] [SALIR]
```

**El boton ORDENES no esta en esa fila y no es un olvido**: con los nombres de concepto de
verdad ya no entra, y se cae antes de montarse sobre otro (ver el parrafo de mas abajo). A las
ordenes se llega con la tecla `o`, que queda escrita arriba a la izquierda.

**Esos dos botones de guardar dicen el nombre del concepto**, asi que en el taller de hoy no
dicen BUENA y MALA: dicen **GUARDAR SIERRA_NUEVA_TP** y **GUARDAR DIENTES_ROTOS** (medido
corriendo `_destinos_rapidos()` con la coleccion del taller). "GUARDAR BUENA" y "GUARDAR MALA"
son los nombres cortos que el programa le pone a los dos conceptos de siempre, `aprobada` y
`rechazada` (`src/afilado/cli/run_live.py:297`), que en esta coleccion no existen. El de la
izquierda es siempre el **primer concepto sano** de la lista y el de la derecha el **primero de
defecto**.

La palabra grande de la tarjeta dice **APROBADO** en verde o **DESAPROBADO** en rojo, con la
tilde o la cruz al lado. Nada mas: ni porcentajes por clase, ni nombres de modelo, ni jerga.
Los dos renglones chicos de abajo son los de la tabla de arriba, y estan para contestar "de
donde salio esto", que es la otra pregunta.

**ANOTAR DIENTE aparece solo en vivo y solo con una orden abierta.** Los tres de la
derecha --OTRO MODELO, ORDENES, CONCEPTOS-- piden lugar de derecha a izquierda, y **si no
entran no se dibujan**: antes que montarse sobre los botones de guardar, desaparecen. Pero
la tecla sigue andando y queda **escrita arriba a la izquierda**, debajo del resumen de la
orden:

```
  teclas:  o ordenes
```

Medido hoy en el taller: dice solo `o ordenes`. El `m modelo` se agrega a ese renglon
**unicamente si hay mas de una foto de modelo cargada**, y la galeria del taller esta
vacia. El `c conceptos` aparece ahi por lo mismo: solo cuando el boton CONCEPTOS tampoco
entro.

Esto no es un caso raro, es el del taller: medido con los nombres de concepto de verdad
(`GUARDAR SIERRA_NUEVA_TP` y `GUARDAR DIENTES_ROTOS`) y la pantalla congelada, el boton
ORDENES **no entra** --ocupa 140 px y quedan 89 libres-- y el unico camino a las ordenes es
ese renglon de teclas. El de OTRO MODELO es el primero que se cae de la barra, porque es el
unico de los tres que no es camino unico: la tecla `m` y el config hacen lo mismo. Los
numeros estan medidos y escritos en `src/afilado/revision.py:1309` y `revision.py:1377`.

**Cada panel escribe su propio renglon de teclas abajo, y ese renglon se ajusta a lo que
hay.** No hay que acordarse de nada: lo que se puede apretar esta escrito en la pantalla que
se esta mirando.

| Pantalla | El renglon dice |
|---|---|
| CONCEPTOS | `teclas:   1-9 elegir   ENTER guardar aca   n nuevo   d bueno/malo   v ver fotos   b borrar   ESC volver` |
| VER FOTOS | `teclas:   1-8 elegir   b borrar la elegida   espacio ver mas   ESC volver` |
| La ORDEN | `teclas:   s siguiente sierra   x saltear este diente   i informe   n orden nueva   ESC volver` |
| El HISTORIAL | `teclas:   1-4 elegir   ENTER retomar   e exportar   b borrar   n orden nueva   espacio ver mas   ESC volver` |
| BORRAR UNA ORDEN | `teclas:   s = si, borrala      n o ESC = no, dejala` |

En el historial **solo se nombran los digitos que de verdad hay**: con 4 ordenes dice `1-4`,
con 3 ordenes en una pagina de 7 dice `1-3`, y con el historial vacio cambia a
`teclas:   n orden nueva   ESC volver`, porque no hay nada que elegir, retomar, exportar ni
borrar (`src/afilado/panel.py:1075`). Una tecla escrita que no hace nada es lo mismo que un
boton pintado que no se puede apretar. Al reves pasa una sola vez: la confirmacion de
borrado tambien se cancela con la **`q`**, aunque el renglon nombre solo `n o ESC`
(`src/afilado/cli/run_live.py:1182`). Ahi hay una tecla de mas, no una de menos.

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

**Parpadear entre aprobado y rechazado.** Para el veredicto del **modelo entrenado** hay un
freno: ese cartel solo cambia cuando `interfaz.frames_para_confirmar` evaluaciones seguidas
dan lo mismo (3 de fabrica). Con una imagen quieta y un veredicto estable, confirma en 3
evaluaciones.

> **OJO: HOY ESE FRENO NO CORRE, Y ES LO QUE EL OPERARIO VA A VER.** El contador de
> repeticiones vive **adentro** de la rama del clasificador
> (`src/afilado/cli/run_live.py:2134` a `run_live.py:2141`), y sin modelo entrenado esa rama
> no corre: `crear_clasificador()` devuelve None porque `detector.pesos:
> models/afilado_best.pt` no existe --ni en el repo ni al lado del `.exe` del taller--. El
> cartel de hoy lo pone **la memoria**, en otra rama (`run_live.py:2182` a `run_live.py:2210`),
> que reescribe el veredicto y el porcentaje en **cada analisis** --uno de cada cinco
> cuadros, `detector.clasificar_cada_n_frames: 5`-- sin contar ninguna repeticion. O sea:
> mientras no haya modelo, el cartel puede cambiar en cada analisis y tocar
> `frames_para_confirmar` **no cambia nada**. No esta roto: es que el freno es del otro
> veredicto.

Si la pieza se mueve, el veredicto anterior **se borra** en vez de quedar en pantalla:
mostrar un resultado que ya no describe lo que se ve es peor que no mostrar nada.

Con `evaluar_en_vivo: false` el programa no opina hasta que se aprieta REVISAR. Conviene si
la camara mira una zona donde pasa gente o herramientas.

#### De donde sale ese porcentaje

Hay **cuatro** fuentes posibles, y la pantalla **siempre dice cual fue**, abajo del cartel.
Se usa la primera que este disponible:

| Orden | Quien opina | Que dice el detalle en pantalla |
|---|---|---|
| 1 | El **modelo entrenado**, si existe `models/afilado_best.*` | `87% - segun el modelo entrenado` |
| 2 | La **memoria** juzgando: todas las fotos que el taller archivo | `se parece a rechazada (82%) - entre las 24 fotos guardadas` |
| 3 | La **foto archivada** mas parecida, cuando la memoria tiene fotos pero todavia no puede juzgar | `82% - se parece a aprobada, la foto archivada mas parecida` |
| 4 | El **parecido** contra las fotos de modelo | `80% - la que mas se parece de las fotos de modelo: modelo_20261112_161523` |
| 5 | Nadie todavia | `SIN MODELO` + lo que la memoria lleva juntado |

La fila 3 es la del primer dia: la memoria ya tiene fotos pero todavia no hay dos conceptos
con el minimo, asi que no puede dar un veredicto. En vez de saltar a la foto de modelo, se
compara contra **todas las fotos archivadas** y se muestra la que mas se parece, nombrando
su concepto. Catorce fotos archivadas dicen mucho mas que una sola foto de modelo: medido
sobre las fotos del taller, el parecido contra **una** foto suelta se mueve entre **42% y
73%** segun como se apoyo la pieza (`src/afilado/memoria.py:709`).

Que el origen este SIEMPRE escrito no es un adorno. En este proyecto aparecio un
**DESAPROBADO 36%** con las dos imagenes practicamente iguales: el numero venia de un modelo
entrenado con fotos de otra cosa, y mirando la pantalla no habia forma de saberlo. Un
porcentaje sin su origen es indistinguible de uno inventado.

**Que mide cada uno:**

- **Modelo entrenado:** cuanta probabilidad le da a que la pieza sea de las aprobadas. Es lo
  mas confiable, cuando existe. Umbral: `interfaz.umbral_aprobacion` (0,70 de fabrica).
- **Memoria:** cuanto se parece a las fotos ya archivadas, y en que concepto estan esas. Ver
  mas abajo. Ese `82%` es de la foto mas parecida **dentro de** `rechazada`, no la mas
  parecida de toda la memoria: el porcentaje tiene que ser del concepto que el cartel
  nombra. Antes era el mejor de todos y la pantalla llego a decir `se parece a filo_quemado
  (100%)` cuando ese 100% era de una foto archivada en `aprobada`
  (`src/afilado/memoria.py:684`).
- **Parecido contra las fotos de modelo:** compara contra **todas** las fotos de la galeria,
  no solo contra la que esta elegida con la `m`, y se queda con la mejor. El detalle
  **nombra cual la dio**, porque casi nunca es la que eligio el operario: medido, el maximo
  era **80%** con `modelo_20261112_161523` y contra la que nombraba la cabecera
  (`modelo_20261112_161319`) daba **62%**, o sea el veredicto al reves
  (`src/afilado/cli/run_live.py:456`). El texto `comparando con la foto de modelo`, sin
  nombre, solo sale cuando no se pudo saber cual gano. Hay un tope: se miran **24** fotos de
  modelo y las que sobran **no se comparan contra nada**; el parecido contra 24 cuesta 58 ms
  y contra 50 cuesta 117 ms (`run_live.py:1974`).
  Encendido por pedido del taller (`interfaz.veredicto_por_parecido`), con su propio umbral
  `interfaz.umbral_parecido`. **Ojo con lo que dice:** medido sobre las fotos del taller,
  dos tomas de la MISMA pieza dan 78% y dos de piezas DISTINTAS dan 79%. O sea que este
  APROBADO significa "se parece bastante a la foto de al lado", **no** "el filo esta sano".

El umbral de 0,70 es un punto de partida: si pasan piezas que no deberian, subirlo; si frena
piezas buenas, bajarlo. Conviene tocarlo recien cuando haya un modelo entrenado con fotos del
microscopio y se vea como se porta.

| Tecla | Boton | Que hace |
|---|---|---|
| `espacio` o `ENTER` | **REVISAR** | Congela lo que se ve y lo juzga. |
| `espacio` o `ENTER` | **OTRA PIEZA** | Vuelve al vivo. |
| `a` | **GUARDAR BUENA** | Archiva la foto en el primer concepto sano. |
| `r` | **GUARDAR MALA** | Archiva la foto en el primer concepto de rechazo. |
| `g` | **ANOTAR DIENTE** | Dice "este diente ya lo mire": lo anota en la orden con el veredicto que esta en pantalla, **sin** archivar la foto en la coleccion. |
| `c` | **CONCEPTOS** | Abre el panel: elegir a que carpeta va la foto, crear conceptos nuevos, ver y borrar fotos. |
| `o` | **ORDENES** | Abre el **historial de ordenes**: todas las guardadas, la mas nueva arriba. De ahi se retoma una, se exporta, se borra o se carga una nueva. |
| `f` | — | **Agrega** lo que se ve como una foto de modelo nueva. |
| `m` | **OTRO MODELO** | Pasa a la siguiente foto de modelo. |
| `q` o `ESC` | **SALIR** | Cierra el programa. |

**El texto de los botones de guardar sigue al nombre del concepto.** Dice `GUARDAR BUENA` y
`GUARDAR MALA` solo cuando los conceptos se llaman `aprobada` y `rechazada`; con los del
taller dice `GUARDAR SIERRA_NUEVA_TP` y `GUARDAR DIENTES_ROTOS`. Si el nombre es largo el
boton no entra en la barra y **no se dibuja**, pero la tecla sigue archivando igual. Y si no
hay **ningun** concepto sano, la `a` **no hace nada**, a proposito: archivar en el primero de
la lista porque es el primero seria inventar una etiqueta. Lo mismo la `r` sin conceptos de
defecto.

**La `g` es la unica que anota sin archivar.** El boton aparece en vivo y solo con una orden
abierta; sin orden abierta la tecla avisa `NO HAY ORDEN ABIERTA - apreta 'o'`. Despues de
anotar **hay que mover la sierra**: sobre la misma imagen avisa `ESTE DIENTE YA SE ANOTO -
mueva la sierra`. La `a` y la `r` hacen las dos cosas: archivan la foto **y**, con orden
abierta, anotan el diente.

**La `f` no tiene boton en ninguna pantalla.** Es la unica tecla asi en la pantalla del
taller, y al reves, CERRAR ORDEN es lo unico que solo se hace con el mouse.

#### Un resultado POR DIENTE

> **HOY ESTO ESTA APAGADO** (`interfaz.recuadros_de_diente: false`), y el taller tuvo
> razon en pedirlo: *"capta cualquier imagen"*.
>
> El detector esta construido sobre una sola idea --**la cara iluminada del diente es lo
> mas claro de la foto**-- que era cierta con la luz de antes. Desde el 13/11 el taller
> ilumina **a contraluz**: el diente sale negro contra un fondo blanco. Entonces el umbral
> se queda con el fondo y el recuadro abarca todo. Medido sobre las dos fotos reales de
> ese dia: caja `(0,0,640,480)` en las dos, el cuadro entero marcado como si fuera un
> diente.
>
> Y tampoco era solo la luz nueva. Sobre imagenes donde **no hay ningun diente** que
> encontrar:
>
> | lo que se le muestra | dibuja recuadro |
> |---|---|
> | la sierra entera de lejos o contra la pared | **65 de 66** |
> | una mano delante del microscopio | si, del 29% del cuadro |
> | un papel con una sombra | si, del 52% del cuadro |
>
> **Con esto apagado la pieza es LA FOTO**: se juzga el cuadro entero y se archiva el
> cuadro entero. Eso ademas empareja la coleccion --todas las fotos del mismo tamano--
> que es lo que evita que la memoria aprenda el encuadre en vez del filo.
>
> **Lo que se pierde**: el analisis de a dos dientes en una misma foto y las teclas `1..9`
> para elegir cual se archiva. Para volver a encenderlo hay que arreglar antes el detector
> para la luz a contraluz, donde el diente es lo mas **oscuro**. Lo que sigue abajo
> describe como funciona cuando esta encendido.

Lo normal es que entre **un diente** y se analice ese. Pero si entran dos --o tres-- el
programa los separa y **juzga cada uno por su cuenta**: sobre cada diente aparece un
recuadro con **su porcentaje**, verde si paso y rojo si no. El numero y no la palabra: el
color ya dice si paso, y el porcentaje es el dato que se puede discutir.

```
   SIERRA                                ASI SE VE CADA ESTADO
  +--------------------------------+    +---------------+---------------+
  | +--- 1  93% ---+ +-- 2  81% --+|    |               |               |
  | |   (verde)    | |   (rojo)   ||    |   1 afilado   | 2 mal_afilado |
  | +--------------+ +------------+|    +---------------+---------------+
  +--------------------------------+

                    2 DIENTES
   1 bien, 1 para rechazar  -  el porcentaje de cada uno esta sobre su recuadro
   se va a guardar el diente 1   -   con las teclas 1 2 elegis cual
```

A la derecha, en vez de la foto de modelo, aparece **un ejemplo de cada estado**: la foto
guardada a la que se parecio cada diente, con el mismo numero que su recuadro. Asi se ve
contra que se comparo cada cual.

**Se guarda de a un diente.** Con dos en pantalla, las teclas `1` y `2` eligen cual se
archiva (el elegido tiene el borde mas grueso). Marcar los dos con el mismo estado de un
saque etiquetaria mal el que no se miro, y una foto mal etiquetada no se nota hasta que el
modelo sale raro meses despues. Lo que se guarda es **el recorte de ese diente**, no la foto
entera: la memoria juzga el recorte, asi que lo que el programa aprende y lo que compara
tienen que ser lo mismo.

**Como los separa, sin modelo entrenado.** La cara iluminada del diente es lo mas claro de
la imagen; Otsu encuentra ese corte solo, sin ningun numero puesto a mano. Se probaron cinco
caminos sobre las 19 fotos reales del microscopio y sobre fotos armadas pegando dos y tres
dientes:

| metodo | 1 diente da 1 | 2 dientes dan 2 |
|---|---|---|
| **otsu por brillo** | **19 de 19** | **16 de 19** |
| otsu invertido | 19 de 19 | 3 de 19 |
| por textura | 17 de 19 | 2 de 19 |
| umbral adaptativo | 3 de 19 | 2 de 19 |
| por columnas | 8 de 19 | 3 de 19 |

Con tres dientes devuelve los tres 13 de 19 veces y 2 las otras 6. **Cuando no encuentra
nada** --la tapa puesta, una pared-- el diente es la foto entera y el programa se comporta
como venia: no se pierde nada por no detectar.

> Se probo tambien repetir la busqueda tapando lo ya encontrado, para rescatar un segundo
> diente mas oscuro. **Empeora**: sube de 16 a 17 de 19 en el caso de dos, pero en el de UN
> diente empieza a inventar un segundo (de 19 a 11 de 19). Inventar un diente es peor que
> perderlo, porque termina como un resultado de algo que no existe en el informe de la orden.

#### La memoria: el programa aprende de cada foto que se archiva

**No hay cupo que llenar.** Cada foto que se guarda en un concepto se suma AL INSTANTE a lo
que el programa compara, y el veredicto de la foto siguiente ya la tiene en cuenta. No hay
que juntar cincuenta y entrenar: eso sigue existiendo, y es mejor, pero es para despues.

Lo que la memoria hace es preguntarle a las fotos guardadas: *cuales se parecen mas a esta, y
en que concepto estan archivadas esas*. Dos cosas importantes salen de que sea comparacion y
no una red: se puede explicar foto por foto, y no inventa nada que no este guardado.

**La memoria se mide a si misma.** Esconde cada foto guardada y le pregunta a las demas en
que concepto va. De ahi sale el renglon que aparece en pantalla y en el panel:

```
  memoria: acierta 10 de las 13 veces que decidio  -  15 fotos (Diente_sin_placa 2  Dientes_rotos 8  Sierra_Afilada 3  Sierra_Nueva_TP 2)
```

Ese renglon es el de la coleccion del taller, **medido corriendo `Memoria.resumen()` sobre
una copia de `data/coleccion`**: 15 fotos, 13 decisiones, 10 aciertos. Es el unico numero
honesto sobre cuanta precision tiene esto: sale de las fotos de ESTE taller y sube (o baja)
con cada foto que entra. Ese `10 de 13` es un 77% (`src/afilado/memoria.py:801`).

**En ese renglon los conceptos salen en orden alfabetico**, no por cuantas fotos tienen: se
arma con `sorted(conteo.items())` (`memoria.py:879`). Las **filas** del panel de CONCEPTOS, en
cambio, van en el orden de `conceptos.json`. O sea que el renglon de arriba y la lista de
abajo pueden no coincidir, y eso no es una falla.

**Los dos numeros del renglon son distintos a proposito: 13 y 15.** El primero son
**decisiones**, el segundo son **fotos**. Hay fotos sobre las que el programa no decide
--porque lo que mas se les parece es un concepto al que le faltan fotos-- y esas no cuentan
ni como acierto ni como error: quedan aparte, como "sin decidir". En la coleccion del taller
son **2 de 15**, las de `Sierra_Nueva_TP`, que tiene 2 fotos de las 3 que hacen falta
(`memoria.py:802`). El acierto va **adelante** porque con muchos conceptos la linea se
cortaba en el borde de la pantalla justo en el numero que importa.

> Lo que **no** se saca de encima son las fotos que el programa contesta **mal con otro
> nombre**: esas cuentan como error aunque el concepto verdadero sea flaco. Ese diente va a
> llegar al microscopio y va a salir con un nombre equivocado, asi que esconderlo de la
> cuenta seria mentir.

> Ese numero es **optimista si se sacan varias tomas de la misma pieza**: al esconder una, la
> otra sigue ahi y la ubica sin esfuerzo. Para que signifique lo que parece, las fotos de un
> concepto tienen que ser de **piezas distintas**.

> De los 3 errores del 16/11, **2 son un defecto llamado con el nombre de otro defecto** y
> el diente se rechaza igual: del lado correcto de la linea aprobado/rechazado quedan **12
> de 13**. El unico que cruza la linea --`Sierra_Afilada` llamado `Dientes_rotos`-- salio con
> **1,8 puntos** de margen (`memoria.py:805`).

#### Los TRES carteles de "no decidio"

Cuando la memoria no da un veredicto, **no se queda callada**: pone un cartel ambar y abajo
escribe **siempre el motivo**. Son tres carteles distintos porque lo que el operario tiene
que hacer con cada uno es distinto:

| Cartel | Que paso | Que hay que hacer |
|---|---|---|
| **FALTAN FOTOS** | La memoria **si** reconoce lo que ve, pero ese concepto todavia no llego al minimo de fotos | Sacarle **una foto mas a ESE concepto** y archivarla ahi. Es una tarea de un minuto |
| **NO ESTOY SEGURO** | O la foto no se parece a nada de lo guardado, o reconoce dos cosas parejas y no se anima | Mirar el diente a ojo. Si es un caso nuevo, archivarlo donde va |
| **SIN MEMORIA** | La memoria no pudo analizar la imagen | Volver a sacar la foto |

**FALTAN FOTOS es el que mas destraba.** Un concepto al que le faltan fotos **no puede ser
un veredicto**, y tampoco se contesta con el segundo en su lugar, que seria contestar otra
cosa que la que se vio. Se dice a que se parece y cuanto le falta, que es lo unico que el
operario puede arreglar:

```
  FALTAN FOTOS
  se parece a Sierra_Nueva_TP, que tiene 2 de las 3 fotos que hacen falta
  para decidir: sacale una mas y archivala ahi
```

Antes competian todos y un concepto de **2 fotos** podia ganar la votacion. Medido: ese
concepto acertaba **0 de 2** --las dos las llamaba `Dientes_rotos`-- y esos dos errores
dejaban la confianza en 62% para toda la coleccion (`memoria.py:503`,
`configs/config.yaml:281`).

**NO ESTOY SEGURO tapa dos cosas**, y el motivo de abajo dice cual. Una es que la foto no
llegue al `memoria.parecido_minimo`: ahi dice `esta foto no se parece a ninguna de las N
guardadas (la que mas, 0%)`. La otra es el empate. **El orden importa:** si la foto no se
parece a nada, gana este cartel aunque lo que mas se le parezca sea un concepto al que le
faltan fotos. FALTAN FOTOS aparece solo si la foto **si** se parece a algo.

**SIN MEMORIA casi no se puede ver, y conviene saberlo.** Mientras no haya dos conceptos con
el minimo, el programa **ni le pregunta** a la memoria: la pantalla sigue mostrando el
parecido contra las fotos ya archivadas (fila 3 de la tabla de arriba) y recien cuando no hay
ninguna foto aparece `SIN MODELO`. O sea que el primer dia **no** se ve SIN MEMORIA. Queda
para el caso en que la memoria no puede analizar la imagen.

#### La confianza: de donde sale el "no me animo"

Un APROBADO salido de un empate es peor que no decir nada, porque el operario no tiene como
saber que fue un empate. Asi que antes de contestar, el programa se pregunta cuanta ventaja
le saca el primer concepto al segundo --**el margen**-- y mira **que paso de verdad** en sus
propias decisiones igual de ajustadas. Si de las doce veces que estuvo asi de parejo acerto
seis, la confianza es 50%, no llega al minimo de 70% (`memoria.confianza_minima`) y sale NO
ESTOY SEGURO. No hay ningun umbral de margen inventado.

**El margen se mide solo entre los conceptos que pueden decidir**, o sea los que ya llegaron
al minimo de fotos. Un concepto al que le faltan fotos no puede ser el veredicto, asi que
tampoco tiene por que achicarle el margen a los dos que si pueden. Antes un concepto flaco se
metia en el medio y bajaba la confianza de una decision que en realidad era comoda.

> **EL PORCENTAJE DE LA PANTALLA NO ES EL NUMERO CON EL QUE SE DECIDE, y restarlos a mano no
> da el margen.** Son dos cuentas distintas sobre las mismas fotos:
>
> | Numero | Que es | Donde esta |
> |---|---|---|
> | el **porcentaje que se ve** | la foto mas parecida **de adentro** del concepto que gano | `memoria.py:691` |
> | el **puntaje** de cada concepto | el **promedio** de sus mejores candidatas (`vecinos_por_concepto`, 3 de fabrica) | `memoria.py:777` |
>
> El margen --y con el la confianza-- se mide entre los **puntajes**, no entre los
> porcentajes de pantalla (`memoria.py:629`). O sea que leer `82%` en una consulta y `80%` en
> otra **no** quiere decir que el margen haya sido 2 puntos. El porcentaje de pantalla esta
> ahi para contestar "a que se parece esto y cuanto", que es otra pregunta.

**Pero eso es lo que hace CUANDO TIENE CON QUE.** Hay tres situaciones y el cartel dice en
cual esta, con otras palabras:

| Decisiones medidas | Que mira | Como lo dice la pantalla |
|---|---|---|
| menos de **8** | nada: no hay numero que signifique algo, y decide **sin** ese control | `todavia hay pocas fotos para medir cuanto acierta` (`src/afilado/revision.py:943`) |
| de **8 a 15** | su acierto **global**, y dice que son pocas para medir esta decision aparte | `acierta 10 de las 13 veces que decidio, y son muy pocas para medir aparte una decision tan ajustada como esta (Sierra_Afilada contra Dientes_rotos, 1.8 puntos)` (`memoria.py:681`) |
| desde **16** | la banda de margenes parecidos a este | `Sierra_Afilada y Dientes_rotos quedaron a 1.8 puntos, y en decisiones asi de ajustadas la memoria acierta X% de las veces` (`memoria.py:667`) |

El cartel del medio **no dice "la memoria acierta X%" a secas**, y es a proposito: ese numero
es el acierto de **todas** sus decisiones, no el de una decision ajustada como la que se esta
mirando, asi que el cartel dice las dos cosas --cuanto acierta en general y que son muy pocas
para medir esta aparte-- en vez de ponerle a un numero el nombre de otro. Es la leccion del
62% que freno el taller.

**Por que 16.** La banda es el 25% de casos mas cercanos a este margen, con piso de 8. Con 15
casos manda el piso: la ventana son 8 casos, o sea el **53%** de la memoria, y adentro entran
margenes de **1,8 a 17,9 puntos**. Eso no es "decisiones asi de ajustadas", es el acierto de
casi toda la memoria con otro nombre. Recien desde 16 decisiones la ventana vuelve a ser una
banda de verdad (`memoria.py:230` y `memoria.py:243`, comprobado en
`tests/test_memoria.py:701`).

> **Este es el numero que freno el taller el 16/11/2026.** Con 15 fotos y 4 conceptos la
> ventana daba **62% para cualquier margen por debajo de 11 puntos** --62% a 0, 2, 5, 6.8 y 10
> puntos-- asi que el programa dijo NO ESTOY SEGURO en **todos** los dientes del dia. No
> dudaba de ese diente: dudaba de todos. El 62% estaba bien calculado; lo que estaba mal era
> el nombre que el cartel le ponia, porque no media "decisiones asi de ajustadas". Medido y
> escrito en `memoria.py:266` y `configs/config.yaml:305`.

Ademas del porcentaje, cuando la diferencia esta **en un lugar** y no repartida por toda la
imagen, lo dice: *la mayor diferencia esta arriba a la derecha*. Para que nombre una zona,
esa zona tiene que destacarse **3,0 veces** sobre la mediana de las nueve: medido, el mismo
diente con otra exposicion da 1,03-1,04, dos tomas distintas 1,47-2,83 y una mancha agregada
3,58 (`memoria.py:375`).

**Para que empiece a opinar** hacen falta `memoria.minimo_por_concepto` fotos (3 de fabrica)
en al menos **dos** conceptos distintos. Con uno solo no hay decision posible: todo se parece
a lo unico que hay. **Ese 3 no es un numero medido**, y queda dicho:
`configs/config.yaml:273` lo deja escrito asi.

> **HOY LA MEMORIA ESTA A UNA FOTO DE DEJAR DE JUZGAR, y conviene saberlo antes de borrar
> nada.** Medido sobre la coleccion del taller: `Dientes_rotos` 8, `Sierra_Afilada` 3,
> `Diente_sin_placa` 2, `Sierra_Nueva_TP` 2. Los unicos **dos** conceptos que pueden decidir
> son `Dientes_rotos` y `Sierra_Afilada`, y `Sierra_Afilada` tiene **exactamente** el minimo
> de 3. Si una de esas tres se manda a la papelera desde VER FOTOS, queda un solo concepto
> que decide, `lista_para_juzgar()` pasa a false (`memoria.py:486`), el programa deja de
> preguntarle a la memoria y la pantalla vuelve al `se parece a ..., la foto archivada mas
> parecida` de la fila 3. No es una rotura: es que se cayo abajo del minimo. Se arregla
> archivando una foto mas ahi.

> **`memoria.habilitada` apaga la memoria entera.** Es la llave que no se parece a las
> otras: `minimo_por_concepto`, `parecido_minimo` y `confianza_minima` ajustan **como**
> decide la memoria, pero esta decide **si existe**. En false el programa ni la construye
> (`src/afilado/cli/run_live.py:1671`) y el veredicto vuelve al parecido contra la foto de
> modelo, con los dos avisos que ese numero se lleva puestos (78% la misma pieza, 79% dos
> piezas distintas). Si la pantalla dejo de nombrar conceptos de un dia para el otro, esta
> es la llave que hay que mirar primero.

**Se corrige sola, archivando.** Si la memoria dice "aprobada" y era mala, se aprieta GUARDAR
MALA: esa foto entra al concepto correcto y la proxima vez pesa. Eso es todo el
entrenamiento que hay que hacer en el dia a dia.

Cuesta poco, y lo que esta medido es esto: una comparacion exacta de dos fotos sale en
**0,47 ms**, y un veredicto con cinco conceptos son 60 comparaciones, **28 ms**, cueste lo
que cueste la memoria entera --porque primero descarta barato (0,1 microsegundo por foto) y
solo compara en serio las mejores candidatas de cada concepto--. Esta escrito al lado del
codigo que lo hace, en `src/afilado/memoria.py:24` a `memoria.py:31`.

Lo que **si** se siente es volver a medirse: archivar una foto hace que la memoria se mida
contra si misma de nuevo, y eso tarda **118 ms con 15 fotos, 575 ms con 41 y 1.554 ms con
121** (`src/afilado/cli/run_live.py:1861`). Por eso el cartel sale primero y la cuenta
despues.

En `data/memoria/` queda guardado tambien **el acierto medido junto con la regla
con la que se midio**: si se toca `memoria.minimo_por_concepto`, la medicion vieja se tira y
se vuelve a medir una vez al abrir. Por eso el numero puede cambiar despues de tocar el
config, y no es un error. La cache vive en `data/memoria/` y se puede borrar sin perder nada: se rehace
sola leyendo las fotos.

> **TODOS LOS CONCEPTOS SE SACAN CON EL MISMO ENCUADRE.** Esta es la regla que mas plata
> puede costar si no se respeta, y tambien sale de una medicion.
>
> La memoria separa por **como se ve la foto entera**, y el encuadre es parte de como se ve.
> Medido sobre las 14 fotos que saco el taller el 12/11/2026, que estan **todas** archivadas
> en `afilado` --para el operario son todas la misma cosa-- pero 10 salieron del cuadro casi
> entero y 4 son recortes chicos de un diente: etiquetandolas a proposito por encuadre y
> escondiendo una por vez, la memoria acerto **14 de 14, todas al 100% de confianza**. No
> miro el filo ni una vez.
>
> El mismo efecto esta en el porcentaje de la pantalla. Comparando cada foto contra las otras
> trece:
>
> | encuadre | parecido | pasan el 70% |
> |---|---|---|
> | las 10 de cuadro entero | 81,6% a 92,1% (mediana 88,7%) | 10 de 10 |
> | los 4 recortes de diente | 39,0% a 68,6% (mediana 58,9%) | **0 de 4** |
>
> Y la foto mas parecida fue siempre del mismo encuadre, 14 de 14.
>
> **Que significa esto en la practica:** si `afilado` se saca de lejos y `mal_afilado` de
> cerca, el programa va a decir `mal_afilado 100%` y va a estar leyendo el encuadre, no el
> filo. El informe de un reclamo diria que un diente esta mal por como se apoyo la camara.
> Saca todas las fotos de todos los conceptos a la misma distancia y con el diente ocupando
> el mismo lugar del cuadro.
>
> De donde salen hoy los dos encuadres: lo que se archiva es el **recorte del diente
> elegido**, y cuando el programa no logra separar ningun diente, ese "recorte" termina
> siendo el cuadro entero. Por eso en la misma carpeta hay fotos de 640x446 y de 253x257.

> **APOYAR LA PIEZA SIEMPRE CON LA MISMA ORIENTACION.** Es la otra recomendacion de uso que
> hace falta, y sale de una medicion, no de una corazonada. Volviendo a juzgar las **14
> fotos del microscopio de hoy** despues de perturbarlas como pasa de verdad al reapoyar la
> pieza, el parecido se mantiene sobre el 70% en:
>
> | perturbacion | se mantiene | | perturbacion | se mantiene |
> |---|---|---|---|---|
> | quieta (control) | 14 de 14 | | mas luz (+25%) | 14 de 14 |
> | corrida 10 px | 14 de 14 | | menos luz (-25%) | 14 de 14 |
> | corrida 25 px | 14 de 14 | | girada 3 grados | 13 de 14 |
> | corrida 50 px | 14 de 14 | | girada 7 grados | 8 de 14 |
> | | | | girada 15 grados | **1 de 14** |
>
> O sea: **correrla y la luz no mueven el veredicto; girarla si**, y mas de lo que decia la
> tabla anterior, que estaba medida con la camara que ya no se usa y daba 10 de 19 a 15
> grados. El programa normaliza la luz y busca el mejor calce para el encuadre --por eso
> aguanta 50 px de corrimiento sin perder un punto-- pero no prueba angulos, asi que una
> pieza bien girada se le vuelve otra pieza.

#### El panel de CONCEPTOS (tecla `c`)

Un **concepto** es una carpeta de fotos que significan lo mismo: `aprobada`, `rechazada`,
`diente_astillado`, `filo_quemado`. Los decide el taller, no el config.

Asi se ve hoy con los conceptos del taller (copiado de lo que dibuja el programa):

```
  CONCEPTOS
  memoria: acierta 10 de las 13 veces que decidio  -  15 fotos (Diente_sin_placa 2  Dientes_rotos 8  Sierra_Afilada 3  Sierra_Nueva_TP 2)

  [ ] 1  foto de modelo       0 fotos  -  es la foto con la que se compara, no un concepto
  (o) 2  Dientes_rotos        8 fotos  -  la memoria acierta 8 de 8                  <- rojo: para tirar
  (o) 3  Sierra_Nueva_TP      2 fotos  -  le falta 1 foto mas para que la memoria pueda decidir con el
  (o) 4  Diente_sin_placa     2 fotos  -  le falta 1 foto mas para que la memoria pueda decidir con el
  (o) 5  Sierra_Afilada       3 fotos  -  la memoria acierta 2 de 3                  <- verde: buena

  teclas:   1-9 elegir   ENTER guardar aca   n nuevo   d bueno/malo   v ver fotos   b borrar   ESC volver

  [GUARDAR ACA] [NUEVO CONCEPTO] [VER FOTOS] [BORRAR CONCEPTO]        [VOLVER]
```

Ese dibujo es el de **hoy**, copiado de lo que dibuja `panel.componer` con la coleccion y la
galeria de verdad del taller, abierto como lo abre `run_live`. Tres cosas que se leen ahi y
conviene saber:

- **La galeria de fotos de modelo se lleva la fila 1**, con un cuadrado hueco en vez de punto
  (`src/afilado/panel.py:419`), asi que **el primer concepto es la tecla `2`**, no la `1`. Hoy
  esa fila dice `0 fotos` porque la galeria del taller esta vacia: por eso el panel derecho de
  la pantalla principal pide poner una pieza buena y apretar `f`.
- **El orden de las filas es el de `conceptos.json`, no alfabetico.** El renglon de la memoria
  de arriba **si** es alfabetico (`sorted`, `memoria.py:879`), asi que los dos no coinciden y
  no es un error: son dos listas con dos ordenes.
- **GUARDAR ACA esta SIEMPRE**, porque al panel se entra con el cuadro de la pantalla en la
  mano. No hay forma de abrirlo "para mirar": `run_live` lo abre con
  `pendiente=a_mostrar.copy()` (`src/afilado/cli/run_live.py:2447`), tanto con la `c` como
  con la `o`, asi que la condicion de `panel.py:470` siempre da verdadera. **Medido**
  componiendo el panel como lo abre `run_live` con la coleccion del taller: los botones son
  `GUARDAR ACA`, `NUEVO CONCEPTO`, `VER FOTOS`, `BORRAR CONCEPTO` y `VOLVER`, y
  `guardar_aqui` esta dibujado (x=18..205).

> **OJO CON EL `ENTER` DE REFLEJO: la fila que viene elegida es la 1, `foto de modelo`.** El
> panel abre con `elegido = 0` (`src/afilado/panel.py:145`) y la fila 1 de esa lista es la
> galeria de modelos (`_lista_para_panel`, `run_live.py:579`). El `ENTER` no pregunta nada y no
> mira si habia algo pendiente: archiva el cuadro en la fila elegida y **cierra el panel**
> (`run_live.py:1248`). O sea que un `ENTER` apretado por costumbre al entrar **agrega el
> cuadro que estaba en pantalla como foto de modelo nueva y la deja elegida**
> (`galeria.agregar` + `recargar(preferida=...)`, `run_live.py:1845`): cambia la foto contra la
> que se compara todo lo demas. **Para solo mirar se sale con `ESC`, `q` o VOLVER**, que no
> archivan nada.

**Las filas 3 y 4 son las que destraban el cartel FALTAN FOTOS.** Ahi el operario lee que le
falta **una** foto y a que concepto: es una tarea de un minuto. Sin ese renglon la fila decia
`2 fotos` y nada mas, porque un concepto flaco no aparece en la cuenta de aciertos --la
memoria no decide con el, asi que no lo mide--.

Cuando la autoevaluacion midio **menos fotos de las que tiene el concepto**, la fila lo dice
con otras palabras: `acierta 9 de las 10 que midio` en vez de `acierta 9 de 10`. Con muchas
fotos se mide sobre un recorte, y un `acierta 9 de 10` al lado de muchas fotos se leeria como
si las hubiera medido todas.

**Desde cuando se mide sobre un recorte:** el tope es `memoria.maximo_autoevaluacion: 120`
(`configs/config.yaml:356`). Arriba de 120 **fotos** se mide sobre las 120 mas nuevas por
fecha y el renglon pasa a decir `de las ultimas` (`memoria.py:892`). Con 0 se mide siempre
sobre todas, cueste lo que cueste. **Ojo con los dos numeros:** ese tope cuenta **fotos** y el
de la pantalla cuenta **decisiones**, y no son el mismo --medido, 15 fotos dan 13
decisiones--. Una orden con muchas fotos archivadas en un dia llega al tope, y ahi el numero
de la pantalla cambia sin que nadie toque nada.

- **A que carpeta va esta foto**: se elige la fila (con el mouse o con las teclas `1`..`9`) y
  se aprieta **GUARDAR ACA** o `ENTER`. A la derecha se ve la foto que se esta por archivar.
  Al guardar, el panel **se cierra** y se vuelve a la pantalla del taller. El numero de fila
  se dibuja hasta la `9` nada mas, porque del decimo en adelante no hay digito.
- **Si hay mas conceptos que filas en pantalla**: se baja con `s` y se sube con `w`, dando la
  vuelta. La lista se corre sola para que el elegido siempre se vea, asi que **ningun
  concepto queda fuera de alcance**, y el renglon de teclas lo avisa cuando hace falta.
- **Crear un concepto**: **NUEVO CONCEPTO** o `n`, se escribe el nombre y `ENTER`. Queda
  creada la carpeta y se puede usar en el acto, sin cerrar el programa. En esa pantalla se
  confirma con `ENTER` o **CREAR** y se deja con `ESC` o **CANCELAR**. Ojo: ahi las teclas
  son **letras**, asi que la `q` escribe una q y no cierra nada. Los espacios pasan a `_`.
- **Decir si es bueno o malo**: el **punto de color** de la izquierda se puede apretar, o la
  tecla `d`. Verde = pieza buena, rojo = pieza para tirar. **Esto es lo que decide si la
  pantalla pinta APROBADO o DESAPROBADO** cuando la memoria elige ese concepto, asi que un
  concepto nuevo como `diente_astillado` hay que marcarlo en rojo. Sobre la fila
  `foto de modelo` la `d` **no hace nada** y lo dice: no es un concepto, no es ni bueno ni
  malo.
- **Ver y tirar fotos**: **VER FOTOS** o `v` muestra las del concepto elegido. Hay **ocho
  miniaturas por pagina**; se elige con `1`..`8` o con el mouse, y los digitos solo alcanzan
  a las que **se ven**. Se pasa de pagina con la **barra espaciadora** o con el boton
  **SIGUIENTES**, que aparece solo si hay mas de ocho fotos. La `b` --boton **BORRAR
  ESTA**-- manda la elegida a la papelera; aca no hay pantalla de confirmacion, porque va a
  `_papelera` y de ahi se saca. Se vuelve a la lista con `ESC`, `q`, otra vez `v`, o
  **VOLVER**. Sirve para sacar la foto movida o la que quedo en la carpeta equivocada.
- **Borrar un concepto**: **BORRAR CONCEPTO** o `b`, con una pantalla de confirmacion que se
  contesta con `s` (o **SI, BORRAR**) y se cancela con `n`, `ESC` o **NO, DEJALO**. **El
  `ENTER` ahi no confirma, a proposito**: en la lista el `ENTER` es "guardar aca", y una `b`
  de mas seguida del `ENTER` de siempre se llevaria el concepto. Sobre `foto de modelo` no se
  puede borrar y el programa lo dice: sus fotos se sacan de a una desde VER FOTOS.
- **Salir del panel**: `ESC`, `q` o el boton **VOLVER**.

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

#### El HISTORIAL de ordenes (tecla `o` / boton ORDENES)

**La `o` abre el historial, no el formulario.** Es la lista de **todas** las ordenes
guardadas, la mas nueva arriba, con la que esta en curso ya elegida. Es la puerta a todo lo
que tiene que ver con ordenes, y la unica pantalla desde donde se puede contestar el reclamo
de la semana pasada sin abrir el Explorador de Windows.

Asi se ve hoy, con las cuatro ordenes de verdad del taller (copiado de lo que dibuja
`panel.componer` leyendo los cuatro `.json`):

```
  ORDENES DE AFILADO
  4 orden(es)  -  pagina 1 de 1  -  la mas nueva arriba  -  elegida: 5113546

  ( ) 1  5113546   546464                                                    SIN PRUEBA GUARDADA
         abierta 16/11/2026 09:34:20  -  cerrada 16/11/2026 12:11:25  -  15 de 80 revisados  -  15 para rechazar
  ( ) 2  1234      1234                                                            7 CON PRUEBA
         abierta 12/11/2026 16:33:22  -  cerrada 12/11/2026 16:39:09  -  7 de 80 revisados  -  0 para rechazar
  ( ) 3  123       5564                                                            7 CON PRUEBA
         abierta 12/11/2026 16:14:21  -  cerrada 12/11/2026 16:16:52  -  7 de 80 revisados  -  0 para rechazar
  ( ) 4  213132    sin cliente anotado                                      SIN PRUEBA GUARDADA
         abierta 12/11/2026 15:26:37  -  cerrada 12/11/2026 15:27:44  -  0 de 96 revisados  -  0 para rechazar

  2 ordenes ya tienen prueba guardada: a esas los .exe de respaldo del taller ya no las abren.
  teclas:   1-4 elegir   ENTER retomar   e exportar   b borrar   n orden nueva   espacio ver mas   ESC volver

  [RETOMAR ESTA ORDEN] [EXPORTAR A UN ZIP] [BORRAR] [ORDEN NUEVA] [VOLVER]
```

**Cada orden son DOS renglones, no uno.** Arriba van el punto, el digito, el numero, la
etiqueta **ABIERTA** si lo esta y la columna de la prueba pegada a la derecha; abajo, mas
chico, la fecha de apertura, la de cierre, cuantos dientes se revisaron, cuantos para
rechazar y --cuando hace falta-- el nombre del archivo o el aviso de los nombres de foto
sueltos (`src/afilado/panel.py:811` y `panel.py:996`).

**El punto de la izquierda dice en que anda la orden**, con el mismo idioma que la lista de
conceptos (`panel.py:953`):

| Punto | Que es |
|---|---|
| circulo **verde lleno** | la orden **abierta**, la que esta en curso |
| circulo **hueco** | una orden **cerrada** |
| circulo **rojo lleno** | el `.json` **no se pudo leer** |

En el dibujo de arriba los cuatro puntos son huecos porque **las cuatro ordenes del taller
estan cerradas**, con su fecha de cierre cargada en el `.json`. Ninguna esta abierta hoy.

**La bajada dice cual fila esta elegida y en que pagina se esta** (`panel.py:924`). Eso no es
un adorno: **EXPORTAR y BORRAR trabajan sobre la elegida**, y con el paginado la elegida puede
estar en una pagina que no se esta viendo. El numero que figura ahi es el que se va a exportar
o a borrar.

> **HOY NINGUNA DE LAS CUATRO SE PUEDE RETOMAR.** Las cuatro tienen el campo `cerrada`
> cargado --la 123 el 12/11 16:16:52, la 1234 el 12/11 16:39:09, la 213132 el 12/11 15:27:44 y
> la 5113546 el 16/11 12:11:25-- y una orden cerrada no se reabre
> (`src/afilado/cli/run_live.py:998`). Sobre estas cuatro, del historial sirven la `e`
> (exportar), la `b` (borrar) y la `n` (cargar una nueva). El `ENTER` contesta por que se
> niega y manda a exportar.

| Tecla | Boton | Que hace |
|---|---|---|
| `1`..`N` | clic en la fila | Elegir una fila **de esta pagina**: las que se ven. Cuantas entran depende de la camara |
| `w` / `s` | — | Subir y bajar por **toda** la lista, cambiando de pagina sola |
| `espacio` | — | Pasar a la pagina siguiente |
| `ENTER` | **RETOMAR ESTA ORDEN** | Entra a la orden elegida y la deja en curso |
| `e` | **EXPORTAR A UN ZIP** | Arma el paquete del reclamo |
| `b` | **BORRAR** | Lleva a la confirmacion |
| `n` | **ORDEN NUEVA** | Abre el formulario para cargar una a mano |
| `ESC` o `q` | **VOLVER** | Cierra el panel y vuelve a la pantalla del taller |

**Los digitos solo alcanzan a las filas que se ven, y es a proposito**: lo que se aprieta
tiene que ser lo que se ve. El renglon de teclas nombra solo los digitos que de verdad hay
--con 3 ordenes en una pagina de 7, el `5` no elige nada--. A las filas que no entran se
llega con `w`/`s` o con la barra espaciadora. **Cuantas filas entran no es un 8 fijo**: lo
mide el alto del lienzo, y esta medido: camara de 640x480 da 8 filas; 1280x720 y 1920x1080
dan 7, y 8 cuando hay una orden abierta con la barra de dientes
(`src/afilado/panel.py:110`).

**RETOMAR se niega en tres casos y dice por que, y en cada uno queda algo distinto por
hacer:**

| Por que se niega | Que se puede hacer igual |
|---|---|
| El `.json` **no se puede leer** | **Nada mas que mandarla a la papelera** (`b`) y abrirla a mano con el Explorador |
| La orden esta **CERRADA** --su informe ya se entrego-- | Exportarla con `e`, o borrarla |
| El archivo **no se llama como su numero**, porque anotar ahi escribiria en otro archivo | Exportarla o borrarla, nada mas |

> **DE UNA ORDEN ILEGIBLE NO SE PUEDE EXPORTAR TAMPOCO**, y esa es la unica de las tres que
> queda sin salida: la `e` corta con `el archivo X.json no se puede leer: no se exporta como
> prueba algo que el programa no pudo abrir` (`src/afilado/cli/run_live.py:1043`), y por
> abajo `Ordenes.exportar()` devuelve None igual (`src/afilado/ordenes.py:1691`). A esa fila
> **ni le dibuja los botones** RETOMAR ni EXPORTAR (`src/afilado/panel.py:1094`). No es una
> falla: un zip de reclamo armado con un archivo que nadie pudo leer seria presentar como
> prueba algo que no se sabe que dice.

**Que se ve cuando un `.json` no se puede leer.** La fila sale igual --esconderla seria
esconder justo la que hay que arreglar-- con el punto **rojo lleno**, la columna de la prueba
diciendo **NO SE PUDO LEER** en rojo (`panel.py:788`) y, abajo, la pantalla escribe
`Esa orden no se puede leer: solo se puede mandar a la papelera y mirarla a mano.`
(`panel.py:1027`). Lo que hay que hacer es abrir el `.json` con el Explorador y ver que tiene;
si no se arregla, `b` la saca del medio y queda guardada en la papelera.

**De esta pantalla, la unica tecla que pide confirmacion es la `b`.** Las otras trabajan de
una: el `ENTER` retoma la orden y escribe en el disco sin preguntar nada
(`src/afilado/cli/run_live.py:1029`), la `n` abre el formulario de una (`run_live.py:981`) y
`ESC` cierra el panel de una (`run_live.py:935`). **La `e` no la pide porque solo agrega:**
arma **un** `.zip` en `data/ordenes/exportado/` para mandar por mail o llevar en un pendrive,
y el cartel dice la carpeta, el nombre y **cuanto pesa**. Adentro va:

- `orden.json`, copiado **tal cual** del disco, porque es el documento.
- `informe.txt` y `informe.csv` **armados en ese momento**, no la copia vieja que esta al
  lado de la orden.
- las **fotos de prueba que los dientes nombran**, en la misma ruta que el informe les da.
- `procedencia.txt`, si lo hay.

Lo que **ningun diente nombra no viaja y no se borra**: queda en la carpeta y el informe dice
cuantas quedaron. Y si una prueba nombrada no entro al paquete, el informe pone un
`(NO ESTA)` al lado de ese diente en vez de callarselo.

> **Cada `e` escribe un zip NUEVO. Nadie los borra, ni cuando se borra la orden.** No hay
> tope ni limpieza: esa carpeta se vacia a mano con el Explorador cuando moleste. Medido, de
> **2,8 KiB** una orden sin pruebas a **26,9 MiB** (en 847 ms) una de 4 sierras de 80 dientes
> con sus 320 fotos: entre el mas liviano y el mas pesado hay diez mil veces. Diez `e` de esa
> orden gorda en un dia son **269 MiB** que nadie borra. Por eso el cartel dice cuanto pesa
> cada uno (`src/afilado/cli/run_live.py:316` y `run_live.py:323`).
>
> **Y DOS `e` SEGUIDAS NO SE PISAN: la segunda sale con `_1` pegado al nombre.** El sello del
> nombre tiene resolucion de un segundo y exportar tarda mucho menos --medido, la orden de
> verdad 5113546 entre 3,1 y 15,7 ms-- asi que dos `e` del mismo segundo caen en el mismo
> nombre sin ninguna dificultad. Antes el segundo zip pisaba al primero, quedaba uno solo y el
> cartel decia igual que se habia escrito uno nuevo (`src/afilado/ordenes.py:1720`). O sea que
> el que vaya a buscar el archivo del cartel a la carpeta se va a encontrar **dos nombres
> parecidos**: el del cartel es el que vale.

**BORRAR (`b`) no borra de un saque**: lleva a una confirmacion que se contesta con `s` (o
**SI, BORRAR**) y se cancela con `n`, `ESC`, la `q` o **NO, DEJALA**. El `ENTER` **no
confirma**, a proposito: en el historial el `ENTER` es "retomar". Lo que hace es **mover** a
`data/ordenes/_papelera/<fecha y hora>/` el `.json` de la orden, sus informes y su carpeta de
fotos.

- **A la papelera va LO QUE HABIA, no una lista fija.** El `.json` siempre; los dos informes,
  cada uno **si esta** al lado (`src/afilado/ordenes.py:1578`); y la carpeta de fotos **solo
  si la orden tenia prueba guardada** (`ordenes.py:1564`, que mueve el deposito nada mas que
  `if pruebas is not None and pruebas.is_dir()`). La pantalla de confirmacion promete
  exactamente eso y no mas: el renglon "con su informe y sus fotos" sale **solo** cuando la
  fila tiene pruebas (`src/afilado/panel.py:1255`).
- **Si la fila decia SIN PRUEBA GUARDADA o PRUEBAS QUE NO ESTAN, en la papelera no hay
  ninguna foto que rescatar.** Hoy es el caso de dos de las cuatro ordenes del taller, la
  213132 y la 5113546: `data/pruebas` solo tiene las carpetas `123` y `1234`. Esas fotos hay
  que buscarlas donde dice el aviso de prueba perdida.
- **Se mueve la carpeta de pruebas ENTERA**, asi que tambien viajan las fotos **huerfanas**
  --las que ningun diente nombra, de un guardado que fallo-- y esas la pantalla de
  confirmacion **no las cuenta** (`ordenes.py:1564`). En la papelera puede haber **mas** fotos
  que las N que se prometieron. Nunca menos: eso esta medido en cuatro casos
  (`panel.py:1105`).
- **De ahi se saca con el Explorador**: se copia el `.json` de vuelta a `data/ordenes` y la
  carpeta de fotos a `data/pruebas`, con su nivel de carpeta.
- **El `.json` es lo ultimo que se mueve**, y eso importa: si algo se traba --una foto abierta
  en el visor de Windows-- la orden sigue **entera y visible** en el historial, que es la
  unica pantalla desde donde se la puede exportar.
- **Nada se deshace solo.** Si el movimiento quedo a medias el programa lo dice, nombra la
  carpeta de papelera donde mirar, y ese aviso **queda pegado**: no lo tapa la tecla
  siguiente.
- **Apretar `b` otra vez completa ESA misma carpeta de papelera, no abre otra**
  (`src/afilado/ordenes.py:2134`). Es a proposito: si abriera una nueva, la orden quedaria
  partida en dos carpetas con fotos repetidas y el que va a rescatarla no sabria cual es la
  buena. O sea que el reintento se hace desde la misma fila, y la carpeta que nombro el aviso
  sigue siendo la que hay que mirar.
- **Si la que se borra es la orden ABIERTA, el programa queda sin ninguna orden en curso** y
  los dientes que se revisen despues **no se anotan en ningun lado** hasta abrir otra. La
  pantalla de confirmacion lo avisa en ambar antes de apretar --`ES LA ORDEN ABIERTA: el
  programa va a quedar sin ninguna orden abierta.` (`panel.py:1250`)-- y el programa lo repite
  despues de borrar.
- **Los `.zip` ya exportados no se van**: esos quedan en `exportado/`.

**ORDEN NUEVA (`n`) es el unico camino al formulario.** La `o` ya no cae ahi.

#### La ORDEN DE AFILADO y su informe (se entra con `ENTER` desde el historial)

Sirve para una cosa concreta: **cuando llega un reclamo**, poder contestar que se vio en
cada diente de esa sierra, con fecha y hora.

Asi se ve la orden 5113546 del taller (copiado de lo que dibuja el programa leyendo su
`.json`):

```
  ORDEN DE AFILADO
  ORDEN 5113546   -   sierra 1 de 1   -   diente 16 de 80

  Nota de pedido:  55131
  Cliente:  546464
  Abierta:  16/11/2026 09:34:20
  Sierras:  1   -   dientes en total: 80
  Revisados:  15 de 80   -   para rechazar: 15

  SIERRA 1:  15 de 80 dientes revisados   -   15 para rechazar
  (la barra de 80 cuadritos: los primeros 15 en rojo, los otros 65 en gris)

  teclas:   s siguiente sierra   x saltear este diente   i informe   n orden nueva   ESC volver

  [SIGUIENTE SIERRA] [SALTEAR DIENTE] [GUARDAR INFORME] [ORDEN NUEVA] [CERRAR ORDEN] [VOLVER]
```

**Los cinco datos van uno abajo del otro, no en dos columnas**, y el tercero es la **fecha de
apertura** de la orden (`src/afilado/panel.py:596`). La fecha importa: es el dato con el que
se contesta "cuando se revisio esta sierra".

**La barra de dientes son cuadritos de COLOR, nada mas.** No hay ninguna `X` ni ningun `?`
dibujados (`panel.py:630`): lo unico que distingue a un diente rechazado es el **rojo**.

| Color del cuadrito | Que es ese diente |
|---|---|
| verde | revisado y **aprobado** |
| rojo | revisado y **para rechazar** |
| ambar | revisado y **no se pudo decidir** |
| gris | **todavia no se reviso** |

| Tecla | Boton | Que hace |
|---|---|---|
| `s` | **SIGUIENTE SIERRA** | Pasa a la sierra siguiente y avisa `SIERRA n de N` |
| `x` | **SALTEAR DIENTE** | Saltea el diente que seguia. Con la sierra completa lo dice en vez de saltear |
| `i` | **GUARDAR INFORME** | Escribe el informe y la planilla |
| `n` | **ORDEN NUEVA** | Deja el formulario vacio encima de la orden en curso |
| — | **CERRAR ORDEN** | Cierra la orden y recien despues escribe el informe |
| `ESC` o `q` | **VOLVER** | Sube al **historial**, que se relee del disco |

> **CERRAR ORDEN es lo unico del programa que se hace SOLO con el mouse**: no tiene tecla, y
> por eso el renglon de teclas de esa pantalla no lo nombra. Cierra primero y escribe el
> informe despues, porque al reves el papel que se le entrega al cliente salia impreso con
> "Cerrada: sigue abierta".

**El formulario** (se llega con `n`, o con **ORDEN NUEVA** desde el historial) son cinco
campos en orden: Orden de afilado, Nota de pedido, Cliente, Cuantas sierras y Dientes por
sierra. Se escribe campo por campo hasta 40 letras, el `ENTER` pasa al siguiente y el `ENTER`
del ultimo --o el boton **EMPEZAR LA ORDEN**-- la crea. En los dos campos de numero una letra
no entra y avisa `aca va un numero`. La `q` ahi es **una letra** y no cierra nada, porque hay
clientes que se llaman Quiroga.

**`ESC` o CANCELAR vuelve de donde se vino**, no siempre al historial
(`src/afilado/cli/run_live.py:647`):

- si el formulario se abrio **desde el historial**, se cae en el historial;
- si se abrio con la `n` **estando adentro de una orden**, se vuelve a **esa orden**, a su
  pantalla de avance. La orden en curso no se pierde por abrir el formulario y arrepentirse.

- **Los dientes por sierra son los MISMOS para todas.** Una orden con sierras de 24 y de 36
  hay que partirla en dos.
- **`r` / RETOMAR LA GUARDADA**: si con ese numero ya hay una orden **sin cerrar**, el
  programa ofrece seguir esa en vez de empezar de nuevo. El `ENTER` **no la retoma**, a
  proposito: solo pregunta, las veces que se lo apriete, para que dos `ENTER` seguidos no
  abran la orden de ayer. Contesta la `r` --y solo en un campo de numero, porque en un campo
  de texto la `r` es una letra y hay clientes que se llaman Rodriguez-- o el boton, que
  aparece solo cuando hay algo que retomar.
- **Si con ese numero ya hay una orden CERRADA, el programa NO la deja crear** y avisa
  `ya hay una orden guardada en <archivo>.json; usa otro numero`
  (`src/afilado/cli/run_live.py:816`). No hay nada que retomar --una orden cerrada ya entrego
  su informe-- y escribirle encima borraria la unica copia del trabajo de ese dia. Ahi hay que
  usar **otro numero**: la sierra que vuelve del cliente es otra pasada por el taller, no la
  misma orden. **Se nombra el archivo y no el numero tecleado** porque dos numeros distintos
  pueden caer en el mismo archivo: `OA 123` y `OA/123` dan los dos `OA_123.json`.
- **Los datos se cargan a mano**: numero de orden, nota de pedido, cliente, cuantas sierras
  y cuantos dientes tiene cada una. La idea es que vengan de la app de afilados, pero eso
  **todavia no esta conectado**: conectarlo a medias seria peor que nada, porque un numero
  de orden equivocado en un informe de reclamo es peor que no tener informe. El dia que se
  conecte, lo unico que cambia es de donde salen los mismos campos.
- **Cada diente que archivas queda anotado** y el contador avanza solo: arriba de la
  pantalla del taller se ve siempre `sierra 1 de 3 - diente 15 de 24`. Lo que vale como
  estado es **donde lo archivaste vos**; lo que opinaba el programa queda anotado al lado.
- **Un diente que no se pudo mirar se saltea** con la `x` --boton **SALTEAR DIENTE**-- y
  figura como tal. Que falte uno es un dato, no un hueco.
- **GUARDAR INFORME** (`i`) deja dos archivos en `data/ordenes/`: `informe_5113546.txt` para
  imprimir o mandar, y `informe_5113546.csv` para abrir en Excel, con **una fila por
  diente**. Al cerrar la orden se escriben solos. **Los dos se escriben por separado y la
  pantalla dice que mitad salio**: si falta el `.txt` nombra la planilla, si falta el `.csv`
  nombra el informe, y si no salio ninguno dice que hay que revisar el disco.
- Se guarda **despues de cada diente**, no al final: un corte de luz a mitad de una sierra
  no puede borrar dos horas de trabajo. Cuesta poco: medido, **0,8 ms** sin fsync y **1,2 ms**
  con los datos forzados al disco (`src/afilado/ordenes.py:1239`). Y si cerras el programa,
  al abrirlo retoma la orden donde estaba.

#### Las dos columnas de foto de la planilla, que no son lo mismo

En el `.csv` hay **dos** columnas de foto por diente, y confundirlas seria hacer pasar una
por la otra:

| Columna | Que es | Sirve de prueba |
|---|---|---|
| `foto_de_entrenamiento` | El nombre **pelado** de la foto que entro a la coleccion, sin carpeta | **No.** Ni el programa la encuentra, y se puede haber borrado al borrar el concepto |
| `prueba` | La ruta de la foto que respalda el veredicto de ese diente | **Si.** Es la que viaja en el zip del reclamo |

El informe `.txt` avisa con todas las letras que la primera **no** es la prueba. Las ordenes
viejas solo tienen la primera, y el historial las cuenta **aparte** justamente para no
hacerlas pasar por prueba: un diente asi sale como `7 nombres de foto sueltos, no son prueba`.

#### Que archivos deja UNA orden en el disco

Seis cosas, todas abajo de `data/`:

| Donde queda | Que es |
|---|---|
| `data/ordenes/<numero>.json` | **La orden.** Es el documento que se copia tal cual al zip |
| `data/ordenes/informe_<numero>.txt` | El informe para imprimir o mandar |
| `data/ordenes/informe_<numero>.csv` | La planilla, una fila por diente |
| `data/ordenes/.abierta` | El marcador de cual orden retomar al abrir el programa |
| `data/pruebas/<nombre del .json>/` | Las fotos que respaldan cada diente |
| `data/ordenes/_papelera/<fecha y hora>/` | Lo que se borro |
| `data/ordenes/exportado/orden_<nombre del .json>_<fecha y hora>.zip` | Los paquetes de reclamo |

> **EL NOMBRE DEL ZIP SALE DEL ARCHIVO, NO DEL NUMERO DE ADENTRO**, igual que la carpeta de
> pruebas: se arma con el nombre del `.json` tal cual (`src/afilado/ordenes.py:1692`). O sea
> que exportar `123 - copia.json` no da `orden_123_...` sino `orden_123 - copia_...`, aunque
> adentro el numero diga 123. Es la misma regla de la carpeta de pruebas, y por la misma
> razon: lo que el historial leyo es el **archivo**.

> **EL `procedencia.txt` NO LO DEJA EL PROGRAMA.** Es un papel que se escribe **a mano**,
> cuando las fotos de una carpeta de pruebas **no** se sacaron en el momento de juzgar el
> diente: es el caso de las 14 fotos rescatadas de la papelera de la coleccion para las
> ordenes 123 y 1234. Lo unico que el programa hace con ese papel es **llevarlo en el zip si
> esta** (`ordenes.py:1789`) y **no contarlo** entre las fotos que ningun diente nombra
> (`ordenes.py:1340`). Buscarlo en la carpeta de una orden nueva es buscar algo que nadie
> escribio: una orden recien revisada deja su carpeta de pruebas **sin** `procedencia.txt`.
> Sirve para que el que recibe el reclamo sepa de donde salio cada foto, asi que si se
> rescatan fotos a mano, **el papel lo tiene que escribir quien las rescata**.

Afuera de `data/` queda `registro.txt`, al lado del programa: 1 MB con dos vueltas. **Un
`.tmp` que aparezca es un guardado que se corto**; se puede borrar.

> **La carpeta de las fotos lleva el nombre del ARCHIVO `.json`, no el numero que figura
> adentro.** Por eso una orden copiada a mano con el Explorador no se lleva --ni promete-- las
> fotos de la original: `123 - copia.json` tiene el numero 123 adentro, pero su carpeta seria
> `data/pruebas/123 - copia`. Y a una orden cuyo archivo no se llama como su numero el
> programa **no la deja retomar**: se la puede exportar o borrar, nada mas.

> **Una foto cuenta como prueba de un diente solo si esta ADENTRO de la carpeta de pruebas de
> esa orden.** Una ruta que apunta a otra parte --aunque la foto exista-- no es prueba de esa
> orden: se cuenta aparte como **prueba perdida**, sale en rojo en el historial
> (`N PRUEBAS QUE NO ESTAN`) y **no viaja en el reclamo**. Mostrar como prueba algo que no lo
> es seria exactamente lo que este programa no hace.
>
> Y "prueba perdida" **no** quiere decir que la foto no exista: hay que ir a buscarla a la
> papelera de la coleccion, al respaldo, o a la PC de donde se copio la carpeta. Paso de
> verdad: las 14 fotos de las ordenes 123 y 1234 estaban en la papelera de la coleccion y de
> ahi se las rescato el 16/11.

> **AVISO PARA LA PC DEL TALLER.** Una orden que ya tiene fotos de prueba guardadas **no la
> abren los `.exe` de respaldo** que estan al lado de los datos: son anteriores al campo
> `prueba` y una clave que no conocen les hace ilegible la **orden entera**, no el campo.
> Medido hoy contra los `.json` del taller: **2 de las 4 ordenes** (la 123 y la 1234, con 7
> dientes con prueba cada una) ya no se abren con esos `.exe`; la 213132 y la 5113546 si. El
> programa lo avisa en el historial. Una orden **sin** pruebas queda con la misma forma que
> antes y esos `.exe` la siguen abriendo, porque el campo vacio no se escribe.

#### Con la orden abierta, los dientes se anotan solos

Pararse a apretar una tecla por diente son veinticuatro pulsaciones por sierra, y obliga a
soltar la pieza. Con una orden abierta el programa **mira cinco fotos por segundo** --una cada
0,2 s-- y cuando **cinco seguidas son la misma foto con el mismo veredicto**, da ese diente
por revisado, lo anota en la orden y pasa al siguiente. El operario mueve la sierra y no
toca una tecla.

> **HAY QUE APOYAR CADA DIENTE 1,2 SEGUNDOS.** El minimo teorico seria un segundo, pero esta
> medido que un segundo **no alcanza**: con el bucle de verdad y las 15 fotos del taller del
> 16/11, a 1,0 s anota **8 de 15** dientes y a 1,2 s anota **14 de 15** (a 0,6 s, ninguno; a
> 2,0 s, los mismos 14). Y con la mano temblando hay que apoyar **mas tiempo**, no apretar mas
> fuerte: con 2 px de temblor la racha cierra el 44% de las veces y a 1,2 s anota 7 de 15.
> Medido y escrito en `src/afilado/config.py:174` y `config.py:196`.
>
> Arrancar en 1,0 s entre fotos era inusable: eran cinco segundos de sierra quieta por diente,
> casi siete minutos de inmovilidad en una sierra de 80.

Arriba, al lado de la orden, se ve en que esta: `mirando`, `confirmando: 3 de 5`, o
`diente anotado - pase al siguiente`.

**Por que cinco y no una.** Una sola foto puede agarrar el momento en que la pieza se esta
acomodando, el reflejo de una lampara o la mano cruzando. Cinco seguidas que dan lo mismo
quieren decir dos cosas a la vez: que la pieza esta **quieta** y que el veredicto es
**estable**. Cualquiera de las dos sola no alcanza.

> **Despues de anotar hay que mover la pieza.** Es lo que evita el peor error posible: que
> una sierra olvidada sobre la mesa se anote veinticuatro veces sola y el informe diga que
> se reviso entera sin que nadie la haya tocado. El corte de "la imagen cambio" es el mismo
> que ya usa el programa para no guardar dos veces la misma foto, y esta medido: archivo
> identico 0.0000, el mismo frame con otra exposicion 0.0063-0.0074, tomas realmente
> distintas 0.0315-0.3296.

#### De cada diente anotado QUEDA UNA FOTO

**La foto que el programa juzgo se guarda como prueba.** Va a
`data/pruebas/<nombre del .json de la orden>/`, con la sierra y el diente en el nombre
--`s01_d007_20261116_093432_481.jpg`-- a calidad 95, y la orden anota esa ruta en el campo
`prueba` de ese diente. Es la foto que viaja en el zip del reclamo.

Pasa por los **tres** caminos que anotan un diente: la captura automatica, la tecla `g` y las
teclas `a`/`r`.

> **OJO CON LAS DOS CARPETAS QUE YA EXISTEN EN EL TALLER.** Las unicas dos ordenes que hoy
> tienen prueba guardada --la **123** y la **1234**-- tienen las fotos con el **nombre viejo de
> la coleccion**, no con el `s01_d007_...` de arriba: medido leyendo
> `data/pruebas/123` y `data/pruebas/1234`, son **7 fotos cada una**, todas del tipo
> `afilado_20261112_161511_098.jpg`, mas el `procedencia.txt` escrito a mano. Es porque esas
> fotos no se sacaron al juzgar el diente: se rescataron despues de la papelera de la
> coleccion. El que busque un nombre con sierra y diente en esas dos carpetas **no lo va a
> encontrar**, y no es que falte la foto: es que tiene el otro nombre. El `.json` de la orden
> dice cual foto es de cual diente.

**Lo que si se tira** son los cuadros que la captura mira y **no** anota: cinco por segundo.
El que se guarda es **el que cierra la racha**.

> **Esa foto NO entra a la coleccion de entrenamiento.** Juntar fotos para que el programa
> aprenda sigue siendo a mano con las teclas, eligiendo en que concepto va cada una. El
> deposito de pruebas esta afuera de la coleccion justamente por eso: si la prueba entrara,
> archivar pasaria de 209 ms con 15 fotos a 3.153 ms con 320 --y 320 fotos es UNA orden de 4
> sierras de 80 dientes-- contra los 1.200 ms que el operario apoya cada diente
> (`src/afilado/ordenes.py:1110`).

**Lo que cuesta y lo que ocupa, medido en esta PC** con una foto de verdad del microscopio de
640x480 y 80 dientes seguidos:

| Que | Cuanto |
|---|---|
| Anotar un diente con su foto de prueba | mediana **4,98 ms**, p95 7,18 ms (sin prueba, 2,27 ms: la prueba agrega 2,7) |
| Eso contra los 1.200 ms que se apoya el diente | **0,4 %** del tiempo |
| Las fotos de una sierra de 80 dientes | **7,32 MiB** (el `.json` queda en 30.770 bytes) |
| Una orden de 4 sierras de 80 | **29,3 MiB** |
| Lo que entra con los 116,7 GiB libres del C: | mas de **4.000** ordenes asi |

Escrito en `src/afilado/cli/run_live.py:1805` y `run_live.py:1812`.

La captura automatica se apaga con `interfaz.segundos_entre_fotos: 0` --y ahi los dientes se
anotan a mano con la `g`-- y se ajusta con `interfaz.fotos_iguales_para_anotar`. Las dos
llaves estan escritas en `configs/config.yaml` abajo de `interfaz:`, con sus valores de hoy y
la medicion al lado: `segundos_entre_fotos: 0.2` (`config.yaml:201`) y
`fotos_iguales_para_anotar: 5` (`config.yaml:207`).

> **EL DIA QUE APAREZCA UN MODELO ENTRENADO, ESTO DEJA DE ANOTAR SOLO.** Conviene saberlo de
> antemano para no buscarlo como si fuera una rotura nueva. Hoy el veredicto de cada diente
> --y la cuenta de la racha-- se llenan **unicamente** en la rama que corre cuando **no** hay
> clasificador (`src/afilado/cli/run_live.py:2182`). Con un `.pt` o un `.onnx` en `models/`
> esa rama no corre, `_resultado_del_diente` no encuentra dientes y devuelve None, y la
> **captura automatica no confirma nunca**: los dientes hay que anotarlos a mano con la `g`.
> Esta escrito al lado del codigo, en `run_live.py:2171` a `run_live.py:2180`, con las dos
> cosas que faltan para que ande con modelo. Hasta que eso se haga, **copiar el modelo
> entrenado al taller apaga el anotado solo**.

#### La barra de dientes, abajo del veredicto

Debajo del cartel de APROBADO/DESAPROBADO aparece **un cuadrito por diente de la sierra**:
verde el que paso, rojo el que no, ambar el que no se pudo decidir y gris el que falta. De
un vistazo se ve como viene la sierra y cuanto queda, sin soltar la pieza ni abrir el panel.

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

Es la pantalla pensada para el **microscopio USB**. Se activa poniendo `interfaz.modo:
revision` en el config. **NO es como viene:** de fabrica el config trae `modo: simple`
(`configs/config.yaml:128`), que es la pantalla de la seccion 5.1; y si no hay config, el
programa arranca en `clasico` (`src/afilado/config.py:100`). Para usar esta pantalla hay que
cambiar esa llave a mano.

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
| `espacio` o `ENTER` | **REVISAR** | Congela lo que se ve y lo analiza. |
| `espacio` o `ENTER` | **OTRA PIEZA** | Descongela y vuelve al vivo, para la siguiente. |
| `a` | **GUARDAR \<primer concepto sano\>** | Archiva la foto congelada en el **primer concepto sano**. Hoy, `Sierra_Nueva_TP`. |
| `r` | **GUARDAR \<primer concepto de defecto\>** | Archiva la foto congelada en el **primer concepto de defecto**. Hoy, `Dientes_rotos`. |
| `f` | — | **Agrega** lo que se ve como una foto de modelo nueva y la deja elegida. No pisa ninguna. |
| `q` o `ESC` | **SALIR** | Cierra el programa. |

**Los dos botones de guardar siguen a los conceptos que hay**, igual que en la pantalla
simple: con los conceptos de hoy dicen **GUARDAR SIERRA_NUEVA_TP** y **GUARDAR DIENTES_ROTOS**
(medido corriendo `_destinos_rapidos()` con la coleccion del taller;
`src/afilado/revision.py`, dentro de `componer()`, en la rama `if estado.congelado:`). El
orden sale de `conceptos.json`, no del alfabeto: la `a` va al **primer** concepto sano de esa
lista y la `r` al **primero de defecto**, que es lo mismo que hacen las teclas en el modo
simple (`src/afilado/cli/run_live.py:273`).

> Antes esos dos botones estaban clavados en las claves `aprobada` y `rechazada`, y con los
> conceptos de verdad del taller **no hacian nada**: se apretaban y la pantalla no decia por
> que. Ya esta arreglado. Un boton que no hace lo que dice es lo mismo que prometer un
> veredicto que no se va a dar.

**El renglon de ayuda de esta pantalla es fijo y dice menos de lo que hay** (`revision.py:388`):

```
  espacio o clic = REVISAR   |   a aprobada   |   r rechazada   |   f fijar referencia   |   q salir
```

Dos cosas para tener en cuenta leyendolo: el **`ENTER` tambien congela y descongela**, igual
que la barra espaciadora --el bucle es el mismo para los dos modos, `run_live.py:2325`-- y
`a aprobada` / `r rechazada` son los nombres viejos: lo que esas teclas archivan es el primer
concepto sano y el primero de defecto, o sea los mismos destinos que dicen los botones. Si hay
fotos guardadas en esta sesion, al renglon se le suma `   |   guardadas: N`
(`revision.py:389`).

**Y hay teclas que aqui andan pero no tienen boton ni estan escritas en la pantalla**, porque
las resuelve el mismo bucle que el modo simple: `c` conceptos, `o` historial de ordenes, `g`
anotar diente y `m` otra foto de modelo. Si hacen falta todos los dias, conviene el modo
simple, que si las muestra.

> **CON UNA ORDEN ABIERTA, ESTA PANTALLA NO MUESTRA LA ORDEN NI EL AVANCE.** El resumen de la
> orden, el `confirmando: 3 de 5` y la barra de dientes los dibuja **solo** `componer_simple`;
> `componer()` --la pantalla de este modo, `src/afilado/revision.py:278` a `revision.py:412`--
> no dibuja ninguno de los tres. O sea que aca la `g` y la captura automatica **anotan a
> ciegas**: el diente se anota igual, pero no se ve `sierra 1 de 3 - diente 15 de 24` ni en
> cual va. Si se trabaja con ordenes todos los dias, la pantalla es la del modo **simple**.

Casi todo tiene tecla **y** boton --si en alguna maquina el mouse no responde bien dentro de
la ventana de OpenCV, el teclado sigue funcionando-- pero **no todo**, y conviene saber las
excepciones:

| Que | Como se hace |
|---|---|
| **CERRAR ORDEN** | Solo con el **mouse**: no tiene tecla |
| La tecla **`f`** (agregar foto de modelo) | Solo con el **teclado**: no tiene boton en ninguna pantalla |
| **`w`** / **`s`** (moverse por la lista de conceptos y por el historial) | Solo con el **teclado**: no hay boton ni en CONCEPTOS (`src/afilado/panel.py:469`) ni en el historial (`panel.py:1093`) |
| Pasar de pagina en el historial y en VER FOTOS | Solo con la **barra espaciadora**, salvo el boton SIGUIENTES de VER FOTOS |

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

**Lo que se valido con estas fotos reales:**
- El **filtro de nitidez** descarto solo las tomas movidas (de 51 fotos de la rechazada, ~35
  borrosas fuera; de 15 de la aprobada, 9 fuera). Funciona sobre fotos de taller, no de laboratorio.
- El **camino de clasificacion** (`construir_clasificacion` → `train`) corre de punta a punta y
  produce el `best.pt`. Quedaron 18 fotos de entrenamiento (6 aprobada + curadas de 12 rechazada).

**Lo que todavia NO funciona (y es esperado):**
- El modelo entrenado con ~22 fotos es **inservible**: da 100% en 4 fotos de validacion porque
  las memoriza, no porque haya aprendido. No es cautela: **faltan cientos de fotos por clase**.
  Por eso el `best.pt` de prueba se descarto y el sistema sigue en modo geometrico hasta juntar
  datos de verdad. Ver la seccion 7.
- **El veredicto en vivo YA esta conectado** (`run_live` en modo `classify` muestra la banda
  aprobada/rechazada y archiva fotos por clase con `a`/`r`). Verificado con las sierras de
  referencia. Lo unico que falta para que sirva de verdad es el modelo entrenado con suficientes
  fotos: hasta entonces arranca en modo recoleccion. Ver la seccion 5.1.

## 6. EL CICLO DE APRENDIZAJE (la parte mas importante)

El sistema no se entrena una vez y listo. **Es un circulo que se repite y cada vuelta lo
deja mejor.**

> **OJO: ESTE CICLO ES DEL MODO DETECCION, NO DE LA PANTALLA QUE CORRE HOY EN EL TALLER.**
> El dibujo de abajo habla de cajas, de Roboflow y de la tecla `e`, y eso es la seccion
> 5.2 (`tarea: detect` o `segment`). El taller corre `tarea: classify` con
> `interfaz.modo: simple`, y **ahi la tecla `e` no hace nada**: no esta ligada a ninguna
> accion y el programa no avisa nada cuando se la aprieta. Peor: con el panel de ORDENES
> abierto, la `e` **exporta un zip de reclamo** de la orden elegida.
>
> **En la pantalla de hoy el error se corrige archivando la foto**: `a` o `r` la mandan al
> primer concepto sano o al primero de defecto, y con `c` se entra a CONCEPTOS para elegir
> cualquier otro. Eso entra en la memoria **en el acto**, sin fin de semana, sin Roboflow y
> sin reentrenar nada: el diente siguiente ya se compara contra esa foto. Ver "La memoria"
> en la seccion 5.1.
>
> El ciclo de abajo queda escrito porque es a donde se va el dia que haya un modelo
> entrenado y se midan dientes, no porque sea lo que se hace hoy.

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
**Solo en el modo deteccion.** En la pantalla del taller la `e` no esta ligada a nada; ahi
el paso 2 es archivar la foto con `a`, con `r` o desde CONCEPTOS.

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
revisan las cajas pre-dibujadas, se corrigen las que estan mal, se ajusta la clase --la del
config, que hoy es `aprobada` / `rechazada`, o el concepto que corresponda-- y se exporta.

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

| Que | Cantidad |
|---|---|
| Por concepto | **Cientos.** 200-300 como piso, 500+ para andar bien. |
| Total para un primer modelo usable | ~1.000 a 1.500 imagenes |
| Total para un modelo solido | ~3.000+ |

Son los conceptos que el taller tenga en `data/coleccion`, no una lista fija de defectos: hoy
`Dientes_rotos`, `Sierra_Nueva_TP`, `Diente_sin_placa` y `Sierra_Afilada`.

Reglas para que las fotos sirvan:

- **Todos los conceptos representados.** Si hay 900 fotos de uno y 30 de otro, el modelo va a
  contestar siempre el primero, porque acierta casi siempre haciendo eso. Hay que buscar
  activamente las piezas feas.
- **Misma luz y mismo angulo que la version final.** Las fotos de entrenamiento tienen que
  parecerse a lo que la camara va a ver todos los dias. Si se cambia la iluminacion o la
  altura de la camara despues de entrenar, hay que volver a juntar fotos.
- **Variedad dentro de cada concepto.** Distintas piezas, distintas rotaciones, distintas
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
│   ├── construir_clasificacion.py <- arma el dataset desde data/coleccion/
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
│   ├── ordenes.py            <- la orden, el informe, el zip y la papelera
│   ├── panel.py              <- las pantallas: conceptos, historial, orden
│   ├── revision.py           <- la pantalla del taller y sus botones
│   ├── memoria.py            <- la memoria: compara, juzga y se mide a si misma
│   ├── conceptos.py          <- las carpetas de la coleccion y cual es defecto
│   ├── dientes.py            <- separa los dientes de una foto
│   ├── automatico.py         <- la captura automatica con la orden abierta
│   ├── referencias.py        <- la galeria de fotos de modelo
│   ├── tipografia.py         <- mide y dibuja los textos de pantalla
│   ├── tipos.py              <- los tipos que comparten los modulos
│   └── cli/
│       ├── run_live.py       <- el programa del taller
│       ├── train.py          <- entrenamiento
│       └── prepare_dataset.py<- preparacion del dataset
├── tests/                    <- pruebas automaticas del codigo
├── registro.txt              <- el registro del programa (1 MB, con dos vueltas)
└── data/                     <- todo lo que se genera trabajando
    ├── ordenes/              <- LAS ORDENES DE AFILADO
    │   ├── 5113546.json      <- la orden: es lo que se copia tal cual al zip
    │   ├── informe_5113546.txt  <- el informe para imprimir o mandar
    │   ├── informe_5113546.csv  <- la planilla, una fila por diente
    │   ├── .abierta          <- cual orden retomar al abrir el programa
    │   ├── _papelera/        <- lo borrado, una carpeta por fecha y hora
    │   │   └── 20261116_121100/
    │   └── exportado/        <- los zip de reclamo (NADIE los borra)
    │       └── orden_5113546_20261116_121100.zip
    ├── pruebas/              <- LAS FOTOS QUE RESPALDAN CADA DIENTE
    │   └── 5113546/          <- se llama como el .json, no como el numero de adentro
    │       ├── s01_d007_20261116_093432_481.jpg
    │       └── procedencia.txt  <- NO lo deja el programa: se escribe a mano cuando
    │                               las fotos no se sacaron al juzgar el diente
    ├── coleccion/            <- LA COLECCION DE ENTRENAMIENTO (los conceptos)
    │   ├── Sierra_Afilada/   <- un concepto = una carpeta de fotos
    │   ├── _papelera/        <- las fotos y conceptos borrados
    │   └── conceptos.json    <- cual cuenta como defecto y en que orden se muestran
    ├── memoria/              <- la cache de la memoria y su acierto medido
    ├── referencia/
    │   └── microscopio/      <- la galeria de fotos de modelo (tecla f y m)
    ├── clasificacion/        <- el dataset que arma construir_clasificacion.py
    │   ├── train/<concepto>/  <- (se puede borrar: se rehace desde coleccion/)
    │   └── val/<concepto>/
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

**Las tres carpetas de `data/` que el taller toca todos los dias** son `ordenes/`,
`pruebas/` y `coleccion/`. Las otras se rehacen solas o las arma un script. Dos avisos que ya
estan mas arriba y conviene tener a mano aca:

- **`data/ordenes/exportado/` crece sin tope y nadie la limpia**, ni cuando se borra la orden.
  Se vacia a mano con el Explorador.
- **`data/pruebas/<carpeta>/` se llama como el ARCHIVO `.json`**, no como el numero que
  figura adentro. Una orden copiada a mano con el Explorador no se lleva las fotos de la
  original.

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
| **El modelo solo sabe lo que se le enseno** | Si aparece un tipo de herramienta que nunca vio, va a decir cualquier cosa. La respuesta es archivar esa foto en el concepto que corresponda (`a`, `r`, o CONCEPTOS con la `c`): la memoria la toma en el acto. En el modo deteccion, el ciclo de la seccion 6. |

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

**El cartel dice FALTAN FOTOS.**
La memoria **si** reconoce lo que ve, pero ese concepto todavia no tiene las 3 fotos que
hacen falta para poder ser un veredicto. El cartel dice **cual** concepto y **cuantas** fotos
le faltan. Hay que sacarle una foto mas a ESE concepto y archivarla ahi con la tecla `c`: es
una tarea de un minuto y el panel de CONCEPTOS tambien lo muestra, fila por fila. No es una
falla.

**El cartel dice NO ESTOY SEGURO.**
Dos cosas distintas, y el motivo de abajo dice cual: o la foto no se parece a nada de lo
guardado --ahi hay que mirar el diente a ojo y, si es un caso nuevo, archivarlo donde va-- o
reconoce dos conceptos parejos y no se anima. Ver "Los TRES carteles de no decidio".

**No encuentro las ordenes. Como abro una orden de la semana pasada.**
Tecla `o` o boton **ORDENES**: eso abre el **historial**, con todas las ordenes guardadas y
la mas nueva arriba. Se elige la fila con el digito o el mouse y se entra con `ENTER`. Para
cargar una nueva, la `n`. Una orden ya **cerrada** no se reabre --su informe ya se entrego--
pero se puede exportar con la `e`.

**No veo el boton ORDENES en la pantalla.**
Con nombres de concepto largos y la pantalla congelada no entra en la barra y no se dibuja.
La tecla `o` sigue andando y queda **escrita arriba a la izquierda**: `teclas:  o ordenes`.
Con la barra espaciadora se vuelve al vivo, donde el boton si entra.

**Donde quedo la foto que respalda un diente.**
En `data/pruebas/<nombre del .json de la orden>/`, con la sierra y el diente en el nombre:
`s01_d007_20261116_093432_481.jpg`. Es la que viaja en el zip del reclamo (tecla `e` del
historial). Ojo con las ordenes **123** y **1234**, las dos unicas que hoy tienen prueba: sus
fotos tienen el nombre viejo de la coleccion (`afilado_20261112_161511_098.jpg`), no el de
sierra y diente.

**El historial dice PRUEBAS QUE NO ESTAN en rojo.**
Quiere decir que la foto **no esta donde la orden la nombra**, y nada mas que eso: el programa
pregunto por esa ruta y no la encontro, **no fue a buscarla a ninguna otra parte**
(`src/afilado/ordenes.py:2097`). Asi que **no dice si la foto existe o no**. Hay que ir a
buscarla a la **papelera de la coleccion** (`data/coleccion/_papelera/`), al respaldo, o a la
PC de donde se copio la carpeta. Si no esta en ninguna de las tres, esa prueba se perdio y el
informe lo va a decir con un `(NO ESTA)` al lado del diente. Paso de verdad y se recupero: las
14 fotos de las ordenes 123 y 1234 estaban en la papelera de la coleccion.

**Borre una orden por error.**
No se borro: esta en `data/ordenes/_papelera/<fecha y hora>/`. Se saca con el Explorador,
copiando el `.json` de vuelta a `data/ordenes` y la carpeta de fotos a `data/pruebas`. **Pero
ahi va lo que habia**: el `.json` siempre, los informes que estuvieran al lado, y la carpeta
de fotos **solo si esa orden tenia prueba guardada**. Si la fila decia **SIN PRUEBA GUARDADA**
o **PRUEBAS QUE NO ESTAN**, en la papelera **no hay ninguna foto** que rescatar: hay que
buscarlas donde dice la respuesta de arriba. Y si el borrado quedo a medias, apretar `b` otra
vez sobre la misma fila **completa esa misma carpeta de papelera**, no abre otra.

**La captura automatica no anota los dientes.**
Hay que **apoyar cada diente 1,2 segundos**. Esta medido: con un segundo justo anota 8 de 15
y con 1,2 s anota 14 de 15. Y si la mano tiembla hay que apoyar **mas tiempo**, no apretar
mas fuerte. Despues de anotar hay que **mover la sierra**: la misma imagen no se anota dos
veces.

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
(`--conf 0.9`). Y sobre todo: **archivar esa foto en el concepto bueno** --`a`, o `c` y
GUARDAR ACA-- cada vez que pasa. Eso entra en la memoria en el acto y el diente siguiente
ya se compara contra ella. (La tecla `e` del ciclo de la seccion 6 no hace nada en esta
pantalla, y con el historial de ordenes abierto exporta un zip.)

**Se movio la camara.**
Apretar `r` para reiniciar la calibracion. Si el cambio fue grande (altura o angulo),
verificar los 90 grados y controlar unas piezas contra el calibre antes de confiar.

---

*Consultas: Valentin y Leandro, taller.*
