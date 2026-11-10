#Requires -Version 5.1
<#
.SINOPSIS
    Prepara el entorno de desarrollo de Analisis-afilado.

.DESCRIPCION
    Verifica la version de Python (exige 3.10 o superior), crea el entorno virtual .venv,
    actualiza pip, instala las dependencias, registra src/ en el entorno, detecta si hay
    GPU NVIDIA y sugiere el comando de torch con CUDA correspondiente.

    Tiene dos modos. El COMPLETO (por defecto) es para la maquina donde se entrena:
    instala ultralytics y torch, ~2.5 GB en disco. El modo TALLER (-Taller) es para la
    maquina que solo mira sierras y da el veredicto: instala unicamente lo necesario para
    correr el modelo ya exportado a .onnx, sin torch ni ultralytics. Medido: 244 MB de
    entorno contra 1.2 GB, y el mismo veredicto.

    Es idempotente: se puede correr las veces que haga falta. No borra nada sin avisar;
    si .venv ya existe lo reutiliza (para rehacerlo desde cero, usa -Recrear).

.PARAMETER Python
    Ejecutable de Python a usar. Si se omite, busca uno compatible con el lanzador "py".

.PARAMETER Recrear
    Borra el .venv existente y lo crea de nuevo. Pide confirmacion.

.PARAMETER Taller
    Instalacion liviana para la PC del taller: solo lo necesario para CORRER el sistema.
    No instala ultralytics ni torch, asi que esa maquina no puede entrenar ni exportar;
    levanta el modelo .onnx que se le copia desde la maquina de entrenamiento.

.EJEMPLO
    .\scripts\setup.ps1
    .\scripts\setup.ps1 -Taller
    .\scripts\setup.ps1 -Python "C:\Python312\python.exe"
    .\scripts\setup.ps1 -Recrear
#>

[CmdletBinding()]
param(
    [string]$Python = "",
    [switch]$Recrear,
    [switch]$Taller
)

$ErrorActionPreference = "Stop"

$RaizRepo = Split-Path -Parent $PSScriptRoot
$RutaVenv = Join-Path $RaizRepo ".venv"
$RutaRequirements = Join-Path $RaizRepo "requirements.txt"

function Escribir-Titulo([string]$Texto) {
    Write-Host ""
    Write-Host "=== $Texto ===" -ForegroundColor Cyan
}

function Escribir-Ok([string]$Texto)    { Write-Host "  [OK] $Texto" -ForegroundColor Green }
function Escribir-Aviso([string]$Texto) { Write-Host "  [!]  $Texto" -ForegroundColor Yellow }
function Escribir-Error([string]$Texto) { Write-Host "  [X]  $Texto" -ForegroundColor Red }

# --- Version de Python -------------------------------------------------------
# Minimo 3.10 por la sintaxis de tipos que usa el paquete (X | None).
# No hay techo de version: verificado el 2026-07-17 que torch 2.13 publica wheels cp314,
# asi que 3.14 funciona. Si en el futuro pip no encuentra torch para una version muy
# nueva, el aviso de mas abajo explica como salir del paso con un Python anterior.
$VersionMinima = [Version]"3.10"

function Obtener-VersionPython([string]$Ejecutable) {
    try {
        $salida = & $Ejecutable -c "import sys; print('%d.%d' % sys.version_info[:2])" 2>$null
        if ($LASTEXITCODE -ne 0 -or [string]::IsNullOrWhiteSpace($salida)) { return $null }
        return [Version]$salida.Trim()
    } catch {
        return $null
    }
}

function Buscar-PythonCompatible {
    # Primero el "python" del PATH: si ya sirve, es el que el usuario espera usar.
    foreach ($nombre in @("python", "python3")) {
        $comando = Get-Command $nombre -ErrorAction SilentlyContinue
        if ($null -eq $comando) { continue }
        $version = Obtener-VersionPython $comando.Source
        if ($null -ne $version -and $version -ge $VersionMinima) {
            return $comando.Source
        }
    }
    # Si el del PATH no sirve (o no hay), probamos el lanzador "py", de mas nuevo a mas viejo.
    foreach ($etiqueta in @("3.14", "3.13", "3.12", "3.11", "3.10")) {
        try {
            $ruta = & py "-$etiqueta" -c "import sys; print(sys.executable)" 2>$null
            if ($LASTEXITCODE -eq 0 -and -not [string]::IsNullOrWhiteSpace($ruta)) {
                return $ruta.Trim()
            }
        } catch {
            # el lanzador py no esta instalado o no tiene esa version; seguimos probando
        }
    }
    return $null
}

Escribir-Titulo "Verificando Python"

if ([string]::IsNullOrWhiteSpace($Python)) {
    $Python = Buscar-PythonCompatible
    if ($null -eq $Python) {
        Escribir-Error "No se encontro ningun Python $VersionMinima o superior."
        $comandoPython = Get-Command python -ErrorAction SilentlyContinue
        if ($null -ne $comandoPython) {
            $detectada = Obtener-VersionPython $comandoPython.Source
            if ($null -ne $detectada) {
                Write-Host "       El 'python' de tu PATH es $detectada ($($comandoPython.Source))." -ForegroundColor Red
            }
        }
        Write-Host ""
        Write-Host "  SOLUCION: instala Python desde https://www.python.org/downloads/" -ForegroundColor Yellow
        Write-Host "    1. Descarga 'Windows installer (64-bit)'" -ForegroundColor Yellow
        Write-Host "    2. Marca 'Add python.exe to PATH' y 'py launcher' durante la instalacion" -ForegroundColor Yellow
        Write-Host "    3. Volve a correr este script" -ForegroundColor Yellow
        Write-Host ""
        Write-Host "  Si ya tenes un Python en una ruta rara, pasalo a mano:" -ForegroundColor Yellow
        Write-Host "    .\scripts\setup.ps1 -Python 'C:\Python314\python.exe'" -ForegroundColor Yellow
        exit 1
    }
}

if (-not (Test-Path $Python)) {
    $comandoDado = Get-Command $Python -ErrorAction SilentlyContinue
    if ($null -eq $comandoDado) {
        Escribir-Error "No existe el ejecutable de Python indicado: $Python"
        exit 1
    }
    $Python = $comandoDado.Source
}

$VersionPython = Obtener-VersionPython $Python
if ($null -eq $VersionPython) {
    Escribir-Error "No se pudo determinar la version de: $Python"
    exit 1
}
if ($VersionPython -lt $VersionMinima) {
    Escribir-Error "Python $VersionPython es demasiado viejo. Se requiere $VersionMinima o superior."
    exit 1
}
Escribir-Ok "Python $VersionPython -> $Python"

# --- Entorno virtual ---------------------------------------------------------
Escribir-Titulo "Entorno virtual"

if ((Test-Path $RutaVenv) -and $Recrear) {
    Write-Host "  Se va a BORRAR el entorno existente: $RutaVenv" -ForegroundColor Yellow
    $respuesta = Read-Host "  Confirmas? (s/N)"
    if ($respuesta -notmatch '^[sSyY]$') {
        Escribir-Aviso "Cancelado por el usuario. No se borro nada."
        exit 1
    }
    Remove-Item -Recurse -Force $RutaVenv -Confirm:$false
    Escribir-Ok "Entorno anterior borrado"
}

if (Test-Path $RutaVenv) {
    Escribir-Ok "Ya existe .venv, se reutiliza (para rehacerlo: -Recrear)"
} else {
    & $Python -m venv $RutaVenv
    if ($LASTEXITCODE -ne 0) {
        Escribir-Error "Fallo la creacion del entorno virtual."
        exit 1
    }
    Escribir-Ok "Entorno virtual creado en .venv"
}

$PythonVenv = Join-Path $RutaVenv "Scripts\python.exe"
if (-not (Test-Path $PythonVenv)) {
    Escribir-Error "El .venv existe pero no tiene Scripts\python.exe. Esta corrupto."
    Write-Host "  Rehacelo con: .\scripts\setup.ps1 -Recrear" -ForegroundColor Red
    exit 1
}

$VersionVenv = Obtener-VersionPython $PythonVenv
if ($null -ne $VersionVenv -and $VersionVenv -lt $VersionMinima) {
    Escribir-Error "El .venv existente usa Python $VersionVenv, por debajo del minimo $VersionMinima."
    Write-Host "  Rehacelo con: .\scripts\setup.ps1 -Recrear" -ForegroundColor Red
    exit 1
}

# --- pip y dependencias ------------------------------------------------------
Escribir-Titulo "Actualizando pip"
& $PythonVenv -m pip install --upgrade pip setuptools wheel --quiet
if ($LASTEXITCODE -ne 0) {
    Escribir-Error "Fallo la actualizacion de pip."
    exit 1
}
Escribir-Ok "pip actualizado"

Escribir-Titulo "Instalando dependencias"

# El modo taller instala una lista aparte, sin ultralytics ni torch: esa maquina solo corre
# el modelo ya exportado a .onnx. Medido en un i5-12400: 244 MB de entorno contra 1.2 GB,
# con el mismo veredicto.
if ($Taller) {
    $RutaLista = Join-Path $RaizRepo "requirements-taller.txt"
    $DetalleLista = "modo TALLER: solo lo necesario para correr (~250 MB, sin torch)"
} else {
    $RutaLista = $RutaRequirements
    $DetalleLista = "instalacion COMPLETA: incluye ultralytics y torch para entrenar (~2.5 GB)"
}

if (-not (Test-Path $RutaLista)) {
    Escribir-Error "No se encontro la lista de dependencias en $RutaLista"
    exit 1
}
Write-Host "  $DetalleLista"
Write-Host "  Esto puede tardar varios minutos..."
& $PythonVenv -m pip install -r $RutaLista
if ($LASTEXITCODE -ne 0) {
    if ($Taller) {
        # onnxruntime es lo unico opcional de la lista del taller. Si no hay wheel para esta
        # version de Python, el sistema sigue andando con el cv2.dnn que ya trae opencv:
        # mas lento (5.8 ms contra 1.8 ms) pero el mismo resultado. No es motivo para
        # dejar la maquina sin instalar.
        Escribir-Aviso "Fallo la instalacion; reintento sin onnxruntime, que es opcional."
        & $PythonVenv -m pip install "opencv-contrib-python>=4.8,<5.0" "numpy>=1.24,<3.0" "pyyaml>=6.0" "pillow>=10.0"
        if ($LASTEXITCODE -ne 0) {
            Escribir-Error "Fallo la instalacion de dependencias. Revisa el error de pip mas arriba."
            exit 1
        }
        Escribir-Aviso "Sin onnxruntime: el .onnx va a correr con cv2.dnn. Anda igual, mas lento."
    } else {
        Escribir-Error "Fallo la instalacion de dependencias. Revisa el error de pip mas arriba."
        exit 1
    }
}
Escribir-Ok "Dependencias instaladas"

# --- Que "python -m afilado..." encuentre el paquete -------------------------
# El codigo vive en src/ y el proyecto no se instala como paquete: se trabaja sobre los
# archivos, sin reinstalar en cada cambio. El precio es que src/ no esta en el path, y
# entonces "python -m afilado.cli.run_live" --el comando que documenta el README-- falla
# con ModuleNotFoundError. Un archivo .pth en site-packages agrega esa carpeta al path:
# es el mismo mecanismo que usa "pip install -e", sin instalar ni copiar nada.
Escribir-Titulo "Registrando src/ en el entorno"

$sitePackages = & $PythonVenv -c "import sysconfig; print(sysconfig.get_paths()['purelib'])" 2>$null
if ($LASTEXITCODE -ne 0 -or [string]::IsNullOrWhiteSpace($sitePackages)) {
    Escribir-Aviso "No se pudo ubicar site-packages del entorno."
    Write-Host "  Vas a tener que anteponer PYTHONPATH en cada corrida:" -ForegroundColor Yellow
    Write-Host "    `$env:PYTHONPATH = 'src'" -ForegroundColor White
} else {
    $rutaPth = Join-Path $sitePackages.Trim() "afilado-src.pth"
    $rutaSrc = Join-Path $RaizRepo "src"
    # Sin BOM: Python lee los .pth linea por linea y un BOM al principio le arruina la ruta.
    $sinBom = New-Object System.Text.UTF8Encoding($false)
    [System.IO.File]::WriteAllText($rutaPth, $rutaSrc + [Environment]::NewLine, $sinBom)

    $comprobacion = & $PythonVenv -c "import afilado; print(afilado.__file__)" 2>$null
    if ($LASTEXITCODE -eq 0 -and -not [string]::IsNullOrWhiteSpace($comprobacion)) {
        Escribir-Ok "python -m afilado.cli.run_live ya encuentra el paquete"
    } else {
        Escribir-Aviso "Se escribio $rutaPth pero 'import afilado' sigue fallando."
        Write-Host "  Como salida de emergencia, antes de cada corrida:" -ForegroundColor Yellow
        Write-Host "    `$env:PYTHONPATH = 'src'" -ForegroundColor White
    }
}

# --- Conflicto de OpenCV -----------------------------------------------------
# ultralytics declara opencv-python como dependencia, asi que pip lo instala igual y
# pisa el cv2 de contrib, dejandonos sin cv2.aruco. Hay que corregirlo despues.
Escribir-Titulo "Verificando OpenCV (modulo aruco)"

$paquetesOpencv = & $PythonVenv -m pip list --format=freeze 2>$null |
    Where-Object { $_ -match '^opencv-(python|python-headless)==' }

if ($paquetesOpencv) {
    Escribir-Aviso "Se detecto opencv-python instalado junto a opencv-contrib-python."
    Write-Host "  No pueden convivir: ambos instalan 'cv2' y el ultimo pisa al otro," -ForegroundColor Yellow
    Write-Host "  dejando el sistema sin cv2.aruco (la referencia de escala). Corrigiendo..." -ForegroundColor Yellow
    & $PythonVenv -m pip uninstall -y opencv-python opencv-python-headless --quiet
    # El pin <5.0 va SI O SI: sin el, pip trae OpenCV 5.x y nos saltea la restriccion de
    # requirements.txt por la puerta de atras. Las comillas tambien: sin ellas PowerShell
    # interpreta el '>' como redireccion a un archivo.
    & $PythonVenv -m pip install --force-reinstall --no-deps "opencv-contrib-python>=4.8,<5.0" --quiet
}

$tieneAruco = & $PythonVenv -c "import cv2; print(hasattr(cv2, 'aruco'))" 2>$null
if ($tieneAruco -match "True") {
    $versionCv = & $PythonVenv -c "import cv2; print(cv2.__version__)" 2>$null
    Escribir-Ok "OpenCV $($versionCv.Trim()) con modulo aruco disponible"
} else {
    Escribir-Aviso "cv2.aruco NO esta disponible. Corregilo a mano con:"
    Write-Host "    .\.venv\Scripts\python.exe -m pip uninstall -y opencv-python opencv-python-headless" -ForegroundColor Yellow
    Write-Host "    .\.venv\Scripts\python.exe -m pip install --force-reinstall --no-deps 'opencv-contrib-python>=4.8,<5.0'" -ForegroundColor Yellow
}

# --- GPU ---------------------------------------------------------------------
# En el taller no se entrena, y sin torch instalado la pregunta no tiene sentido.
if ($Taller) {
    Escribir-Titulo "GPU"
    Escribir-Ok "Modo taller: no se busca GPU (esta maquina no entrena)"
    $hayGpu = $false
    $torchConCuda = $false
} else {

Escribir-Titulo "Detectando GPU NVIDIA"

$comandoNvidiaSmi = Get-Command nvidia-smi -ErrorAction SilentlyContinue
$hayGpu = $false
if ($null -ne $comandoNvidiaSmi) {
    $nombresGpu = & nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>$null
    if ($LASTEXITCODE -eq 0 -and -not [string]::IsNullOrWhiteSpace($nombresGpu)) {
        $hayGpu = $true
        foreach ($gpu in $nombresGpu) {
            if (-not [string]::IsNullOrWhiteSpace($gpu)) { Escribir-Ok "GPU: $($gpu.Trim())" }
        }
    }
}

$torchConCuda = $false
if ($hayGpu) {
    $salidaTorch = & $PythonVenv -c "import torch; print(torch.__version__, torch.cuda.is_available())" 2>$null
    if ($LASTEXITCODE -eq 0 -and $salidaTorch -match "True") {
        $torchConCuda = $true
        Escribir-Ok "torch ve la GPU: $($salidaTorch.Trim())"
    } else {
        Escribir-Aviso "Hay GPU NVIDIA pero torch esta compilado solo para CPU."
        Write-Host ""
        Write-Host "  ultralytics instala el torch CPU por defecto. Para entrenar en GPU," -ForegroundColor Yellow
        Write-Host "  reinstala la build CUDA 12.4 con este comando:" -ForegroundColor Yellow
        Write-Host ""
        Write-Host "    .\.venv\Scripts\python.exe -m pip install --force-reinstall torch torchvision --index-url https://download.pytorch.org/whl/cu126" -ForegroundColor White
        Write-Host ""
        Write-Host "  Son ~2.5 GB de descarga. Despues verifica con:" -ForegroundColor Yellow
        Write-Host "    .\.venv\Scripts\python.exe -c `"import torch; print(torch.cuda.is_available())`"" -ForegroundColor White
    }
} else {
    Escribir-Aviso "No se detecto GPU NVIDIA (nvidia-smi no respondio)."
    Write-Host "  La inferencia en vivo funciona igual en CPU con los modelos chicos (yolo11n)." -ForegroundColor Yellow
    Write-Host "  Pero ENTRENAR en CPU es lentisimo: dias en vez de horas. Si tenes una" -ForegroundColor Yellow
    Write-Host "  NVIDIA y aun asi no aparece, actualiza el driver desde nvidia.com." -ForegroundColor Yellow
}

}

# --- Proximos pasos ----------------------------------------------------------
Escribir-Titulo "Listo. Proximos pasos"

# El taller tiene otro flujo: no genera marcador ni entrena. Lo unico que hace es recibir
# el modelo desde la maquina buena y mirar sierras.
if ($Taller) {
    Write-Host ""
    Write-Host "  1. Activa el entorno en cada terminal nueva:" -ForegroundColor White
    Write-Host "       .\.venv\Scripts\Activate.ps1"
    Write-Host "     Si PowerShell bloquea el script, habilitalo una sola vez con:"
    Write-Host "       Set-ExecutionPolicy -Scope CurrentUser RemoteSigned"
    Write-Host ""
    Write-Host "  2. Copia el modelo desde la maquina de entrenamiento:" -ForegroundColor White
    Write-Host "       models\afilado_best.onnx        <- el modelo"
    Write-Host "       models\afilado_best.onnx.json   <- sus clases y tamano de entrada"
    Write-Host "     Los DOS archivos, siempre juntos. Sin el .json el programa se cae a las"
    Write-Host "     clases del config, y una lista corrida convierte aprobada en rechazada."
    Write-Host "     Despues, en configs\config.yaml:"
    Write-Host "       detector:"
    Write-Host "         pesos: models/afilado_best.onnx"
    Write-Host ""
    Write-Host "  3. Arranca la inspeccion en vivo:" -ForegroundColor White
    Write-Host "       python -m afilado.cli.run_live"
    Write-Host "     Sin modelo copiado arranca igual, en modo recoleccion: no da veredicto,"
    Write-Host "     pero deja archivar fotos por clase."
    Write-Host ""
    Write-Host "  4. Teclas del visor:" -ForegroundColor White
    Write-Host "       a = archivar APROBADA  |  r = archivar RECHAZADA  |  q = salir"
    Write-Host "     Las fotos van a data\coleccion\<clase>\ y son las que despues se llevan"
    Write-Host "     a la maquina de entrenamiento para mejorar el modelo."
    Write-Host ""
    Escribir-Aviso "Esta maquina NO entrena ni exporta: no tiene ultralytics ni torch."
    Write-Host "  Es a proposito: asi el entorno ocupa ~250 MB en vez de 2.5 GB." -ForegroundColor Yellow
    Write-Host "  Para entrenar y exportar el .onnx, corre .\scripts\setup.ps1 SIN -Taller" -ForegroundColor Yellow
    Write-Host "  en la maquina buena." -ForegroundColor Yellow
    Write-Host ""
    exit 0
}

Write-Host ""
Write-Host "  1. Activa el entorno en cada terminal nueva:" -ForegroundColor White
Write-Host "       .\.venv\Scripts\Activate.ps1"
Write-Host "     Si PowerShell bloquea el script, habilitalo una sola vez con:"
Write-Host "       Set-ExecutionPolicy -Scope CurrentUser RemoteSigned"
Write-Host ""
Write-Host "  2. Genera e imprime el marcador ArUco (la referencia de escala):" -ForegroundColor White
Write-Host "       python scripts\make_aruco.py --lado-mm 30 --dpi 300 --salida marcador.png"
Write-Host "     Imprimilo al 100% en papel MATE, sin plastificar, y verifica el lado"
Write-Host "     con calibre. Colocalo a la MISMA ALTURA que la cara que vas a medir."
Write-Host ""
Write-Host "  3. Arranca la inspeccion en vivo:" -ForegroundColor White
Write-Host "       python -m afilado.cli.run_live"
Write-Host "     Sin modelo entrenado usa el detector geometrico: ya mide y te deja"
Write-Host "     recolectar datos desde el dia 1."
Write-Host ""
Write-Host "  4. Recolecta datos con las teclas del visor:" -ForegroundColor White
Write-Host "       e = la IA se equivoco  |  g = ejemplo bueno  |  espacio = captura"
Write-Host "     Se guardan en data\feedback\<fecha>\ con pre-etiquetado YOLO."
Write-Host ""
Write-Host "  5. Cuando tengas unos cientos de ejemplos corregidos, entrena:" -ForegroundColor White
Write-Host "       python -m afilado.cli.prepare_dataset dividir --origen data\dataset\crudo --salida data\dataset"
Write-Host "       python -m afilado.cli.train --datos data\dataset\data.yaml --epocas 150"
Write-Host ""

if (-not $torchConCuda -and $hayGpu) {
    Escribir-Aviso "Recorda instalar torch+cu126 (paso indicado arriba) antes de entrenar."
}

exit 0
