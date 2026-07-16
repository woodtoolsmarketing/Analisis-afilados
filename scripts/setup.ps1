#Requires -Version 5.1
<#
.SINOPSIS
    Prepara el entorno de desarrollo de Analisis-afilado.

.DESCRIPCION
    Verifica la version de Python (exige 3.10-3.12), crea el entorno virtual .venv,
    actualiza pip, instala requirements.txt, detecta si hay GPU NVIDIA y sugiere el
    comando de torch+cu124 correspondiente.

    Es idempotente: se puede correr las veces que haga falta. No borra nada sin avisar;
    si .venv ya existe lo reutiliza (para rehacerlo desde cero, usa -Recrear).

.PARAMETER Python
    Ejecutable de Python a usar. Si se omite, busca uno compatible con el lanzador "py".

.PARAMETER Recrear
    Borra el .venv existente y lo crea de nuevo. Pide confirmacion.

.EJEMPLO
    .\scripts\setup.ps1
    .\scripts\setup.ps1 -Python "C:\Python312\python.exe"
    .\scripts\setup.ps1 -Recrear
#>

[CmdletBinding()]
param(
    [string]$Python = "",
    [switch]$Recrear
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
# torch (y por lo tanto ultralytics) solo publica wheels hasta 3.12. En 3.13+ el pip
# install falla con "no matching distribution", que es un mensaje que no le dice nada
# a nadie. Preferimos cortar aca con una explicacion clara.
$VersionMinima = [Version]"3.10"
$VersionMaximaExclusiva = [Version]"3.13"

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
    # Prioriza el lanzador "py" con version explicita: es la forma fiable de agarrar
    # un 3.12 aunque el "python" del PATH sea 3.14.
    foreach ($etiqueta in @("3.12", "3.11", "3.10")) {
        try {
            $ruta = & py "-$etiqueta" -c "import sys; print(sys.executable)" 2>$null
            if ($LASTEXITCODE -eq 0 -and -not [string]::IsNullOrWhiteSpace($ruta)) {
                return $ruta.Trim()
            }
        } catch {
            # el lanzador py no esta instalado o no tiene esa version; seguimos probando
        }
    }
    foreach ($nombre in @("python", "python3")) {
        $comando = Get-Command $nombre -ErrorAction SilentlyContinue
        if ($null -eq $comando) { continue }
        $version = Obtener-VersionPython $comando.Source
        if ($null -ne $version -and $version -ge $VersionMinima -and $version -lt $VersionMaximaExclusiva) {
            return $comando.Source
        }
    }
    return $null
}

Escribir-Titulo "Verificando Python"

if ([string]::IsNullOrWhiteSpace($Python)) {
    $Python = Buscar-PythonCompatible
    if ($null -eq $Python) {
        Escribir-Error "No se encontro ningun Python entre 3.10 y 3.12."
        $comandoPython = Get-Command python -ErrorAction SilentlyContinue
        if ($null -ne $comandoPython) {
            $detectada = Obtener-VersionPython $comandoPython.Source
            if ($null -ne $detectada) {
                Write-Host "       El 'python' de tu PATH es $detectada ($($comandoPython.Source))." -ForegroundColor Red
            }
        }
        Write-Host ""
        Write-Host "  Python 3.13 y 3.14 NO sirven para este proyecto: PyTorch todavia no" -ForegroundColor Red
        Write-Host "  publica wheels para esas versiones, y ultralytics depende de torch." -ForegroundColor Red
        Write-Host "  El pip install fallaria con 'no matching distribution found for torch'." -ForegroundColor Red
        Write-Host ""
        Write-Host "  SOLUCION: instala Python 3.12 desde https://www.python.org/downloads/" -ForegroundColor Yellow
        Write-Host "    1. Descarga 'Windows installer (64-bit)' de la serie 3.12.x" -ForegroundColor Yellow
        Write-Host "    2. Marca 'Add python.exe to PATH' y 'py launcher' durante la instalacion" -ForegroundColor Yellow
        Write-Host "    3. No hace falta desinstalar el Python que ya tenes: pueden convivir" -ForegroundColor Yellow
        Write-Host "    4. Volve a correr este script (usara 3.12 automaticamente via 'py -3.12')" -ForegroundColor Yellow
        Write-Host ""
        Write-Host "  Si ya tenes un 3.12 en una ruta rara, pasalo a mano:" -ForegroundColor Yellow
        Write-Host "    .\scripts\setup.ps1 -Python 'C:\Python312\python.exe'" -ForegroundColor Yellow
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
    Escribir-Error "Python $VersionPython es demasiado viejo. Se requiere 3.10 o superior."
    exit 1
}
if ($VersionPython -ge $VersionMaximaExclusiva) {
    Escribir-Error "Python $VersionPython no esta soportado (maximo: 3.12)."
    Write-Host "  PyTorch no publica wheels para 3.13+. Instala Python 3.12 desde python.org" -ForegroundColor Red
    Write-Host "  y volve a correr el script. Ver el detalle mas arriba." -ForegroundColor Red
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
if ($null -ne $VersionVenv -and $VersionVenv -ge $VersionMaximaExclusiva) {
    Escribir-Error "El .venv existente usa Python $VersionVenv, que no esta soportado."
    Write-Host "  Rehacelo con: .\scripts\setup.ps1 -Recrear -Python 'ruta\a\python3.12.exe'" -ForegroundColor Red
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
if (-not (Test-Path $RutaRequirements)) {
    Escribir-Error "No se encontro requirements.txt en $RutaRequirements"
    exit 1
}
Write-Host "  Esto puede tardar varios minutos (ultralytics arrastra torch, ~2.5 GB)..."
& $PythonVenv -m pip install -r $RutaRequirements
if ($LASTEXITCODE -ne 0) {
    Escribir-Error "Fallo la instalacion de dependencias. Revisa el error de pip mas arriba."
    exit 1
}
Escribir-Ok "Dependencias instaladas"

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
    & $PythonVenv -m pip install --force-reinstall --no-deps opencv-contrib-python --quiet
}

$tieneAruco = & $PythonVenv -c "import cv2; print(hasattr(cv2, 'aruco'))" 2>$null
if ($tieneAruco -match "True") {
    $versionCv = & $PythonVenv -c "import cv2; print(cv2.__version__)" 2>$null
    Escribir-Ok "OpenCV $($versionCv.Trim()) con modulo aruco disponible"
} else {
    Escribir-Aviso "cv2.aruco NO esta disponible. Corregilo a mano con:"
    Write-Host "    .\.venv\Scripts\python.exe -m pip uninstall -y opencv-python opencv-python-headless" -ForegroundColor Yellow
    Write-Host "    .\.venv\Scripts\python.exe -m pip install --force-reinstall opencv-contrib-python" -ForegroundColor Yellow
}

# --- GPU ---------------------------------------------------------------------
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
        Write-Host "    .\.venv\Scripts\python.exe -m pip install --force-reinstall torch torchvision --index-url https://download.pytorch.org/whl/cu124" -ForegroundColor White
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

# --- Proximos pasos ----------------------------------------------------------
Escribir-Titulo "Listo. Proximos pasos"

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
    Escribir-Aviso "Recorda instalar torch+cu124 (paso indicado arriba) antes de entrenar."
}

exit 0
