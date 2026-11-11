<#
.SYNOPSIS
    Crea el acceso directo de "Analisis de afilado" en el Escritorio.

.DESCRIPTION
    El taller no entra a una consola ni busca un .exe entre carpetas: abre el programa
    desde un icono en el Escritorio. Esto crea ese icono.

    El acceso apunta al .exe y, sobre todo, le fija la CARPETA DE TRABAJO. Eso importa:
    al lado del ejecutable viven las fotos que archiva el operario (data\coleccion), las
    fotos de modelo (data\referencia), el modelo entrenado (models\) y el registro
    (registro.txt). Un acceso sin carpeta de trabajo deja al programa escribiendo donde
    Windows se le ocurra.

.PARAMETER Exe
    Ruta al ejecutable. Por defecto busca AnalisisAfilado.exe al lado de este script,
    despues en dist\, despues en el Escritorio.

.PARAMETER Nombre
    Como se va a llamar el acceso. Por defecto "Analisis de afilado".

.PARAMETER Destino
    Donde dejarlo. Por defecto el Escritorio del usuario actual.

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File scripts\crear_acceso.ps1

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File scripts\crear_acceso.ps1 -Exe C:\Afilado\AnalisisAfilado.exe
#>
[CmdletBinding()]
param(
    [string]$Exe,
    [string]$Nombre = "Analisis de afilado",
    [string]$Destino,
    [string]$Icono
)

$ErrorActionPreference = "Stop"
$aca = Split-Path -Parent $MyInvocation.MyCommand.Path
$raiz = Split-Path -Parent $aca

# ---- encontrar el ejecutable ------------------------------------------------
if (-not $Exe) {
    $candidatos = @(
        (Join-Path $aca "AnalisisAfilado.exe"),
        (Join-Path $raiz "AnalisisAfilado.exe"),
        (Join-Path $raiz "dist\AnalisisAfilado.exe"),
        (Join-Path ([Environment]::GetFolderPath("Desktop")) "Afilado\AnalisisAfilado.exe"),
        (Join-Path ([Environment]::GetFolderPath("Desktop")) "AnalisisAfilado.exe")
    )
    $Exe = $candidatos | Where-Object { Test-Path $_ } | Select-Object -First 1
}

if (-not $Exe -or -not (Test-Path $Exe)) {
    Write-Host "ERROR: no encontre AnalisisAfilado.exe." -ForegroundColor Red
    Write-Host "       Pasalo a mano:  -Exe C:\ruta\AnalisisAfilado.exe"
    exit 2
}

$Exe = (Resolve-Path $Exe).Path
$carpetaExe = Split-Path -Parent $Exe

# La carpeta tiene que ser escribible: ahi van las fotos y el registro. Si no lo es, el
# programa arranca igual y despues falla al guardar, que es mucho peor que avisar ahora.
$prueba = Join-Path $carpetaExe ".permiso_de_escritura"
try {
    New-Item -ItemType File -Path $prueba -Force | Out-Null
    Remove-Item $prueba -Force
} catch {
    Write-Host "AVISO: no se puede escribir en $carpetaExe" -ForegroundColor Yellow
    Write-Host "       El programa no va a poder guardar las fotos ni el registro."
    Write-Host "       Move el .exe al Escritorio y volve a correr esto."
}

# ---- el icono ---------------------------------------------------------------
# Se apunta AL PROPIO .EXE, que ya lleva el icono adentro desde que construir_exe.py lo
# empaqueta con --icon. Apuntar al assets\afilado.ico del repositorio seria un error: en
# la PC del taller ese repositorio no existe, y el acceso quedaria con el icono en blanco.
# El .exe viaja solo; el acceso tiene que depender unicamente de el.
if (-not $Icono) { $Icono = $Exe }

# ---- crear el acceso --------------------------------------------------------
if (-not $Destino) { $Destino = [Environment]::GetFolderPath("Desktop") }
$lnk = Join-Path $Destino "$Nombre.lnk"

$shell = New-Object -ComObject WScript.Shell
$acceso = $shell.CreateShortcut($lnk)
$acceso.TargetPath = $Exe
$acceso.WorkingDirectory = $carpetaExe
$acceso.IconLocation = "$Icono,0"
$acceso.Description = "Revisa el afilado de la sierra con el microscopio"
$acceso.WindowStyle = 1
$acceso.Save()

Write-Host ""
Write-Host "Listo: $lnk" -ForegroundColor Green
Write-Host "  apunta a:   $Exe"
Write-Host "  trabaja en: $carpetaExe"
Write-Host ""
Write-Host "Doble clic y arranca. Si algo falla, el detalle queda en:"
Write-Host "  $(Join-Path $carpetaExe 'registro.txt')"
