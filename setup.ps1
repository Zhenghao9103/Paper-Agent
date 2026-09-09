[CmdletBinding()]
param()

$ErrorActionPreference = 'Stop'
function Invoke-Native {
    param(
        [Parameter(Mandatory = $true)][string]$FilePath,
        [Parameter(Mandatory = $true)][string[]]$ArgumentList
    )
    & $FilePath @ArgumentList
    if ($LASTEXITCODE -ne 0) {
        throw "Native command failed with exit code ${LASTEXITCODE}: $FilePath"
    }
}

$repoRoot = [System.IO.Path]::GetFullPath($PSScriptRoot)
if ($repoRoot.StartsWith('\\')) {
    throw 'Paper-Agent full installation requires a local Windows drive, not a UNC path.'
}
$driveRoot = [System.IO.Path]::GetPathRoot($repoRoot)
$driveName = $driveRoot.Substring(0, 1)
$freeBytes = (Get-PSDrive -Name $driveName).Free
if ($freeBytes -lt 20GB) {
    throw 'Paper-Agent full installation requires at least 20 GB free space.'
}

if (-not [Environment]::Is64BitOperatingSystem) {
    throw 'Paper-Agent requires 64-bit Windows.'
}

$python = Get-Command python -ErrorAction Stop
$version = Invoke-Native -FilePath $python.Source -ArgumentList @(
    '-c', 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")'
)
if ($version -ne '3.11') {
    throw "Paper-Agent requires Python 3.11; found $version."
}

$git = Get-Command git -ErrorAction Stop
Invoke-Native -FilePath $git.Source -ArgumentList @('lfs', 'version') | Out-Null

Push-Location -LiteralPath $repoRoot
try {
    if (-not (Test-Path -LiteralPath '.venv\Scripts\python.exe')) {
        Invoke-Native -FilePath $python.Source -ArgumentList @('-m', 'venv', '.venv')
    }
    $venvPython = Join-Path $repoRoot '.venv\Scripts\python.exe'
    Invoke-Native -FilePath $venvPython -ArgumentList @('-m', 'pip', 'install', '--upgrade', 'pip')
    Invoke-Native -FilePath $venvPython -ArgumentList @('-m', 'pip', 'install', '-r', 'requirements.txt')

    $figurePython = Join-Path $repoRoot '.mineru\figure-env\Scripts\python.exe'
    if (-not (Test-Path -LiteralPath $figurePython)) {
        Invoke-Native -FilePath $python.Source -ArgumentList @(
            '-m', 'venv', (Join-Path $repoRoot '.mineru\figure-env')
        )
    }
    Invoke-Native -FilePath $figurePython -ArgumentList @(
        '-m', 'pip', 'install', '-r', 'backend\requirements-figure.txt'
    )

    Invoke-Native -FilePath $git.Source -ArgumentList @(
        'lfs', 'pull', '--include=models/**/*.gguf'
    )
    Invoke-Native -FilePath $venvPython -ArgumentList @(
        'scripts\bootstrap.py', 'install-resources', '--root', $repoRoot
    )
    Invoke-Native -FilePath $venvPython -ArgumentList @(
        'scripts\bootstrap.py', 'configure', '--root', $repoRoot
    )
    Invoke-Native -FilePath $venvPython -ArgumentList @(
        'scripts\post_install_check.py', '--root', $repoRoot
    )
} finally {
    Pop-Location
}
