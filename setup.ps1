[CmdletBinding()]
param()

$ErrorActionPreference = 'Stop'
$repoRoot = [System.IO.Path]::GetFullPath($PSScriptRoot)
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
$version = & $python.Source -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")'
if ($version -ne '3.11') {
    throw "Paper-Agent requires Python 3.11; found $version."
}

$git = Get-Command git -ErrorAction Stop
& $git.Source lfs version | Out-Null
if ($LASTEXITCODE -ne 0) { throw 'Git LFS is required.' }

Push-Location -LiteralPath $repoRoot
try {
    if (-not (Test-Path -LiteralPath '.venv\Scripts\python.exe')) {
        & $python.Source -m venv .venv
    }
    $venvPython = Join-Path $repoRoot '.venv\Scripts\python.exe'
    & $venvPython -m pip install --upgrade pip
    & $venvPython -m pip install -r requirements.txt

    $figurePython = Join-Path $repoRoot '.mineru\figure-env\Scripts\python.exe'
    if (-not (Test-Path -LiteralPath $figurePython)) {
        & $python.Source -m venv (Join-Path $repoRoot '.mineru\figure-env')
    }
    & $figurePython -m pip install -r backend\requirements-figure.txt

    & $git.Source lfs pull --include='models/**/*.gguf'
    & $venvPython scripts\bootstrap.py install-resources --root $repoRoot
    & $venvPython scripts\bootstrap.py configure --root $repoRoot
    & $venvPython scripts\post_install_check.py --root $repoRoot
} finally {
    Pop-Location
}
