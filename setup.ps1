[CmdletBinding()]
param(
    [ValidateSet('local', 'jev')][string]$RouterProvider
)

$ErrorActionPreference = 'Stop'
function Resolve-RouterProvider {
    param(
        [string]$RepoRoot,
        [string]$Override
    )
    if ($Override) {
        return $Override
    }
    $envPath = Join-Path $RepoRoot '.env'
    if (-not (Test-Path -LiteralPath $envPath)) {
        return 'local'
    }
    $selected = 'local'
    foreach ($line in [System.IO.File]::ReadAllLines($envPath)) {
        if ($line -match '^\s*ROUTER_PROVIDER\s*=\s*(.*?)\s*$') {
            $selected = $Matches[1].Trim().Trim('"', "'").ToLowerInvariant()
        }
    }
    if ($selected -notin @('local', 'jev')) {
        throw "Invalid ROUTER_PROVIDER in .env: use local or jev."
    }
    return $selected
}
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
$RouterProvider = Resolve-RouterProvider -RepoRoot $repoRoot -Override $RouterProvider
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

if ($RouterProvider -eq 'local') {
    $git = Get-Command git -ErrorAction Stop
    Invoke-Native -FilePath $git.Source -ArgumentList @('lfs', 'version') | Out-Null
}

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

    if ($RouterProvider -eq 'local') {
        Invoke-Native -FilePath $git.Source -ArgumentList @(
            'lfs', 'pull', '--include=models/**/*.gguf'
        )
    }
    Invoke-Native -FilePath $venvPython -ArgumentList @(
        'scripts\bootstrap.py', 'install-resources', '--root', $repoRoot,
        '--router-provider', $RouterProvider
    )
    Invoke-Native -FilePath $venvPython -ArgumentList @(
        'scripts\bootstrap.py', 'configure', '--root', $repoRoot,
        '--router-provider', $RouterProvider
    )
    Invoke-Native -FilePath $venvPython -ArgumentList @(
        'scripts\post_install_check.py', '--root', $repoRoot,
        '--router-provider', $RouterProvider
    )
} finally {
    Pop-Location
}
