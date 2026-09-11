param(
    [switch]$Strict
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$python = Join-Path $root ".venv\Scripts\python.exe"
$failed = $false

function Write-Check([string]$name, [bool]$ok, [string]$detail) {
    $mark = if ($ok) { "PASS" } else { "FAIL" }
    $color = if ($ok) { "Green" } else { "Red" }
    Write-Host ("[{0}] {1}: {2}" -f $mark, $name, $detail) -ForegroundColor $color
    if (-not $ok) { $script:failed = $true }
}

Write-Host "Ark 5.0 environment doctor"
Write-Host ("root: {0}" -f $root)

$cfg = Join-Path $root ".venv\pyvenv.cfg"
Write-Check "venv files" ((Test-Path $python) -and (Test-Path $cfg)) "$python"

if (Test-Path $python) {
    $env:PYTHONUTF8 = "1"
    $env:PYTHONIOENCODING = "utf-8"
    $version = & $python -c "import sys; print(sys.version.split()[0]); print(sys.executable)" 2>&1
    $exit = $LASTEXITCODE
    Write-Check "venv interpreter" ($exit -eq 0) (($version -join " | "))

    $pipCheck = & $python -m pip check 2>&1
    $exit = $LASTEXITCODE
    Write-Check "pip dependency consistency" ($exit -eq 0) (($pipCheck -join " | "))

    $imports = & $python -c "import importlib.util; names=['requests','PIL','pydantic','aiohttp','mcp','faster_whisper','yt_dlp']; print(' '.join(f'{n}={'yes' if importlib.util.find_spec(n) else 'no'}' for n in names))" 2>&1
    $exit = $LASTEXITCODE
    Write-Check "Python imports" ($exit -eq 0) (($imports -join " | "))
}

foreach ($tool in @("ffmpeg", "agently-cli", "obsidian")) {
    $cmd = Get-Command $tool -ErrorAction SilentlyContinue
    if ($null -eq $cmd) {
        Write-Host ("[SKIP] {0}: command not installed (integration dependency)" -f $tool) -ForegroundColor Yellow
        if ($Strict) { $failed = $true }
    } else {
        Write-Check ("external " + $tool) $true $cmd.Source
    }
}

if ($failed) {
    Write-Host "Doctor result: FAIL" -ForegroundColor Red
    exit 1
}
Write-Host "Doctor result: PASS (missing optional integration commands are reported as SKIP)" -ForegroundColor Green
exit 0
