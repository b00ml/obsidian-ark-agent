param(
    [switch]$Strict
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$python = Join-Path $root ".venv\Scripts\python.exe"
if (-not (Test-Path $python)) { Write-Error "venv interpreter not found: $python"; exit 1 }
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"
& $python (Join-Path $PSScriptRoot "doctor.py")
$code = $LASTEXITCODE
if ($Strict -and $code -eq 0) {
    # Optional integrations are reported by doctor; strict mode requires them.
    foreach ($tool in @("ffmpeg", "agently-cli", "obsidian")) {
        if (-not (Get-Command $tool -ErrorAction SilentlyContinue)) { $code = 1 }
    }
}
exit $code
