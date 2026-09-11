param(
    [switch]$SkipArk
)

$ErrorActionPreference = "Continue"
$root = Split-Path -Parent $PSScriptRoot
$python = Join-Path $root ".venv\Scripts\python.exe"
$results = @()

if (-not (Test-Path $python)) {
    Write-Host "[FAIL] .venv interpreter not found: $python" -ForegroundColor Red
    exit 1
}

$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"

function Invoke-PythonSuite([string]$name, [string]$workdir, [string[]]$pythonArgs) {
    Write-Host ("`n=== {0} ===" -f $name)
    Push-Location $workdir
    try {
        & $python @pythonArgs
        $code = $LASTEXITCODE
    } finally {
        Pop-Location
    }
    $script:results += [pscustomobject]@{ Name = $name; Code = $code }
}

Invoke-PythonSuite "agentlab" (Join-Path $root "agentlab") @("-m", "unittest", "discover", "-s", "tests", "-v")
Invoke-PythonSuite "vault-gateway / brain" $root @("-m", "unittest", "obsidian_agent_brain.test_mcp", "-v")
Invoke-PythonSuite "content pipelines / bili" $root @("-m", "unittest", "discover", "-s", "bili_summarizer", "-p", "test_*.py", "-v")
Invoke-PythonSuite "inbox collector" $root @("-m", "unittest", "discover", "-s", "inbox_collector", "-p", "test_*.py", "-v")

if (-not $SkipArk) {
    $npm = Get-Command npm -ErrorAction SilentlyContinue
    if ($null -eq $npm) {
        Write-Host "`n[SKIP] Ark build: npm is not installed (integration dependency)" -ForegroundColor Yellow
    } else {
        Write-Host "`n=== Ark plugin ==="
        Push-Location (Join-Path $root "ark")
        try {
            & npm run build
            $code = $LASTEXITCODE
        } finally {
            Pop-Location
        }
        $results += [pscustomobject]@{ Name = "Ark plugin"; Code = $code }
    }
}

Write-Host "`n=== summary ==="
$failed = $false
foreach ($result in $results) {
    if ($result.Code -eq 0) {
        Write-Host ("[PASS] {0}" -f $result.Name) -ForegroundColor Green
    } else {
        Write-Host ("[FAIL] {0} (exit {1})" -f $result.Name, $result.Code) -ForegroundColor Red
        $failed = $true
    }
}
if ($failed) { exit 1 }
Write-Host "All requested suites passed; unavailable integrations were skipped." -ForegroundColor Green
exit 0
