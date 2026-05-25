#requires -Version 5
<#
  Brevet-GPT launcher.

  Starts the Django async API and the Angular dev server, each in its own
  PowerShell window so you can watch their logs and Ctrl-C them independently.

  It does NOT start, stop, or modify LM Studio, MySQL, or ChromaDB — those are
  managed separately and assumed to be running already.

  Usage:
    .\start.ps1          # start backend + frontend
    .\start.ps1 -Open    # also open the browser once the frontend is ready
#>
param([switch]$Open)

$ErrorActionPreference = 'Stop'
$root     = $PSScriptRoot
$backend  = Join-Path $root 'backend'
$frontend = Join-Path $root 'frontend'

if (-not (Test-Path (Join-Path $backend 'manage.py')))     { throw "backend not found at $backend" }
if (-not (Test-Path (Join-Path $frontend 'package.json'))) { throw "frontend not found at $frontend" }

Write-Host 'Brevet-GPT launcher' -ForegroundColor Cyan
Write-Host '  LM Studio / MySQL / ChromaDB are NOT managed here.' -ForegroundColor DarkGray
Write-Host '  Make sure LM Studio is running with a model loaded.' -ForegroundColor DarkGray

# Backend — Django async API (uvicorn) on :8000
Start-Process powershell -ArgumentList @(
  '-NoExit', '-Command',
  "`$host.UI.RawUI.WindowTitle = 'Brevet-GPT backend :8000'; Set-Location '$backend'; python manage.py brevet"
)

# Frontend — Angular dev server on :4200
Start-Process powershell -ArgumentList @(
  '-NoExit', '-Command',
  "`$host.UI.RawUI.WindowTitle = 'Brevet-GPT frontend :4200'; Set-Location '$frontend'; npm start"
)

Write-Host "`nStarted in separate windows:" -ForegroundColor Green
Write-Host '  backend  -> http://localhost:8000  (POST /api/ask · /api/ask/stream · GET /api/health)'
Write-Host '  frontend -> http://localhost:4200'

if ($Open) {
  Write-Host "`nWaiting for the frontend to compile..." -ForegroundColor DarkGray
  for ($i = 0; $i -lt 90; $i++) {
    try { Invoke-WebRequest 'http://localhost:4200' -UseBasicParsing -TimeoutSec 2 | Out-Null; break }
    catch { Start-Sleep -Seconds 2 }
  }
  Start-Process 'http://localhost:4200'
}
