# PowerShell script to test audio loading
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $scriptDir

Write-Host "Testing audio loading functionality..." -ForegroundColor Green

try {
    python simple_test.py
} catch {
    Write-Host "Error running test: $_" -ForegroundColor Red
}

Write-Host "Test completed." -ForegroundColor Yellow
