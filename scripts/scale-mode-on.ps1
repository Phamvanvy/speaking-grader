# Bật profile "nhiều user": tắt llama-server (giải phóng cả 2 GPU cho
# Whisper/wav2vec), rồi recreate container với .env.scale (backend openrouter).
#
# Chạy từ gốc repo:  powershell -File scripts\scale-mode-on.ps1

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $repoRoot

$llama = Get-Process -Name "llama-server" -ErrorAction SilentlyContinue
if ($llama) {
    Write-Host "Đang tắt llama-server (PID $($llama.Id))..." -ForegroundColor Yellow
    Stop-Process -Id $llama.Id -Force
    Start-Sleep -Seconds 2
    Write-Host "Đã tắt llama-server." -ForegroundColor Green
} else {
    Write-Host "llama-server không chạy — bỏ qua bước tắt." -ForegroundColor Gray
}

Write-Host "Recreate container với profile scale (.env.scale)..." -ForegroundColor Yellow
docker compose -f docker-compose.yml -f docker-compose.scale.yml up -d --force-recreate
if ($LASTEXITCODE -ne 0) { throw "docker compose up thất bại (exit $LASTEXITCODE)." }

Write-Host "`nProfile scale đã bật. Kiểm tra:" -ForegroundColor Green
Write-Host "  curl http://localhost:8000/health"
Write-Host "`nQuay lại bình thường: powershell -File scripts\scale-mode-off.ps1" -ForegroundColor Gray
