# Tắt profile "nhiều user": recreate container về cấu hình thường (.env,
# backend local). KHÔNG tự khởi động lại llama-server — script không biết
# lệnh/tham số bạn dùng để chạy nó (model path, context size, tensor-split...),
# tự đoán sai dễ khởi động nhầm cấu hình. Tự chạy lại llama-server thủ công
# nếu muốn dùng lại backend local.
#
# Chạy từ gốc repo:  powershell -File scripts\scale-mode-off.ps1

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $repoRoot

Write-Host "Recreate container với cấu hình thường (.env)..." -ForegroundColor Yellow
docker compose up -d --force-recreate
if ($LASTEXITCODE -ne 0) { throw "docker compose up thất bại (exit $LASTEXITCODE)." }

Write-Host "`nĐã quay lại cấu hình thường." -ForegroundColor Green
Write-Host "LƯU Ý: llama-server CHƯA được khởi động lại — .env đang backend=local" -ForegroundColor Yellow
Write-Host "cần llama-server chạy thì mới chấm được. Tự bật lại bằng lệnh cũ của bạn." -ForegroundColor Yellow
