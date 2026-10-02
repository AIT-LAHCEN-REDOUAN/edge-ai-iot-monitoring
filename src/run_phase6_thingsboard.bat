@echo off
title PHASE 6 - ThingsBoard Test

echo ========================================
echo   PHASE 6 — ThingsBoard FULL TEST
echo ========================================
echo.

cd /d D:\github\IOT_Project

:: --------------------------------------------------
:: 1. Health Check
:: --------------------------------------------------
echo [1] Checking VM health...
curl http://localhost:5001/health
echo.
curl http://localhost:5002/health
echo.
curl http://localhost:5003/health
echo.
timeout /t 2 >nul

:: --------------------------------------------------
:: 2. Prepare payload (STATIC - from your test)
:: --------------------------------------------------
echo [2] Preparing payload...

set BODY={"slice_path":"/app/data/processed/slices/test/0/lh.pptnIXI329-HH-1908-MADisoTFE1_-s3T181_-0301-00003-000001-01_slice_032.npy","patient_id":"lh.pptnIXI329-HH-1908-MADisoTFE1_-s3T181_-0301-00003-000001-01"}

echo %BODY%
timeout /t 2 >nul

:: --------------------------------------------------
:: 3. Inference test
:: --------------------------------------------------
echo [3] Running inference...

curl -X POST http://localhost:5001/infer -H "Content-Type: application/json" -d "%BODY%"
echo.
curl -X POST http://localhost:5002/infer -H "Content-Type: application/json" -d "%BODY%"
echo.
curl -X POST http://localhost:5003/infer -H "Content-Type: application/json" -d "%BODY%"
echo.
timeout /t 2 >nul

:: --------------------------------------------------
:: 4. Deployment matrix
:: --------------------------------------------------
echo [4] Running deployment matrix...

python src\deployment\measure_vm_matrix.py

echo.
echo ===== SUMMARY =====
type results\deployment\vm_tech_matrix_summary.txt
timeout /t 2 >nul

:: --------------------------------------------------
:: 5. ThingsBoard config
:: --------------------------------------------------
echo [5] Configuring ThingsBoard...

set TB_HOST=localhost
set TB_PORT=1883
set TB_TOKEN=i2d4b9rpmz3lq23ba65h

:: --------------------------------------------------
:: 6. Publish scores
:: --------------------------------------------------
echo [6] Publishing scores...

python src\thingsboard\publish_scores.py
timeout /t 2 >nul

:: --------------------------------------------------
:: 7. Collective evaluation
:: --------------------------------------------------
echo [7] Running collective evaluation...

python src\collective\run_collective.py

echo.
echo ===== COLLECTIVE SUMMARY =====
type results\collective\summary.txt
timeout /t 2 >nul

:: --------------------------------------------------
:: 8. Publish collective results
:: --------------------------------------------------
echo [8] Publishing collective summary...

python src\thingsboard\publish_collective_summary.py
timeout /t 2 >nul

:: --------------------------------------------------
:: 9. Live telemetry simulation
:: --------------------------------------------------
echo [9] Sending live telemetry...

for /l %%i in (1,1,10) do (
    curl -X POST http://localhost:5001/infer -H "Content-Type: application/json" -d "%BODY%" >nul
    curl -X POST http://localhost:5002/infer -H "Content-Type: application/json" -d "%BODY%" >nul
    curl -X POST http://localhost:5003/infer -H "Content-Type: application/json" -d "%BODY%" >nul
    timeout /t 1 >nul
)

echo Live telemetry sent.
echo.

:: --------------------------------------------------
:: 10. Open ThingsBoard
:: --------------------------------------------------
echo [10] Opening ThingsBoard...
start http://localhost:9090

echo.
echo ========================================
echo   ✅ PHASE 6 TEST COMPLETED
echo ========================================

pause