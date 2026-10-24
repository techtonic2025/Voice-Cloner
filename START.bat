@echo off
setlocal
cd /d "%~dp0"
where node >nul 2>&1
if errorlevel 1 (
  echo Node.js LTS non e' installato.
  echo Installa Node.js LTS da https://nodejs.org e riapri questo file.
  pause
  exit /b 1
)

set "PORT=4587"
node "%~dp0server.mjs" --open
echo.
echo Il server si e' fermato. Copia qui sopra l'eventuale errore.
pause
