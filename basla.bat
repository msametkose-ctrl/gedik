@echo off
rem Gedik - derle, sunucuyu baslat, tarayiciyi ac.
rem   basla.bat        -> http://127.0.0.1:8080
rem   basla.bat 9000   -> baska port
setlocal
cd /d "%~dp0"
set PORT=%1
if "%PORT%"=="" set PORT=8080

where cargo >nul 2>nul
if errorlevel 1 (
  if exist "%USERPROFILE%\.cargo\bin\cargo.exe" set "PATH=%USERPROFILE%\.cargo\bin;%PATH%"
)
where cargo >nul 2>nul
if errorlevel 1 (
  echo Rust kurulu degil: https://rustup.rs
  pause & exit /b 1
)

echo Derleniyor...
cargo build --release
if errorlevel 1 ( echo Derleme basarisiz. & pause & exit /b 1 )

start "" http://127.0.0.1:%PORT%
target\release\gedik.exe serve %PORT%
