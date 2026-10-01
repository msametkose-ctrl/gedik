@echo off
rem Gedik - masaustunden tek tikla calistir.
rem Her calistiginda GitHub'daki guncel surumu indirir/gunceller, derler,
rem sunucuyu baslatir ve tarayicida web arayuzunu acar.
rem
rem Kendi calisma klasorune dokunmaz: ayri bir klasor kullanir (DIR).
rem O klasordeki yerel degisiklikler her guncellemede silinir.
rem
rem Gerekenler: Git (https://git-scm.com/download/win) ve Rust (https://rustup.rs)
setlocal
set "REPO=https://github.com/msametkose-ctrl/gedik.git"
set "BRANCH=claude/dreamy-fermat-lz00su"
set "DIR=%USERPROFILE%\gedik-guncel"
set "PORT=8080"
title Gedik

where git >nul 2>nul
if errorlevel 1 (
  echo Git kurulu degil: https://git-scm.com/download/win
  pause & exit /b 1
)
where cargo >nul 2>nul
if errorlevel 1 (
  if exist "%USERPROFILE%\.cargo\bin\cargo.exe" set "PATH=%USERPROFILE%\.cargo\bin;%PATH%"
)
where cargo >nul 2>nul
if errorlevel 1 (
  echo Rust kurulu degil: https://rustup.rs
  pause & exit /b 1
)

if not exist "%DIR%\.git" (
  echo Ilk kurulum: %DIR% klasorune indiriliyor...
  git clone --branch "%BRANCH%" "%REPO%" "%DIR%"
  if errorlevel 1 (
    echo Indirme basarisiz. Internet baglantisini ve GitHub girisini kontrol et.
    pause & exit /b 1
  )
) else (
  echo Guncel surum aliniyor...
  git -C "%DIR%" fetch origin "%BRANCH%"
  if errorlevel 1 (
    echo Guncelleme alinamadi, eldeki surumle devam ediliyor.
  ) else (
    git -C "%DIR%" checkout -f -B "%BRANCH%" "origin/%BRANCH%"
  )
)

cd /d "%DIR%"
echo Derleniyor (ilk sefer biraz surer)...
cargo build --release --bin gedik
if errorlevel 1 (
  echo Derleme basarisiz.
  pause & exit /b 1
)

rem Sunucu bu pencerede calisir; tarayici 2 saniye sonra acilir.
start "" /b cmd /c "ping -n 3 127.0.0.1 >nul & start "" http://127.0.0.1:%PORT%"
echo.
echo Tarayicida: http://127.0.0.1:%PORT%
echo Kapatmak icin bu pencereyi kapat ya da Ctrl-C.
echo.
target\release\gedik.exe serve %PORT%
pause
