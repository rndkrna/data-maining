@echo off
chcp 65001 >nul
title Website Demonstrasi Analisis Sentimen Berita Politik - UMRAH
cls
echo ==============================================================================
echo   MEMULAI SERVER APLIKASI WEB DEMONSTRASI ANALISIS SENTIMEN
echo ==============================================================================
echo   Program Studi Teknik Informatika - Universitas Maritim Raja Ali Haji
echo   Teknik Penambangan Data: InSet Lexicon + TF-IDF + Linear SVM & Naive Bayes
echo ==============================================================================
echo.

where python >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Python tidak terdeteksi di sistem PATH!
    echo Harap pastikan Python 3 sudah terpasang.
    pause
    exit /b
)

echo [+] Memulai server web lokal dan membuka peramban otomatis...
echo [+] Tekan CTRL + C di jendela ini untuk mematikan server.
echo.

python app.py --open

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [!] Server berhenti dengan kode error: %ERRORLEVEL%
    pause
)
