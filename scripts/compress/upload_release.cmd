@echo off
chcp 65001 >nul
echo ============================================================
echo  Mage-VL upload : starting in a MINIMIZED window.
echo  You may close THIS window; the upload keeps running there.
echo  Log: E:\MageVL\compress\upload-*.log
echo  Stop it by closing the minimized "MageVL upload" window.
echo ============================================================
start "MageVL upload" /min powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0upload_release.ps1" %*
timeout /t 6 >nul