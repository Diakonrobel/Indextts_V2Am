@echo off
REM IndexTTS2 Model Downloader for Windows
echo ========================================
echo IndexTTS2 Model Downloader
echo ========================================
echo.

python download_models.py

if %ERRORLEVEL% EQU 0 (
    echo.
    echo ========================================
    echo Download completed successfully!
    echo ========================================
    pause
) else (
    echo.
    echo ========================================
    echo Download failed or incomplete.
    echo Please check the error messages above.
    echo ========================================
    pause
    exit /b 1
)
