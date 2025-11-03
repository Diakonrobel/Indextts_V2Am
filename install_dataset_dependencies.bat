@echo off
echo ========================================
echo Installing Dataset Processing Dependencies
echo ========================================
echo.

echo Installing Python packages...
pip install pysrt>=1.1.2 webvtt-py>=0.4.6 yt-dlp>=2023.12.0 noisereduce>=2.0.1 rich>=13.0.0

echo.
echo ========================================
echo Installation Complete!
echo ========================================
echo.
echo IMPORTANT: You also need FFmpeg installed.
echo Download from: https://ffmpeg.org
echo.
echo To verify FFmpeg:
ffmpeg -version

echo.
pause
