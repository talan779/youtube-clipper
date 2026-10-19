@echo off
title Clipper
cd /d "%~dp0"
call .venv\Scripts\activate.bat
clipper ui
pause
