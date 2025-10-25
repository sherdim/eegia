@echo off

cd /d "%~dp0"
call venv\Scripts\activate
python training_app.py
@REM waitress-serve --host=127.0.0.1 --port=8050 main:server

@REM pause