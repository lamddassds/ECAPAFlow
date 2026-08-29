@echo off
rem ECAPAFlow launcher — uses the localvoice venv (speechbrain + supertonic +
rem onnxruntime-directml already installed). Falls back to plain `python`.
set PY=C:\Users\lauro\.venvs\localvoice\Scripts\python.exe
if not exist "%PY%" set PY=python
"%PY%" "%~dp0app.py"
pause
