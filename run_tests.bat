@echo off
set ROOT=c:\Users\LENOVO\shivansh\RAGABS~1
set PY=%ROOT%\.venv\Scripts\python.exe

echo === FAST SUITE (not slow) ===
%PY% -m pytest -v -m "not slow"
echo.
echo === FULL SUITE (including slow) ===
%PY% -m pytest -v
