@echo off
set ROOT=c:\Users\LENOVO\shivansh\RAGABS~1
set PY=%ROOT%\.venv\Scripts\python.exe
%PY% -m pip show datasets huggingface_hub pyarrow
echo.
%PY% -m pytest -v
