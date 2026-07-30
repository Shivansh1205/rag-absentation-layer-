@echo off
set ROOT=c:\Users\LENOVO\shivansh\RAGABS~1
set PY=%ROOT%\.venv\Scripts\python.exe
%PY% scripts\generate_dataset.py --n-train 10 --n-eval 5
