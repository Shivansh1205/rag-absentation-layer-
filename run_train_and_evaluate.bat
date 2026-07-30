@echo off
set ROOT=c:\Users\LENOVO\shivansh\RAGABS~1
set PY=%ROOT%\.venv\Scripts\python.exe
cd /d %ROOT%

%PY% scripts\train_and_evaluate.py --data-dir data --output-dir artifacts\eval
