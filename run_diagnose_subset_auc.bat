@echo off
set ROOT=c:\Users\LENOVO\shivansh\RAGABS~1
set PY=%ROOT%\.venv\Scripts\python.exe
cd /d %ROOT%

%PY% scripts\diagnose_subset_auc.py
