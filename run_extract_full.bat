@echo off
set ROOT=c:\Users\LENOVO\shivansh\RAGABS~1
set PY=%ROOT%\.venv\Scripts\python.exe
echo === Running full feature extraction ===
%PY% scripts\extract_features.py
