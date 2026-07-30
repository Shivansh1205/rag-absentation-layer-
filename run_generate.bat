@echo off
set ROOT=c:\Users\LENOVO\shivansh\RAGABS~1
set PY=%ROOT%\.venv\Scripts\python.exe
echo === Generating full 5000/1000 dataset ===
%PY% scripts\generate_dataset.py --n-train 5000 --n-eval 1000 --seed 42 --out-dir data
