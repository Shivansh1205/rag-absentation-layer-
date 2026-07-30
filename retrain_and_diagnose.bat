@echo off
set ROOT=c:\Users\LENOVO\shivansh\RAGABS~1
set PY=%ROOT%\.venv\Scripts\python.exe
cd /d %ROOT%

echo === STEP A: retrain (re-pickle with current sklearn) ===
%PY% scripts\train_and_evaluate.py --data-dir data --output-dir artifacts\eval
echo.
echo === STEP B: clean-subset AUC diagnostic ===
%PY% scripts\diagnose_subset_auc.py
