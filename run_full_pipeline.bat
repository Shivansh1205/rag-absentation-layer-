@echo off
cd /d c:\Users\LENOVO\shivansh\RAGABS~1

echo === Step 1: download reranker model ===
c:\Users\LENOVO\shivansh\RAGABS~1\.venv\Scripts\python.exe -c "from sentence_transformers import CrossEncoder; m = CrossEncoder('cross-encoder/ms-marco-MiniLM-L6-v2'); print('reranker loaded OK')"
if errorlevel 1 exit /b 1

echo.
echo === Step 2: re-extract all 6000 rows ===
c:\Users\LENOVO\shivansh\RAGABS~1\.venv\Scripts\python.exe scripts\extract_features.py
if errorlevel 1 exit /b 1

echo.
echo === Step 3: retrain classifier ===
c:\Users\LENOVO\shivansh\RAGABS~1\.venv\Scripts\python.exe scripts\train_and_evaluate.py
if errorlevel 1 exit /b 1

echo.
echo === Step 4: subset AUC diagnostic ===
c:\Users\LENOVO\shivansh\RAGABS~1\.venv\Scripts\python.exe scripts\diagnose_subset_auc.py
