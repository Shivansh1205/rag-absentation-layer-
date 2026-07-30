@echo off
cd /d c:\Users\LENOVO\shivansh\RAGABS~1

echo === Step 1: verify train has reranker columns ===
c:\Users\LENOVO\shivansh\RAGABS~1\.venv\Scripts\python.exe -c "import pandas as pd; print('train cols:', list(pd.read_parquet('data/train_features.parquet').columns))"
if errorlevel 1 exit /b 1

echo.
echo === Step 2: re-extract EVAL only (hiding train.parquet) ===
rename data\train.parquet train.parquet.bak
c:\Users\LENOVO\shivansh\RAGABS~1\.venv\Scripts\python.exe scripts\extract_features.py
rename data\train.parquet.bak train.parquet
if errorlevel 1 exit /b 1

echo.
echo === Step 3: verify BOTH parquets have reranker columns ===
c:\Users\LENOVO\shivansh\RAGABS~1\.venv\Scripts\python.exe -c "import pandas as pd; t = pd.read_parquet('data/train_features.parquet'); e = pd.read_parquet('data/eval_features.parquet'); print('train cols:', list(t.columns), 'shape:', t.shape); print('eval cols:', list(e.columns), 'shape:', e.shape); assert 'reranker_max_score' in t.columns, 'TRAIN MISSING RERANKER'; assert 'reranker_max_score' in e.columns, 'EVAL MISSING RERANKER'; print('BOTH PARQUETS HAVE RERANKER COLUMNS - OK')"
if errorlevel 1 exit /b 1

echo.
echo === Step 4: retrain ===
c:\Users\LENOVO\shivansh\RAGABS~1\.venv\Scripts\python.exe scripts\train_and_evaluate.py
if errorlevel 1 exit /b 1

echo.
echo === Step 5: verify new model has reranker features ===
c:\Users\LENOVO\shivansh\RAGABS~1\.venv\Scripts\python.exe -c "import joblib; m = joblib.load('artifacts/eval/model.joblib'); print('model feature_names_in_:', list(m.feature_names_in_)); assert 'reranker_max_score' in m.feature_names_in_, 'MODEL STILL HAS OLD FEATURES'; print('MODEL HAS RERANKER FEATURES - OK')"
if errorlevel 1 exit /b 1

echo.
echo === Step 6: subset AUC diagnostic ===
c:\Users\LENOVO\shivansh\RAGABS~1\.venv\Scripts\python.exe scripts\diagnose_subset_auc.py
