@echo off
set ROOT=c:\Users\LENOVO\shivansh\RAGABS~1
set PY=%ROOT%\.venv\Scripts\python.exe
%PY% scripts\extract_features.py --limit 20
echo.
echo === Output parquet shape and columns ===
%PY% -c "import pandas as pd; df=pd.read_parquet('data/train_features.parquet'); print('train_features shape:', df.shape); print('columns:', list(df.columns))"
%PY% -c "import pandas as pd; df=pd.read_parquet('data/eval_features.parquet'); print('eval_features shape:', df.shape); print('columns:', list(df.columns))"
