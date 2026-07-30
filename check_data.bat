@echo off
set ROOT=c:\Users\LENOVO\shivansh\RAGABS~1
set PY=%ROOT%\.venv\Scripts\python.exe
echo === Checking existing parquet row counts ===
%PY% -c "import pandas as pd; df=pd.read_parquet('data/train.parquet'); print('train.parquet rows:', len(df))"
%PY% -c "import pandas as pd; df=pd.read_parquet('data/eval.parquet'); print('eval.parquet rows:', len(df))"
