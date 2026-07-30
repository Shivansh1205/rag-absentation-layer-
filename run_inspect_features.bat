@echo off
cd /d c:\Users\LENOVO\shivansh\RAGABS~1
c:\Users\LENOVO\shivansh\RAGABS~1\.venv\Scripts\python.exe -c "import pandas as pd; df = pd.read_parquet('data/train_features.parquet'); print('columns:', list(df.columns)); print('shape:', df.shape)"
