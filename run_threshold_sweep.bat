@echo off
cd /d c:\Users\LENOVO\shivansh\RAGABS~1
c:\Users\LENOVO\shivansh\RAGABS~1\.venv\Scripts\python.exe -c "import pandas as pd; pd.set_option('display.max_columns', None); pd.set_option('display.width', 200); df = pd.read_csv('artifacts/eval_clean_subset/threshold_sweep.csv'); print(df.to_string(index=False))"
