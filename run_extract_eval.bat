@echo off
cd /d c:\Users\LENOVO\shivansh\RAGABS~1
rename data\train.parquet train.parquet.bak
c:\Users\LENOVO\shivansh\RAGABS~1\.venv\Scripts\python.exe scripts\extract_features.py
rename data\train.parquet.bak train.parquet
