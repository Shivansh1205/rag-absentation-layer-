@echo off
cd /d c:\Users\LENOVO\shivansh\RAGABS~1
c:\Users\LENOVO\shivansh\RAGABS~1\.venv\Scripts\python.exe -m pip install fastapi "uvicorn[standard]"
c:\Users\LENOVO\shivansh\RAGABS~1\.venv\Scripts\python.exe -c "from api.app import app; print('app imports OK')"
