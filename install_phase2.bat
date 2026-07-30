@echo off
set ROOT=c:\Users\LENOVO\shivansh\RAGABS~1
set PY=%ROOT%\.venv\Scripts\python.exe

echo === Installing torch from PyTorch CPU-only wheel index ===
%PY% -m pip install torch --index-url https://download.pytorch.org/whl/cpu

echo.
echo === Installing sentence-transformers ===
%PY% -m pip install "sentence-transformers>=3.0"

echo.
echo === Verifying torch is CPU-only (no +cuXXX suffix) ===
%PY% -c "import torch; print('torch version:', torch.__version__)"

echo.
echo === Checking for any nvidia-* packages (should be none) ===
%PY% -m pip list | findstr /i nvidia
if errorlevel 1 echo (none found -- good)
