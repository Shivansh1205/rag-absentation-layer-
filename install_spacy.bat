@echo off
set ROOT=c:\Users\LENOVO\shivansh\RAGABS~1
set PY=%ROOT%\.venv\Scripts\python.exe

echo === Installing spacy^>=3.7 ===
%PY% -m pip install "spacy>=3.7"

echo.
echo === Checking for any unexpected new packages (torch must stay CPU-only) ===
%PY% -m pip list | findstr /i nvidia
if errorlevel 1 echo (no nvidia-* packages -- good)

echo.
echo === Verifying torch is still CPU-only ===
%PY% -c "import torch; print('torch version:', torch.__version__)"

echo.
echo === Downloading en_core_web_sm model ===
%PY% -m spacy download en_core_web_sm

echo.
echo === Confirming en_core_web_sm loads ===
%PY% -c "import spacy; nlp = spacy.load('en_core_web_sm'); print('OK')"
