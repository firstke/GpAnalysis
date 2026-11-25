@echo off
setlocal
cd /d "%~dp0"

set "PY=python"
if not exist .venv (
  %PY% -m venv .venv
)
call .venv\Scripts\activate.bat

python -m pip install --upgrade pip --disable-pip-version-check
python -m pip install -r requirements-dev.txt --disable-pip-version-check

python -m PyInstaller --noconfirm --clean --onedir --windowed --name StockAnalysis --add-data "src\gp\AkShareDemo.py;." --collect-all streamlit --collect-all numpy --collect-all scipy --collect-all pandas --collect-all akshare --collect-all plotly --collect-all sklearn --collect-all backports --collect-all webview --hidden-import streamlit.web.cli desktop_launcher.py

echo Windows build finished: dist\StockAnalysis\StockAnalysis.exe
endlocal
