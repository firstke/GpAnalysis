#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

PYTHON="${PYTHON:-python3}"
if [ ! -d .venv ]; then
  "$PYTHON" -m venv .venv
fi

.venv/bin/python -m pip install --upgrade pip --disable-pip-version-check
.venv/bin/python -m pip install -r requirements-dev.txt --disable-pip-version-check

.venv/bin/python -m PyInstaller \
  --noconfirm \
  --clean \
  --onedir \
  --windowed \
  --name StockAnalysis \
  --osx-bundle-identifier com.example.stockanalysis \
  --add-data "src/gp/AkShareDemo.py:." \
  --collect-all streamlit \
  --collect-all numpy \
  --collect-all scipy \
  --collect-all pandas \
  --collect-all akshare \
  --collect-all plotly \
  --collect-all sklearn \
  --collect-all backports \
  --collect-all webview \
  --hidden-import streamlit.web.cli \
  desktop_launcher.py

echo "macOS build finished: dist/StockAnalysis"
