@echo off
setlocal
cd /d %~dp0

:: 1. 仮想環境が存在すれば有効化
if exist .venv\Scripts\activate (
    call .venv\Scripts\activate
)

:: 2. Python スクリプトを実行
python main.py

pause