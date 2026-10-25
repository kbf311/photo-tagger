@echo off
setlocal
cd /d %~dp0

:: ============================================
:: 設定項目
:: 作成したい Python のバージョンと仮想環境名
:: ============================================
set "PYTHON_VERSION=3.11"
set "VENV_DIR=.venv"

echo ============================================
echo   環境セットアップスクリプト (uv)
echo ============================================

:: 1. uv がインストールされているかチェック
where uv >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] uv がインストールされていないか、PATHが通っていません。
    echo.
    echo 以下のコマンドを PowerShell で実行して uv をインストールしてください:
    echo powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
    echo.
    echo ※ インストール完了後、再度このバッチを実行してください。
    echo ============================================
    pause
    exit /b 1
)

echo [OK] uv が見つかりました。
uv self update
echo --------------------------------------------

:: 2. 特定の Python バージョンで仮想環境 (.venv) を作成
:: (該当バージョンの Python が未インストールの場合は uv が自動でダウンロードしてくれます)
echo Python %PYTHON_VERSION% の仮想環境 (%VENV_DIR%) を作成しています...
uv venv %VENV_DIR% --python %PYTHON_VERSION% --clear
if %errorlevel% neq 0 (
    echo [ERROR] 仮想環境の作成に失敗しました。
    pause
    exit /b 1
)

:: 3. インストールするファイルの選択
echo.
echo ============================================
echo   どちらのモードでインストールしますか？
echo ============================================
echo   [1] requirements.txt
echo       - パッケージを最新バージョンでインストールします
echo       - 新機能を使いたい場合や、通常はこちらを推奨
echo.
echo   [2] requirements-lock.txt
echo       - 動作確認済みのバージョンに固定してインストールします
echo       - エラーを避けて、確実に動かしたい場合はこちらを推奨
echo ============================================
echo   [1] または [2] を入力して Enter を押してください。
echo.

:choice
set SELECT=1
set /p SELECT="番号を入力してください (1 or 2) [Enterで1を選択]: "

if "%SELECT%"=="1" goto req
if "%SELECT%"=="2" goto lock
echo [ERROR] 1か2を入力してください。
echo.
goto choice

:req
set "TARGET_FILE=requirements.txt"
goto install

:lock
set "TARGET_FILE=requirements-lock.txt"
goto install

:install
:: 4. パッケージのインストール
echo.
if not exist "%TARGET_FILE%" (
    echo [ERROR] %TARGET_FILE% が見つかりませんでした。
    pause
    exit /b 1
)

echo %TARGET_FILE% からパッケージをインストールしています...
uv pip install --python "%VENV_DIR%" -r "%TARGET_FILE%"
if errorlevel 1 (
    echo [ERROR] パッケージのインストールに失敗しました。
    pause
    exit /b 1
)
echo [OK] パッケージのインストールが完了しました。

:: [1] を選択し、かつ requirements-lock.txt が存在しない場合は作成する
if "%SELECT%"=="1" if not exist "requirements-lock.txt" (
    echo.
    echo requirements-lock.txt が存在しないため、現在の環境から作成します...
    uv pip freeze --python "%VENV_DIR%" > requirements-lock.txt
    echo [OK] requirements-lock.txt を作成しました。
)

echo.
echo ============================================
echo   セットアップが正常に完了しました！
echo ============================================
pause