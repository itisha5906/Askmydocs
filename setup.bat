@echo off
REM setup.bat — Windows setup script for AskMyDocs
REM Run once: setup.bat

echo ============================================
echo  AskMyDocs — Windows Setup
echo ============================================

REM Create virtual environment
if not exist "venv" (
    echo Creating virtual environment...
    python -m venv venv
)

REM Activate venv
call venv\Scripts\activate.bat

REM Upgrade pip
python -m pip install --upgrade pip

REM Install dependencies
echo Installing requirements...
pip install -r requirements.txt

REM Copy .env if it does not exist
if not exist ".env" (
    echo Copying .env.example to .env ...
    copy .env.example .env
    echo IMPORTANT: Edit .env and add your GROQ_API_KEY before running the app.
)

REM Create data directory
if not exist "data\faiss_index" (
    mkdir data\faiss_index
)

echo.
echo Setup complete!
echo.
echo Next steps:
echo   1. Edit .env and set your GROQ_API_KEY
echo   2. Run: venv\Scripts\activate.bat
echo   3. Run: streamlit run app.py
