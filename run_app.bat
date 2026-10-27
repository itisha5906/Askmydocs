@echo off
title AskMyDocs Web App
cd /d "%~dp0"
echo ========================================================
echo Starting AskMyDocs Web Application...
echo Browser will open automatically at http://localhost:8501
echo ========================================================
"C:\Users\itish\AppData\Local\Programs\Python\Python313\python.exe" -m streamlit run app.py
pause
