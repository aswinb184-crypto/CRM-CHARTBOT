@echo off
setlocal
cd /d "%~dp0"
if errorlevel 1 (
  echo Could not switch to the project folder.
  pause
  exit /b 1
)
if not exist ".venv\Scripts\python.exe" (
  echo Project Python environment not found. Run setup.bat first.
  pause
  exit /b 1
)

echo Starting Streamlit on http://127.0.0.1:8501
echo Keep this window open. Press Ctrl+C here to stop the server.
".venv\Scripts\python.exe" -u -m streamlit run "%~dp0streamlit_app.py" --server.address 127.0.0.1 --server.port 8501 --server.headless true --browser.gatherUsageStats false
set "STREAMLIT_EXIT_CODE=%ERRORLEVEL%"

echo Streamlit stopped with exit code %STREAMLIT_EXIT_CODE%.
echo Review the output above if it stopped before you pressed Ctrl+C.
pause >nul
exit /b %STREAMLIT_EXIT_CODE%
