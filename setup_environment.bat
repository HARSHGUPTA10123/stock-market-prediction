@echo off
echo ========================================
echo  Stock Market Analysis - Environment Setup
echo ========================================
echo.

echo Step 1: Creating Python virtual environment...
python -m venv stock_env

echo Step 2: Activating environment...
call stock_env\Scripts\activate.bat

echo Step 3: Upgrading pip...
pip install --upgrade pip

echo Step 4: Installing dependencies...
pip install -r requirements.txt

echo.
echo ========================================
echo  SETUP COMPLETE!
echo ========================================
echo.
echo To activate environment:
echo   stock_env\Scripts\activate.bat
echo.
echo To run the application:
echo   streamlit run webapp.py
echo.
pause