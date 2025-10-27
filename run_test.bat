@echo off
echo ===============================================
echo Nong-View DB Test Environment Setup
echo ===============================================
echo.

echo [1] Installing required packages...
C:\Users\ASUS\anaconda3\python.exe -m pip install sqlalchemy==2.0.23 python-dotenv==1.0.0 alembic==1.12.1

echo.
echo [2] Running database test...
C:\Users\ASUS\anaconda3\python.exe D:\Nong-View\test_db.py

echo.
pause