@echo off
chcp 65001 >nul
echo =========================================
echo Execute all scripts in the Text_Sequence folder with one click
echo =========================================
echo.

cd /d "%~dp0"

echo Start executing scripts...
echo.

echo [1/2] Execute fuse_e3_S_D_C.py
python fuse_e3_S_D_C.py
if errorlevel 1 (
    echo Error: fuse_e3_S_D_C.py execution failed
    pause
    exit /b 1
)
echo Completed
echo.

echo [2/2] Execute Classification_e3_S_D_C.py
python Classification_e3_S_D_C.py
if errorlevel 1 (
    echo Error: Classification_e3_S_D_C.py execution failed
    pause
    exit /b 1
)
echo Completed
echo.


echo =========================================
echo All scripts have been executed successfully!
echo =========================================
pause