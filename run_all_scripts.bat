@echo off
chcp 65001 >nul

echo =========================================
echo Execute run_all_scripts.bat in all folders
echo =========================================
echo.

set "folder1=.\Text_Sequence"
set "folder2=.\Dynamic_Path"
set "folder3=.\Figure_Structure"
set "folder4=.\Feature_Fusion"

echo [1/4] Executing scripts in Folder1
cd /d "%folder1%"
call run_all_scripts.bat
if errorlevel 1 (
    echo Error: Script execution in Folder1 failed
    pause
    exit /b 1
)
echo Folder1 scripts execution completed
echo.

echo [2/4] Executing scripts in Folder2
cd /d "%folder2%"
call run_all_scripts.bat
if errorlevel 1 (
    echo Error: Script execution in Folder2 failed
    pause
    exit /b 1
)
echo Folder2 scripts execution completed
echo.

echo [3/4] Executing scripts in Folder3
cd /d "%folder3%"
call run_all_scripts.bat
if errorlevel 1 (
    echo Error: Script execution in Folder3 failed
    pause
    exit /b 1
)
echo Folder3 scripts execution completed
echo.

echo [4/4] Executing scripts in Folder4
cd /d "%folder4%"
call run_all_scripts.bat
if errorlevel 1 (
    echo Error: Script execution in Folder3 failed
    pause
    exit /b 1
)
echo Folder3 scripts execution completed
echo.


echo =========================================
echo All folder scripts have been executed successfully!
echo =========================================
pause