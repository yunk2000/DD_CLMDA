@echo off
chcp 65001 >nul
echo =========================================
echo Execute all scripts in the Text_Sequence folder with one click
echo =========================================
echo.

cd /d "%~dp0"

echo Start executing scripts...
echo.

echo [1/8] Execute or_generate_CPG.py
python or_generate_CPG.py
if errorlevel 1 (
    echo Error: or_generate_CPG.py execution failed
    pause
    exit /b 1
)
echo Completed
echo.

echo [2/8] Execute or_make_to_graphData.py
python or_make_to_graphData.py
if errorlevel 1 (
    echo Error: or_make_to_graphData.py execution failed
    pause
    exit /b 1
)
echo Completed
echo.

echo [3/8] Execute or_encoder_GAT_cpg.py
python or_encoder_GAT_cpg.py
if errorlevel 1 (
    echo Error: or_encoder_GAT_cpg.py execution failed
    pause
    exit /b 1
)
echo Completed
echo.

echo [4/8] Execute ta_generate_CPG.py
python ta_generate_CPG.py
if errorlevel 1 (
    echo Error: ta_generate_CPG.py execution failed
    pause
    exit /b 1
)
echo Completed
echo.

echo [5/8] Execute ta_make_to_graphData.py
python ta_make_to_graphData.py
if errorlevel 1 (
    echo Error: ta_make_to_graphData.py execution failed
    pause
    exit /b 1
)
echo Completed
echo.

echo [6/8] Execute ta_encoder_GAT_cpg.py
python ta_encoder_GAT_cpg.py
if errorlevel 1 (
    echo Error: ta_encoder_GAT_cpg.py execution failed
    pause
    exit /b 1
)
echo Completed
echo.

echo [7/8] Execute data_aug.py
python data_aug.py
if errorlevel 1 (
    echo Error: data_aug.py execution failed
    pause
    exit /b 1
)
echo Completed
echo.

echo [8/8] Execute encode_dann_train.py
python encode_dann_train.py
if errorlevel 1 (
    echo Error: encode_dann_train.py execution failed
    pause
    exit /b 1
)
echo Completed
echo.


echo =========================================
echo All scripts have been executed successfully!
echo =========================================
pause