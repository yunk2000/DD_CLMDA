@echo off
chcp 65001 >nul
echo =========================================
echo Execute all scripts in the Text_Sequence folder with one click
echo =========================================
echo.

cd /d "%~dp0"

echo Start executing scripts...
echo.

echo [1/8] Execute or_get_dy_label.py
python or_get_dy_label.py
if errorlevel 1 (
    echo Error: or_get_dy_label.py execution failed
    pause
    exit /b 1
)
echo Completed
echo.

echo [2/8] Execute or_data_process.py
python or_data_process.py
if errorlevel 1 (
    echo Error: or_data_process.py execution failed
    pause
    exit /b 1
)
echo Completed
echo.

echo [3/8] Execute or_transformer_encoder_dy.py
python or_transformer_encoder_dy.py
if errorlevel 1 (
    echo Error: or_transformer_encoder_dy.py execution failed
    pause
    exit /b 1
)
echo Completed
echo.

echo [4/8] Execute ta_get_dy_label.py
python ta_get_dy_label.py
if errorlevel 1 (
    echo Error: ta_get_dy_label.py execution failed
    pause
    exit /b 1
)
echo Completed
echo.

echo [5/8] Execute ta_data_process.py
python ta_data_process.py
if errorlevel 1 (
    echo Error: ta_data_process.py execution failed
    pause
    exit /b 1
)
echo Completed
echo.

echo [6/8] Execute ta_transformer_encoder_dy.py
python ta_transformer_encoder_dy.py
if errorlevel 1 (
    echo Error: ta_transformer_encoder_dy.py execution failed
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

echo [8/8] Execute encoder_dann_train.py
python encoder_dann_train.py
if errorlevel 1 (
    echo Error: encoder_dann_train.py execution failed
    pause
    exit /b 1
)
echo Completed
echo.


echo =========================================
echo All scripts have been executed successfully!
echo =========================================
pause