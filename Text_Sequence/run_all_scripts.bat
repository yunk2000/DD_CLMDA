@echo off
chcp 65001 >nul
echo =========================================
echo Execute all scripts in the Text_Sequence folder with one click
echo =========================================
echo.

cd /d "%~dp0"

echo Start executing scripts...
echo.

echo [1/12] Execute or_ast_generator.py
python or_ast_generator.py
if errorlevel 1 (
    echo Error: or_ast_generator.py execution failed
    pause
    exit /b 1
)
echo Completed
echo.

echo [2/12] Execute or_ast_to_sequence.py
python or_ast_to_sequence.py
if errorlevel 1 (
    echo Error: or_ast_to_sequence.py execution failed
    pause
    exit /b 1
)
echo Completed
echo.

echo [3/12] Execute or_ast_sequence_labeling.py
python or_ast_sequence_labeling.py
if errorlevel 1 (
    echo Error: or_ast_sequence_labeling.py execution failed
    pause
    exit /b 1
)
echo Completed
echo.

echo [4/12] Execute or_data_processor.py
python or_data_processor.py
if errorlevel 1 (
    echo Error: or_data_processor.py execution failed
    pause
    exit /b 1
)
echo Completed
echo.

echo [5/12] Execute or_encoder_trainer.py
python or_encoder_trainer.py
if errorlevel 1 (
    echo Error: or_encoder_trainer.py execution failed
    pause
    exit /b 1
)
echo Completed
echo.

echo [6/12] Execute ta_ast_generator.py
python ta_ast_generator.py
if errorlevel 1 (
    echo Error: ta_ast_generator.py execution failed
    pause
    exit /b 1
)
echo Completed
echo.

echo [7/12] Execute ta_ast_to_sequence.py
python ta_ast_to_sequence.py
if errorlevel 1 (
    echo Error: ta_ast_to_sequence.py execution failed
    pause
    exit /b 1
)
echo Completed
echo.

echo [8/12] Execute ta_ast_sequence_labeling.py
python ta_ast_sequence_labeling.py
if errorlevel 1 (
    echo Error: ta_ast_sequence_labeling.py execution failed
    pause
    exit /b 1
)
echo Completed
echo.

echo [9/12] Execute ta_data_processor.py
python ta_data_processor.py
if errorlevel 1 (
    echo Error: ta_data_processor.py execution failed
    pause
    exit /b 1
)
echo Completed
echo.

echo [10/12] Execute ta_encoder_trainer.py
python ta_encoder_trainer.py
if errorlevel 1 (
    echo Error: ta_encoder_trainer.py execution failed
    pause
    exit /b 1
)
echo Completed
echo.

echo [11/12] Execute data_augmentation.py
python data_augmentation.py
if errorlevel 1 (
    echo Error: data_augmentation.py execution failed
    pause
    exit /b 1
)
echo Completed
echo.

echo [12/12] Execute dann_encoder_trainer.py
python dann_encoder_trainer.py
if errorlevel 1 (
    echo Error: dann_encoder_trainer.py execution failed
    pause
    exit /b 1
)
echo Completed
echo.



echo =========================================
echo All scripts have been executed successfully!
echo =========================================
pause