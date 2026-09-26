@echo off
chcp 65001 >nul
echo ============================================
echo  Run: запуск solution.py
echo ============================================
echo.

if not exist venv (
    echo venv не найден. Сначала запусти setup.bat
    pause
    exit /b 1
)

call venv\Scripts\activate.bat
python solution.py

echo.
echo Готово. Результат: submission.csv
pause
