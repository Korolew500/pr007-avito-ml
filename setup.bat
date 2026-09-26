@echo off
chcp 65001 >nul
echo ============================================
echo  Setup: создание venv и установка зависимостей
echo ============================================
echo.

if not exist venv (
    echo [1/3] Создаю виртуальное окружение venv...
    python -m venv venv
    if errorlevel 1 (
        echo.
        echo ОШИБКА: python не найден в PATH.
        echo Установи Python 3.10+ с https://python.org и добавь в PATH.
        pause
        exit /b 1
    )
) else (
    echo [1/3] venv уже существует, пропускаю создание.
)

echo [2/3] Обновляю pip...
call venv\Scripts\activate.bat
python -m pip install --upgrade pip >nul

echo [3/3] Устанавливаю зависимости из requirements.txt...
pip install -r requirements.txt
if errorlevel 1 (
    echo.
    echo ОШИБКА при установке зависимостей.
    pause
    exit /b 1
)

echo.
echo Готово. Теперь запусти run.bat
pause
