@echo off
REM ============================================================
REM EJECUTA LA APP CON MODELO POOLED + CURVAS HILL POR CLIENTE
REM ============================================================

echo.
echo ========================================================
echo   APP OPTIMIZER - Curvas Hill por Cliente
echo ========================================================
echo.

cd /d "%~dp0"

REM 1. Verificar modelo
if not exist "modelo_notebook2.pkl" (
    echo [1/3] Cargando modelo del notebook 2...
    python cargar_modelo_notebook.py
    if %errorlevel% neq 0 (
        echo.
        echo [ERROR] No se pudo cargar el modelo
        echo.
        pause
        exit /b 1
    )
    echo      ✓ Modelo cargado
    echo.
) else (
    echo [1/3] ✓ Modelo encontrado
    echo.
)

REM 2. Generar curvas Hill
if not exist "curvas_hill_por_cliente.pkl" (
    echo [2/3] Generando curvas Hill por cliente...
    python ajustar_curvas_por_cliente.py
    if %errorlevel% neq 0 (
        echo      ⚠ Error al generar curvas (continuando...)
    ) else (
        echo      ✓ Curvas generadas
    )
    echo.
) else (
    echo [2/3] ✓ Curvas Hill encontradas
    echo.
)

REM 3. Ejecutar app
echo [3/3] Iniciando Streamlit...
echo.
echo ========================================================
echo   La app se abrira en: http://localhost:8501
echo ========================================================
echo.

python -m streamlit run app_streamlit_pooled.py

pause

