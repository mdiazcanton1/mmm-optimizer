#!/bin/bash
# ============================================================
# EJECUTA LA APP CON MODELO POOLED + CURVAS HILL POR CLIENTE
# ============================================================

echo ""
echo "========================================================"
echo "  APP OPTIMIZER - Curvas Hill por Cliente"
echo "========================================================"
echo ""

# Cambiar al directorio del script
cd "$(dirname "$0")"

# 1. Verificar modelo
if [ ! -f "modelo_notebook2.pkl" ]; then
    echo "[1/3] Cargando modelo del notebook 2..."
    python cargar_modelo_notebook.py
    if [ $? -ne 0 ]; then
        echo ""
        echo "[ERROR] No se pudo cargar el modelo"
        echo ""
        exit 1
    fi
    echo "     ✓ Modelo cargado"
    echo ""
else
    echo "[1/3] ✓ Modelo encontrado"
    echo ""
fi

# 2. Generar curvas Hill
if [ ! -f "curvas_hill_por_cliente.pkl" ]; then
    echo "[2/3] Generando curvas Hill por cliente..."
    python ajustar_curvas_por_cliente.py
    if [ $? -ne 0 ]; then
        echo "     ⚠ Error al generar curvas (continuando...)"
    else
        echo "     ✓ Curvas generadas"
    fi
    echo ""
else
    echo "[2/3] ✓ Curvas Hill encontradas"
    echo ""
fi

# 3. Ejecutar app
echo "[3/3] Iniciando Streamlit..."
echo ""
echo "========================================================"
echo "  La app se abrira en: http://localhost:8501"
echo "========================================================"
echo ""

python -m streamlit run app_streamlit_pooled.py

