# 🍎 Instrucciones para Ejecutar en Mac/Linux

## ⚠️ Problema Común

Si ejecutas directamente:
```bash
python -m streamlit run app_streamlit_pooled.py
```

**La app va a dar errores** porque le faltan archivos que debe generar primero.

## ✅ Solución: Usar el Script de Ejecución

### Pasos para ejecutar correctamente:

1. **Activar el entorno virtual:**
   ```bash
   source venv_mmm/bin/activate
   ```

2. **Dar permisos de ejecución al script** (solo la primera vez):
   ```bash
   chmod +x EJECUTAR_AQUI.sh
   ```

3. **Ejecutar el script:**
   ```bash
   ./EJECUTAR_AQUI.sh
   ```

## 🔍 ¿Qué hace el script automáticamente?

El script `EJECUTAR_AQUI.sh` hace **3 pasos importantes**:

### [1/3] Verificar/Cargar Modelo
- Busca el archivo `modelo_notebook2.pkl`
- Si no existe, lo genera automáticamente ejecutando `cargar_modelo_notebook.py`
- Este archivo contiene el modelo MMM entrenado (R² = 0.90)

### [2/3] Verificar/Generar Curvas Hill
- Busca el archivo `curvas_hill_por_cliente.pkl`
- Si no existe, lo genera automáticamente ejecutando `ajustar_curvas_por_cliente.py`
- Este archivo contiene las curvas de saturación por cliente

### [3/3] Ejecutar Streamlit
- Una vez que tiene todos los archivos necesarios, lanza la app
- La app se abre en `http://localhost:8501`

## 📋 Archivos que DEBEN existir para que la app funcione:

- ✅ `modelo_notebook2.pkl` - Modelo MMM entrenado
- ✅ `curvas_hill_por_cliente.pkl` - Curvas Hill por cliente
- ✅ `dataset_limpio_sin_multicolinealidad.csv` - Dataset limpio
- ✅ `atribucion_incremental.csv` - Atribución de transacciones

Si alguno falta, el script los genera automáticamente.

## 🐛 Solución de Problemas

### Problema: "Permission denied"
```bash
chmod +x EJECUTAR_AQUI.sh
```

### Problema: "modelo_notebook2.pkl not found"
El script lo genera automáticamente, pero necesita estos archivos:
- `metricas_modelo.csv`
- `coeficientes_modelo.csv`
- `parametros_modelo.csv`
- `atribucion_incremental.csv`

Si no los tienes, ejecuta el notebook `2_Modelo_MMM.ipynb` primero.

### Problema: "dataset_limpio_sin_multicolinealidad.csv not found"
Ejecuta el notebook `1_EDA_y_Correlaciones.ipynb` primero.

## 💡 Comparación Windows vs Mac

| Acción | Windows | Mac/Linux |
|--------|---------|-----------|
| Ejecutar app | `EJECUTAR_AQUI.bat` | `./EJECUTAR_AQUI.sh` |
| Activar venv | `venv_mmm\Scripts\activate` | `source venv_mmm/bin/activate` |
| Separador de rutas | `\` | `/` |

## ✨ Resultado Esperado

Deberías ver algo como esto:

```
========================================================
  APP OPTIMIZER - Curvas Hill por Cliente
========================================================

[1/3] ✓ Modelo encontrado

[2/3] ✓ Curvas Hill encontradas

[3/3] Iniciando Streamlit...

========================================================
  La app se abrira en: http://localhost:8501
========================================================

  You can now view your Streamlit app in your browser.

  Local URL: http://localhost:8501
  Network URL: http://192.168.1.XXX:8501
```

---

**¿Dudas?** Contacta a Matías, Liam o Ariel.

