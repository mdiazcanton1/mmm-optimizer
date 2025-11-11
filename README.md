# Marketing Mix Modeling (MMM) - Optimizer de Inversión Publicitaria

Proyecto Final ITBA - Optimizer de presupuesto publicitario con análisis de ROI/ROAS y saturación para múltiples clientes.

## 📋 Descripción

Este proyecto implementa un **Marketing Mix Model (MMM)** que permite:
- 📊 Analizar el impacto de inversión publicitaria en transacciones
- 💰 Optimizar la distribución de presupuesto entre META y Google Ads
- 📈 Identificar puntos de saturación y ROI marginal
- 🎯 Generar curvas de respuesta Hill por cliente
- 🚀 Visualizar resultados en una app interactiva de Streamlit

## 🏗️ Estructura del Proyecto

```
📁 Proyecto Final/
├── 1_EDA_y_Correlaciones.ipynb          # Análisis exploratorio y limpieza
├── 2_Modelo_MMM.ipynb                    # Entrenamiento del modelo pooled
├── 3_Curvas_Respuesta_Optimizacion.ipynb # Generación de curvas Hill
├── app_streamlit_pooled.py               # Aplicación web interactiva
├── EJECUTAR_AQUI.bat                     # Script para ejecutar la app
├── ajustar_curvas_por_cliente.py         # Generación de curvas Hill
├── cargar_modelo_notebook.py             # Carga del modelo entrenado
├── dataset_consolidado_completo.csv      # Dataset principal
└── requirements.txt                       # Dependencias Python
```

## 🚀 Instalación y Ejecución

### 1. Clonar el repositorio

```bash
git clone https://github.com/TU_USUARIO/TU_REPO.git
cd TU_REPO
```

### 2. Crear entorno virtual

```bash
python -m venv venv_mmm
```

### 3. Activar entorno virtual

**Windows:**
```bash
venv_mmm\Scripts\activate
```

**Linux/Mac:**
```bash
source venv_mmm/bin/activate
```

### 4. Instalar dependencias

```bash
pip install -r requirements.txt
```

### 5. Ejecutar la aplicación

**Windows:**
```bash
EJECUTAR_AQUI.bat
```

**Linux/Mac:**
```bash
python -m streamlit run app_streamlit_pooled.py
```

La app se abrirá en `http://localhost:8501`

## 📊 Flujo de Trabajo

### Paso 1: EDA y Limpieza de Datos
Ejecutar `1_EDA_y_Correlaciones.ipynb`:
- Carga y limpieza de datos
- Análisis de multicolinealidad (VIF)
- Generación de `dataset_limpio_sin_multicolinealidad.csv`

### Paso 2: Entrenar Modelo MMM
Ejecutar `2_Modelo_MMM.ipynb`:
- Entrenamiento de modelo pooled
- Cálculo de atribución incremental
- Generación de `modelo_notebook2.pkl` y `atribucion_incremental.csv`

### Paso 3: Generar Curvas Hill
Ejecutar `3_Curvas_Respuesta_Optimizacion.ipynb`:
- Ajuste de curvas de respuesta por cliente
- Generación de `curvas_hill_por_cliente.pkl`

### Paso 4: Usar la App
Ejecutar `EJECUTAR_AQUI.bat`:
- La app verifica y genera archivos faltantes automáticamente
- Interfaz interactiva con 5 páginas:
  - 📁 **Datos**: Visualización de datos históricos
  - 🤖 **Modelo Pooled**: Diagnóstico del modelo
  - 💰 **Optimizar Presupuesto**: Optimización de inversión
  - 📉 **Análisis de Saturación**: Curvas de profit y ROI
  - 📈 **Dashboards**: Visualizaciones generales

## 🔧 Tecnologías Utilizadas

- **Python 3.12**
- **Streamlit**: Interfaz web interactiva
- **Pandas & NumPy**: Manipulación de datos
- **Scikit-learn**: Modelo de regresión
- **SciPy**: Optimización no lineal
- **Plotly**: Gráficos interactivos
- **Statsmodels**: Análisis estadístico

## 📈 Funcionalidades Principales

### Optimización de Presupuesto
- Distribución óptima entre META y Google Ads
- Maximización de profit (revenue - inversión)
- Restricciones personalizables por canal

### Análisis de Saturación
- Identificación de punto óptimo de inversión
- Cálculo de ROI y ROAS marginal
- Detección de sobresaturación

### Validación de Resultados
- Verificación de R² de curvas Hill
- Alertas de resultados no confiables
- Recomendaciones basadas en calidad de datos

## ⚠️ Notas Importantes

1. **Datos en USD**: Todos los valores monetarios están en dólares (conversión automática desde pesos argentinos)
2. **Modelo Pooled**: Entrenado con datos de múltiples clientes para mayor robustez
3. **Curvas Hill**: Representan saturación de respuesta a inversión publicitaria
4. **R² < 0.70**: Resultados de saturación pueden no ser confiables

## 📝 Requisitos del Sistema

- Python 3.8+
- 4GB RAM mínimo
- 500MB espacio en disco

## 🤝 Contribuciones

Este es un proyecto académico (ITBA). Para consultas o mejoras, contactar al autor.

## 📄 Licencia

Proyecto académico - ITBA 2025

---

**Autor**: Matias  
**Institución**: ITBA  
**Fecha**: Noviembre 2025

