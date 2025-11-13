# -*- coding: utf-8 -*-
"""
Grid Search para encontrar los mejores parámetros de Adstock y Hill por canal
Objetivo: Maximizar R² en validación
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from scipy.signal import lfilter as recursive_filter
import warnings
import time
from itertools import product

warnings.filterwarnings('ignore')

print("="*80)
print("  GRID SEARCH: OPTIMIZACIÓN DE PARÁMETROS ADSTOCK & HILL POR CANAL")
print("="*80)

# =============================================================================
# FUNCIONES DE TRANSFORMACIÓN
# =============================================================================

def safe_zscore(s):
    """Z-score seguro (maneja std=0)"""
    s = pd.Series(s, dtype="float64")
    std = s.std()
    return (s - s.mean())/std if std and std != 0 else (s - s.mean())

def adstock(x, theta=0.6):
    """Efecto de memoria (carry-over)"""
    x = np.asarray(x, float)
    # AR(1): y[t] = x[t] + theta * y[t-1]
    # En lfilter: b=[1], a=[1, -theta]
    return recursive_filter([1], [1, -theta], x)

def hill(x, k=1.0, alpha=1.2):
    """Curva de saturación (diminishing returns)"""
    x = np.maximum(np.asarray(x, float), 0.0)
    return np.power(x, alpha) / (np.power(k, alpha) + np.power(x, alpha))

def media_response(series, theta=0.6, alpha=1.2, zscale=True):
    """Transformación completa: z-score + adstock + saturación"""
    x = np.asarray(series, float)
    if zscale:
        x = safe_zscore(pd.Series(x)).values
        x = x - x.min()  # Evita negativos
    a = adstock(x, theta=theta)
    k = np.median(a) or 1.0
    return hill(a, k=k, alpha=alpha)

# =============================================================================
# CARGAR DATOS
# =============================================================================

print("\n[1] Cargando datos...")

try:
    df = pd.read_csv("dataset_limpio_sin_multicolinealidad.csv")
    print(f"   OK Dataset cargado: {df.shape}")
except FileNotFoundError:
    print("ERROR: dataset_limpio_sin_multicolinealidad.csv no encontrado")
    exit(1)

# =============================================================================
# PREPARAR DATOS BASE
# =============================================================================

print("\n[2] Preparando datos base...")

TARGET = "transactions_GA"
EVENTS = ["evt_Hot Sale", "evt_Cyber Monday", "evt_Black Friday", "evt_Navidad"]

# Fechas
df["Fecha"] = pd.to_datetime(df["Fecha"])
df = df.sort_values(["empresa", "Fecha"]).reset_index(drop=True)

# Estacionalidad
week = df.index.map(lambda idx: pd.to_datetime(df.loc[idx, "Fecha"]).isocalendar().week)
df["sin_y"] = np.sin(2*np.pi*week/52.18)
df["cos_y"] = np.cos(2*np.pi*week/52.18)

# Eventos
for e in EVENTS:
    if e not in df.columns:
        df[e] = 0

# Dummies de empresa
empresas = sorted(df["empresa"].unique())
for emp in empresas[1:]:  # Skip first (reference)
    df[f"empresa_{emp}"] = (df["empresa"] == emp).astype(int)

print(f"   OK {len(empresas)} empresas, {len(df)} observaciones")

# =============================================================================
# GRID SEARCH
# =============================================================================

print("\n[3] Definiendo grid de busqueda...")

# Grid de parámetros (81 combinaciones = 3×3×3×3)
grid = {
    'theta_meta': [0.30, 0.40, 0.50],  # META: carry-over moderado-alto
    'alpha_meta': [0.80, 1.00, 1.20],  # META: saturación moderada
    'theta_gads': [0.10, 0.20, 0.30],  # GADS: carry-over bajo
    'alpha_gads': [1.00, 1.30, 1.60],  # GADS: saturación gradual
}

# Baseline (parámetros actuales)
baseline_params = {
    'theta_meta': 0.30,
    'alpha_meta': 1.00,
    'theta_gads': 0.30,
    'alpha_gads': 1.00
}

# Generar todas las combinaciones
param_combinations = list(product(
    grid['theta_meta'],
    grid['alpha_meta'],
    grid['theta_gads'],
    grid['alpha_gads']
))

print(f"   Total de combinaciones a probar: {len(param_combinations)}")
print(f"   Baseline: theta_META={baseline_params['theta_meta']}, alpha_META={baseline_params['alpha_meta']}, "
      f"theta_GADS={baseline_params['theta_gads']}, alpha_GADS={baseline_params['alpha_gads']}")

# =============================================================================
# EVALUAR CADA COMBINACIÓN
# =============================================================================

print("\n[4] Ejecutando grid search (esto tomara 2-5 minutos)...")

results = []
start_time = time.time()

for i, (theta_meta, alpha_meta, theta_gads, alpha_gads) in enumerate(param_combinations):
    try:
        # Copiar dataframe
        df_temp = df.copy()
        
        # Aplicar transformaciones de medios con estos parámetros
        if "impressions_META" in df_temp.columns:
            df_temp["META_resp"] = (df_temp.groupby("empresa")["impressions_META"]
                                   .transform(lambda s: media_response(s.fillna(0), 
                                              theta=theta_meta, alpha=alpha_meta, zscale=True)))
        else:
            df_temp["META_resp"] = 0.0
        
        if "impressions_GADS" in df_temp.columns:
            df_temp["GADS_resp"] = (df_temp.groupby("empresa")["impressions_GADS"]
                                   .transform(lambda s: media_response(s.fillna(0), 
                                              theta=theta_gads, alpha=alpha_gads, zscale=True)))
        else:
            df_temp["GADS_resp"] = 0.0
        
        # Features
        feature_cols = ["META_resp", "GADS_resp", "sin_y", "cos_y"] + EVENTS
        empresa_cols = [c for c in df_temp.columns if c.startswith("empresa_")]
        feature_cols += empresa_cols
        
        # Preparar X, y
        X = df_temp[feature_cols].values
        y = df_temp[TARGET].values
        
        # Split train/valid/test (60/20/20)
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X, y, test_size=0.20, random_state=42
        )
        X_train, X_valid, y_train, y_valid = train_test_split(
            X_train_val, y_train_val, test_size=0.25, random_state=42  # 0.25 * 0.80 = 0.20
        )
        
        # Escalar
        scaler = StandardScaler()
        X_train_sc = scaler.fit_transform(X_train)
        X_valid_sc = scaler.transform(X_valid)
        X_test_sc = scaler.transform(X_test)
        
        # Entrenar Ridge
        ridge = Ridge(alpha=1.0, random_state=42)
        ridge.fit(X_train_sc, y_train)
        
        # Evaluar
        r2_train = ridge.score(X_train_sc, y_train)
        r2_valid = ridge.score(X_valid_sc, y_valid)
        r2_test = ridge.score(X_test_sc, y_test)
        
        # Guardar resultados
        results.append({
            'theta_meta': theta_meta,
            'alpha_meta': alpha_meta,
            'theta_gads': theta_gads,
            'alpha_gads': alpha_gads,
            'r2_train': r2_train,
            'r2_valid': r2_valid,
            'r2_test': r2_test,
            'is_baseline': (theta_meta == baseline_params['theta_meta'] and 
                          alpha_meta == baseline_params['alpha_meta'] and
                          theta_gads == baseline_params['theta_gads'] and
                          alpha_gads == baseline_params['alpha_gads'])
        })
        
        # Progreso cada 10 iteraciones
        if (i + 1) % 10 == 0:
            elapsed = time.time() - start_time
            eta = elapsed / (i + 1) * (len(param_combinations) - i - 1)
            print(f"   Progreso: {i+1}/{len(param_combinations)} "
                  f"({(i+1)/len(param_combinations)*100:.0f}%) - "
                  f"ETA: {eta:.0f}s")
    
    except Exception as e:
        print(f"   ADVERTENCIA: Error en combinacion {i+1}: {e}")
        continue

elapsed_total = time.time() - start_time
print(f"\n   OK Grid search completado en {elapsed_total:.1f} segundos")

# =============================================================================
# ANALIZAR RESULTADOS
# =============================================================================

print("\n[5] Analizando resultados...")

df_results = pd.DataFrame(results)
df_results = df_results.sort_values('r2_valid', ascending=False)

# Guardar todos los resultados
df_results.to_csv("grid_search_resultados.csv", index=False)
print("   OK Resultados guardados: grid_search_resultados.csv")

# Mejores parámetros (según R² validación)
best = df_results.iloc[0]
baseline = df_results[df_results['is_baseline']].iloc[0]

print("\n" + "="*80)
print("  RESULTADOS DEL GRID SEARCH")
print("="*80)

print(f"\n[MEJOR COMBINACION] (R2 validacion = {best['r2_valid']:.4f}):")
print(f"   theta_META = {best['theta_meta']:.2f}  |  alpha_META = {best['alpha_meta']:.2f}")
print(f"   theta_GADS = {best['theta_gads']:.2f}  |  alpha_GADS = {best['alpha_gads']:.2f}")
print(f"\n   R2 Train: {best['r2_train']:.4f}")
print(f"   R2 Valid: {best['r2_valid']:.4f}")
print(f"   R2 Test:  {best['r2_test']:.4f}")

print(f"\n[BASELINE ACTUAL] (R2 validacion = {baseline['r2_valid']:.4f}):")
print(f"   theta_META = {baseline['theta_meta']:.2f}  |  alpha_META = {baseline['alpha_meta']:.2f}")
print(f"   theta_GADS = {baseline['theta_gads']:.2f}  |  alpha_GADS = {baseline['alpha_gads']:.2f}")
print(f"\n   R2 Train: {baseline['r2_train']:.4f}")
print(f"   R2 Valid: {baseline['r2_valid']:.4f}")
print(f"   R2 Test:  {baseline['r2_test']:.4f}")

# Mejora
mejora_valid = (best['r2_valid'] - baseline['r2_valid']) * 100
mejora_test = (best['r2_test'] - baseline['r2_test']) * 100

print(f"\n[MEJORA]:")
print(f"   R2 Valid: {mejora_valid:+.2f} puntos porcentuales")
print(f"   R2 Test:  {mejora_test:+.2f} puntos porcentuales")

if mejora_valid > 0.5:
    print(f"\n   >> Mejora significativa! Vale la pena actualizar el modelo")
elif mejora_valid > 0:
    print(f"\n   >> Mejora marginal. Evalua si vale la pena actualizar")
else:
    print(f"\n   >> No hay mejora. Los parametros actuales son optimos")

# Top 5 combinaciones
print(f"\n[TOP 5 COMBINACIONES]:")
print(f"\n{'Rank':<6}{'theta_META':<12}{'alpha_META':<12}{'theta_GADS':<12}{'alpha_GADS':<12}{'R2 Valid':<12}{'R2 Test':<12}")
print("-"*76)
for idx, row in df_results.head(5).iterrows():
    rank = df_results.index.get_loc(idx) + 1
    print(f"{rank:<6}{row['theta_meta']:<12.2f}{row['alpha_meta']:<12.2f}"
          f"{row['theta_gads']:<12.2f}{row['alpha_gads']:<12.2f}"
          f"{row['r2_valid']:<12.4f}{row['r2_test']:<12.4f}")

# =============================================================================
# GUARDAR MEJORES PARÁMETROS
# =============================================================================

print("\n[6] Guardando mejores parametros...")

best_params = {
    'THETA_META': float(best['theta_meta']),
    'ALPHA_META': float(best['alpha_meta']),
    'THETA_GADS': float(best['theta_gads']),
    'ALPHA_GADS': float(best['alpha_gads']),
    'R2_TRAIN': float(best['r2_train']),
    'R2_VALID': float(best['r2_valid']),
    'R2_TEST': float(best['r2_test']),
    'MEJORA_VALID_PP': float(mejora_valid),
    'MEJORA_TEST_PP': float(mejora_test)
}

# Guardar como CSV
pd.DataFrame([best_params]).to_csv("mejores_parametros_canales.csv", index=False)
print("   OK Mejores parametros guardados: mejores_parametros_canales.csv")

# Guardar como archivo Python (para fácil importación)
with open("mejores_parametros_canales.py", "w", encoding="utf-8") as f:
    f.write("# -*- coding: utf-8 -*-\n")
    f.write('"""\n')
    f.write("Mejores parametros encontrados por grid search\n")
    f.write(f"R2 validacion: {best['r2_valid']:.4f}\n")
    f.write(f"Mejora vs baseline: {mejora_valid:+.2f} pp\n")
    f.write('"""\n\n')
    f.write(f"THETA_META = {best['theta_meta']:.2f}\n")
    f.write(f"ALPHA_META = {best['alpha_meta']:.2f}\n")
    f.write(f"THETA_GADS = {best['theta_gads']:.2f}\n")
    f.write(f"ALPHA_GADS = {best['alpha_gads']:.2f}\n")

print("   OK Archivo Python creado: mejores_parametros_canales.py")

print("\n" + "="*80)
print(">> GRID SEARCH COMPLETADO")
print("="*80)

if mejora_valid > 0.5:
    print("\n[SIGUIENTE PASO]:")
    print("   Los nuevos parametros mejoran significativamente el modelo.")
    print("   Opcion 1: Actualizar manualmente en 2_Modelo_MMM.ipynb")
    print("   Opcion 2: Ejecutar script de re-entrenamiento automatico")
    print(f"\n   Nuevos valores:")
    print(f"   THETA_META, ALPHA_META = {best['theta_meta']:.2f}, {best['alpha_meta']:.2f}")
    print(f"   THETA_GADS, ALPHA_GADS = {best['theta_gads']:.2f}, {best['alpha_gads']:.2f}")
else:
    print("\n>> Los parametros actuales ya son optimos (o muy cercanos).")
    print("   No es necesario actualizar el modelo.")

print("="*80 + "\n")

