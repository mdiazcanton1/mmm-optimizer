# -*- coding: utf-8 -*-
"""
Ajusta curvas Hill por cliente usando la atribución incremental
Esto permite optimizar presupuesto con curvas específicas de cada cliente
"""

import pandas as pd
import numpy as np
import pickle
from scipy.optimize import least_squares
import warnings

warnings.filterwarnings('ignore')

def hill_scaled(x, alpha, k, beta):
    """Curva Hill escalada"""
    x = np.clip(np.asarray(x, float), 0, None)
    alpha = max(float(alpha), 1e-8)
    k = max(float(k), 1e-8)
    beta = max(float(beta), 1e-12)
    return beta * (np.power(x, alpha) / (np.power(k, alpha) + np.power(x, alpha)))

def fit_hill_curve(x, y, alpha0=1.2, k0=None, beta0=None, min_k=1.0):
    """
    Ajusta curva Hill a datos (x, y)
    
    Args:
        x: Variable independiente (ej: inversión)
        y: Variable dependiente (ej: transacciones incrementales)
        alpha0: Valor inicial de alpha (forma de curva)
        k0: Valor inicial de k (punto medio)
        beta0: Valor inicial de beta (máximo)
        min_k: Valor mínimo aceptable para k (default: 1.0)
    
    Returns:
        dict con parámetros ajustados y métricas, o None si la curva no es válida
    """
    x = np.asarray(x, float)
    y = np.clip(np.asarray(y, float), 0, None)
    
    # Validación: Y debe tener varianza
    if np.std(y) < 1e-3:
        print(f"  ⚠️ Y casi constante (std={np.std(y):.6f}) - curva no confiable")
        return None
    
    # Valores iniciales
    if k0 is None:
        k0 = np.median(x[x > 0]) if np.any(x > 0) else 1.0
    if beta0 is None:
        beta0 = max(np.nanmax(y) if np.isfinite(np.nanmax(y)) else 1.0, 1.0)
    
    p0 = np.array([alpha0, float(k0), float(beta0)], dtype=float)
    
    # Bounds (k mínimo más realista)
    lb = np.array([0.3, min_k, 1e-6], dtype=float)
    ub = np.array([5.0, 1e12, 1e12], dtype=float)
    
    # Residuos
    def resid(p):
        a, k, b = p
        return hill_scaled(x, a, k, b) - y
    
    # Optimización
    try:
        res = least_squares(resid, p0, bounds=(lb, ub), loss="soft_l1", method="trf")
        a, k, b = map(float, res.x)
        
        # Predicciones y R²
        y_hat = hill_scaled(x, a, k, b)
        ss_res = float(np.sum((y - y_hat)**2))
        ss_tot = float(np.sum((y - y.mean())**2)) + 1e-12
        r2 = 1.0 - ss_res/ss_tot
        
        # Validaciones post-ajuste
        if r2 > 0.999:
            print(f"  ⚠️ R² = {r2:.4f} (demasiado perfecto, posible overfitting)")
            return None
        
        if k < min_k * 2:
            print(f"  ⚠️ k = ${k:.2f} (muy bajo, saturación instantánea sospechosa)")
            return None
        
        return {
            "alpha": a, 
            "k": k, 
            "beta": b, 
            "r2": r2, 
            "y_hat": y_hat, 
            "success": bool(res.success),
            "n_obs": len(x)
        }
    except Exception as e:
        print(f"  ⚠️ Error al ajustar curva: {str(e)}")
        return None

def ajustar_curvas_por_cliente():
    """Ajusta curvas Hill por cliente"""
    
    print("="*80)
    print("  AJUSTE DE CURVAS HILL POR CLIENTE")
    print("="*80)
    
    # 1. Cargar modelo y datos
    print("\n1️⃣ Cargando datos...")
    
    try:
        with open("modelo_notebook2.pkl", "rb") as f:
            model = pickle.load(f)
    except FileNotFoundError:
        print("❌ Modelo no encontrado. Ejecuta: python cargar_modelo_notebook.py")
        return None
    
    try:
        df_hist = pd.read_csv("dataset_limpio_sin_multicolinealidad.csv")
    except FileNotFoundError:
        print("❌ Dataset no encontrado")
        return None
    
    atribucion = model["atribucion"]
    
    print(f"   ✓ Modelo cargado: {len(model['empresas'])} empresas")
    print(f"   ✓ Atribución: {len(atribucion)} observaciones")
    
    # 2. Ajustar curvas por cliente
    print("\n2️⃣ Ajustando curvas por cliente...")
    
    curvas_por_cliente = {}
    resultados = []
    
    for cliente in sorted(model['empresas']):
        print(f"\n   📊 {cliente}")
        
        # Datos del cliente
        df_cliente_attr = atribucion[atribucion["empresa"] == cliente]
        df_cliente_hist = df_hist[df_hist["empresa"] == cliente]
        
        if len(df_cliente_attr) < 10 or len(df_cliente_hist) < 10:
            print(f"      ⚠️ Muy pocos datos ({len(df_cliente_attr)} obs), saltando...")
            continue
        
        # Preparar datos para META
        if "invest_META" in df_cliente_hist.columns:
            # Merge para obtener invest + incremental
            df_merge = df_cliente_hist[["Fecha", "invest_META"]].copy()
            df_merge["Fecha"] = pd.to_datetime(df_merge["Fecha"])
            
            df_attr_temp = df_cliente_attr[["Fecha", "META_incr"]].copy()
            df_attr_temp["Fecha"] = pd.to_datetime(df_attr_temp["Fecha"])
            
            df_meta = df_merge.merge(df_attr_temp, on="Fecha", how="inner")
            
            # Filtrar observaciones con inversión > 0
            df_meta = df_meta[df_meta["invest_META"] > 0]
            
            if len(df_meta) >= 5:
                x_meta = df_meta["invest_META"].values
                y_meta = df_meta["META_incr"].values
                
                curva_meta = fit_hill_curve(x_meta, y_meta)
                
                if curva_meta and curva_meta["success"]:
                    print(f"      ✅ META: α={curva_meta['alpha']:.2f}, k=${curva_meta['k']:,.0f}, β={curva_meta['beta']:.2f}, R²={curva_meta['r2']:.3f}")
                else:
                    print(f"      ❌ META: No convergió")
                    curva_meta = None
            else:
                print(f"      ⚠️ META: Muy pocos datos con inversión > 0 ({len(df_meta)})")
                curva_meta = None
        else:
            curva_meta = None
        
        # Preparar datos para GADS
        if "invest_GADS" in df_cliente_hist.columns:
            df_merge = df_cliente_hist[["Fecha", "invest_GADS"]].copy()
            df_merge["Fecha"] = pd.to_datetime(df_merge["Fecha"])
            
            df_attr_temp = df_cliente_attr[["Fecha", "GADS_incr"]].copy()
            df_attr_temp["Fecha"] = pd.to_datetime(df_attr_temp["Fecha"])
            
            df_gads = df_merge.merge(df_attr_temp, on="Fecha", how="inner")
            df_gads = df_gads[df_gads["invest_GADS"] > 0]
            
            if len(df_gads) >= 5:
                x_gads = df_gads["invest_GADS"].values
                y_gads = df_gads["GADS_incr"].values
                
                curva_gads = fit_hill_curve(x_gads, y_gads)
                
                if curva_gads and curva_gads["success"]:
                    print(f"      ✅ GADS: α={curva_gads['alpha']:.2f}, k=${curva_gads['k']:,.0f}, β={curva_gads['beta']:.2f}, R²={curva_gads['r2']:.3f}")
                else:
                    print(f"      ❌ GADS: No convergió")
                    curva_gads = None
            else:
                print(f"      ⚠️ GADS: Muy pocos datos con inversión > 0 ({len(df_gads)})")
                curva_gads = None
        else:
            curva_gads = None
        
        # Guardar curvas del cliente
        if curva_meta or curva_gads:
            # Baseline (orgánico)
            baseline = df_cliente_attr["y_base"].mean()
            
            curvas_por_cliente[cliente] = {
                "META": curva_meta,
                "GADS": curva_gads,
                "baseline": baseline
            }
            
            resultados.append({
                "cliente": cliente,
                "baseline": baseline,
                "META_alpha": curva_meta["alpha"] if curva_meta else np.nan,
                "META_k": curva_meta["k"] if curva_meta else np.nan,
                "META_beta": curva_meta["beta"] if curva_meta else np.nan,
                "META_r2": curva_meta["r2"] if curva_meta else np.nan,
                "GADS_alpha": curva_gads["alpha"] if curva_gads else np.nan,
                "GADS_k": curva_gads["k"] if curva_gads else np.nan,
                "GADS_beta": curva_gads["beta"] if curva_gads else np.nan,
                "GADS_r2": curva_gads["r2"] if curva_gads else np.nan,
            })
    
    # 3. Guardar resultados
    print("\n3️⃣ Guardando curvas...")
    
    with open("curvas_hill_por_cliente.pkl", "wb") as f:
        pickle.dump(curvas_por_cliente, f)
    
    print(f"   ✓ Curvas guardadas: curvas_hill_por_cliente.pkl")
    
    if resultados:
        df_resultados = pd.DataFrame(resultados)
        df_resultados.to_csv("curvas_hill_resumen.csv", index=False)
        print(f"   ✓ Resumen guardado: curvas_hill_resumen.csv")
    
    # 4. Estadísticas
    print("\n4️⃣ Estadísticas:")
    print(f"   Clientes totales: {len(model['empresas'])}")
    print(f"   Curvas ajustadas: {len(curvas_por_cliente)}")
    
    if resultados:
        df_res = pd.DataFrame(resultados)
        print(f"\n   R² promedio META: {df_res['META_r2'].mean():.3f}")
        print(f"   R² promedio GADS: {df_res['GADS_r2'].mean():.3f}")
        
        print(f"\n   Top 5 clientes por R² META:")
        top_meta = df_res.nlargest(5, 'META_r2')[['cliente', 'META_r2', 'META_k']]
        for _, row in top_meta.iterrows():
            print(f"      • {row['cliente']}: R²={row['META_r2']:.3f}, k=${row['META_k']:,.0f}")
    
    print("\n" + "="*80)
    print("✅ CURVAS HILL AJUSTADAS POR CLIENTE")
    print("="*80)
    
    return curvas_por_cliente

if __name__ == "__main__":
    print("\n🚀 Iniciando ajuste de curvas por cliente...\n")
    curvas = ajustar_curvas_por_cliente()
    
    if curvas:
        print(f"\n✅ Proceso completado: {len(curvas)} clientes con curvas")
        print("\n💡 Ahora puedes usar el optimizer con curvas reales por cliente")
    else:
        print("\n❌ No se pudieron ajustar curvas")

