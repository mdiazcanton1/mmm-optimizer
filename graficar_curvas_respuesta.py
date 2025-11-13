"""
Script para graficar TODAS las curvas de respuesta Hill por cliente y canal
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

# Configuración de estilo
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['figure.figsize'] = (20, 12)
plt.rcParams['font.size'] = 10

# Cargar curvas Hill
print("Cargando curvas Hill...")
with open('curvas_hill_por_cliente.pkl', 'rb') as f:
    curvas = pickle.load(f)

print(f"Total de clientes: {len(curvas)}")

# Función Hill
def hill_scaled(x, alpha, k, beta):
    x = np.clip(np.asarray(x, float), 0, None)
    alpha = max(float(alpha), 1e-8)
    k = max(float(k), 1e-8)
    beta = max(float(beta), 1e-12)
    return beta * (np.power(x, alpha) / (np.power(k, alpha) + np.power(x, alpha)))

# Preparar datos
clientes_con_curvas = []
for cliente, c in curvas.items():
    meta_ok = c['META'] and c['META']['r2'] > 0.3  # R² mínimo
    gads_ok = c['GADS'] and c['GADS']['r2'] > 0.3
    if meta_ok or gads_ok:
        clientes_con_curvas.append((cliente, c, meta_ok, gads_ok))

print(f"Clientes con curvas válidas (R² > 0.3): {len(clientes_con_curvas)}")

# Crear figura con subplots
n_clientes = len(clientes_con_curvas)
n_cols = 4
n_rows = (n_clientes + n_cols - 1) // n_cols

fig = plt.figure(figsize=(24, 6 * n_rows))
gs = gridspec.GridSpec(n_rows, n_cols, figure=fig, hspace=0.4, wspace=0.3)

# Graficar cada cliente
for idx, (cliente, c, meta_ok, gads_ok) in enumerate(clientes_con_curvas):
    row = idx // n_cols
    col = idx % n_cols
    ax = fig.add_subplot(gs[row, col])
    
    # Determinar rango del gráfico
    max_invest = 0
    if meta_ok:
        max_invest = max(max_invest, c['META']['k'] * 3)
    if gads_ok:
        max_invest = max(max_invest, c['GADS']['k'] * 3)
    
    max_invest = max(max_invest, 1000)  # Mínimo $1,000
    x_range = np.linspace(0, max_invest, 200)
    
    # Graficar META
    if meta_ok:
        meta_params = c['META']
        y_meta = hill_scaled(x_range, meta_params['alpha'], meta_params['k'], meta_params['beta'])
        ax.plot(x_range, y_meta, 'b-', linewidth=2.5, 
                label=f"META (R²={meta_params['r2']:.2f})", alpha=0.8)
        
        # Marcar punto k (saturación)
        y_at_k = hill_scaled(meta_params['k'], meta_params['alpha'], 
                            meta_params['k'], meta_params['beta'])
        ax.scatter([meta_params['k']], [y_at_k], c='blue', s=100, 
                  zorder=5, marker='o', edgecolors='darkblue', linewidths=2)
        ax.axvline(meta_params['k'], color='blue', linestyle='--', 
                  alpha=0.3, linewidth=1)
    
    # Graficar GADS
    if gads_ok:
        gads_params = c['GADS']
        y_gads = hill_scaled(x_range, gads_params['alpha'], gads_params['k'], gads_params['beta'])
        ax.plot(x_range, y_gads, 'r-', linewidth=2.5, 
                label=f"GADS (R²={gads_params['r2']:.2f})", alpha=0.8)
        
        # Marcar punto k (saturación)
        y_at_k = hill_scaled(gads_params['k'], gads_params['alpha'], 
                            gads_params['k'], gads_params['beta'])
        ax.scatter([gads_params['k']], [y_at_k], c='red', s=100, 
                  zorder=5, marker='o', edgecolors='darkred', linewidths=2)
        ax.axvline(gads_params['k'], color='red', linestyle='--', 
                  alpha=0.3, linewidth=1)
    
    # Configuración del gráfico
    ax.set_xlabel('Inversión Semanal (USD)', fontsize=10, fontweight='bold')
    ax.set_ylabel('Transacciones Incrementales', fontsize=10, fontweight='bold')
    
    # Título con baseline
    baseline_text = f"Baseline: {c['baseline']:.1f} trans"
    ax.set_title(f"{cliente}\n{baseline_text}", 
                fontsize=11, fontweight='bold', pad=10)
    
    ax.legend(loc='lower right', fontsize=9, framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)
    ax.set_xlim(0, max_invest)
    ax.set_ylim(0, None)
    
    # Formato de eje x
    ax.ticklabel_format(style='plain', axis='x')

# Título general
fig.suptitle('Curvas de Respuesta Hill por Cliente y Canal\n(Transacciones Incrementales vs Inversión Semanal)', 
            fontsize=18, fontweight='bold', y=0.995)

# Guardar
output_path = 'curvas_respuesta_todas.png'
plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
print(f"\n[OK] Grafico guardado: {output_path}")

# Mostrar
plt.show()

print("\n" + "="*70)
print("LEYENDA:")
print("="*70)
print("- Línea azul: Canal META (Facebook/Instagram)")
print("- Línea roja: Canal GADS (Google Ads)")
print("- Círculos: Punto k de saturación (donde alcanza ~63% del máximo)")
print("- Líneas punteadas verticales: Marcan el punto k de cada canal")
print("- R²: Calidad del ajuste (>0.7 = excelente, >0.5 = bueno, >0.3 = aceptable)")
print("="*70)

# Estadísticas adicionales
print("\nESTADÍSTICAS GENERALES:")
print("="*70)

meta_count = sum(1 for _, c, meta_ok, _ in clientes_con_curvas if meta_ok)
gads_count = sum(1 for _, c, _, gads_ok in clientes_con_curvas if gads_ok)

print(f"Clientes con curva META válida: {meta_count}/{len(curvas)} ({meta_count/len(curvas)*100:.1f}%)")
print(f"Clientes con curva GADS válida: {gads_count}/{len(curvas)} ({gads_count/len(curvas)*100:.1f}%)")

# Promedios de R²
r2_meta = [c['META']['r2'] for _, c, meta_ok, _ in clientes_con_curvas if meta_ok]
r2_gads = [c['GADS']['r2'] for _, c, _, gads_ok in clientes_con_curvas if gads_ok]

if r2_meta:
    print(f"\nR² promedio META: {np.mean(r2_meta):.3f} (rango: [{np.min(r2_meta):.3f}, {np.max(r2_meta):.3f}])")
if r2_gads:
    print(f"R² promedio GADS: {np.mean(r2_gads):.3f} (rango: [{np.min(r2_gads):.3f}, {np.max(r2_gads):.3f}])")

print("\n[OK] Analisis completo!")

