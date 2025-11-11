# -*- coding: utf-8 -*-
"""
App Streamlit: Optimizer usando Modelo Pooled (R² = 0.90) + Análisis ROAS/ROI
Con análisis de saturación y punto óptimo de inversión
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.graph_objects as go
import plotly.express as px
from scipy.optimize import differential_evolution, NonlinearConstraint

# Configuración
st.set_page_config(
    page_title="Optimizer - Modelo Pooled + ROAS",
    page_icon="📊",
    layout="wide"
)

# =============================================================================
# FUNCIONES DE CARGA
# =============================================================================

@st.cache_resource
def load_pooled_model():
    """Carga el modelo pooled del notebook 2"""
    try:
        with open("modelo_notebook2.pkl", "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        return None

@st.cache_resource
def load_curvas_hill():
    """Carga curvas Hill por cliente"""
    try:
        with open("curvas_hill_por_cliente.pkl", "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        return None

@st.cache_data
def load_data():
    """Carga dataset limpio"""
    try:
        df = pd.read_csv("dataset_limpio_sin_multicolinealidad.csv")
        if "Fecha" in df.columns:
            df["Fecha"] = pd.to_datetime(df["Fecha"])
        return df
    except FileNotFoundError:
        return None

def get_ticket_usd(df, cliente):
    """Extrae ticket promedio USD del cliente desde el dataset"""
    df_cliente = df[df['empresa'] == cliente]
    
    # Intentar con diferentes columnas de ticket en orden de preferencia
    for col in ['ticket_usd', 'Ticket_promedio_usd', 'ticket_est_usd']:
        if col in df_cliente.columns:
            ticket = df_cliente[col].median()
            if not pd.isna(ticket) and ticket > 0:
                return ticket
    
    # Fallback: calcular desde revenue/transactions si está disponible
    if 'revenue_META' in df_cliente.columns and 'transactions_META' in df_cliente.columns:
        df_with_trans = df_cliente[df_cliente['transactions_META'] > 0]
        if len(df_with_trans) > 0:
            ticket = (df_with_trans['revenue_META'] / df_with_trans['transactions_META']).median()
            if not pd.isna(ticket) and ticket > 0:
                return ticket
    
    # Default conservador
    return 50.0

# =============================================================================
# FUNCIONES DE OPTIMIZACIÓN
# =============================================================================

def hill_scaled(x, alpha, k, beta):
    """Curva Hill escalada"""
    x = np.clip(np.asarray(x, float), 0, None)
    alpha = max(float(alpha), 1e-8)
    k = max(float(k), 1e-8)
    beta = max(float(beta), 1e-12)
    return beta * (np.power(x, alpha) / (np.power(k, alpha) + np.power(x, alpha)))

def estimate_transactions(invest_META, invest_GADS, cliente, model, df_hist, curvas_hill=None):
    """Estima transacciones usando curvas Hill del cliente o aproximación"""
    
    # Si hay curvas Hill disponibles, usarlas
    if curvas_hill and cliente in curvas_hill:
        curva_cliente = curvas_hill[cliente]
        baseline = curva_cliente["baseline"]
        
        # Calcular incremental usando curvas Hill
        META_incr = 0
        if curva_cliente["META"]:
            curva_meta = curva_cliente["META"]
            META_incr = hill_scaled(invest_META, curva_meta["alpha"], 
                                   curva_meta["k"], curva_meta["beta"])
        
        GADS_incr = 0
        if curva_cliente["GADS"]:
            curva_gads = curva_cliente["GADS"]
            GADS_incr = hill_scaled(invest_GADS, curva_gads["alpha"], 
                                   curva_gads["k"], curva_gads["beta"])
        
        return max(0, baseline + META_incr + GADS_incr)
    
    # Fallback: aproximación logarítmica (si no hay curvas)
    atribucion = model["atribucion"]
    df_cliente_attr = atribucion[atribucion["empresa"] == cliente]
    df_cliente_hist = df_hist[df_hist["empresa"] == cliente]
    
    if len(df_cliente_attr) == 0 or len(df_cliente_hist) == 0:
        return 0
    
    invest_META_hist = df_cliente_hist[df_cliente_hist["invest_META"] > 0]["invest_META"].mean()
    invest_GADS_hist = df_cliente_hist[df_cliente_hist["invest_GADS"] > 0]["invest_GADS"].mean()
    
    if pd.isna(invest_META_hist) or invest_META_hist == 0:
        invest_META_hist = 1000
    if pd.isna(invest_GADS_hist) or invest_GADS_hist == 0:
        invest_GADS_hist = 1000
    
    META_incr_hist = df_cliente_attr["META_incr"].mean()
    GADS_incr_hist = df_cliente_attr["GADS_incr"].mean()
    y_base = df_cliente_attr["y_base"].mean()
    
    ratio_META = np.log1p(invest_META) / np.log1p(invest_META_hist)
    ratio_GADS = np.log1p(invest_GADS) / np.log1p(invest_GADS_hist)
    
    META_incr = META_incr_hist * ratio_META * 0.8
    GADS_incr = GADS_incr_hist * ratio_GADS * 0.8
    
    return max(0, y_base + META_incr + GADS_incr)

def optimize_budget_roi(cliente, total_budget, ticket_usd, model, df_hist, curvas_hill=None,
                        min_invest_meta=0, min_invest_gads=0, optimize_for='profit'):
    """
    Optimiza distribución META/GADS maximizando ROI/ROAS o profit
    
    Args:
        optimize_for: 'profit' (revenue - inversión), 'roi' (ROI %), 'roas' (ROAS), 'transactions'
    """
    
    def objective(x):
        invest_META, invest_GADS = x[0], x[1]
        invest_total = invest_META + invest_GADS
        
        # Estimar transacciones
        trans = estimate_transactions(invest_META, invest_GADS, cliente, model, df_hist, curvas_hill)
        
        # Calcular revenue
        revenue = trans * ticket_usd
        
        # Según el objetivo
        if optimize_for == 'profit':
            return -(revenue - invest_total)  # Maximizar profit
        elif optimize_for == 'roi':
            roi = (revenue - invest_total) / invest_total if invest_total > 0 else 0
            return -roi  # Maximizar ROI
        elif optimize_for == 'roas':
            roas = revenue / invest_total if invest_total > 0 else 0
            return -roas  # Maximizar ROAS
        else:  # transactions
            return -trans
    
    # Constraint: x[0] + x[1] <= total_budget
    def budget_constraint_fun(x):
        return np.array([total_budget - x[0] - x[1]])
    
    nlc = NonlinearConstraint(budget_constraint_fun, 0, np.inf)
    
    result = differential_evolution(
        objective,
        bounds=[(min_invest_meta, total_budget), (min_invest_gads, total_budget)],
        constraints=(nlc,),
        seed=42,
        maxiter=100
    )
    
    invest_META_opt = result.x[0]
    invest_GADS_opt = result.x[1]
    invest_total_opt = invest_META_opt + invest_GADS_opt
    trans_opt = estimate_transactions(invest_META_opt, invest_GADS_opt, cliente, model, df_hist, curvas_hill)
    revenue_opt = trans_opt * ticket_usd
    profit_opt = revenue_opt - invest_total_opt
    roi_opt = (revenue_opt - invest_total_opt) / invest_total_opt if invest_total_opt > 0 else 0
    roas_opt = revenue_opt / invest_total_opt if invest_total_opt > 0 else 0
    
    # Caso actual (50/50)
    invest_META_actual = total_budget * 0.5
    invest_GADS_actual = total_budget * 0.5
    trans_actual = estimate_transactions(invest_META_actual, invest_GADS_actual, 
                                        cliente, model, df_hist, curvas_hill)
    revenue_actual = trans_actual * ticket_usd
    profit_actual = revenue_actual - total_budget
    roi_actual = (revenue_actual - total_budget) / total_budget if total_budget > 0 else 0
    roas_actual = revenue_actual / total_budget if total_budget > 0 else 0
    
    return {
        'invest_META_opt': invest_META_opt,
        'invest_GADS_opt': invest_GADS_opt,
        'invest_total_opt': invest_total_opt,
        'trans_opt': trans_opt,
        'revenue_opt': revenue_opt,
        'profit_opt': profit_opt,
        'roi_opt': roi_opt,
        'roas_opt': roas_opt,
        'invest_META_actual': invest_META_actual,
        'invest_GADS_actual': invest_GADS_actual,
        'trans_actual': trans_actual,
        'revenue_actual': revenue_actual,
        'profit_actual': profit_actual,
        'roi_actual': roi_actual,
        'roas_actual': roas_actual,
        'delta_trans': trans_opt - trans_actual,
        'delta_revenue': revenue_opt - revenue_actual,
        'delta_profit': profit_opt - profit_actual,
        'delta_roi': roi_opt - roi_actual,
        'delta_roas': roas_opt - roas_actual
    }

def analyze_saturation(cliente, ticket_usd, model, df_hist, curvas_hill=None, max_budget=50000, n_points=100):
    """
    Analiza punto de saturación donde ROI marginal = 0
    
    Returns dict con:
        - budgets: array de presupuestos
        - transactions: transacciones esperadas
        - revenues: revenue esperado
        - profits: profit esperado
        - rois: ROI en cada punto
        - roas: ROAS en cada punto
        - marginal_roi: ROI marginal (derivada)
        - optimal_budget: presupuesto óptimo (donde ROI marginal ≈ 0)
    """
    budgets = np.linspace(0, max_budget, n_points)
    results = []
    
    for budget in budgets:
        if budget == 0:
            results.append({
                'budget': 0,
                'trans': 0,
                'revenue': 0,
                'profit': 0,
                'roi': 0,
                'roas': 0
            })
            continue
        
        # Optimizar para este presupuesto
        opt_result = optimize_budget_roi(cliente, budget, ticket_usd, model, df_hist, 
                                        curvas_hill, optimize_for='profit')
        
        results.append({
            'budget': budget,
            'trans': opt_result['trans_opt'],
            'revenue': opt_result['revenue_opt'],
            'profit': opt_result['profit_opt'],
            'roi': opt_result['roi_opt'],
            'roas': opt_result['roas_opt']
        })
    
    df_results = pd.DataFrame(results)
    
    # Calcular ROI marginal (derivada numérica)
    marginal_roi = np.gradient(df_results['profit'], df_results['budget'])
    
    # Encontrar punto óptimo (donde ROI marginal cruza 1.0, es decir, cada $1 adicional genera $1)
    # ROI marginal = d(profit)/d(budget) = d(revenue - budget)/d(budget) = d(revenue)/d(budget) - 1
    # Queremos d(revenue)/d(budget) = 1, es decir, marginal_roi = 0
    optimal_idx = np.argmin(np.abs(marginal_roi))
    optimal_budget = df_results['budget'].iloc[optimal_idx]
    
    return {
        'budgets': df_results['budget'].values,
        'transactions': df_results['trans'].values,
        'revenues': df_results['revenue'].values,
        'profits': df_results['profit'].values,
        'rois': df_results['roi'].values,
        'roas': df_results['roas'].values,
        'marginal_roi': marginal_roi,
        'optimal_budget': optimal_budget,
        'optimal_profit': df_results['profit'].iloc[optimal_idx],
        'optimal_roi': df_results['roi'].iloc[optimal_idx],
        'optimal_roas': df_results['roas'].iloc[optimal_idx]
    }

# =============================================================================
# INTERFAZ PRINCIPAL
# =============================================================================

st.title("📊 Optimizer de Inversión Publicitaria + ROAS/ROI")
st.markdown("### Modelo Pooled con Análisis de Saturación")

# Sidebar
st.sidebar.title("🎯 Navegación")
page = st.sidebar.radio("Selecciona una página:", 
                        ["📁 Datos", "🤖 Modelo Pooled", "💰 Optimizar Presupuesto", 
                         "📉 Análisis de Saturación", "📈 Dashboards"])

# Cargar datos
df = load_data()
model = load_pooled_model()
curvas_hill = load_curvas_hill()

# =============================================================================
# PÁGINA 1: DATOS
# =============================================================================

if page == "📁 Datos":
    st.header("📁 Datos de Clientes")
    
    if df is None:
        st.error("❌ No se encontró 'dataset_limpio_sin_multicolinealidad.csv'")
        st.stop()
    
    st.success(f"✅ Dataset cargado: {df.shape[0]} observaciones, {df.shape[1]} columnas")
    
    # Resumen
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("📊 Clientes", df['empresa'].nunique())
    with col2:
        st.metric("📅 Semanas", len(df))
    with col3:
        if 'transactions_GA' in df.columns:
            st.metric("💰 Trans Totales", f"{df['transactions_GA'].sum():,.0f}")
    
    # Filtros
    st.subheader("🔍 Explorar Datos")
    
    selected_client = st.selectbox("Selecciona un cliente:", sorted(df['empresa'].unique()))
    
    df_client = df[df['empresa'] == selected_client]
    
    st.write(f"**{selected_client}**: {len(df_client)} semanas de datos")
    
    # Métricas del cliente
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        if 'invest_META' in df_client.columns:
            st.metric("💵 Invest META (USD)", f"${df_client['invest_META'].sum():,.0f}")
    with col2:
        if 'invest_GADS' in df_client.columns:
            st.metric("💵 Invest GADS (USD)", f"${df_client['invest_GADS'].sum():,.0f}")
    with col3:
        if 'transactions_GA' in df_client.columns:
            st.metric("🛒 Transacciones", f"{df_client['transactions_GA'].sum():,.0f}")
    with col4:
        ticket = get_ticket_usd(df, selected_client)
        st.metric("🎫 Ticket Promedio (USD)", f"${ticket:.2f}")
    
    # Gráfico de serie temporal
    if 'Fecha' in df_client.columns and 'transactions_GA' in df_client.columns:
        fig = px.line(df_client.sort_values('Fecha'), 
                     x='Fecha', y='transactions_GA', 
                     title=f"Transacciones - {selected_client}")
        st.plotly_chart(fig, use_container_width=True)
    
    # Tabla de datos
    with st.expander("📊 Ver datos del cliente"):
        st.dataframe(df_client)

# =============================================================================
# PÁGINA 2: MODELO POOLED
# =============================================================================

elif page == "🤖 Modelo Pooled":
    st.header("🤖 Modelo Pooled (R² = 0.90)")
    
    if model is None:
        st.error("❌ Modelo pooled no encontrado")
        st.info("Ejecuta: `python cargar_modelo_notebook.py`")
        st.stop()
    
    st.success("✅ Modelo pooled cargado exitosamente")
    
    # Métricas del modelo
    st.subheader("📊 Métricas del Modelo")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        r2_test = model['metrics']['r2_test']
        st.metric("R² Test", f"{r2_test:.4f}", 
                 "Excelente" if r2_test > 0.8 else "Bueno")
    with col2:
        st.metric("R² Train", f"{model['metrics']['r2_train']:.4f}")
    with col3:
        st.metric("R² Valid", f"{model['metrics']['r2_valid']:.4f}")
    with col4:
        st.metric("RMSE Test", f"{model['metrics']['rmse_test']:.1f}")
    
    # Coeficientes
    st.subheader("🔑 Coeficientes de Medios")
    
    col1, col2 = st.columns(2)
    with col1:
        beta_meta = model['coeficientes']['beta_META']
        st.metric("β(META)", f"{beta_meta:+.4f}", 
                 "✅ Positivo" if beta_meta > 0 else "❌ Negativo")
    with col2:
        beta_gads = model['coeficientes']['beta_GADS']
        st.metric("β(GADS)", f"{beta_gads:+.4f}",
                 "✅ Positivo" if beta_gads > 0 else "❌ Negativo")
    
    st.info(f"""
    **Interpretación**:
    - El modelo fue entrenado con **{model['n_obs']:,} observaciones** de **{model['n_empresas']} empresas**
    - Usa transformaciones avanzadas: Adstock (θ={model['transform_params']['theta_meta']}) + Saturación Hill (α={model['transform_params']['alpha_meta']})
    - R² test = {r2_test:.4f} indica que el modelo explica **{r2_test*100:.1f}% de la varianza** en test
    - GADS tiene un efecto {"mayor" if beta_gads > beta_meta else "menor"} que META
    """)
    
    # Clientes en el modelo
    st.subheader("📁 Clientes Incluidos")
    
    empresas_df = pd.DataFrame({
        'Cliente': model['empresas']
    })
    
    if 'atribucion' in model:
        atrib = model['atribucion']
        empresas_df['Observaciones'] = empresas_df['Cliente'].apply(
            lambda x: len(atrib[atrib['empresa'] == x])
        )
        empresas_df['Trans Promedio'] = empresas_df['Cliente'].apply(
            lambda x: atrib[atrib['empresa'] == x]['y_real'].mean()
        )
        empresas_df = empresas_df.sort_values('Trans Promedio', ascending=False)
    
    st.dataframe(empresas_df, use_container_width=True)

# =============================================================================
# PÁGINA 3: OPTIMIZAR PRESUPUESTO (CON ROI)
# =============================================================================

elif page == "💰 Optimizar Presupuesto":
    st.header("💰 Optimizar Presupuesto Publicitario + ROI")
    
    if model is None:
        st.error("❌ Modelo no encontrado. Ejecuta: `python cargar_modelo_notebook.py`")
        st.stop()
    
    if df is None:
        st.error("❌ Dataset no encontrado")
        st.stop()
    
    # Info sobre curvas Hill
    if curvas_hill:
        st.success(f"✅ Usando **curvas Hill individuales** para {len(curvas_hill)} clientes (más preciso)")
    else:
        st.warning("⚠️ No se encontraron curvas Hill. Usando aproximación logarítmica (menos preciso)")
        st.info("💡 Para mejor precisión, ejecuta: `python ajustar_curvas_por_cliente.py`")
    
    st.info("💡 El optimizer encuentra la mejor distribución META/GADS maximizando profit (revenue - inversión)")
    
    # Seleccionar cliente
    st.subheader("1️⃣ Selecciona Cliente")
    selected_client = st.selectbox("Cliente:", sorted(model['empresas']))
    
    # Obtener ticket USD del cliente
    ticket_default = get_ticket_usd(df, selected_client)
    
    # Mostrar info del cliente
    col1, col2, col3 = st.columns(3)
    with col1:
        if 'atribucion' in model:
            atrib_cliente = model['atribucion'][model['atribucion']['empresa'] == selected_client]
            st.metric("Trans Promedio/Semana", f"{atrib_cliente['y_real'].mean():.1f}")
    with col2:
        st.metric("Ticket Promedio (USD)", f"${ticket_default:.2f}")
    with col3:
        df_cliente = df[df['empresa'] == selected_client]
        if 'invest_total_paid' in df_cliente.columns:
            invest_hist = df_cliente['invest_total_paid'].mean()
            st.metric("Invest Histórico/Semana", f"${invest_hist:,.0f}")
    
    # Permitir editar ticket USD
    st.subheader("2️⃣ Parámetros Económicos")
    col1, col2 = st.columns(2)
    with col1:
        ticket_usd = st.number_input(
            "Ticket Promedio (USD) - Editable",
            min_value=1.0,
            value=float(ticket_default),
            step=1.0,
            help="Extraído del dataset. Puedes modificarlo si conoces un valor más preciso."
        )
    with col2:
        optimize_for = st.selectbox(
            "Optimizar para:",
            options=['profit', 'roi', 'roas', 'transactions'],
            index=0,
            help="profit = maximizar revenue - inversión | roi = maximizar ROI % | roas = maximizar ROAS | transactions = maximizar transacciones"
        )
    
    # Mostrar curvas Hill del cliente
    if curvas_hill and selected_client in curvas_hill:
        with st.expander("📈 Ver Curvas Hill del Cliente"):
            curva = curvas_hill[selected_client]
            
            col1, col2 = st.columns(2)
            
            with col1:
                if curva["META"]:
                    meta_params = curva["META"]
                    st.markdown("**META:**")
                    st.write(f"- α (forma): {meta_params['alpha']:.2f}")
                    st.write(f"- k (saturación): ${meta_params['k']:,.0f}")
                    st.write(f"- β (máximo): {meta_params['beta']:.2f}")
                    st.write(f"- R²: {meta_params['r2']:.3f}")
                else:
                    st.warning("Sin curva META")
            
            with col2:
                if curva["GADS"]:
                    gads_params = curva["GADS"]
                    st.markdown("**GADS:**")
                    st.write(f"- α (forma): {gads_params['alpha']:.2f}")
                    st.write(f"- k (saturación): ${gads_params['k']:,.0f}")
                    st.write(f"- β (máximo): {gads_params['beta']:.2f}")
                    st.write(f"- R²: {gads_params['r2']:.3f}")
                else:
                    st.warning("Sin curva GADS")
            
            # Visualización de las curvas
            if curva["META"] or curva["GADS"]:
                fig = go.Figure()
                
                # Rango de inversión para visualizar
                max_invest = 10000
                x_range = np.linspace(0, max_invest, 200)
                
                if curva["META"]:
                    y_meta = hill_scaled(x_range, curva["META"]["alpha"], 
                                       curva["META"]["k"], curva["META"]["beta"])
                    fig.add_trace(go.Scatter(x=x_range, y=y_meta, mode='lines',
                                            name='META', line=dict(color='#3498db', width=3)))
                
                if curva["GADS"]:
                    y_gads = hill_scaled(x_range, curva["GADS"]["alpha"], 
                                       curva["GADS"]["k"], curva["GADS"]["beta"])
                    fig.add_trace(go.Scatter(x=x_range, y=y_gads, mode='lines',
                                            name='GADS', line=dict(color='#e74c3c', width=3)))
                
                fig.update_layout(
                    title=f"Curvas de Respuesta - {selected_client}",
                    xaxis_title="Inversión Semanal (USD)",
                    yaxis_title="Transacciones Incrementales",
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)
    
    # Definir presupuesto
    st.subheader("3️⃣ Define Presupuesto")
    
    total_budget = st.number_input(
        "Presupuesto total semanal (USD)",
        min_value=0.0,
        value=5000.0,
        step=100.0
    )
    
    col1, col2 = st.columns(2)
    with col1:
        min_meta = st.number_input("Mínimo META (USD)", min_value=0.0, value=0.0, step=100.0)
    with col2:
        min_gads = st.number_input("Mínimo GADS (USD)", min_value=0.0, value=0.0, step=100.0)
    
    # Optimizar
    st.subheader("4️⃣ Optimizar")
    
    if st.button("🎯 Calcular Óptimo", type="primary"):
        with st.spinner("Optimizando..."):
            result = optimize_budget_roi(
                selected_client,
                total_budget,
                ticket_usd,
                model,
                df,
                curvas_hill,
                min_meta,
                min_gads,
                optimize_for
            )
            
            if result:
                st.success("✅ Optimización completada")
                
                # ============================================================
                # VALIDACIÓN ESPECIAL: INVERSIÓN = 0 (ANTES DE MOSTRAR MÉTRICAS)
                # ============================================================
                
                if result['invest_total_opt'] < 0.01:  # Detectar $0 o muy cerca de $0
                    st.error("⚠️ **CASO ESPECIAL: Inversión Recomendada = $0**")
                    st.warning(f"""
                    **El optimizer recomienda NO invertir nada en medios pagos.**
                    
                    **¿Por qué?**
                    
                    El modelo estima que el **baseline orgánico** ({result['trans_opt']:.0f} trans/semana) 
                    es tan alto que el incremental de META/GADS no justifica la inversión.
                    
                    **Análisis:**
                    - 🎯 Transacciones orgánicas (baseline): **{result['trans_opt']:.0f}**
                    - 💰 Revenue orgánico: **${result['revenue_opt']:,.0f} USD/semana**
                    - 📈 Incremental META/GADS estimado: Muy bajo
                    - 📊 Trans promedio histórico: {df[df['empresa']==selected_client]['transactions_GA'].mean():.1f}
                    """)
                    
                    st.info("""
                    **¿Es esto correcto?**
                    
                    **Probablemente NO.** Esto indica que:
                    
                    1. ❌ **Faltan datos de períodos SIN inversión**
                       - El modelo necesita semanas con $0 de inversión para calibrar el baseline real
                       - Sin estos datos, el modelo sobrestima el tráfico orgánico
                    
                    2. ❌ **El baseline está sobreestimado**
                       - El modelo asigna casi todas las transacciones al baseline
                       - El incremental real es mayor del estimado
                    
                    3. ❌ **Las curvas Hill no capturan bien el efecto incremental**
                       - La separación baseline vs incremental es incorrecta
                    
                    **¿Qué hacer?**
                    
                    1. ⚠️ **NO sigas esta recomendación literalmente**
                       - Continúa invirtiendo según tus datos históricos
                    
                    2. ✅ **Revisa datos históricos** (pestaña "Datos")
                       - ¿Hubo períodos SIN inversión? (para medir baseline real)
                       - Si siempre invertiste, el modelo no puede separar bien
                    
                    3. ✅ **Considera experimentación controlada**
                       - Prueba 2-3 semanas con $0 de inversión
                       - Esto calibrará el baseline real
                    
                    4. ✅ **Usa datos históricos como guía**
                       - Tu inversión histórica promedio funciona
                       - No dejes de invertir basándote solo en este resultado
                    
                    **Nota técnica:** ROI y ROAS no se muestran cuando inversión = $0 
                    porque serían matemáticamente infinitos (división por 0).
                    """)
                    
                    # Mostrar solo métricas básicas (sin ROI/ROAS)
                    st.subheader("📊 Resultados (sin inversión)")
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric(
                            "Inversión Óptima",
                            "$0",
                            "⚠️ NO confiable"
                        )
                    with col2:
                        st.metric(
                            "Transacciones (solo baseline)",
                            f"{result['trans_opt']:.0f}",
                            "Sin incremental"
                        )
                    with col3:
                        st.metric(
                            "Revenue (solo orgánico)",
                            f"${result['revenue_opt']:,.0f}",
                            "Sin inversión"
                        )
                    
                    st.error("❌ **NO uses estos resultados para tomar decisiones de inversión**")
                    
                    # Salir sin mostrar más detalles
                    st.stop()
                
                # ============================================================
                # MÉTRICAS NORMALES (solo si inversión > 0)
                # ============================================================
                
                st.subheader("📊 Resultados")
                
                col1, col2, col3, col4, col5 = st.columns(5)
                with col1:
                    st.metric(
                        "META Óptimo",
                        f"${result['invest_META_opt']:,.0f}",
                        f"{result['invest_META_opt']/total_budget*100:.0f}%"
                    )
                with col2:
                    st.metric(
                        "GADS Óptimo",
                        f"${result['invest_GADS_opt']:,.0f}",
                        f"{result['invest_GADS_opt']/total_budget*100:.0f}%"
                    )
                with col3:
                    st.metric(
                        "Revenue (USD)",
                        f"${result['revenue_opt']:,.0f}",
                        f"${result['delta_revenue']:+,.0f}"
                    )
                with col4:
                    st.metric(
                        "ROI",
                        f"{result['roi_opt']*100:.1f}%",
                        f"{result['delta_roi']*100:+.1f}pp"
                    )
                with col5:
                    st.metric(
                        "ROAS",
                        f"{result['roas_opt']:.2f}x",
                        f"{result['delta_roas']:+.2f}x"
                    )
                
                # Segunda fila de métricas
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric(
                        "Transacciones",
                        f"{result['trans_opt']:.0f}",
                        f"{result['delta_trans']:+.0f}"
                    )
                with col2:
                    st.metric(
                        "Profit (USD)",
                        f"${result['profit_opt']:,.0f}",
                        f"${result['delta_profit']:+,.0f}"
                    )
                with col3:
                    cpa_opt = result['invest_total_opt'] / result['trans_opt'] if result['trans_opt'] > 0 else 0
                    st.metric(
                        "CPA (USD)",
                        f"${cpa_opt:.2f}",
                        f"vs ticket ${ticket_usd:.2f}"
                    )
                
                # ============================================================
                # VALIDACIONES DE CONFIABILIDAD
                # ============================================================
                
                warnings = []
                is_reliable = True
                
                # 1. Verificar R² de curvas Hill
                if curvas_hill and selected_client in curvas_hill:
                    curva = curvas_hill[selected_client]
                    
                    r2_meta = curva["META"]["r2"] if curva["META"] else None
                    r2_gads = curva["GADS"]["r2"] if curva["GADS"] else None
                    
                    if r2_meta and r2_meta < 0.70:
                        warnings.append(f"⚠️ R² curva META = {r2_meta:.3f} (< 0.70) - Curva NO confiable")
                        is_reliable = False
                    
                    if r2_gads and r2_gads < 0.70:
                        warnings.append(f"⚠️ R² curva GADS = {r2_gads:.3f} (< 0.70) - Curva NO confiable")
                        is_reliable = False
                    
                    if not curva["META"] and not curva["GADS"]:
                        warnings.append(f"❌ No hay curvas Hill para {selected_client}")
                        is_reliable = False
                
                # 2. Sanity checks (solo si R² es bajo o valores extremos)
                # Si R² de curvas es bueno (> 0.70), confiar más en los resultados
                r2_meta_val = None
                r2_gads_val = None
                if curvas_hill and selected_client in curvas_hill:
                    curva = curvas_hill[selected_client]
                    r2_meta_val = curva["META"]["r2"] if curva["META"] else None
                    r2_gads_val = curva["GADS"]["r2"] if curva["GADS"] else None
                
                # Si ambas curvas tienen R² alto, permitir métricas más agresivas
                has_good_curves = (r2_meta_val and r2_meta_val > 0.70) or (r2_gads_val and r2_gads_val > 0.70)
                
                # Umbrales ajustados según calidad de curvas
                if has_good_curves:
                    # Con curvas buenas, umbrales más relajados (marketing digital puede ser muy eficiente)
                    roi_threshold = 50.0  # 5000% (50x)
                    roas_threshold = 100.0  # 100x
                    cpa_min_ratio = 0.05  # 5% del ticket
                    invest_min_ratio = 0.02  # 2% del presupuesto
                else:
                    # Con curvas malas, umbrales más estrictos
                    roi_threshold = 5.0  # 500%
                    roas_threshold = 10.0  # 10x
                    cpa_min_ratio = 0.2  # 20% del ticket
                    invest_min_ratio = 0.1  # 10% del presupuesto
                
                if result['roi_opt'] > roi_threshold:
                    warnings.append(f"🚨 ROI = {result['roi_opt']*100:.0f}% es extremadamente alto (> {roi_threshold*100:.0f}%)")
                    is_reliable = False
                
                if result['roas_opt'] > roas_threshold:
                    warnings.append(f"🚨 ROAS = {result['roas_opt']:.1f}x es extremadamente alto (> {roas_threshold:.0f}x)")
                    is_reliable = False
                
                if cpa_opt > 0 and cpa_opt < ticket_usd * cpa_min_ratio:
                    warnings.append(f"🚨 CPA = ${cpa_opt:.2f} es muy bajo comparado con ticket ${ticket_usd:.2f} (< {cpa_min_ratio*100:.0f}%)")
                    is_reliable = False
                
                if result['invest_total_opt'] < total_budget * invest_min_ratio:
                    warnings.append(f"🚨 Inversión recomendada (${result['invest_total_opt']:,.0f}) es muy baja vs presupuesto (${total_budget:,.0f}) (< {invest_min_ratio*100:.0f}%)")
                    is_reliable = False
                
                # Mostrar alertas si hay problemas
                if warnings:
                    st.error("⚠️ **RESULTADOS NO CONFIABLES**")
                    st.warning("""
                    **Los resultados NO son confiables debido a:**
                    """)
                    for warning in warnings:
                        st.markdown(f"- {warning}")
                    
                    st.info(f"""
                    **¿Por qué pasa esto?**
                    
                    - **R² bajo de curva Hill**: La curva no ajusta bien los datos históricos del cliente
                    - **Pocos datos**: El cliente tiene pocas observaciones con inversión
                    - **Alta variabilidad**: Los datos históricos son muy variables
                    - **Valores extremos**: Las métricas superan umbrales realistas
                    
                    **¿Qué hacer?**
                    
                    1. ✅ **Usa el modelo pooled** (R² = 0.90) en lugar de curvas individuales
                    2. ✅ **Revisa datos históricos** del cliente (pestaña "Datos")
                    3. ✅ **Incrementa inversión gradualmente** para generar más datos
                    4. ⚠️ **NO confíes en estos números** para tomar decisiones
                    
                    **Umbrales de confiabilidad:**
                    - R² > 0.70 → Curva confiable
                    - R² 0.50-0.70 → Usar con precaución
                    - R² < 0.50 → NO usar
                    
                    **Umbrales de métricas** (aplicados según R²):
                    - Con R² > 0.70: ROI < 5000%, ROAS < 100x (marketing digital eficiente)
                    - Con R² < 0.70: ROI < 500%, ROAS < 10x (valores conservadores)
                    """)
                
                # Indicador de confiabilidad
                if is_reliable:
                    st.success("✅ Resultados confiables - R² de curvas Hill > 0.70")
                else:
                    st.error(f"❌ Resultados NO confiables - Revisar alertas arriba")
                
                # Gráficos
                col1, col2 = st.columns(2)
                
                with col1:
                    fig1 = go.Figure()
                    fig1.add_trace(go.Bar(name='META', 
                                         x=['Actual (50/50)', 'Óptimo'], 
                                         y=[result['invest_META_actual'], result['invest_META_opt']],
                                         marker_color='#3498db'))
                    fig1.add_trace(go.Bar(name='GADS', 
                                         x=['Actual (50/50)', 'Óptimo'], 
                                         y=[result['invest_GADS_actual'], result['invest_GADS_opt']],
                                         marker_color='#e74c3c'))
                    fig1.update_layout(title='Distribución de Inversión (USD)',
                                      yaxis_title='Inversión (USD)',
                                      barmode='stack', height=400)
                    st.plotly_chart(fig1, use_container_width=True)
                
                with col2:
                    fig2 = go.Figure()
                    fig2.add_trace(go.Bar(
                        x=['Actual', 'Óptimo'],
                        y=[result['profit_actual'], result['profit_opt']],
                        marker_color=['#95a5a6', '#2ecc71'],
                        text=[f"${result['profit_actual']:,.0f}", f"${result['profit_opt']:,.0f}"],
                        textposition='outside'
                    ))
                    fig2.update_layout(title='Profit Esperado (USD)',
                                      yaxis_title='Profit (USD)', height=400)
                    st.plotly_chart(fig2, use_container_width=True)
                
                # Comparación ROI/ROAS
                col1, col2 = st.columns(2)
                
                with col1:
                    fig3 = go.Figure()
                    fig3.add_trace(go.Bar(
                        x=['Actual', 'Óptimo'],
                        y=[result['roi_actual']*100, result['roi_opt']*100],
                        marker_color=['#e67e22', '#27ae60'],
                        text=[f"{result['roi_actual']*100:.1f}%", f"{result['roi_opt']*100:.1f}%"],
                        textposition='outside'
                    ))
                    fig3.update_layout(title='ROI (%)',
                                      yaxis_title='ROI (%)', height=400)
                    st.plotly_chart(fig3, use_container_width=True)
                
                with col2:
                    fig4 = go.Figure()
                    fig4.add_trace(go.Bar(
                        x=['Actual', 'Óptimo'],
                        y=[result['roas_actual'], result['roas_opt']],
                        marker_color=['#e67e22', '#27ae60'],
                        text=[f"{result['roas_actual']:.2f}x", f"{result['roas_opt']:.2f}x"],
                        textposition='outside'
                    ))
                    fig4.update_layout(title='ROAS',
                                      yaxis_title='ROAS (x)', height=400)
                    st.plotly_chart(fig4, use_container_width=True)
                
                # Recomendación
                st.subheader("💡 Recomendación")
                
                profit_color = "green" if result['profit_opt'] > 0 else "red"
                
                st.markdown(f"""
                **Para {selected_client}:**
                
                - **Presupuesto total:** ${total_budget:,.0f} USD
                - **Distribución óptima:** ${result['invest_META_opt']:,.0f} META ({result['invest_META_opt']/total_budget*100:.0f}%) + ${result['invest_GADS_opt']:,.0f} GADS ({result['invest_GADS_opt']/total_budget*100:.0f}%)
                - **Revenue esperado:** ${result['revenue_opt']:,.0f} USD ({result['trans_opt']:.0f} trans × ${ticket_usd:.2f})
                - **Profit esperado:** <span style="color:{profit_color}">**${result['profit_opt']:,.0f} USD**</span>
                - **ROI:** {result['roi_opt']*100:.1f}% | **ROAS:** {result['roas_opt']:.2f}x
                - **Mejora vs 50/50:** +{result['delta_profit']:,.0f} USD profit ({result['delta_roi']*100:+.1f}pp ROI)
                
                ✅ Modelo confiable: R² = {model['metrics']['r2_test']:.4f}
                """, unsafe_allow_html=True)
                
                if result['profit_opt'] < 0:
                    st.warning(f"⚠️ **Profit negativo:** Con este presupuesto y ticket, se espera pérdida. Considera: 1) Reducir presupuesto, 2) Aumentar ticket promedio, 3) Mejorar eficiencia de campaña.")

# =============================================================================
# PÁGINA 4: ANÁLISIS DE SATURACIÓN
# =============================================================================

elif page == "📉 Análisis de Saturación":
    st.header("📉 Análisis de Saturación + Punto Óptimo de Inversión")
    
    if model is None or df is None:
        st.error("❌ Faltan modelo o datos")
        st.stop()
    
    st.info("""
    💡 **¿Qué es el Punto de Saturación?**
    
    Es el nivel de inversión donde **invertir $1 adicional genera menos de $1 de revenue**. 
    Más allá de este punto, el ROI marginal es negativo y **no conviene seguir invirtiendo**.
    
    Esta sección te muestra:
    - Curvas de ROI y ROAS vs presupuesto
    - Punto óptimo donde maximizas profit
    - Presupuesto máximo recomendado
    """)
    
    # Seleccionar cliente
    st.subheader("1️⃣ Selecciona Cliente")
    selected_client = st.selectbox("Cliente:", sorted(model['empresas']), key='sat_client')
    
    # Obtener ticket USD
    ticket_default = get_ticket_usd(df, selected_client)
    
    col1, col2 = st.columns(2)
    with col1:
        ticket_usd = st.number_input(
            "Ticket Promedio (USD)",
            min_value=1.0,
            value=float(ticket_default),
            step=1.0,
            key='sat_ticket'
        )
    with col2:
        max_budget_analysis = st.number_input(
            "Presupuesto máximo a analizar (USD)",
            min_value=1000.0,
            value=20000.0,
            step=1000.0,
            key='sat_max_budget'
        )
    
    # Ejecutar análisis
    if st.button("🔍 Analizar Saturación", type="primary"):
        with st.spinner("Analizando curva de saturación..."):
            sat_analysis = analyze_saturation(
                selected_client,
                ticket_usd,
                model,
                df,
                curvas_hill,
                max_budget=max_budget_analysis,
                n_points=50
            )
            
            # Resultados principales
            st.success("✅ Análisis completado")
            
            st.subheader("📊 Punto Óptimo de Inversión")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric(
                    "Presupuesto Óptimo",
                    f"${sat_analysis['optimal_budget']:,.0f}",
                    "USD/semana"
                )
            with col2:
                st.metric(
                    "Profit Máximo",
                    f"${sat_analysis['optimal_profit']:,.0f}",
                    "USD"
                )
            with col3:
                st.metric(
                    "ROI Óptimo",
                    f"{sat_analysis['optimal_roi']*100:.1f}%",
                    "percent"
                )
            with col4:
                st.metric(
                    "ROAS Óptimo",
                    f"{sat_analysis['optimal_roas']:.2f}x",
                    "times"
                )
            
            # ============================================================
            # VALIDACIONES DE CONFIABILIDAD
            # ============================================================
            
            warnings_sat = []
            is_reliable_sat = True
            
            # 1. Verificar R² de curvas Hill
            if curvas_hill and selected_client in curvas_hill:
                curva = curvas_hill[selected_client]
                
                r2_meta = curva["META"]["r2"] if curva["META"] else None
                r2_gads = curva["GADS"]["r2"] if curva["GADS"] else None
                
                if r2_meta and r2_meta < 0.70:
                    warnings_sat.append(f"⚠️ R² curva META = {r2_meta:.3f} (< 0.70) - Análisis de saturación NO confiable")
                    is_reliable_sat = False
                
                if r2_gads and r2_gads < 0.70:
                    warnings_sat.append(f"⚠️ R² curva GADS = {r2_gads:.3f} (< 0.70) - Análisis de saturación NO confiable")
                    is_reliable_sat = False
            
            # 2. Sanity checks (ajustados según calidad de curvas)
            # Si curvas tienen R² alto, permitir métricas más agresivas
            has_good_curves_sat = (r2_meta and r2_meta > 0.70) or (r2_gads and r2_gads > 0.70)
            
            if has_good_curves_sat:
                roi_threshold_sat = 50.0  # 5000%
                roas_threshold_sat = 100.0  # 100x
            else:
                roi_threshold_sat = 5.0  # 500%
                roas_threshold_sat = 10.0  # 10x
            
            if sat_analysis['optimal_roi'] > roi_threshold_sat:
                warnings_sat.append(f"🚨 ROI óptimo = {sat_analysis['optimal_roi']*100:.0f}% es extremadamente alto (> {roi_threshold_sat*100:.0f}%)")
                is_reliable_sat = False
            
            if sat_analysis['optimal_roas'] > roas_threshold_sat:
                warnings_sat.append(f"🚨 ROAS óptimo = {sat_analysis['optimal_roas']:.1f}x es extremadamente alto (> {roas_threshold_sat:.0f}x)")
                is_reliable_sat = False
            
            # Mostrar alertas si hay problemas
            if warnings_sat:
                st.error("⚠️ **ANÁLISIS DE SATURACIÓN NO CONFIABLE**")
                st.warning("""
                **El análisis NO es confiable debido a:**
                """)
                for warning in warnings_sat:
                    st.markdown(f"- {warning}")
                
                st.info(f"""
                **¿Por qué?**
                
                El análisis de saturación depende de las curvas Hill individuales.
                Si las curvas tienen R² bajo, las predicciones de profit/ROI son incorrectas.
                
                **Para {selected_client}:**
                - R² META: {f'{r2_meta:.3f}' if r2_meta is not None else 'N/A'}
                - R² GADS: {f'{r2_gads:.3f}' if r2_gads is not None else 'N/A'}
                
                **¿Qué hacer?**
                
                1. ❌ **NO uses estos resultados** para decisiones de inversión
                2. ✅ **Revisa datos históricos** en pestaña "Datos"
                3. ✅ **Espera más observaciones** para ajustar mejor las curvas
                4. ✅ **Usa modelos pooled** como referencia general
                
                **Clientes con R² > 0.70** tienen análisis confiables.
                """)
            else:
                st.success("✅ Análisis de saturación confiable - R² de curvas Hill > 0.70")
            
            # Gráfico 1: Profit vs Presupuesto
            st.subheader("📈 Curva de Profit vs Presupuesto")
            
            fig1 = go.Figure()
            
            fig1.add_trace(go.Scatter(
                x=sat_analysis['budgets'],
                y=sat_analysis['profits'],
                mode='lines',
                name='Profit',
                line=dict(color='#2ecc71', width=3)
            ))
            
            # Marcar punto óptimo
            fig1.add_trace(go.Scatter(
                x=[sat_analysis['optimal_budget']],
                y=[sat_analysis['optimal_profit']],
                mode='markers+text',
                name='Punto Óptimo',
                marker=dict(size=15, color='red', symbol='star'),
                text=[f"${sat_analysis['optimal_budget']:,.0f}"],
                textposition='top center'
            ))
            
            # Línea en profit = 0
            fig1.add_hline(y=0, line_dash="dash", line_color="gray", 
                          annotation_text="Break-even")
            
            fig1.update_layout(
                title=f"Profit vs Presupuesto - {selected_client}",
                xaxis_title="Presupuesto Semanal (USD)",
                yaxis_title="Profit (USD)",
                hovermode='x unified',
                height=500
            )
            
            st.plotly_chart(fig1, use_container_width=True)
            
            # Gráfico 2: ROI y ROAS vs Presupuesto
            st.subheader("📊 ROI y ROAS vs Presupuesto")
            
            fig2 = go.Figure()
            
            fig2.add_trace(go.Scatter(
                x=sat_analysis['budgets'],
                y=sat_analysis['rois'] * 100,
                mode='lines',
                name='ROI (%)',
                line=dict(color='#3498db', width=3),
                yaxis='y'
            ))
            
            fig2.add_trace(go.Scatter(
                x=sat_analysis['budgets'],
                y=sat_analysis['roas'],
                mode='lines',
                name='ROAS (x)',
                line=dict(color='#e74c3c', width=3),
                yaxis='y2'
            ))
            
            # Marcar punto óptimo
            fig2.add_vline(x=sat_analysis['optimal_budget'], 
                          line_dash="dash", line_color="red",
                          annotation_text=f"Óptimo: ${sat_analysis['optimal_budget']:,.0f}")
            
            fig2.update_layout(
                title=f"ROI y ROAS vs Presupuesto - {selected_client}",
                xaxis_title="Presupuesto Semanal (USD)",
                yaxis=dict(title="ROI (%)", titlefont=dict(color='#3498db')),
                yaxis2=dict(title="ROAS (x)", overlaying='y', side='right', 
                           titlefont=dict(color='#e74c3c')),
                hovermode='x unified',
                height=500
            )
            
            st.plotly_chart(fig2, use_container_width=True)
            
            # Gráfico 3: ROI Marginal
            st.subheader("🎯 ROI Marginal (Derivada del Profit)")
            
            fig3 = go.Figure()
            
            fig3.add_trace(go.Scatter(
                x=sat_analysis['budgets'],
                y=sat_analysis['marginal_roi'],
                mode='lines',
                name='ROI Marginal',
                line=dict(color='#9b59b6', width=3),
                fill='tozeroy'
            ))
            
            # Línea en ROI marginal = 0
            fig3.add_hline(y=0, line_dash="dash", line_color="red", 
                          annotation_text="ROI Marginal = 0 (no conviene invertir más)")
            
            # Marcar punto óptimo
            fig3.add_vline(x=sat_analysis['optimal_budget'], 
                          line_dash="dash", line_color="green",
                          annotation_text=f"Óptimo: ${sat_analysis['optimal_budget']:,.0f}")
            
            fig3.update_layout(
                title=f"ROI Marginal vs Presupuesto - {selected_client}",
                xaxis_title="Presupuesto Semanal (USD)",
                yaxis_title="ROI Marginal (d(Profit)/d(Budget))",
                hovermode='x unified',
                height=500
            )
            
            st.plotly_chart(fig3, use_container_width=True)
            
            # Recomendaciones
            st.subheader("💡 Recomendaciones")
            
            # Encontrar presupuesto donde profit = 0 (break-even)
            break_even_idx = np.argmin(np.abs(sat_analysis['profits']))
            break_even_budget = sat_analysis['budgets'][break_even_idx]
            
            st.markdown(f"""
            **Análisis de Saturación para {selected_client}:**
            
            1. **Presupuesto Óptimo:** ${sat_analysis['optimal_budget']:,.0f} USD/semana
               - Este presupuesto maximiza el profit absoluto
               - Profit esperado: ${sat_analysis['optimal_profit']:,.0f} USD
               - ROI: {sat_analysis['optimal_roi']*100:.1f}% | ROAS: {sat_analysis['optimal_roas']:.2f}x
            
            2. **Break-even:** ~${break_even_budget:,.0f} USD/semana
               - Por debajo de este presupuesto, hay pérdida neta
               - Por encima, comienza a generar profit
            
            3. **Punto de Saturación:** Cuando ROI marginal ≈ 0
               - Más allá de ${sat_analysis['optimal_budget']:,.0f}, cada $1 adicional genera <$1 de revenue
               - **No se recomienda invertir más** que este monto
            
            4. **Recomendación Final:**
               - **Presupuesto mínimo:** ${break_even_budget:,.0f} USD (para no perder dinero)
               - **Presupuesto óptimo:** ${sat_analysis['optimal_budget']:,.0f} USD (maximiza profit)
               - **Presupuesto máximo:** ${sat_analysis['optimal_budget']*1.2:,.0f} USD (límite conservador)
            
            ✅ Ticket promedio usado: ${ticket_usd:.2f} USD
            """)
            
            if sat_analysis['optimal_profit'] < 0:
                st.error("""
                ⚠️ **Alerta:** El profit máximo es negativo. Esto significa que:
                - El ticket promedio es muy bajo para este cliente
                - Los costos de adquisición son muy altos
                - Se recomienda: 1) Revisar ticket promedio, 2) Mejorar eficiencia de campañas, 3) Considerar no invertir en medios pagos
                """)

# =============================================================================
# PÁGINA 5: DASHBOARDS
# =============================================================================

elif page == "📈 Dashboards":
    st.header("📈 Dashboard General")
    
    if model is None or df is None:
        st.error("❌ Faltan datos o modelo")
        st.stop()
    
    if 'atribucion' not in model:
        st.error("❌ No hay datos de atribución en el modelo")
        st.stop()
    
    atrib = model['atribucion']
    
    # KPIs generales
    st.subheader("🎯 KPIs Generales")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Clientes", len(model['empresas']))
    with col2:
        st.metric("R² Test", f"{model['metrics']['r2_test']:.4f}")
    with col3:
        st.metric("Trans Totales", f"{atrib['y_real'].sum():,.0f}")
    with col4:
        share_gads = atrib['GADS_incr'].sum() / (atrib['META_incr'].sum() + atrib['GADS_incr'].sum())
        st.metric("Share GADS", f"{share_gads*100:.0f}%")
    
    # Por cliente
    st.subheader("📊 Análisis por Cliente")
    
    cliente_stats = []
    for cliente in sorted(model['empresas']):
        atrib_cliente = atrib[atrib['empresa'] == cliente]
        ticket_cliente = get_ticket_usd(df, cliente)
        
        cliente_stats.append({
            'Cliente': cliente,
            'Observaciones': len(atrib_cliente),
            'Trans Promedio': atrib_cliente['y_real'].mean(),
            'Baseline': atrib_cliente['y_base'].mean(),
            'META Incr': atrib_cliente['META_incr'].mean(),
            'GADS Incr': atrib_cliente['GADS_incr'].mean(),
            'Ticket USD': ticket_cliente
        })
    
    df_stats = pd.DataFrame(cliente_stats).sort_values('Trans Promedio', ascending=False)
    
    # Gráficos
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.bar(df_stats.head(15), x='Cliente', y='Trans Promedio',
                    title='Top 15 Clientes por Transacciones Promedio',
                    color='Trans Promedio', color_continuous_scale='RdYlGn')
        fig.update_layout(xaxis_tickangle=-45, height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = go.Figure()
        fig.add_trace(go.Bar(name='META', x=df_stats['Cliente'].head(15), 
                            y=df_stats['META Incr'].head(15)))
        fig.add_trace(go.Bar(name='GADS', x=df_stats['Cliente'].head(15), 
                            y=df_stats['GADS Incr'].head(15)))
        fig.update_layout(title='Incremental por Canal (Top 15)',
                         barmode='group', xaxis_tickangle=-45, height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    # Gráfico de Ticket USD
    st.subheader("🎫 Ticket Promedio por Cliente (USD)")
    fig = px.bar(df_stats.head(15), x='Cliente', y='Ticket USD',
                title='Top 15 Clientes por Ticket Promedio',
                color='Ticket USD', color_continuous_scale='Blues')
    fig.update_layout(xaxis_tickangle=-45, height=400)
    st.plotly_chart(fig, use_container_width=True)
    
    # Tabla completa
    st.subheader("📋 Resumen Completo")
    st.dataframe(df_stats, use_container_width=True)

# =============================================================================
# FOOTER
# =============================================================================

st.sidebar.markdown("---")
st.sidebar.info("""
**📚 Guía Rápida:**

1. **Datos**: Visualiza datos por cliente + ticket USD
2. **Modelo Pooled**: Ve métricas del modelo (R² = 0.90)
3. **Optimizar**: Calcula mix óptimo maximizando ROI/ROAS
4. **Saturación**: Encuentra punto óptimo de inversión
5. **Dashboard**: Compara todos los clientes

**💰 Nuevas funcionalidades:**
- ✅ Análisis de ROAS y ROI
- ✅ Ticket promedio en USD (editable)
- ✅ Punto de saturación óptimo
- ✅ Recomendaciones de presupuesto máximo

**📊 R² Test**: """ + (f"{model['metrics']['r2_test']:.4f}" if model else "No cargado") + """
""")

st.sidebar.markdown("---")
st.sidebar.caption("v3.0 - Optimizer con ROAS/ROI + Análisis de Saturación")
