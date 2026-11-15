import streamlit as st
import duckdb
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns

# ==============================================================================
# CONFIGURACIÓN DE PÁGINA Y ENCABEZADO EJECUTIVO
# ==============================================================================
st.set_page_config(
    page_title="Proyecto Integrador U4 - Agrotecnología",
    page_icon="🌱",
    layout="wide"
)

st.title("🌱 Optimización Inferencial del Rendimiento Agrícola y Eficiencia de Riego")
st.markdown("""
**Asignatura:** Análisis y Exploración de Datos (`INC-2505`) | **Proyecto Integrador Unidad 4**  
**Escenario:** Monitoreo inteligente de parcelas de jitomate/aguacate en invernadero mediante telemetría SQL, regresión lineal múltiple (Subtema 4.2) e inferencia estadística de tratamientos (Subtema 4.3).
""")

# ==============================================================================
# HITO 1: CARGA Y PREPARACIÓN DE TELEMETRÍA AGROINDUSTRIAL (DUCKDB)
# ==============================================================================
@st.cache_data
def generar_telemetria():
    con = duckdb.connect(database=':memory:')
    # Paso 1.1: Semilla determinista global
    con.execute("SELECT setseed(0.42);")
    
    # Paso 1.2: Tabla de N=100 parcelas según el algoritmo físico-matemático
    con.execute("""
    CREATE TABLE parcelas_invernadero AS
    WITH base AS (
        SELECT
            parcela_id,
            ROUND(100.0 + (random() * 200.0), 2) AS volumen_riego,   -- 100.0 a 300.0 m³/ha
            ROUND(50.0 + (random() * 100.0), 2) AS fertilizante_kg,  -- 50.0 a 150.0 kg/ha
            CASE
                WHEN random() < 0.50 THEN 'Tratamiento_A'
                ELSE 'Tratamiento_B'
            END AS tipo_tratamiento
        FROM generate_series(1, 100) AS t(parcela_id)
    )
    SELECT
        parcela_id,
        volumen_riego,
        fertilizante_kg,
        tipo_tratamiento,
        ROUND(
            5.0
            + (0.035 * volumen_riego)
            + (0.020 * fertilizante_kg)
            + CASE WHEN tipo_tratamiento = 'Tratamiento_B' THEN 1.8 ELSE 0.0 END
            + ((random() * 1.6) - 0.8),
            2
        ) AS rendimiento_ton
    FROM base;
    """)
    
    # Paso 1.3: Limpieza y extracción SQL hacia Pandas DataFrame
    df = con.execute("""
        SELECT *
        FROM parcelas_invernadero
        WHERE volumen_riego IS NOT NULL
          AND fertilizante_kg IS NOT NULL
          AND rendimiento_ton IS NOT NULL
        ORDER BY parcela_id ASC;
    """).df()
    return df

df_parcelas = generar_telemetria()

# ==============================================================================
# CÁLCULOS ANALÍTICOS GLOBALES (PARA KPIs Y MODELOS)
# ==============================================================================
# Hito 2: Regresión Lineal Múltiple (OLS)
X = df_parcelas[['volumen_riego', 'fertilizante_kg']]
Y = df_parcelas['rendimiento_ton']
X_const = sm.add_constant(X)
modelo_ols = sm.OLS(Y, X_const).fit()

y_pred = modelo_ols.predict(X_const)
residuos = Y - y_pred
rmse = np.sqrt(mean_squared_error(Y, y_pred))
matriz_corr = df_parcelas[['volumen_riego', 'fertilizante_kg', 'rendimiento_ton']].corr()

# Hito 3: Inferencia Estadística de Tratamientos (t-Student unilateral: H1: mu_B > mu_A)
grupo_A = df_parcelas[df_parcelas['tipo_tratamiento'] == 'Tratamiento_A']['rendimiento_ton']
grupo_B = df_parcelas[df_parcelas['tipo_tratamiento'] == 'Tratamiento_B']['rendimiento_ton']

# Prueba t de dos muestras independientes (unilateral: alternative='greater' para B > A)
t_stat, p_val_unilateral = stats.ttest_ind(grupo_B, grupo_A, equal_var=True, alternative='greater')

# Intervalo de Confianza al 95% para la diferencia de medias (mu_B - mu_A)
diff_medias = grupo_B.mean() - grupo_A.mean()
n_A, n_B = len(grupo_A), len(grupo_B)
s_p2 = (((n_A - 1) * grupo_A.var(ddof=1)) + ((n_B - 1) * grupo_B.var(ddof=1))) / (n_A + n_B - 2)
se_diff = np.sqrt(s_p2 * ((1 / n_A) + (1 / n_B)))
gl = n_A + n_B - 2
t_crit = stats.t.ppf(0.975, df=gl)
ic_inf = diff_medias - (t_crit * se_diff)
ic_sup = diff_medias + (t_crit * se_diff)

# ==============================================================================
# FASE 5 (ENLACE HITO 4): TARJETAS KPI DE RESUMEN SUPERIORES
# ==============================================================================
st.divider()
kpi1, kpi2, kpi3 = st.columns(3)
kpi1.metric(
    label="🌾 Rendimiento Promedio Global",
    value=f"{df_parcelas['rendimiento_ton'].mean():.2f} ton/ha",
    delta=f"± {df_parcelas['rendimiento_ton'].std():.2f} ton/ha (Desv. Est.)"
)
kpi2.metric(
    label="📈 R² Ajustado del Modelo OLS",
    value=f"{modelo_ols.rsquared_adj * 100:.2f}%",
    delta=f"RMSE: {rmse:.4f} ton/ha"
)
kpi3.metric(
    label="🧪 P-Valor Prueba Inferencial (t-Student)",
    value=f"{p_val_unilateral:.4e}",
    delta="Significativo (p < 0.05)" if p_val_unilateral < 0.05 else "No significativo (p >= 0.05)"
)
st.divider()

# ==============================================================================
# HITO 4 (CONTROLES LATERALES): FILTROS DINÁMICOS Y SIMULADOR
# ==============================================================================
st.sidebar.header("🎛️ Panel de Control Agronómico")
filtro_tratamiento = st.sidebar.multiselect(
    "Filtrar por Tipo de Tratamiento:",
    options=['Tratamiento_A', 'Tratamiento_B'],
    default=['Tratamiento_A', 'Tratamiento_B']
)

rango_riego = st.sidebar.slider(
    "Rango de Volumen de Riego (m³/ha):",
    min_value=100.0,
    max_value=300.0,
    value=(100.0, 300.0)
)

# Aplicar filtros dinámicos para la vista de telemetría y gráficas filtrables
df_filtrado = df_parcelas[
    (df_parcelas['tipo_tratamiento'].isin(filtro_tratamiento)) &
    (df_parcelas['volumen_riego'].between(rango_riego[0], rango_riego[1]))
]

if df_filtrado.empty:
    st.warning("⚠️ Selecciona al menos un tratamiento o amplía el rango de riego en el panel lateral.")
    st.stop()

st.sidebar.divider()
st.sidebar.subheader("🧮 Simulador de Cosecha (OLS)")
sim_riego = st.sidebar.slider("Volumen de Riego (m³/ha)", 100.0, 300.0, 200.0, 1.0)
sim_fert = st.sidebar.slider("Fertilizante Aplicado (kg/ha)", 50.0, 150.0, 100.0, 1.0)
pred_cosecha = modelo_ols.params['const'] + (modelo_ols.params['volumen_riego'] * sim_riego) + (modelo_ols.params['fertilizante_kg'] * sim_fert)
st.sidebar.metric("Rendimiento Predicho", f"{pred_cosecha:.2f} ton/ha")

# ==============================================================================
# ESTRUCTURA POR HITOS DE TRABAJO (TABS EJECUTIVOS)
# ==============================================================================
tab_h1, tab_h2, tab_h3, tab_h4 = st.tabs([
    "Hito 1: Telemetría SQL",
    "Hito 2: Regresión y Correlación",
    "Hito 3: Inferencia de Tratamientos",
    "Hito 4: Diseño del Dashboard"
])

# ------------------------------------------------------------------------------
# HITO 1: TELEMETRÍA SQL (DUCKDB)
# ------------------------------------------------------------------------------
with tab_h1:
    st.header("Hito 1: Carga y Preparación de Telemetría Agroindustrial (DuckDB)")
    st.markdown(f"""
    * **Semilla determinista:** `setseed(0.42)` | **Registros totales:** `100` parcelas | **Registros visibles (filtrados):** `{len(df_filtrado)}` parcelas.
    * **Distribución de grupos:** `Tratamiento_A` (Estándar) = **{n_A}** parcelas | `Tratamiento_B` (Bio-estimulante) = **{n_B}** parcelas.
    """)
    
    st.subheader("Vista de Telemetría de Parcelas")
    st.dataframe(df_filtrado, use_container_width=True)
    
    with st.expander("Ver Estadística Descriptiva por Tratamiento"):
        resumen_grupos = df_parcelas.groupby('tipo_tratamiento')[['volumen_riego', 'fertilizante_kg', 'rendimiento_ton']].agg(['count', 'mean', 'std', 'min', 'max'])
        st.dataframe(resumen_grupos.round(2), use_container_width=True)

# ------------------------------------------------------------------------------
# HITO 2: REGRESIÓN Y CORRELACIÓN (SUBTEMA 4.2)
# ------------------------------------------------------------------------------
with tab_h2:
    st.header("Hito 2: Análisis de Regresión Lineal y Correlación (Subtema 4.2)")
    
    col_corr, col_eq = st.columns([1, 1])
    with col_corr:
        st.subheader("Paso 2.1: Matriz de Correlación de Pearson (r)")
        st.dataframe(matriz_corr.round(4), use_container_width=True)
        r_riego = matriz_corr.loc['volumen_riego', 'rendimiento_ton']
        r_fert = matriz_corr.loc['fertilizante_kg', 'rendimiento_ton']
        factor_dom = "Volumen de Riego" if abs(r_riego) > abs(r_fert) else "Fertilizante Aplicado"
        st.info(f"📌 **Factor dominante:** `{factor_dom}` presenta la mayor asociación lineal con el rendimiento ($r_{{riego}} = {r_riego:.4f}$ vs $r_{{fert}} = {r_fert:.4f}$).")

    with col_eq:
        st.subheader("Paso 2.2: Ecuación de Regresión OLS Ajustada")
        b0 = modelo_ols.params['const']
        b1 = modelo_ols.params['volumen_riego']
        b2 = modelo_ols.params['fertilizante_kg']
        st.latex(rf"Rendimiento = {b0:.4f} + ({b1:.4f} \cdot Riego) + ({b2:.4f} \cdot Fertilizante)")
        st.markdown(f"""
        * **Interpretación de $\\beta_1$ (Riego):** Por cada $m^3/ha$ adicional de agua, el rendimiento aumenta en **{b1:.4f} ton/ha** manteniendo constante el fertilizante.
        * **Interpretación de $\\beta_2$ (Fertilizante):** Por cada $kg/ha$ adicional de fertilizante, el rendimiento aumenta en **{b2:.4f} ton/ha** manteniendo constante el riego.
        """)

    st.subheader("Paso 2.3: Validación de Métricas y Significancia de Coeficientes")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("R² (Determinación)", f"{modelo_ols.rsquared:.4f}")
    m2.metric("R² Ajustado", f"{modelo_ols.rsquared_adj:.4f}")
    m3.metric("RMSE", f"{rmse:.4f} ton/ha")
    m4.metric("Prob (F-statistic)", f"{modelo_ols.f_pvalue:.2e}")

    tabla_ols = pd.DataFrame({
        "Coeficiente (β)": modelo_ols.params,
        "Error Estándar": modelo_ols.bse,
        "Estadístico t": modelo_ols.tvalues,
        "Valor p (P>|t|)": modelo_ols.pvalues,
        "IC 2.5%": modelo_ols.conf_int()[0],
        "IC 97.5%": modelo_ols.conf_int()[1]
    })
    st.dataframe(
        tabla_ols.style.format({
            "Coeficiente (β)": "{:.4f}",
            "Error Estándar": "{:.4f}",
            "Estadístico t": "{:.3f}",
            "Valor p (P>|t|)": "{:.4e}",
            "IC 2.5%": "{:.4f}",
            "IC 97.5%": "{:.4f}"
        }),
        use_container_width=True
    )

    with st.expander("Ver salida técnica completa (OLS Regression Results)"):
        st.code(modelo_ols.summary().as_text(), language="text")

# ------------------------------------------------------------------------------
# HITO 3: INFERENCIA DE TRATAMIENTOS (SUBTEMA 4.3)
# ------------------------------------------------------------------------------
with tab_h3:
    st.header("Hito 3: Inferencia Estadística de Tratamientos en Campo (Subtema 4.3)")
    
    st.subheader("Paso 3.1: Formulación de Hipótesis")
    st.latex(r"H_0: \mu_B - \mu_A \le 0 \quad \text{(El Tratamiento B no incrementa el rendimiento promedio respecto al Tratamiento A)}")
    st.latex(r"H_1: \mu_B - \mu_A > 0 \quad \text{(El Tratamiento B produce significativamente más rendimiento que el Tratamiento A)}")

    st.subheader("Paso 3.2: Prueba t-Student de Dos Muestras Independientes")
    t1, t2, t3, t4 = st.columns(4)
    t1.metric("Media Tratamiento A", f"{grupo_A.mean():.2f} ton/ha", f"n = {n_A}")
    t2.metric("Media Tratamiento B", f"{grupo_B.mean():.2f} ton/ha", f"n = {n_B}")
    t3.metric("Diferencia (μB - μA)", f"+{diff_medias:.2f} ton/ha", f"IC 95%: [{ic_inf:.2f}, {ic_sup:.2f}]")
    t4.metric("Estadístico t / Valor p", f"t = {t_stat:.3f}", f"p = {p_val_unilateral:.4e}")

    st.subheader("Paso 3.3: Veredicto de Ingeniería (α = 0.05)")
    if p_val_unilateral < 0.05:
        st.success(f"""
        ✅ **SE RECHAZA LA HIPÓTESIS NULA ($H_0$)** dado que el valor $p = {p_val_unilateral:.4e} < 0.05$.  
        **Decisión Técnica y Financiera:** Existe evidencia estadística significativa al 95% de confianza (con una ganancia media estimada entre **{ic_inf:.2f}** y **{ic_sup:.2f} ton/ha**) para afirmar que el bio-estimulante (**Tratamiento_B**) incrementa el rendimiento agrícola. **Se recomienda autorizar la inversión para escalar el Tratamiento_B a todo el invernadero.**
        """)
    else:
        st.warning(f"""
        ⚠️ **NO SE RECHAZA LA HIPÓTESIS NULA ($H_0$)** dado que el valor $p = {p_val_unilateral:.4e} \ge 0.05$.  
        **Decisión Técnica:** No hay evidencia suficiente para afirmar que el Tratamiento_B supere al método estándar.
        """)

# ------------------------------------------------------------------------------
# HITO 4: DISEÑO DEL DASHBOARD (CUADRÍCULA 2x2 CON LAS 4 GRÁFICAS)
# ------------------------------------------------------------------------------
with tab_h4:
    st.header("Hito 4: Dashboard Visual Agroindustrial (Cuadrícula 2×2)")
    st.write("Visualizaciones interactivas actualizadas según los filtros seleccionados en el panel lateral.")

    sns.set_theme(style="whitegrid")
    
    # Fila 1 de la cuadrícula 2x2
    row1_col1, row1_col2 = st.columns(2)

    with row1_col1:
        st.subheader("1. Mapa de Calor de Correlación (Pearson)")
        fig1, ax1 = plt.subplots(figsize=(6, 4.5))
        corr_filtrada = df_filtrado[['volumen_riego', 'fertilizante_kg', 'rendimiento_ton']].corr()
        sns.heatmap(corr_filtrada, annot=True, fmt=".3f", cmap="YlGnBu", cbar=True, ax=ax1)
        ax1.set_title("Matriz de Correlación: Insumos vs Rendimiento")
        plt.tight_layout()
        st.pyplot(fig1)

    with row1_col2:
        st.subheader("2. Dispersión con Línea de Regresión OLS")
        fig2, ax2 = plt.subplots(figsize=(6, 4.5))
        sns.regplot(
            data=df_filtrado,
            x='volumen_riego',
            y='rendimiento_ton',
            scatter_kws={'alpha': 0.7, 'color': '#2E86AB'},
            line_kws={'color': '#D90429', 'linewidth': 2, 'label': 'Ajuste OLS + IC 95%'},
            ax=ax2
        )
        ax2.set_title("Rendimiento (Y) vs Volumen de Riego (X)")
        ax2.set_xlabel("Volumen de Riego (m³/ha)")
        ax2.set_ylabel("Rendimiento (ton/ha)")
        ax2.legend()
        plt.tight_layout()
        st.pyplot(fig2)

    # Fila 2 de la cuadrícula 2x2
    row2_col1, row2_col2 = st.columns(2)

    with row2_col1:
        st.subheader("3. Diagrama de Cajas Comparativo (A vs B)")
        fig3, ax3 = plt.subplots(figsize=(6, 4.5))
        sns.boxplot(
            data=df_filtrado,
            x='tipo_tratamiento',
            y='rendimiento_ton',
            palette={'Tratamiento_A': '#8D99AE', 'Tratamiento_B': '#2B9348'},
            ax=ax3
        )
        sns.stripplot(
            data=df_filtrado,
            x='tipo_tratamiento',
            y='rendimiento_ton',
            color='#1D3557',
            alpha=0.5,
            jitter=True,
            ax=ax3
        )
        ax3.set_title("Comparación de Rendimiento por Tratamiento")
        ax3.set_xlabel("Tipo de Tratamiento")
        ax3.set_ylabel("Rendimiento (ton/ha)")
        plt.tight_layout()
        st.pyplot(fig3)

    with row2_col2:
        st.subheader("4. Histograma y Densidad de Residuos")
        fig4, ax4 = plt.subplots(figsize=(6, 4.5))
        sns.histplot(residuos, kde=True, color='#6A4C93', bins=15, ax=ax4)
        ax4.axvline(0, color='#D90429', linestyle='--', linewidth=2, label='Media Teórica (0)')
        ax4.set_title("Distribución de Residuos del Modelo OLS (e = y - ŷ)")
        ax4.set_xlabel("Residuo (ton/ha)")
        ax4.set_ylabel("Frecuencia")
        ax4.legend()
        plt.tight_layout()
        st.pyplot(fig4)
