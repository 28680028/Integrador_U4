import streamlit as st
import duckdb
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.stattools import durbin_watson
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# Configuración inicial de la página
st.set_page_config(
    page_title="Práctica 4.2 - Regresión Lineal y Correlación",
    page_icon="💧",
    layout="wide"
)

st.title("💧 Práctica 4.2: Regresión Lineal Múltiple y Correlación")
st.markdown("""
**Asignatura:** Análisis y Exploración de Datos (INC-2505) | **Unidad 4:** Análisis Estadístico Inferencial  
**Caso de Estudio:** Predicción del Consumo Per Cápita de Agua Potable (CPC) basado en *Arellano & Peña (2020)*.
""")
st.divider()

# ==============================================================================
# PASO 1: SIMULACIÓN DE DATOS RELACIONALES EN DUCKDB (REPRODUCIBLE)
# ==============================================================================
@st.cache_data
def cargar_datos():
    con = duckdb.connect(database=':memory:')
    con.execute("SELECT setseed(0.42);")
    con.execute("""
    CREATE TABLE consumo_agua AS
    SELECT
        residencia_id,
        ROUND(20 + (random() * 15), 2) AS temperatura_max,    -- 20°C a 35°C
        ROUND(40 + (random() * 50), 2) AS humedad_max,        -- 40% a 90%
        ROUND(2 + (random() * 4), 0) AS personas_familia,     -- 2 a 6 hab/vivienda
        ROUND(0.5 + (random() * 2.5), 2) AS consumo_bidon_pc, -- 0.5 a 3.0 lt/hab-día
        ROUND(
            95.0 + (2.5 * (20 + random() * 15))
                 - (0.8 * (40 + random() * 50))
                 - (5.2 * ROUND(2 + random() * 4, 0))
                 + (15.0 * (0.5 + random() * 2.5))
                 + (random() * 6.0 - 3.0), 2
        ) AS cpc_consumo
    FROM generate_series(1, 100) AS t(residencia_id);
    """)
    return con.execute("SELECT * FROM consumo_agua").df()

df_agua = cargar_datos()

st.header("1. Simulación de Datos Relacionales (DuckDB)")
st.write("Primeros 5 registros extraídos con semilla determinista `setseed(0.42)` (100 observaciones totales):")
st.dataframe(df_agua.head(), use_container_width=True)

with st.expander("Ver base de datos completa (100 registros)"):
    st.dataframe(df_agua, use_container_width=True)

st.divider()

# ==============================================================================
# PASO 2: MATRIZ DE CORRELACIÓN DE PEARSON (r)
# ==============================================================================
st.header("2. Matriz de Correlación de Pearson (r)")
matriz_correlacion = df_agua[['temperatura_max', 'humedad_max', 'personas_familia', 'consumo_bidon_pc', 'cpc_consumo']].corr()

col1, col2 = st.columns([2, 1])
with col1:
    st.subheader("Matriz Completa")
    st.dataframe(matriz_correlacion.round(4), use_container_width=True)
with col2:
    st.subheader("Correlación con CPC")
    corr_cpc = matriz_correlacion['cpc_consumo'].drop('cpc_consumo').round(4)
    st.dataframe(corr_cpc, use_container_width=True)

st.divider()

# ==============================================================================
# PASO 3: REGRESIÓN LINEAL MÚLTIPLE CON STATSMODELS (OLS)
# ==============================================================================
st.header("3. Resumen del Modelo OLS (statsmodels)")
X = df_agua[['temperatura_max', 'humedad_max', 'personas_familia', 'consumo_bidon_pc']]
Y = df_agua['cpc_consumo']
X_const = sm.add_constant(X)

modelo_ols = sm.OLS(Y, X_const).fit()
st.text(modelo_ols.summary().as_text())

st.divider()

# ==============================================================================
# PASO 4: DIAGNÓSTICO DE RESIDUOS Y SUPUESTOS
# ==============================================================================
st.header("4. Diagnóstico de Residuos y Supuestos")
residuos = modelo_ols.resid
stat_sw, p_val_sw = stats.shapiro(residuos)
dw_stat = durbin_watson(residuos)

col_sw, col_dw = st.columns(2)
with col_sw:
    st.subheader("Prueba de Normalidad (Shapiro-Wilk)")
    st.metric(label="Estadístico W", value=f"{stat_sw:.4f}", delta=f"p-valor = {p_val_sw:.4f}")
    if p_val_sw > 0.05:
        st.success("✅ Los residuos cumplen con el supuesto de NORMALIDAD (p > 0.05).")
    else:
        st.error("❌ Se rechaza la normalidad de los residuos (p <= 0.05).")

with col_dw:
    st.subheader("Independencia de Errores (Durbin-Watson)")
    st.metric(label="Estadístico DW", value=f"{dw_stat:.4f}")
    st.info("ℹ️ Valores entre 1.5 y 2.5 indican ausencia de autocorrelación.")

st.divider()

# ==============================================================================
# PASO 5: EVALUACIÓN PREDICTIVA CON SCIKIT-LEARN
# ==============================================================================
st.header("5. Métricas de Rendimiento Predictivo (scikit-learn)")
modelo_sk = LinearRegression()
modelo_sk.fit(X, Y)
y_pred = modelo_sk.predict(X)

r2_sk = r2_score(Y, y_pred)
rmse_sk = np.sqrt(mean_squared_error(Y, y_pred))

m1, m2 = st.columns(2)
m1.metric("Coeficiente de Determinación (R²)", f"{r2_sk * 100:.2f}%")
m2.metric("Error Cuadrático Medio Raíz (RMSE)", f"{rmse_sk:.4f} lt/hab-día")

st.subheader("Ecuación del Modelo Ajustado:")
st.latex(
    rf"CPC = {modelo_sk.intercept_:.2f} + ({modelo_sk.coef_[0]:.2f} \cdot Temp) + ({modelo_sk.coef_[1]:.2f} \cdot Humedad) + ({modelo_sk.coef_[2]:.2f} \cdot Personas) + ({modelo_sk.coef_[3]:.2f} \cdot Bidón)"
)

st.divider()

# ==============================================================================
# PASO 6: PANEL DE DIAGNÓSTICO VISUAL (GRÁFICAS)
# ==============================================================================
st.header("6. Panel de Diagnóstico Visual")

sns.set_theme(style="whitegrid")
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle('Panel de Diagnóstico Visual: Regresión Lineal Múltiple (Subtema 4.2)', fontsize=16, fontweight='bold')

# 1. Mapa de Calor de Correlación
sns.heatmap(matriz_correlacion, annot=True, fmt=".2f", cmap="coolwarm", cbar=True, ax=axes[0, 0])
axes[0, 0].set_title('1. Matriz de Correlación de Pearson (r)')

# 2. Scatter Plot: Humedad vs Consumo
sns.regplot(data=df_agua, x='humedad_max', y='cpc_consumo', color='b', ax=axes[0, 1],
            line_kws={'color': 'red', 'linewidth': 2})
axes[0, 1].set_title('2. Tendencia: Humedad Máx vs Consumo')
axes[0, 1].set_xlabel('Humedad Máxima (%)')
axes[0, 1].set_ylabel('CPC Consumo (lt/hab-día)')

# 3. Valores Reales vs Predichos
axes[0, 2].scatter(y_pred, Y, alpha=0.7, color='teal')
axes[0, 2].plot([Y.min(), Y.max()], [Y.min(), Y.max()], 'r--', lw=2)
axes[0, 2].set_title(f'3. Val. Reales vs Predichos (R² = {r2_sk * 100:.1f}%)')
axes[0, 2].set_xlabel('CPC Predicho (ŷ)')
axes[0, 2].set_ylabel('CPC Real (y)')

# 4. Residuos vs Valores Ajustados (Homocedasticidad)
axes[1, 0].scatter(y_pred, residuos, color='purple', alpha=0.6)
axes[1, 0].axhline(y=0, color='r', linestyle='--', lw=2)
axes[1, 0].set_title('4. Residuos vs Ajustados (Homocedasticidad)')
axes[1, 0].set_xlabel('Valores Predichos (ŷ)')
axes[1, 0].set_ylabel('Residuos (e = y - ŷ)')

# 5. Gráfico Q-Q Normal (Normalidad de Errores)
sm.qqplot(residuos, line='45', fit=True, ax=axes[1, 1])
axes[1, 1].set_title('5. Gráfico Q-Q Normal (Residuos)')

# Ocultar la celda sobrante (posición [1, 2])
axes[1, 2].axis('off')

plt.tight_layout()
st.pyplot(fig)

# ==============================================================================
# BARRA LATERAL: SIMULADOR INTERACTIVO DE PREDICCIÓN
# ==============================================================================
st.sidebar.header("🧮 Simulador de Consumo (CPC)")
st.sidebar.write("Ajusta las variables para predecir el consumo per cápita de agua:")

in_temp = st.sidebar.slider("Temperatura Máxima (°C)", 20.0, 35.0, 27.5, 0.1)
in_hum = st.sidebar.slider("Humedad Máxima (%)", 40.0, 90.0, 65.0, 0.5)
in_pers = st.sidebar.slider("Personas por Familia", 2, 6, 4, 1)
in_bidon = st.sidebar.slider("Consumo Bidón (lt/hab-día)", 0.5, 3.0, 1.5, 0.1)

entrada_usuario = pd.DataFrame({
    'temperatura_max': [in_temp],
    'humedad_max': [in_hum],
    'personas_familia': [in_pers],
    'consumo_bidon_pc': [in_bidon]
})

prediccion_cpc = modelo_sk.predict(entrada_usuario)[0]
st.sidebar.divider()
st.sidebar.metric("CPC Estimado", f"{prediccion_cpc:.2f} lt/hab-día")