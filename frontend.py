# frontend.py - VERSIÓN FINAL COMPLETA
import streamlit as st
import requests
import pandas as pd
import json

# Configuración de página
st.set_page_config(page_title="Super Financial Engine", layout="wide", page_icon="💰")

# --- ESTILOS CSS (Opcional, para que se vea más limpio) ---
st.markdown("""
<style>
    .stButton>button { width: 100%; border-radius: 5px; }
    .metric-box { border: 1px solid #e6e6e6; padding: 10px; border-radius: 5px; }
</style>
""", unsafe_allow_html=True)

# Título Principal
st.title("🤖 Motor Financiero con IA")
st.markdown("Optimización, Backtesting y Análisis Inteligente en un solo lugar.")

# CONFIGURACIÓN DE CONEXIÓN
API_URL = "http://127.0.0.1:8005"

# --- SIDEBAR: ESTADO Y API KEY ---
st.sidebar.header("⚙️ Configuración")

# Verificación de API
if st.sidebar.button("Revisar Conexión API"):
    try:
        r = requests.get(f"{API_URL}/")
        if r.status_code == 200:
            st.sidebar.success("Backend Online 🟢")
        else:
            st.sidebar.error("Backend con Error 🔴")
    except:
        st.sidebar.error("Backend Apagado 🔌")

st.sidebar.markdown("---")
# Campo para la API Key de Google (para no quemarla en código)
google_api_key = st.sidebar.text_input("🔑 Google Gemini API Key", type="password")
st.sidebar.info("Necesaria para la pestaña 'Consultor IA'")

# --- SECCIÓN 1: INPUTS (Siempre visible) ---
st.markdown("### 1. Define tu Estrategia")

c1, c2, c3 = st.columns(3)
with c1:
    tickers_in = st.text_input("Tickers (ej: AAPL, MSFT, SPY)", "AAPL,MSFT,GOOGL,AMZN,TSLA,SPY")
with c2:
    risk_in = st.selectbox("Perfil de Riesgo", ["Conservador", "Moderado", "Arriesgado"])
with c3:
    cap_in = st.number_input("Capital Inicial (USD)", value=10000, step=1000)

# Inicializar estado si no existe (Memoria del navegador)
if 'opt_data' not in st.session_state:
    st.session_state['opt_data'] = None

# BOTÓN DE OPTIMIZACIÓN
if st.button("🚀 GENERAR PORTAFOLIO ÓPTIMO", type="primary"):
    tickers_list = [t.strip().upper() for t in tickers_in.split(",") if t.strip()]
    
    if not tickers_list:
        st.error("Ingresa al menos un ticker.")
    else:
        payload = {
            "tickers": tickers_list,
            "risk_profile": risk_in,
            "initial_capital": cap_in
        }
        
        with st.spinner("Calculando la frontera eficiente..."):
            try:
                resp = requests.post(f"{API_URL}/api/v1/optimize", json=payload)
                if resp.status_code == 200:
                    # GUARDAMOS EL RESULTADO EN MEMORIA (SESSION STATE)
                    st.session_state['opt_data'] = resp.json()
                    st.session_state['user_inputs'] = payload # Guardamos también los inputs
                    st.success("¡Portafolio Optimizado!")
                else:
                    st.error(f"Error: {resp.text}")
            except Exception as e:
                st.error(f"Error de conexión: {e}")

# --- SECCIÓN 2: RESULTADOS (PESTAÑAS) ---
# Solo mostramos esto si ya tenemos datos en memoria
if st.session_state['opt_data']:
    data = st.session_state['opt_data']
    inputs = st.session_state['user_inputs']
    
    st.markdown("---")
    # Creamos las pestañas
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Distribución", "⏳ Backtest (Histórico)", "🧠 Consultor IA", "📥 Exportar Excel"])

    # === PESTAÑA 1: DISTRIBUCIÓN (Lo que ya tenías) ===
    with tab1:
        st.subheader("Composición del Portafolio")
        
        col_res1, col_res2 = st.columns([1, 1])
        
        with col_res1:
            # Gráfico de pesos
            weights = data["weights"]
            df_w = pd.DataFrame(list(weights.items()), columns=["Activo", "Peso"])
            st.bar_chart(df_w.set_index("Activo"))
            
        with col_res2:
            # Métricas grandes
            met = data["metrics"]
            st.metric("Retorno Anual Esperado", f"{met['ret_anual']:.1%}")
            st.metric("Volatilidad (Riesgo)", f"{met['vol_anual']:.1%}")
            st.metric("Sharpe Ratio", f"{met['sharpe']:.2f}")

    # === PESTAÑA 2: BACKTEST (Nuevo) ===
    with tab2:
        st.subheader("Prueba Histórica")
        st.caption("¿Qué hubiera pasado si invertías este dinero en 2020?")
        
        start_date_bt = st.date_input("Fecha de Inicio", pd.to_datetime("2020-01-01"))
        
        if st.button("Correr Backtest"):
            # Usamos los pesos que ya calculó la optimización
            bt_payload = {
                "tickers": inputs["tickers"],
                "weights": data["weights"], # Pesos optimizados
                "initial_capital": inputs["initial_capital"],
                "start_date": str(start_date_bt)
            }
            
            with st.spinner("Viajando en el tiempo..."):
                try:
                    bt_resp = requests.post(f"{API_URL}/api/v1/backtest", json=bt_payload)
                    if bt_resp.status_code == 200:
                        bt_data = bt_resp.json()
                        
                        # Métricas del Backtest
                        b1, b2, b3 = st.columns(3)
                        b1.metric("Saldo Final", f"${bt_data['final_balance']:,.2f}")
                        b2.metric("Retorno Total", f"{bt_data['total_return_pct']}%")
                        b3.metric("Peor Caída (Max Drawdown)", f"{bt_data['max_drawdown_pct']}%", delta_color="inverse")
                        
                        # Gráfico de Línea
                        hist_data = bt_data["history"] # Es una lista de dicts
                        df_hist = pd.DataFrame(hist_data)
                        df_hist['date'] = pd.to_datetime(df_hist['date'])
                        df_hist = df_hist.set_index('date')
                        
                        st.line_chart(df_hist)
                    else:
                        st.error("Error en Backtest")
                except Exception as e:
                    st.error(f"Error: {e}")

    # === PESTAÑA 3: INTELIGENCIA ARTIFICIAL (Nuevo) ===
    with tab3:
        st.subheader("Análisis de Inversión con Gemini")
        
        if not google_api_key:
            st.warning("⚠️ Por favor ingresa tu API Key de Google en la barra lateral izquierda para usar esta función.")
        else:
            if st.button("Consultar a la IA"):
                ai_payload = {
                    "weights": data["weights"],
                    "metrics": data["metrics"],
                    "risk_profile": inputs["risk_profile"],
                    "api_key": google_api_key
                }
                
                with st.spinner("La IA está analizando tus activos..."):
                    try:
                        ai_resp = requests.post(f"{API_URL}/api/v1/analyze", json=ai_payload)
                        if ai_resp.status_code == 200:
                            analysis_text = ai_resp.json().get("ai_analysis", "Sin respuesta")
                            st.success("Análisis completado:")
                            st.markdown(analysis_text) # Renderiza el Markdown bonito
                        else:
                            st.error(f"Error IA: {ai_resp.text}")
                    except Exception as e:
                        st.error(f"Error: {e}")

    # === PESTAÑA 4: EXPORTAR (Nuevo) ===
    with tab4:
        st.subheader("Descargar Reporte")
        st.write("Genera un archivo Excel con todos los cálculos técnicos.")
        
        # Preparamos los datos para enviar
        export_payload = {
            "weights": data["weights"],
            "metrics": data["metrics"]
        }
        
        # Lógica para descargar
        # Streamlit necesita leer los bytes primero
        if st.button("Generar Excel"):
            with st.spinner("Generando archivo..."):
                try:
                    # Hacemos el POST pero pedimos el contenido binario
                    xls_resp = requests.post(f"{API_URL}/api/v1/export", json=export_payload)
                    
                    if xls_resp.status_code == 200:
                        st.download_button(
                            label="📥 Descargar Reporte (.xlsx)",
                            data=xls_resp.content,
                            file_name="reporte_inversion_pro.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                        )
                        st.success("¡Archivo listo para descargar!")
                    else:
                        st.error("Error generando Excel")
                except Exception as e:
                    st.error(f"Error: {e}")