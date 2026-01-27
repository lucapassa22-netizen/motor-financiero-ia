# frontend.py - VERSIÓN 2.0 (CON MONTE CARLO Y BENCHMARK)
import streamlit as st
import requests
import pandas as pd
import json

# Configuración de página
st.set_page_config(page_title="Super Financial Engine", layout="wide", page_icon="💰")

# --- ESTILOS CSS ---
st.markdown("""
<style>
    .stButton>button { width: 100%; border-radius: 5px; }
    .metric-box { border: 1px solid #e6e6e6; padding: 10px; border-radius: 5px; }
</style>
""", unsafe_allow_html=True)

# Título Principal
st.title("🤖 Motor Financiero con IA")
st.markdown("Optimización, Proyección y Análisis Inteligente.")

# CONFIGURACIÓN DE CONEXIÓN
# NOTA: Si estás probando en tu PC, usa "http://127.0.0.1:8000"
# Si vas a subir a Render, usa tu URL de Render:
API_URL = "https://mi-motor-financiero-ia.onrender.com"

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

# Inicializar estado
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
                    st.session_state['opt_data'] = resp.json()
                    st.session_state['user_inputs'] = payload
                    st.success("¡Portafolio Optimizado!")
                else:
                    st.error(f"Error: {resp.text}")
            except Exception as e:
                st.error(f"Error de conexión: {e}")

# --- SECCIÓN 2: RESULTADOS (PESTAÑAS) ---
if st.session_state['opt_data']:
    data = st.session_state['opt_data']
    inputs = st.session_state['user_inputs']
    
    st.markdown("---")
    
    # AHORA SON 6 PESTAÑAS
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📊 Distribución", 
        "🆚 Benchmark", 
        "🔮 Monte Carlo", 
        "⏳ Backtest", 
        "🧠 Consultor IA", 
        "📥 Exportar"
    ])

    # === TAB 1: DISTRIBUCIÓN (Igual que antes) ===
    with tab1:
        st.subheader("Composición del Portafolio")
        col_res1, col_res2 = st.columns([1, 1])
        with col_res1:
            weights = data["weights"]
            df_w = pd.DataFrame(list(weights.items()), columns=["Activo", "Peso"])
            st.bar_chart(df_w.set_index("Activo"))
        with col_res2:
            met = data["metrics"]
            st.metric("Retorno Anual Esperado", f"{met['ret_anual']:.1%}")
            st.metric("Volatilidad (Riesgo)", f"{met['vol_anual']:.1%}")
            st.metric("Sharpe Ratio", f"{met['sharpe']:.2f}")

    # === TAB 2: BENCHMARK (NUEVO) ===
    with tab2:
        st.subheader("Tu Portafolio vs S&P 500")
        st.caption("Compara tu estrategia contra el mercado desde una fecha específica.")
        
        start_date_bm = st.date_input("Fecha Inicio Comparación", pd.to_datetime("2021-01-01"))
        
        if st.button("Comparar con Mercado"):
            bm_payload = {
                "tickers": inputs["tickers"],
                "weights": data["weights"],
                "start_date": str(start_date_bm)
            }
            with st.spinner("Descargando datos del mercado..."):
                try:
                    r = requests.post(f"{API_URL}/api/v1/benchmark", json=bm_payload)
                    if r.status_code == 200:
                        bm_data = r.json()
                        if "error" in bm_data:
                             st.error(bm_data["error"])
                        else:
                            # Crear DataFrame para el gráfico
                            df_bm = pd.DataFrame({
                                "Tu Portafolio": bm_data["portfolio"],
                                "S&P 500 (Mercado)": bm_data["benchmark"]
                            }, index=pd.to_datetime(bm_data["dates"]))
                            
                            st.line_chart(df_bm)
                            
                            # Calcular quién ganó
                            final_port = bm_data["portfolio"][-1]
                            final_spy = bm_data["benchmark"][-1]
                            diff = (final_port - final_spy) * 100
                            
                            c1, c2 = st.columns(2)
                            c1.metric("Retorno Acumulado Tuyo", f"{(final_port-1)*100:.2f}%")
                            c2.metric("Retorno Mercado (SPY)", f"{(final_spy-1)*100:.2f}%", delta=f"{diff:.2f}%")
                            
                            if final_port > final_spy:
                                st.success(f"¡Le ganaste al mercado por {diff:.2f} puntos!")
                            else:
                                st.warning(f"El mercado ganó esta vez por {abs(diff):.2f} puntos.")
                    else:
                        st.error("Error en cálculo de Benchmark")
                except Exception as e:
                    st.error(f"Error: {e}")

    # === TAB 3: MONTE CARLO (NUEVO) ===
    with tab3:
        st.subheader("Simulación Futura (Monte Carlo)")
        st.caption("Proyección probabilística a 1 año (252 días bursátiles).")
        
        
        sims = st.slider("Número de escenarios a simular", 100, 1000, 500)
        
        if st.button("Correr Simulación"):
            mc_payload = {
                "tickers": inputs["tickers"],
                "weights": data["weights"],
                "initial_capital": inputs["initial_capital"],
                "simulations": sims
            }
            
            with st.spinner("Simulando futuros posibles..."):
                try:
                    r = requests.post(f"{API_URL}/api/v1/montecarlo", json=mc_payload)
                    if r.status_code == 200:
                        mc_data = r.json()
                        if "error" in mc_data:
                            st.error(mc_data["error"])
                        else:
                            # Gráfico
                            df_mc = pd.DataFrame({
                                "Escenario Optimista (95%)": mc_data["optimistic"],
                                "Escenario Base (Mediana)": mc_data["median"],
                                "Escenario Pesimista (5%)": mc_data["pessimistic"]
                            })
                            st.line_chart(df_mc)
                            
                            final_median = mc_data['median'][-1]
                            ganancia = final_median - inputs['initial_capital']
                            
                            st.info(f"En el escenario base, tus ${inputs['initial_capital']:,} se podrían convertir en **${final_median:,.2f}** (Ganancia: ${ganancia:,.2f})")
                    else:
                        st.error("Error en simulación")
                except Exception as e:
                    st.error(f"Error: {e}")

    # === TAB 4: BACKTEST (Igual que antes) ===
    with tab4:
        st.subheader("Backtest Detallado")
        start_date_bt = st.date_input("Fecha Inicio Backtest", pd.to_datetime("2020-01-01"))
        if st.button("Correr Backtest Detallado"):
            bt_payload = {
                "tickers": inputs["tickers"],
                "weights": data["weights"],
                "initial_capital": inputs["initial_capital"],
                "start_date": str(start_date_bt)
            }
            with st.spinner("Calculando..."):
                try:
                    bt_resp = requests.post(f"{API_URL}/api/v1/backtest", json=bt_payload)
                    if bt_resp.status_code == 200:
                        bt_data = bt_resp.json()
                        b1, b2, b3 = st.columns(3)
                        b1.metric("Saldo Final", f"${bt_data['final_balance']:,.2f}")
                        b2.metric("Retorno Total", f"{bt_data['total_return_pct']}%")
                        b3.metric("Max Drawdown", f"{bt_data['max_drawdown_pct']}%")
                        
                        hist_data = bt_data["history"]
                        df_hist = pd.DataFrame(hist_data)
                        df_hist['date'] = pd.to_datetime(df_hist['date'])
                        st.line_chart(df_hist.set_index('date'))
                    else:
                        st.error("Error Backtest")
                except Exception as e:
                    st.error(f"Error: {e}")

    # === TAB 5: IA (Igual que antes) ===
    with tab5:
        st.subheader("Análisis Gemini IA")
        if not google_api_key:
            st.warning("⚠️ Ingresa tu API Key en la barra lateral.")
        else:
            if st.button("Consultar a la IA"):
                ai_payload = {
                    "weights": data["weights"],
                    "metrics": data["metrics"],
                    "risk_profile": inputs["risk_profile"],
                    "api_key": google_api_key
                }
                with st.spinner("Analizando..."):
                    try:
                        ai_resp = requests.post(f"{API_URL}/api/v1/analyze", json=ai_payload)
                        if ai_resp.status_code == 200:
                            st.success("Análisis completado:")
                            st.markdown(ai_resp.json().get("ai_analysis", "Sin respuesta"))
                        else:
                            st.error(f"Error IA: {ai_resp.text}")
                    except Exception as e:
                        st.error(f"Error: {e}")

    # === TAB 6: EXPORTAR (Igual que antes) ===
    with tab6:
        st.subheader("Descargar Reporte")
        export_payload = { "weights": data["weights"], "metrics": data["metrics"] }
        if st.button("Generar Excel"):
            with st.spinner("Generando..."):
                try:
                    xls_resp = requests.post(f"{API_URL}/api/v1/export", json=export_payload)
                    if xls_resp.status_code == 200:
                        st.download_button(
                            label="📥 Descargar Reporte (.xlsx)",
                            data=xls_resp.content,
                            file_name="reporte_inversion_pro.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                        )
                        st.success("¡Listo!")
                    else:
                        st.error("Error Excel")
                except Exception as e:
                    st.error(f"Error: {e}")