import pandas as pd
import numpy as np
import yfinance as yf
import io
import json
import google.generativeai as genai
from pypfopt import risk_models, expected_returns, EfficientFrontier
from pypfopt import HRPOpt, black_litterman

class FinancialEngine:
    def __init__(self):
        """Inicializa el motor financiero."""
        pass

    # ==========================================
    # 1. GESTIÓN DE DATOS (DATA FEED)
    # ==========================================
    def get_market_data(self, tickers, start_date, end_date):
        """Descarga precios de Yahoo Finance."""
        if not tickers: return pd.DataFrame()
        
        # Aseguramos SPY para benchmarking interno, pero marcamos qué pidió el usuario
        tickers_to_download = list(set(tickers + ['SPY']))
        
        try:
            data = yf.download(tickers_to_download, start=start_date, end=end_date, progress=False)
            
            # Manejo robusto de MultiIndex de yfinance
            if isinstance(data.columns, pd.MultiIndex):
                col = 'Adj Close' if 'Adj Close' in data.columns else 'Close'
                prices = data[col]
            else:
                prices = data['Adj Close'] if 'Adj Close' in data.columns else data['Close']

            prices = prices.dropna(how='all')
            return prices
        except Exception as e:
            raise RuntimeError(f"Error descargando datos: {e}")

    def get_real_market_caps(self, tickers):
        """
        Obtiene Market Caps reales para Black-Litterman.
        Usa fast_info para no ralentizar la app.
        """
        mcaps = {}
        for t in tickers:
            try:
                # fast_info es mucho más rápido que .info
                mcaps[t] = yf.Ticker(t).fast_info['market_cap']
            except:
                # Fallback realista: Promedio del resto o valor base para no romper
                mcaps[t] = 1e9 
        return mcaps

    # ==========================================
    # 2. MOTOR DE OPTIMIZACIÓN (CEREBRO)
    # ==========================================
    def optimize_portfolio(self, prices, method='markowitz', objective='max_sharpe', 
                          risk_free_rate=0.02, bounds=(0.0, 1.0), target_return=None,
                          market_benchmark='SPY'):
        """
        Retorna: weights (dict), performance (tuple), status (dict)
        """
        # Filtramos SPY si el usuario NO lo pidió explícitamente en la lista de tickers del DF
        # (Se asume que 'prices' trae solo lo que el usuario seleccionó + SPY si faltaba)
        opt_cols = [c for c in prices.columns if c != market_benchmark]
        # Si el usuario SI eligió SPY en su lista, lo dejamos
        # (Lógica simplificada: Optimizamos sobre todo lo que no sea el benchmark forzado)
        if market_benchmark in prices.columns and len(prices.columns) > 1:
             # Verificación simple: Si SPY está en prices pero no fue solicitado por usuario, idealmente 
             # se debería filtrar antes. Aquí asumimos que el Dashboard pasa prices limpios o filtramos:
             pass 

        opt_prices = prices[opt_cols] if len(opt_cols) > 0 else prices
        
        weights = {}
        perf = (0, 0, 0)
        status = {"success": True, "message": "Optimización exitosa"}
        ef = None

        try:
            # --- A. HRP (Hierarchical Risk Parity) ---
            if method == 'hrp':
                rets = expected_returns.returns_from_prices(opt_prices)
                hrp = HRPOpt(rets)
                hrp.optimize()
                weights = hrp.clean_weights()
                perf = hrp.portfolio_performance(verbose=False, risk_free_rate=risk_free_rate)
            
            # --- B. BLACK-LITTERMAN (Realista con Market Caps) ---
            elif method == 'black_litterman':
                # 1. Matriz Covarianza
                S = risk_models.sample_cov(opt_prices)
                
                # 2. Delta de Mercado (Aversión al riesgo global)
                if market_benchmark in prices.columns:
                    delta = black_litterman.market_implied_risk_aversion(prices[market_benchmark])
                else:
                    delta = 2.5 # Valor estándar académico si falla la descarga
                
                # 3. Market Caps Reales (Tu petición de realismo)
                mcaps = self.get_real_market_caps(opt_prices.columns.tolist())
                
                # 4. Retornos Implícitos de Mercado (Prior)
                mu = black_litterman.market_implied_prior_returns(mcaps, delta, S)
                
                # 5. Optimizar sobre la Frontera con los nuevos retornos
                ef = EfficientFrontier(mu, S, weight_bounds=bounds)
                if objective == 'min_volatility':
                    weights = ef.min_volatility()
                else:
                    weights = ef.max_sharpe(risk_free_rate=risk_free_rate)
                
                weights = ef.clean_weights()
                perf = ef.portfolio_performance(verbose=False, risk_free_rate=risk_free_rate)

            # --- C. MARKOWITZ CLÁSICO ---
            else:
                mu = expected_returns.mean_historical_return(opt_prices)
                S = risk_models.sample_cov(opt_prices)
                ef = EfficientFrontier(mu, S, weight_bounds=bounds)
                
                if objective == 'min_volatility':
                    weights = ef.min_volatility()
                elif objective == 'target_return' and target_return:
                    weights = ef.efficient_return(target_return=target_return)
                else:
                    weights = ef.max_sharpe(risk_free_rate=risk_free_rate)
                
                weights = ef.clean_weights()
                perf = ef.portfolio_performance(verbose=False, risk_free_rate=risk_free_rate)

        except Exception as e:
            # Fallback seguro (Equiponderado) + Aviso
            status = {"success": False, "message": f"Fallo en optimización ({str(e)}). Usando equiponderado."}
            n = len(opt_prices.columns)
            weights = {t: 1/n for t in opt_prices.columns}
            # Estimación rápida de performance del fallback
            mu_f = expected_returns.mean_historical_return(opt_prices)
            S_f = risk_models.sample_cov(opt_prices)
            w_arr = np.array(list(weights.values()))
            ret_f = np.sum(w_arr * mu_f)
            vol_f = np.sqrt(np.dot(w_arr.T, np.dot(S_f, w_arr)))
            perf = (ret_f, vol_f, 0.0)

        return weights, perf, status

    # ==========================================
    # 3. CÁLCULO DE CURVAS (NAV)
    # ==========================================
    def calculate_portfolio_nav(self, prices, weights, capital, freq='M', fee_perc=0.0):
        if isinstance(weights, dict): weights = pd.Series(weights)
        
        assets = weights.index.intersection(prices.columns)
        if assets.empty: return None, None

        prices_clean = prices[assets].dropna()
        weights_clean = weights[assets]
        asset_rets = prices_clean.pct_change().dropna()
        
        # Retorno Bruto
        port_daily_ret = (asset_rets * weights_clean).sum(axis=1)
        
        # Ajuste de Costos (Realismo vs Vectorización)
        # Si hay fees, los aplicamos como "expense ratio" diario para mantener la curva suave
        if fee_perc > 0:
            daily_fee = fee_perc / 252
            port_daily_ret = port_daily_ret - daily_fee
        
        nav_curve = (1 + port_daily_ret).cumprod() * capital
        return port_daily_ret, nav_curve

    def align_and_benchmark(self, prices, port_ret, capital):
        # Obtener SPY
        spy_ret = pd.Series(dtype=float)
        if 'SPY' in prices.columns:
            spy_ret = prices['SPY'].pct_change().fillna(0)
        else:
            # Si no estaba, descarga rápida de fallback
             spy_data = yf.download('SPY', start=prices.index[0], end=prices.index[-1], progress=False)
             col = 'Adj Close' if 'Adj Close' in spy_data.columns else 'Close'
             spy_ret = spy_data[col].pct_change().fillna(0)

        common = port_ret.index.intersection(spy_ret.index)
        
        port_a = port_ret.loc[common]
        spy_a = spy_ret.loc[common]
        
        c_port = (1 + port_a).cumprod() * capital
        c_spy = (1 + spy_a).cumprod() * capital
        
        df = pd.concat([c_port, c_spy], axis=1)
        df.columns = ['Portafolio Estratégico', 'S&P 500 Benchmark']
        return port_a, spy_a, df.dropna()

    # ==========================================
    # 4. RIESGO Y MÉTRICAS
    # ==========================================
    def calculate_metrics(self, daily_ret, benchmark_ret=None, risk_free=0.02):
        if daily_ret.empty: return {}
        
        ann_ret = daily_ret.mean() * 252
        ann_vol = daily_ret.std() * np.sqrt(252)
        sharpe = (ann_ret - risk_free) / ann_vol if ann_vol > 0 else 0
        
        cum = (1 + daily_ret).cumprod()
        dd = (cum / cum.cummax()) - 1
        max_dd = dd.min()
        
        sortino = (ann_ret - risk_free) / (daily_ret[daily_ret<0].std()*np.sqrt(252)) if len(daily_ret[daily_ret<0]) > 0 else 0
        calmar = ann_ret / abs(max_dd) if max_dd != 0 else 0
        
        var95 = np.percentile(daily_ret, 5)
        cvar95 = daily_ret[daily_ret <= var95].mean()

        alpha, beta = 0.0, 0.0
        if benchmark_ret is not None:
            common = daily_ret.index.intersection(benchmark_ret.index)
            if len(common) > 20:
                cov = np.cov(daily_ret.loc[common], benchmark_ret.loc[common])
                beta = cov[0,1]/cov[1,1]
                alpha = ann_ret - (risk_free + beta*(benchmark_ret.loc[common].mean()*252 - risk_free))

        return {
            "ret_anual": ann_ret, "vol_anual": ann_vol, "sharpe": sharpe,
            "sortino": sortino, "calmar": calmar, "max_dd": max_dd,
            "var_95": var95, "cvar_95": cvar95, "alpha": alpha, "beta": beta,
            "dd_series": dd
        }

    def get_rolling(self, ret, w=126):
        return ret.rolling(w).std()*np.sqrt(252), ret.rolling(w).mean()*252

    # ==========================================
    # 5. IA & MONTE CARLO
    # ==========================================
    def run_monte_carlo(self, capital, mu, sigma, days=252, sims=1000):
        dt = 1/252
        drift = (mu - 0.5 * sigma**2) * dt
        diff = sigma * np.sqrt(dt)
        Z = np.random.normal(0, 1, (days, sims))
        ret_sim = np.exp(drift + diff * Z)
        
        paths = np.zeros((days+1, sims))
        paths[0] = capital
        for t in range(1, days+1): paths[t] = paths[t-1] * ret_sim[t-1]
            
        return np.arange(days+1), np.percentile(paths, 50, axis=1), np.percentile(paths, 95, axis=1), np.percentile(paths, 5, axis=1)

    def ask_ai(self, key, prompt):
        if not key: return "Sin API Key."
        try:
            genai.configure(api_key=key)
            return genai.GenerativeModel('models/gemini-flash-latest').generate_content(prompt).text
        except Exception as e: return f"Error IA: {e}"

    def export_excel(self, weights, metrics, perf):
        buf = io.BytesIO()
        with pd.ExcelWriter(buf, engine='xlsxwriter') as w:
            weights.to_excel(w, sheet_name='Pesos')
            perf.to_excel(w, sheet_name='Performance')
            pd.DataFrame([metrics]).to_excel(w, sheet_name='Metricas')
        return buf.getvalue()
    # ==========================================
    # 6. MÉTODOS ESPECIALES PARA LA API (WRAPPERS)
    # ==========================================
    def api_monte_carlo(self, tickers, weights, initial_capital, simulations=1000):
        """
        Función autónoma para la API: Descarga datos, calcula Mu/Sigma y corre la simulación.
        Devuelve JSON puro.
        """
        try:
            # 1. Obtener datos frescos (últimos 2 años para estimar volatilidad reciente)
            prices = self.get_market_data(tickers, start_date="2022-01-01", end_date=None)
            if prices.empty: return {"error": "No data found"}
            
            # 2. Calcular retornos logarítmicos
            log_rets = np.log(prices / prices.shift(1)).dropna()
            
            # 3. Calcular Mu y Sigma del PORTAFOLIO
            # Convertimos pesos a array en el orden correcto de las columnas
            w_list = [weights.get(col, 0) for col in prices.columns]
            w = np.array(w_list)
            
            # Retorno esperado diario (promedio simple) y Volatilidad diaria
            port_ret_daily = np.sum(log_rets.mean() * w)
            port_vol_daily = np.sqrt(np.dot(w.T, np.dot(log_rets.cov(), w)))
            
            # 4. Anualizar para la función run_monte_carlo (que usa inputs anualizados o diarios según tu lógica)
            # Tu función original 'run_monte_carlo' usa drift diario internamente (drift = mu - 0.5...),
            # pero asumamos que le pasamos los parámetros anualizados para mantener consistencia 
            # o pasamos los diarios y ajustamos. 
            # NOTA: Tu función 'run_monte_carlo' ya hace dt = 1/252. 
            # Pasemos Mu y Sigma ANUALIZADOS que es lo estándar.
            mu_annual = port_ret_daily * 252
            sigma_annual = port_vol_daily * np.sqrt(252)
            
            # 5. Correr Simulación
            days_arr, median, optimistic, pessimistic = self.run_monte_carlo(
                capital=initial_capital, 
                mu=mu_annual, 
                sigma=sigma_annual, 
                days=252, 
                sims=simulations
            )
            
            # 6. Retornar listas (JSON serializable)
            return {
                "days": days_arr.tolist(),
                "median": median.tolist(),
                "optimistic": optimistic.tolist(),
                "pessimistic": pessimistic.tolist()
            }
        except Exception as e:
            return {"error": str(e)}

    def api_benchmark(self, tickers, weights, start_date="2020-01-01"):
        """
        Función autónoma para la API: Reconstruye la historia y compara con SPY.
        """
        try:
            # 1. Obtener historia completa
            prices = self.get_market_data(tickers, start_date=start_date, end_date=None)
            
            # 2. Calcular curva del usuario
            # Reconstruimos los pesos en orden
            w_series = pd.Series(weights)
            # Filtramos solo activos validos
            valid_tickers = prices.columns.intersection(w_series.index)
            
            # Retornos ponderados
            rets = prices[valid_tickers].pct_change().dropna()
            port_ret = (rets * w_series[valid_tickers]).sum(axis=1)
            
            # 3. Usar tu función existente de benchmark
            # (Le pasamos capital=1 para obtener retornos normalizados)
            _, _, df_compare = self.align_and_benchmark(prices, port_ret, capital=100)
            
            # 4. Formatear para gráfica (índice fecha a string)
            df_compare.index = df_compare.index.strftime('%Y-%m-%d')
            
            return {
                "dates": df_compare.index.tolist(),
                "portfolio": df_compare['Portafolio Estratégico'].tolist(),
                "benchmark": df_compare['S&P 500 Benchmark'].tolist()
            }
        except Exception as e:
            return {"error": str(e)}