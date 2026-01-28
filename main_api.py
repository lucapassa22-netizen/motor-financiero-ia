# main_api.py - VERSIÓN 2.1 (SAAS EDITION FIX)
from fastapi import FastAPI, HTTPException, Depends, Header
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Dict, Optional, Any
import pandas as pd
import numpy as np
import yfinance as yf
from pypfopt import EfficientFrontier, risk_models, expected_returns
import google.generativeai as genai
import os
import io
import xlsxwriter
from supabase import create_client, Client
from datetime import datetime # MOVIDO AQUÍ PARA EVITAR ERRORES

# --- 1. CONFIGURACIÓN DE CREDENCIALES (RENDER) ---
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Conexión a Supabase
supabase: Client = None
if not SUPABASE_URL or not SUPABASE_KEY:
    print("⚠️ ADVERTENCIA: Faltan credenciales de Supabase. La validación de usuarios podría fallar.")
else:
    try:
        supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
        print("✅ Conexión a Supabase inicializada.")
    except Exception as e:
        print(f"❌ Error conectando a Supabase: {e}")

app = FastAPI(title="Financial Engine SaaS", version="2.1")

# --- 2. EL PORTERO (SEGURIDAD CON BASE DE DATOS) ---
async def verify_api_key(x_api_key: str = Header(..., alias="X-API-Key")):
    # MODO DEBUG: Si no hay Supabase configurado, dejamos pasar una llave maestra para pruebas
    if not supabase:
        if x_api_key == "sk_live_master_key_123":
            return {"user_id": "test_user", "plan_type": "admin", "is_active": True}
        raise HTTPException(status_code=500, detail="Error: Backend sin conexión a Base de Datos.")

    try:
        # Consultando Supabase
        response = supabase.table("user_api_keys").select("*").eq("api_key", x_api_key).execute()
        
        if not response.data:
            raise HTTPException(status_code=403, detail="API Key no válida o no encontrada.")
        
        user_data = response.data[0]
        
        if not user_data.get("is_active", True):
            raise HTTPException(status_code=403, detail="Tu plan está inactivo.")
            
        return user_data 

    except Exception as e:
        # Si ya es HTTPException, la relanzamos
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(status_code=500, detail=f"Error interno auth: {str(e)}")

def get_clean_data(tickers, start_date=None, period=None):
    """Descarga datos de forma robusta."""
    try:
        # Aseguramos que sea una lista única
        tickers = list(set(tickers))
        
        if period:
            df = yf.download(tickers, period=period, progress=False, auto_adjust=False)
        else:
            df = yf.download(tickers, start=start_date, progress=False, auto_adjust=False)
        
        if df.empty:
            return pd.DataFrame()

        # MANEJO DE ESTRUCTURA YFINANCE (El punto crítico)
        # yfinance devuelve MultiIndex si hay >1 ticker, y simple si hay 1.
        
        # 1. Intentamos obtener 'Adj Close', si no 'Close'
        if 'Adj Close' in df:
            prices = df['Adj Close']
        elif 'Close' in df:
            prices = df['Close']
        else:
            # Si df ya es directamente los precios (caso raro)
            prices = df

        # 2. Si descargamos 1 solo ticker, prices es una Series. Lo convertimos a DataFrame.
        if isinstance(prices, pd.Series):
            prices = prices.to_frame(name=tickers[0])
            
        # 3. Limpieza final
        prices = prices.dropna(how='all').fillna(method='ffill').dropna()
        return prices

    except Exception as e:
        print(f"Error descargando datos: {e}")
        return pd.DataFrame()

# --- 3. MODELOS DE DATOS ---
class OptimizationRequest(BaseModel):
    tickers: List[str]
    risk_profile: str
    initial_capital: float = 10000

class BacktestRequest(BaseModel):
    tickers: List[str]
    weights: Dict[str, float]
    start_date: str = "2020-01-01"
    end_date: Optional[str] = None
    initial_capital: float = 10000

class SimulationRequest(BaseModel):
    tickers: List[str]
    weights: Dict[str, float]
    initial_capital: float = 10000
    simulations: int = 500

class BenchmarkRequest(BaseModel):
    tickers: List[str]
    weights: Dict[str, float]
    start_date: str = "2021-01-01"

class AIAnalysisRequest(BaseModel):
    weights: Dict[str, float]
    metrics: Dict[str, float]
    risk_profile: str
    api_key: str 

class ExportRequest(BaseModel):
    weights: Dict[str, float]
    metrics: Dict[str, float]

class CreateKeyRequest(BaseModel):
    user_id: str
    plan_type: str = "free"

# --- 4. ENDPOINTS ---

@app.post("/api/v1/create_key")
def create_user_key(req: CreateKeyRequest):
    """Genera una nueva llave (Solo uso interno/Admin)"""
    import secrets
    new_key = f"sk_live_{secrets.token_hex(16)}"
    
    if supabase:
        try:
            data = {
                "user_id": req.user_id,
                "api_key": new_key,
                "plan_type": req.plan_type,
                "is_active": True
            }
            supabase.table("user_api_keys").insert(data).execute()
            return {"status": "ok", "api_key": new_key}
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))
    return {"error": "No DB connection"}

@app.get("/")
def home():
    status = "Connected to Supabase 🟢" if supabase else "No DB 🔴"
    return {"message": f"Financial Engine SaaS v2.1 - {status}"}

@app.post("/api/v1/optimize")
def optimize(request: OptimizationRequest, user=Depends(verify_api_key)):
    try:
        data = get_clean_data(request.tickers, period="2y")
        if data.empty: 
            raise HTTPException(404, "No se pudieron descargar datos.")
        
        # Filtramos tickers que realmente bajaron
        valid_tickers = data.columns.tolist()
        
        mu = expected_returns.mean_historical_return(data)
        S = risk_models.sample_cov(data)
        ef = EfficientFrontier(mu, S)
        
        if request.risk_profile == "Conservador":
            weights = ef.min_volatility()
        elif request.risk_profile == "Arriesgado":
            weights = ef.max_sharpe()
        else:
            weights = ef.max_sharpe() 
            
        clean_weights = ef.clean_weights()
        perf = ef.portfolio_performance(verbose=False)
        
        return {
            "weights": clean_weights,
            "metrics": {"ret_anual": perf[0], "vol_anual": perf[1], "sharpe": perf[2]},
            "plan": user.get("plan_type")
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Opt error: {str(e)}")

@app.post("/api/v1/backtest")
def backtest(request: BacktestRequest, user=Depends(verify_api_key)):
    try:
        # Usamos get_clean_data para consistencia
        df = get_clean_data(request.tickers, start_date=request.start_date)
        
        if df.empty: raise HTTPException(404, "Sin datos para backtest")

        # Normalizar a base 1.0
        norm_df = df / df.iloc[0]
        port_val = pd.Series(0, index=norm_df.index)
        
        for t, w in request.weights.items():
            if t in norm_df.columns:
                port_val += norm_df[t] * w
        
        equity_curve = port_val * request.initial_capital
        
        total_ret = (equity_curve.iloc[-1] / request.initial_capital) - 1
        cummax = equity_curve.cummax()
        drawdown = (equity_curve - cummax) / cummax
        max_dd = drawdown.min()
        
        # Reset index para que JSON serialice fechas
        hist_df = equity_curve.reset_index()
        hist_df.columns = ['date', 'value']
        # Convertimos fecha a string
        history = hist_df.apply(lambda x: {"date": x['date'].strftime('%Y-%m-%d'), "value": x['value']}, axis=1).tolist()
        
        return {
            "final_balance": round(equity_curve.iloc[-1], 2),
            "total_return_pct": round(total_ret * 100, 2),
            "max_drawdown_pct": round(max_dd * 100, 2),
            "history": history[::5] # Downsample
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Backtest error: {str(e)}")

@app.post("/api/v1/montecarlo")
def montecarlo(request: SimulationRequest, user=Depends(verify_api_key)):
    try:
        # Usamos get_clean_data para consistencia
        data = get_clean_data(request.tickers, period="1y")
        returns = data.pct_change().dropna()
        
        mean_daily_ret = 0
        var_daily_ret = 0
        
        # Calculamos mu y sigma del portafolio
        w_list = np.array([request.weights.get(t, 0) for t in data.columns])
        
        # Return portfolio diario
        port_rets = returns.dot(w_list)
        
        mean = port_rets.mean()
        std = port_rets.std()
        
        days = 252
        sim_data = pd.DataFrame()
        
        # Vectorizado para velocidad (mucho más rápido que bucle for)
        # Z es una matriz (days, simulations)
        Z = np.random.normal(mean, std, (days, request.simulations))
        daily_returns_sim = 1 + Z
        
        price_paths = np.vstack([np.ones((1, request.simulations)) * request.initial_capital, np.zeros((days, request.simulations))])
        
        # Acumulamos retornos
        for t in range(1, days + 1):
            price_paths[t] = price_paths[t-1] * daily_returns_sim[t-1]
            
        final_paths = pd.DataFrame(price_paths)
        
        return {
            "median": final_paths.median(axis=1).tolist(),
            "optimistic": final_paths.quantile(0.95, axis=1).tolist(),
            "pessimistic": final_paths.quantile(0.05, axis=1).tolist()
        }
    except Exception as e:
         raise HTTPException(status_code=500, detail=f"Montecarlo error: {str(e)}")

@app.post("/api/v1/benchmark")
def benchmark(request: BenchmarkRequest, user=Depends(verify_api_key)):
    try:
        tickers_all = list(set(request.tickers + ["SPY"]))
        df = get_clean_data(tickers_all, start_date=request.start_date)
        
        # Filtro de seguridad
        if "SPY" not in df.columns:
            raise HTTPException(500, "No se pudo descargar SPY para benchmark")

        df = df.dropna()
        norm = df / df.iloc[0]
        
        port_series = pd.Series(0, index=norm.index)
        for t, w in request.weights.items():
            if t in norm.columns:
                port_series += norm[t] * w
                
        return {
            "dates": [d.strftime('%Y-%m-%d') for d in port_series.index],
            "portfolio": port_series.tolist(),
            "benchmark": norm["SPY"].tolist()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/analyze")
def analyze_ai(request: AIAnalysisRequest, user=Depends(verify_api_key)):
    try:
        genai.configure(api_key=request.api_key)
        
        # Prompt estándar
        prompt = f"""
        Actúa como un experto financiero. Analiza brevemente:
        - Perfil: {request.risk_profile}
        - Pesos: {request.weights}
        - Métricas: {request.metrics}
        Dame 3 puntos clave (pros/contras) y una conclusión.
        """

        try:
            # INTENTO 1: Modelo moderno (Rápido y barato)
            model = genai.GenerativeModel('gemini-1.5-flash')
            response = model.generate_content(prompt)
            return {"ai_analysis": response.text}
            
        except Exception as e_flash:
            print(f"⚠️ Falló gemini-1.5-flash: {e_flash}")
            
            # INTENTO 2: Fallback al modelo clásico
            try:
                model = genai.GenerativeModel('gemini-pro')
                response = model.generate_content(prompt)
                return {"ai_analysis": response.text}
            except Exception as e_pro:
                # SI TODO FALLA: Listar qué modelos ve el servidor
                try:
                    available = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
                    error_msg = f"Error de Modelos. Disponibles en el servidor: {available}"
                except:
                    error_msg = f"Error crítico IA: {str(e_pro)}"
                
                print(error_msg)
                return {"ai_analysis": error_msg}

    except Exception as e:
        return {"ai_analysis": f"Error configuración IA: {str(e)}"}

@app.post("/api/v1/export")
def export_excel(request: ExportRequest, user=Depends(verify_api_key)):
    output = io.BytesIO()
    workbook = xlsxwriter.Workbook(output)
    ws = workbook.add_worksheet()
    ws.write(0, 0, "Ticker")
    ws.write(0, 1, "Peso")
    row = 1
    for t, w in request.weights.items():
        ws.write(row, 0, t)
        ws.write(row, 1, w)
        row += 1
    workbook.close()
    output.seek(0)

    return StreamingResponse(
        output, 
        headers={"Content-Disposition": "attachment; filename=reporte.xlsx"}, 
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )


