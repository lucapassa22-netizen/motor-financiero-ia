# main_api.py - VERSIÓN 2.0 (SAAS EDITION CON SUPABASE)
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

# --- 1. CONFIGURACIÓN DE CREDENCIALES (RENDER) ---
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Conexión a Supabase
if not SUPABASE_URL or not SUPABASE_KEY:
    print("⚠️ ADVERTENCIA: Faltan credenciales de Supabase. La seguridad fallará.")
    supabase: Client = None
else:
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

app = FastAPI(title="Financial Engine SaaS", version="2.0")

# --- 2. EL PORTERO (SEGURIDAD CON BASE DE DATOS) ---
# Reemplaza la función verify_api_key anterior por esta:

async def verify_api_key(x_api_key: str = Header(..., alias="X-API-Key")):
    print(f"\n--- 🕵️ DEBUG START ---")
    print(f"1. Llave recibida del Frontend: '{x_api_key}'")
    
    if not supabase:
        print("❌ Error: Cliente Supabase es None. Revisa credenciales.")
        raise HTTPException(status_code=500, detail="Error configuración DB")

    try:
        # Intentamos buscar la llave
        print(f"2. Consultando Supabase tabla 'user_api_keys'...")
        
        # Hacemos la consulta
        response = supabase.table("user_api_keys").select("*").eq("api_key", x_api_key).execute()
        
        print(f"3. Respuesta cruda de Supabase: {response}")
        print(f"4. Datos (data): {response.data}")
        
        # Verificar si se encontró algo
        if not response.data:
            print("⛔ Resultado: Lista vacía. La llave no coincide con ninguna fila.")
            raise HTTPException(status_code=403, detail="API Key no encontrada en DB")
        
        user_data = response.data[0]
        print(f"✅ Usuario encontrado: {user_data.get('user_id')}")
        
        if not user_data.get("is_active", True):
            print("⛔ Usuario inactivo.")
            raise HTTPException(status_code=403, detail="Tu plan está inactivo.")
            
        print("--- DEBUG END ---\n")
        return user_data 

    except Exception as e:
        print(f"🔥 EXCEPCIÓN CRÍTICA: {str(e)}")
        # Si ya es HTTPException, la relanzamos
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(status_code=500, detail=f"Error interno: {str(e)}")

def get_clean_data(tickers, start_date=None, period=None):
    """Descarga datos de forma segura evitando errores de columnas"""
    try:
        if period:
            df = yf.download(tickers, period=period, progress=False)
        else:
            df = yf.download(tickers, start=start_date, progress=False)
        
        # Corrección para el error 'Adj Close'
        if 'Adj Close' in df:
            return df['Adj Close']
        elif 'Close' in df:
            print("⚠️ 'Adj Close' no encontrado, usando 'Close'")
            return df['Close']
        else:
            # Caso raro: yfinance devuelve los datos sin multi-index si es 1 solo ticker
            return df
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

# --- 4. ENDPOINTS DE GESTIÓN (NUEVO) ---

@app.post("/api/v1/create_key")
def create_user_key(req: CreateKeyRequest):
    """Genera una nueva llave y la guarda en Supabase (Solo uso interno)"""
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

# --- 5. ENDPOINTS FINANCIEROS (PROTEGIDOS) ---

@app.get("/")
def home():
    return {"message": "Financial Engine v2.0 (SaaS Mode) - Connected to Supabase 🟢"}

@app.post("/api/v1/optimize")
def optimize(request: OptimizationRequest, user=Depends(verify_api_key)):
    try:
        data = get_clean_data(request.tickers, period="2y")
        if data.empty or data.shape[1] == 0: 
            raise HTTPException(404, "No se pudieron descargar datos de Yahoo Finance.")
        # Limpieza extra para evitar NaN
        data = data.dropna(axis=1, how='all').dropna()
        
        mu = expected_returns.mean_historical_return(data)
        S = risk_models.sample_cov(data)
        ef = EfficientFrontier(mu, S)
        
        if request.risk_profile == "Conservador":
            weights = ef.min_volatility()
        elif request.risk_profile == "Arriesgado":
            weights = ef.max_sharpe()
        else:
            # Moderado: Max Sharpe con restricción de volatilidad (simplificado aquí como max_sharpe standard)
            weights = ef.max_sharpe() 
            
        clean_weights = ef.clean_weights()
        perf = ef.portfolio_performance(verbose=False)
        
        return {
            "weights": clean_weights,
            "metrics": {"ret_anual": perf[0], "vol_anual": perf[1], "sharpe": perf[2]},
            "plan": user.get("plan_type")
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/backtest")
def backtest(request: BacktestRequest, user=Depends(verify_api_key)):
    try:
        end = request.end_date if request.end_date else datetime.today().strftime('%Y-%m-%d')
        from datetime import datetime
        df = yf.download(request.tickers, start=request.start_date)['Adj Close']
        
        # Normalizar y calcular
        norm_df = df / df.iloc[0]
        port_val = pd.Series(0, index=norm_df.index)
        for t, w in request.weights.items():
            if t in norm_df.columns:
                port_val += norm_df[t] * w
        
        equity_curve = port_val * request.initial_capital
        
        # Métricas
        total_ret = (equity_curve.iloc[-1] / request.initial_capital) - 1
        drawdown = (equity_curve - equity_curve.cummax()) / equity_curve.cummax()
        max_dd = drawdown.min()
        
        history = [{"date": str(d.date()), "value": v} for d, v in equity_curve.items()]
        
        return {
            "final_balance": round(equity_curve.iloc[-1], 2),
            "total_return_pct": round(total_ret * 100, 2),
            "max_drawdown_pct": round(max_dd * 100, 2),
            "history": history[::5] # Retornar 1 de cada 5 datos para no saturar
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/montecarlo")
def montecarlo(request: SimulationRequest, user=Depends(verify_api_key)):
    # Simulación Montecarlo simplificada dentro de la API
    try:
        data = yf.download(request.tickers, period="1y")['Adj Close']
        returns = data.pct_change().dropna()
        
        mean_daily_ret = 0
        var_daily_ret = 0
        for t, w in request.weights.items():
             if t in returns.columns:
                 mean_daily_ret += returns[t].mean() * w
                 var_daily_ret += (returns[t].std()**2) * (w**2)
        
        vol_daily = np.sqrt(var_daily_ret)
        days = 252
        
        sim_data = pd.DataFrame()
        for i in range(request.simulations):
            # Proyección geométrica browniana simple
            daily_noise = np.random.normal(mean_daily_ret, vol_daily, days)
            price_series = [request.initial_capital]
            for r in daily_noise:
                price_series.append(price_series[-1] * (1 + r))
            sim_data[f"sim_{i}"] = price_series
            
        return {
            "median": sim_data.median(axis=1).tolist(),
            "optimistic": sim_data.quantile(0.95, axis=1).tolist(),
            "pessimistic": sim_data.quantile(0.05, axis=1).tolist()
        }
    except Exception as e:
         raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/benchmark")
def benchmark(request: BenchmarkRequest, user=Depends(verify_api_key)):
    try:
        tickers_all = request.tickers + ["SPY"]
        df = yf.download(tickers_all, start=request.start_date)['Adj Close']
        df = df.dropna()
        norm = df / df.iloc[0]
        
        port_series = pd.Series(0, index=norm.index)
        for t, w in request.weights.items():
            if t in norm.columns:
                port_series += norm[t] * w
                
        return {
            "dates": [str(d.date()) for d in port_series.index],
            "portfolio": port_series.tolist(),
            "benchmark": norm["SPY"].tolist()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/analyze")
def analyze_ai(request: AIAnalysisRequest, user=Depends(verify_api_key)):
    try:
        genai.configure(api_key=request.api_key)
        model = genai.GenerativeModel('gemini-pro')
        prompt = f"Analiza este portafolio con perfil {request.risk_profile}: {request.weights}. Métricas: {request.metrics}. Dame 3 recomendaciones."
        response = model.generate_content(prompt)
        return {"ai_analysis": response.text}
    except Exception as e:
        return {"ai_analysis": "Error IA: " + str(e)}

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

    return StreamingResponse(output, headers={"Content-Disposition": "attachment; filename=reporte.xlsx"}, media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

