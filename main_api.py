# main_api.py (VERSIÓN 1.6.0 - SECURED B2B)
from fastapi import FastAPI, HTTPException, Security, Depends
from fastapi.security import APIKeyHeader
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from datetime import datetime
import pandas as pd
import numpy as np
import io
import os

# Importamos TU cerebro financiero
from financial_engine import FinancialEngine

# --- CONFIGURACIÓN DE SEGURIDAD (EL PORTERO) ---
API_KEY_NAME = "X-API-Key"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

# LISTA DE CLIENTES AUTORIZADOS (Esto en el futuro iría a una base de datos)
VALID_API_KEYS = [
    "PRUEBA_GRATIS_123",    # Para tus tests
    "CLIENTE_BANCO_A",      # Cliente real 1
    "CLIENTE_FONDO_B"       # Cliente real 2
]

# Función que verifica la llave
async def get_api_key(api_key_header: str = Security(api_key_header)):
    if api_key_header in VALID_API_KEYS:
        return api_key_header
    else:
        raise HTTPException(
            status_code=403, 
            detail="⛔ Acceso Denegado: API Key inválida o faltante."
        )

# --- INICIALIZACIÓN DE LA APP ---
# dependencies=[Depends(get_api_key)] protege TODAS las rutas automáticamente
app = FastAPI(
    title="Financial Engine API (B2B)",
    description="Motor financiero profesional securizado para integración bancaria.",
    version="1.6.0",
    dependencies=[Depends(get_api_key)] 
)

engine = FinancialEngine()

# --- MODELOS DE DATOS ---

class OptimizeRequest(BaseModel):
    tickers: List[str]
    risk_profile: str
    initial_capital: float = 10000.0

class BacktestRequest(BaseModel):
    tickers: List[str]
    weights: Dict[str, float]
    start_date: str = "2020-01-01"
    end_date: Optional[str] = None
    initial_capital: float = 10000.0

class AIAnalysisRequest(BaseModel):
    weights: Dict[str, float]
    metrics: Dict[str, float]
    risk_profile: str
    api_key: str # Esta es la key de Gemini (Google), distinta a la de tu API

class ExportRequest(BaseModel):
    weights: Dict[str, float]
    metrics: Dict[str, float]

class SimulationRequest(BaseModel):
    tickers: List[str]
    weights: Dict[str, float]
    initial_capital: float = 10000
    simulations: int = 500

class BenchmarkRequest(BaseModel):
    tickers: List[str]
    weights: Dict[str, float]
    start_date: str = "2020-01-01"

class PortfolioResponse(BaseModel):
    weights: Dict[str, float]
    metrics: Dict[str, float]
    status: Dict[str, Any]
    tickers_analyzed: List[str]

# --- ENDPOINTS ---

# Nota: Incluso el Home está protegido ahora.
@app.get("/")
def home():
    return {"status": "online", "message": "Financial Engine B2B Ready 🔒"}

@app.post("/api/v1/optimize", response_model=PortfolioResponse)
def optimize_portfolio(request: OptimizeRequest):
    try:
        if not request.tickers:
            raise HTTPException(status_code=400, detail="Lista de tickers vacía.")

        today = datetime.today().strftime('%Y-%m-%d')
        prices = engine.get_market_data(request.tickers, start_date="2020-01-01", end_date=today)
        
        if prices.empty:
            raise HTTPException(status_code=404, detail="No se pudieron descargar datos.")

        method = 'markowitz'
        objective = 'max_sharpe'
        if request.risk_profile == "Conservador":
            objective = 'min_volatility'
            
        weights, perf_tuple, status = engine.optimize_portfolio(
            prices, 
            method=method, 
            objective=objective,
            market_benchmark='SPY'
        )

        if not status['success']:
             raise HTTPException(status_code=400, detail=status['message'])

        metrics_dict = {
            "ret_anual": perf_tuple[0],
            "vol_anual": perf_tuple[1],
            "sharpe": perf_tuple[2]
        }
        
        clean_weights = {k: v for k, v in weights.items() if v > 0.001}

        return {
            "weights": clean_weights,
            "metrics": metrics_dict,
            "status": status,
            "tickers_analyzed": list(prices.columns)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/backtest")
def backtest_portfolio(request: BacktestRequest):
    try:
        end = request.end_date if request.end_date else datetime.today().strftime('%Y-%m-%d')
        prices = engine.get_market_data(request.tickers, start_date=request.start_date, end_date=end)
        
        if prices.empty:
            raise HTTPException(status_code=404, detail="No hay datos.")

        returns = prices.pct_change().dropna()
        
        portfolio_weights = []
        available_tickers = returns.columns.tolist()
        
        for ticker in available_tickers:
            w = request.weights.get(ticker, 0.0)
            portfolio_weights.append(w)
            
        weights_array = np.array(portfolio_weights)
        if weights_array.sum() > 0:
            weights_array = weights_array / weights_array.sum()
        else:
            raise HTTPException(status_code=400, detail="Pesos inválidos.")

        portfolio_returns = returns.dot(weights_array)
        cumulative_returns = (1 + portfolio_returns).cumprod()
        equity_curve = request.initial_capital * cumulative_returns
        
        final_balance = equity_curve.iloc[-1]
        total_return_pct = (final_balance - request.initial_capital) / request.initial_capital
        
        rolling_max = equity_curve.cummax()
        drawdown = (equity_curve - rolling_max) / rolling_max
        max_drawdown = drawdown.min()

        chart_data = []
        for date, value in equity_curve.iloc[::5].items():
            chart_data.append({
                "date": date.strftime('%Y-%m-%d'),
                "value": round(value, 2)
            })

        return {
            "initial_capital": request.initial_capital,
            "final_balance": round(final_balance, 2),
            "total_return_pct": round(total_return_pct * 100, 2),
            "max_drawdown_pct": round(max_drawdown * 100, 2),
            "history": chart_data
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/analyze")
def analyze_with_ai(request: AIAnalysisRequest):
    try:
        prompt = f"""
        Actúa como un Asesor Financiero experto. Analiza este portafolio:
        RIESGO: {request.risk_profile}
        PESOS: {request.weights}
        MÉTRICAS: {request.metrics}
        
        Dame:
        1. Comentario de diversificación.
        2. Riesgos específicos.
        3. Conclusión profesional.
        Responde en español y con formato Markdown.
        """
        analysis = engine.ask_ai(key=request.api_key, prompt=prompt)
        return {"ai_analysis": analysis}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error IA: {str(e)}")

@app.post("/api/v1/export")
def export_to_excel(request: ExportRequest):
    try:
        w_series = pd.Series(request.weights, name="Pesos")
        metrics_df = pd.DataFrame.from_dict(request.metrics, orient='index', columns=['Valor'])
        excel_binary = engine.export_excel(w_series, request.metrics, metrics_df)
        
        return StreamingResponse(
            io.BytesIO(excel_binary),
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            headers={"Content-Disposition": "attachment; filename=reporte_inversion.xlsx"}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error Excel: {str(e)}")
    
@app.post("/api/v1/montecarlo")
def endpoint_montecarlo(request: SimulationRequest):
    try:
        results = engine.api_monte_carlo(
            request.tickers, 
            request.weights, 
            request.initial_capital, 
            request.simulations
        )
        if "error" in results:
            raise HTTPException(status_code=500, detail=results["error"])
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/benchmark")
def endpoint_benchmark(request: BenchmarkRequest):
    try:
        results = engine.api_benchmark(
            request.tickers, 
            request.weights, 
            request.start_date
        )
        if "error" in results:
            raise HTTPException(status_code=500, detail=results["error"])
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))