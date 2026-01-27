# main_api.py (VERSIÓN 1.5.0 - COMPLETA CON EXCEL)
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from datetime import datetime
import pandas as pd
import numpy as np
import io # Necesario para manejar el archivo en memoria

# Importamos TU cerebro financiero existente
from financial_engine import FinancialEngine

app = FastAPI(
    title="Financial Engine API",
    description="API Completa: Optimización, Backtest, IA y Reportes.",
    version="1.5.0"
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
    api_key: str

# NUEVO MODELO PARA EXCEL
class ExportRequest(BaseModel):
    weights: Dict[str, float]
    metrics: Dict[str, float]

class PortfolioResponse(BaseModel):
    weights: Dict[str, float]
    metrics: Dict[str, float]
    status: Dict[str, Any]
    tickers_analyzed: List[str]

# --- ENDPOINTS ---

@app.get("/")
def home():
    return {"status": "online", "message": "All systems operational 🚀"}

# 1. OPTIMIZAR
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

# 2. BACKTEST
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

# 3. CONSULTOR IA
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

# 4. EXPORTAR EXCEL (NUEVO)
@app.post("/api/v1/export")
def export_to_excel(request: ExportRequest):
    """
    Genera un archivo Excel descargable con el portafolio.
    """
    try:
        # Convertimos los diccionarios JSON a Pandas Series/DataFrames
        # Esto es necesario porque engine.export_excel espera objetos de Pandas
        w_series = pd.Series(request.weights, name="Pesos")
        
        # Creamos un DataFrame para performance (visualización limpia)
        metrics_df = pd.DataFrame.from_dict(request.metrics, orient='index', columns=['Valor'])
        
        # Llamamos a la función de tu motor
        # Nota: Pasamos metrics_df dos veces porque tu motor pide (weights, metrics, perf)
        # y metrics ya contiene la data de perf, así que reutilizamos para que no falle.
        excel_binary = engine.export_excel(w_series, request.metrics, metrics_df)
        
        # Devolvemos el archivo como un "stream" para que el navegador lo descargue
        return StreamingResponse(
            io.BytesIO(excel_binary),
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            headers={"Content-Disposition": "attachment; filename=reporte_inversion.xlsx"}
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error Excel: {str(e)}")