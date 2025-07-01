from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import uvicorn
import DataPrep

app = FastAPI()

# Allow CORS for local frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class PredictionRequest(BaseModel):
    target_ticker: str
    related_tickers: List[str]
    prediction_days: int = 30

@app.post("/predict")
def predict_stock(req: PredictionRequest):
    predictions, stock_data, sentiment_data, model, metrics = DataPrep.run_stock_prediction(
        req.target_ticker, req.related_tickers, req.prediction_days
    )
    # Prepare response
    preds = predictions.to_dict(orient="records")
    current_price = float(stock_data['y'].iloc[-1])
    predicted_30d = float(predictions['Predicted_Price'].iloc[-1])
    change_30d = ((predicted_30d - current_price) / current_price) * 100
    return {
        "ticker": req.target_ticker,
        "current_price": current_price,
        "predicted_30d": predicted_30d,
        "change_30d": change_30d,
        "predictions": preds,
        "metrics": metrics
    }

if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
