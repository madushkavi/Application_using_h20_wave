from fastapi import Depends, FastAPI, HTTPException
from h2o_wave import Q, ui
from predictor import Predictor
from fastapi.responses import JSONResponse
from pydantic import BaseModel

app = FastAPI()

predictor = Predictor()

class PredictionRequest(BaseModel):
    mfr: str
    type: str
    calories: int

class PredictionResponse(BaseModel):
    predicted_rating: float

@app.post('/cereal', response_model=PredictionResponse)
async def serve(q: Q = Depends(), request: PredictionRequest = Depends()):
    prediction = None  # Initialize prediction variable
    if q.args.predict:
        try:
            prediction = predictor.predict_rating(request)
            q.page['result'] = ui.text(f'Predicted Rating: {prediction}')
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    content = ui.form_card(
        box='1 1 10 10',
        items=[
            ui.choice(label='Manufacturer', name='mfr', choices=['A', 'B', 'C']),
            ui.choice(label='Type', name='type', choices=['C', 'H']),
            ui.int(label='Calories', name='calories'),
            ui.button(name='predict', label='Predict'),
        ],
    )

    if 'result' in q.page:
        content += q.page['result']

    q.page['content'] = content

    return PredictionResponse(predicted_rating=prediction)
