from h2o_wave import main, app, Q, ui, on, run_on
import h2o
from h2o.automl import H2OAutoML

h2o.init()

aml = H2OAutoML.load_model("your_trained_model_path")

def predict_movie_type(title):
    try:
        df = h2o.H2OFrame({"title": [title]})
        prediction = aml.predict(df)
        return prediction[0, "predict"]
    except Exception as e:
        return f"Prediction failed. Error: {str(e)}"

def add_card(q, name, card) -> None:
    q.client.cards.add(name)
    q.page[name] = card

def clear_cards(q, ignore=[]):
    if not q.client.cards:
        return
    for name in q.client.cards.copy():
        if name not in ignore:
            del q.page[name]
            q.client.cards.remove(name)

async def init(q: Q) -> None:
    pass

@on('#page1')
async def page1(q: Q):
    clear_cards(q)
    add_card(q, 'prediction_form', ui.form_card(box='vertical', items=[
        ui.textbox(name='movie_title', label='Enter Movie Title:'),
        ui.buttons(items=[ui.button(name='predict_movie', label='Predict', primary=True)]),
        ui.text(name='predicted_type', content='', style='margin-top: 20px; font-weight: bold;')
    ]))

@on('predict_movie')
async def predict_movie(q: Q):
    movie_title = q.args.movie_title
    predicted_type = predict_movie_type(movie_title)

    q.page['prediction_form'].items[2].content = f'Predicted Movie Type: {predicted_type}'

@app('/movie')
async def serve(q: Q):
    if not q.client.initialized:
        q.client.cards = set()
        await init(q)
        q.client.initialized = True

    await run_on(q)
    await q.page.save()

if __name__ == '__main__':
    main()
