# predictor.py
import h2o
import numpy as np

class Predictor:
    def __init__(self):
        pass

    def predict_rating(self, args):
        try:
            input_data = {
                'mfr': args.mfr,
                'type': args.type,
                'calories': args.calories,
                'protein': args.protein,
                'fat': args.fat,
                'sodium': args.sodium,
                'fiber': args.fiber,
                'carbo': args.carbo,
                'sugars': args.sugars,
                'potass': args.potass,
                'vitamins': args.vitamins,
            }

            input_data['mfr'] = str(input_data['mfr'])
            input_data['type'] = str(input_data['type'])

            input_frame = h2o.H2OFrame([[input_data[key] for key in input_data]])

            prediction = aml_leader.predict(input_frame)
            rating = prediction.as_data_frame().iloc[0]['predict']

            return round(rating, 2)
        except Exception as e:
            raise e
