# train.py
from h2o_wave import Q, ui
import h2o
from h2o.automl import H2OAutoML

def train_model():
    h2o.init()
    df = h2o.import_file('./cereal.csv') 
    train, test = df.split_frame(ratios=[0.8], seed=10)

    response_col = 'rating'
    predictor_cols = df.columns.remove(response_col)

    aml = H2OAutoML(
        max_models=10,
        seed=10,
        stopping_tolerance=0.012,
        stopping_rounds=3,
        nfolds=8
    )

    aml.train(x=predictor_cols, y=response_col, training_frame=train)

    model_path = "./model/" 
    aml_path = h2o.save_model(model=aml.leader, path=model_path, force=True)

    lb = aml.leaderboard
    print(lb)

    preds = aml.predict(test)

    h2o.cluster().shutdown(prompt=False)

    print(f"Trained model saved at: {aml_path}")

if __name__ == '__main__':
    train_model()
