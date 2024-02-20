import h2o
from h2o.automl import H2OAutoML
import pandas as pd

h2o.init()

file_path = r'netflix_titles.csv'
df = pd.read_csv(file_path)
h2o_df = h2o.H2OFrame(df)
h2o_df['listed_in'] = h2o_df['listed_in'].asfactor()

train, test = h2o_df.split_frame(ratios=[0.8], seed=42)
x = h2o_df.names[2:-1]
y = 'listed_in'

aml = H2OAutoML(max_models=5, seed=1)
print('before_train')
aml.train(x=x, y=y, training_frame=train)
print('train')
model_path = "./best_model/"
aml_path = h2o.save_model(model=aml.leader, path=model_path, force=True)
lb = aml.leaderboard
print(lb)
preds = aml.predict(test)

h2o.cluster().shutdown()

print(f"Trained model saved at: {aml_path}")
