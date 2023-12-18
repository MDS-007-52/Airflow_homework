# <YOUR_IMPORTS>
import os
import dill
import pandas as pd
import json
from datetime import datetime
import logging

path = os.environ.get('PROJECT_PATH', '.')


def predict():
    # <YOUR_CODE>
    # Find the latest pickled model file and load it
    model_path = os.path.join(os.path.join(path, 'data'), 'models')
    model_files = [f for f in os.listdir(model_path) if os.path.isfile(os.path.join(model_path, f))]
    fn_latest_model = os.path.join(model_path, model_files[-1])
    with open(fn_latest_model, 'rb') as file:
        predict_model = dill.load(file)

    # Load list of json files for applying our prediction model
    test_path = os.path.join(os.path.join(path, 'data'), 'test')
    test_files = [os.path.join(test_path, f) for f in os.listdir(test_path)
                  if os.path.isfile(os.path.join(test_path, f))]
    with open(test_files[0], 'rb') as tf:
        test_columns = dict(json.load(tf)).keys()
    test_series = [pd.read_json(f, typ='series') for f in test_files]
    test_df = pd.DataFrame(test_series, columns=test_columns)

    predicted_data = pd.Series(predict_model.predict(test_df))
    pred_df = pd.concat([test_df['id'], predicted_data], axis=1).rename(columns={'id': 'car_id', 0: 'pred'})
    pred_fn = os.path.join(os.path.join(os.path.join(path, 'data'),
                                        'predictions'),
                           f'cars_pipe_{datetime.now().strftime("%Y%m%d%H%M")}.csv')
    logging.info(pred_df.to_string())
    pred_df.to_csv(pred_fn, index=None)


if __name__ == '__main__':
    predict()
