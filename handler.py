import pandas as pd
from rossmann import Rossmann
from flask import Flask, request, Response
import os

app = Flask(__name__)
@app.route('/predict', methods=['POST'])

def rossmann_predict():
    df_raw_json = request.get_json()

    if df_raw_json:
        if isinstance(df_raw_json, dict):
            df_raw = pd.DataFrame(df_raw_json, index=[0])

        else:
            df_raw = pd.DataFrame(df_raw_json, columns=df_raw_json[0].keys())

        papeline = Rossmann()
        df = papeline.data_cleaning(df_raw)
        df = papeline.feature_engineering(df)
        df = papeline.data_filtering(df)
        df = papeline.data_preparation(df)
        df = papeline.get_predictions(df, df_raw)
        df_json = df.to_json(orient='records')

        return df_json

    else:
        return Response('{}', status=200, mimetype='application/json')


if __name__ == "__main__":
    port = os.environ.get('PORT', 5000)
    app.run(host='0.0.0.0', port=port)
