import pickle

from flask import Flask
from flask import request
from flask import jsonify

dv_file = 'dv.bin'
model_file = 'model2.bin'

with open(dv_file, 'rb') as dv_in:
    dv = pickle.load(dv_in)

with open(model_file, 'rb') as model_in:
    model = pickle.load(model_in)

app = Flask('card')

@app.route('/predict', methods=['POST'])
def predict():
    customer = request.get_json()
    X = dv.transform([customer])
    y_pred = model.predict_proba(X)[0,1]
    owner = y_pred >= 0.5

    result = {
        'ownership_probability': float(y_pred),
        'owner': bool(owner)
    }

    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)
