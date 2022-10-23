import bentoml
from bentoml.io import NumpyNdarray

model_ref = bentoml.sklearn.get("mlzoomcamp_homework:qtzdz3slg6mwwdu5")
#model_ref2 = bentoml.sklearn.get("mlzoomcamp_homework:jsi67fslz6txydu5")

model_runner = model_ref.to_runner()
#model_runner2 = model_ref2.to_runner()

svc = bentoml.Service("credit_risk_classifier", runners=[model_runner])

@svc.api(input=NumpyNdarray(), output=NumpyNdarray())
def classify(application_data):
    prediction = model_runner.predict.run(application_data)
    return prediction
