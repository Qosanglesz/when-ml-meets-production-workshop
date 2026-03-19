import joblib
import pandas as pd
from sklearn.metrics import accuracy_score


prod = pd.read_csv("data/loan_production.csv")

X = prod.drop("default", axis=1)
y = prod["default"]

model_v1 = joblib.load("model_v1.joblib")
model_v2 = joblib.load("model_v2.joblib")

pred1 = model_v1.predict(X)
pred2 = model_v2.predict(X)

print("Model v1 accuracy:", accuracy_score(y, pred1))
print("Model v2 accuracy:", accuracy_score(y, pred2))