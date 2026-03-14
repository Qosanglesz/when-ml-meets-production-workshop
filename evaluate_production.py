import joblib
import pandas as pd
from sklearn.metrics import accuracy_score


model = joblib.load("model.joblib")

prod = pd.read_csv("data/loan_production.csv")

X = prod.drop("default", axis=1)
y = prod["default"]

pred = model.predict(X)

print("Production accuracy:", accuracy_score(y, pred))
