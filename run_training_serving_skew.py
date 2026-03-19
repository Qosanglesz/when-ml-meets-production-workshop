import joblib
import pandas as pd
from sklearn.metrics import accuracy_score


prod = pd.read_csv("data/loan_production_skew.csv")

X_prod = prod.drop("default", axis=1)
y_prod = prod["default"]

model = joblib.load("model.joblib")
pred = model.predict(X_prod)

print("Production accuracy:", accuracy_score(y_prod, pred))