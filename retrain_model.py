import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier


train = pd.read_csv("data/loan_train.csv")
prod = pd.read_csv("data/loan_production.csv")

df = pd.concat([train, prod])

X = df.drop("default", axis=1)
y = df["default"]

model = RandomForestClassifier()
model.fit(X, y)

joblib.dump(model, "model_v2.joblib")
print("Saved model_v2.joblib")
