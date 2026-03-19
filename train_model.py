import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


train = pd.read_csv("data/loan_train.csv")

X = train.drop("default", axis=1)
y = train["default"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = RandomForestClassifier()
model.fit(X_train, y_train)

pred = model.predict(X_test)

print("Training accuracy:", accuracy_score(y_test, pred))

# Save the model
joblib.dump(model, "model.joblib")
print("Model saved to model.joblib")