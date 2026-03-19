import pandas as pd
from evidently import Dataset, DataDefinition, Report
from evidently.presets import DataDriftPreset


ref = pd.read_csv("data/loan_train.csv")
cur = pd.read_csv("data/loan_production.csv")

schema = DataDefinition(
    numerical_columns=["age", "income", "loan_amount", "credit_score", "default"],
)

ref_dataset = Dataset.from_pandas(
    pd.DataFrame(ref),
    data_definition=schema
)

cur_dataset = Dataset.from_pandas(
    pd.DataFrame(cur),
    data_definition=schema
)

report = Report([
    DataDriftPreset(),
])
my_eval = report.run(reference_data=ref_dataset, current_data=cur_dataset)
my_eval.save_html("evidently_drift_report.html")

print("Drift report saved to drift_report.html")