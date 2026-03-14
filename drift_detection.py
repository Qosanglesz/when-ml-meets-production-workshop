import pandas as pd
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset


ref = pd.read_csv("data/loan_train.csv")
cur = pd.read_csv("data/loan_production.csv")

report = Report(metrics=[DataDriftPreset()])
report.run(reference_data=ref, current_data=cur)

report.save_html("drift_report.html")
print("Drift report saved to drift_report.html")
