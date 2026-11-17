
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error
)
import numpy as np

# ==========================================
# HOSPITAL DEMAND MODEL EVALUATION
# ==========================================

PROJECT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    PROJECT / "data" / "processed" /
    "resource_metrics" / "part-m-00000"
)

RESULTS_DIR = PROJECT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = RESULTS_DIR / "model_evaluation.csv"

# 1. Read processed Pig data

columns = [
    "date",
    "department",
    "admissions",
    "available_beds",
    "occupied_beds",
    "occupancy_percentage",
    "bed_shortage",
    "staff_shortage",
    "age_group",
    "gender"
]

if not INPUT_FILE.exists():
    raise FileNotFoundError(
        f"Pig output not found: {INPUT_FILE}"
    )

data = pd.read_csv(INPUT_FILE, names=columns)

data["date"] = pd.to_datetime(data["date"])

# 2. Prepare date-based features

data["day"] = data["date"].dt.day
data["month"] = data["date"].dt.month
data["day_of_week"] = data["date"].dt.dayofweek

feature_columns = [
    "day",
    "month",
    "day_of_week"
]

all_results = []

# 3. Evaluate each department separately

for department in sorted(data["department"].unique()):

    dept_data = (
        data[data["department"] == department]
        .sort_values("date")
        .copy()
    )

    # Aggregate admissions by date
    dept_data = (
        dept_data.groupby("date", as_index=False)
        ["admissions"].sum()
    )

    dept_data["day"] = dept_data["date"].dt.day
    dept_data["month"] = dept_data["date"].dt.month
    dept_data["day_of_week"] = (
        dept_data["date"].dt.dayofweek
    )

    # Require at least 3 dates
    if len(dept_data) < 3:
        print(
            f"Skipping {department}: "
            "not enough dates for evaluation."
        )
        continue

    # First 80% for training, last 20% for testing
    split_index = int(len(dept_data) * 0.8)
    split_index = max(2, split_index)

    train_data = dept_data.iloc[:split_index]
    test_data = dept_data.iloc[split_index:]

    if test_data.empty:
        print(
            f"Skipping {department}: "
            "no test records available."
        )
        continue

    X_train = train_data[feature_columns]
    y_train = train_data["admissions"]

    X_test = test_data[feature_columns]
    y_test = test_data["admissions"]

    # Train model
    model = RandomForestRegressor(
        n_estimators=100,
        random_state=42
    )

    model.fit(X_train, y_train)

    # Predict unseen dates
    predictions = model.predict(X_test)

    # Calculate evaluation metrics
    mae = mean_absolute_error(
        y_test,
        predictions
    )

    mse = mean_squared_error(
        y_test,
        predictions
    )

    rmse = np.sqrt(mse)

    # Store actual vs predicted values
    for i in range(len(test_data)):

        all_results.append({
            "department": department,
            "date": test_data.iloc[i]["date"].date(),
            "actual_admissions": int(y_test.iloc[i]),
            "predicted_admissions": round(
                float(predictions[i]), 2
            ),
            "absolute_error": round(
                abs(
                    float(y_test.iloc[i]) -
                    float(predictions[i])
                ), 2
            ),
            "MAE": round(float(mae), 2),
            "MSE": round(float(mse), 2),
            "RMSE": round(float(rmse), 2)
        })

# 4. Save evaluation results

if not all_results:
    raise ValueError(
        "No evaluation results generated."
    )

evaluation = pd.DataFrame(all_results)

evaluation.to_csv(
    OUTPUT_FILE,
    index=False
)

# 5. Display results

print("\nHOSPITAL DEMAND MODEL EVALUATION")
print("--------------------------------")

print(
    evaluation[
        [
            "department",
            "date",
            "actual_admissions",
            "predicted_admissions",
            "absolute_error"
        ]
    ].to_string(index=False)
)

print("\nEVALUATION METRICS")
print("------------------")

metrics = (
    evaluation.groupby("department")
    [["MAE", "MSE", "RMSE"]]
    .first()
)

print(metrics.to_string())

print("\nEvaluation completed successfully!")
print("Results saved to:", OUTPUT_FILE)
