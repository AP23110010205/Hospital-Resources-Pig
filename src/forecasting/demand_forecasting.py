
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor

# ==========================================
# HOSPITAL DEMAND FORECASTING
# ==========================================

# Project paths
PROJECT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    PROJECT / "data" / "processed" /
    "resource_metrics" / "part-m-00000"
)

OUTPUT_DIR = PROJECT / "results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = OUTPUT_DIR / "demand_predictions.csv"

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
        f"Pig output file not found: {INPUT_FILE}"
    )

data = pd.read_csv(INPUT_FILE, names=columns)

data["date"] = pd.to_datetime(data["date"])

# 2. Aggregate admissions by date and department

daily_data = (
    data.groupby(["date", "department"], as_index=False)
    ["admissions"].sum()
)

# 3. Create date-based features

daily_data["day"] = daily_data["date"].dt.day
daily_data["month"] = daily_data["date"].dt.month
daily_data["day_of_week"] = daily_data["date"].dt.dayofweek

# 4. Train model and predict next-day admissions

predictions = []

for department in daily_data["department"].unique():

    dept_data = daily_data[
        daily_data["department"] == department
    ].copy()

    X = dept_data[
        ["day", "month", "day_of_week"]
    ]

    y = dept_data["admissions"]

    model = RandomForestRegressor(
        n_estimators=100,
        random_state=42
    )

    model.fit(X, y)

    next_date = (
        dept_data["date"].max() + pd.Timedelta(days=1)
    )

    next_features = pd.DataFrame({
        "day": [next_date.day],
        "month": [next_date.month],
        "day_of_week": [next_date.dayofweek]
    })

    predicted_admissions = model.predict(
        next_features
    )[0]

    predictions.append({
        "date": next_date.date(),
        "department": department,
        "predicted_admissions": round(
            max(0, predicted_admissions), 2
        )
    })

# 5. Save predictions

prediction_data = pd.DataFrame(predictions)

prediction_data.to_csv(
    OUTPUT_FILE,
    index=False
)

# 6. Display results

print("\nHOSPITAL DEMAND FORECASTING RESULTS")
print("-----------------------------------")

print(prediction_data.to_string(index=False))

print("\nTotal departments:", len(prediction_data))
print("Predictions saved to:", OUTPUT_FILE)
