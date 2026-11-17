
import pandas as pd
from pathlib import Path
import math

# ==========================================
# HOSPITAL RESOURCE OPTIMIZATION
# ==========================================

PROJECT = Path(__file__).resolve().parents[2]

PREDICTION_FILE = (
    PROJECT / "results" / "demand_predictions.csv"
)

RAW_DATA_FILE = (
    PROJECT / "data" / "raw" / "hospital_resources.csv"
)

OUTPUT_DIR = PROJECT / "results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = OUTPUT_DIR / "resource_optimization.csv"

# 1. Read forecasting results

predictions = pd.read_csv(PREDICTION_FILE)

# 2. Read original hospital dataset

hospital_data = pd.read_csv(RAW_DATA_FILE)

# 3. Calculate average resources by department

department_resources = (
    hospital_data.groupby("department")
    .agg(
        available_beds=("available_beds", "mean"),
        occupied_beds=("occupied_beds", "mean"),
        staff_available=("staff_available", "mean"),
        staff_required=("staff_required", "mean")
    )
    .reset_index()
)

# 4. Combine predictions with resource information

result = predictions.merge(
    department_resources,
    on="department",
    how="left"
)

# 5. Estimate required beds and staff

# Demonstration planning assumptions
BED_BUFFER = 1.10
STAFF_PER_ADMISSION = 0.08

result["estimated_required_beds"] = (
    result["predicted_admissions"] * BED_BUFFER
).apply(math.ceil)

result["estimated_required_staff"] = (
    result["predicted_admissions"] * STAFF_PER_ADMISSION
).apply(math.ceil)

# 6. Calculate additional resources required

result["additional_beds_needed"] = (
    result["estimated_required_beds"]
    - result["available_beds"]
).clip(lower=0).apply(math.ceil)

result["additional_staff_needed"] = (
    result["estimated_required_staff"]
    - result["staff_available"]
).clip(lower=0).apply(math.ceil)

# 7. Generate recommendations

def recommendation(row):

    recommendations = []

    if row["additional_beds_needed"] > 0:
        recommendations.append("Allocate additional beds")

    if row["additional_staff_needed"] > 0:
        recommendations.append("Allocate additional staff")

    if not recommendations:
        return "Resources sufficient"

    return " and ".join(recommendations)


result["recommendation"] = result.apply(
    recommendation,
    axis=1
)

# 8. Save optimization results

result.to_csv(OUTPUT_FILE, index=False)

# 9. Display results

display_columns = [
    "department",
    "predicted_admissions",
    "estimated_required_beds",
    "additional_beds_needed",
    "estimated_required_staff",
    "additional_staff_needed",
    "recommendation"
]

print("\nHOSPITAL RESOURCE OPTIMIZATION RESULTS")
print("--------------------------------------")

print(result[display_columns].to_string(index=False))

print("\nOptimization results saved to:")
print(OUTPUT_FILE)
