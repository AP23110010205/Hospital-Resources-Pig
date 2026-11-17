
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ==========================================
# HOSPITAL RESOURCE FAIRNESS ANALYSIS
# ==========================================

PROJECT = Path(__file__).resolve().parents[2]

DATA_FILE = (
    PROJECT / "data" / "raw" / "hospital_resources.csv"
)

RESULTS_DIR = PROJECT / "results"
GRAPH_DIR = RESULTS_DIR / "graphs"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
GRAPH_DIR.mkdir(parents=True, exist_ok=True)

# 1. Load hospital dataset

data = pd.read_csv(DATA_FILE)

# 2. Check required columns

required_columns = [
    "department",
    "available_beds",
    "occupied_beds",
    "staff_available",
    "staff_required",
    "age_group",
    "gender"
]

missing = [
    col for col in required_columns
    if col not in data.columns
]

if missing:
    raise ValueError(
        f"Missing required columns: {missing}"
    )

# 3. Calculate resource utilization

data["bed_occupancy_percentage"] = (
    data["occupied_beds"] /
    data["available_beds"].replace(0, pd.NA)
) * 100

data["bed_shortage"] = (
    data["occupied_beds"] - data["available_beds"]
).clip(lower=0)

data["staff_shortage"] = (
    data["staff_required"] - data["staff_available"]
).clip(lower=0)

# 4. Fairness analysis by age group

age_fairness = (
    data.groupby("age_group")
    .agg(
        records=("department", "count"),
        average_occupancy=("bed_occupancy_percentage", "mean"),
        average_bed_shortage=("bed_shortage", "mean"),
        average_staff_shortage=("staff_shortage", "mean")
    )
    .reset_index()
)

age_fairness.to_csv(
    RESULTS_DIR / "age_fairness_report.csv",
    index=False
)

# 5. Fairness analysis by gender

gender_fairness = (
    data.groupby("gender")
    .agg(
        records=("department", "count"),
        average_occupancy=("bed_occupancy_percentage", "mean"),
        average_bed_shortage=("bed_shortage", "mean"),
        average_staff_shortage=("staff_shortage", "mean")
    )
    .reset_index()
)

gender_fairness.to_csv(
    RESULTS_DIR / "gender_fairness_report.csv",
    index=False
)

# 6. Department-level resource analysis

department_analysis = (
    data.groupby("department")
    .agg(
        average_occupancy=("bed_occupancy_percentage", "mean"),
        average_bed_shortage=("bed_shortage", "mean"),
        average_staff_shortage=("staff_shortage", "mean")
    )
    .reset_index()
)

department_analysis.to_csv(
    RESULTS_DIR / "department_resource_report.csv",
    index=False
)

# 7. Calculate group differences

def calculate_gap(series):
    values = series.dropna()

    if len(values) < 2:
        return 0.0

    return float(values.max() - values.min())


age_occupancy_gap = calculate_gap(
    age_fairness["average_occupancy"]
)

gender_occupancy_gap = calculate_gap(
    gender_fairness["average_occupancy"]
)

age_staff_gap = calculate_gap(
    age_fairness["average_staff_shortage"]
)

gender_staff_gap = calculate_gap(
    gender_fairness["average_staff_shortage"]
)

# 8. Generate fairness summary

summary = pd.DataFrame([
    {
        "comparison": "Age group occupancy gap",
        "gap": age_occupancy_gap
    },
    {
        "comparison": "Gender occupancy gap",
        "gap": gender_occupancy_gap
    },
    {
        "comparison": "Age group staff shortage gap",
        "gap": age_staff_gap
    },
    {
        "comparison": "Gender staff shortage gap",
        "gap": gender_staff_gap
    }
])

summary.to_csv(
    RESULTS_DIR / "fairness_summary.csv",
    index=False
)

# 9. Display reports

print("\nAGE GROUP FAIRNESS ANALYSIS")
print("---------------------------")
print(age_fairness.to_string(index=False))

print("\nGENDER FAIRNESS ANALYSIS")
print("-----------------------")
print(gender_fairness.to_string(index=False))

print("\nDEPARTMENT RESOURCE ANALYSIS")
print("----------------------------")
print(department_analysis.to_string(index=False))

print("\nFAIRNESS GAP SUMMARY")
print("--------------------")
print(summary.to_string(index=False))

# 10. Create fairness charts

plt.figure(figsize=(8, 5))
plt.bar(
    age_fairness["age_group"],
    age_fairness["average_occupancy"]
)
plt.title("Average Bed Occupancy by Age Group")
plt.xlabel("Age Group")
plt.ylabel("Average Occupancy (%)")
plt.tight_layout()
plt.savefig(GRAPH_DIR / "age_group_occupancy.png")
plt.close()

plt.figure(figsize=(8, 5))
plt.bar(
    gender_fairness["gender"],
    gender_fairness["average_occupancy"]
)
plt.title("Average Bed Occupancy by Gender")
plt.xlabel("Gender")
plt.ylabel("Average Occupancy (%)")
plt.tight_layout()
plt.savefig(GRAPH_DIR / "gender_occupancy.png")
plt.close()

print("\nFairness analysis completed successfully!")
print("Reports saved in:", RESULTS_DIR)
print("Charts saved in:", GRAPH_DIR)
