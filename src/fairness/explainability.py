
import pandas as pd
from pathlib import Path

# ==========================================
# HOSPITAL RESOURCE EXPLAINABILITY
# ==========================================

PROJECT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    PROJECT / "results" / "resource_optimization.csv"
)

OUTPUT_FILE = (
    PROJECT / "results" / "explainability_report.csv"
)

# 1. Check input file

if not INPUT_FILE.exists():
    raise FileNotFoundError(
        f"Optimization results not found: {INPUT_FILE}"
    )

# 2. Read optimization results

data = pd.read_csv(INPUT_FILE)

# 3. Explain resource recommendations

def explain_recommendation(row):

    explanations = []

    # Explain bed requirement

    if row["additional_beds_needed"] > 0:
        explanations.append(
            f"Additional {int(row['additional_beds_needed'])} "
            "beds recommended because estimated bed demand "
            "exceeds available beds."
        )
    else:
        explanations.append(
            "No additional beds recommended because "
            "estimated bed demand is within available capacity."
        )

    # Explain staff requirement

    if row["additional_staff_needed"] > 0:
        explanations.append(
            f"Additional {int(row['additional_staff_needed'])} "
            "staff members recommended because estimated "
            "staff requirement exceeds available staff."
        )
    else:
        explanations.append(
            "No additional staff recommended because "
            "estimated staff requirement is within available capacity."
        )

    return " ".join(explanations)


# 4. Generate explanations

data["explanation"] = data.apply(
    explain_recommendation,
    axis=1
)

# 5. Create explainability report

report_columns = [
    "department",
    "predicted_admissions",
    "available_beds",
    "staff_available",
    "estimated_required_beds",
    "estimated_required_staff",
    "additional_beds_needed",
    "additional_staff_needed",
    "recommendation",
    "explanation"
]

report = data[report_columns].copy()

# 6. Save report

report.to_csv(
    OUTPUT_FILE,
    index=False
)

# 7. Display explanations

print("\nHOSPITAL RESOURCE EXPLAINABILITY REPORT")
print("---------------------------------------")

for _, row in report.iterrows():

    print(f"\nDepartment: {row['department']}")
    print(
        "Predicted admissions:",
        row["predicted_admissions"]
    )
    print(
        "Recommendation:",
        row["recommendation"]
    )
    print("Explanation:")
    print(row["explanation"])

print("\nExplainability analysis completed successfully!")
print("Report saved to:", OUTPUT_FILE)
