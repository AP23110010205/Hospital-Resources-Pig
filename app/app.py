import streamlit as st
import pandas as pd
from pathlib import Path

st.set_page_config(
    page_title="Hospital Resource Management",
    page_icon="🏥",
    layout="wide"
)


st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #eaf4ff, #e4faf5);
}
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #063b65, #087f8c);
}
[data-testid="stSidebar"] * {
    color: white !important;
}
h1 {
    color: #075985 !important;
    font-weight: 800 !important;
}
h2, h3 {
    color: #087e8b !important;
}
[data-testid="stMetric"] {
    background: linear-gradient(135deg, #ffffff, #d9f4f0);
    border: 1px solid #a5dcd5;
    padding: 20px;
    border-radius: 15px;
    box-shadow: 0 4px 12px rgba(0,70,100,0.12);
}
.stButton > button,
.stFormSubmitButton > button {
    background: linear-gradient(90deg, #087e8b, #0eaaa0);
    color: white;
    border-radius: 10px;
    border: none;
    font-weight: bold;
}
.stButton > button:hover {
    background: #075985;
    color: white;
}
</style>
""", unsafe_allow_html=True)

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
DATA = ROOT / "data" / "raw" / "hospital_resources.csv"

st.title("🏥 Hospital Resource Management")
st.subheader("Smart Hospital Analytics Dashboard")

st.markdown("""
Welcome to our Hospital Resource Management project.

This application uses Big Data Analytics with Hadoop,
Apache Pig and Python to analyze hospital resources,
forecast patient demand and support resource optimization.
""")

st.sidebar.title("🏥 Hospital Analytics")

page = st.sidebar.radio("Choose a page", [
    "Home",
    "Dashboard",
    "Hospital Dataset",
    "Demand Forecasting",
    "Resource Optimization",
    "Fairness Analysis",
    "Model Evaluation",
    "Explainability",
    "Hadoop & Pig Analytics",
    "Patient & Resource Input"
])

def load_report(filename):
    path = RESULTS / filename
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()

def show_report(df):
    if df.empty:
        st.warning("No report data found.")
    else:
        st.dataframe(df, use_container_width=True)
if page == "Home":
    st.header("Welcome to Hospital Analytics")
    st.success("Hospital Resource Management Website is Running!")

    col1, col2, col3 = st.columns(3)
    col1.metric("Departments", "4")
    col2.metric("Technology", "Hadoop + Pig")
    col3.metric("Project Status", "Running")

    st.subheader("Project Modules")
    st.write("""
    - Hospital Dataset Analysis
    - Patient Demand Forecasting
    - Resource Optimization
    - Fairness Analysis
    - Model Evaluation
    - Model Explainability
    - Hadoop and Apache Pig Analytics
    """)

elif page == "Dashboard":
    st.header("Hospital Dashboard")
    df = load_report("resource_optimization.csv")

    if not df.empty:
        col1, col2, col3 = st.columns(3)
        col1.metric("Departments", df["department"].nunique())
        col2.metric(
            "Predicted Admissions",
            round(df["predicted_admissions"].sum(), 2)
        )
        col3.metric(
            "Available Beds",
            int(df["available_beds"].sum())
        )

        st.subheader("Department Resource Report")
        st.dataframe(df, use_container_width=True)

        if "predicted_admissions" in df.columns:
            st.bar_chart(
                df.set_index("department")["predicted_admissions"]
            )
    else:
        st.warning("Resource optimization report not found.")

elif page == "Hospital Dataset":
    st.header("Hospital Dataset")

    if DATA.exists():
        df = pd.read_csv(DATA)
        st.metric("Total Records", len(df))
        st.dataframe(df, use_container_width=True)
        st.download_button(
            "Download Dataset",
            df.to_csv(index=False),
            "hospital_resources.csv"
        )
    else:
        st.error("Hospital dataset not found.")
elif page == "Demand Forecasting":
    st.header("Patient Demand Forecasting")
    df = load_report("demand_predictions.csv")
    show_report(df)

    if not df.empty:
        numeric_cols = df.select_dtypes(include="number").columns
        if len(numeric_cols) > 0:
            selected = st.selectbox("Select prediction value", numeric_cols)
            st.line_chart(df[selected])

elif page == "Resource Optimization":
    st.header("Hospital Resource Optimization")
    df = load_report("resource_optimization.csv")
    show_report(df)

    if not df.empty:
        if "department" in df.columns:
            st.subheader("Predicted Admissions")
            st.bar_chart(
                df.set_index("department")["predicted_admissions"]
            )

elif page == "Fairness Analysis":
    st.header("Fairness Analysis")

    st.subheader("Fairness Summary")
    show_report(load_report("fairness_summary.csv"))

    st.subheader("Age Group Analysis")
    show_report(load_report("age_fairness_report.csv"))

    st.subheader("Gender Analysis")
    show_report(load_report("gender_fairness_report.csv"))

    graphs = RESULTS / "graphs"
    for filename, title in [
        ("age_group_occupancy.png", "Age Group Occupancy"),
        ("gender_occupancy.png", "Gender Occupancy")
    ]:
        image_path = graphs / filename
        if image_path.exists():
            st.subheader(title)
            st.image(str(image_path), use_container_width=True)
elif page == "Model Evaluation":
    st.header("Model Evaluation")
    df = load_report("model_evaluation.csv")
    show_report(df)

    if not df.empty and "department" in df.columns:
        for column in ["MAE", "RMSE"]:
            if column in df.columns:
                st.subheader(column + " by Department")
                st.bar_chart(df.set_index("department")[column])

elif page == "Explainability":
    st.header("Model Explainability")
    df = load_report("explainability_report.csv")
    show_report(df)

elif page == "Hadoop & Pig Analytics":
    st.header("Hadoop and Apache Pig Analytics")

    pig_file = ROOT / "data" / "processed" / "resource_metrics" / "part-m-00000"

    if pig_file.exists():
        st.success("Pig output file found!")
        try:
            pig_df = pd.read_csv(
                pig_file,
                sep=None,
                engine="python",
                header=None
            )
            show_report(pig_df)
        except Exception:
            st.text(pig_file.read_text(errors="replace"))

        st.download_button(
            "Download Pig Output",
            pig_file.read_bytes(),
            "pig_resource_metrics.txt"
        )
    else:
        st.warning("Pig output file not found.")

    pig_script = ROOT / "pig" / "scripts" / "resource_analysis.pig"
    if pig_script.exists():
        st.subheader("Apache Pig Script")
        st.code(pig_script.read_text(), language="sql")

elif page == "Patient & Resource Input":
    st.header("Patient & Hospital Resource Input")
    st.write("Enter details below to check hospital bed availability.")

    with st.form("hospital_input_form"):
        st.subheader("Patient Details")
        age = st.number_input("Patient Age", min_value=0, max_value=120, value=30)
        gender = st.selectbox("Gender", ["Female", "Male", "Other"])
        department_name = st.selectbox(
            "Department",
            ["Cardiology", "Emergency", "Neurology", "Orthopedics"]
        )

        st.subheader("Hospital Resource Details")
        total_beds = st.number_input("Total Bed Capacity", min_value=1, value=100)
        occupied_beds = st.number_input(
            "Currently Occupied Beds", min_value=0, value=60
        )
        expected_admissions = st.number_input(
            "Expected New Admissions", min_value=0, value=20
        )
        available_staff = st.number_input(
            "Available Staff", min_value=0, value=25
        )

        submitted = st.form_submit_button("Calculate Resource Availability")

    if submitted:
        if occupied_beds > total_beds:
            st.error("Occupied beds cannot exceed total bed capacity.")
        else:
            free_beds = total_beds - occupied_beds
            beds_needed = max(0, expected_admissions - free_beds)
            projected_occupancy = (
                (occupied_beds + expected_admissions) / total_beds
            ) * 100

            st.subheader("Resource Analysis Results")
            c1, c2, c3 = st.columns(3)
            c1.metric("Currently Free Beds", free_beds)
            c2.metric("Additional Beds Needed", beds_needed)
            c3.metric("Projected Occupancy", f"{projected_occupancy:.1f}%")

            st.write("**Patient Age:**", age)
            st.write("**Gender:**", gender)
            st.write("**Department:**", department_name)
            st.write("**Available Staff:**", available_staff)

            if beds_needed > 0:
                st.warning(
                    f"{department_name} may need {beds_needed} additional beds."
                )
            else:
                st.success("Current bed capacity is sufficient for the expected admissions.")

            if projected_occupancy > 100:
                st.error("Projected admissions exceed total bed capacity.")

st.divider()
st.caption("Hospital Resource Management | Hadoop | Apache Pig | Python")
