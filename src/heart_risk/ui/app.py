"""
Interactive Clinical Decision Support Dashboard (Streamlit).
Early Cardiovascular Risk Stratification, Explainable AI (XAI), and Counterfactual Lifestyle Simulator.
"""

import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from src.heart_risk.config import load_config
from src.heart_risk.data.loader import load_dataset
from src.heart_risk.data.schemas import PatientInput
from src.heart_risk.evaluation.clinical_utility import (
    compute_decision_curve_analysis,
)
from src.heart_risk.explainability.counterfactual import CounterfactualSimulator
from src.heart_risk.explainability.shap_engine import (
    FEATURE_DISPLAY_NAMES,
    ClinicalExplainer,
)
from src.heart_risk.models.train import train_and_export_models
from src.heart_risk.ui.components import (
    plot_dca_curve,
    plot_waterfall_chart,
    render_risk_gauge,
)
from src.heart_risk.ui.report_generator import generate_clinical_html_report

st.set_page_config(
    page_title="Cardiovascular Risk CDS System",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for polished medical UI
st.markdown(
    """
    <style>
    .main-title {
        font-size: 28px;
        font-weight: 800;
        color: #0f172a;
        margin-bottom: 4px;
    }
    .sub-title {
        font-size: 15px;
        color: #64748b;
        margin-bottom: 20px;
    }
    .metric-badge {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 12px;
    }
    .info-card {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 16px;
        margin-bottom: 16px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_system():
    """Load configuration, background dataset, model pipeline, and explainers."""
    cfg = load_config()
    df = load_dataset()
    import joblib

    model_path = Path(cfg.paths.models_dir) / "best_model.joblib"
    if model_path.exists():
        model = joblib.load(model_path)
    else:
        trainer = train_and_export_models()
        model = trainer.best_pipeline_

    bg_df = df.drop("target", axis=1)
    explainer = ClinicalExplainer(model, bg_df)
    simulator = CounterfactualSimulator(model)

    # Load benchmark results
    bench_path = Path(cfg.paths.processed_data_dir) / "benchmark_results.json"
    bench_data = {}
    if bench_path.exists():
        with open(bench_path, "r", encoding="utf-8") as f:
            bench_data = json.load(f)

    return cfg, df, model, explainer, simulator, bench_data


cfg, df, model, explainer, simulator, bench_data = load_system()

# Sidebar Navigation & Preset Patient Selector
st.sidebar.markdown("## 🏥 Clinical Control Center")
st.sidebar.markdown(
    f"**System Version:** `v{cfg.project.version}`\n\n"
    f"**Active Model:** `{bench_data.get('best_model', 'Logistic Regression (Calibrated)')}`"
)

preset_option = st.sidebar.selectbox(
    "Load Clinical Case Profile:",
    [
        "Custom Input",
        "Case 1: High Risk Elderly (Chest Pain & ST Depression)",
        "Case 2: Young Active Adult (Low Risk Baseline)",
        "Case 3: Middle-Aged Hypertensive (Moderate Risk)",
    ],
)

# Preset feature defaults
if preset_option == "Case 1: High Risk Elderly (Chest Pain & ST Depression)":
    default_vals = {
        "age": 67,
        "sex": 1,
        "cp": 0,
        "trestbps": 160,
        "chol": 286,
        "fbs": 1,
        "restecg": 0,
        "thalach": 108,
        "exang": 1,
        "oldpeak": 2.6,
        "slope": 1,
        "ca": 3,
        "thal": 2,
    }
elif preset_option == "Case 2: Young Active Adult (Low Risk Baseline)":
    default_vals = {
        "age": 35,
        "sex": 0,
        "cp": 2,
        "trestbps": 115,
        "chol": 182,
        "fbs": 0,
        "restecg": 1,
        "thalach": 174,
        "exang": 0,
        "oldpeak": 0.0,
        "slope": 2,
        "ca": 0,
        "thal": 2,
    }
elif preset_option == "Case 3: Middle-Aged Hypertensive (Moderate Risk)":
    default_vals = {
        "age": 54,
        "sex": 1,
        "cp": 1,
        "trestbps": 142,
        "chol": 245,
        "fbs": 0,
        "restecg": 1,
        "thalach": 145,
        "exang": 0,
        "oldpeak": 1.2,
        "slope": 1,
        "ca": 1,
        "thal": 2,
    }
else:
    default_vals = {
        "age": 58,
        "sex": 1,
        "cp": 1,
        "trestbps": 135,
        "chol": 230,
        "fbs": 0,
        "restecg": 1,
        "thalach": 155,
        "exang": 0,
        "oldpeak": 0.8,
        "slope": 2,
        "ca": 0,
        "thal": 2,
    }

# Main Application Title
st.markdown(
    '<div class="main-title">❤️ Cardiovascular Risk Stratification & Clinical Decision Support System</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="sub-title">Multi-Model Calibrated Machine Learning, Explainable AI (SHAP), and Actionable Counterfactual Simulation</div>',
    unsafe_allow_html=True,
)

# Tabs
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
    [
        "🩺 Patient Triage",
        "🧠 Explainable AI (XAI)",
        "🔄 What-If Simulator",
        "📊 Model Benchmarking",
        "📁 Batch Population Screening",
        "📑 Clinical Report Export",
    ]
)

# ----------------- TAB 1: PATIENT TRIAGE -----------------
with tab1:
    st.markdown("### 📋 Enter Patient Biomarkers & Clinical Parameters")

    col_input1, col_input2, col_input3 = st.columns(3)

    with col_input1:
        st.markdown("**Demographics & Baseline Vitals**")
        age = st.slider(
            "Age (years)", 18, 100, default_vals["age"], help="Patient age"
        )
        sex = st.radio(
            "Biological Sex",
            [1, 0],
            index=0 if default_vals["sex"] == 1 else 1,
            format_func=lambda x: "Male" if x == 1 else "Female",
            horizontal=True,
        )
        trestbps = st.slider(
            "Resting Blood Pressure (mmHg)",
            80,
            220,
            default_vals["trestbps"],
            help="Resting BP on admission",
        )
        chol = st.slider(
            "Serum Cholesterol (mg/dL)",
            100,
            500,
            default_vals["chol"],
            help="Total serum cholesterol",
        )

    with col_input2:
        st.markdown("**Symptoms & Exercise Stress Test**")
        cp = st.selectbox(
            "Chest Pain Type",
            [0, 1, 2, 3],
            index=default_vals["cp"],
            format_func=lambda x: {
                0: "Typical Angina (Substernal chest pressure)",
                1: "Atypical Angina (Dyspnea / Epigastric)",
                2: "Non-Anginal Pain (Musculoskeletal / Pleuritic)",
                3: "Asymptomatic (Silent presentation)",
            }[x],
        )
        thalach = st.slider(
            "Max Heart Rate Achieved (bpm)",
            60,
            220,
            default_vals["thalach"],
            help="Peak HR during treadmill stress test",
        )
        exang = st.radio(
            "Exercise-Induced Angina",
            [1, 0],
            index=0 if default_vals["exang"] == 1 else 1,
            format_func=lambda x: "Yes (Angina during test)"
            if x == 1
            else "No",
            horizontal=True,
        )
        fbs = st.radio(
            "Fasting Blood Sugar > 120 mg/dL",
            [1, 0],
            index=0 if default_vals["fbs"] == 1 else 1,
            format_func=lambda x: "Yes (> 120 mg/dL)" if x == 1 else "No",
            horizontal=True,
        )

    with col_input3:
        st.markdown("**Electrocardiogram & Fluoroscopy**")
        oldpeak = st.slider(
            "ST Depression Oldpeak (mm)",
            0.0,
            6.2,
            float(default_vals["oldpeak"]),
            step=0.1,
            help="ST depression induced by exercise relative to rest",
        )
        slope = st.selectbox(
            "ST Segment Peak Slope",
            [0, 1, 2],
            index=default_vals["slope"],
            format_func=lambda x: {
                0: "Upsloping (Normal)",
                1: "Flat (Ischemic pattern)",
                2: "Downsloping (Severe ischemia)",
            }[x],
        )
        restecg = st.selectbox(
            "Resting ECG Results",
            [0, 1, 2],
            index=default_vals["restecg"],
            format_func=lambda x: {
                0: "Normal",
                1: "ST-T Wave Abnormality",
                2: "Left Ventricular Hypertrophy (LVH)",
            }[x],
        )
        ca = st.selectbox(
            "Major Vessels Colored (Fluoroscopy)",
            [0, 1, 2, 3, 4],
            index=default_vals["ca"],
            help="Number of major vessels (0-4) visible",
        )
        thal = st.selectbox(
            "Thalassemia Status",
            [1, 2, 3],
            index=[1, 2, 3].index(default_vals.get("thal", 2))
            if default_vals.get("thal", 2) in [1, 2, 3]
            else 1,
            format_func=lambda x: {
                1: "Fixed Defect (Prior infarction)",
                2: "Normal Blood Flow",
                3: "Reversible Defect (Inducible ischemia)",
            }.get(x, "Unknown"),
        )

    # Build active patient dataframe
    patient_dict = {
        "age": age,
        "sex": sex,
        "cp": cp,
        "trestbps": trestbps,
        "chol": chol,
        "fbs": fbs,
        "restecg": restecg,
        "thalach": thalach,
        "exang": exang,
        "oldpeak": oldpeak,
        "slope": slope,
        "ca": ca,
        "thal": thal,
    }
    patient_df = pd.DataFrame([patient_dict])

    # Predict Risk
    pred_prob = float(model.predict_proba(patient_df)[0, 1])
    risk_pct = pred_prob * 100.0

    if risk_pct >= 90:
        category = "Critical Risk"
    elif risk_pct >= 75:
        category = "High Risk"
    elif risk_pct >= 50:
        category = "Moderate Risk"
    else:
        category = "Low Risk"

    st.markdown("---")

    # Display Triage Results
    col_res1, col_res2 = st.columns([1.2, 2])

    with col_res1:
        render_risk_gauge(risk_pct, category)

    with col_res2:
        st.markdown("#### 🩺 Clinical Decision Recommendation")
        if category in ("Critical Risk", "High Risk"):
            st.error(
                f"**{category.upper()} DETECTED ({risk_pct:.1f}%)**\n\n"
                "• **Immediate Action:** Urgent cardiology consultation & 12-lead ECG.\n"
                "• **Diagnostics:** Cardiac biomarker assay (hs-cTnI / hs-cTnT), bedside echocardiogram.\n"
                "• **Intervention:** Assess eligibility for urgent coronary angiography / revascularization."
            )
        elif category == "Moderate Risk":
            st.warning(
                f"**{category.upper()} ({risk_pct:.1f}%)**\n\n"
                "• **Recommendation:** Exercise stress echocardiography or myocardial perfusion imaging.\n"
                "• **Guideline Therapy:** Initiate statin therapy and antihypertensive optimization."
            )
        else:
            st.success(
                f"**{category.upper()} ({risk_pct:.1f}%)**\n\n"
                "• **Recommendation:** Routine preventive cardiology follow-up.\n"
                "• **Counseling:** Encourage physical activity and dietary maintenance."
            )

# ----------------- TAB 2: EXPLAINABLE AI (XAI) -----------------
with tab2:
    st.markdown("### 🧠 Patient-Specific Feature Attribution (SHAP-Style XAI)")
    st.markdown(
        "Understand **why** the model produced this prediction. Red bars increase risk, Green bars reduce risk."
    )

    explanation = explainer.explain_instance(patient_df, top_k=8)

    col_chart, col_table = st.columns([1.8, 1.2])

    with col_chart:
        fig_waterfall = plot_waterfall_chart(
            explanation["top_contributions"],
            explanation["base_risk"],
            explanation["predicted_risk"],
        )
        st.pyplot(fig_waterfall)

    with col_table:
        st.markdown("#### Top Driving Biomarkers")
        for item in explanation["top_contributions"]:
            icon = "🔴" if item.direction == "increases_risk" else "🟢"
            st.markdown(
                f"{icon} **{item.display_name}**: `{item.value}`  \n"
                f"<span style='color: {'#dc2626' if item.direction=='increases_risk' else '#16a34a'}; font-size:12px;'>"
                f"Impact: {item.contribution*100:+.1f}% points</span>",
                unsafe_allow_html=True,
            )

    st.markdown("---")
    st.markdown("#### 🌐 Global Cohort Feature Importance")
    global_imp = explainer.get_global_feature_importance()
    top_global = dict(list(global_imp.items())[:8])
    fig_glob, ax_glob = plt.subplots(figsize=(8, 3))
    ax_glob.barh(
        list(top_global.keys()),
        list(top_global.values()),
        color="#0284c7",
        height=0.55,
    )
    ax_glob.set_xlabel("Relative Importance Weight")
    ax_glob.set_title("Top Population-Level Predictors", fontweight="bold")
    ax_glob.invert_yaxis()
    ax_glob.grid(axis="x", linestyle=":", alpha=0.5)
    st.pyplot(fig_glob)

# ----------------- TAB 3: WHAT-IF SIMULATOR -----------------
with tab3:
    st.markdown("### 🔄 Counterfactual Lifestyle & Risk Reduction Simulator")
    st.markdown(
        "Simulate evidence-based clinical interventions on **modifiable risk factors** to demonstrate risk reduction pathways."
    )

    sim_res = simulator.simulate_interventions(patient_df)

    col_sim1, col_sim2, col_sim3 = st.columns(3)
    col_sim1.metric(
        "Baseline Estimated Risk",
        f"{sim_res['baseline_risk_pct']:.1f}%",
    )
    col_sim2.metric(
        "Target Optimized Risk",
        f"{sim_res['counterfactual_risk_pct']:.1f}%",
        delta=f"-{sim_res['absolute_risk_reduction_pct']:.1f}%",
        delta_color="normal",
    )
    col_sim3.metric(
        "Relative Risk Reduction (RRR)",
        f"{sim_res['relative_risk_reduction_pct']:.1f}%",
    )

    st.markdown("#### 📋 Actionable Clinical Recommendations")
    if sim_res["recommendations"]:
        for rec in sim_res["recommendations"]:
            st.info(
                f"🎯 **{rec.factor}** (Current: `{rec.current_value}` → Target: `{rec.target_value}`):  \n"
                f"{rec.action_text}  \n"
                f"**Potential Risk Reduction: -{rec.risk_reduction_pct}%**"
            )
    else:
        st.success(
            "Patient is already operating at optimal modifiable clinical baselines!"
        )

# ----------------- TAB 4: MODEL BENCHMARKING -----------------
with tab4:
    st.markdown("### 📊 Repeated Stratified Cross-Validation Benchmark Suite")
    st.markdown(
        "All models evaluated under **strict leak-free 5-Fold x 3-Repeat Nested Cross-Validation**."
    )

    if bench_data and "models" in bench_data:
        summary_rows = []
        for m_name, m_res in bench_data["models"].items():
            summary_rows.append(
                {
                    "Algorithm": m_name.replace("_", " ").title(),
                    "ROC-AUC (Mean ± Std)": f"{m_res['roc_auc_mean']:.3f} ± {m_res['roc_auc_std']:.3f}",
                    "Sensitivity (Recall)": f"{m_res['sensitivity_mean']*100:.1f}%",
                    "Specificity": f"{m_res['specificity_mean']*100:.1f}%",
                    "Accuracy": f"{m_res['accuracy_mean']*100:.1f}%",
                    "F1-Score": f"{m_res['f1_score_mean']:.3f}",
                    "Brier Score": f"{m_res['brier_score_mean']:.3f}",
                }
            )
        bench_df = pd.DataFrame(summary_rows)
        st.dataframe(bench_df, use_container_width=True)

        st.markdown("#### 📈 Decision Curve Analysis (DCA - Net Clinical Benefit)")
        X_all = df.drop("target", axis=1)
        y_all = df["target"]
        probs_all = model.predict_proba(X_all)[:, 1]
        dca_dict = compute_decision_curve_analysis(y_all, probs_all)
        fig_dca = plot_dca_curve(dca_dict)
        st.pyplot(fig_dca)

# ----------------- TAB 5: BATCH SCREENING -----------------
with tab5:
    st.markdown("### 📁 Population Batch Risk Screening")
    st.markdown(
        "Upload a clinical cohort CSV to run automated batch triage and stratify high-risk patients."
    )

    uploaded_file = st.file_uploader(
        "Upload CSV file (Expected columns: age, sex, cp, trestbps, chol, etc.)",
        type=["csv"],
    )

    if uploaded_file is not None:
        batch_df = pd.read_csv(uploaded_file)
        st.write(f"Loaded {len(batch_df)} patient records.")
        if st.button("Run Batch Triage Analysis", type="primary"):
            feature_cols = [c for c in batch_df.columns if c != "target"]
            probs = model.predict_proba(batch_df[feature_cols])[:, 1]
            batch_df["Estimated_Risk_Pct"] = np.round(probs * 100.0, 1)
            batch_df["Risk_Category"] = [
                "Critical Risk"
                if p >= 0.90
                else "High Risk"
                if p >= 0.75
                else "Moderate Risk"
                if p >= 0.50
                else "Low Risk"
                for p in probs
            ]

            st.dataframe(batch_df, use_container_width=True)

            # Download analyzed batch CSV
            csv_data = batch_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "📥 Download Triage Summary CSV",
                data=csv_data,
                file_name="cardiovascular_triage_results.csv",
                mime="text/csv",
            )
    else:
        st.info(
            "No file uploaded. Click below to load sample cohort from test set."
        )
        if st.button("Load Sample Test Cohort"):
            sample_cohort = df.sample(20, random_state=42).copy()
            feature_cols = [c for c in sample_cohort.columns if c != "target"]
            probs = model.predict_proba(sample_cohort[feature_cols])[:, 1]
            sample_cohort["Estimated_Risk_Pct"] = np.round(probs * 100.0, 1)
            sample_cohort["Risk_Category"] = [
                "Critical Risk"
                if p >= 0.90
                else "High Risk"
                if p >= 0.75
                else "Moderate Risk"
                if p >= 0.50
                else "Low Risk"
                for p in probs
            ]
            st.dataframe(sample_cohort, use_container_width=True)

# ----------------- TAB 6: CLINICAL REPORT EXPORT -----------------
with tab6:
    st.markdown("### 📑 Clinical Assessment PDF & HTML Report Generator")
    st.markdown(
        "Generate a structured medical summary document for the current active patient profile."
    )

    pred_data = {
        "risk_percentage": round(risk_pct, 1),
        "risk_category": category,
        "clinical_recommendation": generate_clinical_recommendation(
            pred_prob, category
        ),
    }

    report_html = generate_clinical_html_report(
        patient_data=patient_dict,
        prediction_data=pred_data,
        explanation_data=explanation,
        counterfactual_data=sim_res,
    )

    st.download_button(
        label="📥 Download Printable HTML Clinical Report",
        data=report_html,
        file_name=f"patient_cardiac_assessment_{patient_dict['age']}yo_{'male' if patient_dict['sex']==1 else 'female'}.html",
        mime="text/html",
    )

    st.markdown("#### 👁️ Report Preview")
    st.components.v1.html(report_html, height=650, scrolling=True)
