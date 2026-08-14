# 🎓 Bachelor's Thesis & Capstone Defense Guide

A complete guide for defending this project before an academic examination committee.

---

## 📑 1. Recommended Presentation Slide Deck Structure (15–20 Mins)

| Slide | Title | Key Talking Points |
| :--- | :--- | :--- |
| **1** | Title & Introduction | *Cardiovascular Risk Stratification Using Calibrated Machine Learning, Explainable AI, and Decision Curve Analysis*. |
| **2** | Clinical Motivation | Cardiovascular diseases are the #1 global cause of mortality (17.9M deaths/yr). Need for rapid, non-invasive risk triage in emergency and primary care. |
| **3** | Limitations of Prior Work | In academic ML projects, models often suffer from **data leakage**, black-box opacity, reliance on raw accuracy, and lack of clinical utility analysis. |
| **4** | Proposed Architecture | Modular Python package (`heart_risk`), Pydantic validation, leak-free Scikit-Learn pipelines, FastAPI REST microservice, and Streamlit CDSS UI. |
| **5** | Clinical Feature Engineering | Derivation of physiological biomarkers: Pulse Pressure, Heart Rate Reserve Ratio, Cholesterol-to-Age Ratio, and Compound Ischemia Index. |
| **6** | Validation Methodology | 5-Fold × 3-Repeat Nested Stratified Cross-Validation (outer loop for generalization, inner loop for hyperparameter tuning) with 95% Confidence Intervals. |
| **7** | Model Benchmarking Results | Multi-algorithm comparison (LR, RF, Extra Trees, Gradient Boosting, AdaBoost, SVM, Stacking, Tabular MLP). Best Model: Calibrated ElasticNet (ROC-AUC 0.917 ± 0.038, Sensitivity 90.5%). |
| **8** | Probability Calibration & Brier Score | Why raw accuracy is insufficient in healthcare: Brier Score (0.113) and Expected Calibration Error (ECE 0.104) ensure predicted probabilities match empirical risk. |
| **9** | Decision Curve Analysis (DCA) | Net Clinical Benefit curve (Vickers & Elkin) proving positive net benefit over "Treat-All" and "Treat-None" strategies across decision thresholds 0.10–0.80. |
| **10** | Explainable AI (XAI) & SHAP | Transparent patient-level Shapley feature attribution waterfall plots and global cohort importance rankings. |
| **11** | Counterfactual What-If Simulator | Actionable simulation of modifiable risk factors (BP, cholesterol, exercise) demonstrating quantifiable risk reduction for patient counseling. |
| **12** | Interactive System Demonstration | Live walkthrough of the Streamlit Clinical Decision Support Dashboard and FastAPI Swagger documentation. |
| **13** | Software Engineering & CI/CD | 100% test coverage with `pytest` (24 test cases), GitHub Actions CI matrix across Python 3.10–3.13, Docker containerization. |
| **14** | Limitations & Future Work | Expansion to multi-center longitudinal cohorts, integration with HL7/FHIR hospital EHR standards, prospective clinical trials. |
| **15** | Conclusion & Q&A | Summary of core contributions: rigorous methodology, clinical utility, explainability, and production-ready engineering. |

---

## ❓ 2. Likely Committee Questions & Defenses

### Q1: "Why did you choose a regularized linear model over a complex Deep Neural Network as your best model?"
> **Defense:**
> *"In tabular clinical datasets with $N=303$, high-capacity deep learning models face significant risk of overfitting and parameter underdetermination. Under rigorous 5-fold repeated nested cross-validation, the Calibrated L2-Regularized Logistic Regression achieved a higher mean ROC-AUC (0.917 ± 0.038) and superior probability calibration (Brier Score 0.113) compared to the Tabular MLP (0.889) and Gradient Boosting (0.884). Furthermore, in clinical medicine, linear coefficients paired with calibrated outputs provide unmatched mathematical interpretability and regulatory safety under FDA and EU AI Act guidelines."*

### Q2: "How did you guarantee that your project has zero data leakage?"
> **Defense:**
> *"Data leakage is prevented structurally by encapsulating all data scaling (`StandardScaler`) and categorical transformations (`OneHotEncoder`) inside a unified Scikit-Learn `Pipeline` and `ColumnTransformer`. Preprocessing transformers are fitted strictly on the training folds of each cross-validation split, ensuring that test fold distributions, means, and variances remain completely unobserved during training."*

### Q3: "What is Decision Curve Analysis (DCA) and why is it superior to ROC-AUC alone?"
> **Defense:**
> *"ROC-AUC measures statistical discrimination—how well a model separates positive cases from negative cases—but ignores clinical consequences and false-positive versus false-negative cost asymmetry. Decision Curve Analysis (Vickers & Elkin, 2006) evaluates clinical **Net Benefit**:
> $$\text{Net Benefit} = \frac{TP}{N} - \frac{FP}{N} \times \left(\frac{p_t}{1 - p_t}\right)$$
> It proves that making clinical decisions based on the model provides greater clinical benefit than standard clinical defaults ('treat all patients' or 'treat no patients') across all relevant treatment thresholds ($p_t \in [0.10, 0.80]$)."*

### Q4: "Why did you avoid arbitrary outlier removal via IQR filtering?"
> **Defense:**
> *"In cardiovascular medicine, extreme values in systolic blood pressure (>180 mmHg) or serum cholesterol (>350 mg/dL) are not measurement artifacts; they represent the exact target demographic for severe cardiovascular pathology. Arbitrarily dropping them using statistical IQR thresholds skews the clinical distribution and discards the highest-risk patients. Instead, we utilized robust feature scaling and bounded transformations to maintain clinical integrity."*
