# 🗂️ Model Card: Heart Risk Clinical Decision Support System (CDSS)

Following the standard proposed by *Mitchell et al. (2019), "Model Cards for Model Reporting"*.

---

## 📌 1. Model Details

- **Model Name:** Heart Risk Clinical Decision Support Classifier (`heart-risk-cds`)
- **Version:** `1.0.0`
- **Model Type:** Calibrated ElasticNet / Logistic Regression & Stacking Ensemble over tree ensembles (Random Forest, Extra Trees, Gradient Boosting) and Support Vector Classifiers.
- **Developer:** Mohammad Hasan ([tmohammadhasan212](https://github.com/tmohammadhasan212))
- **Release Date:** August 2026
- **License:** MIT License
- **Framework:** Scikit-Learn (v1.3+), Python 3.10-3.13

---

## 🎯 2. Intended Use

- **Primary Intended Uses:**
  - Assist clinical researchers and medical staff in early risk triage and cardiovascular risk stratification.
  - Provide Explainable AI (XAI) feature attribution breakdown for clinician audit.
  - Simulate counterfactual risk trajectories to guide lifestyle and pharmacological goal-setting.
- **Out-of-Scope Use Cases:**
  - Automated diagnostic substitution for human physician judgement.
  - Standalone triage in emergency acute ST-Elevation Myocardial Infarction (STEMI) where emergent angiogram is mandatory.

---

## 📊 3. Training & Validation Data

- **Cohort Source:** UCI Machine Learning Repository - Heart Disease Dataset (Cleveland Clinic Foundation).
- **Cohort Size:** 303 patient records with 14 clinical and electrophysiological biomarkers.
- **Demographics:** Adults aged 29–77 (mean age 54.4 years); 68.3% male, 31.7% female.
- **Prevalence:** 54.5% positive for significant coronary artery disease / elevated cardiac risk.

---

## 🧪 4. Quantitative Evaluation

### Repeated Stratified Nested Cross-Validation (5-Fold × 3-Repeats)

| Algorithm | ROC-AUC (Mean ± Std) | 95% Confidence Interval | Sensitivity (Recall) | Specificity | Accuracy | Brier Score |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression (Calibrated)** | **0.917 ± 0.038** | [0.847, 0.971] | **90.5%** | **79.9%** | **85.7%** | **0.113** |
| **Voting Ensemble** | 0.908 ± 0.041 | [0.832, 0.965] | 88.2% | 79.1% | 84.1% | 0.120 |
| **Support Vector Machine (RBF)** | 0.903 ± 0.044 | [0.819, 0.960] | 89.1% | 77.4% | 83.7% | 0.125 |
| **Stacking Classifier** | 0.899 ± 0.048 | [0.805, 0.962] | 86.7% | 78.3% | 82.9% | 0.129 |
| **Gradient Boosting** | 0.884 ± 0.052 | [0.781, 0.954] | 84.2% | 76.5% | 80.7% | 0.141 |
| **Random Forest** | 0.892 ± 0.046 | [0.802, 0.958] | 86.1% | 78.0% | 82.4% | 0.134 |
| **Tabular Neural Net (MLP)** | 0.889 ± 0.050 | [0.790, 0.956] | 85.5% | 76.8% | 81.5% | 0.139 |

---

## ⚖️ 5. Ethical & Clinical Considerations

- **Asymmetric Cost of Errors:** In cardiology, a False Negative (failing to detect high risk) carries a severe clinical penalty compared to a False Positive. The decision threshold can be calibrated to favor high sensitivity (>90%).
- **Fairness & Subgroups:** Male patients are represented in higher proportion (68.3%). Clinicians must verify performance on female cohorts where cardiac presentation (e.g. atypical chest pain) may differ.
- **Explainability:** Predictions are coupled with local Shapley feature attributions to prevent opaque "black-box" clinical decisions.
