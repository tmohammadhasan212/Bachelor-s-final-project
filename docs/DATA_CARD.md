# 🗂️ Data Card: UCI Heart Disease Dataset

Comprehensive metadata, clinical definitions, and provenance of the dataset.

---

## 📌 1. Dataset Overview

- **Dataset Name:** Cleveland Heart Disease Database
- **Originating Institution:** Cleveland Clinic Foundation (Dr. Robert Detrano, M.D., Ph.D.)
- **Primary Repository:** UCI Machine Learning Repository
- **Instances:** 303 clinical records
- **Features:** 13 input features + 1 binary target outcome

---

## 🩺 2. Clinical Feature Dictionary

| Feature Name | Type | Unit / Values | Clinical Description & Significance |
| :--- | :--- | :--- | :--- |
| `age` | Integer | Years (29–77) | Patient chronological age; major non-modifiable cardiovascular risk factor. |
| `sex` | Binary | 1 = Male, 0 = Female | Biological sex; male sex associated with earlier onset CAD. |
| `cp` | Categorical | 0 = Typical Angina<br>1 = Atypical Angina<br>2 = Non-Anginal<br>3 = Asymptomatic | Chest pain presentation; key diagnostic symptom for coronary ischemia. |
| `trestbps` | Integer | mmHg (94–200) | Resting systolic blood pressure upon admission. |
| `chol` | Integer | mg/dL (126–564) | Serum total cholesterol level; marker for atherogenic risk. |
| `fbs` | Binary | 1 = >120 mg/dL, 0 = ≤120 | Fasting blood sugar; indicator of metabolic dysfunction / diabetes mellitus. |
| `restecg` | Categorical | 0 = Normal<br>1 = ST-T wave abnormality<br>2 = LVH | Baseline 12-lead electrocardiographic findings. |
| `thalach` | Integer | bpm (71–202) | Maximum heart rate achieved during exercise stress testing. |
| `exang` | Binary | 1 = Yes, 0 = No | Occurrence of angina pectoris provoked by physical exertion. |
| `oldpeak` | Float | mm (0.0–6.2) | ST segment depression induced by exercise relative to resting baseline. |
| `slope` | Categorical | 0 = Upsloping<br>1 = Flat<br>2 = Downsloping | Morphology of peak exercise ST segment slope. |
| `ca` | Integer | 0 to 4 | Number of major coronary arteries with >50% luminal diameter stenosis visible on fluoroscopy. |
| `thal` | Categorical | 1 = Fixed defect<br>2 = Normal<br>3 = Reversible defect | Thallium-201 myocardial perfusion scintigraphy status. |
| `target` | Binary | 1 = High Risk / Disease<br>0 = Low Risk / Absence | Diagnosis of coronary heart disease (>50% diameter narrowing). |

---

## 🔬 3. Preprocessing & Leak-Free Architecture

1. **No Data Leakage:** Scaling and one-hot encoding are isolated within Scikit-Learn `ColumnTransformer` pipelines fit strictly inside cross-validation training folds.
2. **Derived Biomarkers (Clinical Feature Engineering):**
   - **Pulse Pressure:** `trestbps - 80` (marker of arterial stiffness).
   - **Heart Rate Reserve Ratio:** `thalach / (220 - age)` (fraction of age-predicted peak aerobic capacity).
   - **Cholesterol-to-Age Ratio:** `chol / age` (accumulation rate).
   - **Exercise Ischemia Index:** `oldpeak * (exang + 1.0)`.
