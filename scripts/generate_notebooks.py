"""
Script to generate clean, publication-grade Jupyter Notebooks for EDA and Modeling/XAI.
"""

import json
from pathlib import Path


def create_nb(cells):
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.11.0"},
        },
        "nbformat": 4,
        "nbformat_minor": 4,
    }


def md(text):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [l + "\n" for l in text.strip().split("\n")],
    }


def code(text):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [l + "\n" for l in text.strip().split("\n")],
    }


def generate_eda_notebook():
    cells = [
        md(
            """# 📊 Cardiovascular Risk: Exploratory Data Analysis & Clinical Insights

**Bachelor Capstone Project** • *Mohammad Hasan*

This notebook conducts a rigorous, leak-free exploratory data analysis of the Cleveland Heart Disease cohort ($N=303$).
"""
        ),
        code(
            """import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
%matplotlib inline"""
        ),
        md("## 1. Data Ingestion & Schema Verification"),
        code(
            """from src.heart_risk.data.loader import load_dataset

df = load_dataset()
print(f"Dataset Dimensions: {df.shape[0]} rows, {df.shape[1]} features")
df.head()"""
        ),
        md("## 2. Summary Statistics & Data Types"),
        code(
            """df.info()
df.describe().T"""
        ),
        md("## 3. Class Balance & Demographics"),
        code(
            """fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

# Target Balance
sns.countplot(x='target', data=df, ax=axes[0], palette=['#16a34a', '#dc2626'])
axes[0].set_xticklabels(['Low Risk (0)', 'High Risk (1)'])
axes[0].set_title('Target Class Distribution', fontweight='bold')

# Age Distribution by Target
sns.histplot(data=df, x='age', hue='target', kde=True, ax=axes[1], palette=['#16a34a', '#dc2626'])
axes[1].set_title('Age Distribution by Risk Category', fontweight='bold')

plt.tight_layout()
plt.show()"""
        ),
        md("## 4. Key Clinical Biomarker Distributions"),
        code(
            """num_cols = ['trestbps', 'chol', 'thalach', 'oldpeak']
fig, axes = plt.subplots(2, 2, figsize=(14, 9))
axes = axes.flatten()

for idx, col in enumerate(num_cols):
    sns.boxplot(x='target', y=col, data=df, ax=axes[idx], palette=['#16a34a', '#dc2626'])
    axes[idx].set_xticklabels(['Low Risk', 'High Risk'])
    axes[idx].set_title(f'Distribution of {col} by Cardiac Risk', fontweight='bold')

plt.tight_layout()
plt.show()"""
        ),
        md("## 5. Correlation Heatmap of Physiological Features"),
        code(
            """plt.figure(figsize=(11, 8))
corr = df.corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='coolwarm', cbar_kws={'shrink': .8})
plt.title('Correlation Matrix of Clinical Markers', fontsize=14, fontweight='bold', pad=12)
plt.show()"""
        ),
        md("## 6. Categorical Symptom Analysis"),
        code(
            """fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

# Chest Pain Type
sns.countplot(x='cp', hue='target', data=df, ax=axes[0], palette=['#16a34a', '#dc2626'])
axes[0].set_xticklabels(['Typical', 'Atypical', 'Non-anginal', 'Asymptomatic'])
axes[0].set_title('Chest Pain vs. Cardiac Risk', fontweight='bold')

# Exercise Induced Angina
sns.countplot(x='exang', hue='target', data=df, ax=axes[1], palette=['#16a34a', '#dc2626'])
axes[1].set_xticklabels(['No Angina', 'Exercise Angina'])
axes[1].set_title('Exercise-Induced Angina vs. Cardiac Risk', fontweight='bold')

plt.tight_layout()
plt.show()"""
        ),
        md(
            """## 📌 Summary of EDA Insights
- Higher maximum heart rate (`thalach`) during exercise stress tests and lower exercise-induced ST depression (`oldpeak`) distinguish lower-risk patients.
- Symptom presentation (chest pain type `cp` and presence of `exang`) provides strong predictive signal.
- Extreme values in blood pressure and cholesterol reflect genuine physiological stress and are preserved with robust scaling."""
        ),
    ]

    Path("notebooks").mkdir(parents=True, exist_ok=True)
    with open("notebooks/01_exploratory_data_analysis.ipynb", "w") as f:
        json.dump(create_nb(cells), f, indent=2)


def generate_modeling_notebook():
    cells = [
        md(
            """# 🤖 Clinical Machine Learning Benchmarking, Evaluation & Explainable AI (XAI)

**Bachelor Capstone Project** • *Mohammad Hasan*

This notebook demonstrates the end-to-end modeling workflow using the `heart_risk` package:
1. Leak-Free Preprocessing Pipelines
2. Repeated Stratified Nested Cross-Validation (5-Fold × 3-Repeats)
3. Model Benchmarking & Statistical Evaluation
4. Probability Calibration & Brier Score
5. Decision Curve Analysis (DCA - Net Clinical Benefit)
6. Explainable AI (SHAP Waterfall Attributions)
7. Counterfactual "What-If" Lifestyle Simulations
"""
        ),
        code(
            """import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from src.heart_risk.config import load_config
from src.heart_risk.data.loader import load_dataset, get_data_splits
from src.heart_risk.features.pipeline import create_preprocessing_pipeline
from src.heart_risk.models.train import ModelTrainer
from src.heart_risk.evaluation.metrics import evaluate_classification_performance
from src.heart_risk.evaluation.clinical_utility import compute_decision_curve_analysis
from src.heart_risk.explainability.shap_engine import ClinicalExplainer
from src.heart_risk.explainability.counterfactual import CounterfactualSimulator
from src.heart_risk.ui.components import plot_waterfall_chart, plot_dca_curve

%matplotlib inline"""
        ),
        md("## 1. Data Ingestion & Stratified Splits"),
        code(
            """df = load_dataset()
X_train, X_test, y_train, y_test = get_data_splits(df, test_size=0.20, random_seed=42)
print(f"Train samples: {len(X_train)}, Test samples: {len(X_test)}")"""
        ),
        md(
            "## 2. Multi-Model Repeated Stratified Nested Cross-Validation Benchmark"
        ),
        code(
            """cfg = load_config()
trainer = ModelTrainer(cfg)

# Run full cross-validation benchmarking across all algorithms
benchmark_results = trainer.benchmark_all_models(df, save_results=True)

# Format summary table
summary = []
for name, res in benchmark_results.items():
    summary.append({
        'Algorithm': name.replace('_', ' ').title(),
        'ROC-AUC': f"{res['roc_auc_mean']:.3f} ± {res['roc_auc_std']:.3f}",
        'Sensitivity (Recall)': f"{res['sensitivity_mean']*100:.1f}%",
        'Specificity': f"{res['specificity_mean']*100:.1f}%",
        'Accuracy': f"{res['accuracy_mean']*100:.1f}%",
        'F1-Score': f"{res['f1_score_mean']:.3f}",
        'Brier Score': f"{res['brier_score_mean']:.3f}"
    })

pd.DataFrame(summary).sort_values(by='ROC-AUC', ascending=False)"""
        ),
        md("## 3. Best Model Fitting & Holdout Test Evaluation"),
        code(
            """best_pipeline, test_metrics = trainer.fit_and_calibrate_best_model(df)
print(f"🏆 Selected Best Model: {trainer.best_model_name_}")
print("\\n📊 Holdout Test Performance Metrics:")
for k, v in test_metrics.items():
    if isinstance(v, float):
        print(f"  • {k}: {v:.4f}")
    elif k != 'confusion_matrix':
        print(f"  • {k}: {v}")"""
        ),
        md("## 4. Decision Curve Analysis (DCA - Clinical Net Benefit)"),
        code(
            """y_prob_test = best_pipeline.predict_proba(X_test)[:, 1]
dca = compute_decision_curve_analysis(y_test, y_prob_test)

fig_dca = plot_dca_curve(dca)
plt.show()"""
        ),
        md("## 5. Explainable AI (XAI) - Patient Feature Attributions"),
        code(
            """bg_df = X_train.copy()
explainer = ClinicalExplainer(best_pipeline, bg_df)

# Take a sample patient from test set
sample_patient = X_test.iloc[[0]]
explanation = explainer.explain_instance(sample_patient, top_k=7)

fig_waterfall = plot_waterfall_chart(
    explanation['top_contributions'],
    explanation['base_risk'],
    explanation['predicted_risk']
)
plt.show()"""
        ),
        md('## 6. Counterfactual "What-If" Lifestyle Simulation'),
        code(
            """simulator = CounterfactualSimulator(best_pipeline)
sim_res = simulator.simulate_interventions(sample_patient)

print(f"Baseline Risk: {sim_res['baseline_risk_pct']}%")
print(f"Optimized Target Risk: {sim_res['counterfactual_risk_pct']}%")
print(f"Absolute Risk Reduction: -{sim_res['absolute_risk_reduction_pct']}%")
print("\\n🎯 Clinical Action Recommendations:")
for rec in sim_res['recommendations']:
    print(f"  • {rec.factor}: {rec.action_text} [Reduction: -{rec.risk_reduction_pct}%]")"""
        ),
    ]

    with open("notebooks/02_clinical_modeling_and_xai.ipynb", "w") as f:
        json.dump(create_nb(cells), f, indent=2)


if __name__ == "__main__":
    generate_eda_notebook()
    generate_modeling_notebook()
    print("Clean publication-grade notebooks created successfully.")
