"""
Clinical Assessment Report Generator for Cardiovascular Risk Stratification.
Formats comprehensive medical evaluation summaries in clean HTML and text formats.
"""

from datetime import datetime
from typing import Any, Dict, List
import pandas as pd


def generate_clinical_html_report(
    patient_data: Dict[str, Any],
    prediction_data: Dict[str, Any],
    explanation_data: Dict[str, Any],
    counterfactual_data: Dict[str, Any],
) -> str:
    """Generate a printable HTML clinical cardiology report."""
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M")
    prob_pct = prediction_data.get("risk_percentage", 0.0)
    category = prediction_data.get("risk_category", "Unknown")

    badge_color = {
        "Critical Risk": "#dc2626",
        "High Risk": "#ea580c",
        "Moderate Risk": "#d97706",
        "Low Risk": "#16a34a",
    }.get(category, "#2563eb")

    contributions_html = ""
    for item in explanation_data.get("top_contributions", [])[:5]:
        direction_icon = "🔺" if item.direction == "increases_risk" else "🔻"
        color = "#b91c1c" if item.direction == "increases_risk" else "#15803d"
        contributions_html += f"""
        <tr>
            <td style="padding: 8px; border-bottom: 1px solid #e2e8f0;">{item.display_name}</td>
            <td style="padding: 8px; border-bottom: 1px solid #e2e8f0; font-weight: 600;">{item.value}</td>
            <td style="padding: 8px; border-bottom: 1px solid #e2e8f0; color: {color};">{direction_icon} {item.contribution:+.3f}</td>
        </tr>
        """

    recs_html = ""
    for rec in counterfactual_data.get("recommendations", []):
        recs_html += f"""
        <li style="margin-bottom: 8px;">
            <strong>{rec.factor}</strong> (Current: {rec.current_value} &rarr; Target: {rec.target_value}):
            {rec.action_text} <span style="color: #16a34a; font-weight: 600;">[Estimated Risk Reduction: -{rec.risk_reduction_pct}%]</span>
        </li>
        """

    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <title>Cardiovascular Risk Assessment Report</title>
        <style>
            body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; color: #1e293b; background: #ffffff; padding: 24px; line-height: 1.5; }}
            .card {{ border: 1px solid #cbd5e1; border-radius: 8px; padding: 20px; margin-bottom: 20px; }}
            .header {{ display: flex; justify-content: space-between; border-bottom: 2px solid #0284c7; padding-bottom: 12px; margin-bottom: 20px; }}
            .badge {{ display: inline-block; padding: 6px 16px; border-radius: 9999px; color: #ffffff; font-weight: 700; font-size: 14px; background-color: {badge_color}; }}
            table {{ width: 100%; border-collapse: collapse; margin-top: 10px; font-size: 14px; }}
            th {{ text-align: left; background: #f8fafc; padding: 8px; border-bottom: 2px solid #cbd5e1; color: #475569; }}
        </style>
    </head>
    <body>
        <div class="header">
            <div>
                <h1 style="margin: 0; font-size: 22px; color: #0f172a;">Cardiovascular Risk Assessment Report</h1>
                <p style="margin: 4px 0 0 0; color: #64748b; font-size: 13px;">Clinical Decision Support System (CDSS) • Generated on {now_str}</p>
            </div>
            <div>
                <span class="badge">{category.upper()} ({prob_pct}%)</span>
            </div>
        </div>

        <div class="card" style="background: #f8fafc;">
            <h3 style="margin-top: 0; color: #0f172a;">🩺 Patient Baseline Vitals & Biomarkers</h3>
            <table style="margin-top: 5px;">
                <tr>
                    <td><strong>Age:</strong> {patient_data.get('age')} years</td>
                    <td><strong>Sex:</strong> {'Male' if patient_data.get('sex')==1 else 'Female'}</td>
                    <td><strong>Resting BP:</strong> {patient_data.get('trestbps')} mmHg</td>
                    <td><strong>Cholesterol:</strong> {patient_data.get('chol')} mg/dL</td>
                </tr>
                <tr>
                    <td><strong>Chest Pain:</strong> Type {patient_data.get('cp')}</td>
                    <td><strong>Max Heart Rate:</strong> {patient_data.get('thalach')} bpm</td>
                    <td><strong>ST Depression:</strong> {patient_data.get('oldpeak')} mm</td>
                    <td><strong>Exercise Angina:</strong> {'Yes' if patient_data.get('exang')==1 else 'No'}</td>
                </tr>
            </table>
        </div>

        <div class="card">
            <h3 style="margin-top: 0; color: #0f172a;">📋 Clinical Triage & Recommendation</h3>
            <p style="font-size: 15px; color: #1e293b; background: #f1f5f9; padding: 12px; border-left: 4px solid {badge_color}; border-radius: 4px;">
                {prediction_data.get('clinical_recommendation')}
            </p>
        </div>

        <div class="card">
            <h3 style="margin-top: 0; color: #0f172a;">🧠 Key Feature Contributions (Explainable AI Attribution)</h3>
            <table>
                <thead>
                    <tr>
                        <th>Clinical Marker</th>
                        <th>Recorded Value</th>
                        <th>Risk Attribution Impact</th>
                    </tr>
                </thead>
                <tbody>
                    {contributions_html}
                </tbody>
            </table>
        </div>

        <div class="card">
            <h3 style="margin-top: 0; color: #0f172a;">🎯 Counterfactual Risk-Reduction Pathways</h3>
            <p style="font-size: 13px; color: #64748b;">
                Simulated achievable risk reduction if patient achieves evidence-based therapeutic targets:
            </p>
            <ul style="padding-left: 20px; font-size: 14px;">
                {recs_html if recs_html else "<li>Patient biomarkers already at optimal baseline. Maintain preventive maintenance.</li>"}
            </ul>
            <p style="margin-top: 12px; font-weight: 600; color: #0f172a;">
                Potential Optimized Risk: <span style="color: #16a34a;">{counterfactual_data.get('counterfactual_risk_pct', prob_pct)}%</span>
                (Absolute Risk Reduction: <span style="color: #16a34a;">-{counterfactual_data.get('absolute_risk_reduction_pct', 0.0)}%</span>)
            </p>
        </div>

        <div style="font-size: 11px; color: #94a3b8; text-align: center; margin-top: 24px; border-top: 1px solid #e2e8f0; padding-top: 12px;">
            ⚠️ Academic & Clinical Decision Support Demonstration. For diagnostic confirmation, consult a board-certified cardiologist.
        </div>
    </body>
    </html>
    """
    return html
