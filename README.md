🎓 The Dropout Signal

Fair Early-Warning Pipeline for Student Dropout Prediction

"Databricks" (https://img.shields.io/badge/Databricks-FF3621?style=for-the-badge&logo=databricks&logoColor=white)
"PySpark" (https://img.shields.io/badge/PySpark-E25A1C?style=for-the-badge&logo=apache-spark&logoColor=white)
"MLflow" (https://img.shields.io/badge/MLflow-0194E2?style=for-the-badge&logo=mlflow&logoColor=white)
"XGBoost" (https://img.shields.io/badge/XGBoost-1793D1?style=for-the-badge)

---

🧠 Simple Explanation

This project predicts whether a student is likely to drop out using machine learning.
It helps institutions identify at-risk students early and take timely action through clear, explainable insights.

---

📌 Overview

An end-to-end, fair, and explainable machine learning pipeline built on Databricks to predict student dropout risk.
The system transforms raw student data into actionable risk assessments for academic advisors.

«Key Highlight: Generates human-readable explanations ("reason_text") and uses calibrated probabilities to ensure reliable intervention decisions.»

---

🌟 Key Features

- 📊 Predicts dropout probability
- 🧠 Machine Learning-based analysis (XGBoost + Logistic Regression)
- ⚖️ Fairness-aware model with bias detection
- 🔍 SHAP-based explainability
- 🗣️ Natural language insights ("reason_text")
- 📈 Risk-based intervention tiers (High / Medium / Low)

---

🏗️ Architecture (Medallion Lakehouse)

The pipeline is implemented in Databricks using a structured 6-stage workflow:

1. Bronze Layer → Raw data ingestion
2. Silver Layer → Feature engineering
3. Model Training → XGBoost + Logistic Regression with Platt Calibration
4. Fairness Audit → Detects bias across demographic groups
5. Explainability → SHAP-based feature importance
6. Gold Layer → Final dataset with predictions & explanations

---

🌟 Key Differentiators

1. 🗣️ Jargon-Free Explainability

Converts complex model outputs into simple sentences:
"Grade fell 2.3 pts; low engagement; financial stress detected."

---

2. 📊 Statistically Reliable Predictions

Uses Platt Calibration so probabilities are meaningful:
A score of "0.85" = 85% actual risk

---

3. ⚖️ Intersectional Fairness

Ensures the model does NOT discriminate across:

- Gender
- Socio-economic status
- Combined vulnerable groups

---

📂 Project Structure

app.py → Main application  
templates/ → Frontend HTML  
static/ → CSS & JS  
notebooks/ → ML pipeline  
dataset → Student data  

---

▶️ How to Run (Local)

pip install -r requirements.txt
python app.py

---

📊 Final Output

SELECT student_id, risk_score, intervention_tier, reason_text 
FROM gold.at_risk_students
WHERE intervention_tier IN ('HIGH', 'MEDIUM')
ORDER BY risk_score DESC;

---

👥 Contributors

- Harsha B
- Tarun N
- B V Hitesh Sai

---

🚀 Future Improvements

- Improve model accuracy
- Add real-time prediction
- Enhance UI/UX
- Deploy as a live web application

---

🏁 Conclusion

This project bridges the gap between advanced machine learning and real-world usability by combining accuracy, fairness, and explainability into a single system.

---

Built for a 24-hour hackathon.
