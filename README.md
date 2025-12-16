\# 🏦 Bank Loan Default Prediction System




\## 📋 Executive Summary

This project implements a machine learning pipeline to predict loan defaults. By comparing performance with and without Credit Scores, the analysis proved that \*\*financial fundamentals (Debt-to-Income ratios)\*\* alone are insufficient for prediction in this dataset, while the \*\*Credit History (CIBIL Score)\*\* is the dominant predictor.



\## 📊 Key Results



| Model | Accuracy | ROC-AUC | Key Insight |

| :--- | :--- | :--- | :--- |

| \*\*Logistic Regression\*\* | 92.04% | 0.97 | Strong baseline but missed non-linear edge cases. |

| \*\*Random Forest\*\* | \*\*99.88%\*\* | \*\*1.00\*\* | Near-perfect classification using feature engineering + SMOTE. |

| \*\*Decision Tree\*\* | \*\*100.0%\*\* | \*\*1.00\*\* | Identified the exact credit score threshold used by the bank. |



\## 🛠️ The Analysis Story



\### 1. The "Stress Test" (Financials Only)

I initially removed the `cibil\_score` to see if a user's income and assets were enough to predict default.

\* \*\*Result:\*\* Accuracy dropped to ~53% (Random Guessing).

\* \*\*Conclusion:\*\* In this specific dataset, loan approval decisions are strictly rule-based around credit scores.



\### 2. Feature Engineering

To try and recover performance without the credit score, I engineered behavioral risk ratios:

\* \*\*`loan\_to\_income`\*\*: Ratio of loan amount to annual income.

\* \*\*`asset\_to\_loan`\*\*: Coverage of the loan by available assets.

\* \*Result:\* These features captured ~13% of the decision logic, but could not replace credit history.



\### 3. Handling Imbalance

The dataset was highly imbalanced. I used \*\*SMOTE (Synthetic Minority Over-sampling Technique)\*\* to balance the training data, ensuring the model learned to identify defaults rather than just guessing "Approved".



\## 🚀 How to Run

1\. Clone the repository:

&nbsp;  ```bash

&nbsp;  git clone \[https://github.com/YOUR_USERNAME/bank-loan-prediction.git](https://github.com/YOUR_USERNAME/bank-loan-prediction.git)

