# Loan Prediction Model

An XGBoost binary classifier that predicts whether a loan application should be **approved (1)** or **rejected (0)**, based on an applicant's cash flow behavior, requested loan terms, and credit repayment history.

## Model Details

| Property | Value |
|---|---|
| Algorithm | XGBoost (`XGBClassifier`) |
| Objective | `binary:logistic` |
| Number of trees (`n_estimators`) | 300 |
| Max tree depth | 4 |
| Learning rate | 0.1 |
| Random state | 42 |
| Number of input features | 9 |
| Output classes | `0` = Rejected, `1` = Approved |

## Input Features

The model expects a single row of 9 numeric features, in this exact order:

| # | Feature | Description |
|---|---|---|
| 1 | `Monthly_Credit_Velocity` | Rate/frequency of money credited into the applicant's account per month |
| 2 | `Monthly_Debit_Velocity` | Rate/frequency of money debited from the applicant's account per month |
| 3 | `Monthly_Cashflow_Surplus` | Net surplus (income − expenses) available monthly |
| 4 | `Requested_Loan_Amount` | Total loan amount requested by the applicant |
| 5 | `Requested_Loan_EMI` | Equated Monthly Installment (EMI) for the requested loan |
| 6 | `Loan_Tenure_Months` | Requested loan repayment tenure, in months |
| 7 | `DTI_Percent` | Debt-to-Income ratio, as a percentage |
| 8 | `Credit_Repayment_History_Good` | Binary flag (1/0) — applicant has a good repayment history |
| 9 | `Credit_Repayment_History_Poor` | Binary flag (1/0) — applicant has a poor repayment history |

> **Note:** Missing values are natively supported by XGBoost (`missing=nan`) and do not need to be imputed before prediction.

## Feature Importance

Based on the trained model, the features driving predictions the most are:

| Feature | Importance |
|---|---|
| `Credit_Repayment_History_Good` | 35.6% |
| `Credit_Repayment_History_Poor` | 24.1% |
| `Monthly_Cashflow_Surplus` | 15.2% |
| `Requested_Loan_Amount` | 14.3% |
| `DTI_Percent` | 3.3% |
| `Requested_Loan_EMI` | 3.7% |
| `Monthly_Credit_Velocity` | 2.0% |
| `Monthly_Debit_Velocity` | 1.1% |
| `Loan_Tenure_Months` | 0.8% |

**Takeaway:** An applicant's past repayment behavior (good/poor history) is by far the strongest predictor of loan approval, followed by their monthly cashflow surplus and the requested loan amount.

## Requirements

```
python >= 3.9
xgboost
scikit-learn
pandas
numpy
```

Install with:

```bash
pip install xgboost scikit-learn pandas numpy
```

## Usage

```python
import pickle
import pandas as pd

# Load the model
with open("Loan_prediction_model.pkl", "rb") as f:
    model = pickle.load(f)

# Prepare input data (single applicant example)
sample = pd.DataFrame([{
    "Monthly_Credit_Velocity": 12,
    "Monthly_Debit_Velocity": 9,
    "Monthly_Cashflow_Surplus": 25000,
    "Requested_Loan_Amount": 500000,
    "Requested_Loan_EMI": 12500,
    "Loan_Tenure_Months": 48,
    "DTI_Percent": 32,
    "Credit_Repayment_History_Good": 1,
    "Credit_Repayment_History_Poor": 0
}])

# Predict class (0 = Rejected, 1 = Approved)
prediction = model.predict(sample)

# Predict probability of approval
probability = model.predict_proba(sample)[:, 1]

print("Prediction:", "Approved" if prediction[0] == 1 else "Rejected")
print(f"Approval probability: {probability[0]:.2%}")
```

**Important:** The input DataFrame's columns must match the 9 feature names above, in the same order, or be passed as a dict/DataFrame with matching column names (XGBoost's sklearn wrapper will align by name if a DataFrame is used).

## Output

| Method | Returns |
|---|---|
| `model.predict(X)` | `0` (Rejected) or `1` (Approved) |
| `model.predict_proba(X)` | `[P(class=0), P(class=1)]` — probability for each class |

## File Info

- **File:** `Loan_prediction_model.pkl`
- **Format:** Python pickle (serialized `xgboost.sklearn.XGBClassifier` object)
- **Loading:** Use `pickle.load()` — requires `xgboost` to be installed in the environment, since the object is an XGBoost-native class, not a plain Booster file.

## 📄 License

This project is licensed under the MIT License.

## 👨‍💻 Author

Parv Shah

If you find any bugs, have feature requests, or would like to contribute, feel free to open an Issue or submit a Pull Request.
