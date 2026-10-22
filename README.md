# CreditWise — Modern Fintech SaaS (Streamlit)

## Project structure

```text
creditwise_redesign/
├── app.py
├── requirements.txt
├── loan_approval_data.csv
└── models/
    └── loan_model.pkl
```

Copy your existing `loan_approval_data.csv` into the project root and your existing `models/loan_model.pkl` into the `models/` folder. The app does not retrain the model.

## Run locally

In the terminal, from the project folder:

```bash
pip install -r requirements.txt
python -m streamlit run app.py
```

## Expected model artifact

The app supports the dict artifact saved by the supplied training script:

```python
{
  "model": fitted_pipeline_or_estimator,
  "feature_columns": [...],
  "numeric_columns": [...],
  "categorical_columns": [...],
  "metrics": {...}
}
```

It also tolerates a plain fitted estimator, but then it needs to infer feature fields from the CSV. If the model was trained with different feature names, the form must be adjusted to those exact names.

## Notes

- The design uses a horizontal top navigation rather than the previous sidebar layout.
- Application history is stored in Streamlit session state and is temporary. It is not a permanent database.
- Do not use this demo for real lending decisions without security, authentication, validation, monitoring, fairness testing and compliance review.
