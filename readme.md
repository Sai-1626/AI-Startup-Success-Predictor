# AI Startup Success Predictor

## Overview
AI Startup Success Predictor is a Machine Learning web application developed using Python and Streamlit. It predicts whether a startup is likely to succeed based on funding, milestones, relationships, location, and business category.

## Features
- Predict startup success
- User-friendly Streamlit interface
- Random Forest Classifier
- Data preprocessing using StandardScaler
- Real-time prediction

## Technologies Used
- Python
- Pandas
- NumPy
- Scikit-learn
- Streamlit
- Joblib

## Machine Learning Model
- Random Forest Classifier
- Accuracy: **82.16%**

## Dataset
Startup Success Prediction Dataset

## Project Structure

```
AI Startup Success Predictor/
│
├── app.py
├── feature_columns.py
├── models/
│   ├── best_model.pkl
│   └── scaler.pkl
├── requirements.txt
└── README.md
```

## How to Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Author

Saijeshwanth