import joblib
import pandas as pd

model = joblib.load("model/final_model.pkl")

def predict(sample_dict):
    sample_df = pd.DataFrame([sample_dict])

    prediction = model.predict(sample_df)[0]

    return prediction