import os
import joblib
import pandas as pd
import numpy as np
from flask import Flask, render_template, request

app = Flask(__name__)

# --- PATH CONFIGURATION ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "src", "saved_models")

# --- LOAD ALL ARTIFACTS ---
model = joblib.load(os.path.join(MODEL_PATH, "best_model_decision_tree.pkl"))
scaler = joblib.load(os.path.join(MODEL_PATH, "minmax_scaler.pkl"))
target_encoder = joblib.load(os.path.join(MODEL_PATH, "target_encoder.pkl"))
label_encoder = joblib.load(os.path.join(MODEL_PATH, "target_label_encoder.pkl"))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # 1. Capture data from form
        raw_data = request.form.to_dict()
        
        # 2. Convert to DataFrame (Essential for Scikit-Learn 1.0+)
        # Ensure the keys in your index.html match your training CSV columns exactly
        input_df = pd.DataFrame([raw_data])

        # 3. Apply Encoders (Convert categories to numbers)
        # Note: If your target_encoder is a 'category_encoders' object, use .transform
        encoded_data = target_encoder.transform(input_df)

        # 4. Apply Scaler (Normalize the numbers)
        final_input = scaler.transform(encoded_data)

        # 5. Make Prediction (0 or 1)
        prediction_numeric = model.predict(final_input)

        # 6. Translate (0 -> Fail, 1 -> Pass) using your Label Encoder
        prediction_text = label_encoder.inverse_transform(prediction_numeric)[0]
        
        # UI styling based on result
        result_class = "success-msg" if "pass" in str(prediction_text).lower() or "1" in str(prediction_text) else "error-msg"

        return render_template('index.html', 
                               prediction_text=f"Prediction: {prediction_text}",
                               result_class=result_class)

    except Exception as e:
        return render_template('index.html', 
                               prediction_text=f"Error in processing: {str(e)}", 
                               result_class="error-msg")

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)