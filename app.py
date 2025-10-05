import pandas as pd
import joblib
from flask import Flask, render_template, request

# Initialize the Flask application
app = Flask(__name__)

# --- Load the pre-trained models and objects ---
try:
    model = joblib.load("KNN_heart_model.pkl")
    scaler = joblib.load("scaler.pkl")
    expected_columns = joblib.load("columns.pkl")
except FileNotFoundError:
    print("Error: One or more .pkl files not found. Make sure they are in the root directory.")
    model, scaler, expected_columns = None, None, None

# --- Define the routes ---

# Route for the home page (where the form is)
@app.route('/')
def home():
    if model is None:
        # Handle the case where models failed to load
        return "Error: Model files not loaded. Please check the server logs.", 500
    return render_template('index.html')

# Route for handling the prediction
@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return render_template('index.html', prediction_text="⚠️ Error: Model not loaded.")

    # Get user input from the form
    age = int(request.form['age'])
    sex = request.form['sex']
    chest_pain = request.form['chest_pain']
    resting_bp = int(request.form['resting_bp'])
    cholesterol = int(request.form['cholesterol'])
    fasting_bs = int(request.form['fasting_bs'])
    resting_ecg = request.form['resting_ecg']
    max_hr = int(request.form['max_hr'])
    exercise_angina = request.form['exercise_angina']
    oldpeak = float(request.form['oldpeak'])
    st_slope = request.form['st_slope']

    # --- Preprocessing: Same logic as your Streamlit app ---

    # Create a raw dictionary for one-hot encoding
    raw_input = {
        'Age': age,
        'RestingBP': resting_bp,
        'Cholesterol': cholesterol,
        'FastingBS': fasting_bs,
        'MaxHR': max_hr,
        'Oldpeak': oldpeak,
        'Sex_' + sex: 1,
        'ChestPainType_' + chest_pain: 1,
        'RestingECG_' + resting_ecg: 1,
        'ExerciseAngina_' + exercise_angina: 1,
        'ST_Slope_' + st_slope: 1
    }

    # Create a DataFrame from the raw input
    input_df = pd.DataFrame([raw_input])

    # Add missing columns and fill with 0, then reorder
    for col in expected_columns:
        if col not in input_df.columns:
            input_df[col] = 0
    input_df = input_df[expected_columns]

    # Scale the input features
    scaled_input = scaler.transform(input_df)

    # Make the prediction
    prediction = model.predict(scaled_input)[0]

    # Determine the result message
    if prediction == 1:
        result_text = "⚠️ High Risk of Heart Disease"
    else:
        result_text = "✅ Low Risk of Heart Disease"

    # Render the page again with the prediction result
    return render_template('index.html', prediction_text=result_text)

# Run the app
if __name__ == "__main__":
    app.run(debug=True)