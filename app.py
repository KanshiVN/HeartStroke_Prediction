import pandas as pd
import joblib
from flask import Flask, render_template, request
import os
import requests
import zipfile

# Initialize the Flask application
app = Flask(__name__)

# --- NEW: Logic to Download, Unzip, and Load Models ---

def download_and_unzip_models(url, dest_folder='/tmp'):
    """Downloads and unzips model files from a URL."""
    zip_filename = 'models.zip'
    zip_filepath = os.path.join(dest_folder, zip_filename)
    
    # Define the path for one of the expected pkl files to check if extraction is needed
    model_check_path = os.path.join(dest_folder, 'KNN_heart_model.pkl')
    
    if not os.path.exists(model_check_path):
        print("Downloading models...")
        try:
            with requests.get(url, stream=True) as r:
                r.raise_for_status()
                with open(zip_filepath, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
            print("Download complete.")

            print("Unzipping models...")
            with zipfile.ZipFile(zip_filepath, 'r') as zip_ref:
                zip_ref.extractall(dest_folder)
            print("Unzipping complete.")
            
            # Clean up the downloaded zip file
            os.remove(zip_filepath)
        except Exception as e:
            print(f"An error occurred: {e}")
            return False
    else:
        print("Models already exist in /tmp directory.")
    return True

# URL from your GitHub Release for the models.zip file
MODELS_ZIP_URL = 'PASTE_YOUR_MODELS.ZIP_URL_HERE'

# Call the function to ensure models are ready
models_ready = download_and_unzip_models(MODELS_ZIP_URL)

# Define paths to the extracted models in the /tmp directory
model_path = os.path.join('/tmp', 'KNN_heart_model.pkl')
scaler_path = os.path.join('/tmp', 'scaler.pkl')
columns_path = os.path.join('/tmp', 'columns.pkl')

# Load the pre-trained models and objects using joblib
try:
    if models_ready:
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        expected_columns = joblib.load(columns_path)
    else:
        raise FileNotFoundError
except FileNotFoundError:
    print("Error: Model files not found after download attempt.")
    model, scaler, expected_columns = None, None, None

# --- Your existing Flask routes (no changes needed below) ---

@app.route('/')
def home():
    if model is None:
        return "Error: Model files not loaded. Please check the server logs.", 500
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return render_template('index.html', prediction_text="⚠️ Error: Model not loaded.")

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

    raw_input = {
        'Age': age, 'RestingBP': resting_bp, 'Cholesterol': cholesterol,
        'FastingBS': fasting_bs, 'MaxHR': max_hr, 'Oldpeak': oldpeak,
        'Sex_' + sex: 1, 'ChestPainType_' + chest_pain: 1,
        'RestingECG_' + resting_ecg: 1, 'ExerciseAngina_' + exercise_angina: 1,
        'ST_Slope_' + st_slope: 1
    }

    input_df = pd.DataFrame([raw_input])
    for col in expected_columns:
        if col not in input_df.columns:
            input_df[col] = 0
    input_df = input_df[expected_columns]

    scaled_input = scaler.transform(input_df)
    prediction = model.predict(scaled_input)[0]

    if prediction == 1:
        result_text = "⚠️ High Risk of Heart Disease"
    else:
        result_text = "✅ Low Risk of Heart Disease"

    return render_template('index.html', prediction_text=result_text)

if __name__ == "__main__":
    app.run(debug=True)
