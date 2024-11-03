import logging
from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

# Load the trained model
model = joblib.load('liver_disease_model.pkl')

# Set up logging
logging.basicConfig(filename='predictions.log', level=logging.INFO,
                    format='%(asctime)s:%(levelname)s:%(message)s')

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Get input data from form
        age = int(request.form['age'])
        gender = int(request.form['gender'])  # 1 for male, 2 for female
        total_bilirubin = float(request.form['total_bilirubin'])
        direct_bilirubin = float(request.form['direct_bilirubin'])
        alkaline_phosphatase = float(request.form['alkaline_phosphatase'])
        alanine_aminotransferase = float(request.form['alanine_aminotransferase'])
        aspartate_aminotransferase = float(request.form['aspartate_aminotransferase'])
        total_proteins = float(request.form['total_proteins'])
        albumin = float(request.form['albumin'])
        albumin_and_globulin_ratio = float(request.form['albumin_and_globulin_ratio'])
        
        # Prepare input for prediction
        input_features = [[age, gender, total_bilirubin, direct_bilirubin,
                           alkaline_phosphatase, alanine_aminotransferase,
                           aspartate_aminotransferase, total_proteins,
                           albumin, albumin_and_globulin_ratio]]

        # Make prediction
        prediction = model.predict(input_features)[0]

        # Log the prediction
        logging.info(f'Input: {input_features}, Prediction: {prediction}')

        return render_template('result.html', prediction=prediction)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
