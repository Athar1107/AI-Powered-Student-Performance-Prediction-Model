from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
import joblib
import os

# Initialize Flask app
app = Flask(__name__)

# Load model and preprocessor
MODEL_PATH = os.path.join("student_performance_model.pkl")
PREPROCESSOR_PATH = os.path.join("data_preprocessor.pkl")

model = joblib.load(MODEL_PATH)
preprocessor = joblib.load(PREPROCESSOR_PATH)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Option 1: If user uploads a CSV
        if 'file' in request.files:
            file = request.files['file']
            if file and file.filename.endswith('.csv'):
                data = pd.read_csv(file)
            else:
                return render_template('index.html', error="Please upload a valid CSV file.")
        else:
            # Option 2: Manual form input
            data = pd.DataFrame([{
                'Quiz01 [10]': float(request.form['quiz1']),
                'Assignment01 [8]': float(request.form['assign1']),
                'Midterm Exam [20]': float(request.form['midterm']),
                'Assignment02 [12]': float(request.form['assign2']),
                'Assignment03 [25]': float(request.form['assign3']),
                'Final Exam [35]': float(request.form['final'])
            }])

        # Preprocess input
        X_processed = preprocessor.transform(data)

        # Make prediction
        prediction = model.predict(X_processed)

        # If multiple rows, show top few predictions
        if len(prediction) > 1:
            result = pd.DataFrame(prediction, columns=['Predicted Class'])
            return render_template('index.html', tables=[result.to_html(classes='table table-striped', index=False)])

        return render_template('index.html', prediction_text=f"Predicted Class: {prediction[0]}")

    except Exception as e:
        return render_template('index.html', error=f"Error: {str(e)}")

if __name__ == "__main__":
    app.run(debug=True)
