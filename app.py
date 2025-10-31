from flask import Flask, request, jsonify
import pandas as pd
import joblib
import os

app = Flask(__name__)

# Load model and preprocessor
try:
    model = joblib.load('student_performance_model.pkl')
    preprocessor = joblib.load('data_preprocessor.pkl')
    print("✅ Model and preprocessor loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None
    preprocessor = None

@app.route('/')
def home():
    with open('home.html', 'r', encoding='utf-8') as f:
        return f.read()

@app.route('/predict')
def predict_page():
    with open('index.html', 'r', encoding='utf-8') as f:
        return f.read()

@app.route('/sample')
def sample_csv():
    from flask import send_file
    return send_file('sample_data.csv', as_attachment=True, download_name='sample_student_data.csv')

@app.route('/dashboard')
def dashboard():
    with open('dashboard.html', 'r', encoding='utf-8') as f:
        return f.read()

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if not model or not preprocessor:
            return jsonify({'error': 'Model not loaded properly'}), 500
        
        data = request.json
        
        # Create DataFrame with the exact column names expected by the model
        df = pd.DataFrame([{
            'Quiz01 [10]': float(data['quiz1']),
            'Assignment01 [8]': float(data['assign1']),
            'Midterm Exam [20]': float(data['midterm']),
            'Assignment02 [12]': float(data['assign2']),
            'Assignment03 [25]': float(data['assign3']),
            'Final Exam [35]': float(data['final'])
        }])
        
        # Preprocess and predict
        X_processed = preprocessor.transform(df)
        prediction = model.predict(X_processed)[0]
        
        # Calculate additional metrics
        total = sum([float(data[key]) for key in ['quiz1', 'assign1', 'midterm', 'assign2', 'assign3', 'final']])
        percentage = (total / 110) * 100
        
        return jsonify({
            'prediction': prediction,
            'total': round(total, 1),
            'percentage': round(percentage, 1)
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/predict_csv', methods=['POST'])
def predict_csv():
    try:
        if not model or not preprocessor:
            return jsonify({'error': 'Model not loaded properly'}), 500
        
        data = request.json
        csv_data = data['csv_data']
        
        # Parse CSV data
        lines = csv_data.strip().split('\n')
        headers = [h.strip() for h in lines[0].split(',')]
        
        results = []
        for i in range(1, len(lines)):
            if lines[i].strip():
                values = [float(v.strip()) for v in lines[i].split(',')]
                if len(values) >= 6:
                    # Create DataFrame
                    df = pd.DataFrame([{
                        'Quiz01 [10]': values[0],
                        'Assignment01 [8]': values[1],
                        'Midterm Exam [20]': values[2],
                        'Assignment02 [12]': values[3],
                        'Assignment03 [25]': values[4],
                        'Final Exam [35]': values[5]
                    }])
                    
                    # Predict
                    X_processed = preprocessor.transform(df)
                    prediction = model.predict(X_processed)[0]
                    
                    total = sum(values[:6])
                    percentage = (total / 110) * 100
                    
                    results.append({
                        'original_data': values,
                        'prediction': prediction,
                        'total': round(total, 1),
                        'percentage': round(percentage, 1)
                    })
        
        return jsonify({'results': results, 'headers': headers})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True, port=5000)