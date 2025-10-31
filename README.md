# 🎓 AI-Powered Student Performance Prediction

A modern web application that uses machine learning to predict student performance based on assignment scores and exam results.

## ✨ Features

- **🤖 AI Predictions**: Trained machine learning model for accurate performance forecasting
- **📊 Interactive Dashboard**: Comprehensive analytics with charts and statistics
- **📁 CSV Processing**: Batch upload and process multiple student records
- **🎨 3D Interface**: Modern UI with 3D animations and effects
- **📈 Real-time Analytics**: Live performance tracking and insights

## 🚀 Quick Start

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the Application**
   ```bash
   python app.py
   ```

3. **Open Browser**
   ```
   http://localhost:5000
   ```

## 📁 Project Structure

```
├── app.py                 # Flask backend server
├── home.html             # Landing page with 3D effects
├── index.html            # Prediction interface
├── dashboard.html        # Analytics dashboard
├── sample_data.csv       # Sample student data
├── requirements.txt      # Python dependencies
├── student_performance_model.pkl    # Trained ML model
├── data_preprocessor.pkl           # Data preprocessing pipeline
└── README.md            # Project documentation
```

## 🎯 Usage

### Manual Prediction
1. Navigate to the prediction page
2. Enter individual assignment scores
3. Get instant AI-powered performance prediction

### CSV Batch Processing
1. Upload a CSV file with student data
2. View batch predictions in a table format
3. All data is automatically saved to the dashboard

### Analytics Dashboard
1. View comprehensive performance statistics
2. Interactive charts showing grade distribution
3. Performance trends across all assessments

## 📊 Model Information

- **Algorithm**: Random Forest Classifier
- **Features**: Quiz scores, assignment grades, exam results
- **Output**: Performance classification (A, B, C, F grades)
- **Accuracy**: Optimized through hyperparameter tuning

## 🛠️ Technology Stack

- **Backend**: Flask (Python)
- **Frontend**: HTML5, CSS3, JavaScript
- **ML**: scikit-learn, pandas, joblib
- **Charts**: Chart.js
- **Styling**: Modern CSS with 3D transforms

## 📝 CSV Format

```csv
Quiz01 [10],Assignment01 [8],Midterm Exam [20],Assignment02 [12],Assignment03 [25],Final Exam [35]
8.5,7.2,18.0,10.5,22.0,32.0
```

## 🎨 Features

- **3D Animations**: Floating particles and rotating cubes
- **Responsive Design**: Works on all device sizes
- **Real-time Updates**: Dashboard updates automatically
- **Modern UI**: Gradient backgrounds and smooth transitions

## 📄 License

This project is open source and available under the MIT License.