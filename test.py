import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
import joblib
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# --- 1. CONFIGURATION AND DATA LOADING ---

# IMPORTANT: The raw string (r"...") is required for Windows paths like this.
# This assumes the file is accessible at the path you provided.
FILE_PATH = r"C:\Users\athar\coding\AI-Powered Student Performance Prediction Mode\Student Performance Prediction - Multiclass Case\Student Performance Prediction-Multi.csv"

try:
    df = pd.read_csv(FILE_PATH)
except FileNotFoundError:
    # Use a relative path as a fallback for execution environment if absolute path fails
    FILE_PATH = 'Student Performance Prediction - Multiclass Case/Student Performance Prediction-Multi.csv'
    try:
        df = pd.read_csv(FILE_PATH)
    except FileNotFoundError:
        print(f"❌ ERROR: Cannot find the data file at either the absolute or relative path.")
        raise

print("\n--- Dataset Info ---")
df.info()
print("\n--- First 5 Rows ---")
print(df.head())
print("\n--- Column Names ---")
print(df.columns.tolist())

# --- 2. TARGET IDENTIFICATION AND DATA PREPARATION ---

# FIX: The true multiclass target is 'Class' (G, F, W). 
# We manually set it and exclude columns that cause data leakage ('Course Grade', 'Total [100]').
TARGET_COL = 'Class' 
print(f"\n✅ Manually set target column to: '{TARGET_COL}'")


# Define columns to exclude: Unique ID, the final class target, and the score columns (Data Leakage)
EXCLUDE_COLS = ['Student ID', TARGET_COL, 'Course Grade', 'Total [100]']
all_features = [c for c in df.columns if c not in EXCLUDE_COLS]

# Define features and target
X = df[all_features]
y = df[TARGET_COL]

# Filter features by data type (object for categorical, numeric for numerical)
# Based on the data info, the remaining features are all numerical (int64)
numerical_cols = [c for c in all_features if pd.api.types.is_numeric_dtype(X[c])]
categorical_cols = [c for c in all_features if pd.api.types.is_object_dtype(X[c])]

print(f"\nNumerical columns used: {numerical_cols}")
print(f"Categorical columns used: {categorical_cols}")

# --- 3. PREPROCESSING PIPELINE DEFINITION ---

# Define preprocessing pipelines
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# FIX: Since there are no remaining categorical features to encode, we use a simple placeholder pipeline
# or just ensure the categorical list is empty. Based on the output, categorical_cols should be empty.
if categorical_cols:
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    transformers = [
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ]
else:
    # Use only the numerical transformer if no categorical columns remain
    transformers = [
        ('num', numerical_transformer, numerical_cols)
    ]
    
# Combine into a ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=transformers,
    remainder='drop'  # drop unused columns (like the original 'Student ID')
)

# Apply preprocessing
X_processed = preprocessor.fit_transform(X)

# Get transformed feature names
# We use .get_feature_names_out() to get the final list of column names after OneHotEncoding
feature_names = preprocessor.get_feature_names_out()

# Convert to DataFrame
X_df = pd.DataFrame(X_processed, columns=feature_names)

print("\n--- Processed Feature Snapshot (First 5 Rows) ---")
print(X_df.head())

# --- 4. CRITICAL FIX: DATA SPLITTING ---
# You need to split the data *after* preprocessing and *before* training.
print("\n--- Data Splitting (Train/Test) ---")
X_train, X_test, y_train, y_test = train_test_split(
    X_df, y,
    test_size=0.2, # 20% for testing
    random_state=42,
    stratify=y # Ensures grades are proportionally distributed
)
print(f"X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")


# --- 5. BASELINE MODEL TRAINING ---

rf_baseline = RandomForestClassifier(random_state=42)

print("\nStarting Random Forest Baseline Training...")
rf_baseline.fit(X_train, y_train)
print("Training complete.")

# Baseline Prediction and Evaluation
y_pred_baseline = rf_baseline.predict(X_test)
print("\n### Baseline Model Performance (Default Settings) ###")
print(f"Accuracy: {accuracy_score(y_test, y_pred_baseline):.4f}")
print("\n--- Classification Report ---")
print(classification_report(y_test, y_pred_baseline))


# --- 6. HYPERPARAMETER TUNING (Grid Search) ---

param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, None],
    'min_samples_leaf': [1, 2],
    'criterion': ['gini', 'entropy']
}

grid_search = GridSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_grid=param_grid,
    cv=5,
    scoring='f1_weighted',
    verbose=1,
    n_jobs=-1
)

print("\nStarting Hyperparameter Tuning (Grid Search)...")
grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_
print("\n### Tuning Results ###")
print(f"Best F1-Weighted Score: {grid_search.best_score_:.4f}")
print(f"Best Parameters Found: {grid_search.best_params_}")


# --- 7. FINAL MODEL EVALUATION ---

y_pred_tuned = best_model.predict(X_test)
print("\n### Final Optimized Model Performance ###")
print(f"Test Accuracy: {accuracy_score(y_test, y_pred_tuned):.4f}")
print("\n--- Classification Report (Optimized Model) ---")
print(classification_report(y_test, y_pred_tuned))

# Confusion Matrix Visualization
plt.figure(figsize=(8, 6))
sns.heatmap(confusion_matrix(y_test, y_pred_tuned), annot=True, fmt='d', cmap='Blues',
            xticklabels=best_model.classes_, yticklabels=best_model.classes_)
plt.title('Confusion Matrix (Optimized Random Forest)')
plt.ylabel('True Grade')
plt.xlabel('Predicted Grade')
plt.show()

# --- 8. FEATURE IMPORTANCE ANALYSIS ---

importances = best_model.feature_importances_
feature_importances = pd.Series(importances, index=X_df.columns).sort_values(ascending=False)

plt.figure(figsize=(10, 8))
# Use the actual column names from the processed DataFrame for the plot
sns.barplot(x=feature_importances.head(15), y=feature_importances.head(15).index)
plt.title('Top 15 Feature Importances for Performance Prediction')
plt.xlabel('Importance Score')
plt.ylabel('Feature')
plt.tight_layout()
plt.show()

# Save model and preprocessor
joblib.dump(best_model, 'student_performance_model.pkl')
joblib.dump(preprocessor, 'data_preprocessor.pkl')

print("✅ Model and preprocessor saved successfully!")
