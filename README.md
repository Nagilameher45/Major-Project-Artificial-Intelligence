# Major-Project-Artificial-Intelligence
A Complete Machine Learning & Streamlit Web Application

This repository contains a full-featured Heart Disease Prediction web application built using Python, Streamlit, and Machine Learning.
The app allows users to:

Upload a semicolon-separated CSV dataset
Automatically fix and clean the dataset
Perform EDA (Histograms, Correlation Matrix)
Train and evaluate 5 ML models
View confusion matrices, ROC curves, and metrics
Auto-select and save the best-performing model
Predict heart disease for a single patient using a trained model
Download the best model in .joblib format

🚀 Features
✔ Dataset Handling
Upload semicolon-separated CSV
Auto-split into correct columns
Automatic data cleaning and numeric coercion

✔ Exploratory Data Analysis (EDA)
Descriptive statistics
Histograms for numerical features
Correlation heatmap

✔ Machine Learning Models
The following models are trained and evaluated:
Logistic Regression
K-Nearest Neighbors (KNN)
Decision Tree
Random Forest
Support Vector Machine (SVM)

Using:
80/20 split
StandardScaler
5-fold Stratified Cross Validation

✔ Evaluation Metrics
Accuracy
Precision
Recall
F1 Score
ROC-AUC Score
Confusion Matrix
ROC Curve

✔ Model Export
Automatically identifies the best model based on test accuracy
Retrains best model on full dataset
Saves as:
best_model_cardio.joblib


Model can be downloaded directly from the UI

✔ Real-Time Prediction
Enter:
Age
Gender
Blood pressure
Glucose, cholesterol
Lifestyle factors (smoking, alcohol, activity)

Get:
Prediction label (Low or High chance of heart disease)
Probability score

📁 Project Structure
📦 HeartDiseaseApp
 ┣ 📜 app.py                 # Main Streamlit application
 ┣ 📜 best_model_cardio.joblib (generated after training)
 ┣ 📜 README.md              # Project documentation
 ┗ 📄 requirements.txt       # Python dependencies

🛠 Installation & Setup
1. Clone the Repository
git clone https://github.com/yourusername/HeartDiseaseApp.git
cd HeartDiseaseApp

2. Install Dependencies
Create a virtual environment (recommended):
pip install -r requirements.txt


Suggested requirements.txt content:

streamlit
pandas
numpy
scikit-learn
matplotlib
seaborn
joblib

▶ Run the Application
streamlit run app.py


Streamlit will start your local server, usually at:

http://localhost:8501/

🌐 Running in Google Colab (Optional)

If you're using Google Colab, use the following to expose Streamlit via ngrok:

!pip install pyngrok streamlit
from pyngrok import ngrok
ngrok.set_auth_token("YOUR_TOKEN")
public_url = ngrok.connect(8501)
public_url


Then run:

streamlit run app.py --server.port 8501 --server.address 0.0.0.0

📊 Dataset Format (Required)
Your dataset must be semicolon-separated with these columns:
id;age;gender;height;weight;ap_hi;ap_lo;cholesterol;gluc;smoke;alco;active;cardio


The app auto-fixes this format if provided incorrectly.

🧠 Model Details

After training, the app generates:
Cross-validation results
Test set evaluation table
Best model selection summary
Downloadable trained model
The saved model contains:
{
  "scaler": StandardScaler(),
  "model": BestEstimator,
  "features": [...feature list...]
}

🎯 Use Cases

This project is ideal for:

✔ Machine learning students
✔ Healthcare analytics demos
✔ Data science portfolio projects
✔ Colleges/universities practical submissions
✔ Medical prediction prototypes

Nagila Meher A
Machine Learning Intern
