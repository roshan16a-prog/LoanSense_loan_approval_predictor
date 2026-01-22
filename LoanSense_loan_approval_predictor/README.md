# Loan Approval Predictor

This project is a Machine Learning application that predicts loan approval eligibility based on applicant details. It uses a trained model (Logistic Regression/KNN/Naive Bayes) served via a Streamlit web interface.

## Project Structure

- `app.py`: The main Streamlit application for the user interface and prediction.
- `train_and_evaluate.py`: Script to train the machine learning models, evaluate them, and save the best one.
- `check_model.py`: Utility script to check the saved model assets.
- `loan_approval_data.csv`: The dataset used for training.
- `best_model_assets.pkl`: The serialized best model and preprocessing objects.
- `requirements.txt`: List of Python dependencies.

## Setup & Installation

1.  **Install Dependencies**
    Ensure you have Python installed. It is recommended to use a virtual environment.
    ```bash
    pip install -r requirements.txt
    ```

## Usage

### 1. Train the Model (Optional)
If you want to retrain the model or regenerate the assets:
```bash
python train_and_evaluate.py
```
This will generate `best_model_assets.pkl`.

### 2. Run the Application
Start the Streamlit web app:
```bash
streamlit run app.py
```
The application will open in your default web browser (usually at `http://localhost:8501`).

## Features
- **Input Form**: Enter applicant details like Income, Credit Score, Loan Amount, etc.
- **Real-time Prediction**: Get an instant "Approved" or "Rejected" status along with a confidence score.
- **Categorical Handling**: Automatically handles categorical inputs like Education, Marital Status, etc.
"# LoanSense_loan_approval_predictor" 
