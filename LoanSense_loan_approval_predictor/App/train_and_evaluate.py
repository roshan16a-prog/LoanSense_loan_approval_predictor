
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import joblib

# 1. Load Data
print("Loading data...")
df = pd.read_csv("loan_approval_data.csv")

# 2. Preprocessing
print("Preprocessing...")
# Drop rows where target is missing
df.dropna(subset=['Loan_Approved'], inplace=True)
print(f"Rows after dropping missing targets: {len(df)}")

# Handling missing values
numerical_cols = df.select_dtypes(include=['number']).columns
categorical_cols = df.select_dtypes(include=['object']).columns
categorical_cols = [c for c in categorical_cols if c != 'Loan_Approved']

# Check unique values in target
print(f"Unique values in Loan_Approved before encoding: {df['Loan_Approved'].unique()}")

# Impute
num_imp = SimpleImputer(strategy='mean')
df[numerical_cols] = num_imp.fit_transform(df[numerical_cols])

cat_imp = SimpleImputer(strategy='most_frequent')
# Ensure Loan_Approved is not imputed if it was excluded (it was). 
# But wait, I excluded it from categorical_cols list.
df[categorical_cols] = cat_imp.fit_transform(df[categorical_cols])

# Encode Categorical Features
le_dict = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    le_dict[col] = le

# Encode Target
le_target = LabelEncoder()
df['Loan_Approved'] = le_target.fit_transform(df['Loan_Approved'].astype(str))
print(f"Unique values in Loan_Approved after encoding: {df['Loan_Approved'].unique()}")

# Feature Engineering (as per notebook)
df['DTI_Ratio_sq'] = df['DTI_Ratio'] ** 2
df['Credit_Score_sq'] = df['Credit_Score'] ** 2

# Drop Applicant_ID if present, as it is an ID
if 'Applicant_ID' in df.columns:
    df.drop(columns=['Applicant_ID'], inplace=True)

X = df.drop(columns=['Loan_Approved', 'Credit_Score', 'DTI_Ratio'])
y = df['Loan_Approved']
feature_names = X.columns.tolist()

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 3. Model Training & Evaluation
models = {
    "Logistic Regression": LogisticRegression(random_state=42),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "Naive Bayes": GaussianNB()
}

results = {}
best_model_name = ""
best_precision = 0
best_model_obj = None

print("\n--- Model Evaluation ---")
for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    
    acc = accuracy_score(y_test, y_pred)
    # Using pos_label=1 assuming 1 is the positive class, or average='weighted' to be safe
    # If binary, it should be 0 and 1.
    try:
        prec = precision_score(y_test, y_pred, pos_label=1)
        rec = recall_score(y_test, y_pred, pos_label=1)
        f1 = f1_score(y_test, y_pred, pos_label=1)
    except ValueError:
        # Fallback for multiclass or if inferred wrong
        print(f"Warning: Multiclass target detected for {name}, switching to weighted average.")
        prec = precision_score(y_test, y_pred, average='weighted')
        rec = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
    
    results[name] = {"Accuracy": acc, "Precision": prec, "Recall": rec, "F1": f1}
    print(f"\n{name}:")
    print(f"  Accuracy: {acc:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall: {rec:.4f}")
    print(f"  F1 Score: {f1:.4f}")
    
    # Selection criteria: Notebook chose Precision
    if prec > best_precision:
        best_precision = prec
        best_model_name = name
        best_model_obj = model
    elif prec == best_precision:
        # Tie breaker: F1
        if f1 > results[best_model_name]["F1"]:
            best_model_name = name
            best_model_obj = model

print(f"\nBest Model selected based on Precision: {best_model_name}")

# 4. Save Best Model
print("Saving best model...")
assets = {
    'model': best_model_obj,
    'scaler': scaler,
    'le_dict': le_dict,
    'le_target': le_target,
    'feature_names': feature_names
}
joblib.dump(assets, 'best_model_assets.pkl')
print(f"Saved {best_model_name} to best_model_assets.pkl")

# 5. Verify Prediction (Sample)
print("\n--- Verification Prediction ---")
sample_input = X_test_scaled[0].reshape(1, -1)
prediction = best_model_obj.predict(sample_input)[0]
prediction_decoded = le_target.inverse_transform([prediction])[0]
actual = le_target.inverse_transform([y_test.iloc[0]])[0]

print(f"Sample Test Input Index: 0")
print(f"Predicted: {prediction_decoded}")
print(f"Actual: {actual}")
if prediction_decoded == actual:
    print("Prediction is CORRECT for this sample.")
else:
    print("Prediction is INCORRECT for this sample.")
