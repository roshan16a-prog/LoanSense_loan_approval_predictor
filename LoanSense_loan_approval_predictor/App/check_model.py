
import joblib
import pandas as pd

try:
    assets = joblib.load('best_model_assets.pkl')
    model = assets['model']
    print(f"Loaded model type: {type(model).__name__}")
    print(f"Model parameters: {model.get_params()}")
    
    if 'scaler' in assets:
        print("Scaler is present.")
    if 'le_dict' in assets:
        print(f"Label encoders present for: {list(assets['le_dict'].keys())}")
        
except Exception as e:
    print(f"Error loading model: {e}")
