import joblib

# Load the trained model
model = joblib.load('best_xgb_model_2.pkl')

# Get the feature names
feature_names = model.get_booster().feature_names

print("Feature Names:", feature_names)
