import pickle

# Load the model
with open('best_xgb_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Try to access feature names
try:
    if hasattr(model, 'feature_names_in_'):
        feature_names = model.feature_names_in_
    else:
        raise AttributeError("Feature names are not stored in the model.")
    print("Feature names:", feature_names)
except AttributeError as e:
    print(e)
