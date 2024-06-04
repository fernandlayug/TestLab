import pickle

# Load the .pkl model
with open('best_xgb_model_2.pkl', 'rb') as file:
    model = pickle.load(file)

# If the model is a scikit-learn model, you can check feature names like this:
try:
    # If the model has a 'feature_importances_' attribute (e.g., DecisionTree, RandomForest)
    if hasattr(model, 'feature_importances_'):
        print("Feature importances:", model.feature_importances_)
    
    # If the model has a 'coef_' attribute (e.g., LinearRegression, LogisticRegression)
    if hasattr(model, 'coef_'):
        print("Coefficients:", model.coef_)
    
    # If the model has been fitted with a DataFrame, feature names might be stored in the 'columns' attribute
    if hasattr(model, 'feature_names_in_'):
        print("Feature names:", model.feature_names_in_)
except Exception as e:
    print("An error occurred while accessing model attributes:", e)
