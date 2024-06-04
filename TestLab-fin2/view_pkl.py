import pickle

# Replace 'path_to_your_file.pkl' with the actual path to your .pkl file
file_path = 'best_xgb_model_4.pkl'

with open(file_path, 'rb') as file:
    data = pickle.load(file)

print(data)
