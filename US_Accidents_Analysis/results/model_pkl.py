import pickle

with open("results/random_forest_model.pkl", "wb") as f:
    pickle.dump(rf_model, f)

#Save trained machine learning models for future use