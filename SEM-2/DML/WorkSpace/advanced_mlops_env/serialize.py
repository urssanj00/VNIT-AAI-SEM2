from  hparam_tuning import *


import pickle

# Serialize the best model
with open("best_svm_model.pkl", "wb") as f:
    pickle.dump(best_model, f)
print("Model serialized and saved as 'best_svm_model.pkl'.")
