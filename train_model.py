import pandas as pd
from sklearn.svm import SVC
import pickle

# Load your training data
training = pd.read_csv('datasets/Training.csv')

# Prepare your data
X = training.iloc[:, :-1]  # Features
y = training.iloc[:, -1]   # Target

# Create and train the model
svc = SVC()
svc.fit(X, y)

# Save the model
with open('models/svc.pkl', 'wb') as f:
    pickle.dump(svc, f) 