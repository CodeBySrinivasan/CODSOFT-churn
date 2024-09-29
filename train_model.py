# train_model.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import pickle

# Load dataset
df = pd.read_csv('Churn_Modelling.csv')

# Data Preprocessing
le_gender = LabelEncoder()
df['Gender'] = le_gender.fit_transform(df['Gender'])  # 1 for female, 0 for male

le_geo = LabelEncoder()
df['Geography'] = le_geo.fit_transform(df['Geography'])  # Geography encoding

# Split data into features and target variable
X = df.drop(columns=['RowNumber', 'CustomerId', 'Surname', 'Exited'])
y = df['Exited']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=3)

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the Random Forest Classifier
rf_clf = RandomForestClassifier(n_estimators=100, random_state=3)
rf_clf.fit(X_train_scaled, y_train)

# Save the model and the scaler
with open('trained_rf_model.pkl', 'wb') as model_file:
    pickle.dump(rf_clf, model_file)

with open('scaler.pkl', 'wb') as scaler_file:
    pickle.dump(scaler, scaler_file)

print("Model and Scaler saved to disk.")
