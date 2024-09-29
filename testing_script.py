import pandas as pd
import pickle
import numpy as np

# Load the dataset
df = pd.read_csv('Churn_Modelling.csv')

# Check the distribution of the target variable
print("Distribution of the target variable:")
print(df['Exited'].value_counts())

# Load the trained model and scaler
model = pickle.load(open('trained_rf_model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))
le_gender = pickle.load(open('le_gender.pkl', 'rb'))
le_geo = pickle.load(open('le_geo.pkl', 'rb'))

# Create sample input data
input_data = [
    [500, 'France', 'Female', 28, 1, 1000, 1, 0, 0, 30000],
    [700, 'Spain', 'Male', 35, 5, 75000, 2, 1, 1, 60000],
    [450, 'Germany', 'Female', 22, 1, 500, 1, 0, 0, 25000],
    [600, 'France', 'Male', 45, 10, 150000, 1, 1, 1, 80000],
    [580, 'Spain', 'Female', 30, 2, 2000, 1, 0, 0, 30000]
]

for data in input_data:
    # Encode categorical features
    data[1] = le_geo.transform([data[1]])[0]  # Geography
    data[2] = le_gender.transform([data[2]])[0]  # Gender
    
    # Convert to DataFrame for scaling
    input_df = pd.DataFrame([data], columns=['CreditScore', 'Geography', 'Gender', 'Age', 
                                             'Tenure', 'Balance', 'NumOfProducts', 
                                             'HasCrCard', 'IsActiveMember', 'EstimatedSalary'])
    
    # Scale the input
    input_scaled = scaler.transform(input_df)

    # Make a prediction
    prediction = model.predict(input_scaled)
    print(f"Input: {data} -> Prediction: {'Churn' if prediction[0] == 1 else 'Not Churn'}")
