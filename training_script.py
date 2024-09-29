import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Load the dataset
df = pd.read_csv('Churn_Modelling.csv')

# Encode categorical variables
le_gender = LabelEncoder()
df['Gender'] = le_gender.fit_transform(df['Gender'])

le_geo = LabelEncoder()
df['Geography'] = le_geo.fit_transform(df['Geography'])

# Prepare features and target variable
X = df[['CreditScore', 'Geography', 'Gender', 'Age', 'Tenure', 'Balance', 
         'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary']]
y = df['Exited']

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Train the model
model = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
model.fit(X_train_scaled, y_train)

# Save the model, scaler, and encoders
pickle.dump(model, open('trained_rf_model.pkl', 'wb'))
pickle.dump(scaler, open('scaler.pkl', 'wb'))
pickle.dump(le_gender, open('le_gender.pkl', 'wb'))
pickle.dump(le_geo, open('le_geo.pkl', 'wb'))

print("Model training complete and files saved.")
