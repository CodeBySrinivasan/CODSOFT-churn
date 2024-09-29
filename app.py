from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained model and scaler
model = pickle.load(open('trained_rf_model.pkl', 'rb'))  # Adjust path as necessary
scaler = pickle.load(open('scaler.pkl', 'rb'))  # Adjust path as necessary

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract features from the form data
        credit_score = float(request.form['CreditScore'])
        geography = int(request.form['Geography'])
        gender = int(request.form['Gender'])
        age = float(request.form['Age'])
        tenure = float(request.form['Tenure'])
        balance = float(request.form['Balance'])
        num_of_products = int(request.form['NumOfProducts'])
        has_cr_card = int(request.form['HasCrCard'])
        is_active_member = int(request.form['IsActiveMember'])
        estimated_salary = float(request.form['EstimatedSalary'])

        # Prepare the input for the model
        input_data = np.array([[credit_score, geography, gender, age, tenure, balance, num_of_products, has_cr_card, is_active_member, estimated_salary]])

        # Scale the input data
        input_data_scaled = scaler.transform(input_data)

        # Make a prediction
        prediction = model.predict(input_data_scaled)[0]  # Get the prediction

        # Log raw predictions
        print(f"Raw prediction: {prediction}")

        # Convert the prediction to a standard Python type
        prediction_result = 'Churn' if int(prediction) == 1 else 'Not Churn'

        # Return a JSON response
        return jsonify({'prediction': prediction_result})

    except KeyError as e:
        return jsonify({'error': f'Missing parameter: {str(e)}'}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)
