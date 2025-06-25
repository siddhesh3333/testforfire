from flask import Flask, render_template, request
import pickle
import numpy as np

application = Flask(__name__)
app = application

# Load the Ridge regression model and StandardScaler
try:
    ridge_model = pickle.load(open('model/ridge.pkl', 'rb'))
    standard_scaler = pickle.load(open('model/scaler.pkl', 'rb'))
except Exception as e:
    print("Error loading model or scaler:", e)
    ridge_model = None
    standard_scaler = None

@app.route("/")
def index():
    # Render the home page (index.html)
    return render_template("index.html")

@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == "POST":
        try:
            # Extract features from the form
            Temperature = float(request.form.get('Temperature'))
            RH = float(request.form.get('RH'))
            Ws = float(request.form.get('Ws'))
            Rain = float(request.form.get('Rain'))
            FFMC = float(request.form.get('FFMC'))
            DMC = float(request.form.get("DMC"))
            ISI = float(request.form.get('ISI'))
            Classes = float(request.form.get('Classes'))
            Region = float(request.form.get('Region'))

            # Prepare and scale the input data
            input_data = np.array([[Temperature, RH, Ws, Rain, FFMC, DMC, ISI, Classes, Region]])
            new_data_scaled = standard_scaler.transform(input_data)
            # Make prediction
            result = ridge_model.predict(new_data_scaled)
            # Render result in home.html
            return render_template('home.html', results=result[0])
        except Exception as e:
            # If there's an error, show it in the template
            return render_template('home.html', results=f"Error: {e}")
    else:
        # For GET request, just render the home page
        return render_template('home.html')

if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)
