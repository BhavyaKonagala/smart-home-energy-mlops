from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)

MODEL_PATH = "smart_home_energy_model.pkl"

model = joblib.load(MODEL_PATH)


@app.route("/")
def home():
    return jsonify({
        "message": "Smart Home Energy Prediction API",
        "status": "running"
    })


@app.route("/health")
def health():
    return jsonify({
        "status": "healthy",
        "model": "smart_home_energy_model.pkl"
    })


@app.route("/predict", methods=["POST"])
def predict():

    data = request.get_json()

    required_features = [
        "Household_Size",
        "Monthly_Income",
        "Home_Area_sqft",
        "AC_Hours_Daily",
        "Refrigerator_Hours",
        "Washing_Machine_Uses",
        "TV_Hours_Daily",
        "Computer_Hours_Daily",
        "Lighting_Hours_Daily",
        "Smart_Appliances"
    ]

    input_data = {
        feature: data[feature]
        for feature in required_features
    }

    prediction = model.predict([[
        input_data[feature]
        for feature in required_features
    ]])[0]

    return jsonify({
        "prediction": int(prediction)
    })


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=5000
    )
