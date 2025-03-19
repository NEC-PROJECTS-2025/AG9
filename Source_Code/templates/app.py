from flask import Flask, jsonify, request, render_template, redirect, url_for
from pathlib import Path
import pandas as pd
import joblib

# Initialize Flask app
app = Flask(__name__)

# Load the simple model
def load_simple_model():
    model_path = Path("C:\\Users\\user\\Desktop\\SRILATHA\\final\\model\\lgbm_model.pkl")
    print(f"Loading simple model from: {model_path}")
    model = joblib.load(model_path)
    return model

simple_model = load_simple_model()

# Class names for the custom model
class_names = {
    0: "Insufficient Weight",
    1: "Normal Weight",
    2: "Obesity Type I",
    3: "Obesity Type II",
    4: "Obesity Type III",
    5: "Overweight Level I",
    6: "Overweight Level II",
}

# Helper function for prediction
def predict_sample(sample: dict) -> dict:
    sample = sample['data']
    sample = [sample]
    sample_df = pd.DataFrame(sample)

    # Perform feature engineering here (dummy example)
    # Replace with your actual feature engineering logic
    feature = sample_df  # Modify as per your logic

    predictions = simple_model.predict(feature)
    return {"prediction": predictions.item(), "class": class_names.get(predictions.item(), "Unknown")}

@app.route('/')
def home():
    return render_template("home.html", title="Home Page")

@app.route('/about')
def about():
    return render_template("about.html", title="About Project")

@app.route('/prediction', methods=['GET', 'POST'])
def prediction():
    if request.method == 'POST':
        input_data = request.form.get('input_data')
        try:
            prediction = simple_model.predict([[float(input_data)]])  # Modify based on model input
            result = f"Prediction: {prediction[0]}"
        except Exception as e:
            result = f"Error: {str(e)}"
        return render_template("prediction.html", result=result, title="Predictions")
    return render_template("prediction.html", title="Predictions")

@app.route('/obesityRiskForm', methods=["GET", "POST"])
def obesity_risk_form():
    if request.method == "POST":
        # Redirect to the 'predict_obesity' route for handling form submission
        return redirect(url_for("prediction_obesity"))
    return render_template("obesityRiskForm.html")

@app.route('/prediction_obesity', methods=["POST", "GET"])
def predict_obesity():
    if request.method == "POST":
        try:
            # Collect the form data
            sample = {
                "data": {
                    "Gender": request.form.get("gender"),
                    "Age": int(request.form.get("age")),
                    "Height": float(request.form.get("height").replace(",", ".")) / 100,
                    "Weight": float(request.form.get("weight")),
                    "FamHist": request.form.get("fam"),
                    "FAVC": request.form.get("favc"),
                    "FCVC": request.form.get("fcvc"),
                    "NCP": request.form.get("ncp"),
                    "CAEC": request.form.get("caec"),
                    "SMOKE": request.form.get("smoke"),
                    "CH2O": request.form.get("ch2o"),
                    "SCC": request.form.get("scc"),
                    "FAF": request.form.get("faf"),
                    "TUE": request.form.get("tue"),
                    "CALC": request.form.get("calc"),
                    "MTRANS": request.form.get("mtrans"),
                }
            }

            # Preprocess categorical features into numeric values
            # Example: Replace strings with corresponding numeric encodings
            sample["data"]["Gender"] = 1 if sample["data"]["Gender"] == "Male" else 0
            sample["data"]["FamHist"] = 1 if sample["data"]["FamHist"] == "yes" else 0
            sample["data"]["FAVC"] = 1 if sample["data"]["FAVC"] == "yes" else 0
            sample["data"]["CAEC"] = {"no": 0, "Sometimes": 1, "Frequently": 2, "Always": 3}.get(sample["data"]["CAEC"], 0)
            sample["data"]["SMOKE"] = 1 if sample["data"]["SMOKE"] == "yes" else 0
            sample["data"]["SCC"] = 1 if sample["data"]["SCC"] == "yes" else 0
            sample["data"]["MTRANS"] = {
                "Public_Transportation": 0,
                "Automobile": 1,
                "Motorbike": 2,
                "Bike": 3,
                "Walking": 4,
            }.get(sample["data"]["MTRANS"], 0)

            # Convert other fields to numeric values
            sample["data"]["FCVC"] = float(sample["data"]["FCVC"])
            sample["data"]["NCP"] = float(sample["data"]["NCP"])
            sample["data"]["CH2O"] = float(sample["data"]["CH2O"])
            sample["data"]["FAF"] = float(sample["data"]["FAF"])
            sample["data"]["TUE"] = float(sample["data"]["TUE"])
            sample["data"]["CALC"] = {"no": 0, "Sometimes": 1, "Frequently": 2, "Always": 3}.get(sample["data"]["CALC"], 0)

            # Make prediction
            prediction = predict_sample(sample)
            return render_template("prediction.html", prediction_class=prediction["class"], title="Prediction Result")
        except Exception as e:
            return render_template("prediction.html", prediction_class=f"Error: {str(e)}", title="Prediction Error")
    return render_template("prediction.html", prediction_class="No Prediction Yet", title="Predictions")

@app.route('/metrics')
def metrics():
    return render_template("metrics.html", title="Model Evaluation Metrics")

@app.route('/flowchart')
def flowchart():
    return render_template("flowchart.html", title="Project Flowchart")

@app.route("/prediction_api", methods=["POST"])
def predict_api():
    data = request.json
    try:
        sample = {"data": data}
        result_data = predict_sample(sample)
        return jsonify(result_data)
    except Exception as e:
        return jsonify({"error": str(e)})

# Run the app
if __name__ == "__main__":
    app.run(debug=True)
