
from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)

model = joblib.load("student_pass_model.joblib")


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    if not data or "study_hours" not in data or "attendance" not in data:
        return jsonify({
            "error": "Please provide study_hours and attendance"
        }), 400

    try:
        study_hours = float(data["study_hours"])
        attendance = float(data["attendance"])
    except (ValueError, TypeError):
        return jsonify({
            "error": "Inputs must be numbers"
        }), 400

    if study_hours < 0:
        return jsonify({
            "error": "Study hours cannot be negative"
        }), 400

    if not 0 <= attendance <= 100:
        return jsonify({
            "error": "Attendance must be between 0 and 100"
        }), 400

    prediction = model.predict([[study_hours]])[0]

    return jsonify({
        "study_hours": study_hours,
        "attendance": attendance,
        "prediction": prediction
    })


if __name__ == "__main__":
    app.run(debug=True)
