
import joblib
from sklearn.linear_model import LogisticRegression

# Training data
study_hours = [[1], [2], [3], [4], [5], [6], [7], [8]]

results = [
    "Fail", "Fail", "Fail", "Pass",
    "Pass", "Pass", "Pass", "Pass"
]

# Train the model
model = LogisticRegression()
model.fit(study_hours, results)

# Save the model
joblib.dump(model, "student_pass_model.joblib")
print("Model saved successfully!")

# Load the saved model
loaded_model = joblib.load("student_pass_model.joblib")

# Make three predictions
test_hours = [[2], [4], [7]]

for sample in test_hours:
    prediction = loaded_model.predict([sample])
    print(f"Study Hours: {sample[0]}, Prediction: {prediction[0]}")

