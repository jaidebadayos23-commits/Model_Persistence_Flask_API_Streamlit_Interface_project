
from sklearn.tree import DecisionTreeClassifier
import joblib

# Sample student performance data
# Study hours: 1, 2, 3, 4, 5, 6, 7, 8
X = [[1], [2], [3], [4], [5], [6], [7], [8]]
y = [
    "Fail", "Fail", "Fail", "Pass",
    "Pass", "Pass", "Pass", "Pass"
]

# Train the model
model = DecisionTreeClassifier(random_state=42)
model.fit(X, y)

# Save the trained model
joblib.dump(model, "student_pass_model.joblib")
print("Model saved successfully!")

# Load the saved model
loaded_model = joblib.load("student_pass_model.joblib")
print("Model loaded successfully!")

# Make three predictions

for hours in [2, 5, 8]:
    prediction = loaded_model.predict([[hours]])[0]
    print(f"Study Hours: {hours}, Prediction: {prediction}")

