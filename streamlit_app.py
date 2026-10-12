
import streamlit as st
import joblib
import pandas as pd

# Load the saved model
model = joblib.load("student_pass_model.joblib")

st.title("Student Pass Prediction System")
st.write("Predict a student's result based on study hours.")

# Student name input
student_name = st.text_input("Student Name")

# Study hours input
study_hours = st.number_input(
    "Study Hours",
    min_value=0.0,
    max_value=24.0,
    value=2.0,
    step=0.5
)

# Prediction button
if st.button("Predict Result"):
    if not student_name.strip():
        st.warning("Please enter the student's name.")
    else:
        prediction = model.predict([[study_hours]])[0]

        st.subheader("Prediction Result")
        st.write(f"Student Name: {student_name}")
        st.write(f"Study Hours: {study_hours}")

        if prediction == "Pass":
            st.success(f"Predicted Class: {prediction}")
        else:
            st.error(f"Predicted Class: {prediction}")

        st.write(
            "Explanation: The system uses a previously trained "
            "machine learning model to predict the result based "
            "on the student's study hours."
        )

        # Bonus chart
        chart_data = pd.DataFrame({
            "Study Hours": [study_hours],
            "Predicted Class": [str(prediction)]
        })

        st.subheader("Prediction Chart")
        st.bar_chart(
            chart_data,
            x="Predicted Class",
            y="Study Hours"
        )
