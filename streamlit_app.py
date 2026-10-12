
import streamlit as st
import joblib
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="Student Pass Predictor")

st.title("Student Pass Predictor")
st.write("Predict a student's result based on study hours.")

model = joblib.load("student_pass_model.joblib")

student_name = st.text_input("Student Name")
study_hours = st.number_input(
    "Study Hours",
    min_value=0.0,
    max_value=24.0,
    value=4.0,
    step=0.5
)

if st.button("Predict Result"):
    if student_name.strip():
        prediction = model.predict([[study_hours]])[0]

        st.subheader("Prediction Result")
        st.write(f"Student Name: {student_name}")
        st.write(f"Study Hours: {study_hours}")
        st.success(f"Predicted Result: {prediction}")

        st.info(
            "This prediction is based on study hours only. "
            "It is an estimate, not a guarantee of actual performance."
        )
    else:
        st.warning("Please enter the student's name.")

st.subheader("Study Hours Comparison")

hours_list = list(range(1, 9))
predictions = [
    model.predict([[hours]])[0] for hours in hours_list
]

chart_data = pd.DataFrame({
    "Study Hours": hours_list,
    "Predicted Result": predictions
})

chart_data["Pass (1) / Fail (0)"] = (
    chart_data["Predicted Result"] == "Pass"
).astype(int)

st.bar_chart(
    chart_data,
    x="Study Hours",
    y="Pass (1) / Fail (0)"
)

st.caption("0 = Fail, 1 = Pass")
