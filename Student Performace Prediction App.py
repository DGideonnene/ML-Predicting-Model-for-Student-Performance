import numpy as np
import pandas as pd
import joblib
import streamlit as st
import matplotlib.pyplot as plt

# Load Model
# model = joblib.load('C:/Users/chiom/Documents/Axia Africa Store/web scraping/new_student_performance_model.pkl')
model = joblib.load("new_student_performance_model.pkl")

# Streamlit UI Customization
st.markdown(
    """
    <style>
    body {
        background-color: #000000;
        color: #E32636;
    }
    .stTextInput, .stButton>button {
        background-color: #E32636 !important;
        color: white !important;
        border-radius: 5px;
    }
    .stMarkdown {
        color: white !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Prediction Function
def performance_prediction(input_data):
    # Define column names expected by the model
    column_names = [
        'absence_days', 'weekly_self_study_hours', 'math_score',
        'history_score', 'physics_score', 'chemistry_score',
        'biology_score', 'english_score', 'geography_score'
    ]
    
    # Convert input to NumPy array and reshape correctly
    input_data = np.array(input_data).reshape(1, -1)
    
    # Convert array directly to DataFrame
    input_df = pd.DataFrame(input_data, columns=column_names)
    
    # Make prediction
    pred = model.predict(input_df)
    prediction = np.round(pred, 2)
    
    if prediction[0] > 70:
        return f"🌟 Excellent Performance: {prediction[0]}"
    elif 50 < prediction[0] <= 70:
        return f"✅ Good Performance: {prediction[0]}"
    else:
        return f"⚠️ Poor Performance: {prediction[0]}"

# Streamlit UI
def main():
    st.title("📊 Student Performance Prediction")
    st.markdown("**Enter the following details to predict the student's performance:**")

    # Input Fields
    absence_days = st.number_input('Number of Absence Days', min_value=0, step=1)
    weekly_self_study_hours = st.number_input('Weekly Self-Study Hours', min_value=0.0, step=0.1)
    math_score = st.number_input('Math Score', min_value=0.0, max_value=100.0, step=0.1)
    history_score = st.number_input('History Score', min_value=0.0, max_value=100.0, step=0.1)
    physics_score = st.number_input('Physics Score', min_value=0.0, max_value=100.0, step=0.1)
    chemistry_score = st.number_input('Chemistry Score', min_value=0.0, max_value=100.0, step=0.1)
    biology_score = st.number_input('Biology Score', min_value=0.0, max_value=100.0, step=0.1)
    english_score = st.number_input('English Score', min_value=0.0, max_value=100.0, step=0.1)
    geography_score = st.number_input('Geography Score', min_value=0.0, max_value=100.0, step=0.1)

    prediction = ""

    # Button for Prediction
    if st.button("📈 Predict Performance"):
        input_data = [absence_days, weekly_self_study_hours, math_score,
                      history_score, physics_score, chemistry_score,
                      biology_score, english_score, geography_score]
        prediction = performance_prediction(input_data)
        
    st.success(prediction)

    # Visualization
    st.subheader("📊 Subject Score Distribution")
    subjects = ['Math', 'History', 'Physics', 'Chemistry', 'Biology', 'English', 'Geography']
    scores = [math_score, history_score, physics_score, chemistry_score, biology_score, english_score, geography_score]
    
    fig, ax = plt.subplots()
    ax.bar(subjects, scores, color='#E32636')
    ax.set_ylabel("Score")
    ax.set_xlabel("Subjects")
    ax.set_title("Student's Subject Score Overview")
    st.pyplot(fig)

if __name__ == '__main__':
    main()





    
    
    
    
    
