import streamlit as st
import pandas as pd
import joblib

# Set Page Config with a professional style
st.set_page_config(
    page_title="Employee Attrition Predictor Pro",
    page_icon="💼",
    layout="wide"
)

# Custom CSS for high-quality, professional aesthetics
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    
    .main-title {
        font-size: 2.8rem;
        font-weight: 700;
        background: linear-gradient(90deg, #FF4B4B, #FF8E53);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    
    .sub-title {
        font-size: 1.1rem;
        color: #7f8c8d;
        margin-bottom: 2rem;
    }
    
    /* Result Cards */
    .result-card-low {
        background: linear-gradient(135deg, rgba(46, 204, 113, 0.15) 0%, rgba(39, 174, 96, 0.15) 100%);
        border: 2px solid #2ecc71;
        border-radius: 12px;
        padding: 30px;
        text-align: center;
        margin-top: 25px;
        box-shadow: 0 4px 12px rgba(46, 204, 113, 0.1);
    }
    .result-card-high {
        background: linear-gradient(135deg, rgba(231, 76, 60, 0.15) 0%, rgba(192, 41, 43, 0.15) 100%);
        border: 2px solid #e74c3c;
        border-radius: 12px;
        padding: 30px;
        text-align: center;
        margin-top: 25px;
        box-shadow: 0 4px 12px rgba(231, 76, 60, 0.1);
    }
    .result-text-big {
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 12px;
    }
    .result-text-sub {
        font-size: 1.1rem;
        color: #2c3e50;
        font-weight: 500;
    }
    
    /* Tabs Custom Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 12px;
        border-bottom: 2px solid #f1f2f6;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: #f8f9fa;
        border-radius: 8px 8px 0px 0px;
        padding: 12px 24px;
        font-weight: 600;
        color: #4b6584;
        transition: all 0.2s ease;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(90deg, #FF4B4B, #FF8E53) !important;
        color: white !important;
    }
    
    /* Predict Button */
    div.stButton > button:first-child {
        background: linear-gradient(90deg, #FF4B4B, #FF8E53);
        color: white;
        font-size: 1.3rem;
        font-weight: 600;
        padding: 14px 40px;
        border-radius: 8px;
        border: none;
        box-shadow: 0 4px 15px rgba(255, 75, 75, 0.3);
        transition: all 0.3s ease;
        width: 100%;
        margin-top: 20px;
        cursor: pointer;
    }
    div.stButton > button:first-child:hover {
        background: linear-gradient(90deg, #FF3333, #FF7E33);
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(255, 75, 75, 0.4);
    }
</style>
""", unsafe_allow_html=True)

# Layout Title
st.markdown('<div class="main-title">Employee Attrition Prediction Pro</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">An intelligent HR analytics tool to predict employee attrition risk.</div>', unsafe_allow_html=True)

# Sidebar for Model Settings
with st.sidebar:
    st.image("https://img.icons8.com/color/144/user-group.png", width=90)
    st.markdown("### Model Settings")
    model_choice = st.selectbox(
        "Prediction Engine",
        ["XGBoost (High Accuracy)", "Random Forest"]
    )
    
    st.markdown("---")
    st.markdown("### Dashboard Insights")
    st.write(
        "Fill in the employee's information across the tabs. The application "
        "will encode and scale inputs automatically before performing real-time inference."
    )
    st.markdown("---")
    st.markdown("**Accuracy Reference:**")
    st.markdown("- **XGBoost**: 92.91% Accuracy")
    st.markdown("- **Random Forest**: 91.30% Accuracy")

# Load selected model
if model_choice == "XGBoost (High Accuracy)":
    model = joblib.load("xgboost_attrition.pkl")
else:
    model = joblib.load("random_forest_attrition.pkl")

# Organize inputs into tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "📋 Demographics & Personal", 
    "💼 Job & Profile", 
    "💰 Compensation & Performance", 
    "⏳ Tenure & Experience"
])

# Demographics & Personal
with tab1:
    col1, col2 = st.columns(2)
    with col1:
        age = st.slider("Age", min_value=18, max_value=60, value=35, step=1)
        gender = st.selectbox("Gender", ["Female", "Male"])
        marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
    with col2:
        distance_from_home = st.slider("Distance From Home (miles)", min_value=1, max_value=29, value=10, step=1)
        education = st.selectbox("Education Level", [1, 2, 3, 4, 5], index=2, 
                                 format_func=lambda x: {1: "1 - Below College", 2: "2 - College", 3: "3 - Bachelor", 4: "4 - Master", 5: "5 - Doctor"}[x])
        education_field = st.selectbox("Education Field", ["Life Sciences", "Medical", "Marketing", "Technical Degree", "Human Resources", "Other"])

# Job & Profile
with tab2:
    col1, col2 = st.columns(2)
    with col1:
        department = st.selectbox("Department", ["Research & Development", "Sales", "Human Resources"])
        job_role = st.selectbox("Job Role", [
            "Sales Executive", "Research Scientist", "Laboratory Technician", 
            "Manufacturing Director", "Healthcare Representative", "Manager", 
            "Sales Representative", "Research Director", "Human Resources"
        ])
        job_level = st.slider("Job Level", min_value=1, max_value=5, value=2, step=1)
        business_travel = st.selectbox("Business Travel Frequency", ["Non-Travel", "Travel Rarely", "Travel Frequently"], index=1)
    with col2:
        overtime = st.selectbox("Overtime Required", ["No", "Yes"])
        job_involvement = st.selectbox("Job Involvement Level", [1, 2, 3, 4], index=2,
                                      format_func=lambda x: {1: "1 - Low", 2: "2 - Medium", 3: "3 - High", 4: "4 - Very High"}[x])
        env_satisfaction = st.selectbox("Environment Satisfaction", [1, 2, 3, 4], index=2,
                                       format_func=lambda x: {1: "1 - Low", 2: "2 - Medium", 3: "3 - High", 4: "4 - Very High"}[x])
        job_satisfaction = st.selectbox("Job Satisfaction", [1, 2, 3, 4], index=2,
                                       format_func=lambda x: {1: "1 - Low", 2: "2 - Medium", 3: "3 - High", 4: "4 - Very High"}[x])

# Compensation & Performance
with tab3:
    col1, col2 = st.columns(2)
    with col1:
        monthly_income = st.number_input("Monthly Income ($)", min_value=1000, max_value=20000, value=5700, step=100)
        daily_rate = st.number_input("Daily Rate ($)", min_value=100, max_value=1500, value=780, step=10)
        hourly_rate = st.number_input("Hourly Rate ($)", min_value=30, max_value=100, value=66, step=1)
    with col2:
        monthly_rate = st.number_input("Monthly Rate ($)", min_value=2000, max_value=27000, value=14300, step=100)
        percent_salary_hike = st.slider("Percent Salary Hike (%)", min_value=11, max_value=25, value=15, step=1)
        performance_rating = st.selectbox("Performance Rating", [3, 4], index=0,
                                          format_func=lambda x: {3: "3 - Excellent", 4: "4 - Outstanding"}[x])
        stock_option_level = st.selectbox("Stock Option Level", [0, 1, 2, 3], index=1)

# Tenure & Experience
with tab4:
    col1, col2 = st.columns(2)
    with col1:
        total_working_years = st.slider("Total Working Years", min_value=0, max_value=40, value=10, step=1)
        years_at_company = st.slider("Years At Company", min_value=0, max_value=40, value=6, step=1)
        years_in_current_role = st.slider("Years In Current Role", min_value=0, max_value=18, value=4, step=1)
        years_since_last_promotion = st.slider("Years Since Last Promotion", min_value=0, max_value=15, value=2, step=1)
    with col2:
        years_with_curr_manager = st.slider("Years With Current Manager", min_value=0, max_value=17, value=3, step=1)
        num_companies_worked = st.slider("Number of Companies Worked", min_value=0, max_value=9, value=3, step=1)
        training_times_last_year = st.slider("Training Times Last Year", min_value=0, max_value=6, value=3, step=1)
        relationship_satisfaction = st.selectbox("Relationship Satisfaction", [1, 2, 3, 4], index=2,
                                                 format_func=lambda x: {1: "1 - Low", 2: "2 - Medium", 3: "3 - High", 4: "4 - Very High"}[x])
        work_life_balance = st.selectbox("Work Life Balance Rating", [1, 2, 3, 4], index=2,
                                         format_func=lambda x: {1: "1 - Bad", 2: "2 - Good", 3: "3 - Better", 4: "4 - Best"}[x])

# Construct the input DataFrame mapping to all 44 expected features
input_dict = {
    'Age': [age],
    'DailyRate': [daily_rate],
    'DistanceFromHome': [distance_from_home],
    'Education': [education],
    'EnvironmentSatisfaction': [env_satisfaction],
    'HourlyRate': [hourly_rate],
    'JobInvolvement': [job_involvement],
    'JobLevel': [job_level],
    'JobSatisfaction': [job_satisfaction],
    'MonthlyIncome': [monthly_income],
    'MonthlyRate': [monthly_rate],
    'NumCompaniesWorked': [num_companies_worked],
    'PercentSalaryHike': [percent_salary_hike],
    'PerformanceRating': [performance_rating],
    'RelationshipSatisfaction': [relationship_satisfaction],
    'StockOptionLevel': [stock_option_level],
    'TotalWorkingYears': [total_working_years],
    'TrainingTimesLastYear': [training_times_last_year],
    'WorkLifeBalance': [work_life_balance],
    'YearsAtCompany': [years_at_company],
    'YearsInCurrentRole': [years_in_current_role],
    'YearsSinceLastPromotion': [years_since_last_promotion],
    'YearsWithCurrManager': [years_with_curr_manager],
    
    # Categorical one-hot encoded variables
    'BusinessTravel_Travel_Frequently': [1 if business_travel == "Travel Frequently" else 0],
    'BusinessTravel_Travel_Rarely': [1 if business_travel == "Travel Rarely" else 0],
    'Department_Research & Development': [1 if department == "Research & Development" else 0],
    'Department_Sales': [1 if department == "Sales" else 0],
    'EducationField_Life Sciences': [1 if education_field == "Life Sciences" else 0],
    'EducationField_Marketing': [1 if education_field == "Marketing" else 0],
    'EducationField_Medical': [1 if education_field == "Medical" else 0],
    'EducationField_Other': [1 if education_field == "Other" else 0],
    'EducationField_Technical Degree': [1 if education_field == "Technical Degree" else 0],
    'Gender_Male': [1 if gender == "Male" else 0],
    'JobRole_Human Resources': [1 if job_role == "Human Resources" else 0],
    'JobRole_Laboratory Technician': [1 if job_role == "Laboratory Technician" else 0],
    'JobRole_Manager': [1 if job_role == "Manager" else 0],
    'JobRole_Manufacturing Director': [1 if job_role == "Manufacturing Director" else 0],
    'JobRole_Research Director': [1 if job_role == "Research Director" else 0],
    'JobRole_Research Scientist': [1 if job_role == "Research Scientist" else 0],
    'JobRole_Sales Executive': [1 if job_role == "Sales Executive" else 0],
    'JobRole_Sales Representative': [1 if job_role == "Sales Representative" else 0],
    'MaritalStatus_Married': [1 if marital_status == "Married" else 0],
    'MaritalStatus_Single': [1 if marital_status == "Single" else 0],
    'OverTime_Yes': [1 if overtime == "Yes" else 0]
}

input_data = pd.DataFrame(input_dict)

# Prediction execution
if st.button("Predict Attrition"):
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    if prediction == 1:
        st.markdown(f"""
        <div class="result-card-high">
            <div class="result-text-big">⚠️ High Attrition Risk</div>
            <div class="result-text-sub">Probability of leaving: {probability:.2%}</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="result-card-low">
            <div class="result-text-big">✅ Low Attrition Risk</div>
            <div class="result-text-sub">Probability of leaving: {probability:.2%}</div>
        </div>
        """, unsafe_allow_html=True)