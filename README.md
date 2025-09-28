# FloodSight: A flood risk classifier

## **Problem Statement**

- Timely and accurate flood risk assessment is critical for everyday commuters to recognize if the route is safe or not.
- Existing systems often rely on static models or delayed reporting, failing to provide the dynamic, hyper-localized information needed for rapid decision-making.
- The FloodSight project addresses this by developing a machine learning-powered system that ingests real-time environmental data (rainfall, tide levels, weather forecasts) to classify flood risk severity.
- The core objective is to deliver immediate, actionable intelligence through a responsive dashboard featuring three distinct severity levels:
  - 🟢 Safe: No immediate risk detected.
  - 🟡 Warning: Elevated risk; monitor conditions closely.
  - 🔴 Evacuation: Imminent danger; immediate action required.
- Crucially, the system integrates with the OpenWeathermanAPI to provide users whether the area is safe or not, with the weather forecast and also 3 navigable route suggestion.
- This combination of classification and dynamic routing transforms raw data into life-saving operational capacity.

## **Team Members & Tech Stack**
1. Mrunmayee Ovhal: Data Integration, Map Visualization, Streamlit UI, Weather API
2. Karan Saun: Model Training, Classification Logic
3. Prachita Jadhav: Application Logic, UI, Model Trainging


## **Technology Stack**
1. Application Framework- Streamlit: Building the entire interactive web dashboard and alert system purely in Python for rapid development
2. Core Logic- Python: Unified language for all logic, data handling, and ML model serving.
3. Development Environment- VS Code: Primary IDE used for development, debugging, and execution.
4. Machine Learning- Python (Pandas, scikit-learn): Data cleaning, feature engineering, and training the flood risk classification model (XGBoost).

## **Screenshots**
## **1. Flood Risk Analysis and Weather:**
<img width="1820" height="899" alt="image" src="https://github.com/user-attachments/assets/da2214b8-09a1-4ec5-af67-c479bc1ff98b" />
<img width="1772" height="931" alt="image" src="https://github.com/user-attachments/assets/82ca864b-bff9-4cb9-88c9-cc21b7074c6e" />
<img width="1612" height="868" alt="image" src="https://github.com/user-attachments/assets/74c4f855-40f7-4e5b-b2f3-3e9ffbca1431" />

## **2. Interactive Route Map**
<img width="1592" height="954" alt="image" src="https://github.com/user-attachments/assets/b874d93b-85a0-4bea-b733-4c52f1bc43db" />
<img width="1676" height="879" alt="image" src="https://github.com/user-attachments/assets/ab809a7e-6d12-4b13-a267-1b8c7a0bca44" />
