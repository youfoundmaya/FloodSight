# FloodSight

## **Problem Statement**

- Timely and accurate flood risk assessment is critical for saving lives and minimizing property damage. Existing systems often rely on static models or delayed reporting, failing to provide the dynamic, hyper-localized information needed for rapid decision-making.
- The FloodSight project addresses this by developing a machine learning-powered system that ingests real-time environmental data (rainfall, river levels, terrain, weather forecasts) to classify flood risk severity. The core objective is to deliver immediate, actionable intelligence through a responsive dashboard featuring three distinct severity levels:

- 🟢 Safe: No immediate risk detected.
- 🟡 Warning: Elevated risk; monitor conditions closely.
- 🔴 Evacuation: Imminent danger; immediate action required.

- Crucially, the system integrates with the Google Maps API to provide users in "Warning" or "Evacuation" zones with safe, navigable route suggestions to designated safe zones, bypassing flooded or blocked areas in real-time. This combination of predictive classification and dynamic routing transforms raw data into life-saving operational capacity.

## **Team Members & Tech Stack**
Name                           Focus              
Mrunmayee Ovhal           Data Integration, Map Visualization, Streamlit UI, Weather API
Karan Saun                Model Training, Classification Logic
Prachita Jadhav           Application Logic, UI, Model Training  


## **Technology Stack**
Category                                      Technology                                                                                Purpose                                            
1. Application Framework                      Streamlit                                         Building the entire interactive web dashboard and alert system purely in Python for rapid development.
2. Core Logic                                 Python                                            Unified language for all logic, data handling, and ML model serving.
3. Development Environment                    VS Code                                           Primary IDE used for development, debugging, and execution.
4. Machine Learning                           Python (Pandas, scikit-learn/TensorFlow)          Data cleaning, feature engineering, and training the flood risk classification model.

