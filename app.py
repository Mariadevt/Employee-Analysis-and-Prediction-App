#Import the neessary libraries

import streamlit as st 
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV




#Load the Excel file into a DataFrame

df = pd.read_excel("Employee_Dataset.xlsx")
#Add the title of our app

st.title("Employee Analysis Prediction App")

#Add an image to our dataset

st.image("Employee.webp")

#Add a header

st.header("Dataset Concept", divider="orange")

#Add a paragraph

st.write("""
         Our Employee Analysis Prediction App offers insights into employee performance based on a comprehensive dataset. 
         By leveraging machine learning algorithms, such as RandomForestClassifier, we provide predictive analysis on key performance metrics. Users can interact with the app to explore trends,
         identify patterns, and make informed decisions to enhance employee productivity and satisfaction. 
         The app's intuitive interface, powered by Streamlit and Visual Studio Code, ensures a seamless user experience, enabling HR professionals and managers to optimize their workforce strategies""")

#Display the header

st.header("Explanatory Data Analysis(EDA)", divider="orange")

if st.checkbox("Dataset Info"):
    st.write("Dataset Information", df.info())
    
if st.checkbox("Number of Rows"):
    st.write("Number of Rows", df.shape[0])
    
if st.checkbox("Number of Columns"):
    st.write("Number of Columns", df.shape[1])
    
if st.checkbox("Column Names"):
    st.write("Column Names", df.columns.tolist())
    
if st.checkbox("Data Types"):
    st.write("Data Types", df.dtypes)
    
if st.checkbox("Missing Values"):
    st.write("Missing Values", df.isnull().sum())
    
if st.checkbox("Statistical Summary"):
    st.write("Statistical Summary", df.describe())

if st.checkbox("Duplicates"):
    st.write("Duplicates", df.duplicated().sum())
    
#Visualisation

st.header("Visualization(VIZ)", divider="orange")

if st.checkbox("Histogram"):
    fig, ax = plt.subplots(figsize=(12,6))
    sns.histplot(df["Age"], kde=True, ax=ax)
    ax.set_title("Age Histogram")
    ax.set_xlabel("Age")
    ax.set_ylabel("Count")
    st.pyplot(fig)
     
if st.checkbox("BoxPlot"):
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.boxplot(x="EmpJobSatisfaction", y="DistanceFromHome", data=df, ax=ax)
    ax.set_title("Boxplot of DistanceFromHome by EmpJobSatisfaction")
    ax.set_xlabel("EmpJobSatisfaction")
    ax.set_ylabel("DistanceFromHome")
    st.pyplot(fig)


if st.checkbox("BarChart"):
    plt.figure(figsize=(12, 6))
    sns.countplot(x="EmpJobSatisfaction", data=df)
    plt.title("BarChart of JobSatisfaction")
    plt.xlabel("JobSatisfaction")
    plt.ylabel("Count")
    st.pyplot(plt.gcf())
    
    
if st.checkbox("Scatter Plot"):
    plt.figure(figsize=(12, 6))
    sns.scatterplot(x="ExperienceYearsAtThisCompany", y="TotalWorkExperienceInYears", data=df)
    plt.title("Scatter Plot of ExperienceYearsAtThisCompany vs. TotalWorkExperienceInYears")
    plt.xlabel("ExperienceYearsAtThisCompany")
    plt.ylabel("TotalWorkExperienceInYears")
    st.pyplot(plt.gcf())
    

# Encode categorical columns using LabelEncoder
label_columns = ['Gender', 'EducationBackground', 'MaritalStatus', 'EmpDepartment', 'EmpJobRole', 'BusinessTravelFrequency', 'OverTime', 'Attrition']
label_encoders = {}

def encode_labels(df, columns):
    for column in label_columns:
        label_encoders[column] = LabelEncoder()
        df[column] = label_encoders[column].fit_transform(df[column])
    return df

df_encoded = encode_labels(df.copy(), label_columns)

# Select the relevant features from the encoded DataFrame
X = df_encoded.drop(['PerformanceRating', 'EmpNumber'], axis=1)
y = df_encoded['PerformanceRating']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Train the RandomForestClassifier model
clf = RandomForestClassifier(n_estimators=100, max_depth=20, random_state=42)
clf.fit(X_train, y_train)

# Define a function to encode categorical columns
def encode_categorical(df):
    label_encoders = {}
    for column in df.columns:
        if df[column].dtype == 'object':
            label_encoders[column] = LabelEncoder()
            df[column] = label_encoders[column].fit_transform(df[column])
    return df, label_encoders


# Make predictions
y_pred = clf.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

# Display the model evaluation metrics
st.header("Model Evaluation", divider="orange")
st.write(f"Accuracy: {accuracy}")


# Create a function to predict performance rating based on user input
def predict_performance_rating(user_input):
    user_input_encoded = pd.DataFrame(user_input, index=[0])
    for column, encoder in label_encoders.items():
        user_input_encoded[column] = encoder.transform([user_input_encoded[column].iloc[0]])[0]
    prediction = clf.predict(user_input_encoded)
    return prediction[0]


# Add an interface for user input
st.sidebar.header("Enter Prediction Values:", divider="orange")

user_input = {}
for column in X.columns:
    user_input[column] = st.sidebar.selectbox(f"Select {column}:", df[column].unique())

if st.sidebar.button("Predict Performance Rating"):
    prediction = predict_performance_rating(user_input)
    st.sidebar.write("Predicted Performance Rating:", prediction)

