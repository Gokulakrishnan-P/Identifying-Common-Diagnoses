# Import necessary libraries
import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Load data from SQL file
def load_data(sql_script_path):
    # Create SQLite connection
    conn = sqlite3.connect(':memory:')
    cursor = conn.cursor()
    
    # Load SQL script
    with open(sql_script_path, 'r') as file:
        sql_script = file.read()

    # Remove unnecessary SQL commands for SQLite
    sql_script = sql_script.replace('create database identify;', '').replace('use identify;', '')

    # Execute script to create tables and insert data
    cursor.executescript(sql_script)
    
    # Load the data into pandas DataFrame
    df = pd.read_sql_query("SELECT * FROM PatientRecords", conn)
    
    return df

# Data preprocessing: Encoding categorical variables
def preprocess_data(df):
    le_gender = LabelEncoder()
    le_city = LabelEncoder()
    le_insurance = LabelEncoder()
    le_diagnosis = LabelEncoder()

    df['Gender'] = le_gender.fit_transform(df['Gender'])
    df['City'] = le_city.fit_transform(df['City'])
    df['InsuranceProvider'] = le_insurance.fit_transform(df['InsuranceProvider'])
    df['Diagnosis'] = le_diagnosis.fit_transform(df['Diagnosis'])

    X = df[['Gender', 'City', 'InsuranceProvider']]
    y = df['Diagnosis']
    
    return X, y

# Exploratory Data Analysis (EDA)
def perform_eda(df):
    print("Data Head:\n", df.head())
    print("\nData Info:")
    print(df.info())
    print("\nData Description:")
    print(df.describe())

    # Plot the distribution of diagnosis
    plt.figure(figsize=(8, 6))
    sns.countplot(x='Diagnosis', data=df)
    plt.title('Distribution of Diagnoses')
    plt.show()

    # Plot the gender distribution
    plt.figure(figsize=(8, 6))
    sns.countplot(x='Gender', data=df)
    plt.title('Gender Distribution')
    plt.show()

# Split the data into training and testing sets
def split_data(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

# Train Random Forest model
def train_random_forest(X_train, y_train):
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    return model

# Train Logistic Regression model
def train_logistic_regression(X_train, y_train):
    model = LogisticRegression(random_state=42)
    model.fit(X_train, y_train)
    return model

# Evaluate the model
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {accuracy}")
    print("Classification Report:\n", classification_report(y_test, y_pred))

# Save the model
def save_model(model, filename):
    joblib.dump(model, filename)

# Load the model
def load_model(filename):
    return joblib.load(filename)

# Main function to run the entire ML pipeline
def main():
    # Step 1: Load the data from SQL
    sql_script_path = 'identify.sql'  # Update the path if necessary
    df = load_data(sql_script_path)

    # Step 2: Exploratory Data Analysis (EDA)
    perform_eda(df)

    # Step 3: Preprocess the data
    X, y = preprocess_data(df)

    # Step 4: Split the data into training and testing sets
    X_train, X_test, y_train, y_test = split_data(X, y)

    # Step 5: Train Random Forest model
    rf_model = train_random_forest(X_train, y_train)

    # Step 6: Evaluate the Random Forest model
    print("\nRandom Forest Model Evaluation:")
    evaluate_model(rf_model, X_test, y_test)

    # Step 7: Train Logistic Regression model
    log_reg_model = train_logistic_regression(X_train, y_train)

    # Step 8: Evaluate Logistic Regression model
    print("\nLogistic Regression Model Evaluation:")
    evaluate_model(log_reg_model, X_test, y_test)

    # Step 9: Save the best model
    save_model(rf_model, 'random_forest_model.pkl')
    print("Random Forest model saved as 'random_forest_model.pkl'.")

# Run the project
if __name__ == '__main__':
    main()