import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# This code use a single spilit instead of cross validation, can be used to compare in report



# Load the dataset
# Replace 'your_dataset.csv' with the path to your actual dataset
data = pd.read_csv(r'C:\Users\amroh\Desktop\Ahmed\AAI\AAI-assignment\Datasets and old trials\dementia_data-MRI-features.csv')

# Check column names to ensure 'Gender' is correct
print("Columns in the dataset:", data.columns)

# If 'Gender' is found as 'M/F' instead, rename it
if 'M/F' in data.columns:
    data = data.rename(columns={'M/F': 'Gender'})

# Handle missing values (e.g., using the median for continuous columns)
data['MMSE'].fillna(data['MMSE'].mean(), inplace=True)
data['SES'].fillna(data['SES'].mode()[0], inplace=True)

# Handle the 'Group' feature to have only 'Demented' and 'Nondemented'
data['Group'] = data['Group'].apply(lambda x: 'Demented' if x == 'Demented' else 'Nondemented')

# Encode categorical variables like 'Gender' and 'Group'
gender_encoder = LabelEncoder()
data['Gender'] = gender_encoder.fit_transform(data['Gender'])  # Transform 'M' and 'F' to numeric values

group_encoder = LabelEncoder()
data['Group'] = group_encoder.fit_transform(data['Group'])  # Transform 'Demented' and 'Nondemented' to numeric values

# Define the features and target variable
X = data[['Visit', 'Gender', 'Age', 'EDUC', 'SES', 'MMSE', 'CDR', 'eTIV', 'nWBV', 'ASF']]  # Features
y = data['Group']  # Target variable

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the model (Random Forest Classifier)
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

# Predict for a specific query
sample_query = pd.DataFrame({
    'Visit': [1],
    'Gender': gender_encoder.transform(['M']),  # Ensure using the same encoder here
    'Age': [75],
    'EDUC': [12],
    'SES': [2],
    'MMSE': [23],
    'CDR': [0.5],
    'eTIV': [1678],
    'nWBV': [0.736],
    'ASF': [1.046]
})

# Make prediction for the sample query
query_prediction = model.predict(sample_query)
predicted_group = group_encoder.inverse_transform(query_prediction)
print(f"Predicted group for the sample query: {predicted_group[0]}")
