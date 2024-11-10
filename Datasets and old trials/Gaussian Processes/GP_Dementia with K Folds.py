import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Using cross validation with K = 5 that can me changed , 
# Plotting the learning curve and the features impoortance.

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

# Initialize the model (Random Forest Classifier)
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Use cross-validation to evaluate the model's performance
cv_scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')  # 5-fold cross-validation

# Print cross-validation accuracy scores for each fold
print(f"Cross-validation accuracy scores: {cv_scores}")
print(f"Average cross-validation accuracy: {cv_scores.mean():.4f}")

# Feature Importance Plot
model.fit(X, y)  # Fit the model to the entire dataset to extract feature importances
importances = model.feature_importances_
indices = importances.argsort()

plt.figure(figsize=(10, 6))
plt.title("Feature Importances")
plt.barh(range(len(indices)), importances[indices], align="center")
plt.yticks(range(len(indices)), [X.columns[i] for i in indices])
plt.xlabel("Relative Importance")
plt.show()

# Learning Curve Plot (using training set sizes)
from sklearn.model_selection import learning_curve

train_sizes, train_scores, test_scores = learning_curve(model, X, y, cv=5, scoring='accuracy', n_jobs=-1)

# Calculate mean and std deviation of the training and testing scores
train_mean = train_scores.mean(axis=1)
train_std = train_scores.std(axis=1)
test_mean = test_scores.mean(axis=1)
test_std = test_scores.std(axis=1)

# Plot learning curve
plt.figure(figsize=(10, 6))
plt.title("Learning Curve")
plt.plot(train_sizes, train_mean, label="Training score", color="r")
plt.plot(train_sizes, test_mean, label="Test score", color="g")

# Show the shaded region for standard deviation
plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, color="r", alpha=0.2)
plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, color="g", alpha=0.2)

plt.xlabel("Training Set Size")
plt.ylabel("Accuracy")
plt.legend(loc="best")
plt.show()

# Now, you can make predictions with a sample query
sample_query = pd.DataFrame({
    'Visit': [3],
    'Gender': gender_encoder.transform(['M']),  # Ensure using the same encoder here
    'Age': [80],
    'EDUC': [12],
    'SES': [2],
    'MMSE': [22],
    'CDR': [0.5],
    'eTIV': [1698],
    'nWBV': [0.701],
    'ASF': [1.034]
})

# Make prediction for the sample query
model.fit(X, y)  # Fit the model on the entire dataset again to make a prediction
query_prediction = model.predict(sample_query)

# Inverse transform the predicted group to get original labels
predicted_group = group_encoder.inverse_transform(query_prediction)
print(f"Predicted group for the sample query: {predicted_group[0]}")

# Optionally, if you want to calculate the probability for each class (e.g., 'Demented' or 'Nondemented')
predicted_proba = model.predict_proba(sample_query)
print(f"Prediction probabilities: {predicted_proba[0]}")  # Probabilities for both classes

