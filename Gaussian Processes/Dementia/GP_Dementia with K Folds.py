import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, brier_score_loss
from scipy.stats import entropy
import time
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
from sklearn.model_selection import cross_val_predict
###################################################################################################
# Loading the dataset
data = pd.read_csv(r'C:\CloudDriveSpace\OneDrive\29385647@students.lincoln.ac.uk\Desktop\AAI-assignment\Datasets and old trials\dementia_data-MRI-features.csv')

########################### Data preprocessing #################################################### 
# If 'Gender' is found as 'M/F' instead, rename it
if 'M/F' in data.columns:
    data = data.rename(columns={'M/F': 'Gender'})
# Check column names to ensure 'Gender' is correct
print("Columns in the dataset:", data.columns)

# Handle missing values 
data['MMSE'].fillna(data['MMSE'].mean(), inplace=True)  # using the mean function to impute missing values in MMSE feature
data['SES'].fillna(data['SES'].mode()[0], inplace=True) # using the mode function to impute missing values in SES feature

# Handle the 'Group' feature to have only 'Demented' and 'Nondemented'
data['Group'] = data['Group'].apply(lambda x: 'Nondemented' if x == 'Nondemented' else 'Demented') # All 'converted' valeus will be replaced with 'Demented' 

# Encode categorical variables like 'Gender' and 'Group'
gender_encoder = LabelEncoder()
data['Gender'] = gender_encoder.fit_transform(data['Gender'])  # Transform 'M' and 'F' to binary values

group_encoder = LabelEncoder()
data['Group'] = group_encoder.fit_transform(data['Group'])  # Transform 'Demented' and 'Nondemented' to binary values

# Define the features and target variable
X = data[['Visit', 'Gender', 'Age', 'EDUC', 'SES', 'MMSE', 'CDR', 'eTIV', 'nWBV', 'ASF']]  # Features
y = data['Group']  # Target variable

################################################################################################
# Initialize the model (Random Forest Classifier)
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Start measuring time
start_time = time.time()

# Use cross-validation to evaluate the model's performance
cv_scores = cross_val_score(model, X, y, cv=4, scoring='accuracy')  # 4-fold cross-validation

# End measuring time
cv_time = time.time() - start_time

# Print cross-validation accuracy score
print(f"Average cross-validation accuracy: {cv_scores.mean():.4f}") # Average accuracy 


################## Predictions with a sample query ####################################
query = pd.DataFrame({
    'Visit': [2],
    'Gender': gender_encoder.transform(['F']),  # Ensure using the same encoder here
    'Age': [88],
    'EDUC': [14],
    'SES': [2],
    'MMSE': [30],
    'CDR': [0],
    'eTIV': [2004],
    'nWBV': [0.681],
    'ASF': [0.876]
})

# Make prediction for the sample query
model.fit(X, y)  # Fit the model on the entire dataset again to make a prediction
query_prediction = model.predict(query)

# Inverse transform the predicted group to get original labels
predicted_group = group_encoder.inverse_transform(query_prediction)
print(f"Predicted group for the sample query: {predicted_group[0]}")

# probability for each class ('Demented' or 'Nondemented')
predicted_proba = model.predict_proba(query)
print(f"Prediction probabilities: {predicted_proba[0]}")  # Probabilities for both classes

############################ Evaluation ##############################################
# Cross-validation performance metrics
# Get cross-validated predictions to calculate AUC and other metrics
cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)
predictions = cross_val_predict(model, X, y, cv=cv, method='predict_proba')

# AUC (Area Under the Curve)
auc_value = roc_auc_score(y, predictions[:, 1])  # Using probabilities for class 1

# Brier Score
brier_score = brier_score_loss(y, predictions[:, 1])

# Kullback-Leibler Divergence
kl_divergence = entropy(y, predictions[:, 1])

# Print the performance metrics
print(f"Cross-validation Time (seconds): {cv_time:.4f}")
print(f"Area Under the Curve  (  AUC  ): {auc_value:.4f}")
print(f"Kullback-Leibler Divergence    : {kl_divergence:.4f}")
print(f"Brier Score : {brier_score:.4f}")

########################### Plotting ##################################################
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
train_sizes, train_scores, test_scores = learning_curve(model, X, y, cv=5, scoring='accuracy', n_jobs=-1)

# Calculate mean and std deviation of the training and testing scores
train_mean = train_scores.mean(axis=1)
# train_std = train_scores.std(axis=1)
test_mean = test_scores.mean(axis=1)
# test_std = test_scores.std(axis=1)

# Plot learning curve
plt.figure(figsize=(10, 6))
plt.title("Learning Curve")
plt.plot(train_sizes, train_mean, label="Training score", color="r")
plt.plot(train_sizes, test_mean, label="Test score", color="g")
plt.xlabel("Training Set Size")
plt.ylabel("Accuracy")
plt.legend(loc="best")
plt.show()
