import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, brier_score_loss
from scipy.stats import entropy
import time
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
from sklearn.metrics import roc_curve, auc
###################################################################################################
# Load the dataset
file_path = r'C:\Users\amroh\Desktop\Ahmed\AAI\AAI-assignment\Datasets and old trials\parkinsons_data-VOICE-features.csv'
data = pd.read_csv(file_path)

########################### Data preprocessing #################################################### 
# Split features (X) and target (y)
X = data[['MDVP:Fo(Hz)', 'MDVP:Fhi(Hz)', 'MDVP:Flo(Hz)', 'MDVP:Jitter(%)', 'MDVP:Jitter(Abs)','MDVP:RAP', 'MDVP:PPQ', 'Jitter:DDP', 'MDVP:Shimmer',
          'MDVP:Shimmer(dB)', 'Shimmer:APQ3', 'Shimmer:APQ5', 'MDVP:APQ', 'Shimmer:DDA', 'NHR', 'HNR', 'RPDE', 'DFA', 'spread1', 'spread2', 'D2', 'PPE']]  # Features
y = data['status']

# Encode the target variable 'status' if it's categorical (0 for no Parkinson's, 1 for Parkinson's)
status_encoder = LabelEncoder()
y_encoded = status_encoder.fit_transform(y)

# Define cross-validation strategy
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

################################################################################################
# Initialize model (Random Forest Classifier)
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Start measuring time
start_time = time.time()

# Perform cross-validation and get predictions
predictions = cross_val_predict(model, X, y_encoded, cv=cv)

# End measuring time
cv_time = time.time() - start_time

# Print cross-validation accuracy score
print(f"Average cross-validation accuracy: {predictions.mean():.4f}") # Average accuracy
################## Predictions with a sample query ####################################
# Define query (based on the feature names from the dataset)
query = {
    'MDVP:Fo(Hz)': 120.992,
    'MDVP:Fhi(Hz)': 120.302,
    'MDVP:Flo(Hz)': 70.997,
    'MDVP:Jitter(%)': 0.1,
    'MDVP:Jitter(Abs)': 0.1,
    'MDVP:RAP': 0.1,
    'MDVP:PPQ': 0.2,
    'Jitter:DDP': 0.1,
    'MDVP:Shimmer': 0.2,
    'MDVP:Shimmer(dB)': 0.426,
    'Shimmer:APQ3': 0.1,
    'Shimmer:APQ5': 0.1,
    'MDVP:APQ': 0.1,
    'Shimmer:DDA': 0.06545,
    'NHR': 0.02211,
    'HNR': 21.033,
    'RPDE': 0.414783,
    'DFA': 0.725,
    'spread1': -0.813031,
    'spread2': 0.106482,
    'D2': 2.301442,
    'PPE': 0.16
}

# Convert query to DataFrame
query_df = pd.DataFrame(query, index=[0])

# Identify missing features and fill them with the mean
missing_features = [col for col in X.columns if col not in query_df.columns]
for feature in missing_features:
    query_df[feature] = X[feature].mean()

# Ensure feature order matches the training data
query_df = query_df[X.columns]

# Train the model on the entire dataset
model.fit(X, y_encoded)

# Get prediction probabilities for the query
prediction_probabilities = model.predict_proba(query_df)
print(f"Probability of No Parkinson (status=0): {prediction_probabilities[0][0]:.4f}")
print(f"Probability of Parkinson's (status=1): {prediction_probabilities[0][1]:.4f}")

############################ Evaluation ##############################################
# Evaluate performance metrics
auc_score = roc_auc_score(y_encoded, predictions)
brier = brier_score_loss(y_encoded, model.predict_proba(X)[:, 1])

# KL Divergence
# Get predicted probabilities for all samples in X
predicted_probabilities = model.predict_proba(X)[:, 1]  # Probabilities for 'status=1' (Parkinson's)

# Create a probability distribution for true labels with high probability for the true class
true_probabilities = np.where(y_encoded == 1, 0.999, 0.001)

# Calculate KL Divergence between true labels and predicted probabilities
kl_divergence = entropy(true_probabilities, predicted_probabilities + 1e-10)  # Add epsilon to avoid log(0)

# Print evaluation metrics
print(f"Cross-validation Time (sec): {cv_time:.4f}")
print(f"Area Under the Curve  (AUC): {auc_score:.4f}")
print(f"Kullback-Leibler Divergence: {kl_divergence:.4f}")
print(f"Brier Score (CV): {brier:.4f}")

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

# Learning Curve Plot
train_sizes, train_scores, test_scores = learning_curve(model, X, y, cv=5, scoring='accuracy', n_jobs=-1)
train_mean = train_scores.mean(axis=1)
test_mean = test_scores.mean(axis=1)
plt.figure(figsize=(10, 6))
plt.title("Learning Curve")
plt.plot(train_sizes, train_mean, label="Training score", color="r")
plt.plot(train_sizes, test_mean, label="Test score", color="g")
plt.xlabel("Training Set Size")
plt.ylabel("Accuracy")
plt.legend(loc="best")
plt.show()

