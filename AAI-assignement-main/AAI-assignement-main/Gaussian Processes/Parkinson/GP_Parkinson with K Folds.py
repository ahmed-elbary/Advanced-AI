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

# Load and preprocess the dataset
file_path = r'C:\Users\amroh\Desktop\Ahmed\AAI\AAI-assignment\AAI-assignement-main\AAI-assignement-main\Datasets and old trials\parkinsons_data-VOICE-features.csv'
data = pd.read_csv(file_path)

# Split features (X) and target (y)
X = data.drop(columns=['status', 'name'])
y = data['status']

# Encode the target variable 'status' if it's categorical (0 for healthy, 1 for Parkinson's)
status_encoder = LabelEncoder()
y_encoded = status_encoder.fit_transform(y)

# Define cross-validation strategy
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Initialize model (Random Forest Classifier)
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Start measuring time
start_time = time.time()

# Perform cross-validation and get predictions
predictions = cross_val_predict(model, X, y_encoded, cv=cv)

# End measuring time
cv_time = time.time() - start_time

# Define query (based on the feature names from the dataset)
query = {
    'MDVP:Fo(Hz)': 140.992,
    'MDVP:Fhi(Hz)': 157.302,
    'MDVP:Flo(Hz)': 74.997,
    'MDVP:Jitter(%)': 0.00784,
    'MDVP:Jitter(Abs)': 0.00007,
    'MDVP:RAP': 0.0037,
    'MDVP:PPQ': 0.00554,
    'Jitter:DDP': 0.01109,
    'MDVP:Shimmer': 0.04374,
    'MDVP:Shimmer(dB)': 0.426,
    'Shimmer:APQ3': 0.02182,
    'Shimmer:APQ5': 0.0313,
    'MDVP:APQ': 0.02971,
    'Shimmer:DDA': 0.06545,
    'NHR': 0.02211,
    'HNR': 18.033,
    'RPDE': 0.414783,
    'DFA': 0.815285,
    'spread1': -4.813031,
    'spread2': 0.266482,
    'D2': 2.301442,
    'PPE': 0.284654
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

# Evaluate performance metrics
accuracy = np.mean(predictions == y_encoded)
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
print(f"Classification Accuracy (CV): {accuracy:.4f}")
print(f"AUC (Area Under the Curve) (CV): {auc_score:.4f}")
print(f"Brier Score (CV): {brier:.4f}")
print(f"Kullback-Leibler Divergence (CV): {kl_divergence:.4f}")
print(f"Cross-validation Time (seconds): {cv_time:.4f}")

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
train_std = train_scores.std(axis=1)
test_mean = test_scores.mean(axis=1)
test_std = test_scores.std(axis=1)

plt.figure(figsize=(10, 6))
plt.title("Learning Curve")
plt.plot(train_sizes, train_mean, label="Training score", color="r")
plt.plot(train_sizes, test_mean, label="Test score", color="g")
plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, color="r", alpha=0.2)
plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, color="g", alpha=0.2)
plt.xlabel("Training Set Size")
plt.ylabel("Accuracy")
plt.legend(loc="best")
plt.show()
