# %%
import pandas as pd
from pgmpy.models import BayesianNetwork
from pgmpy.estimators import MaximumLikelihoodEstimator, HillClimbSearch, BicScore
from pgmpy.inference import VariableElimination
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss, brier_score_loss
from sklearn.model_selection import train_test_split
import numpy as np
import time

# %%
file_path = r'C:\CloudDriveSpace\OneDrive\29385647@students.lincoln.ac.uk\Desktop\AAI-assignment\Datasets and old trials\dementia_data-MRI-features.csv'  # Replace with your local file path
dementia_df = pd.read_csv(file_path)

# %%
# Drop irrelevant columns
dementia_df = dementia_df.drop(columns=['Subject ID', 'MRI ID', 'MR Delay'], errors='ignore')

# %%
# Ensure Group is binary-encoded
dementia_df['Group'] = dementia_df['Group'].apply(lambda x: 1 if x == 'Demented' else 0)

# %%
# Impute missing values for all columns based on their data type
dementia_df['SES'].fillna(dementia_df['SES'].mode()[0], inplace=True)
dementia_df['MMSE'].fillna(dementia_df['MMSE'].mean(), inplace=True)

# %%
# Discretize continuous features with labels=False
dementia_df['Age'] = pd.cut(dementia_df['Age'], bins=4, labels=False)
dementia_df['MMSE'] = pd.cut(dementia_df['MMSE'], bins=4, labels=False)
dementia_df['eTIV'] = pd.cut(dementia_df['eTIV'], bins=4, labels=False)
dementia_df['nWBV'] = pd.cut(dementia_df['nWBV'], bins=4, labels=False)
dementia_df['ASF'] = pd.cut(dementia_df['ASF'], bins=4, labels=False)

# %%
# Split data into training and testing sets
train_df, test_df = train_test_split(dementia_df, test_size=0.2, stratify=dementia_df['Group'], random_state=42)

# %%
# Hill Climb Search to learn Bayesian Network structure
hc = HillClimbSearch(train_df)
best_model = hc.estimate(scoring_method=BicScore(train_df))

# Define the model with the learned structure
learned_structure = BayesianNetwork(best_model.edges())

# %%
# Fit the model with Maximum Likelihood Estimation (MLE) on the learned structure
start_train_time = time.time()
learned_structure.fit(train_df, estimator=MaximumLikelihoodEstimator)
train_time = time.time() - start_train_time

# Initialize inference on the learned model
learned_inference = VariableElimination(learned_structure)

# Define bins for continuous features
age_bins = pd.cut(dementia_df['Age'], bins=4, retbins=True)[1]
mmse_bins = pd.cut(dementia_df['MMSE'], bins=4, retbins=True)[1]
etiv_bins = pd.cut(dementia_df['eTIV'], bins=4, retbins=True)[1]
nwbv_bins = pd.cut(dementia_df['nWBV'], bins=4, retbins=True)[1]
asf_bins = pd.cut(dementia_df['ASF'], bins=4, retbins=True)[1]

# Helper function to map continuous values to bins
def get_bin(value, bins):
    bin_index = pd.cut([value], bins=bins, labels=False)[0]
    return int(bin_index) if not pd.isna(bin_index) else int(len(bins) - 2 if value > bins[-1] else 0)

# %%
# Model Evaluation
y_true = test_df['Group'].values
y_pred = []
y_proba = []

# Start testing time
start_test_time = time.time()
for _, row in test_df.iterrows():
    # Convert continuous values to bins and prepare evidence
    evidence = {
        #'Visit': row['Visit'],
        'Age': get_bin(row['Age'], age_bins),
        'EDUC': row['EDUC'],
        'SES': row['SES'],
        'MMSE': get_bin(row['MMSE'], mmse_bins),
        'CDR': row['CDR'],
        'eTIV': get_bin(row['eTIV'], etiv_bins),
        'nWBV': get_bin(row['nWBV'], nwbv_bins),
        'ASF': get_bin(row['ASF'], asf_bins)
    }
    
    try:
        query_result = learned_inference.query(variables=['Group'], evidence=evidence)
        prob = query_result.values[0]  # Probability of Group=1 (Demented)
        y_proba.append(prob)
        y_pred.append(1 if prob >= 0.5 else 0)
    except KeyError as e:
        print(f"KeyError for evidence {evidence}: {e}")

test_time = time.time() - start_test_time

# %%
# Calculate Metrics
accuracy = accuracy_score(y_true, y_pred)
auc_score = roc_auc_score(y_true, y_proba)
kl_divergence = log_loss(y_true, y_proba)  # Log Loss approximates KL Divergence in binary classification
brier_score = brier_score_loss(y_true, y_proba)

# Print Results
print("Model Evaluation Metrics for Bayesian Network with Hill Climb Structure Learning:")
print(f"Accuracy: {accuracy:.4f}")
print(f"AUC: {auc_score:.4f}")
print(f"Kullback-Leibler Divergence (Log Loss): {kl_divergence:.4f}")
print(f"Brier Score: {brier_score:.4f}")
print(f"Training Time: {train_time:.4f} seconds")
print(f"Testing Time: {test_time:.4f} seconds")

# %%
# Run final query
# Slightly adjust evidence values to test sensitivity
adjusted_query_evidence = {
    'Visit': 1,
    'Age': get_bin(60, age_bins),  # Adjust age to see if results differ
    'EDUC': 12,
    'SES': 2,
    'MMSE': get_bin(16, mmse_bins),  # Adjust MMSE slightly
    'CDR': 0.5,
    'eTIV': get_bin(1698, etiv_bins),
    'nWBV': get_bin(0.701, nwbv_bins),
    'ASF': get_bin(1.034, asf_bins)
}

# Remove 'Visit' if it's not part of the learned structure
if 'Visit' not in learned_structure.nodes():
    adjusted_query_evidence.pop('Visit', None)

# Run the adjusted query
adjusted_query_result = learned_inference.query(variables=['Group'], evidence=adjusted_query_evidence)

print("Adjusted Conditional Probability for Group given modified evidence:")
for state, prob in zip(adjusted_query_result.state_names['Group'], adjusted_query_result.values):
    print(f"P(Group = {state} | adjusted evidence) = {prob:.4f}")