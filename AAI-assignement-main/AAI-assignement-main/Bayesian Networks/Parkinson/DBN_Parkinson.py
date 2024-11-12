# %%
import pandas as pd
from pgmpy.models import BayesianNetwork
from pgmpy.estimators import MaximumLikelihoodEstimator, HillClimbSearch, BicScore
from pgmpy.inference import VariableElimination
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss, brier_score_loss
from sklearn.model_selection import train_test_split
import numpy as np
import time
import matplotlib.pyplot as plt
import networkx as nx
from pgmpy.models import BayesianNetwork

# %%
file_path = r'C:\Users\Student\Downloads\AAI-assignment-main\AAI-assignment-main\Datasets and old trials\parkinsons_data-VOICE-features.csv'  # Replace with your local file path
parkinson_df = pd.read_csv(file_path)

# %%
# Drop irrelevant columns
parkinson_df = parkinson_df.drop(columns=['name'], errors='ignore')

#%%
#check if there is any missing data to be handled
missing_data_count = parkinson_df.isna().sum().sum()
print("Total missing data:", missing_data_count) # Output: 0

# # %%
# # Ensure Group is binary-encoded
# parkinson_df['Group'] = parkinson_df['Group'].apply(lambda x: 1 if x == 'Demented' else 0)


# %%
# Discretize continuous features with labels=False
# Discretize frequency-related features
parkinson_df['MDVP:Fo(Hz)'] = pd.cut(parkinson_df['MDVP:Fo(Hz)'], bins=4, labels=False)
parkinson_df['MDVP:Fhi(Hz)'] = pd.cut(parkinson_df['MDVP:Fhi(Hz)'], bins=4, labels=False)
parkinson_df['MDVP:Flo(Hz)'] = pd.cut(parkinson_df['MDVP:Flo(Hz)'], bins=4, labels=False)

# Discretize jitter-related features
parkinson_df['MDVP:Jitter(%)'] = pd.cut(parkinson_df['MDVP:Jitter(%)'], bins=4, labels=False)
parkinson_df['MDVP:Jitter(Abs)'] = pd.cut(parkinson_df['MDVP:Jitter(Abs)'], bins=4, labels=False)
parkinson_df['MDVP:RAP'] = pd.cut(parkinson_df['MDVP:RAP'], bins=4, labels=False)
parkinson_df['MDVP:PPQ'] = pd.cut(parkinson_df['MDVP:PPQ'], bins=4, labels=False)
parkinson_df['Jitter:DDP'] = pd.cut(parkinson_df['Jitter:DDP'], bins=4, labels=False)

# Discretize shimmer-related features
parkinson_df['MDVP:Shimmer'] = pd.cut(parkinson_df['MDVP:Shimmer'], bins=4, labels=False)
parkinson_df['MDVP:Shimmer(dB)'] = pd.cut(parkinson_df['MDVP:Shimmer(dB)'], bins=4, labels=False)
parkinson_df['Shimmer:APQ3'] = pd.cut(parkinson_df['Shimmer:APQ3'], bins=4, labels=False)
parkinson_df['Shimmer:APQ5'] = pd.cut(parkinson_df['Shimmer:APQ5'], bins=4, labels=False)
parkinson_df['MDVP:APQ'] = pd.cut(parkinson_df['MDVP:APQ'], bins=4, labels=False)
parkinson_df['Shimmer:DDA'] = pd.cut(parkinson_df['Shimmer:DDA'], bins=4, labels=False)

# Discretize NHR and HNR
parkinson_df['NHR'] = pd.cut(parkinson_df['NHR'], bins=4, labels=False)
parkinson_df['HNR'] = pd.cut(parkinson_df['HNR'], bins=4, labels=False)

# Discretize other continuous metrics
parkinson_df['RPDE'] = pd.cut(parkinson_df['RPDE'], bins=4, labels=False)
parkinson_df['DFA'] = pd.cut(parkinson_df['DFA'], bins=4, labels=False)
parkinson_df['spread1'] = pd.cut(parkinson_df['spread1'], bins=4, labels=False)
parkinson_df['spread2'] = pd.cut(parkinson_df['spread2'], bins=4, labels=False)
parkinson_df['D2'] = pd.cut(parkinson_df['D2'], bins=4, labels=False)
parkinson_df['PPE'] = pd.cut(parkinson_df['PPE'], bins=4, labels=False)

# Display the transformed DataFrame
parkinson_df.head(10)

# %%
# Split data into training and testing sets
train_df, test_df = train_test_split(parkinson_df, test_size=0.2, stratify=parkinson_df['status'], random_state=42)

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
# Store the bin edges for each feature instead of discretizing the data directly with labels
# Define bins for frequency-related features
mdvp_fo_bins = pd.cut(parkinson_df['MDVP:Fo(Hz)'], bins=4, retbins=True)[1]
mdvp_fhi_bins = pd.cut(parkinson_df['MDVP:Fhi(Hz)'], bins=4, retbins=True)[1]
mdvp_flo_bins = pd.cut(parkinson_df['MDVP:Flo(Hz)'], bins=4, retbins=True)[1]

# Define bins for jitter-related features
mdvp_jitter_percent_bins = pd.cut(parkinson_df['MDVP:Jitter(%)'], bins=4, retbins=True)[1]
mdvp_jitter_abs_bins = pd.cut(parkinson_df['MDVP:Jitter(Abs)'], bins=4, retbins=True)[1]
mdvp_rap_bins = pd.cut(parkinson_df['MDVP:RAP'], bins=4, retbins=True)[1]
mdvp_ppq_bins = pd.cut(parkinson_df['MDVP:PPQ'], bins=4, retbins=True)[1]
jitter_ddp_bins = pd.cut(parkinson_df['Jitter:DDP'], bins=4, retbins=True)[1]

# Define bins for shimmer-related features
mdvp_shimmer_bins = pd.cut(parkinson_df['MDVP:Shimmer'], bins=4, retbins=True)[1]
mdvp_shimmer_db_bins = pd.cut(parkinson_df['MDVP:Shimmer(dB)'], bins=4, retbins=True)[1]
shimmer_apq3_bins = pd.cut(parkinson_df['Shimmer:APQ3'], bins=4, retbins=True)[1]
shimmer_apq5_bins = pd.cut(parkinson_df['Shimmer:APQ5'], bins=4, retbins=True)[1]
mdvp_apq_bins = pd.cut(parkinson_df['MDVP:APQ'], bins=4, retbins=True)[1]
shimmer_dda_bins = pd.cut(parkinson_df['Shimmer:DDA'], bins=4, retbins=True)[1]

# Define bins for NHR and HNR
nhr_bins = pd.cut(parkinson_df['NHR'], bins=4, retbins=True)[1]
hnr_bins = pd.cut(parkinson_df['HNR'], bins=4, retbins=True)[1]

# Define bins for other continuous metrics
rpde_bins = pd.cut(parkinson_df['RPDE'], bins=4, retbins=True)[1]
dfa_bins = pd.cut(parkinson_df['DFA'], bins=4, retbins=True)[1]
spread1_bins = pd.cut(parkinson_df['spread1'], bins=4, retbins=True)[1]
spread2_bins = pd.cut(parkinson_df['spread2'], bins=4, retbins=True)[1]
d2_bins = pd.cut(parkinson_df['D2'], bins=4, retbins=True)[1]
ppe_bins = pd.cut(parkinson_df['PPE'], bins=4, retbins=True)[1]

# Helper function to map continuous values to bins
def get_bin(value, bins):
    bin_index = pd.cut([value], bins=bins, labels=False)[0]
    return int(bin_index) if not pd.isna(bin_index) else int(len(bins) - 2 if value > bins[-1] else 0)

# %%
# Model Evaluation
y_true = test_df['status'].values
y_pred = []  #predicted labels
y_proba = [] #predicted probabilities

# Start testing time
start_test_time = time.time()
for _, row in test_df.iterrows():
    # Convert continuous values to bins and prepare evidence
    evidence = {
        'MDVP:Fo(Hz)': get_bin(row['MDVP:Fo(Hz)'], mdvp_fo_bins),
        'MDVP:Fhi(Hz)': get_bin(row['MDVP:Fhi(Hz)'], mdvp_fhi_bins),
        'MDVP:Flo(Hz)': get_bin(row['MDVP:Flo(Hz)'], mdvp_flo_bins),
        'MDVP:Jitter(%)': get_bin(row['MDVP:Jitter(%)'], mdvp_jitter_percent_bins),
        'MDVP:Jitter(Abs)': get_bin(row['MDVP:Jitter(Abs)'], mdvp_jitter_abs_bins),
        'MDVP:RAP': get_bin(row['MDVP:RAP'], mdvp_rap_bins),
        'MDVP:PPQ': get_bin(row['MDVP:PPQ'], mdvp_ppq_bins),
        'Jitter:DDP': get_bin(row['Jitter:DDP'], jitter_ddp_bins),
        'MDVP:Shimmer': get_bin(row['MDVP:Shimmer'], mdvp_shimmer_bins),
        'MDVP:Shimmer(dB)': get_bin(row['MDVP:Shimmer(dB)'], mdvp_shimmer_db_bins),
        'Shimmer:APQ3': get_bin(row['Shimmer:APQ3'], shimmer_apq3_bins),
        'Shimmer:APQ5': get_bin(row['Shimmer:APQ5'], shimmer_apq5_bins),
        'MDVP:APQ': get_bin(row['MDVP:APQ'], mdvp_apq_bins),
        'Shimmer:DDA': get_bin(row['Shimmer:DDA'], shimmer_dda_bins),
        'NHR': get_bin(row['NHR'], nhr_bins),
        'HNR': get_bin(row['HNR'], hnr_bins),
        'RPDE': get_bin(row['RPDE'], rpde_bins),
        'DFA': get_bin(row['DFA'], dfa_bins),
        'spread1': get_bin(row['spread1'], spread1_bins),
        'spread2': get_bin(row['spread2'], spread2_bins),
        'D2': get_bin(row['D2'], d2_bins),
        'PPE': get_bin(row['PPE'], ppe_bins)
    }

    try:
        # Query the probabilistic model (assumed to be already trained)
        query_result = learned_inference.query(variables=['status'], evidence=evidence)
        
        # Extract the probability of 'status' = 1 (Parkinson's disease present)
        prob = query_result.values[1]  # Probability of status = 1 (Parkinson's)
        
        # Store the predicted probability
        y_proba.append(prob)
        
        # Store the predicted label based on threshold of 0.5
        y_pred.append(1 if prob >= 0.5 else 0)
        
    except KeyError as e:
        print(f"KeyError for evidence {evidence}: {e}")

# Calculate the time taken for testing
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
# Run final query for Parkinson dataset
# Slightly adjust evidence values to test sensitivity
# phon_R01_S04_5															1						
adjusted_query_evidence = {
    # Adjusting some continuous feature values slightly
    'MDVP:Fo(Hz)': get_bin(144.188, mdvp_fo_bins),   # Adjust MDVP:Fo(Hz)
    'MDVP:Fhi(Hz)': get_bin(349.259, mdvp_fhi_bins), # Adjust MDVP:Fhi(Hz)
    'MDVP:Flo(Hz)': get_bin(111.764, mdvp_flo_bins), # Adjust MDVP:Flo(Hz)
    'MDVP:Jitter(%)': get_bin(0.00544, mdvp_jitter_percent_bins), # Adjust MDVP:Jitter(%)
    'MDVP:Jitter(Abs)': get_bin(0.00004, mdvp_jitter_abs_bins), # Adjust MDVP:Jitter(Abs)
    'MDVP:RAP': get_bin(0.00211, mdvp_rap_bins),    # Adjust MDVP:RAP
    'MDVP:PPQ': get_bin(0.00292, mdvp_ppq_bins),    # Adjust MDVP:PPQ
    'Jitter:DDP': get_bin(0.00632, jitter_ddp_bins), # Adjust Jitter:DDP
    'MDVP:Shimmer': get_bin(0.02047, mdvp_shimmer_bins), # Adjust MDVP:Shimmer
    'MDVP:Shimmer(dB)': get_bin(0.192, mdvp_shimmer_db_bins), # Adjust MDVP:Shimmer(dB)
    'Shimmer:APQ3': get_bin(0.00969, shimmer_apq3_bins),  # Adjust Shimmer:APQ3
    'Shimmer:APQ5': get_bin(0.012, shimmer_apq5_bins), # Adjust Shimmer:APQ5
    'MDVP:APQ': get_bin(0.02074, mdvp_apq_bins),    # Adjust MDVP:APQ
    'Shimmer:DDA': get_bin(0.02908	, shimmer_dda_bins),  # Adjust Shimmer:DDA
    'NHR': get_bin(	0.01859, nhr_bins),              # Adjust NHR
    'HNR': get_bin(17.333, hnr_bins),                # Adjust HNR
    'RPDE': get_bin(0.56738, rpde_bins),             # Adjust RPDE
    'DFA': get_bin(0.644692, dfa_bins),               # Adjust DFA
    'spread1': get_bin(-5.44004, spread1_bins),       # Adjust spread1
    'spread2': get_bin(0.239764, spread2_bins),       # Adjust spread2
    'D2': get_bin(2.264501, d2_bins),                # Adjust D2
    'PPE': get_bin(0.218164, ppe_bins)               # Adjust PPE
}


# Run the adjusted query
adjusted_query_result = learned_inference.query(variables=['status'], evidence=adjusted_query_evidence)

print("Adjusted Conditional Probability for Group given modified evidence:")
for state, prob in zip(adjusted_query_result.state_names['status'], adjusted_query_result.values):
    print(f"P(Group = {state} | adjusted evidence) = {prob:.4f}")



# Running the query with the adjusted evidence to test sensitivity
try:
    # Run the query on the learned inference model (assuming it's already trained)
    query_result = learned_inference.query(variables=['status'], evidence=adjusted_query_evidence)
    
    # Extract the probability of having Parkinson's disease (status = 1)
    prob = query_result.values[1]  # Probability of status = 1 (Parkinson's)
    
    # Print the adjusted result
    print(f"Adjusted probability of Parkinson's (status = 1): {prob:.4f}")
    
except KeyError as e:
    print(f"Error in running query for adjusted evidence: {e}")


# %%

# Plotting the learned Bayesian Network structure (DAG)
plt.figure(figsize=(12, 8))
G = nx.DiGraph()

# Add edges from the learned structure
G.add_edges_from(best_model.edges())

# Draw the graph
pos = nx.spring_layout(G)  # You can use other layouts as well (e.g., circular, shell)
nx.draw(G, pos, with_labels=True, node_size=3000, node_color="skyblue", font_size=10, font_weight="bold", arrowsize=20)
plt.title("DAG of the Learned Bayesian Network")
plt.show()
# %%
