import pandas as pd
from sklearn.impute import SimpleImputer
from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.estimators import MaximumLikelihoodEstimator

# Example: Preprocess data
data = pd.read_csv(r'C:\Users\Student\Documents\Module Dev Containers\AAI-assignment\mycode\dementia_data-MRI-features.csv')  # Assuming data is stored in CSV


################################################################################
# not from pgmpy chatgpt #

# Creating a new column 'Last_Visit' to identify the last visit for each patient
data['Last_Visit'] = data.groupby('Subject ID')['Visit'].transform('max')

# Updating the 'Group' column based on 'Visit' and 'Last_Visit' conditions
data.loc[data['Visit'] < data['Last_Visit'], 'Group'] = 'Nondemented'
data.loc[data['Visit'] == data['Last_Visit'], 'Group'] = 'Demented'

# Dropping the 'Last_Visit' column
data.drop('Last_Visit', axis=1, inplace=True)
#################################################################################


import pandas as pd

# Assuming your dataframe is named 'df'

# Impute missing values in MMSE with the mean
data['MMSE'].fillna(data['MMSE'].mean(), inplace=True)

# Impute missing values in SES with the mode
data['SES'].fillna(data['SES'].mode()[0], inplace=True)

# Verify the changes
print(data[['MMSE', 'SES']].isnull().sum())  # This should show 0 for both columns


# Define the Bayesian Network structure (relationships between nodes)
model = BayesianNetwork([
    ('Age', 'MMSE'),   # MMSE depends on Age
    ('EDUC', 'SES'),   # SES depends on EDUC (education level)
    ('MMSE', 'CDR')    # CDR depends on MMSE
])


# Fit the model to the data using Maximum Likelihood Estimation (MLE)
model.fit(data, estimator=MaximumLikelihoodEstimator)

# After fitting, you can check the CPDs learned by the model
for cpd in model.get_cpds():
    print(cpd)

# Check if the model is valid
assert model.check_model()



########### check for the rest of the code in chatGPT ##########

# https://chatgpt.com/share/672d362c-1274-8010-81b2-513b2a507c45

################################################################











# # Encode categorical variables (simplified example)
# data['M/F'] = data['M/F'].map({'M': 0, 'F': 1})
# data['Group'] = data['Group'].map({'Nondemented': 0, 'Demented': 1, 'Converted': 2})

# # Construct the Bayesian Network (simple structure)
# model = BayesianNetwork([('Age', 'Group'), ('MMSE', 'Group')])

# # Add the data as factors to the model
# # These factors would typically be learned from data, but can also be manually defined
# model.add_factors(
#     DiscreteFactor(['Age'], [4], values=[0.2, 0.3, 0.3, 0.2]),  # Example factor for Age
#     DiscreteFactor(['MMSE'], [4], values=[0.3, 0.2, 0.4, 0.1])   # Example factor for MMSE
# )

# # Check if the model is valid
# assert model.check_model()


# # Perform inference
# inference = VariableElimination(model)
# probability_query = inference.query(variables=['Group'], evidence={'Age': 2, 'MMSE': 3})

# print(probability_query)
