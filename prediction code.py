# Load the trained MLP model
with open('/content/drive/MyDrive/pjt/new.pkl', 'rb') as file:
    mlp_biopsy = joblib.load(file)
    mlp_biopsy.feature_names_in_ = None  # Ensure feature_names_in_ attribute is set to None

# Input the features for the new individual
new_data = pd.DataFrame({
    'Age': [51],
    'Smokes (years)': [34],
    'Smokes (packs/year)': [3.4],
    'Hormonal Contraceptives (years)': [0],
    'IUD': [1],
    'IUD (years)': [7],
    'STDs': [0],
    'STDs (number)': [0],
    'STDs:condylomatosis': [0],
    'STDs:vulvo-perineal condylomatosis': [0],
    'STDs:genital herpes': [0],
    'STDs:HIV': [0],
    'STDs: Number of diagnosis': [0],
    'Dx:Cancer': [0],
    'Dx:CIN': [0],
    'Dx:HPV': [0],
    'Dx': [0],
    'Hinselmann': [0],
    'Schiller': [1],
    'Citology': [0]
})
# Extract the feature names used during feature selection for cervical cancer (Biopsy)
selected_feature_indices_biopsy = selector.get_support(indices=True)

# Verify that the selected indices are valid for the columns in new_data for cervical cancer (Biopsy)
if len(selected_feature_indices_biopsy) == len(new_data.columns):
    selected_new_data_biopsy = new_data  # All features are selected
else:
    selected_new_data_biopsy = new_data.iloc[:, selected_feature_indices_biopsy]

# Predict cervical cancer (Biopsy) for the new individual
prediction_biopsy = mlp_biopsy.predict(selected_new_data_biopsy)

# Convert predictions to "True" or "False" based on their values
biopsy_prediction = "True" if prediction_biopsy[0] == 1 else "False"

# Print the predictions
print("Cervical Cancer (Biopsy) Prediction:", biopsy_prediction)
