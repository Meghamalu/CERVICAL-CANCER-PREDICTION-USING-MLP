import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, confusion_matrix
from imblearn.over_sampling import RandomOverSampler
import seaborn as sns
import matplotlib.pyplot as plt
import zipfile
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import joblib

from google.colab import drive
drive.mount('/content/drive')

# Load the dataset
cancer_df = pd.read_csv('/content/drive/MyDrive/pjt/risk_factors_cervical_cancer.csv')

# Display the last 20 rows of the DataFrame
cancer_df.tail(20)


# print out a summary of the DataFrame's information
cancer_df.info()

#display the descriptive statistics of the numerical columns in the DataFrame
cancer_df.describe()

# REPLACING '?' WITH NaN
cancer_df = cancer_df.replace('?', np.nan)
cancer_df

# PLOTTING HEATMAP TO VISUALIZE THE NUMBER OF NaN'S IN TH DATA

plt.figure(figsize=( 20,20))
sns.heatmap(cancer_df.isnull(), yticklabels = False)
plt.show()

# WE OBSERVE THAT THERE ARE A LOT OF NAN VALUES IN "STD'S: TIME SINCE FIRST DIAGNOSIS" AND "STD'S: TIME SINCE LAST DIAGNOSIS"
# SO WE WILL DROP THESE COLUMNS

cancer_df = cancer_df.drop(['STDs: Time since first diagnosis', 'STDs: Time since last diagnosis'], axis=1)
cancer_df

# Converting the column data types, from object to numeric in order to perform Statistical Analysis of the Data

cancer_df = cancer_df.apply(pd.to_numeric)
cancer_df.info()

#compute the mean of each column in the DataFrame
cancer_df.mean()

# REPLACING NULL/NaN values with the mean values:

cancer_df =  cancer_df.fillna(cancer_df.mean())
cancer_df

# PLOTTING HEATMAP AGAIN TO VISUALIZE AND CHECK OUR DATA CLEANSING

plt.figure(figsize=(8,20))
sns.heatmap(cancer_df.isnull(), yticklabels = False)
plt.xticks(rotation=90)
plt.tick_params(labelsize=8)
plt.show()

cancer_df.describe()

# WE'LL TRY TO OBSERVE THE CORELATION BETWEEN DIFFERENT FEATURES IN OUR DATASETS:

corr_matrix = cancer_df.corr()

corr_matrix

# PLOTTING THE HEATMAP FOR CORRELATION MATRIX

plt.figure(figsize = (20,20))

sns.heatmap(corr_matrix, annot=True)

plt.xticks(rotation=90)

plt.yticks(rotation=360)

plt.tick_params(labelsize=8)

plt.show()


# VISUALIZING THE WHOLE DATAFRAME BY PLOTTING HISTOGRAM
cancer_df.hist(bins = 10, figsize = (10,10), color='blue')
plt.show()

#the features are separated from the target variable. 'X' contains the features, excluding the 'Biopsy' column.
X = cancer_df.drop(['Biopsy'], axis=1)

# Define the target variable y
y = cancer_df['Biopsy']

# Perform feature selection with SelectKBest and Chi-squared test
selector = SelectKBest(score_func=chi2, k=20)
selected_features = selector.fit_transform(X, y)

# Get selected feature names
selected_feature_names = X.columns[selector.get_support()]

# Get feature scores
feature_scores = selector.scores_

# Get indices of selected features
selected_indices = selector.get_support(indices=True)

# Print importance of selected features
print("Importance of Selected Features:")
for name, score in zip(selected_feature_names, feature_scores[selected_indices]):
    print(f"{name}: {score}")

# Resample the features and target variable using RandomOverSampler
ros = RandomOverSampler(random_state=42)
X_resampled, y_resampled = ros.fit_resample(selected_features, y)

import matplotlib.pyplot as plt

# Plot importance of selected features
plt.figure(figsize=(10, 6))  # Adjust the figure size as needed
plt.bar(selected_feature_names, feature_scores[selected_indices], color='skyblue')
plt.xlabel('Selected Features')
plt.ylabel('Feature Score')
plt.title('Importance of Selected Features')
plt.xticks(rotation=90)  # Rotate x-axis labels for better readability
plt.tight_layout()  # Adjust layout to prevent clipping of labels
plt.show()



# Initialize the MLP classifier
mlp = MLPClassifier(hidden_layer_sizes=(100, 100), activation='relu', max_iter=500, random_state=42)

# Collecting scores for accuracy, F1 score, recall, and precision
accuracy_scores = []
f1_scores = []
recall_scores = []
precision_scores = []
confusion_matrices = []

# Stratified K-Fold cross-validation
for train_index, test_index in StratifiedKFold(n_splits=5, shuffle=True, random_state=42).split(X_resampled, y_resampled):
    X_train, X_test = X_resampled[train_index], X_resampled[test_index]
    y_train, y_test = y_resampled[train_index], y_resampled[test_index]

    # Fit the MLP classifier
    mlp.fit(X_train, y_train)

    # Predict on the test set
    y_pred = mlp.predict(X_test)

    # Calculate evaluation metrics
    accuracy_scores.append(accuracy_score(y_test, y_pred))
    f1_scores.append(f1_score(y_test, y_pred))
    recall_scores.append(recall_score(y_test, y_pred))
    precision_scores.append(precision_score(y_test, y_pred))
    confusion_matrices.append(confusion_matrix(y_test, y_pred))

# Calculate and print average scores
average_accuracy = np.mean(accuracy_scores)
average_f1 = np.mean(f1_scores)
average_recall = np.mean(recall_scores)
average_precision = np.mean(precision_scores)

print("Average Accuracy:", average_accuracy)
print("Average F1 Score:", average_f1)
print("Average Recall:", average_recall)
print("Average Precision:", average_precision)

# Print the confusion matrices
for idx, cm in enumerate(confusion_matrices):
    print(f"Confusion Matrix for Fold {idx + 1}:")
    print(cm)


# Save the trained model to a file
joblib.dump(mlp,'/content/drive/MyDrive/pjt/new.pkl')
