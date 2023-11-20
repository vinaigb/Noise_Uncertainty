
import pandas as pd #for working with dataframes
import numpy as np #for numerical and vector operations
import matplotlib as mpl #for visualization
import matplotlib.pyplot as plt #for visualization
import seaborn as sns #for visualization
import lightgbm as lgb #for light gradient boosting machine
import xgboost as xgb #for extreme gradient boosting 

# Metrics
from sklearn import metrics #metrics
from sklearn.metrics import classification_report #for F1 score and other metrics
from sklearn.metrics import f1_score #for F1 score specifically
from sklearn.metrics import matthews_corrcoef #for Matthews correlation coefficient (MCC)
from sklearn.metrics import confusion_matrix #for confusion matrix
from sklearn.metrics import roc_curve, roc_auc_score #ROC and AUC scores
from sklearn.metrics import ConfusionMatrixDisplay #for plotting confusion matrix
from sklearn.calibration import calibration_curve
from sklearn.metrics import auc

# Importing machine learning tools for preprocessing
from sklearn.model_selection import train_test_split #for splitting the data into training and test sets
from sklearn.preprocessing import StandardScaler #for feature scaling

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, f1_score, matthews_corrcoef, confusion_matrix, roc_auc_score, balanced_accuracy_score, cohen_kappa_score, hamming_loss, log_loss, average_precision_score
import lightgbm as lgb
    
data = pd.read_csv("Data/Data_Detecting_Changeover.csv", sep= ',',header=0, index_col=0)
data.head()
data.info()


# Columns to add noise to
columns_to_add_noise = ['OverrideFeed', 'FeedRate', 'SpindleSpeed', 'IndoorGPSx', 'IndoorGPSy']


# Convert columns to categorical
categorical_columns = ['ProgramStatus', 'ToolNumber', 'PocketTable', 'DriveStatus', 'DoorStatusTooling', 'Phase', 'Phase_compressed','Production']
data[categorical_columns] = data[categorical_columns].astype('category')

data.info()

print(data.describe())

pd.DataFrame(data.ProgramStatus.value_counts())



# Select the columns to encode
columns_to_encode = ['ProgramStatus', 'ToolNumber', 'PocketTable', 'DriveStatus', 'DoorStatusTooling']

# Perform one-hot encoding on the selected columns
encoded_columns = pd.get_dummies(data[columns_to_encode])

# Append the encoded columns to the original dataset
data = pd.concat([data, encoded_columns], axis=1)

# Remove the original columns after encoding
data = data.drop(columns_to_encode, axis=1)

# Move 'Production' column to the rightmost position
production_column = data.pop('Production')
data['Production'] = production_column

# Display the updated dataset
print(data.head())


# Dropping the columns we do not need
df = data.copy()
df.pop('DateTime')
df.pop('Phase')
df.pop('Phase_compressed')
print(df)

df.info()

### HyperParameter Tuning
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    hamming_loss,
    log_loss,
)

# Add periodic noise to the specified columns
for column in columns_to_add_noise:
    noise = np.sin(np.arange(len(data)) / 10)
    data[column] += noise

# Parameters
num_simulations = 1  # Adjust as needed
min_intensity = 0.0
max_intensity = 0.95
num_intensity_steps = 200
columns_to_scale = ['OverrideFeed', 'FeedRate', 'SpindleSpeed', 'IndoorGPSx', 'IndoorGPSy']

# Create an empty list to store results
results_lightgbm = []



# Splitting the data into training and test sets
train_df, test_df = train_test_split(df, test_size=0.2, random_state=0)

# Using numpy to create arrays of labels and features
train_labels = np.array(train_df['Production'])
test_labels = np.array(test_df['Production'])
train_features = np.array(train_df.iloc[:, 0:38])
test_features = np.array(test_df.iloc[:, 0:38])

# Scaling the selected columns using StandardScaler
scaler = StandardScaler()
train_features[:, :5] = scaler.fit_transform(train_features[:, :5])
test_features[:, :5] = scaler.transform(test_features[:, :5])

# Loop over different noise intensity levels
intensity_steps = np.linspace(min_intensity, max_intensity, num=num_intensity_steps)

# Define the hyperparameter grid for grid search
param_grid = {
    'n_estimators': [100, 200, 300],  # Number of boosting rounds
    'learning_rate': [0.01, 0.1, 0.2],  # Learning rate
    'max_depth': [5, 7, 9],  # Maximum tree depth
    'num_leaves': [31, 40, 50],  # Maximum number of leaves in a tree
    'min_child_samples': [10, 20, 30],  # Minimum number of data points in a leaf
    'subsample': [0.8, 0.9, 1.0],  # Fraction of data used for training each tree
    'colsample_bytree': [0.8, 0.9, 1.0],  # Fraction of features used for training each tree
    'reg_alpha': [0.0, 0.1, 0.5],  # L1 regularization term
    'reg_lambda': [0.0, 0.1, 0.5],  # L2 regularization term
    'bagging_fraction': [0.7, 0.8, 0.9],  # Fraction of data used for bagging
    'feature_fraction': [0.7, 0.8, 0.9],  # Fraction of features used for training each tree
    'min_split_gain': [0.0, 0.1, 0.2],  # Minimum loss reduction required for further split
    'min_data_in_leaf': [20, 30, 40]  # Minimum number of data points in a leaf
}

# Create a GridSearchCV object
grid_search = GridSearchCV(estimator=lgb.LGBMClassifier(random_state=0),
                           param_grid=param_grid,
                           scoring='f1_macro',  # scoring metrics
                           cv=5,  # Number of cross-validation folds
                           n_jobs=-1  # Use all available CPU cores
                           )

# Perform the grid search
grid_search.fit(train_features, train_labels)

# Get the best hyperparameters from the grid search
best_params = grid_search.best_params_

# Print the best hyperparameters
print("Best Hyperparameters:")
for param, value in best_params.items():
    print(f"{param}: {value}")

# Loop over different noise intensity levels and train LightGBM models
for intensity_factor in intensity_steps:
    for _ in range(num_simulations):
        # Generate Gaussian noise and add to the data
        noise_train = np.random.normal(0, intensity_factor, size=train_features.shape)
        noise_test = np.random.normal(0, intensity_factor, size=test_features.shape)
        
        train_features_with_noise = train_features + noise_train
        test_features_with_noise = test_features + noise_test

        # Create and train the LightGBM classifier with the best hyperparameters
        clf_lightgbm = lgb.LGBMClassifier(random_state=0, **best_params)
        clf_lightgbm.fit(train_features_with_noise, train_labels)

    # Predict the response for the test dataset
        y_pred_lightgbm = clf_lightgbm.predict(test_features_with_noise)

        # Calculate evaluation metrics
        classification_rep = classification_report(test_labels, y_pred_lightgbm, zero_division=1)  # Set zero_division to 1
        f1_LGBM = f1_score(test_labels, y_pred_lightgbm, average='macro', zero_division=1)  # Set zero_division to 1
        mcc_LGBM = matthews_corrcoef(test_labels, y_pred_lightgbm)
        cm = confusion_matrix(test_labels, y_pred_lightgbm, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        specificity = tn / (tn + fp)
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        fpr = fp / (fp + tn)
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        roc_auc = roc_auc_score(test_labels, y_pred_lightgbm)
        average_precision = average_precision_score(test_labels, y_pred_lightgbm)
        balanced_accuracy = balanced_accuracy_score(test_labels, y_pred_lightgbm)
        kappa = cohen_kappa_score(test_labels, y_pred_lightgbm)
        hl = hamming_loss(test_labels, y_pred_lightgbm)
        logloss = log_loss(test_labels, y_pred_lightgbm)
        auprc = average_precision_score(test_labels, y_pred_lightgbm)

        # Store the results
        results_lightgbm.append({
            'intensity_factor': intensity_factor,
            'classification_report': classification_rep,
            'f1_score': f1_LGBM,
            'MCC': mcc_LGBM,
            'confusion_matrix': cm,
            'specificity': specificity,
            'precision': precision,
            'recall': recall,
            'fpr': fpr,
            'accuracy': accuracy,
            'ROC_AUC': roc_auc,
            'average_precision': average_precision,
            'balanced_accuracy': balanced_accuracy,
            'kappa': kappa,
            'hamming_loss': hl,
            'log_loss': log_loss,
            'auprc': auprc,
            'y_pred': y_pred_lightgbm
        })

for i, result in enumerate(results_lightgbm):
    print(f"Simulation {i+1}")
    print("Classification Report:")
    print("Intensity Factor:", result['intensity_factor'])
    print(result['classification_report'])
    print("F1 Score:", result['f1_score'])
    print("MCC:", result['MCC'])
    print("Confusion Matrix:")
    print(result['confusion_matrix'])
    print("Specificity:", result['specificity'])
    print("ROC AUC:", result['ROC_AUC'])
    print("Average Precision:", result['average_precision'])
    print("Balanced Accuracy:", result['balanced_accuracy'])
    print("Precision:", result['precision'])
    print("Recall:", result['recall'])
    print("False Positive Rate:", result['fpr'])
    print("Accuracy:", result['accuracy'])
    print("Cohen's Kappa:", result['kappa'])
    print("Hamming Loss:", result['hamming_loss'])
    print("Log Loss:", result['log_loss'])
    print("Area Under PR Curve:", result['auprc'])
    print()    # Predict the response for the



######################################
######## Periodic Noise

import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    hamming_loss,
    log_loss,
)

# Add periodic noise to the specified columns
for column in columns_to_add_noise:
    noise = np.sin(np.arange(len(data)) / 10)
    data[column] += noise

# Parameters
num_simulations = 1  # Adjust as needed
min_intensity = 0.0
max_intensity = 0.95
num_intensity_steps = 200
columns_to_scale = ['OverrideFeed', 'FeedRate', 'SpindleSpeed', 'IndoorGPSx', 'IndoorGPSy']

# Create an empty list to store results
results_lightgbm = []

# Splitting the data into training and test sets
train_df, test_df = train_test_split(df, test_size=0.2, random_state=0)

# Using numpy to create arrays of labels and features
train_labels = np.array(train_df['Production'])
test_labels = np.array(test_df['Production'])
train_features = np.array(train_df.iloc[:, 0:38])
test_features = np.array(test_df.iloc[:, 0:38])

# Scaling the selected columns using StandardScaler
scaler = StandardScaler()
train_features[:, :5] = scaler.fit_transform(train_features[:, :5])
test_features[:, :5] = scaler.transform(test_features[:, :5])

# Loop over different noise intensity levels
intensity_steps = np.linspace(min_intensity, max_intensity, num=num_intensity_steps)
for intensity_factor in intensity_steps:
    for _ in range(num_simulations):
        # Generate periodic noise and apply it to the data
        n_train = len(train_features)
        n_test = len(test_features)
        noise_train = np.sin(np.arange(n_train) / 10)
        noise_test = np.sin(np.arange(n_test) / 10)

        periodic_noise_train = noise_train * intensity_factor
        periodic_noise_test = noise_test * intensity_factor
        
        train_features_with_noise = train_features + periodic_noise_train[:, np.newaxis]
        test_features_with_noise = test_features + periodic_noise_test[:, np.newaxis]
        
        # Create and train the LightGBM classifier
        clf_lightgbm = lgb.LGBMClassifier(random_state=0)
        clf_lightgbm.fit(train_features_with_noise, train_labels)

        # Predict the response for the test dataset
        y_pred_lightgbm = clf_lightgbm.predict(test_features_with_noise)

        # Calculate evaluation metrics
        classification_rep = classification_report(test_labels, y_pred_lightgbm, zero_division=1)  # Set zero_division to 1
        f1_LGBM = f1_score(test_labels, y_pred_lightgbm, average='macro', zero_division=1)  # Set zero_division to 1
        mcc_LGBM = matthews_corrcoef(test_labels, y_pred_lightgbm)
        cm = confusion_matrix(test_labels, y_pred_lightgbm, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        specificity = tn / (tn + fp)
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        fpr = fp / (fp + tn)
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        roc_auc = roc_auc_score(test_labels, y_pred_lightgbm)
        average_precision = average_precision_score(test_labels, y_pred_lightgbm)
        balanced_accuracy = balanced_accuracy_score(test_labels, y_pred_lightgbm)
        kappa = cohen_kappa_score(test_labels, y_pred_lightgbm)
        hl = hamming_loss(test_labels, y_pred_lightgbm)
        logloss = log_loss(test_labels, y_pred_lightgbm)
        auprc = average_precision_score(test_labels, y_pred_lightgbm)

        # Store the results
        results_lightgbm.append({
            'intensity_factor': intensity_factor,
            'classification_report': classification_rep,
            'f1_score': f1_LGBM,
            'MCC': mcc_LGBM,
            'confusion_matrix': cm,
            'specificity': specificity,
            'precision': precision,
            'recall': recall,
            'fpr': fpr,
            'accuracy': accuracy,
            'ROC_AUC': roc_auc,
            'average_precision': average_precision,
            'balanced_accuracy': balanced_accuracy,
            'kappa': kappa,
            'hamming_loss': hl,
            'log_loss': log_loss,
            'auprc': auprc,
            'y_pred': y_pred_lightgbm
        })

for i, result in enumerate(results_lightgbm):
    print(f"Simulation {i+1}")
    print("Classification Report:")
    print("Intensity Factor:", result['intensity_factor'])
    print(result['classification_report'])
    print("F1 Score:", result['f1_score'])
    print("MCC:", result['MCC'])
    print("Confusion Matrix:")
    print(result['confusion_matrix'])
    print("Specificity:", result['specificity'])
    print("ROC AUC:", result['ROC_AUC'])
    print("Average Precision:", result['average_precision'])
    print("Balanced Accuracy:", result['balanced_accuracy'])
    print("Precision:", result['precision'])
    print("Recall:", result['recall'])
    print("False Positive Rate:", result['fpr'])
    print("Accuracy:", result['accuracy'])
    print("Cohen's Kappa:", result['kappa'])
    print("Hamming Loss:", result['hamming_loss'])
    print("Log Loss:", result['log_loss'])
    print("Area Under PR Curve:", result['auprc'])
    print()

########################################
###### multiplicative noise

import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    hamming_loss,
    log_loss,
)

# Add multiplicative noise to the specified columns
for column in columns_to_add_noise:
    noise = np.random.uniform(0.9, 1.1, size=len(data))
    data[column] *= noise

# Parameters
num_simulations = 1  # Adjust as needed
min_intensity = 0.0
max_intensity = 0.95
num_intensity_steps = 200
columns_to_scale = ['OverrideFeed', 'FeedRate', 'SpindleSpeed', 'IndoorGPSx', 'IndoorGPSy']

# Create an empty list to store results
results_lightgbm = []

# Splitting the data into training and test sets
train_df, test_df = train_test_split(df, test_size=0.2, random_state=0)

# Using numpy to create arrays of labels and features
train_labels = np.array(train_df['Production'])
test_labels = np.array(test_df['Production'])
train_features = np.array(train_df.iloc[:, 0:38])
test_features = np.array(test_df.iloc[:, 0:38])

# Scaling the selected columns using StandardScaler
scaler = StandardScaler()
train_features[:, :5] = scaler.fit_transform(train_features[:, :5])
test_features[:, :5] = scaler.transform(test_features[:, :5])

# Loop over different noise intensity levels
intensity_steps = np.linspace(min_intensity, max_intensity, num=num_intensity_steps)
for intensity_factor in intensity_steps:
    for _ in range(num_simulations):
        # Generate multiplicative noise and apply it to the data
        n_train = len(train_features)
        n_test = len(test_features)
        noise_train = np.random.uniform(0.9, 1.1, size=n_train)
        noise_test = np.random.uniform(0.9, 1.1, size=n_test)

        train_features_with_noise = train_features * noise_train[:, np.newaxis]
        test_features_with_noise = test_features * noise_test[:, np.newaxis]
        
        # Create and train the LightGBM classifier
        clf_lightgbm = lgb.LGBMClassifier(random_state=0)
        clf_lightgbm.fit(train_features_with_noise, train_labels)

        # Predict the response for the test dataset
        y_pred_lightgbm = clf_lightgbm.predict(test_features_with_noise)

        # Calculate evaluation metrics
        classification_rep = classification_report(test_labels, y_pred_lightgbm, zero_division=1)  # Set zero_division to 1
        f1_LGBM = f1_score(test_labels, y_pred_lightgbm, average='macro', zero_division=1)  # Set zero_division to 1
        mcc_LGBM = matthews_corrcoef(test_labels, y_pred_lightgbm)
        cm = confusion_matrix(test_labels, y_pred_lightgbm, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        specificity = tn / (tn + fp)
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        fpr = fp / (fp + tn)
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        roc_auc = roc_auc_score(test_labels, y_pred_lightgbm)
        average_precision = average_precision_score(test_labels, y_pred_lightgbm)
        balanced_accuracy = balanced_accuracy_score(test_labels, y_pred_lightgbm)
        kappa = cohen_kappa_score(test_labels, y_pred_lightgbm)
        hl = hamming_loss(test_labels, y_pred_lightgbm)
        logloss = log_loss(test_labels, y_pred_lightgbm)
        auprc = average_precision_score(test_labels, y_pred_lightgbm)

        # Store the results
        results_lightgbm.append({
            'intensity_factor': intensity_factor,
            'classification_report': classification_rep,
            'f1_score': f1_LGBM,
            'MCC': mcc_LGBM,
            'confusion_matrix': cm,
            'specificity': specificity,
            'precision': precision,
            'recall': recall,
            'fpr': fpr,
            'accuracy': accuracy,
            'ROC_AUC': roc_auc,
            'average_precision': average_precision,
            'balanced_accuracy': balanced_accuracy,
            'kappa': kappa,
            'hamming_loss': hl,
            'log_loss': logloss,
            'auprc': auprc,
            'y_pred': y_pred_lightgbm
        })

for i, result in enumerate(results_lightgbm):
    print(f"Simulation {i+1}")
    print("Classification Report:")
    print("Intensity Factor:", result['intensity_factor'])
    print(result['classification_report'])
    print("F1 Score:", result['f1_score'])
    print("MCC:", result['MCC'])
    print("Confusion Matrix:")
    print(result['confusion_matrix'])
    print("Specificity:", result['specificity'])
    print("ROC AUC:", result['ROC_AUC'])
    print("Average Precision:", result['average_precision'])
    print("Balanced Accuracy:", result['balanced_accuracy'])
    print("Precision:", result['precision'])
    print("Recall:", result['recall'])
    print("False Positive Rate:", result['fpr'])
    print("Accuracy:", result['accuracy'])
    print("Cohen's Kappa:", result['kappa'])
    print("Hamming Loss:", result['hamming_loss'])
    print("Log Loss:", result['log_loss'])
    print("Area Under PR Curve:", result['auprc'])
    print()



#################################
#########  impulse noise

import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    hamming_loss,
    log_loss,
)

# Add impulse noise to the specified columns
for column in columns_to_add_noise:
    noise = np.random.choice([-1, 1], size=len(data), p=[0.9, 0.1])
    data[column] += noise

# Parameters
num_simulations = 1  # Adjust as needed
min_intensity = 0.0
max_intensity = 0.95
num_intensity_steps = 200
columns_to_scale = ['OverrideFeed', 'FeedRate', 'SpindleSpeed', 'IndoorGPSx', 'IndoorGPSy']

# Create an empty list to store results
results_lightgbm = []

# Splitting the data into training and test sets
train_df, test_df = train_test_split(df, test_size=0.2, random_state=0)

# Using numpy to create arrays of labels and features
train_labels = np.array(train_df['Production'])
test_labels = np.array(test_df['Production'])
train_features = np.array(train_df.iloc[:, 0:38])
test_features = np.array(test_df.iloc[:, 0:38])

# Scaling the selected columns using StandardScaler
scaler = StandardScaler()
train_features[:, :5] = scaler.fit_transform(train_features[:, :5])
test_features[:, :5] = scaler.transform(test_features[:, :5])

# Loop over different noise intensity levels
intensity_steps = np.linspace(min_intensity, max_intensity, num=num_intensity_steps)
for intensity_factor in intensity_steps:
    for _ in range(num_simulations):
        # Generate impulse noise and add to the data
        n_train = len(train_features)
        n_test = len(test_features)
        noise_train = np.random.choice([-1, 1], size=n_train, p=[0.9, 0.1])
        noise_test = np.random.choice([-1, 1], size=n_test, p=[0.9, 0.1])

        impulse_noise_train = noise_train * np.sqrt(intensity_factor)
        impulse_noise_test = noise_test * np.sqrt(intensity_factor)
        
        train_features_with_noise = train_features + impulse_noise_train[:, np.newaxis]
        test_features_with_noise = test_features + impulse_noise_test[:, np.newaxis]
        
        # Create and train the LightGBM classifier
        clf_lightgbm = lgb.LGBMClassifier(random_state=0)
        clf_lightgbm.fit(train_features_with_noise, train_labels)

        # Predict the response for the test dataset
        y_pred_lightgbm = clf_lightgbm.predict(test_features_with_noise)


        classification_rep = classification_report(test_labels, y_pred_lightgbm)
        f1_LGBM = f1_score(test_labels, y_pred_lightgbm, average='macro')
     
        
        # Calculate evaluation metrics
       # classification_rep = classification_report(test_labels, y_pred_lightgbm, zero_division=1)  # Set zero_division to 1
        #f1_LGBM = f1_score(test_labels, y_pred_lightgbm, average='macro', zero_division=1)  # Set zero_division to 1
        mcc_LGBM = matthews_corrcoef(test_labels, y_pred_lightgbm)
        cm = confusion_matrix(test_labels, y_pred_lightgbm, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        specificity = tn / (tn + fp)
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        fpr = fp / (fp + tn) 
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        roc_auc = roc_auc_score(test_labels, y_pred_lightgbm)
        average_precision = average_precision_score(test_labels, y_pred_lightgbm)
        balanced_accuracy = balanced_accuracy_score(test_labels, y_pred_lightgbm)
        kappa = cohen_kappa_score(test_labels, y_pred_lightgbm)
        hl = hamming_loss(test_labels, y_pred_lightgbm)
        logloss = log_loss(test_labels, y_pred_lightgbm)
        auprc = average_precision_score(test_labels, y_pred_lightgbm)

        # Store the results
        results_lightgbm.append({
            'intensity_factor': intensity_factor,
            'classification_report': classification_rep,
            'f1_score': f1_LGBM,
            'MCC': mcc_LGBM,
            'confusion_matrix': cm,
            'specificity': specificity,
            'precision': precision,
            'recall': recall,
            'fpr': fpr,
            'accuracy': accuracy,
            'ROC_AUC': roc_auc,
            'average_precision': average_precision,
            'balanced_accuracy': balanced_accuracy,
            'kappa': kappa,
            'hamming_loss': hl,
            'log_loss': logloss,
            'auprc': auprc,
            'y_pred': y_pred_lightgbm
        })

for i, result in enumerate(results_lightgbm):
    print(f"Simulation {i+1}")
    print("Classification Report:")
    print("Intensity Factor:", result['intensity_factor'])
    print(result['classification_report'])
    print("F1 Score:", result['f1_score'])
    print("MCC:", result['MCC'])
    print("Confusion Matrix:")
    print(result['confusion_matrix'])
    print("Specificity:", result['specificity'])
    print("ROC AUC:", result['ROC_AUC'])
    print("Average Precision:", result['average_precision'])
    print("Balanced Accuracy:", result['balanced_accuracy'])
    print("Precision:", result['precision'])
    print("Recall:", result['recall'])
    print("False Positive Rate:", result['fpr'])
    print("Accuracy:", result['accuracy'])
    print("Cohen's Kappa:", result['kappa'])
    print("Hamming Loss:", result['hamming_loss'])
    print("Log Loss:", result['log_loss'])
    print("Area Under PR Curve:", result['auprc'])
    print()

#########################
##### salt and pepper noise

import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    hamming_loss,
    log_loss,
)

# Add salt and pepper noise to the specified columns
for column in columns_to_add_noise:
    noise = np.random.choice([-1, 0, 1], size=len(data), p=[0.05, 0.9, 0.05])
    data[column] += noise

# Parameters
num_simulations = 1  # Adjust as needed
min_intensity = 0.0
max_intensity = 0.95
num_intensity_steps = 200
columns_to_scale = ['OverrideFeed', 'FeedRate', 'SpindleSpeed', 'IndoorGPSx', 'IndoorGPSy']

# Create an empty list to store results
results_lightgbm = []

# Splitting the data into training and test sets
train_df, test_df = train_test_split(df, test_size=0.2, random_state=0)

# Using numpy to create arrays of labels and features
train_labels = np.array(train_df['Production'])
test_labels = np.array(test_df['Production'])
train_features = np.array(train_df.iloc[:, 0:38])
test_features = np.array(test_df.iloc[:, 0:38])

# Scaling the selected columns using StandardScaler
scaler = StandardScaler()
train_features[:, :5] = scaler.fit_transform(train_features[:, :5])
test_features[:, :5] = scaler.transform(test_features[:, :5])

# Loop over different noise intensity levels
intensity_steps = np.linspace(min_intensity, max_intensity, num=num_intensity_steps)
for intensity_factor in intensity_steps:
    for _ in range(num_simulations):
        # Generate salt and pepper noise and add to the data
        n_train = len(train_features)
        n_test = len(test_features)
        noise_train = np.random.choice([-1, 0, 1], size=n_train, p=[0.05, 0.9, 0.05])
        noise_test = np.random.choice([-1, 0, 1], size=n_test, p=[0.05, 0.9, 0.05])

        salt_pepper_noise_train = noise_train * np.sqrt(intensity_factor)
        salt_pepper_noise_test = noise_test * np.sqrt(intensity_factor)
        
        train_features_with_noise = train_features + salt_pepper_noise_train[:, np.newaxis]
        test_features_with_noise = test_features + salt_pepper_noise_test[:, np.newaxis]
        
        # Create and train the LightGBM classifier
        clf_lightgbm = lgb.LGBMClassifier(random_state=0)
        clf_lightgbm.fit(train_features_with_noise, train_labels)

        # Predict the response for the test dataset
        y_pred_lightgbm = clf_lightgbm.predict(test_features_with_noise)

        # Calculate evaluation metrics
        classification_rep = classification_report(test_labels, y_pred_lightgbm, zero_division=1)  # Set zero_division to 1
        f1_LGBM = f1_score(test_labels, y_pred_lightgbm, average='macro', zero_division=1)  # Set zero_division to 1
        mcc_LGBM = matthews_corrcoef(test_labels, y_pred_lightgbm)
        cm = confusion_matrix(test_labels, y_pred_lightgbm, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        specificity = tn / (tn + fp)
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        fpr = fp / (fp + tn)
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        roc_auc = roc_auc_score(test_labels, y_pred_lightgbm)
        average_precision = average_precision_score(test_labels, y_pred_lightgbm)
        balanced_accuracy = balanced_accuracy_score(test_labels, y_pred_lightgbm)
        kappa = cohen_kappa_score(test_labels, y_pred_lightgbm)
        hl = hamming_loss(test_labels, y_pred_lightgbm)
        logloss = log_loss(test_labels, y_pred_lightgbm)
        auprc = average_precision_score(test_labels, y_pred_lightgbm)

        # Store the results
        results_lightgbm.append({
            'intensity_factor': intensity_factor,
            'classification_report': classification_rep,
            'f1_score': f1_LGBM,
            'MCC': mcc_LGBM,
            'confusion_matrix': cm,
            'specificity': specificity,
            'precision': precision,
            'recall': recall,
            'fpr': fpr,
            'accuracy': accuracy,
            'ROC_AUC': roc_auc,
            'average_precision': average_precision,
            'balanced_accuracy': balanced_accuracy,
            'kappa': kappa,
            'hamming_loss': hl,
            'log_loss': log_loss,
            'auprc': auprc,
            'y_pred': y_pred_lightgbm
        })

for i, result in enumerate(results_lightgbm):
    print(f"Simulation {i+1}")
    print("Classification Report:")
    print("Intensity Factor:", result['intensity_factor'])
    print(result['classification_report'])
    print("F1 Score:", result['f1_score'])
    print("MCC:", result['MCC'])
    print("Confusion Matrix:")
    print(result['confusion_matrix'])
    print("Specificity:", result['specificity'])
    print("ROC AUC:", result['ROC_AUC'])
    print("Average Precision:", result['average_precision'])
    print("Balanced Accuracy:", result['balanced_accuracy'])
    print("Precision:", result['precision'])
    print("Recall:", result['recall'])
    print("False Positive Rate:", result['fpr'])
    print("Accuracy:", result['accuracy'])
    print("Cohen's Kappa:", result['kappa'])
    print("Hamming Loss:", result['hamming_loss'])
    print("Log Loss:", result['log_loss'])
    print("Area Under PR Curve:", result['auprc'])
    print()


####################

###### Uniform Noise
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    hamming_loss,
    log_loss,
)

# Add uniform noise to the specified columns
for column in columns_to_add_noise:
    noise = np.random.uniform(-1, 1, size=len(data))
    data[column] += noise

# Parameters
num_simulations = 1  # Adjust as needed
min_intensity = 0.0
max_intensity = 0.95
num_intensity_steps = 200
columns_to_scale = ['OverrideFeed', 'FeedRate', 'SpindleSpeed', 'IndoorGPSx', 'IndoorGPSy']

# Create an empty list to store results
results_lightgbm = []



# Splitting the data into training and test sets
train_df, test_df = train_test_split(df, test_size=0.2, random_state=0)

# Using numpy to create arrays of labels and features
train_labels = np.array(train_df['Production'])
test_labels = np.array(test_df['Production'])
train_features = np.array(train_df.iloc[:, 0:38])
test_features = np.array(test_df.iloc[:, 0:38])

# Scaling the selected columns using StandardScaler
scaler = StandardScaler()
train_features[:, :5] = scaler.fit_transform(train_features[:, :5])
test_features[:, :5] = scaler.transform(test_features[:, :5])

# Loop over different noise intensity levels
intensity_steps = np.linspace(min_intensity, max_intensity, num=num_intensity_steps)
for intensity_factor in intensity_steps:
    for _ in range(num_simulations):
        # Generate uniform noise and add to the data
        n_train = len(train_features)
        n_test = len(test_features)
        noise_train = np.random.uniform(-1, 1, size=n_train)
        noise_test = np.random.uniform(-1, 1, size=n_test)

        uniform_noise_train = noise_train * np.sqrt(intensity_factor)
        uniform_noise_test = noise_test * np.sqrt(intensity_factor)
        
        train_features_with_noise = train_features + uniform_noise_train[:, np.newaxis]
        test_features_with_noise = test_features + uniform_noise_test[:, np.newaxis]
        
        # Create and train the LightGBM classifier
        clf_lightgbm = lgb.LGBMClassifier(random_state=0)
        clf_lightgbm.fit(train_features_with_noise, train_labels)

        # Predict the response for the test dataset
        y_pred_lightgbm = clf_lightgbm.predict(test_features_with_noise)

        # Calculate evaluation metrics
        classification_rep = classification_report(test_labels, y_pred_lightgbm, zero_division=1)  # Set zero_division to 1
        f1_LGBM = f1_score(test_labels, y_pred_lightgbm, average='macro', zero_division=1)  # Set zero_division to 1
        mcc_LGBM = matthews_corrcoef(test_labels, y_pred_lightgbm)
        cm = confusion_matrix(test_labels, y_pred_lightgbm, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        specificity = tn / (tn + fp)
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        fpr = fp / (fp + tn)
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        roc_auc = roc_auc_score(test_labels, y_pred_lightgbm)
        average_precision = average_precision_score(test_labels, y_pred_lightgbm)
        balanced_accuracy = balanced_accuracy_score(test_labels, y_pred_lightgbm)
        kappa = cohen_kappa_score(test_labels, y_pred_lightgbm)
        hl = hamming_loss(test_labels, y_pred_lightgbm)
        logloss = log_loss(test_labels, y_pred_lightgbm)
        auprc = average_precision_score(test_labels, y_pred_lightgbm)

        # Store the results
        results_lightgbm.append({
            'intensity_factor': intensity_factor,
            'classification_report': classification_rep,
            'f1_score': f1_LGBM,
            'MCC': mcc_LGBM,
            'confusion_matrix': cm,
            'specificity': specificity,
            'precision': precision,
            'recall': recall,
            'fpr': fpr,
            'accuracy': accuracy,
            'ROC_AUC': roc_auc,
            'average_precision': average_precision,
            'balanced_accuracy': balanced_accuracy,
            'kappa': kappa,
            'hamming_loss': hl,
            'log_loss': logloss,
            'auprc': auprc,
            'y_pred': y_pred_lightgbm
        })

for i, result in enumerate(results_lightgbm):
    print(f"Simulation {i+1}")
    print("Classification Report:")
    print("Intensity Factor:", result['intensity_factor'])
    print(result['classification_report'])
    print("F1 Score:", result['f1_score'])
    print("MCC:", result['MCC'])
    print("Confusion Matrix:")
    print(result['confusion_matrix'])
    print("Specificity:", result['specificity'])
    print("ROC AUC:", result['ROC_AUC'])
    print("Average Precision:", result['average_precision'])
    print("Balanced Accuracy:", result['balanced_accuracy'])
    print("Precision:", result['precision'])
    print("Recall:", result['recall'])
    print("False Positive Rate:", result['fpr'])
    print("Accuracy:", result['accuracy'])
    print("Cohen's Kappa:", result['kappa'])
    print("Hamming Loss:", result['hamming_loss'])
    print("Log Loss:", result['log_loss'])
    print("Area Under PR Curve:", result['auprc'])
    print()



############
################  Brown Noise
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    hamming_loss,
    log_loss,
)

# Custom function to generate brown noise
def brown_noise(n):
    # Generate white noise
    white_noise = np.random.normal(0, 1, n)

    # Apply a cumulative sum to the white noise
    brown_noise = np.cumsum(white_noise)

    # Scale the brown noise to have zero mean and unit standard deviation
    brown_noise -= np.mean(brown_noise)
    brown_noise /= np.std(brown_noise)

    return brown_noise

# Add Brown noise to the specified columns
for column in columns_to_add_noise:
    # Generate Brown noise signal
    n = len(data)
    noise = brown_noise(n)

    # Scale the noise to match the column's range
    column_range = data[column].max() - data[column].min()
    scaled_noise = column_range * (noise - np.min(noise)) / (np.max(noise) - np.min(noise))

    # Add the noise to the column
    data[column] += scaled_noise

# Parameters
num_simulations = 1  # Adjust as needed
min_intensity = 0.0
max_intensity = 0.95
num_intensity_steps = 200
columns_to_scale = ['OverrideFeed', 'FeedRate', 'SpindleSpeed', 'IndoorGPSx', 'IndoorGPSy']

# Create an empty list to store results
results_lightgbm = []


# Splitting the data into training and test sets
train_df, test_df = train_test_split(df, test_size=0.2, random_state=0)

# Using numpy to create arrays of labels and features
train_labels = np.array(train_df['Production'])
test_labels = np.array(test_df['Production'])
train_features = np.array(train_df.iloc[:, 0:38])
test_features = np.array(test_df.iloc[:, 0:38])

# Scaling the selected columns using StandardScaler
scaler = StandardScaler()
train_features[:, :5] = scaler.fit_transform(train_features[:, :5])
test_features[:, :5] = scaler.transform(test_features[:, :5])

# Loop over different noise intensity levels
intensity_steps = np.linspace(min_intensity, max_intensity, num=num_intensity_steps)
for intensity_factor in intensity_steps:
    for _ in range(num_simulations):
        # Generate brown noise and add to the data
        n_train = len(train_features)
        n_test = len(test_features)
        noise_train = brown_noise(n_train)
        noise_test = brown_noise(n_test)

        colored_noise_train = noise_train * np.sqrt(intensity_factor)
        colored_noise_test = noise_test * np.sqrt(intensity_factor)
        
        train_features_with_noise = train_features + colored_noise_train[:, np.newaxis]
        test_features_with_noise = test_features + colored_noise_test[:, np.newaxis]
        
        # Create and train the LightGBM classifier
        clf_lightgbm = lgb.LGBMClassifier(random_state=0)
        clf_lightgbm.fit(train_features_with_noise, train_labels)

        # Predict the response for the test dataset
        y_pred_lightgbm = clf_lightgbm.predict(test_features_with_noise)

        # Calculate evaluation metrics
        classification_rep = classification_report(test_labels, y_pred_lightgbm, zero_division=1)  # Set zero_division to 1
        f1_LGBM = f1_score(test_labels, y_pred_lightgbm, average='macro', zero_division=1)  # Set zero_division to 1
        mcc_LGBM = matthews_corrcoef(test_labels, y_pred_lightgbm)
        cm = confusion_matrix(test_labels, y_pred_lightgbm, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        specificity = tn / (tn + fp)
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        fpr = fp / (fp + tn)
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        roc_auc = roc_auc_score(test_labels, y_pred_lightgbm)
        average_precision = average_precision_score(test_labels, y_pred_lightgbm)
        balanced_accuracy = balanced_accuracy_score(test_labels, y_pred_lightgbm)
        kappa = cohen_kappa_score(test_labels, y_pred_lightgbm)
        hl = hamming_loss(test_labels, y_pred_lightgbm)
        logloss = log_loss(test_labels, y_pred_lightgbm)
        auprc = average_precision_score(test_labels, y_pred_lightgbm)

        # Store the results
        results_lightgbm.append({
            'intensity_factor': intensity_factor,
            'classification_report': classification_rep,
            'f1_score': f1_LGBM,
            'MCC': mcc_LGBM,
            'confusion_matrix': cm,
            'specificity': specificity,
            'precision': precision,
            'recall': recall,
            'fpr': fpr,
            'accuracy': accuracy,
            'ROC_AUC': roc_auc,
            'average_precision': average_precision,
            'balanced_accuracy': balanced_accuracy,
            'kappa': kappa,
            'hamming_loss': hl,
            'log_loss': logloss,
            'auprc': auprc,
            'y_pred': y_pred_lightgbm
        })

for i, result in enumerate(results_lightgbm):
    print(f"Simulation {i+1}")
    print("Classification Report:")
    print("Intensity Factor:", result['intensity_factor'])
    print(result['classification_report'])
    print("F1 Score:", result['f1_score'])
    print("MCC:", result['MCC'])
    print("Confusion Matrix:")
    print(result['confusion_matrix'])
    print("Specificity:", result['specificity'])
    print("ROC AUC:", result['ROC_AUC'])
    print("Average Precision:", result['average_precision'])
    print("Balanced Accuracy:", result['balanced_accuracy'])
    print("Precision:", result['precision'])
    print("Recall:", result['recall'])
    print("False Positive Rate:", result['fpr'])
    print("Accuracy:", result['accuracy'])
    print("Cohen's Kappa:", result['kappa'])
    print("Hamming Loss:", result['hamming_loss'])
    print("Log Loss:", result['log_loss'])
    print("Area Under PR Curve:", result['auprc'])
    print()


##########################
#######  1/f Pink Noise
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    hamming_loss,
    log_loss,
)
from scipy.signal import butter, lfilter
sampling_rate = 1000
def pink_noise(n):
    # Generate white noise
    white_noise = np.random.normal(0, 1, n)

    # Apply a one-dimensional IIR filter (Butterworth filter)
    b, a = butter(1, 1/50, output='ba')
    pink_noise = lfilter(b, a, white_noise)

    # Scale the pink noise to have zero mean and unit standard deviation
    pink_noise -= np.mean(pink_noise)
    pink_noise /= np.std(pink_noise)

    return pink_noise

# Add 1/f (pink) noise to the specified columns
for column in columns_to_add_noise:
    # Generate pink noise signal
    n = len(data)
    noise = pink_noise(n)

    # Scale the noise to match the column's range
    column_range = data[column].max() - data[column].min()
    scaled_noise = column_range * (noise - np.min(noise)) / (np.max(noise) - np.min(noise))

    # Add the noise to the column
    data[column] += scaled_noise

# Parameters
num_simulations = 1  
min_intensity = 0.0
max_intensity = 0.95
num_intensity_steps = 200
columns_to_scale = ['OverrideFeed', 'FeedRate', 'SpindleSpeed', 'IndoorGPSx', 'IndoorGPSy']

# Create an empty list to store results
results_lightgbm = []

# Splitting the data into training and test sets
train_df, test_df = train_test_split(df, test_size=0.2, random_state=0)

# Using numpy to create arrays of labels and features
train_labels = np.array(train_df['Production'])
test_labels = np.array(test_df['Production'])
train_features = np.array(train_df.iloc[:, 0:38])
test_features = np.array(test_df.iloc[:, 0:38])

# Scaling the selected columns using StandardScaler
scaler = StandardScaler()
train_features[:, :5] = scaler.fit_transform(train_features[:, :5])
test_features[:, :5] = scaler.transform(test_features[:, :5])

# Loop over different noise intensity levels
intensity_steps = np.linspace(min_intensity, max_intensity, num=num_intensity_steps)
for intensity_factor in intensity_steps:
    for _ in range(num_simulations):
        # Generate pink noise and add to the data
        n_train = len(train_features)
        n_test = len(test_features)
        noise_train = pink_noise(n_train)
        noise_test = pink_noise(n_test)

        b, a = butter(1, 1/50, output='ba')
        colored_noise_train = lfilter(b, a, noise_train) * np.sqrt(intensity_factor * sampling_rate)
        colored_noise_test = lfilter(b, a, noise_test) * np.sqrt(intensity_factor * sampling_rate)
        
        train_features_with_noise = train_features + colored_noise_train[:, np.newaxis]
        test_features_with_noise = test_features + colored_noise_test[:, np.newaxis]
        
        # Create and train the LightGBM classifier
        clf_lightgbm = lgb.LGBMClassifier(random_state=0)
        clf_lightgbm.fit(train_features_with_noise, train_labels)

        # Predict the response for the test dataset
        y_pred_lightgbm = clf_lightgbm.predict(test_features_with_noise)

        # Calculate evaluation metrics
        classification_rep = classification_report(test_labels, y_pred_lightgbm, zero_division=1)  # Set zero_division to 1
        f1_LGBM = f1_score(test_labels, y_pred_lightgbm, average='macro', zero_division=1)  # Set zero_division to 1
        mcc_LGBM = matthews_corrcoef(test_labels, y_pred_lightgbm)
        cm = confusion_matrix(test_labels, y_pred_lightgbm, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        specificity = tn / (tn + fp)
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        fpr = fp / (fp + tn)
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        roc_auc = roc_auc_score(test_labels, y_pred_lightgbm)
        average_precision = average_precision_score(test_labels, y_pred_lightgbm)
        balanced_accuracy = balanced_accuracy_score(test_labels, y_pred_lightgbm)
        kappa = cohen_kappa_score(test_labels, y_pred_lightgbm)
        hl = hamming_loss(test_labels, y_pred_lightgbm)
        logloss = log_loss(test_labels, y_pred_lightgbm)
        auprc = average_precision_score(test_labels, y_pred_lightgbm)

        # Store the results
        results_lightgbm.append({
            'intensity_factor': intensity_factor,
            'classification_report': classification_rep,
            'f1_score': f1_LGBM,
            'MCC': mcc_LGBM,
            'confusion_matrix': cm,
            'specificity': specificity,
            'precision': precision,
            'recall': recall,
            'fpr': fpr,
            'accuracy': accuracy,
            'ROC_AUC': roc_auc,
            'average_precision': average_precision,
            'balanced_accuracy': balanced_accuracy,
            'kappa': kappa,
            'hamming_loss': hl,
            'log_loss': logloss,
            'auprc': auprc,
            'y_pred': y_pred_lightgbm
        })

for i, result in enumerate(results_lightgbm):
    print(f"Simulation {i+1}")
    print("Classification Report:")
    print("Intensity Factor:", result['intensity_factor'])
    print(result['classification_report'])
    print("F1 Score:", result['f1_score'])
    print("MCC:", result['MCC'])
    print("Confusion Matrix:")
    print(result['confusion_matrix'])
    print("Specificity:", result['specificity'])
    print("ROC AUC:", result['ROC_AUC'])
    print("Average Precision:", result['average_precision'])
    print("Balanced Accuracy:", result['balanced_accuracy'])
    print("Precision:", result['precision'])
    print("Recall:", result['recall'])
    print("False Positive Rate:", result['fpr'])
    print("Accuracy:", result['accuracy'])
    print("Cohen's Kappa:", result['kappa'])
    print("Hamming Loss:", result['hamming_loss'])
    print("Log Loss:", result['log_loss'])
    print("Area Under PR Curve:", result['auprc'])
    print()




###################################
######  Colored Noise
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, f1_score, matthews_corrcoef, confusion_matrix, roc_auc_score, average_precision_score, balanced_accuracy_score, cohen_kappa_score, hamming_loss, log_loss

# Generate colored noise function
def generate_colored_noise(length, power):
    noise = np.random.normal(0, 1, size=length)
    b, a = [1], [1, -power]
    colored_noise = lfilter(b, a, noise) * np.sqrt(power * sampling_rate)
    return colored_noise

# Parameters

columns_to_scale = ['OverrideFeed', 'FeedRate', 'SpindleSpeed', 'IndoorGPSx', 'IndoorGPSy']
power = 1.0  # Adjust the power parameter for colored noise
sampling_rate = 1000  # Adjust as needed



from scipy.signal import lfilter


#############################
##Colored Noise
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    hamming_loss,
    log_loss,
)

# Add colored noise to the specified columns
for column in columns_to_add_noise:
    noise = np.random.normal(0, 1, size=len(data))
    b, a = [1], [1, -power]
    colored_noise = lfilter(b, a, noise) * np.sqrt(power * sampling_rate)
    data[column] += colored_noise

# Parameters
num_simulations = 1  # Adjust as needed
min_intensity = 0.0
max_intensity = 0.95
num_intensity_steps = 200
columns_to_scale = ['OverrideFeed', 'FeedRate', 'SpindleSpeed', 'IndoorGPSx', 'IndoorGPSy']

# Create an empty list to store results
results_lightgbm = []



# Splitting the data into training and test sets
train_df, test_df = train_test_split(df, test_size=0.2, random_state=0)

# Using numpy to create arrays of labels and features
train_labels = np.array(train_df['Production'])
test_labels = np.array(test_df['Production'])
train_features = np.array(train_df.iloc[:, 0:38])
test_features = np.array(test_df.iloc[:, 0:38])

# Scaling the selected columns using StandardScaler
scaler = StandardScaler()
train_features[:, :5] = scaler.fit_transform(train_features[:, :5])
test_features[:, :5] = scaler.transform(test_features[:, :5])

# Loop over different noise intensity levels
intensity_steps = np.linspace(min_intensity, max_intensity, num=num_intensity_steps)
for intensity_factor in intensity_steps:
    for _ in range(num_simulations):
        # Generate colored noise and add to the data
        noise_train = np.random.normal(0, 1, size=len(train_features))
        noise_test = np.random.normal(0, 1, size=len(test_features))
        
        b, a = [1], [1, -intensity_factor]
        colored_noise_train = lfilter(b, a, noise_train) * np.sqrt(intensity_factor * sampling_rate)
        colored_noise_test = lfilter(b, a, noise_test) * np.sqrt(intensity_factor * sampling_rate)
        
        train_features_with_noise = train_features + colored_noise_train[:, np.newaxis]
        test_features_with_noise = test_features + colored_noise_test[:, np.newaxis]
        
        # Create and train the LightGBM classifier
        clf_lightgbm = lgb.LGBMClassifier(random_state=0)
        clf_lightgbm.fit(train_features_with_noise, train_labels)

        # Predict the response for the test dataset
        y_pred_lightgbm = clf_lightgbm.predict(test_features_with_noise)

        # Calculate evaluation metrics
        classification_rep = classification_report(test_labels, y_pred_lightgbm, zero_division=1)  # Set zero_division to 1
        f1_LGBM = f1_score(test_labels, y_pred_lightgbm, average='macro', zero_division=1)  # Set zero_division to 1
        mcc_LGBM = matthews_corrcoef(test_labels, y_pred_lightgbm)
        cm = confusion_matrix(test_labels, y_pred_lightgbm, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        specificity = tn / (tn + fp)
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        fpr = fp / (fp + tn)
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        roc_auc = roc_auc_score(test_labels, y_pred_lightgbm)
        average_precision = average_precision_score(test_labels, y_pred_lightgbm)
        balanced_accuracy = balanced_accuracy_score(test_labels, y_pred_lightgbm)
        kappa = cohen_kappa_score(test_labels, y_pred_lightgbm)
        hl = hamming_loss(test_labels, y_pred_lightgbm)
        logloss = log_loss(test_labels, y_pred_lightgbm)
        auprc = average_precision_score(test_labels, y_pred_lightgbm)

        # Store the results
        results_lightgbm.append({
            'intensity_factor': intensity_factor,
            'classification_report': classification_rep,
            'f1_score': f1_LGBM,
            'MCC': mcc_LGBM,
            'confusion_matrix': cm,
            'specificity': specificity,
            'precision': precision,
            'recall': recall,
            'fpr': fpr,
            'accuracy': accuracy,
            'ROC_AUC': roc_auc,
            'average_precision': average_precision,
            'balanced_accuracy': balanced_accuracy,
            'kappa': kappa,
            'hamming_loss': hl,
            'log_loss': logloss,
            'auprc': auprc,
            'y_pred': y_pred_lightgbm
        })

for i, result in enumerate(results_lightgbm):
    print(f"Simulation {i+1}")
    print("Classification Report:")
    print("Intensity Factor:", result['intensity_factor'])
    print(result['classification_report'])
    print("F1 Score:", result['f1_score'])
    print("MCC:", result['MCC'])
    print("Confusion Matrix:")
    print(result['confusion_matrix'])
    print("Specificity:", result['specificity'])
    print("ROC AUC:", result['ROC_AUC'])
    print("Average Precision:", result['average_precision'])
    print("Balanced Accuracy:", result['balanced_accuracy'])
    print("Precision:", result['precision'])
    print("Recall:", result['recall'])
    print("False Positive Rate:", result['fpr'])
    print("Accuracy:", result['accuracy'])
    print("Cohen's Kappa:", result['kappa'])
    print("Hamming Loss:", result['hamming_loss'])
    print("Log Loss:", result['log_loss'])
    print("Area Under PR Curve:", result['auprc'])
    print()


###############################
##LightGBM
## Flicker Noise
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, f1_score, matthews_corrcoef, confusion_matrix, roc_auc_score, balanced_accuracy_score, cohen_kappa_score, hamming_loss, log_loss, average_precision_score
import lightgbm as lgb

# Parameters
num_simulations = 1  
min_intensity = 0.0
max_intensity = 0.95
num_intensity_steps = 200
columns_to_scale = ['OverrideFeed', 'FeedRate', 'SpindleSpeed', 'IndoorGPSx', 'IndoorGPSy']

# Create an empty list to store results
results_lightgbm = []



# Splitting the data into training and test sets
train_df, test_df = train_test_split(df, test_size=0.2, random_state=0)

# Using numpy to create arrays of labels and features
train_labels = np.array(train_df['Production'])
test_labels = np.array(test_df['Production'])
train_features = np.array(train_df.iloc[:, 0:38])
test_features = np.array(test_df.iloc[:, 0:38])

# Scaling the selected columns using StandardScaler
scaler = StandardScaler()
train_features[:, :5] = scaler.fit_transform(train_features[:, :5])
test_features[:, :5] = scaler.transform(test_features[:, :5])

# Loop over different noise intensity levels
intensity_steps = np.linspace(min_intensity, max_intensity, num=num_intensity_steps)
for intensity_factor in intensity_steps:
    for _ in range(num_simulations):
        # Generate flicker noise and add to the data
        noise_train = np.random.normal(0, 1, size=len(train_features))
        noise_test = np.random.normal(0, 1, size=len(test_features))
        
        flicker_noise_train = np.cumsum(noise_train * intensity_factor)
        flicker_noise_test = np.cumsum(noise_test * intensity_factor)
        
        train_features_with_noise = train_features + flicker_noise_train[:, np.newaxis]
        test_features_with_noise = test_features + flicker_noise_test[:, np.newaxis]
        
        # Create and train the LightGBM classifier
        clf_lightgbm = lgb.LGBMClassifier(random_state=0)
        clf_lightgbm.fit(train_features_with_noise, train_labels)

        # Predict the response for the test dataset
        y_pred_lightgbm = clf_lightgbm.predict(test_features_with_noise)

        # Calculate evaluation metrics
        classification_rep = classification_report(test_labels, y_pred_lightgbm)
        f1_LGBM = f1_score(test_labels, y_pred_lightgbm, average='macro')
        mcc_LGBM = matthews_corrcoef(test_labels, y_pred_lightgbm)
        cm = confusion_matrix(test_labels, y_pred_lightgbm, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        specificity = tn / (tn + fp)
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        fpr = fp / (fp + tn)
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        roc_auc = roc_auc_score(test_labels, y_pred_lightgbm)
        average_precision = average_precision_score(test_labels, y_pred_lightgbm)
        balanced_accuracy = balanced_accuracy_score(test_labels, y_pred_lightgbm)
        kappa = cohen_kappa_score(test_labels, y_pred_lightgbm)
        hl = hamming_loss(test_labels, y_pred_lightgbm)
        logloss = log_loss(test_labels, y_pred_lightgbm)
        auprc = average_precision_score(test_labels, y_pred_lightgbm)

        # Store the results
        results_lightgbm.append({
            'intensity_factor': intensity_factor,
            'classification_report': classification_rep,
            'f1_score': f1_LGBM,
            'MCC': mcc_LGBM,
            'confusion_matrix': cm,
            'specificity': specificity,
            'precision': precision,
            'recall': recall,
            'fpr': fpr,
            'accuracy': accuracy,
            'ROC_AUC': roc_auc,
            'average_precision': average_precision,
            'balanced_accuracy': balanced_accuracy,
            'kappa': kappa,
            'hamming_loss': hl,
            'log_loss': logloss,
            'auprc': auprc,
            'y_pred': y_pred_lightgbm
            })


for i, result in enumerate(results_lightgbm):
    print(f"Simulation {i+1}")
    print("Classification Report:")
    print("Intensity Factor:", result['intensity_factor'])
    print(result['classification_report'])
    print("F1 Score:", result['f1_score'])
    print("MCC:", result['MCC'])
    print("Confusion Matrix:")
    print(result['confusion_matrix'])
    print("Specificity:", result['specificity'])
    print("ROC AUC:", result['ROC_AUC'])
    print("Average Precision:", result['average_precision'])
    print("Balanced Accuracy:", result['balanced_accuracy'])
    print("Precision:", result['precision'])
    print("Recall:", result['recall'])
    print("False Positive Rate:", result['fpr'])
    print("Accuracy:", result['accuracy'])
    print("Cohen's Kappa:", result['kappa'])
    print("Hamming Loss:", result['hamming_loss'])
    print("Log Loss:", result['log_loss'])
    print("Area Under PR Curve:", result['auprc'])
    print()

##################################################################
####LightGBM
##Gaussian Noise
# Parameters
num_simulations = 1  # Adjust as needed
min_intensity = 0.0
max_intensity = 0.95
num_intensity_steps = 200
columns_to_scale = ['OverrideFeed', 'FeedRate', 'SpindleSpeed', 'IndoorGPSx', 'IndoorGPSy']

# Create an empty list to store results
results_lightgbm = []

# Splitting the data into training and test sets
train_df, test_df = train_test_split(df, test_size=0.2, random_state=0)

# Using numpy to create arrays of labels and features
train_labels = np.array(train_df['Production'])
test_labels = np.array(test_df['Production'])
train_features = np.array(train_df.iloc[:, 0:38])
test_features = np.array(test_df.iloc[:, 0:38])

# Scaling the selected columns using StandardScaler
scaler = StandardScaler()
train_features[:, :5] = scaler.fit_transform(train_features[:, :5])
test_features[:, :5] = scaler.transform(test_features[:, :5])

# Loop over different noise intensity levels
intensity_steps = np.linspace(min_intensity, max_intensity, num=num_intensity_steps)
for intensity_factor in intensity_steps:
    for _ in range(num_simulations):
        # Generate Gaussian noise and add to the data
        noise_train = np.random.normal(0, intensity_factor, size=train_features.shape)
        noise_test = np.random.normal(0, intensity_factor, size=test_features.shape)
        
        train_features_with_noise = train_features + noise_train
        test_features_with_noise = test_features + noise_test

        # Create and train the LightGBM classifier
        clf_lightgbm = lgb.LGBMClassifier(random_state=0)
        clf_lightgbm.fit(train_features_with_noise, train_labels)

        # Predict the response for the test dataset
        y_pred_lightgbm = clf_lightgbm.predict(test_features_with_noise)

 # Calculate evaluation metrics
        classification_rep = classification_report(test_labels, y_pred_lightgbm)
        f1_LGBM = f1_score(test_labels, y_pred_lightgbm, average='macro')
        mcc_LGBM = matthews_corrcoef(test_labels, y_pred_lightgbm)
        cm = confusion_matrix(test_labels, y_pred_lightgbm, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        specificity = tn / (tn + fp)
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        fpr = fp / (fp + tn)
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        roc_auc = roc_auc_score(test_labels, y_pred_lightgbm)
        average_precision = average_precision_score(test_labels, y_pred_lightgbm)
        balanced_accuracy = balanced_accuracy_score(test_labels, y_pred_lightgbm)
        kappa = cohen_kappa_score(test_labels, y_pred_lightgbm)
        hl = hamming_loss(test_labels, y_pred_lightgbm)
        logloss = log_loss(test_labels, y_pred_lightgbm)
        auprc = average_precision_score(test_labels, y_pred_lightgbm)

        # Store the results
        results_lightgbm.append({
            'intensity_factor': intensity_factor,
            'classification_report': classification_rep,
            'f1_score': f1_LGBM,
            'MCC': mcc_LGBM,
            'confusion_matrix': cm,
            'specificity': specificity,
            'precision': precision,
            'recall': recall,
            'fpr': fpr,
            'accuracy': accuracy,
            'ROC_AUC': roc_auc,
            'average_precision': average_precision,
            'balanced_accuracy': balanced_accuracy,
            'kappa': kappa,
            'hamming_loss': hl,
            'log_loss': logloss,
            'auprc': auprc,
            'y_pred': y_pred_lightgbm
            })


for i, result in enumerate(results_lightgbm):
    print(f"Simulation {i+1}")
    print("Classification Report:")
    print("Intensity Factor:", result['intensity_factor'])
    print(result['classification_report'])
    print("F1 Score:", result['f1_score'])
    print("MCC:", result['MCC'])
    print("Confusion Matrix:")
    print(result['confusion_matrix'])
    print("Specificity:", result['specificity'])
    print("ROC AUC:", result['ROC_AUC'])
    print("Average Precision:", result['average_precision'])
    print("Balanced Accuracy:", result['balanced_accuracy'])
    print("Precision:", result['precision'])
    print("Recall:", result['recall'])
    print("False Positive Rate:", result['fpr'])
    print("Accuracy:", result['accuracy'])
    print("Cohen's Kappa:", result['kappa'])
    print("Hamming Loss:", result['hamming_loss'])
    print("Log Loss:", result['log_loss'])
    print("Area Under PR Curve:", result['auprc'])
    print()

#################################################################

#################################################################

########################################################
from sklearn.ensemble import RandomForestClassifier  # Import the RandomForestClassifier

##Random Forest Classifier
#Gaussian Noise

num_simulations = 1  # Adjust as needed
min_intensity = 0.0
max_intensity = 0.95
num_intensity_steps = 200
columns_to_scale = ['OverrideFeed', 'FeedRate', 'SpindleSpeed', 'IndoorGPSx', 'IndoorGPSy']

# Create an empty list to store results
results_rf = []


# Splitting the data into training and test sets
train_df, test_df = train_test_split(df, test_size=0.2, random_state=0)

# Using numpy to create arrays of labels and features
train_labels = np.array(train_df['Production'])
test_labels = np.array(test_df['Production'])
train_features = np.array(train_df.iloc[:, 0:38])
test_features = np.array(test_df.iloc[:, 0:38])

# Scaling the selected columns using StandardScaler
scaler = StandardScaler()
train_features[:, :5] = scaler.fit_transform(train_features[:, :5])
test_features[:, :5] = scaler.transform(test_features[:, :5])

# Loop over different noise intensity levels
intensity_steps = np.linspace(min_intensity, max_intensity, num=num_intensity_steps)
for intensity_factor in intensity_steps:
    for _ in range(num_simulations):
        # Generate Gaussian noise and add to the data
        noise_train = np.random.normal(0, intensity_factor, size=train_features.shape)
        noise_test = np.random.normal(0, intensity_factor, size=test_features.shape)
        
        train_features_with_noise = train_features + noise_train
        test_features_with_noise = test_features + noise_test

        # Create and train the Random Forest classifier
        clf_rf = RandomForestClassifier(random_state=0)  # Instantiate the RandomForestClassifier
        clf_rf.fit(train_features_with_noise, train_labels)

        # Predict the response for the test dataset
        y_pred_rf = clf_rf.predict(test_features_with_noise)

        # Calculate evaluation metrics
        classification_rep = classification_report(test_labels, y_pred_rf)
        f1_RF = f1_score(test_labels, y_pred_rf, average='macro')
        mcc_RF = matthews_corrcoef(test_labels, y_pred_rf)
        cm = confusion_matrix(test_labels, y_pred_rf, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        specificity = tn / (tn + fp)
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        fpr = fp / (fp + tn)
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        roc_auc = roc_auc_score(test_labels, y_pred_rf)
        average_precision = average_precision_score(test_labels, y_pred_rf)
        balanced_accuracy = balanced_accuracy_score(test_labels, y_pred_rf)
        kappa = cohen_kappa_score(test_labels, y_pred_rf)
        hl = hamming_loss(test_labels, y_pred_rf)
        logloss = log_loss(test_labels, y_pred_rf)
        auprc = average_precision_score(test_labels, y_pred_rf)

        # Store the results
        results_rf.append({
            'intensity_factor': intensity_factor,
            'classification_report': classification_rep,
            'f1_score': f1_RF,
            'MCC': mcc_RF,
            'confusion_matrix': cm,
            'specificity': specificity,
            'precision': precision,
            'recall': recall,
            'fpr': fpr,
            'accuracy': accuracy,
            'ROC_AUC': roc_auc,
            'average_precision': average_precision,
            'balanced_accuracy': balanced_accuracy,
            'kappa': kappa,
            'hamming_loss': hl,
            'log_loss': logloss,
            'auprc': auprc,
            'y_pred': y_pred_rf
        })


####################################################





