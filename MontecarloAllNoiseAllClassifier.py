# Importing libraries
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

## Reading Data
data = pd.read_csv("Data/Data_Detecting_ChangewithoutGPS.csv", sep= ',',header=0, index_col=0)
data.head()
data.info()


# Columns to add noise to
 columns_to_add_noise = ['OverrideFeed', 'FeedRate', 'SpindleSpeed', 'IndoorGPSx', 'IndoorGPSy']
# Add Gaussian white noise to the specified columns
for column in columns_to_add_noise:
    noise = np.random.normal(0, 1, size=len(data))
    data[column] += noise

# Display the updated dataframe
print(data.head())

# Add flicker noise to the specified columns
for column in columns_to_add_noise:
    noise = np.random.normal(0, 1, size=len(data))
    flicker_noise = np.cumsum(noise)  # Apply cumulative sum to create flicker noise
    data[column] += flicker_noise

# Display the updated dataframe
print(data.head())


# Add colored noise 
## we use the lfilter function from scipy.signal to apply 
# a first-order recursive filter to the Gaussian noise array. 
# The filter coefficients b and a are set to [1] and [1, -power] 
# respectively, where power controls the strength of the noise.
# By multiplying the filtered noise with np.sqrt(power * sampling_rate),
# we scale the noise appropriately to match the desired power spectral density.

from scipy.signal import lfilter

# Define the noise parameters
sampling_rate = 1  # Adjust as per your data's sampling rate
power = 0.8  # Adjust the power to control the strength of the noise

# Add colored noise to the specified columns
for column in columns_to_add_noise:
    noise = np.random.normal(0, 1, size=len(data))
    b, a = [1], [1, -power]
    colored_noise = lfilter(b, a, noise) * np.sqrt(power * sampling_rate)
    data[column] += colored_noise

# Display the updated dataframe
print(data.head())


### Add 1/f Noise
from scipy import signal

# Custom function to generate pink noise
def pink_noise(n):
    # Generate white noise
    white_noise = np.random.normal(0, 1, n)

    # Apply a one-dimensional IIR filter (Butterworth filter)
    b, a = signal.butter(1, 1/50, output='ba')
    pink_noise = signal.lfilter(b, a, white_noise)

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

# Display the updated dataframe
print(data.head())


## Add Brown Noise

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

# Display the updated dataframe
print(data.head())

 
 ### Add uniform noise
# Add uniform noise to the specified columns
for column in columns_to_add_noise:
    noise = np.random.uniform(-1, 1, size=len(data))
    data[column] += noise

# Display the updated dataframe
print(data.head())



# Add salt-and-pepper noise to the specified columns
for column in columns_to_add_noise:
    noise = np.random.choice([-1, 0, 1], size=len(data), p=[0.05, 0.9, 0.05])
    data[column] += noise

# Display the updated dataframe
print(data.head())


# Add impulse noise to the specified columns 10% noise level
for column in columns_to_add_noise:
    noise = np.random.choice([-1, 1], size=len(data), p=[0.9, 0.1])
    data[column] += noise

# Display the updated dataframe
print(data.head())


# Add multiplicative noise to the specified columns
for column in columns_to_add_noise:
    noise = np.random.uniform(0.9, 1.1, size=len(data))
    data[column] *= noise

# Display the updated dataframe
print(data.head())


# Add periodic noise to the specified columns
for column in columns_to_add_noise:
    noise = np.sin(np.arange(len(data)) / 10)  # Adjust the frequency of the periodic noise as needed
    data[column] += noise

# Display the updated dataframe
print(data.head())



######################################################


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



###############################################
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, f1_score, matthews_corrcoef, confusion_matrix, roc_auc_score, balanced_accuracy_score, cohen_kappa_score, hamming_loss, log_loss, average_precision_score
import lightgbm as lgb #for light gradient boosting machine
# Define the number of simulations LightGBM
num_simulations = 50

# Placeholder for storing results
results = []

for _ in range(num_simulations):
    # Generate random feature inputs based on normal distribution
    #random_features = np.random.normal(loc=0, scale=1, size=(len(df), 10))
    
    # Split the data into training and test sets
    # Splitting the data into training and test sets
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=None)

    # Using numpy to create arrays of lables and features
    train_labels = np.array(train_df['Production'])
    test_labels = np.array(test_df['Production'])
    train_features = np.array(train_df.iloc[:, 0:38])
    test_features = np.array(test_df.iloc[:, 0:38])

    # Selecting the columns to scale
    columns_to_scale = ['OverrideFeed', 'FeedRate', 'SpindleSpeed', 'IndoorGPSx', 'IndoorGPSy']

    # Scaling the selected columns using StandardScaler
    scaler = StandardScaler()
    train_features[:, :5] = scaler.fit_transform(train_features[:, :5])
    test_features[:, :5] = scaler.transform(test_features[:, :5])
    
    clf_lightgbm = lgb.LGBMClassifier(random_state=0)

    # Train the LightGBM Classifier
    clf_lightgbm.fit(train_features, train_labels)

    # Predict the response for the test dataset
    y_pred_lightgbm = clf_lightgbm.predict(test_features)
    
    # Calculate evaluation metrics
    classification_rep = classification_report(test_labels, y_pred_lightgbm)
    f1_RF = f1_score(test_labels, y_pred_lightgbm, average='macro')
    mcc_RF = matthews_corrcoef(test_labels, y_pred_lightgbm)
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
    results.append({
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
        'y_pred': y_pred_lightgbm
    })


# Print or analyze the results from each simulation
for i, result in enumerate(results):
    print(f"Simulation {i+1}")
    print("Classification Report:")
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



##########################################################
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, f1_score, matthews_corrcoef, confusion_matrix, roc_auc_score, balanced_accuracy_score, cohen_kappa_score, hamming_loss, log_loss, average_precision_score
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingClassifier

# Set the number of simulations
num_simulations = 50

# Placeholder for storing results
results = []

for _ in range(num_simulations):
    # Split the data into training and test sets
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=None)

    # Using numpy to create arrays of labels and features
    train_labels = np.array(train_df['Production'])
    test_labels = np.array(test_df['Production'])
    train_features = np.array(train_df.iloc[:, 0:38])
    test_features = np.array(test_df.iloc[:, 0:38])

    # Selecting the columns to scale
    columns_to_scale = ['OverrideFeed', 'FeedRate', 'SpindleSpeed', 'IndoorGPSx', 'IndoorGPSy']

    # Scaling the selected columns using StandardScaler
    scaler = StandardScaler()
    train_features[:, :5] = scaler.fit_transform(train_features[:, :5])
    test_features[:, :5] = scaler.transform(test_features[:, :5])

    # Train the HistGradientBoosting Classifier
    clf_hist = HistGradientBoostingClassifier(random_state=0)
    clf_hist.fit(train_features, train_labels)

    # Predict the response for the test dataset
    y_pred_hist = clf_hist.predict(test_features)

    # Calculate evaluation metrics
    classification_rep = classification_report(test_labels, y_pred_hist)
    f1_hist = f1_score(test_labels, y_pred_hist, average='macro')
    mcc_hist = matthews_corrcoef(test_labels, y_pred_hist)
    cm = confusion_matrix(test_labels, y_pred_hist, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    fpr = fp / (fp + tn)
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    roc_auc = roc_auc_score(test_labels, y_pred_hist)
    average_precision = average_precision_score(test_labels, y_pred_hist)
    balanced_accuracy = balanced_accuracy_score(test_labels, y_pred_hist)
    kappa = cohen_kappa_score(test_labels, y_pred_hist)
    hl = hamming_loss(test_labels, y_pred_hist)
    logloss = log_loss(test_labels, y_pred_hist)
    auprc = average_precision_score(test_labels, y_pred_hist)

    # Store the results
    results.append({
        'classification_report': classification_rep,
        'f1_score': f1_hist,
        'MCC': mcc_hist,
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
        'y_pred': y_pred_hist
    })

    
    # Print or analyze the results from each simulation
for i, result in enumerate(results):
    print(f"Simulation {i+1}")
    print("Classification Report:")
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
    

###############################################
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, f1_score, matthews_corrcoef, confusion_matrix, roc_auc_score, balanced_accuracy_score, cohen_kappa_score, hamming_loss, log_loss, average_precision_score
from xgboost import XGBClassifier

# Set the number of simulations
num_simulations = 50

# Placeholder for storing results
results = []

for _ in range(num_simulations):
    # Split the data into training and test sets
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=None)

    # Using numpy to create arrays of labels and features
    train_labels = np.array(train_df['Production'])
    test_labels = np.array(test_df['Production'])
    train_features = np.array(train_df.iloc[:, 0:38])
    test_features = np.array(test_df.iloc[:, 0:38])

    # Selecting the columns to scale
    columns_to_scale = ['OverrideFeed', 'FeedRate', 'SpindleSpeed', 'IndoorGPSx', 'IndoorGPSy']

    # Scaling the selected columns using StandardScaler
    scaler = StandardScaler()
    train_features[:, :5] = scaler.fit_transform(train_features[:, :5])
    test_features[:, :5] = scaler.transform(test_features[:, :5])

    # Train the XGBoost Classifier
    clf_xgb = XGBClassifier(random_state=0)
    clf_xgb.fit(train_features, train_labels)

    # Predict the response for the test dataset
    y_pred_xgb = clf_xgb.predict(test_features)

    # Calculate evaluation metrics
    classification_rep = classification_report(test_labels, y_pred_xgb)
    f1_xgb = f1_score(test_labels, y_pred_xgb, average='macro')
    mcc_xgb = matthews_corrcoef(test_labels, y_pred_xgb)
    cm = confusion_matrix(test_labels, y_pred_xgb, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    fpr = fp / (fp + tn)
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    roc_auc = roc_auc_score(test_labels, y_pred_xgb)
    average_precision = average_precision_score(test_labels, y_pred_xgb)
    balanced_accuracy = balanced_accuracy_score(test_labels, y_pred_xgb)
    kappa = cohen_kappa_score(test_labels, y_pred_xgb)
    hl = hamming_loss(test_labels, y_pred_xgb)
    logloss = log_loss(test_labels, y_pred_xgb)
    auprc = average_precision_score(test_labels, y_pred_xgb)

    # Store the results
    results.append({
        'classification_report': classification_rep,
        'f1_score': f1_xgb,
        'MCC': mcc_xgb,
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
        'y_pred': y_pred_xgb
    })


# Print or analyze the results from each simulation
for i, result in enumerate(results):
    print(f"Simulation {i+1}")
    print("Classification Report:")
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



##############################


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score, matthews_corrcoef, confusion_matrix, roc_auc_score, average_precision_score, balanced_accuracy_score, cohen_kappa_score, hamming_loss, log_loss
import numpy as np
import xgboost as xgb
from sklearn.metrics import classification_report, f1_score, matthews_corrcoef, confusion_matrix, ConfusionMatrixDisplay
import lightgbm as lgb

##DART
num_simulations = 50

# Placeholder for storing results
results = []

for _ in range(num_simulations):
    # Generate random feature inputs based on normal distribution
    #random_features = np.random.normal(loc=0, scale=1, size=(len(df), 10))
    
    # Split the data into training and test sets
    # Splitting the data into training and test sets
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=None)

    # Using numpy to create arrays of lables and features
    train_labels = np.array(train_df['Production'])
    test_labels = np.array(test_df['Production'])
    train_features = np.array(train_df.iloc[:, 0:38])
    test_features = np.array(test_df.iloc[:, 0:38])

    # Selecting the columns to scale
    columns_to_scale = ['OverrideFeed', 'FeedRate', 'SpindleSpeed', 'IndoorGPSx', 'IndoorGPSy']

    # Scaling the selected columns using StandardScaler
    scaler = StandardScaler()
    train_features[:, :5] = scaler.fit_transform(train_features[:, :5])
    test_features[:, :5] = scaler.transform(test_features[:, :5])
    
    # Create the DART Classifier
    clf_DART = xgb.XGBClassifier(booster='dart', random_state=0)

    # Train the DART Classifier
    clf_DART.fit(train_features, train_labels)

    # Predict the response for the test dataset
    y_pred_DART = clf_DART.predict(test_features)
    
    
    # Calculate evaluation metrics
    classification_rep = classification_report(test_labels, y_pred_DART)
    f1_RF = f1_score(test_labels, y_pred_DART, average='macro')
    mcc_RF = matthews_corrcoef(test_labels, y_pred_DART)
    cm = confusion_matrix(test_labels, y_pred_DART, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    fpr = fp / (fp + tn)
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    roc_auc = roc_auc_score(test_labels, y_pred_DART)
    average_precision = average_precision_score(test_labels, y_pred_DART)
    balanced_accuracy = balanced_accuracy_score(test_labels, y_pred_DART)
    kappa = cohen_kappa_score(test_labels, y_pred_DART)
    hl = hamming_loss(test_labels, y_pred_DART)
    logloss = log_loss(test_labels, y_pred_DART)
    auprc = average_precision_score(test_labels, y_pred_DART)
    
    # Store the results
    results.append({
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
        'y_pred': y_pred_DART
    })


# Print or analyze the results from each simulation
for i, result in enumerate(results):
    print(f"Simulation {i+1}")
    print("Classification Report:")
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



#######################################################################

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score, matthews_corrcoef, confusion_matrix, roc_auc_score, average_precision_score, balanced_accuracy_score, cohen_kappa_score, hamming_loss, log_loss
from sklearn.metrics import average_precision_score
from sklearn.metrics import balanced_accuracy_score

# Define the number of simulations for Random Forest
num_simulations = 50

# Placeholder for storing results
results = []

for _ in range(num_simulations):
    # Generate random feature inputs based on normal distribution
    #random_features = np.random.normal(loc=0, scale=1, size=(len(df), 10))
    
    # Split the data into training and test sets
    # Splitting the data into training and test sets
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=None)

    # Using numpy to create arrays of lables and features
    train_labels = np.array(train_df['Production'])
    test_labels = np.array(test_df['Production'])
    train_features = np.array(train_df.iloc[:, 0:38])
    test_features = np.array(test_df.iloc[:, 0:38])

    # Selecting the columns to scale
    columns_to_scale = ['OverrideFeed', 'FeedRate', 'SpindleSpeed', 'IndoorGPSx', 'IndoorGPSy']

    # Scaling the selected columns using StandardScaler
    scaler = StandardScaler()
    train_features[:, :5] = scaler.fit_transform(train_features[:, :5])
    test_features[:, :5] = scaler.transform(test_features[:, :5])

    
    # Train the Random Forest Classifier
    clf_RF = RandomForestClassifier(random_state=0)
    clf_RF = clf_RF.fit(train_features, train_labels)
    
    # Predict the response for the test dataset
    y_pred_RF = clf_RF.predict(test_features)
    
    # Calculate evaluation metrics
    classification_rep = classification_report(test_labels, y_pred_RF)
    f1_RF = f1_score(test_labels, y_pred_RF, average='macro')
    mcc_RF = matthews_corrcoef(test_labels, y_pred_RF)
    cm = confusion_matrix(test_labels, y_pred_RF, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    fpr = fp / (fp + tn)
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    roc_auc = roc_auc_score(test_labels, y_pred_RF)
    average_precision = average_precision_score(test_labels, y_pred_RF)
    balanced_accuracy = balanced_accuracy_score(test_labels, y_pred_RF)
    kappa = cohen_kappa_score(test_labels, y_pred_RF)
    hl = hamming_loss(test_labels, y_pred_RF)
    logloss = log_loss(test_labels, y_pred_RF)
    auprc = average_precision_score(test_labels, y_pred_RF)
    
    # Store the results
    results.append({
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
        'y_pred': y_pred_RF
    })


# Print or analyze the results from each simulation
for i, result in enumerate(results):
    print(f"Simulation {i+1}")
    print("Classification Report:")
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
    
    
    #####################################################

    
    
    # Define the ROC curve function with graphical output

def plot_roc_curve(model):

    #setting up the parameters
    try:
        probs = model.predict(test_features)
        fpr, tpr, thresholds = roc_curve(test_labels, probs)
    except ValueError:
        probs = np.argmax(model.predict(test_features, batch_size=1, verbose=0), axis=-1)
        fpr, tpr, thresholds = roc_curve(test_labels, probs)
        
    #plotting the ROC curve
    plt.plot(fpr, tpr, color='orange', label='ROC')
    plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend()
    plt.show()

    #print AUC score
    print(f"AUC score: {roc_auc_score(test_labels, probs)}")
    
    


def plot_calibration_curve(model):
    # Setting up the parameters
    try:
        probs = model.predict(test_features)
    except ValueError:
        probs = np.argmax(model.predict(test_features, batch_size=1, verbose=0), axis=-1)
        
    # Compute calibration curve
    prob_true, prob_pred = calibration_curve(test_labels, probs, n_bins=10)

    # Plot calibration curve
    plt.plot(prob_pred, prob_true, 's-', label=model.__class__.__name__)
    plt.plot([0, 1], [0, 1], '--', color='gray', label='Perfectly calibrated')
    plt.xlabel('Mean predicted probability')
    plt.ylabel('Fraction of positives')
    plt.title('Calibration Curve')
    plt.legend()
    plt.show()



def plot_error_analysis(model):
    # Predict the response for the test dataset
    y_pred = model.predict(test_features)

    # Create a DataFrame for test data and predictions
    df_test = pd.DataFrame(test_features, columns=df.columns[:-1])
    df_test['True Label'] = test_labels
    df_test['Predicted Label'] = y_pred

    # Identify misclassified samples
    misclassified_samples = df_test[df_test['True Label'] != df_test['Predicted Label']]

    # Plot characteristics of misclassified samples
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 10))
    sns.boxplot(x='True Label', y='OverrideFeed', data=misclassified_samples, ax=axes[0, 0])
    sns.boxplot(x='True Label', y='FeedRate', data=misclassified_samples, ax=axes[0, 1])
    sns.boxplot(x='True Label', y='SpindleSpeed', data=misclassified_samples, ax=axes[1, 0])
    sns.boxplot(x='True Label', y='IndoorGPSx', data=misclassified_samples, ax=axes[1, 1])
    plt.tight_layout()
    plt.show()


def plot_lift_curve(model):
    # Get predicted probabilities for positive class
    try:
        probs = model.predict_proba(test_features)[:, 1]
    except ValueError:
        probs = np.max(model.predict_proba(test_features, batch_size=1, verbose=0), axis=-1)
    
    # Sort predictions and true labels by predicted probabilities
    sorted_indices = np.argsort(probs)[::-1]
    sorted_probs = probs[sorted_indices]
    sorted_labels = test_labels[sorted_indices]

    # Calculate cumulative response and cumulative lift
    cum_response = np.cumsum(sorted_labels)
    cum_lift = cum_response / np.sum(test_labels)

    # Calculate random response and random lift
    random_response = np.arange(1, len(test_labels) + 1) / len(test_labels)
    random_lift = cum_lift / random_response

    # Calculate area under lift curve
    auc_lift = auc(random_response, cum_lift)

    # Plot lift curve
    plt.plot(random_response, cum_lift, label='Lift Curve (AUC = {:.2f})'.format(auc_lift))
    plt.plot([0, 1], [1, 1], linestyle='--', color='gray')
    plt.xlabel('Percentage of data')
    plt.ylabel('Lift')
    plt.title('Lift Curve')
    plt.legend()
    plt.show()

def plot_feature_importance(model, feature_names):
    importances = model.feature_importances_
    sorted_indices = np.argsort(importances)[::-1]
    sorted_importances = importances[sorted_indices]
    sorted_features = feature_names[sorted_indices]
    
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(sorted_importances)), sorted_importances, tick_label=sorted_features)
    plt.xticks(rotation='vertical')
    plt.xlabel('Features')
    plt.ylabel('Importance')
    plt.title('Feature Importance')
    plt.show()
    
    
def plot_feature_importance(model, feature_names):
    importances = model.feature_importances_
    sorted_indices = np.argsort(importances)[::-1]

    print("Length of feature_names:", len(feature_names))  # Debugging line

    sorted_importances = importances[sorted_indices]
    sorted_features = feature_names[sorted_indices]

    plt.figure(figsize=(10, 6))
    plt.bar(range(len(sorted_importances)), sorted_importances, tick_label=sorted_features)
    plt.xticks(rotation='vertical')
    plt.xlabel('Features')
    plt.ylabel('Importance')
    plt.title('Feature Importance')
    plt.show()

feature_names = df.columns[:-1]  # Assuming the last column is the target column
print("Number of features:", len(feature_names))  # Debugging line
plot_feature_importance(clf_lightgbm, feature_names)

  

############################################
import lightgbm as lgb
from sklearn.metrics import classification_report

# Create the LightGBM Classifier
clf_lightgbm = lgb.LGBMClassifier(random_state=0)

# Train the LightGBM Classifier
clf_lightgbm.fit(train_features, train_labels)

# Predict the response for the test dataset
y_pred_lightgbm = clf_lightgbm.predict(test_features)

# Get the metrics for the LightGBM Classifier
print(classification_report(test_labels, y_pred_lightgbm))

# Get macro average F1 score
f1_lightgbm = f1_score(test_labels, y_pred_lightgbm, average='macro')
print("F1 Score (DART):", f1_lightgbm)

# Calculate Matthews Correlation coefficient (for future use)
MCC_lightgbm = matthews_corrcoef(test_labels, y_pred_lightgbm)
print("Matthews Correlation Coefficient (DART):", MCC_lightgbm)

# Plot the confusion matrix for the LightGBM Classifier Classifier
cm = confusion_matrix(test_labels, y_pred_lightgbm, labels=[0, 1])
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])
disp.plot()

# Plot the ROC curve and get the AUC score for the Random Forest
plot_roc_curve(clf_lightgbm)

# Plot the Calibration Curve for the Random Forest
plot_calibration_curve(clf_lightgbm)
##The calibration curve shows the relationship between the predicted probabilities and the observed frequencies.

plot_error_analysis(clf_lightgbm)

# Plot the Lift Curve for the Random Forest
plot_lift_curve(clf_lightgbm)

# Plot the Feature Importance for the Random Forest
feature_names = df.columns[:-1]  # Assuming the last column is the target column
plot_feature_importance(clf_lightgbm, feature_names)

  
  ##################################################

    
    
    

    
    
