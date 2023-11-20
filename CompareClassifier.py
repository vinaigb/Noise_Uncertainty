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


    
data = pd.read_csv("Data/Data_Detecting_Changeover.csv", sep= ',',header=0, index_col=0)
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
# we scale the noise appropriately to match the desired power 
# spectral density.

from scipy.signal import lfilter

# Load the dataframe
data = pd.read_csv("Data/Data_Detecting_Changeover.csv", sep=',', header=0, index_col=0)

# Columns to add noise to
columns_to_add_noise = ['OverrideFeed', 'FeedRate', 'SpindleSpeed', 'IndoorGPSx', 'IndoorGPSy']

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

data = pd.read_csv("Data/Data_Detecting_Changeover.csv", sep=',', header=0, index_col=0)
# Columns to add noise to
columns_to_add_noise = ['OverrideFeed', 'FeedRate', 'SpindleSpeed', 'IndoorGPSx', 'IndoorGPSy']


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
data = pd.read_csv("Data/Data_Detecting_Changeover.csv", sep=',', header=0, index_col=0)

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

# Columns to add noise to
columns_to_add_noise = ['OverrideFeed', 'FeedRate', 'SpindleSpeed', 'IndoorGPSx', 'IndoorGPSy']

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
data = pd.read_csv("Data/Data_Detecting_Changeover.csv", sep=',', header=0, index_col=0)
columns_to_add_noise = ['OverrideFeed', 'FeedRate', 'SpindleSpeed', 'IndoorGPSx', 'IndoorGPSy']
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

data = pd.read_csv("Data/Data_Detecting_Changeover.csv", sep=',', header=0, index_col=0)
columns_to_add_noise = ['OverrideFeed', 'FeedRate', 'SpindleSpeed', 'IndoorGPSx', 'IndoorGPSy']

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

########################################
data = pd.read_csv("Data/Data_Detecting_Changeover.csv", sep= ',',header=0, index_col=0)
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


#data.to_csv('Data/Uniform_Noise_Encoded_Data.csv', index=False)

# Dropping the columns we do not need
df = data.copy()
df.pop('DateTime')
df.pop('Phase')
df.pop('Phase_compressed')
print(df)

df.info()
  
    
    
import numpy as np
import pandas as pd
import lightgbm as lgb
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, matthews_corrcoef, confusion_matrix, roc_auc_score, balanced_accuracy_score, cohen_kappa_score, hamming_loss, log_loss, average_precision_score, classification_report

# Set the number of simulations
num_simulations = 50

# Placeholder for storing results
results_hist = []
results_lightgbm = []
results_RF = []
results_xgb = []
results_DART = []

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, f1_score, matthews_corrcoef, confusion_matrix, roc_auc_score, balanced_accuracy_score, cohen_kappa_score, hamming_loss, log_loss, average_precision_score
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingClassifier


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
    results_hist.append({
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



import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, f1_score, matthews_corrcoef, confusion_matrix, roc_auc_score, balanced_accuracy_score, cohen_kappa_score, hamming_loss, log_loss, average_precision_score
from xgboost import XGBClassifier

# Set the number of simulations
num_simulations = 50


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
    results_xgb.append({
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
    results_DART.append({
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




from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score, matthews_corrcoef, confusion_matrix, roc_auc_score, average_precision_score, balanced_accuracy_score, cohen_kappa_score, hamming_loss, log_loss
import numpy as np
import xgboost as xgb
from sklearn.metrics import classification_report, f1_score, matthews_corrcoef, confusion_matrix, ConfusionMatrixDisplay
import lightgbm as lgb
from sklearn.metrics import average_precision_score
from sklearn.metrics import balanced_accuracy_score

# Define the number of Monte Carlo simulations for Random Forest
num_simulations = 50

# Placeholder for storing results
#results = []

for _ in range(num_simulations):

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
    results_RF.append({
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




# Metrics to compare
metrics = ['f1_score', 'MCC', 'specificity', 'ROC_AUC', 'average_precision', 'balanced_accuracy', 'kappa', 'hamming_loss', 'log_loss', 'auprc']

# Create subplots
num_metrics = len(metrics)
fig, axes = plt.subplots(num_metrics, 1, figsize=(10, num_metrics * 5))

# Compare each metric
for i, metric in enumerate(metrics):
    hist_metric_values = [result[metric] for result in results_hist]
    lightgbm_metric_values = [result[metric] for result in results_lightgbm]
    RF_metric_values = [result[metric] for result in results_RF]
    DART_metric_values = [result[metric] for result in results_DART]
    xgb_metric_values = [result[metric] for result in results_xgb]
    
    axes[i].boxplot([hist_metric_values, lightgbm_metric_values, RF_metric_values, DART_metric_values, xgb_metric_values], labels=['HistGradientBoosting', 'LightGBM', 'Random Forest', 'DART', 'XGBoost'])
    axes[i].set_title(f'{metric} Influence of Gaussian Noise')
    axes[i].set_ylabel(metric)

# Adjust layout and display plots
plt.tight_layout()
plt.show()


import numpy as np
import matplotlib.pyplot as plt

# Metrics to compare
metrics = ['f1_score', 'MCC', 'specificity', 'ROC_AUC', 'average_precision', 'balanced_accuracy', 'kappa', 'hamming_loss', 'log_loss', 'auprc']

# Loop through each metric and create a separate plot
for metric in metrics:
    fig, ax = plt.subplots(figsize=(10, 5))
    
    hist_metric_values = [result[metric] for result in results_hist]
    lightgbm_metric_values = [result[metric] for result in results_lightgbm]
    RF_metric_values = [result[metric] for result in results_RF]
    DART_metric_values = [result[metric] for result in results_DART]
    xgb_metric_values = [result[metric] for result in results_xgb]
    
    ax.boxplot([hist_metric_values, lightgbm_metric_values, RF_metric_values, DART_metric_values, xgb_metric_values], labels=['HistGradientBoosting', 'LightGBM', 'Random Forest', 'DART', 'XGBoost'])
    ax.set_title(f'{metric} Influence of Gaussian Noise')
    ax.set_ylabel(metric)
    
    # Adjust layout and save the plot
    plt.tight_layout()
    plt.savefig(f'{metric}_plot.png')  
    plt.close() 



############################################



