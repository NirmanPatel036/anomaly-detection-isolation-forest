# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.experimental import enable_iterative_imputer # and not IteratveImputer as the API might change without any deprecation cycle
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import joblib

# Load the dataset
ccd = pd.read_csv('creditcard.csv')

'''V1 to V28: These are the principal components obtained with PCA (Principal Component Analysis).
V29: This is the amount of the transaction.
V30: This is the time (in seconds) since the first transaction in the dataset.
Class: This is the target variable where 0 indicates a normal transaction and 1 indicates a fraudulent transaction.'''

# Take a look at the first few rows of the dataset
print(ccd.head())

# Check the shape of the dataset
print("Dataset Shape:", ccd.shape)

# Get information about the dataset
print(ccd.info())

# Get summary statistics
print(ccd.describe())

# Count the number of normal and fraudulent transactions
print("Distribution of Target Variable:")
print(ccd['Class'].value_counts())

# Visualize the class distribution
plt.figure(figsize=(8,6))
sns.countplot(x='Class', data=ccd)
plt.title('Distribution of Normal vs Fraudulent Transactions')
plt.show()

# Create histograms for numerical features
ccd.hist(bins=50, figsize=(20, 15))
plt.title('Distribution of Numerical Features')
plt.show()

# Create box plots for numerical features
plt.figure(figsize=(20, 10))
plt.title('Box Plots of Numerical Features')
plt.xticks(rotation=45)
ccd.plot(kind='box', subplots=True, layout=(8, 4))
plt.show()

# Visualize the distribution of the 'amount' feature
plt.figure(figsize=(10,6))
sns.histplot(ccd['Amount'], bins=50, kde=True)
plt.title('Distribution of Transaction Amounts')
plt.xlabel('Amount')
plt.ylabel('Frequency')
plt.show()

# Visualize multiple features using box plots
plt.figure(figsize=(12,8))
sns.boxplot(data=ccd[['Amount', 'Class']], orient='v')
plt.title('Box Plot of Key Features')
plt.show()

# Understanding how features correlate with each other can provide valuable insights:
# Calculate correlation matrix
corr_matrix = ccd.corr()

# Visualize the correlation matrix
plt.figure(figsize=(12,10))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
plt.title('Correlation Matrix')
plt.tight_layout()
plt.show()

# Check for missing values
print("Missing Values Count:")
print(ccd.isnull().sum())

# Visualize missing values
plt.figure(figsize=(10, 6))
ccd.isnull().sum().plot(kind='bar')
plt.title('Missing Values Distribution')
plt.show()

# Identify outliers using Isolation Forest
if_model = IsolationForest(contamination=0.01, random_state=42)
if_model.fit(ccd)
if_predictions = if_model.predict(ccd)

# Create a new dataframe with outlier predictions
outlier_data = pd.DataFrame({'outlier': if_predictions}, index=ccd.index)
print("\nOutlier Predictions:")
print(outlier_data[outlier_data['outlier'] == -1].head())  # -1 indicates outliers

# Visualize outliers on a scatter plot
plt.figure(figsize=(10, 6))
plt.scatter(ccd.iloc[:, 0], ccd.iloc[:, 1], 
             c=['red' if prediction == -1 else 'blue' for prediction in if_predictions])
plt.title('Outlier Visualization')
plt.xlabel(ccd.columns[0])
plt.ylabel(ccd.columns[1])
plt.show()

# handling missing values in the dataset
# Remove rows with missing values
ccd.dropna(inplace=True)

# Alternatively, remove columns with missing values
# ccd.dropna(axis=1, inplace=True)

# methods of imputation(replace missing values): mean/median, mode, regression, KNN, iterative (keyword: SimpleImputer), Iterative imputer
# Initialize iterative imputer
imputer = SimpleImputer(strategy='mean')

# Apply imputation to numerical columns
numerical_cols = ccd.select_dtypes(include=['int64', 'float64']).columns
ccd[numerical_cols] = imputer.fit_transform(ccd[numerical_cols])

# Encode categorical variables (not required for numeriacl columns)
'''le = LabelEncoder()
for col in numerical_cols:
    ccd[col] = le.fit_transform(ccd[col])'''

# data preprocessing
# There are two common approaches for normalization technique:
# Min-Max Scaler (Normalization): Scales data between a specified range (usually 0 to 1)
# Standard Scaler (Standardization): Scales data to have mean=0 and standard deviation=1 >> recommended

# Initialize the StandardScaler
scaler = StandardScaler()

# Fit and transform the data
# Note: We usually don't split the data before scaling when dealing with anomaly detection
# because anomalies could be in both train and test sets
scaled_data = scaler.fit_transform(ccd)

# Convert back to DataFrame
scaled_data = pd.DataFrame(scaled_data, columns=ccd.columns)

# Now split the data into training and testing sets
y = ccd['Class']
X_train, X_test, y_train, y_test = train_test_split(
    scaled_data.drop('Class', axis=1),  # 'Class' is our label column
    ccd['Class'],
    test_size=0.2,
    random_state=42
)

# Important: Save the scaler for future use (you'll need it for deployment)
''' Don't Split Before Scaling: Unlike classification tasks, in anomaly detection, we don't split the data before
scaling because anomalies might exist in both train and test sets. '''
joblib.dump(scaler, 'standard_scaler.pkl')

''' next steps: Importing the Isolation Forest library
				Instantiating the model
				Training the model on the scaled data
				Using the model to predict outliers '''

# check for categorical variables
print(ccd.dtypes)

# Look through the dtype list. If there are any 'object' or 'category types', those are your categorical variables
# if any categorical variables, to handle them:
# le = LabelEncoder()
# ncd['Amount'] = le.fit_transform(ccd['Amount'])

# model implimentation
# instantiating the forest model
if_model = IsolationForest(
    n_estimators=100,  	 # Number of trees in the forest
    max_samples='auto',  # Used to set max_samples to min(256, n_samples)
    contamination=0.1,   # Assuming 10% of the data are outliers
    random_state=42  	 # For reproducibility
)

# training the model
''' The algorithm works by creating multiple isolation trees. Each tree is built by randomly selecting features
and splitting the data until all instances are isolated. The process of isolating instances that are farthest from the others
helps identify anomalies efficiently. '''

# fit the model to the training data
if_model.fit(X_train)

# save the model
joblib.dump(if_model, 'isolation_forest_model.pkl')

'''Feature Selection: At each node, a random feature is selected.
   Data Partitioning: The data is split based on a random value between the minimum and maximum values of the selected feature.
   Isolation: This process continues until all instances are isolated in separate leaves.
   The algorithm keeps track of the number of splits (path lengths) required to isolate each instance. Instances that are isolated
   in fewer splits are more likely to be anomalies. '''

# use the trained model to make predictions on the test data (gives an array with 1 for inliner and -1 for outlier)
y_pred = if_model.predict(X_test)

# convert predictions to binary (0 for inliers, 1 for outliers)
y_pred_binary = np.where(y_pred == -1, 1, 0)

# use precision, recall, and F1-score for evaluating model performamce
# calculating evaluation metrics
default_precision = precision_score(y_test, y_pred_binary, zero_division=0)
default_recall = recall_score(y_test, y_pred_binary, zero_division=0)
default_f1 = f1_score(y_test, y_pred_binary, zero_division=0)

print(f"Precision: {default_precision:.3f}")
print(f"Recall: {default_recall:.3f}")
print(f"F1-score: {default_f1:.3f}")

# creating the confusion matrix to help understand where our model is making mistakes
cm = confusion_matrix(y_test, y_pred_binary)

# Visualize confusion matrix
plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.title("Confusion Matrix")
plt.show()

# evaluating anomoly detection
def evaluate_anomaly_detection(y_true, y_pred):
    """
    Evaluate the performance of an anomaly detection model using precision, recall, and F1-score.
    
    Args:
        y_true (array-like): Ground truth target values (1 for inliers, -1 for outliers)
        y_pred (array-like): Estimated targets from the model (1 for inliers, -1 for outliers)
        
    Returns:
        tuple: (precision, recall, f1_score)
    """
    # Calculate the confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Get the number of true positives, false positives, false negatives, and true negatives
    tp = cm[0][0]  # True positives (correctly predicted inliers)
    fp = cm[0][1]  # False positives (incorrectly predicted inliers)
    fn = cm[1][0]  # False negatives (incorrectly predicted outliers)
    tn = cm[1][1]  # True negatives (correctly predicted outliers)
    
    # Calculate precision
    precision = tp / (tp + fp) if (tp + fp) != 0 else 0
    
    # Calculate recall
    recall = tn / (tn + fn) if (tn + fn) != 0 else 0
    
    # Calculate F1-score
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0
    
    return precision, recall, f1

def visualize_evaluation_metrics(metrics):
    """
    Create bar chart visualization of evaluation metrics
    Args:
        metrics (dict): Dictionary containing precision, recall, and F1-score
    """
    # Extract metrics
    precision = metrics['precision']
    recall = metrics['recall']
    f1 = metrics['f1']
    
    # Create bar chart
    plt.figure(figsize=(10, 6))
    plt.bar(['Precision', 'Recall', 'F1-Score'], [precision, recall, f1])
    plt.title('Evaluation Metrics')
    plt.xlabel('Metric')
    plt.ylabel('Value')
    plt.ylim(0, 1)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.show()

'''
High-Dimensional Data: Most real-world datasets, especially those used in anomaly detection, often have many features.
The Credit Card Fraud dataset, for example, has 30 features.

Visualization Challenges: Visualizing data with more than three dimensions is challenging. Dimensionality reduction helps project this
high-dimensional data into a lower-dimensional space (usually 2D or 3D) while retaining most of the information.

Techniques:
>> Principal Component Analysis (PCA): A linear technique that transforms data into a set of principal components,
retaining as much variance as possible (good for capturing global structure and variance in the data)
>> t-Distributed Stochastic Neighbor Embedding (t-SNE): A non-linear technique that is particularly useful for
visualizing clusters andmanifolds (better for capturing local structures and non-linear relationships)
'''

# dimensionality reduction: since most datasets are high-dimensional, we need to reduce it to 2D for visualization
# PCA Reduction
# create PCA instance with 2 components
pca = PCA(n_components=2, random_state=42)
# fit and transform the data
pca_data = pca.fit_transform(X_test)

# create t-SNE instance
tsne = TSNE(n_components=2, random_state=42, perplexity=30)
# fit and transform the data
tsne_data = tsne.fit_transform(X_test)

# create a figure with two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

# Plot PCA data
sns.scatterplot(x=pca_data[:,0], y=pca_data[:,1], 
                hue=y_pred, 
                palette=['blue', 'red'], 
                ax=ax1,
                legend=False)
ax1.set_title('PCA Visualization')
ax1.set_xlabel('Principal Component 1')
ax1.set_ylabel('Principal Component 2')

# Plot t-SNE data
sns.scatterplot(x=tsne_data[:,0], y=tsne_data[:,1], 
                hue=y_pred, 
                palette=['blue', 'red'], 
                ax=ax2,
                legend=False)
ax2.set_title('t-SNE Visualization')
ax2.set_xlabel('t-SNE Component 1')
ax2.set_ylabel('t-SNE Component 2')

# add counts of inliers and outliers
inlier_count = np.sum(y_pred == 1)
outlier_count = np.sum(y_pred == -1)

plt.text(0.98, 0.98, f'Total points: {len(X_test)}\nInliers: {inlier_count}\nOutliers: {outlier_count}', 
         transform=ax1.transAxes, ha='right')

plt.tight_layout()
plt.show()


# define the range of values we want to test for each parameter
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_samples': [256, 512, 1024],
    'contamination': [0.05, 0.1, 0.15],
    'random_state': [42]
}

'''
grid_search = GridSearchCV(estimator=if_model, 
                            param_grid=param_grid, 
                            cv=5, 
                            n_jobs=-1,
                            verbose=1)
grid_search.fit(X_train, y_train)

best_params = grid_search.best_params_
best_score = grid_search.best_score_
print(f"Best Parameters: {best_params}")
print(f"Best Score: {best_score}")

# Use best model for predictions
best_model = grid_search.best_estimator_
y_pred_best = best_model.predict(X_test)

# evaluate the scores again
best_precision = precision_score(y_test, y_pred_best)
best_recall = recall_score(y_test, y_pred_best)
best_f1 = f1_score(y_test, y_pred_best)

print("Default Model Performance:")
print(f"Precision: {default_precision:.3f}, Recall: {default_recall:.3f}, F1: {default_f1:.3f}")
print("\nBest Model Performance:")
print(f"Precision: {best_precision:.3f}, Recall: {best_recall:.3f}, F1: {best_f1:.3f}")

metrics = ['Precision', 'Recall', 'F1']
default_scores = [default_precision, default_recall, default_f1]
best_scores = [best_precision, best_recall, best_f1]

plt.figure(figsize=(10, 6))
plt.bar(range(3), default_scores, label='Default Model')
plt.bar(range(3), best_scores, label='Best Model')
plt.title('Model Performance Comparison')
plt.xlabel('Metric')
plt.ylabel('Score')
plt.xticks(range(3), metrics)
plt.legend()
plt.show()
'''
