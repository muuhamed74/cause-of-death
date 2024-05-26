#!/usr/bin/env python
# coding: utf-8

# In[6]:


import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder #for encoding 
from sklearn.linear_model import LogisticRegression  #Logistic Regression model
from sklearn.naive_bayes import GaussianNB  #Naive Bayes model
from sklearn.ensemble import RandomForestClassifier #Random Forest
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, roc_auc_score


# In[8]:


# Loading Data
dataset=pd.read_csv('E:/matirials/third year/pattern/project/heart-disease_Testing Part.csv')
# Reading data information
dataset.info()


# In[10]:


dataset.head(10)


# In[14]:


#Preprocessing 
#first step (outliers)
def detect_outliers_iqr(dataset, threshold=1.5):
    Q1 = np.percentile(dataset, 25)
    Q3 = np.percentile(dataset, 75)
    IQR = Q3 - Q1
    lower_bound = Q1 - threshold * IQR
    upper_bound = Q3 + threshold * IQR
    outliers = (dataset < lower_bound) | (dataset > upper_bound)
    return outliers

# Example usage:
outliers = detect_outliers_iqr(dataset['age'])
#X9 stands for average glucose level in blode
print("Number of outliers:", outliers.sum())


# In[15]:


#preprocessisng
#second step (Missing Values) 

#Handling Missing Values:
# Drop rows with missing values
from sklearn.impute import SimpleImputer

# Identify numerical and categorical columns
numerical_columns = dataset.select_dtypes(include=['number']).columns
categorical_columns = dataset.select_dtypes(include=['object']).columns

# Impute missing values for numerical columns
numerical_imputer = SimpleImputer(strategy='mean')
data[numerical_columns] = numerical_imputer.fit_transform(data[numerical_columns])

# Impute missing values for categorical columns
categorical_imputer = SimpleImputer(strategy='most_frequent')
data[categorical_columns] = categorical_imputer.fit_transform(data[categorical_columns])

#no missing values


# In[21]:


#feature engineering
#third step (Feature Extraction)

from sklearn.datasets import load_iris
from sklearn.decomposition import PCA


# Load the iris dataset
iris = load_iris()
X = iris.data  # Features
y = iris.target  # Target

# Perform PCA for feature extraction
pca = PCA(n_components=2)  # Reduce to 2 components for visualization
X_pca = pca.fit_transform(X)

# Create a DataFrame for visualization
df = pd.DataFrame(data=X_pca, columns=['Principal Component 1', 'Principal Component 2'])
df['Target'] = y  # Add target variable to DataFrame

# Visualize the extracted features
print(df.head())


# In[22]:


#Feature engineering 
# fourth step (feature aggregation) //دي زياده محتاجين بونص زياده

# Define the groups for aggregation (e.g., based on a categorical variable)
grouped_column = 'X4'

# Define the numerical columns to aggregate
numerical_columns = ['X3', 'X9', 'X10']

# Define the aggregation functions
aggregation_functions = {
    'X3': 'mean',
    'X3': 'median',
    'X9': 'sum',
    'X10': 'mean'
}

# Perform the aggregation
aggregated_data = data.groupby(grouped_column)[numerical_columns].agg(aggregation_functions).reset_index()

# Rename the columns for clarity
aggregated_data.columns = [grouped_column] + [f'{grouped_column}_{agg}' for agg in aggregation_functions.keys()]

# Merge aggregated data back to the original DataFrame (if needed)
data = data.merge(aggregated_data, on=grouped_column, how='left')


# In[23]:


#preprocessisng
#Fifth step (encoding & scaling) 

categorical_columns = ['X2', 'X6', 'X7', 'X11'  ]  
numerical_columns = ['X3', 'X9', 'X10'] 

# One-hot encoding for categorical variables
encoder = OneHotEncoder(sparse=False)
encoded_categorical = encoder.fit_transform(data[categorical_columns])

# Create DataFrame for encoded categorical columns
encoded_categorical_df = pd.DataFrame(encoded_categorical, columns=encoder.get_feature_names_out(categorical_columns))

# Min-max scaling for numerical features
scaler = MinMaxScaler()
scaled_numerical = scaler.fit_transform(data[numerical_columns])

# Create DataFrame for scaled numerical features
scaled_numerical_df = pd.DataFrame(scaled_numerical, columns=numerical_columns)

# Concatenate encoded categorical columns and scaled numerical features into final DataFrame
final_data = pd.concat([encoded_categorical_df, scaled_numerical_df], axis=1)

# Print the final preprocessed data
print(final_data.head())


# In[24]:


#data splitting
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[31]:


#logisgticRegression with clssification metrics

from sklearn.multiclass import OneVsRestClassifier

# Initialize the logistic regression model
logistic_regression_model = LogisticRegression()

# Initialize the OneVsRestClassifier
ovr_classifier = OneVsRestClassifier(logistic_regression_model)

# Train the model
ovr_classifier.fit(X_train, y_train)

# Predict on the testing set
y_pred = ovr_classifier.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
classification_report_result = classification_report(y_test, y_pred)
confusion_matrix_result = confusion_matrix(y_test, y_pred)

print("Accuracy:", accuracy)
print("Classification Report:\n", classification_report_result)
print("Confusion Matrix:\n", confusion_matrix_result)


# In[26]:


#Random forest with clssification metrics

# Initialize the Random Forest classifier
random_forest_model = RandomForestClassifier()

# Train the model
random_forest_model.fit(X_train, y_train)

# Predict on the testing set
y_pred = random_forest_model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
classification_report_result = classification_report(y_test, y_pred)
confusion_matrix_result = confusion_matrix(y_test, y_pred)

print("Accuracy:", accuracy)
print("Classification Report:\n", classification_report_result)
print("Confusion Matrix:\n", confusion_matrix_result)


# In[27]:


# Naive Bayes with clssification metrics

# Initialize the Naive Bayes classifier
naive_bayes_model = GaussianNB()

# Train the model
naive_bayes_model.fit(X_train, y_train)

# Predict on the testing set
y_pred = naive_bayes_model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
classification_report_result = classification_report(y_test, y_pred)
confusion_matrix_result = confusion_matrix(y_test, y_pred)

print("Accuracy:", accuracy)
print("Classification Report:\n", classification_report_result)
print("Confusion Matrix:\n", confusion_matrix_result)


# In[28]:


sns.countplot(x='X11',hue='Target',data=data)#smoke status


# In[29]:


sns.countplot(x='X2',hue='Target',data=data)#gender compare


# In[30]:


sns.countplot(x='X4',hue='Target',data=data)#if patient has Haipatensonu *ضغط الدم*


# In[ ]:





# In[ ]:





# In[ ]:




