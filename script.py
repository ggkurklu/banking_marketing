import pandas as pd
import numpy as np
from scipy.stats import zscore
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report , mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def clean_dataset_(data):
    """
    Cleans the given DataFrame by handling missing values,
    removing duplicates, and filtering out outliers.
    """
    print("data before cleaning")
    print(data.shape)
    print("Data types:\n", df.dtypes)
    # Handling missing values in the dataset
    # Check for missing values
    missing_values = data.isnull().sum()
    print("Missing values in each column before handling:")
    print(missing_values)

    # Option 1: Drop rows with missing values
    # data.dropna(inplace=True)

    # Option 2: Impute missing values for numeric columns with mean
    numeric_cols = data.select_dtypes(include=np.number).columns
    data[numeric_cols] = data[numeric_cols].fillna(data[numeric_cols].mean())

    # Impute missing values for categorical columns with mode
    categorical_cols = data.select_dtypes(include='object').columns
    for col in categorical_cols:
        data[col].fillna(data[col].mode()[0], inplace=True)

    # Removing any duplicates
    data.drop_duplicates(inplace=True)
    # Using z-score to handle outliers
    numeric_df = data.select_dtypes(include=np.number)
    z_scores = np.abs(zscore(numeric_df))
    print("duplicate removed")
    print(z_scores.shape)

    # Use 3 as z-score threshold
    threshold = 3
    outlier_mask = (z_scores > threshold).any(axis=1)

    # Remove rows with outliers
    clean_df = data[~outlier_mask]

    # Check for missing values after handling
    missing_values_after = clean_df.isnull().sum()
    print("Missing values in each column after handling:")
    print(missing_values_after)

    # Return cleaned dataset
    return clean_df

# Example usage
df = pd.read_csv("https://raw.githubusercontent.com/Timjini/banking_marketing/refs/heads/main/data/bank-data.csv")
month_mapping = {
    'jan': 1,
    'feb': 2,
    'mar': 3,
    'apr': 4,
    'may': 5,
    'jun': 6,
    'jul': 7,
    'aug': 8,
    'sep': 9,
    'oct': 10,
    'nov': 11,
    'dec': 12
}
df['month_numeric'] = df['month'].replace(month_mapping)
df.drop(columns=['month'], inplace=True)
df.rename(columns={'month_numeric': 'month'}, inplace=True)
cleaned_df = clean_dataset_(df)
numeric_df = cleaned_df.select_dtypes(include=['number'])
categorical_cols = cleaned_df.select_dtypes(include=['object'])
# exploratory analysis
# data overview
print(cleaned_df.info())
print(cleaned_df.shape)
print("Data types:\n", cleaned_df.dtypes)
print(cleaned_df.head(100))

# summary stats
print("Descriptive Statistics:\n", cleaned_df.describe())
categorical_cols = cleaned_df.columns
for col in categorical_cols:
    print(f"Value counts for {col}:\n", cleaned_df[col].value_counts())

# correlation matrix
# correlation_matrix = cleaned_df.corr()
# print("Correlation Matrix:\n", correlation_matrix)
colloration_data = numeric_df
correlation_matrix = colloration_data.corr()

# Visualize the correlation matrix using a heatmap
plt.figure(figsize=(15, 10))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', square=True)
plt.title('Correlation Matrix')
plt.show()

cleaned_df.hist(bins=30, figsize=(15, 10))
plt.show()

# average balance by Job
cleaned_df.groupby('job')['balance'].mean().plot(kind='bar', figsize=(10, 6))
plt.title("Average Balance by Job")
plt.ylabel("Average Balance")
plt.xlabel("Job")
plt.show()

z_scores = np.abs(zscore(numeric_df))
print("Outliers detected in each numerical column:\n\n", (z_scores > 3).sum())

sns.scatterplot(data=df, x='age', y='balance', hue='y')
plt.title("Age vs Balance Colored by Target")
plt.show()

sns.countplot(x='job', hue='y', data=df)
plt.title("Job Count Colored by Target")
plt.xticks(rotation=45)
plt.show()

# Desision Tree 
df_encoded = pd.get_dummies(cleaned_df[categorical_cols], drop_first=True)
numerical_cols = numeric_df.columns
# Combine with numerical columns
X = pd.concat([cleaned_df[numerical_cols], df_encoded], axis=1)

# Define the target variable
y = cleaned_df['y'].apply(lambda x: 1 if x == 'yes' else 0)  # Convert target variable to binary

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("x train data shape\n", X_train.shape)
print("x test data shape\n" , X_test.shape)
print("y train data shape\n",y_train.shape)
print("y test data shape\n",y_test.shape)


# Initialize the Decision Tree Classifier
dt_classifier = DecisionTreeClassifier(random_state=42)

# Train the model
dt_classifier.fit(X_train, y_train)

# Make predictions on the test set
y_pred = dt_classifier.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Visualize the decision tree
plt.figure(figsize=(20,10))
plot_tree(dt_classifier, filled=True, feature_names=X.columns, class_names=['No', 'Yes'])
plt.title('Decision Tree Visualization')
plt.show()


# Logistic Regression

log_reg = LogisticRegression(max_iter=1000, random_state=42)

# Train the model
log_reg.fit(X_train, y_train)

# Make predictions on the test set
y_pred_log_reg = log_reg.predict(X_test)

# Calculate accuracy for Logistic Regression
accuracy_log_reg = accuracy_score(y_test, y_pred_log_reg)
print("Logistic Regression Accuracy:", accuracy_log_reg)

# Print classification report for Logistic Regression
print("\nLogistic Regression Classification Report:\n", classification_report(y_test, y_pred_log_reg))

# Confusion Matrix for Logistic Regression
conf_matrix_log_reg = confusion_matrix(y_test, y_pred_log_reg)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix_log_reg, annot=True, fmt='d', cmap='Blues', xticklabels=['No', 'Yes'], yticklabels=['No', 'Yes'])
plt.title('Logistic Regression Confusion Matrix')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()