import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Load the dataset
data = pd.read_csv('customer_churn_data.csv')

# Data Cleaning and Preprocessing
# Handle missing values
data.fillna(method='ffill', inplace=True)

# Encode categorical variables
label_encoder = LabelEncoder()
data['Gender'] = label_encoder.fit_transform(data['Gender'])
data['Geography'] = label_encoder.fit_transform(data['Geography'])

# Feature Engineering
# Create new features if necessary (example: tenure per product)
data['Tenure_Per_Product'] = data['Tenure'] / (data['NumOfProducts'] + 1)

# Split the data into features and target variable
X = data.drop(['Exited', 'CustomerId', 'Surname'], axis=1)
y = data['Exited']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Model Training and Evaluation
# Logistic Regression Model
log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)
y_pred_log_reg = log_reg.predict(X_test)

# Random Forest Classifier Model
rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
rf_clf.fit(X_train, y_train)
y_pred_rf_clf = rf_clf.predict(X_test)

# Model Performance Visualization and Evaluation
def evaluate_model(y_test, y_pred, model_name):
    print(f"Model: {model_name}")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))
    print("\n")

evaluate_model(y_test, y_pred_log_reg, "Logistic Regression")
evaluate_model(y_test, y_pred_rf_clf, "Random Forest Classifier")

# Visualize feature importance for Random Forest Classifier
feature_importances = pd.DataFrame(rf_clf.feature_importances_,
                                   index=X.columns,
                                   columns=['importance']).sort_values('importance', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x=feature_importances.importance, y=feature_importances.index)
plt.title('Feature Importances - Random Forest Classifier')
plt.show()

# Exploratory Data Analysis (EDA)
plt.figure(figsize=(10, 6))
sns.countplot(x='Exited', data=data)
plt.title('Customer Churn Distribution')
plt.show()

plt.figure(figsize=(10, 6))
sns.boxplot(x='Exited', y='CreditScore', data=data)
plt.title('Credit Score vs Exited')
plt.show()

plt.figure(figsize=(10, 6))
sns.boxplot(x='Exited', y='Age', data=data)
plt.title('Age vs Exited')
plt.show()

plt.figure(figsize=(10, 6))
sns.boxplot(x='Exited', y='Balance', data=data)
plt.title('Balance vs Exited')
plt.show()

print("Customer Churn Analysis Completed.")
