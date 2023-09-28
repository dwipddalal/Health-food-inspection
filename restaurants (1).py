import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.utils.class_weight import compute_class_weight

import torch
import torch.nn as nn
import torch.optim as optim

import matplotlib.pyplot as plt
import seaborn as sns

try:
    df_business_master = pd.read_csv('data/InspectionData-scraped/business_master_withyelp.csv', delimiter='\t')
    df_business_master_head = df_business_master.head()
except Exception as e:
    df_business_master_head = str(e)

df_business_master_head

# Counting the number of unique names in the "business_master_withyelp.csv" dataset
unique_names_count = df_business_master['name'].nunique()
unique_names_count

yelpID_unique = df_business_master['YelpID'].nunique()
yelpID_unique

file_path_mapping = 'data/InspectionData-scraped/mapping.csv'
df_mapping = pd.read_csv(file_path_mapping)

df_mapping.head()

# Merging the "business_master_withyelp.csv" and "mapping.csv" datasets based on the "permit_id" column
merged_df = pd.merge(df_business_master, df_mapping, on='permit_id', how='inner')

# Displaying the first few rows of the merged dataset
merged_df.head()

# Counting the number of rows with NaN values in the merged dataset
rows_with_nan = merged_df.isnull().any(axis=1).sum()
rows_with_nan

merged_df.shape

merged_df.columns

# Loading the newly uploaded "cleaned_violation_list.csv" dataset
file_path_violation_list = 'data/InspectionData-scraped/cleaned_violation_list.csv'
df_violation_list = pd.read_csv(file_path_violation_list)

# Displaying the first few rows of the violation list dataset to get an overview
df_violation_list.head()

# Merging the "merged_df" and "cleaned_violation_list.csv" datasets based on the "inspec_id" and "report" columns
final_merged_df = pd.merge(merged_df, df_violation_list, left_on='inspec_id', right_on='report', how='inner')


final_merged_df.head()

final_merged_df.columns

le = LabelEncoder()
final_merged_df['name_encoded'] = le.fit_transform(final_merged_df['name'])

# Feature Engineering: Aggregating the data by 'name_encoded' to get frequencies of 'repeat_violation' and 'crit_viol'
grouped_df = final_merged_df.groupby('name_encoded').agg({'repeat_violation': 'sum', 'crit_viol': 'sum'}).reset_index()

# Target Variable: Creating a binary target variable based on 'repeat_violation'
# If 'repeat_violation' > 0 for a restaurant, then it is likely to commit a violation again in the future (label = 1), else label = 0
grouped_df['likely_to_violate_again'] = grouped_df['repeat_violation'].apply(lambda x: 1 if x > 0 else 0)

X = grouped_df[['name_encoded', 'crit_viol']]
y = grouped_df['likely_to_violate_again']

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Using Random Forest Classifier
clf = RandomForestClassifier(random_state=42)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)

accuracy

print(classification_rep)

# Initialize models
rf_clf = RandomForestClassifier(random_state=42)
logreg_clf = LogisticRegression(random_state=42)
gb_clf = GradientBoostingClassifier(random_state=42)
svc_clf = SVC(random_state=42)

# Store models in a list
classifiers = [rf_clf, logreg_clf, gb_clf, svc_clf]
classifier_names = ['Random Forest', 'Logistic Regression', 'Gradient Boosting', 'Support Vector Machine']

# Initialize report
extensive_report = {}

# Train and evaluate models
for clf, name in zip(classifiers, classifier_names):
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    class_rep = classification_report(y_test, y_pred, zero_division=1)

    extensive_report[name] = {
        'Accuracy': accuracy,
        'Classification Report': class_rep
    }

for i in extensive_report:
    print(i)
    # print(extensive_report[i]['Accuracy'])
    print(extensive_report[i]['Classification Report'])
    print('\n')

# Calculate class weights
class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)

# Print the class weights
print("Class Weights:")
for class_label, weight in zip(np.unique(y_train), class_weights):
    print(f"Class {class_label}: {weight}")

# Initialize models
rf_clf = RandomForestClassifier(random_state=42, class_weight='balanced')

# Store models in a list
classifiers = [rf_clf]
classifier_names = ['Random Forest']

# Initialize report
extensive_report = {}

# Train and evaluate models
for clf, name in zip(classifiers, classifier_names):
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    class_rep = classification_report(y_test, y_pred, zero_division=1)

    extensive_report[name] = {
        'Accuracy': accuracy,
        'Classification Report': class_rep
    }

for i in extensive_report:
    print(i)
    # print(extensive_report[i]['Accuracy'])
    print(extensive_report[i]['Classification Report'])
    print('\n')

"""Observations:
All models have a high accuracy rate, ranging from 87.1% to 91.13%.
All models, except for SVM, failed to correctly identify any restaurant that is likely to commit a violation again (class 1). Even in the case of SVM, while the precision for class 1 is 1.00, the recall is 0.00, indicating that it could not actually identify any such restaurant.
The classifiers appear to be biased towards predicting that restaurants are not likely to commit a violation again. This could be due to class imbalance in the dataset.

## Adressing the issue of class imbalance
"""



# Convert pandas DataFrame to PyTorch tensor
X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32)

# Reshape the y tensors
y_train_tensor = y_train_tensor.view(-1, 1)
y_test_tensor = y_test_tensor.view(-1, 1)

# Define the Focal Loss
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        BCE_loss = nn.BCEWithLogitsLoss()(inputs, targets)
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss
        return F_loss

# Define the neural network model
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(2, 16)
        self.fc2 = nn.Linear(16, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x

# Initialize the model, loss, and optimizer
model = SimpleNN()
criterion = FocalLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
n_epochs = 100
for epoch in range(n_epochs):
    model.train()
    optimizer.zero_grad()

    y_pred = model(X_train_tensor)
    loss = criterion(y_pred, y_train_tensor)

    loss.backward()
    optimizer.step()

    # Print epoch info
    print(f'Epoch {epoch+1}/{n_epochs} - Loss: {loss.item():.4f}')

# Evaluate the model
model.eval()
with torch.no_grad():
    y_pred_test = model(X_test_tensor)
    loss_test = criterion(y_pred_test, y_test_tensor)
    test_loss = loss_test.item()

    # Convert predicted probabilities to binary outputs
    y_pred_binary = (y_pred_test > 0.5).float()
    test_accuracy = (y_pred_binary == y_test_tensor).float().mean().item()

test_loss
print(test_accuracy)

from sklearn.metrics import classification_report

# Convert PyTorch tensor to NumPy array
y_test_np = y_test_tensor.numpy().flatten()
y_pred_binary_np = y_pred_binary.numpy().flatten()

# Generate a classification report
class_report = classification_report(y_test_np, y_pred_binary_np, target_names=["Not Likely to Violate", "Likely to Violate"])

# Print the final report
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")
print("Classification Report:")
print(class_report)

import warnings
warnings.filterwarnings("ignore")

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        BCE_loss = nn.BCEWithLogitsLoss(reduction='none')(inputs, targets)
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss
        return torch.mean(F_loss)

class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(2, 16)
        self.fc2 = nn.Linear(16, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Initialize the model, loss, and optimizer
model = SimpleNN()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
n_epochs = 100
alpha_values = [0.5, 0.75, 0.9, 1]  # Different alpha values for focal loss

for alpha in alpha_values:
    print(f"Training with alpha: {alpha}")
    criterion = FocalLoss(alpha=alpha, gamma=2)

    for epoch in range(n_epochs):
        model.train()
        optimizer.zero_grad()

        y_pred = model(X_train_tensor)
        loss = criterion(y_pred, y_train_tensor)

        loss.backward()
        optimizer.step()

        # Evaluate the model and print the classification report
        model.eval()
        with torch.no_grad():
            y_pred_test = model(X_test_tensor)
            loss_test = criterion(y_pred_test, y_test_tensor)

            y_pred_binary = (torch.sigmoid(y_pred_test) > 0.5).float().numpy()
            y_test_np = y_test_tensor.numpy()

            class_rep = classification_report(y_test_np, y_pred_binary, target_names=["Not Likely to Violate", "Likely to Violate"])

        print(f"Epoch {epoch+1}/{n_epochs} - Test Loss: {loss_test.item():.4f}")
        print("Classification Report:")
        print(class_rep)

"""For alpha = 1

Epoch 16/100 - Test Loss: 0.1741
Classification Report:
                       precision    recall  f1-score   support



Not Likely to Violate       0.94      0.81      0.87       113

    Likely to Violate       0.19      0.45      0.26        11

             accuracy                           0.77       124
            macro avg       0.56      0.63      0.56       124
         weighted avg       0.87      0.77      0.81       124
"""



final_merged_df.head()

# save the dataframe
final_merged_df.to_csv('data/InspectionData-scraped/final_merged_df.csv', index=False)

# Loading the newly uploaded dataset "20160523_1636_All_Service_Requests__Last_30_Days.csv"
file_path_service_requests = 'data/311/20160523_1636_All_Service_Requests__Last_30_Days.csv'
df_service_requests = pd.read_csv(file_path_service_requests)

# Displaying the first few rows of the dataset to get an overview
df_service_requests.head()

waste_keywords = ["waste", "garbage", "trash", "collection"]
hygiene_keywords = ["hygiene", "cleanliness", "sanitation"]

# Function to check if keywords are present in a sentence
def classify_row(row):
    for col in ['SERVICETYPECODEDESCRIPTION', 'SERVICECODEDESCRIPTION', 'DETAILS']:
        description = str(row[col]).lower()

        has_waste = any(keyword in description for keyword in waste_keywords)
        has_hygiene = any(keyword in description for keyword in hygiene_keywords)

        if has_waste or has_hygiene:
            return 1

    return 0

# Apply the classification function to the DataFrame
df_service_requests['IsRelated'] = df_service_requests.apply(classify_row, axis=1)

df_service_requests['IsRelated'].value_counts()

## write code to get the set of zipcodes such that IsRelated = 1
df_service_requests_related = df_service_requests[df_service_requests['IsRelated'] == 1]

l = set(df_service_requests_related['ZIPCODE'].unique())

len(l)

print(l)

# find unique values in the column SERVICECODEDESCRIPTION
df_service_requests['SERVICETYPECODEDESCRIPTION'].unique()

# Loading the dataset
file_path = 'data/Yelp/yelp_data.csv'
yelp_data_df = pd.read_csv(file_path)

# Displaying the first 5 rows of the dataset to get an overview
yelp_data_df.head()

# Setting the style for the plots
sns.set(style="whitegrid")

# Creating subplots
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Plotting the distribution of ratings
sns.histplot(yelp_data_df['rating'], kde=True, bins=9, ax=axes[0])
axes[0].set_title('Distribution of Ratings')
axes[0].set_xlabel('Rating')
axes[0].set_ylabel('Frequency')

# Plotting the distribution of review counts
sns.histplot(yelp_data_df['review_count'], kde=True, bins=50, ax=axes[1])
axes[1].set_title('Distribution of Review Counts')
axes[1].set_xlabel('Review Count')
axes[1].set_ylabel('Frequency')
axes[1].set_yscale('log')  # Using log scale due to the long tail

plt.tight_layout()
plt.show()

# Merge the dataframes based on YelpID and yelpID
merged_yelp_df = pd.merge(final_merged_df, yelp_data_df, left_on='YelpID', right_on='yelpID', how='inner')

# Display first few rows to confirm the merge
merged_yelp_df.head()

"""Following Code checks whether each ZIP code in the "zip" column of the merged_yelp_df dataframe is present in the set l and assigns a corresponding value of 1 or 0 to a new column called 'zip_in_l'."""

l = set(map(int, map(float, l)))
# Define a Function to Convert and Check
def check_and_convert_zip(zip_code):
    try:
        zip_code = int(zip_code)
        return 1 if zip_code in l else 0
    except ValueError:
        return 0  # Treat non-integer values as not found in 'l'

# Create a new column 'zip_in_l' in the dataframe
merged_yelp_df['zip_in_l'] = merged_yelp_df['zip'].apply(lambda x: check_and_convert_zip(x))

# The 'zip_in_l' column will contain 1 if the zip code (as an integer) is in set 'l', and 0 otherwise

merged_yelp_df['zip_in_l'].value_counts()

## So the logic here is that if there is a 1 in the column 'zip_in_l' then the restaurant is in the set of zipcodes that we got from the 311 data which means that there is hygiene related complaint for near that restaurant hence there is likely to be a violation in the future