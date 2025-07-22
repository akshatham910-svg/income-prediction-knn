#
# source: knn_adult_csv updated.ipynb
#
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# [cite_start]1. Load Data [cite: 2]
data = pd.read_csv("./adult.csv") # Make sure adult.csv is in the same folder

# [cite_start]2. Data Cleaning and Preprocessing [cite: 2]

# [cite_start]Handle missing values represented as '?' [cite: 2]
[cite_start]data.workclass.replace({'?':'Others'}, inplace=True) [cite: 2]
[cite_start]data.occupation.replace({'?':'Others'}, inplace=True) [cite: 2]

# [cite_start]Handle outliers in 'age' column [cite: 2]
[cite_start]data = data[(data['age'] <= 75) & (data['age'] >= 17)] [cite: 2]

# [cite_start]Remove irrelevant categories from 'workclass' and 'education' [cite: 2]
[cite_start]data = data[data['workclass'] != 'Without-pay'] [cite: 2]
[cite_start]data = data[data['workclass'] != 'Never-worked'] [cite: 2]
[cite_start]data = data[data['education'] != '1st-4th'] [cite: 2]
[cite_start]data = data[data['education'] != '5th-6th'] [cite: 2]
[cite_start]data = data[data['education'] != 'Preschool'] [cite: 2]

# [cite_start]Drop the 'education' column as 'education-num' is sufficient [cite: 2]
[cite_start]data.drop(columns=['education'], inplace=True) [cite: 2]

# [cite_start]3. Feature Engineering - Convert categorical columns to numerical [cite: 2]
[cite_start]encoder = LabelEncoder() [cite: 2]
[cite_start]categorical_cols = ['workclass', 'marital-status', 'occupation', 'relationship', 'race', 'gender', 'native-country'] [cite: 2]
for col in categorical_cols:
    [cite_start]data[col] = encoder.fit_transform(data[col]) [cite: 2]

# [cite_start]4. Split data into features (X) and target (Y) [cite: 2]
[cite_start]X = data.drop(columns=['income']) [cite: 2]
[cite_start]Y = data['income'] [cite: 2]

# [cite_start]5. Data Scaling [cite: 2]
[cite_start]scaler = MinMaxScaler() [cite: 2]
[cite_start]X_scaled = scaler.fit_transform(X) [cite: 2]

# [cite_start]6. Split data into training and testing sets [cite: 2]
[cite_start]xtrain, xtest, ytrain, ytest = train_test_split(X_scaled, Y, test_size=0.2, random_state=23, stratify=Y) [cite: 2]

# [cite_start]7. Model Training and Prediction [cite: 2]
[cite_start]knn = KNeighborsClassifier() [cite: 2]
[cite_start]knn.fit(xtrain, ytrain) [cite: 2]
[cite_start]predict = knn.predict(xtest) [cite: 2]

# [cite_start]8. Evaluate the Model [cite: 2]
[cite_start]accuracy = accuracy_score(ytest, predict) [cite: 2]
[cite_start]print(f"Model Accuracy: {accuracy}") [cite: 2]
# [cite_start]Expected Output: Model Accuracy: 0.8167786644267114 [cite: 2]
