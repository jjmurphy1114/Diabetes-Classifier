import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

file_path = "diabetes.csv"
df = pd.read_csv(file_path)

# separate features (X) and target variable (y)
# all columns except Outcome
X = df.drop(columns=["Outcome"])
# target variable (y) (0 = No diabetes, 1 = Diabetes)
y = df["Outcome"]

# split into training (80%) and testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# create a dummy classifier that always predicts the majority class
dummy_model = DummyClassifier(strategy="most_frequent")
dummy_model.fit(X_train, y_train)

# predict and calculate accuracy
y_dummy_pred = dummy_model.predict(X_test)
majority_class_accuracy = accuracy_score(y_test, y_dummy_pred) * 100
print(f"Majority Class Baseline Accuracy: {majority_class_accuracy:.4f}%")

# train a logistic regression model
logistic_model = LogisticRegression(max_iter=1000, random_state=42)
logistic_model.fit(X_train, y_train)

# prediction
y_pred = logistic_model.predict(X_test)

# calculate accuracy
logistic_regression_accuracy = accuracy_score(y_test, y_pred) * 100
print(f"Logistic Regression Accuracy: {logistic_regression_accuracy:.4f}%")