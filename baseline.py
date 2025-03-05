import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

# Part 1 - Baseline accuracy

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


# Part 2 - Data visualisation with PCA

def visualise_data(X, y):
    """ Apply PCA to reduce feature dimensions and plot results"""
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis', alpha=0.7)
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")


    plt.title("PCA Visualisation of Diabetes Data")
    plt.colorbar(scatter, label="Outcome")
    plt.show()


# visualise training data
visualise_data(X_train, y_train)

# prep pytorch data - tensor conversion and dataloader creation

# arrays to tensors
X_train_tensor = torch.tensor(X_train.to_numpy(), dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.to_numpy(), dtype=torch.float32).unsqueeze(1)
X_test_tensor = torch.tensor(X_test.to_numpy(), dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.to_numpy(), dtype=torch.float32).unsqueeze(1)

# tensor datasets and dataloaders
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size)


# util functions
def train_model(model, train_loader, criterion, optimizer, num_epochs=100):
    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch_X.size(0)
        epoch_loss /= len(train_loader.dataset)
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}")


def evaluate_model(model, data_loader):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch_X, batch_y in data_loader:
            outputs = model(batch_X)
            preds = (torch.sigmoid(outputs) >= 0.5).float()
            all_preds.append(preds)
            all_labels.append(batch_y)
    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    accuracy = (all_preds.eq(all_labels).sum().item()) / len(all_labels)
    return accuracy


# Part 3 - Simple Logistic Regression Model using PyTorch

class SimpleLogisticModel(nn.Module):
    def __init__(self, input_dim):
        super(SimpleLogisticModel, self).__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x):
        return self.linear(x)  # raw logits

# num of features
input_dim = X_train.shape[1]

simple_model = SimpleLogisticModel(input_dim)
criterion_simple = nn.BCEWithLogitsLoss()
optimizer_simple = optim.Adam(simple_model.parameters(), lr=0.01)

print("\nTraining Simple Logistic Regression Model (PyTorch)...")
train_model(simple_model, train_loader, criterion_simple, optimizer_simple, num_epochs=100)
simple_accuracy = evaluate_model(simple_model, test_loader)
print(f"Simple PyTorch Model Test Accuracy: {simple_accuracy * 100:.2f}%")


# Part 4 - Deep NN with PyTorch
class DeepNN(nn.Module):
    def __init__(self, input_dim):
        super(DeepNN, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1)  # output logits for binary classification
        )

    def forward(self, x):
        return self.model(x)


deep_model = DeepNN(input_dim)
criterion_deep = nn.BCEWithLogitsLoss()
optimizer_deep = optim.Adam(deep_model.parameters(), lr=0.01)

print("\nTraining Deep Neural Network Model (PyTorch)...")
train_model(deep_model, train_loader, criterion_deep, optimizer_deep, num_epochs=100)

deep_accuracy = evaluate_model(deep_model, test_loader)

print(f"Deep Neural Network Test Accuracy: {deep_accuracy * 100:.2f}%")

# Part 5 - Creative and Technically Sound Machine Learning

# using polynomial features, we can create new features based on interactions between existing ones
# helps models capture non-linear relationships in our diabetese dataset
from sklearn.preprocessing import PolynomialFeatures

# use only interaction terms to avoid too many features
poly = PolynomialFeatures(degree=2, interaction_only=True)

X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)


# handle class imbalance using SMOTE (Synthetic Minority Over-sampling Technique)
# SMOTE will generate synthetic examples to balance the dataset
# 500 samples of no diabetes vs 268 samples of diabetes
# stops the model from being more biased towards predicting no diabetes
from imblearn.over_sampling import SMOTE
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Hyperparameter Tuning for logistic regression
# Instead of using default hyperparameters, we search for the best regularization strength
# GridSearchCV is able to use cross-validation to find the best value for alpha

# helps avoid
from sklearn.model_selection import GridSearchCV

# finding the best "C" value (reg. strength)
grid = {"C": [0.01, 0.1, 1, 10]}
# 5-fold cross validation
grid_search = GridSearchCV(LogisticRegression(max_iter=1000), grid, cv=5)

# train on resampled dataset and display best paramater
grid_search.fit(X_train_resampled, y_train_resampled)
print(f"Best Logistic Regression Params: {grid_search.best_params_}")

# By using ensemble learning technique, we are combining Random Forest and Gradient Boosting together
# to improve accuracy and generalization.
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

# using multiple decision trees and averages the outputs to help with overfitting prevention, starting with 100 trees
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

# building trees one right after the other, each tree getting better
gb_model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42)

# train the models using the newly resampled dataset
rf_model.fit(X_train_resampled, y_train_resampled)
gb_model.fit(X_train_resampled, y_train_resampled)

# predictions on the original test set
rf_pred = rf_model.predict(X_test)
gb_pred = gb_model.predict(X_test)

# evaluating performance
from sklearn.metrics import accuracy_score, classification_report
print(f"Random Forest Accuracy: {accuracy_score(y_test, rf_pred) * 100:.2f}%")
print(f"Gradient Boosting Accuracy: {accuracy_score(y_test, gb_pred) * 100:.2f}%")

# classification report with the model performance
# precision, recall, and F1-score for both classes
print("\nClassification Report for Best Model (Gradient Boosting):")
print(classification_report(y_test, gb_pred))
