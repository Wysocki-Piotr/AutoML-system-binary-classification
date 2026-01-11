import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '')))

import pandas as pd
from model_main import MiniAutoML
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')
pd.set_option('display.max_rows', None)

# Wszystkie kolumny
pd.set_option('display.max_columns', None)
cfg = pd.read_json("models_new.json").to_dict(orient="records")

automl = MiniAutoML(cfg, metric="balanced_accuracy")

X = pd.read_csv("test_data/X.csv")
y = pd.read_csv("test_data/y.csv").squeeze().astype(int)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


print(X_test.describe())
print(X_test.info())

# Train the AutoML system and get the best model
best_model = automl.fit(X_train, y_train, cv=5)

# Display the leaderboard
print("Leaderboard:")
print(automl.display_leaderboard(mode="long"))

# Make predictions on the test set
predictions = automl.predict(X_test)
print("Predictions:")
print(predictions)

# Get prediction probabilities for the test set
probabilities = automl.predict_proba(X_test)
print("Prediction Probabilities:")
print(probabilities)

# Ensure y_test and predictions have the same type
if y_test.dtype != predictions.dtype:
    if y_test.dtype == 'object':
        y_test = y_test.map({'No': 0, 'Yes': 1})  # Convert string labels to integers
    elif predictions.dtype == 'object':
        predictions = pd.Series(predictions).map({0: 'No', 1: 'Yes'})  # Convert integer predictions to strings

# Calculate balanced accuracy
accuracy = balanced_accuracy_score(y_test, predictions)
print(f"Balanced accuracy of predictions: {accuracy:.4f}")
