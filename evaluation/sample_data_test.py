import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
from model_main import MiniAutoML
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import train_test_split

cfg = pd.read_json("models.json").to_dict(orient="records")

automl = MiniAutoML(cfg, metric="balanced_accuracy")

X = pd.read_csv("test_data/X.csv")
y = pd.read_csv("test_data/y.csv").values.ravel()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(X_test.describe())
print(X_test.info())

# Train the AutoML system and get the best model
best_model = automl.fit(X_train, y_train, cv=5)

# Display the leaderboard
print("Leaderboard:")
print(automl.display_leaderboard())

# Make predictions on the test set
predictions = automl.predict(X_test)
print("Predictions:")
print(predictions)

# Get prediction probabilities for the test set
probabilities = automl.predict_proba(X_test)
print("Prediction Probabilities:")
print(probabilities)

accuracy = balanced_accuracy_score(y_test, predictions)
print(f"Balanced accuracy of predictions: {accuracy:.4f}")
