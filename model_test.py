import pandas as pd
from sklearn.model_selection import train_test_split
from model_main import MiniAutoML
from sklearn.metrics import accuracy_score

# Load model configurations from JSON
cfg = pd.read_json("models.json").to_dict(orient="records")

# Initialize the MiniAutoML system
automl = MiniAutoML(cfg, metric="brier")

# Load dataset
data = pd.read_csv("Datasets/diabetes_46921.csv")
X = data.drop(columns=["target"])
y = data["target"]

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(X_test.describe())
print(X_test.info())

# Train the AutoML system and get the best model
best_model = automl.fit(X_train, y_train, cv=5)

# Display the leaderboard
print("Leaderboard:")
print(automl.display_leaderboard())

# Convert y_test to integers if predictions are integers
if y_test.dtype == object:  # Check if y_test contains strings
    y_test = y_test.map({'No': 0, 'Yes': 1})

# Make predictions on the test set
predictions = automl.predict(X_test)
print("Predictions:")
print(predictions)


# Get prediction probabilities for the test set
probabilities = automl.predict_proba(X_test)
print("Prediction Probabilities:")
print(probabilities)

accuracy = accuracy_score(y_test, predictions)
print(f"Accuracy of predictions: {accuracy:.4f}")