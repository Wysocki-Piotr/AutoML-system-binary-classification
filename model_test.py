import pandas as pd
from sklearn.model_selection import train_test_split
from model_main import MiniAutoML

cfg = pd.read_json("models.json").to_dict(orient="records")


automl = MiniAutoML(cfg)
data = pd.read_csv("Datasets/diabetes_46921.csv")
X = data.drop(columns=["target"])
y = data["target"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Train and create the leaderboard with cross-validation
leaderboard = automl.fit(X_train, y_train, cv=5)

# Display the leaderboard
print(automl.display_leaderboard())