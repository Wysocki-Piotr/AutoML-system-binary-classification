import pandas as pd
from wrappers.wrapper_model import ModelWrapper
from Preprocessing.SimplePreprocessor import SimplePreprocessor
from sklearn.model_selection import cross_val_score
import numpy as np

class MiniAutoML:
    def __init__(self, models_config, metric="brier"):
        """
        Initialize the MiniAutoML system with model configurations and a chosen metric.

        :param models_config: List of model configurations loaded from a JSON file.
        :param metric: Metric to evaluate models. Default is 'brier'.
        """
        self.models_config = models_config
        self.metric = metric
        self.leaderboard = None
        self.preprocessor = SimplePreprocessor()
        self.best_model = None

    def fit(self, X_train, y_train, cv=5):
        """
        Train and evaluate all models using cross-validation, and select the best model.

        :param X_train: Training features.
        :param y_train: Training labels.
        :param cv: Number of cross-validation folds. Default is 5.
        :return: The best trained model.
        """
        # Apply preprocessing
        X_train, y_train = self.preprocessor.fit_transform(X_train, y_train)

        scores = []

        for model_config in self.models_config:
            wrapper = ModelWrapper(model_config)
            print(f"Evaluating model: {model_config['class']} with parameters: {model_config.get('params', {})}")

            # Perform cross-validation
            cv_scores = cross_val_score(
                wrapper.model, X_train, y_train, cv=cv,
                scoring=self.metric if self.metric != "brier" else "neg_brier_score"
            )
            mean_score = np.mean(cv_scores)

            scores.append({
                "Model Name": model_config["name"],
                "Model Class": model_config["class"],
                "Metric Score": mean_score,
                "Wrapper": wrapper
            })

            print(
                f"Model '{model_config['name']}' ({model_config['class']}) achieved mean {self.metric}: {mean_score:.4f}")
            print("-" * 50)

        # Create a leaderboard DataFrame
        self.leaderboard = pd.DataFrame(scores)

        # Sort the leaderboard by the chosen metric (ascending for 'brier', descending otherwise)
        ascending = self.metric == "brier"
        self.leaderboard = self.leaderboard.sort_values(by="Metric Score", ascending=False).reset_index(drop=True)

        best_model_row = self.leaderboard.iloc[0]
        self.best_model = best_model_row["Wrapper"]

        # Train the best model on the full training data
        self.best_model.fit(X_train, y_train)
        print(f"Best model selected: {best_model_row['Model Name']}")

        return self.best_model

    def predict(self, X_test):
        """
        Return class predictions for the test data.

        :param X_test: Test features.
        :return: Predicted class labels.
        """
        if not self.best_model:
            raise ValueError("No model has been trained. Call fit() first.")

        # POPRAWKA: Rozpakowanie krotki (X_trans, y) -> bierzemy tylko X_test, y ignorujemy (_)
        X_test, _ = self.preprocessor.transform(X_test, None)

        return self.best_model.model.predict(X_test)
    def predict_proba(self, X_test):
        """
        Return probabilities for the positive class for the test data.

        :param X_test: Test features.
        :return: Predicted probabilities for the positive class.
        """
        if not self.best_model:
            raise ValueError("No model has been trained. Call fit() first.")
        X_test , _ = self.preprocessor.transform(X_test, None)
        return self.best_model.predict_proba(X_test)[:, 1]
    def display_leaderboard(self, mode="short"):
        """
        Display the leaderboard in either 'short' or 'long' format.

        :param mode: Display mode, either 'short' or 'long'. Default is 'short'.
        :return: Leaderboard DataFrame in the selected format.
        """
        if self.leaderboard is None:
            raise ValueError("No leaderboard available. Call fit() first.")

        if mode == "short":
            return self.leaderboard[["Model Name", "Metric Score"]]
        elif mode == "long":
            return self.leaderboard
        else:
            raise ValueError("Invalid mode. Choose either 'short' or 'long'.")