import pandas as pd
from wrappers.wrapper_model import ModelWrapper
from Preprocessing.SimplePreprocessor import SimplePreprocessor
from sklearn.model_selection import cross_val_score
import numpy as np

class MiniAutoML:
    def __init__(self, models_config, metric="brier"):
        self.models_config = models_config
        self.metric = metric
        self.leaderboard = None
        self.preprocessor = SimplePreprocessor()

    def fit(self, X_train, y_train, cv=5):
        X_train, y_train = self.preprocessor.fit_transform(X_train, y_train)

        scores = []

        for model_config in self.models_config:
            wrapper = ModelWrapper(model_config)
            print(f"Evaluating model: {model_config['class']} with parameters: {model_config.get('params', {})}")


            cv_scores = cross_val_score(
                wrapper.model, X_train, y_train, cv=cv,
                scoring=self.metric if self.metric != "brier" else "neg_brier_score"
            )
            mean_score = np.mean(cv_scores)

            scores.append({
                "Model Name": model_config["name"],
                "Model Class": model_config["class"],
                "Metric Score": mean_score,
                "Parameters": model_config.get("params", {})
            })

            print(
                f"Model '{model_config['name']}' ({model_config['class']}) achieved mean {self.metric}: {mean_score:.4f}")
            print("-" * 50)


        self.leaderboard = pd.DataFrame(scores)

        self.leaderboard = self.leaderboard.sort_values(by="Metric Score", ascending=False)

        return self.leaderboard
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