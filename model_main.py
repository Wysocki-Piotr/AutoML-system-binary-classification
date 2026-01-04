import pandas as pd
import numpy as np
import time
from wrappers.wrapper_model import ModelWrapper
from sklearn.metrics import balanced_accuracy_score
from Preprocessing.SimplePreprocessor import SimplePreprocessor
from sklearn.model_selection import cross_val_score, ParameterSampler


class MiniAutoML:
    def __init__(self, models_config, metric="balanced_accuracy"):
        """
        Initialize the MiniAutoML system with model configurations and a chosen metric.

        :param models_config: List of model configurations. Each config should ideally contain a 'search_space' key for optimization.
        :param metric: Metric to evaluate models. Default is 'balanced_accuracy'.
        """
        self.models_config = models_config
        self.metric = metric
        self.leaderboard = None
        self.preprocessor = SimplePreprocessor()
        self.best_model = None

    def fit(self, X_train, y_train, cv=5):
        """
        Train and evaluate all models, optimize the top 5 using Random Search (max 3 min/model),
        and select the absolute best model.

        :param X_train: Training features.
        :param y_train: Training labels.
        :param cv: Number of cross-validation folds.
        :return: The best trained model.
        """
        scores = []
        n_samples, n_features = X_train.shape

        # --- ETAP 1: Wstępna ocena wszystkich modeli (Default Params) ---
        print(f"--- Stage 1: Initial Screening of {len(self.models_config)} models ---")

        for model_config in self.models_config:
            constraints = model_config.get("constraints", {})
            max_s = constraints.get("max_samples", float('inf'))
            max_f = constraints.get("max_features", float('inf'))

            if n_samples > max_s or n_features > max_f:
                print(f"Skipping {model_config['name']}: Dataset too large ({n_samples}x{n_features})")
                continue

            use_native_cat = "categorical" in model_config["name"] or "CatBoost" in model_config["class"]
            current_preprocessor = SimplePreprocessor(handle_categorical=not use_native_cat)

            # Preprocessing
            X_temp, y_temp = current_preprocessor.fit_transform(X_train.copy(), y_train.copy())
            cat_cols = X_temp.select_dtypes(include=['category', 'object']).columns.tolist()

            # Inicjalizacja Wrappera
            wrapper = ModelWrapper(model_config)

            # Konfiguracja specyficzna dla bibliotek (CatBoost, XGBoost)
            if "CatBoost" in model_config["class"]:
                wrapper.model.set_params(cat_features=cat_cols)
            if use_native_cat and "XGBClassifier" in model_config["class"]:
                wrapper.model.set_params(enable_categorical=True, tree_method="hist")

            # Cross-validation (Base)
            cv_scores = cross_val_score(
                wrapper.model, X_temp, y_temp, cv=cv,
                scoring=self.metric if self.metric != "balanced_accuracy" else "balanced_accuracy"
            )
            mean_score = np.mean(cv_scores)

            scores.append({
                "Model Name": model_config["name"],
                "Model Class": model_config["class"],
                "Metric Score": mean_score,
                "Wrapper": wrapper,
                "Used_Preprocessor": current_preprocessor,
                "Config": model_config,  # Przechowujemy config do późniejszej optymalizacji
                "Is_Optimized": False,
                "Params": "Default"
            })

            print(f"Model '{model_config['name']}' achieved mean {self.metric}: {mean_score:.4f}")

        # Tworzymy wstępny leaderboard
        self.leaderboard = pd.DataFrame(scores)
        ascending = self.metric == "brier"
        self.leaderboard = self.leaderboard.sort_values(by="Metric Score", ascending=ascending).reset_index(drop=True)

        # --- ETAP 2: Optymalizacja Top 5 (Random Search - Max 3 min) ---
        top_5_models = self.leaderboard.head(5).to_dict('records')
        print("\n" + "=" * 50)
        print(f"--- Stage 2: Optimizing Top {len(top_5_models)} Models (Max 3 min each) ---")

        for row in top_5_models:
            model_config = row["Config"]
            search_space = model_config.get("search_space")

            if not search_space:
                print(f"Skipping optimization for {model_config['name']} (No search_space defined).")
                continue

            print(f"Optimizing {model_config['name']}...")

            # Przygotowanie danych (używamy tego samego preprocessora co w Etapie 1)
            preprocessor = row["Used_Preprocessor"]
            X_temp, y_temp = preprocessor.transform(X_train.copy(), y_train.copy())

            # Inicjalizacja samplera i zmiennych
            # n_iter ustawione wysoko, bo ogranicza nas czas
            param_sampler = ParameterSampler(search_space, n_iter=1000, random_state=42)
            best_opt_score = row["Metric Score"]  # Startujemy od wyniku bazowego
            best_opt_params = None

            start_time = time.time()
            time_limit = 180  # 3 minuty w sekundach

            # Ręczna pętla Random Search z kontrolą czasu
            wrapper = ModelWrapper(model_config)  # Nowa instancja

            # Ponowna konfiguracja specyficzna (CatBoost/XGB)
            cat_cols = X_temp.select_dtypes(include=['category', 'object']).columns.tolist()
            if "CatBoost" in model_config["class"]:
                wrapper.model.set_params(cat_features=cat_cols)
            if "XGBClassifier" in model_config["class"] and "categorical" in model_config["name"]:
                wrapper.model.set_params(enable_categorical=True, tree_method="hist")

            try:
                for params in param_sampler:
                    # Sprawdzenie czasu
                    if time.time() - start_time > time_limit:
                        print(f"  -> Time limit reached for {model_config['name']}.")
                        break

                    # Ustawienie parametrów
                    try:
                        wrapper.model.set_params(**params)
                    except Exception as e:
                        # Ignoruj błędne parametry
                        continue

                    # CV
                    cv_scores_opt = cross_val_score(
                        wrapper.model, X_temp, y_temp, cv=cv,
                        scoring=self.metric if self.metric != "balanced_accuracy" else "balanced_accuracy"
                    )
                    mean_opt_score = np.mean(cv_scores_opt)

                    # Sprawdzenie czy wynik jest lepszy (zależnie od metryki)
                    is_better = (mean_opt_score < best_opt_score) if ascending else (mean_opt_score > best_opt_score)

                    if is_better:
                        best_opt_score = mean_opt_score
                        best_opt_params = params
                        # Kopiujemy najlepszy model (nie tylko params, stan wrappera)
                        # W sklearn params wystarczą, ale dla pewności w strukturze:
            except KeyboardInterrupt:
                print("Optimization interrupted by user.")

            # Jeśli znaleziono lepsze parametry, dodaj do leaderboardu
            if best_opt_params is not None:
                print(f"  -> Found better params! New score: {best_opt_score:.4f}")

                # Tworzymy nowy wrapper z najlepszymi parametrami
                best_wrapper = ModelWrapper(model_config)
                if "CatBoost" in model_config["class"]:
                    best_wrapper.model.set_params(cat_features=cat_cols)
                if "XGBClassifier" in model_config["class"] and "categorical" in model_config["name"]:
                    best_wrapper.model.set_params(enable_categorical=True, tree_method="hist")

                best_wrapper.model.set_params(**best_opt_params)

                scores.append({
                    "Model Name": f"{model_config['name']} (Optimized)",
                    "Model Class": model_config["class"],
                    "Metric Score": best_opt_score,
                    "Wrapper": best_wrapper,
                    "Used_Preprocessor": preprocessor,
                    "Config": model_config,
                    "Is_Optimized": True,
                    "Params": str(best_opt_params)
                })
            else:
                print(f"  -> No improvement found within time limit.")

        # --- FINALIZACJA ---
        # Aktualizacja leaderboardu o wyniki z optymalizacji
        self.leaderboard = pd.DataFrame(scores)
        self.leaderboard = self.leaderboard.sort_values(by="Metric Score", ascending=ascending).reset_index(drop=True)

        # Wybór absolutnie najlepszego modelu
        best_model_row = self.leaderboard.iloc[0]
        self.best_model = best_model_row["Wrapper"]
        self.preprocessor = best_model_row["Used_Preprocessor"]

        # Ostateczne trenowanie na pełnym zbiorze
        X_final, y_final = self.preprocessor.transform(X_train.copy(), y_train.copy())
        self.best_model.fit(X_final, y_final)

        print("=" * 50)
        print(f"Best model selected: {best_model_row['Model Name']} with score {best_model_row['Metric Score']:.4f}")

        return self.best_model

    def predict(self, X_test):
        """Return class predictions for the test data."""
        if not self.best_model:
            raise ValueError("No model has been trained. Call fit() first.")
        X_test, _ = self.preprocessor.transform(X_test, None)
        return self.best_model.model.predict(X_test)

    def predict_proba(self, X_test):
        """Return probabilities for the positive class."""
        if not self.best_model:
            raise ValueError("No model has been trained. Call fit() first.")
        X_test, _ = self.preprocessor.transform(X_test, None)
        return self.best_model.predict_proba(X_test)[:, 1]

    def display_leaderboard(self, mode="short"):
        """Display the leaderboard."""
        if self.leaderboard is None:
            raise ValueError("No leaderboard available. Call fit() first.")

        if mode == "short":
            return self.leaderboard[["Model Name", "Metric Score", "Is_Optimized"]]
        elif mode == "long":
            return self.leaderboard
        else:
            raise ValueError("Invalid mode. Choose either 'short' or 'long'.")