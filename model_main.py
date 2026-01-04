import pandas as pd
import numpy as np
import time
from copy import deepcopy
from wrappers.wrapper_model import ModelWrapper
from sklearn.metrics import balanced_accuracy_score, brier_score_loss, accuracy_score
from Preprocessing.SimplePreprocessor import SimplePreprocessor
from sklearn.model_selection import cross_val_score, ParameterSampler, cross_val_predict, StratifiedKFold
from sklearn.linear_model import LogisticRegression


class StackingEnsemble:
    """
    Wrapper na Ensemble, który zachowuje się jak pojedynczy model (ma fit/predict).
    Obsługuje specyficzne preprocessory dla każdego modelu bazowego.
    """

    def __init__(self, base_models_data, meta_model=None):
        """
        :param base_models_data: Lista słowników {'wrapper': ModelWrapper, 'preprocessor': Preprocessor}
        :param meta_model: Model decyzyjny (np. LogisticRegression). Jeśli None, tworzy domyślny.
        """
        self.base_models_data = base_models_data
        self.meta_model = meta_model if meta_model else LogisticRegression()
        self.fitted_ = False

    def fit(self, X, y):
        """
        Trenuje ensemble:
        1. Trenuje każdy model bazowy na X (używając jego własnego preprocessora).
        2. Generuje OOF (Out-of-Fold) predictions dla X.
        3. Trenuje meta-model na OOF predictions.
        """
        meta_features = []

        # 1. Trenowanie modeli bazowych i generowanie OOF dla Meta-Learnera
        print("  -> Training Ensemble Base Models & Generating OOF predictions...")
        for item in self.base_models_data:
            wrapper = item['wrapper']
            preproc = item['preprocessor']

            # Transformacja danych dedykowana dla tego modelu
            X_trans, y_trans = preproc.transform(X.copy(), y.copy())

            # Ustawienie parametrów dla specyficznych bibliotek
            cat_cols = X_trans.select_dtypes(include=['category', 'object']).columns.tolist()
            if "CatBoost" in wrapper.model.__class__.__name__:
                wrapper.model.set_params(cat_features=cat_cols)
            if "XGBClassifier" in wrapper.model.__class__.__name__ and hasattr(wrapper.model, "enable_categorical"):
                wrapper.model.set_params(enable_categorical=True, tree_method="hist")

            # A) Generowanie OOF predictions (dla meta-modelu)
            # Używamy cross_val_predict, żeby symulować nieznane dane
            try:
                oof_pred = cross_val_predict(
                    wrapper.model, X_trans, y_trans, cv=5, method="predict_proba", n_jobs=-1
                )[:, 1]
            except:
                # Fallback dla modeli bez predict_proba lub błędów
                oof_pred = cross_val_predict(
                    wrapper.model, X_trans, y_trans, cv=5, method="predict", n_jobs=-1
                )
            meta_features.append(oof_pred)

            # B) Trenowanie modelu na pełnym zbiorze (do późniejszego predict)
            wrapper.model.fit(X_trans, y_trans)

        # 2. Trenowanie Meta-Learnera
        X_meta = np.column_stack(meta_features)
        self.meta_model.fit(X_meta, y)
        self.fitted_ = True
        return self

    def predict(self, X):
        if not self.fitted_: raise ValueError("Ensemble not fitted")
        meta_features = self._get_meta_features(X)
        return self.meta_model.predict(meta_features)

    def predict_proba(self, X):
        if not self.fitted_: raise ValueError("Ensemble not fitted")
        meta_features = self._get_meta_features(X)
        return self.meta_model.predict_proba(meta_features)

    def _get_meta_features(self, X):
        preds = []
        for item in self.base_models_data:
            wrapper = item['wrapper']
            preproc = item['preprocessor']
            X_trans, _ = preproc.transform(X.copy(), None)
            preds.append(wrapper.predict_proba(X_trans)[:, 1])
        return np.column_stack(preds)


class MiniAutoML:
    def __init__(self, models_config, metric="balanced_accuracy"):
        self.models_config = models_config
        self.metric = metric
        self.leaderboard = None
        self.preprocessor = SimplePreprocessor()  # Domyślny, ale Ensemble może go nadpisać
        self.best_model = None

    def fit(self, X_train, y_train, cv=5):
        scores = []
        n_samples, n_features = X_train.shape

        # ==============================================================================
        # ETAP 1: SCREENING (Wstępna ocena wszystkich modeli)
        # ==============================================================================
        print(f"--- Stage 1: Initial Screening of {len(self.models_config)} models ---")

        for model_config in self.models_config:
            # Sprawdzanie ograniczeń wielkości danych
            constraints = model_config.get("constraints", {})
            if n_samples > constraints.get("max_samples", float('inf')) or \
                    n_features > constraints.get("max_features", float('inf')):
                continue

            # Wybór preprocessingu
            use_native_cat = "categorical" in model_config["name"] or "CatBoost" in model_config["class"]
            current_preprocessor = SimplePreprocessor(handle_categorical=not use_native_cat)

            # Preprocessing i init
            X_temp, y_temp = current_preprocessor.fit_transform(X_train.copy(), y_train.copy())
            wrapper = ModelWrapper(model_config)

            # Config specyficzny
            cat_cols = X_temp.select_dtypes(include=['category', 'object']).columns.tolist()
            if "CatBoost" in model_config["class"]:
                wrapper.model.set_params(cat_features=cat_cols)
            if use_native_cat and "XGBClassifier" in model_config["class"]:
                wrapper.model.set_params(enable_categorical=True, tree_method="hist")

            # Ocena (CV)
            try:
                cv_scores = cross_val_score(
                    wrapper.model, X_temp, y_temp, cv=cv,
                    scoring=self.metric if self.metric != "balanced_accuracy" else "balanced_accuracy"
                )
                mean_score = np.mean(cv_scores)
            except Exception as e:
                print(f"Error evaluating {model_config['name']}: {e}")
                continue

            scores.append({
                "Model Name": model_config["name"],
                "Model Class": model_config["class"],
                "Metric Score": mean_score,
                "Wrapper": wrapper,
                "Used_Preprocessor": current_preprocessor,
                "Config": model_config,
                "Is_Optimized": False,
                "Params": "Default"
            })
            print(f"Model '{model_config['name']}' achieved mean {self.metric}: {mean_score:.4f}")

        # Wstępny Leaderboard
        self.leaderboard = pd.DataFrame(scores)
        ascending = self.metric == "brier"
        self.leaderboard = self.leaderboard.sort_values(by="Metric Score", ascending=ascending).reset_index(drop=True)

        # ==============================================================================
        # ETAP 2: OPTYMALIZACJA (Top 5 modeli)
        # ==============================================================================
        top_5_models = self.leaderboard.head(5).to_dict('records')
        print("\n" + "=" * 50)
        print(f"--- Stage 2: Optimizing Top {len(top_5_models)} Models (Max 3 min each) ---")

        for row in top_5_models:
            model_config = row["Config"]
            search_space = model_config.get("search_space")

            if not search_space: continue

            print(f"Optimizing {model_config['name']}...")
            preprocessor = row["Used_Preprocessor"]
            X_temp, y_temp = preprocessor.transform(X_train.copy(), y_train.copy())

            # Setup Random Search
            param_sampler = ParameterSampler(search_space, n_iter=1000, random_state=42)
            best_opt_score = row["Metric Score"]
            best_opt_params = None
            start_time = time.time()
            wrapper = ModelWrapper(model_config)

            # Config specyficzny (ponownie, dla nowej instancji)
            cat_cols = X_temp.select_dtypes(include=['category', 'object']).columns.tolist()
            if "CatBoost" in model_config["class"]: wrapper.model.set_params(cat_features=cat_cols)
            if "XGBClassifier" in model_config["class"] and "categorical" in model_config["name"]:
                wrapper.model.set_params(enable_categorical=True, tree_method="hist")

            # Pętla optymalizacyjna
            for params in param_sampler:
                if time.time() - start_time > 60: break  # 3 minuty limit
                try:
                    wrapper.model.set_params(**params)
                    cv_s = cross_val_score(
                        wrapper.model, X_temp, y_temp, cv=cv,
                        scoring=self.metric if self.metric != "balanced_accuracy" else "balanced_accuracy"
                    )
                    mean_opt = np.mean(cv_s)

                    is_better = (mean_opt < best_opt_score) if ascending else (mean_opt > best_opt_score)
                    if is_better:
                        best_opt_score = mean_opt
                        best_opt_params = params
                except:
                    continue

            if best_opt_params:
                print(f"  -> Improved! New score: {best_opt_score:.4f}")
                best_wrapper = ModelWrapper(model_config)
                if "CatBoost" in model_config["class"]: best_wrapper.model.set_params(cat_features=cat_cols)
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

        # Aktualizacja leaderboardu przed Ensemble
        self.leaderboard = pd.DataFrame(scores)
        self.leaderboard = self.leaderboard.sort_values(by="Metric Score", ascending=ascending).reset_index(drop=True)

        # ==============================================================================
        # ETAP 3: STACKING ENSEMBLE (Budowanie i Ocena)
        # ==============================================================================
        print("\n" + "=" * 50)
        print("--- Stage 3: Building & Evaluating Stacking Ensemble ---")

        # Wybór modeli do ensemble (Top 3 unikalne klasy)
        selected_rows = []
        seen_classes = set()
        for _, row in self.leaderboard.iterrows():
            if row["Model Class"] not in seen_classes:
                selected_rows.append(row)
                seen_classes.add(row["Model Class"])
            if len(selected_rows) >= 3: break

        # Jeśli za mało unikalnych, dobierz najlepsze z góry
        if len(selected_rows) < 2:
            selected_rows = self.leaderboard.head(3).to_dict('records')

        print(f"Ensemble components: {[r['Model Name'] for r in selected_rows]}")

        # Przygotowanie danych dla Ensemble
        base_models_data = []
        meta_features_train = []  # Przechowuje predykcje OOF do oceny Ensemble

        for row in selected_rows:
            # Tworzymy głęboką kopię wrappera (z parametrami)
            # Musimy stworzyć nową instancję, bo 'row["Wrapper"]' może być użyty gdzie indziej
            config = row["Config"]
            wrapper_clone = ModelWrapper(config)

            # Aplikujemy parametry (zoptymalizowane lub default)
            # Pobieramy parametry z instancji w leaderboardzie, bo ona trzyma stan po RandomSearch
            current_params = row["Wrapper"].model.get_params()
            wrapper_clone.model.set_params(**current_params)

            # Preprocessor
            preproc = row["Used_Preprocessor"]
            X_trans, y_trans = preproc.transform(X_train.copy(), y_train.copy())

            # Zbieramy OOF predictions dla tego modelu (żeby ocenić jakość Ensemble BEZ wycieku danych)
            # method='predict_proba' dla klasyfikacji
            try:
                oof_pred = cross_val_predict(
                    wrapper_clone.model, X_trans, y_trans, cv=cv, method="predict_proba", n_jobs=-1
                )[:, 1]
            except:
                oof_pred = cross_val_predict(
                    wrapper_clone.model, X_trans, y_trans, cv=cv, method="predict", n_jobs=-1
                )

            meta_features_train.append(oof_pred)

            # Dodajemy do listy składników ensemble (do późniejszego .fit())
            base_models_data.append({
                'wrapper': wrapper_clone,
                'preprocessor': preproc
            })

        # Ocena Ensemble (Symulacja wyniku za pomocą CV na Meta-Modelu)
        X_meta = np.column_stack(meta_features_train)
        meta_learner = LogisticRegression()

        # Szybka ocena meta-modelu (LogReg jest błyskawiczny)
        ens_cv_scores = cross_val_score(
            meta_learner, X_meta, y_train, cv=cv,
            scoring=self.metric if self.metric != "balanced_accuracy" else "balanced_accuracy"
        )
        ens_mean_score = np.mean(ens_cv_scores)

        print(f"Ensemble Estimated {self.metric}: {ens_mean_score:.4f}")

        # Dodanie Ensemble do Leaderboardu
        # Ważne: 'Used_Preprocessor' dajemy None, bo Ensemble sam zarządza preprocessorami swoich dzieci
        # 'Wrapper' to będzie nasza klasa StackingEnsemble
        ensemble_instance = StackingEnsemble(base_models_data, meta_learner)

        scores.append({
            "Model Name": "Stacking Ensemble (Top 3 Unique)",
            "Model Class": "StackingEnsemble",
            "Metric Score": ens_mean_score,
            "Wrapper": ensemble_instance,
            "Used_Preprocessor": None,  # Specjalny przypadek
            "Config": {"name": "Ensemble", "class": "Ensemble"},
            "Is_Optimized": True,
            "Params": "Meta: LogReg"
        })

        # ==============================================================================
        # ETAP 4: FINALIZACJA
        # ==============================================================================
        # Sortowanie końcowe
        self.leaderboard = pd.DataFrame(scores)
        self.leaderboard = self.leaderboard.sort_values(by="Metric Score", ascending=ascending).reset_index(drop=True)

        best_row = self.leaderboard.iloc[0]
        self.best_model = best_row["Wrapper"]
        self.preprocessor = best_row["Used_Preprocessor"]

        print("\n" + "=" * 50)
        print(f"WINNER: {best_row['Model Name']} with score {best_row['Metric Score']:.4f}")
        print("Training winner on full dataset...")

        # Trenowanie zwycięzcy
        if best_row["Model Class"] == "StackingEnsemble":
            # Ensemble bierze surowe dane i sam robi preprocess wewnątrz
            self.best_model.fit(X_train, y_train)
        else:
            # Pojedynczy model potrzebuje zewnętrznego preprocessora
            X_final, y_final = self.preprocessor.transform(X_train.copy(), y_train.copy())
            self.best_model.fit(X_final, y_final)

        return self.best_model

    def predict(self, X_test):
        if not self.best_model: raise ValueError("Call fit() first.")

        # Jeśli wygrał Ensemble, on nie potrzebuje zewnętrznego preprocessora (ma wewnętrzne)
        if isinstance(self.best_model, StackingEnsemble):
            return self.best_model.predict(X_test)

        # Jeśli wygrał pojedynczy model
        X_test, _ = self.preprocessor.transform(X_test, None)
        return self.best_model.model.predict(X_test)

    def predict_proba(self, X_test):
        if not self.best_model: raise ValueError("Call fit() first.")

        if isinstance(self.best_model, StackingEnsemble):
            return self.best_model.predict_proba(X_test)[:, 1]

        X_test, _ = self.preprocessor.transform(X_test, None)
        return self.best_model.predict_proba(X_test)[:, 1]

    def display_leaderboard(self, mode="short"):
        if self.leaderboard is None: raise ValueError("No leaderboard.")
        cols = ["Model Name", "Metric Score", "Is_Optimized"]
        return self.leaderboard[cols] if mode == "short" else self.leaderboard