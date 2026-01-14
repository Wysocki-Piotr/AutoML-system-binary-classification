import pandas as pd
import numpy as np
import time
import warnings
from copy import deepcopy
from sklearn.metrics import balanced_accuracy_score, brier_score_loss, accuracy_score
from sklearn.model_selection import cross_val_score, ParameterSampler, cross_val_predict, KFold, StratifiedKFold
from sklearn.linear_model import LogisticRegression
# Zakładam, że te importy masz w swoim środowisku, jeśli nie - upewnij się, że pliki istnieją
from wrappers.wrapper_model import ModelWrapper
#from Preprocessing.SimplePreprocessor import SimplePreprocessor
from Preprocessing.AutoMLPreprocessor import AutoMLPreprocessor

class StackingEnsemble:
    def __init__(self, base_models, preprocessor, meta_model=None, threshold=0.5, refit_meta=True):
        """
        :param base_models: Lista słowników {'wrapper': ..., 'preprocessor': ...}
        :param refit_meta: Jeśli False, zakłada, że meta_model jest już wytrenowany
                           i pomija kosztowne generowanie OOF w metodzie fit().
        """
        self.base_models = base_models
        self.preprocessor = preprocessor
        self.meta_model = meta_model if meta_model else LogisticRegression()
        self.threshold = threshold
        self.refit_meta = refit_meta
        self.fitted_ = False

    def fit(self, X, y):
        """
        Metoda przyjmuje dane już przetworzone.
        Nie wykonuje transform(), jedynie dostosowuje typy kolumn dla XGB/LGBM.
        """
        print("  -> Fitting base models on full dataset...")
        cat_cols = self.preprocessor.get_categorical_cols(X)
        
        for wrapper in self.base_models:
            # Tworzymy kopię X dla danego modelu, żeby specyficzne rzutowanie nie psuło innych
            X_model = X.copy()
            model_name = wrapper.model.__class__.__name__
            
            # --- Specyficzna obsługa KATEGORII ---
            if "CatBoost" in model_name:
                 wrapper.model.set_params(cat_features=cat_cols)
            
            elif ("XGBClassifier" in model_name or "LGBMClassifier" in model_name) and cat_cols:
                # XGB/LGBM wymagają fizycznego typu 'category'
                for col in cat_cols:
                    X_model[col] = X_model[col].astype("category")
                
                if "XGBClassifier" in model_name:
                    wrapper.model.set_params(enable_categorical=True, tree_method="hist")

            wrapper.model.fit(X_model, y)

        # 2. Meta-model: Trenujemy TYLKO jeśli refit_meta=True
        if self.refit_meta:
            print("  -> Generating OOF predictions for Meta-Learner (Slow)...")
            meta_features = []
            
            for wrapper in self.base_models:
                X_oof = X.copy()
                model_name = wrapper.model.__class__.__name__
                
                # Powtórka logiki kategorii dla OOF
                if "CatBoost" in model_name:
                     wrapper.model.set_params(cat_features=cat_cols)
                elif ("XGBClassifier" in model_name or "LGBMClassifier" in model_name) and cat_cols:
                    for col in cat_cols:
                        X_oof[col] = X_oof[col].astype("category")

                try:
                    oof_pred = cross_val_predict(wrapper.model, X_oof, y, cv=5, method="predict_proba", n_jobs=-1)[:, 1]
                except:
                    oof_pred = cross_val_predict(wrapper.model, X_oof, y, cv=5, method="predict", n_jobs=-1)
                
                meta_features.append(oof_pred)

            X_meta = np.column_stack(meta_features)
            self.meta_model.fit(X_meta, y)
        else:
            print("  -> Using pre-trained Meta-Learner params.")
            # Jeśli meta model nie był trenowany, to musimy go tutaj nauczyć na czymś?
            # W Twoim kodzie MiniAutoML meta_model jest już nauczony "na brudno" przed tworzeniem Ensemble.
            # Więc tutaj zakładamy, że przekazany self.meta_model jest już .fit() z MiniAutoML.
            pass
            
        self.fitted_ = True
        return self

    def predict_proba(self, X_raw):
        """
        Tutaj wchodzą SUROWE dane, więc musimy je przetworzyć raz globalnie.
        """
        if not self.fitted_: raise ValueError("Ensemble not fitted")
        
        # 1. Globalna transformacja
        X_trans = self.preprocessor.transform(X_raw.copy(), None)
        # 2. Generowanie cech dla meta-modelu
        meta_features = self._get_meta_features(X_trans)
        
        # 3. Predykcja meta-modelu
        return self.meta_model.predict_proba(meta_features)

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= self.threshold).astype(int)

    def _get_meta_features(self, X_processed):
        """Generuje predykcje modeli bazowych na PRZETWORZONYCH danych."""
        preds = []
        cat_cols = self.preprocessor.get_categorical_cols(X_processed)
        
        for wrapper in self.base_models:
            X_model = X_processed.copy()
            model_name = wrapper.model.__class__.__name__
            
            if ("XGBClassifier" in model_name or "LGBMClassifier" in model_name) and cat_cols:
                for col in cat_cols:
                    X_model[col] = X_model[col].astype("category")
            
            preds.append(wrapper.model.predict_proba(X_model)[:, 1])
            
        return np.column_stack(preds)


class MiniAutoML:

    def __init__(self, models_config, metric="balanced_accuracy"):
        self.models_config = models_config
        self.metric = metric
        self.leaderboard = None
        self.preprocessor = AutoMLPreprocessor(
                 add_kmeans_features=True,
                 feature_selection= True,
                 add_poly_features=True, 
                 remove_outliers=True, 
                 remove_multicollinearity=True, 
                 multicollinearity_threshold=0.95, 
                 id_threshold=0.95,
                 random_state=42)
        self.best_model = None

    def fit(self, X_train, y_train, cv=5):
        warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')
        from sklearn.experimental import enable_halving_search_cv  # noqa
        from sklearn.model_selection import HalvingRandomSearchCV
        cv_stratedy = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
        scores = []
        n_samples, n_features = X_train.shape
        
        print("Begin preprocessing...")
        X_train_proc, y_train= self.preprocessor.fit_transform(X_train, y_train)

        # dla xgboost/lightgbm konwersja kolumn kategorycznych na 'category'
        # Pobieramy kolumny kategoryczne RAZ
        cat_cols = self.preprocessor.get_categorical_cols(X_train_proc)

        # Przygotowujemy wersję "castowaną" dla XGBoost/LGBM do etapu screening
        X_train_cat = X_train_proc.copy()
        if cat_cols:
            for col in cat_cols:
                X_train_cat[col] = X_train_cat[col].astype("category")
        print("Preprocessing done.")


        # ======================================================================
        # STAGE 1: SCREENING
        # ======================================================================
        print(f"--- Stage 1: Screening {len(self.models_config)} models ---")

        for model_config in self.models_config:
            constraints = model_config.get("constraints", {})
            if n_samples > constraints.get("max_samples", float("inf")):
                continue
            if n_features > constraints.get("max_features", float("inf")):
                continue

            use_native_cat = (
                    "categorical" in model_config["name"]
                    or "CatBoost" in model_config["class"]
            )

            wrapper = ModelWrapper(model_config)
            X_current = X_train_proc

            if "CatBoost" in model_config["class"] and cat_cols:
                    wrapper.model.set_params(cat_features=cat_cols)

            try:
                cv_scores = cross_val_score(
                    wrapper.model,
                    X_current,
                    y_train,
                    cv=cv_stratedy,
                    scoring="balanced_accuracy",
                    n_jobs=-1
                )
                mean_score = np.mean(cv_scores)
            except Exception as e:
                print(f"Error in {model_config['name']}: {e}")
                continue

            scores.append({
                "Model Name": model_config["name"],
                "Model Class": model_config["class"],
                "Metric Score": mean_score,
                "Wrapper": wrapper,
                "Config": model_config,
                "Params": wrapper.model.get_params()  # Zapisujemy domyślne parametry
            })

            print(f"{model_config['name']} → BA = {mean_score:.4f}")

        leaderboard = pd.DataFrame(scores).sort_values(
            by="Metric Score", ascending=False
        ).reset_index(drop=True)

        # ======================================================================
        # STAGE 2: LIGHT OPTIMIZATION (Halving Strategy)
        # ======================================================================
        print("\n--- Stage 2: Light optimization (Top 3) ---")

        top3 = leaderboard.head(3).to_dict("records")

        for row in top3:
            config = row["Config"]
            search_space = config.get("search_space")
            if not search_space:
                print(f" No search space for {config['name']}, skipping optimization.")
                continue


            wrapper = ModelWrapper(config)

            print(f"Optimizing {config['name']}...")
            try:
                search = HalvingRandomSearchCV(
                    wrapper.model,
                    search_space,
                    n_candidates=1000,
                    factor=2,
                    scoring="balanced_accuracy",
                    n_jobs=-1,
                    cv=3,
                    random_state=42,
                    verbose=0
                )

                X_current = X_train_proc

                if "CatBoost" in model_config["class"] and cat_cols:
                        wrapper.model.set_params(cat_features=cat_cols)
                elif "XGBClassifier" in model_config["class"] or "LGBMClassifier" in model_config["class"]:
                    # Używamy wersji z typem 'category'
                    X_current = X_train_cat
                    if "XGBClassifier" in model_config["class"]:
                        wrapper.model.set_params(enable_categorical=True, tree_method="hist")

                search.fit(X_current, y_train)
                
                if search.best_score_ > row["Metric Score"]:
                    # Ważne: Tworzymy nowy wrapper i ustawiamy mu najlepsze parametry
                    optimized_wrapper = ModelWrapper(config)
                    optimized_wrapper.model.set_params(**search.best_params_)

                    scores.append({
                        "Model Name": f"{config['name']} (Opt)",
                        "Model Class": config["class"],
                        "Metric Score": search.best_score_,
                        "Wrapper": optimized_wrapper,
                        "Config": config,
                        "Params": search.best_params_  # Zapisujemy najlepsze parametry
                    })
                    print(f"  -> Improved! BA: {search.best_score_:.4f}")
            except Exception as e:
                print(f" Optimization failed for {config['name']}: {e}")

        # Odśwież leaderboard
        leaderboard = pd.DataFrame(scores).sort_values(by="Metric Score", ascending=False).reset_index(drop=True)

        # ======================================================================
        # STAGE 3: STACKING (OPTIMIZED)
        # ======================================================================
        print("\n--- Stage 3: Stacking Ensemble ---")

        selected = leaderboard.head(3).to_dict("records")

        meta_features = []
        base_wrappers = [] # Zbieramy same obiekty Wrapper, nie słowniki
        ensemble_base_params = {}

        for row in selected:
            wrapper = row["Wrapper"]
            model_name = row["Model Name"]
            ensemble_base_params[model_name] = row["Params"]

            # Decyzja o danych (zwykłe vs category cast)
            X_current = X_train_proc
            if ("XGBClassifier" in row["Config"]["class"] or "LGBMClassifier" in row["Config"]["class"]) and cat_cols:
                X_current = X_train_cat
            
            # Generowanie OOF
            try:
                oof_proba = cross_val_predict(wrapper.model, X_current, y_train, cv=cv, method="predict_proba", n_jobs=-1)[:, 1]
            except:
                oof_proba = cross_val_predict(wrapper.model, X_current, y_train, cv=cv, method="predict", n_jobs=-1)
            
            meta_features.append(oof_proba)

            # Kopia wrappera dla Ensemble
            new_wrapper = ModelWrapper(row["Config"])
            new_wrapper.model.set_params(**wrapper.model.get_params())
            base_wrappers.append(new_wrapper) # Dodajemy wrapper do listy

        X_meta = np.column_stack(meta_features)

        # Trening Meta Modelu
        meta_model = LogisticRegression(class_weight="balanced", solver="lbfgs")

        # Optymalizacja C dla Meta Modelu
        best_meta_score = -np.inf
        best_C = 1.0
        for C in [0.1, 1.0, 10.0]:
            meta_model.set_params(**{'C': C})
            sc = cross_val_score(meta_model, X_meta, y_train, cv=3, scoring="balanced_accuracy").mean()
            if sc > best_meta_score:
                best_meta_score = sc
                best_C = C

        meta_model.set_params(**{'C': best_C})
        meta_model.fit(X_meta, y_train)

        # Optymalizacja Progu
        meta_proba_train = meta_model.predict_proba(X_meta)[:, 1]
        best_thr = 0.5
        best_thr_score = -np.inf
        for thr in np.linspace(0.2, 0.8, 50):
            preds = (meta_proba_train >= thr).astype(int)
            ba = balanced_accuracy_score(y_train, preds)
            if ba > best_thr_score:
                best_thr_score = ba
                best_thr = thr

        ensemble = StackingEnsemble(
            base_models=base_wrappers, 
            preprocessor=self.preprocessor, 
            meta_model=meta_model, 
            threshold=best_thr,
            refit_meta=False 
        )

        # Tworzymy strukture parametrów dla Ensemble
        ensemble_full_params = {
            "type": "ensemble_detailed",
            "meta_params": {"C": best_C, "threshold": best_thr},
            "base_models_params": ensemble_base_params
        }

        scores.append({
            "Model Name": "Stacking Ensemble",
            "Model Class": "Ensemble",
            "Metric Score": best_thr_score,
            "Wrapper": ensemble,
            "Config": {},
            "Params": ensemble_full_params  # Zapisujemy skomplikowaną strukturę
        })

        # ======================================================================
        # FINAL
        # ======================================================================
        leaderboard = pd.DataFrame(scores).sort_values(
            by="Metric Score", ascending=False
        ).reset_index(drop=True)

        best = leaderboard.iloc[0]
        self.leaderboard = leaderboard
        self.best_model = best["Wrapper"]

        print("\n==============================")
        print(f"WINNER: {best['Model Name']}")
        print(f"Balanced Accuracy: {best['Metric Score']:.4f}")
        print("------------------------------")

        # LOGIKA DRUKOWANIA PARAMETRÓW
        params = best["Params"]

        if isinstance(params, dict) and params.get("type") == "ensemble_detailed":
            # Wyświetlanie dla Ensemble
            print(">>> ENSEMBLE STRUCTURE & PARAMETERS <<<")
            print(f"  [Meta-Model] Logistic Regression:")
            print(f"      - C: {params['meta_params']['C']}")
            print(f"      - Threshold: {params['meta_params']['threshold']:.4f}")
            print("\n  [Base Models]:")
            for name, p_dict in params['base_models_params'].items():
                print(f"    * {name}:")
                # Wyświetlamy tylko kluczowe parametry (nie None), żeby nie zaśmiecać,
                # albo wszystkie jeśli wolisz - poniżej wersja skrócona (parametry optymalizowane)
                # Jeśli to dict z HalvingSearch, jest krótki. Jeśli full get_params(), jest długi.
                # Wyświetlamy jako słownik w jednej linii lub ładnie sformatowany
                import json
                try:
                    # Próba ładnego formatowania jeśli parametry są proste
                    print(f"      {p_dict}")
                except:
                    print(f"      {str(p_dict)[:200]}...")  # Przycięcie jeśli za długie
        else:
            # Wyświetlanie dla pojedynczego modelu
            print(">>> BEST MODEL PARAMETERS <<<")
            print(params)

        print("==============================")

        print(f"Final fitting of {best['Model Name']}...")
        
        if isinstance(self.best_model, StackingEnsemble):
            # Ensemble wie, że dostaje przetworzone dane
            self.best_model.fit(X_train_proc, y_train) 
        else:
            # Pojedynczy model
            model_class = best["Config"]["class"]
            
            if ("XGBClassifier" in model_class or "LGBMClassifier" in model_class) and cat_cols:
                # Jeśli wygrał XGB/LGBM, musimy mu dać dane z kategoriami
                self.best_model.fit(X_train_cat, y_train)
            elif "CatBoost" in model_class and cat_cols:
                 self.best_model.model.set_params(cat_features=cat_cols)
                 self.best_model.fit(X_train_proc, y_train)
            else:
                self.best_model.fit(X_train_proc, y_train)

        return self.best_model

    def predict(self, X_test):
        if not self.best_model: raise ValueError("Call fit() first.")

        if isinstance(self.best_model, StackingEnsemble):
            # Ensemble sam sobie robi transform wewnątrz predict
            return self.best_model.predict(X_test)

        # Dla pojedynczego modelu musimy przetworzyć surowe dane
        # Uwaga: używamy transform, nie fit_transform
        X_test_proc = self.preprocessor.transform(X_test, None)
        
        # Obsługa XGBoost przy predykcji pojedynczego modelu
        cat_cols = self.preprocessor.get_categorical_cols(X_test_proc)
        if ("XGBClassifier" in self.best_model.model.__class__.__name__ or 
            "LGBMClassifier" in self.best_model.model.__class__.__name__) and cat_cols:
             for col in cat_cols:
                X_test_proc[col] = X_test_proc[col].astype("category")
                
        return self.best_model.model.predict(X_test_proc)

    def predict_proba(self, X_test):
        if not self.best_model: raise ValueError("Call fit() first.")

        if isinstance(self.best_model, StackingEnsemble):
            return self.best_model.predict_proba(X_test)[:, 1]

        X_test = self.preprocessor.transform(X_test, None)
        return self.best_model.predict_proba(X_test)[:, 1]

    def display_leaderboard(self, mode="short"):
        if self.leaderboard is None: raise ValueError("No leaderboard.")
        print("================ Leaderboard ================")
        if mode == "short":
             cols = ["Model Name", "Metric Score"]
             return self.leaderboard[cols]
        else:
             return self.leaderboard