import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.calibration import LabelEncoder
from sklearn.preprocessing import PolynomialFeatures, StandardScaler, PowerTransformer, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA


import warnings

class AutoMLPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self, target_col=None, add_kmeans_features=True,
                 feature_selection_method='pca', # opcje to None, 'pca' lub 'sfs'
                 n_features=0.25,                # <--- NOWOŚĆ: domyślnie 25% (0.25)
                 add_poly_features=False, remove_outliers=True, 
                 remove_multicollinearity=True, multicollinearity_threshold=0.95,
                 random_state=None):
        
        """
        Kompleksowy preprocesor AutoML, integrujący czyszczenie danych, inżynierię cech
        oraz redukcję wymiarowości.

        Pipeline przetwarzania:
        1. Detekcja typów (liczby, kategorie, daty).
        2. Przetwarzanie dat (cykliczne sin/cos).
        3. Imputacja braków (mediana/moda).
        4. Transformacja Yeo-Johnson i skalowanie.
        5. Generowanie cech (KMeans, Interakcje).
        6. Usuwanie outlierów.
        7. Selekcja cech (PCA/SFS).
        
        Przykładowe użycie:
        
        Processor = AutoMLPreprocessor(feature_selection_method='sfs', add_poly_features=True)
        X_train, X_test, y_train, y_test = Processor.process(X, y)


        Parametry:
        ----------
        target_col : str, opcjonalnie (domyślnie=None)
            Nazwa kolumny docelowej. Jeśli None, klasa spróbuje ją wywnioskować z przekazanego y.

        add_kmeans_features : bool (domyślnie=True)
            Czy generować cechy oparte na klastrowaniu (MiniBatchKMeans).
            Dodaje kolumny z dystansem do centroidów oraz ID klastra.
            Pomaga modelom wykrywać nieliniowe grupy w danych.

        feature_selection_method : {None, 'pca', 'sfs'} (domyślnie='pca')
            
            Metoda redukcji liczby cech:
            - None: brak selekcji cech.
            - 'pca': Szybka redukcja wymiarowości. Tworzy nowe, syntetyczne cechy (PC1, PC2...),
              które maksymalizują wariancję. Zalecane przy dużej liczbie kolumn.
            - 'sfs': Sequential Feature Selection (Backward). Wolniejsza, ale wybiera
              najlepsze *oryginalne* kolumny. Zachowuje interpretowalność biznesową.

        n_features : float lub int (domyślnie=0.25)
            Ile cech zachować po selekcji:
            - float (0.0 - 1.0): Procent początkowych kolumn (np. 0.25 to 25%).
            - int (> 1): Dokładna liczba kolumn do pozostawienia.

        add_poly_features : bool (domyślnie=False)
            Czy tworzyć interakcje między zmiennymi numerycznymi (A*B).
            Może znacznie poprawić wynik, ale generuje dużo nowych kolumn,
            więc zaleca się używanie tego łącznie z `select_features=True`.

        remove_outliers : bool (domyślnie=True)
            Czy usuwać obserwacje odstające (outliery) ze zbioru treningowego
            przy użyciu algorytmu IsolationForest.

        random_state : int, opcjonalnie (domyślnie=None)
            Ziarno losowości dla zapewnienia powtarzalności wyników 
            (dla KMeans, PCA, SFS, IsolationForest).
        """

        self.random_state = random_state
        self.is_fitted = False
        
        self.target_col = target_col
        self.remove_outliers = remove_outliers
        
        # Column types
        self.num_cols = []
        self.cat_cols = []
        self.date_cols = []
        
        # Imputers and transformers
        self.imputer_num = SimpleImputer(strategy='median')
        self.imputer_cat = SimpleImputer(strategy='most_frequent')
        self.power_transformer = PowerTransformer(method='yeo-johnson', standardize=True)
        self.cat_encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)

        # --- Nowe pola dla KMeans ---
        self.add_kmeans_features = add_kmeans_features
        self.kmeans_model = None
        self.kmeans_scaler = StandardScaler() # Osobny skaler tylko dla KMeans (zgodnie z ideą mljar)
        self.kmeans_cols = [] # Lista nazw nowych kolumn

        # --- Nowe pola dla interakcji ---
        self.add_poly_features = add_poly_features
        self.poly_transformer = None

        # Target processing
        self.imputer_y = SimpleImputer(strategy='most_frequent')
        self.encoder_y = LabelEncoder()

        # Feature Selection
        # Parametry selekcji
        self.feature_selection_method = feature_selection_method # <--- Przypisanie
        self.n_features = n_features                             # <--- Przypisanie
        self.selector = None

        # --- Współliniowość ---
        self.remove_multicollinearity = remove_multicollinearity
        self.multicollinearity_threshold = multicollinearity_threshold
        self.collinear_drop_cols = [] # Lista kolumn do usunięcia

    def process(self, X, y=None):
        # 1. Scenariusz: Użytkownik podaje tylko X (dataframe z targetem), y=None
        if y is None:
            if self.target_col is not None and self.target_col in X.columns:
                y = X[self.target_col]
            elif self.target_col is not None:
                raise Exception("Nie podano y ani nie znaleziono target_col w X!")
            else:
                raise Exception("Nie podano y ani target_col! Nie można wywnioskować targetu.")
            # Jeśli target_col jest None i y jest None -> Nie mamy targetu. 

        # 2. Scenariusz: Użytkownik podał y, ale nie podał nazwy target_col w __init__
        if self.target_col is None and y is not None:
            if hasattr(y, 'name'): 
                self.target_col = y.name
            elif hasattr(y, 'columns') and len(y.columns) > 0: 
                self.target_col = y.columns[0]
            else:
                self.target_col = 'target'
        
        # 3. Usuwamy target z X (jeśli tam jest), żeby nie było wycieku danych (Data Leakage)
        if self.target_col and self.target_col in X.columns:
            X = X.drop(columns=[self.target_col], errors='ignore')

        # 4. Split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=self.random_state
        )
        
        X_train, y_train = self.fit_transform(X_train, y_train)
        X_test, y_test = self.transform(X_test, y_test)
        
        return X_train, X_test, y_train, y_test


    def _detect_columns(self, X):
        """Automatyczne wykrywanie typów kolumn (zabezpieczone i wyciszone)."""
        self.date_cols = []
        temp_X = X.copy()
        
        # Iterujemy po potencjalnych kolumnach (object i datetime)
        for col in temp_X.select_dtypes(include=['object', 'datetime']).columns:
            try:
                # 1. Sprawdzamy czy to nie są liczby zapisane jako tekst
                if temp_X[col].dtype == 'object' and temp_X[col].astype(str).str.isnumeric().all():
                    continue
                
                # 2. Próba konwersji na datę
                # Używamy catch_warnings, aby ignorować komunikat "Could not infer format..."
                # Ponieważ właśnie tego chcemy - sprawdzić czy Pandas poradzi sobie ze "zgadywaniem".
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    
                    # Optymalizacja: Sprawdzamy tylko próbkę (np. 100 wierszy) zamiast całej kolumny.
                    # Jeśli próbka jest datą, to cała kolumna prawdopodobnie też.
                    # To znacznie przyspiesza działanie na dużych danych.
                    sample = temp_X[col].dropna().iloc[:100]
                    if len(sample) > 0:
                        pd.to_datetime(sample, errors='raise')
                    else:
                        # Jeśli pusta kolumna, pomijamy
                        continue
                
                # Jeśli przeszło bez błędu, uznajemy to za datę
                self.date_cols.append(col)
                
            except (ValueError, TypeError):
                # Jeśli pd.to_datetime rzuci błąd, to nie jest data
                continue
        
        remaining = X.drop(columns=self.date_cols)
        self.cat_cols = remaining.select_dtypes(include=['object', 'category']).columns.tolist()
        self.num_cols = remaining.select_dtypes(include=['number']).columns.tolist()

    def _process_dates_cyclical(self, df):
        df_out = df.copy()
        for col in self.date_cols:
            if col in df_out.columns:
                dates = pd.to_datetime(df_out[col], errors='coerce')
                cycles = {
                    'month': (dates.dt.month, 12),
                    'day': (dates.dt.day, 31),
                    'dayofweek': (dates.dt.dayofweek, 7),
                    'dayofyear': (dates.dt.dayofyear, 365)
                }
                for part, (values, period) in cycles.items():
                    df_out[f'{col}_{part}_sin'] = np.sin(2 * np.pi * values / period)
                    df_out[f'{col}_{part}_cos'] = np.cos(2 * np.pi * values / period)
                
                df_out[f'{col}_year'] = dates.dt.year
                df_out[f'{col}_is_weekend'] = (dates.dt.dayofweek >= 5).astype(int)
                df_out.drop(columns=[col], inplace=True)
        return df_out

    def _process_target(self, y, fit=False):
        """
        Jeden pipeline dla targetu: Standaryzacja typu -> Imputacja -> Encoding.
        Obsługuje: bool, int, str, object oraz braki danych.
        """
        if y is None:
            return None
        
        # 1. Konwersja na Pandas Series (ułatwia obsługę typów i NaN)
        y = np.ravel(y)
        y_s = pd.Series(y)
        
        # 2. Naprawa błędu "dtype bool":
        # Jeśli dane są boolami (lub zawierają boole i NaNy), zamieniamy je na spójny format.
        # Najbezpieczniej zamienić wszystko na Stringi, ale musimy uważać, 
        # bo astype(str) zamienia np.nan na napis "nan".
        
        # Krok A: Zamiana na stringi (True -> "True", 1 -> "1", NaN -> "nan")
        y_s = y_s.astype(str)
        
        # Krok B: Przywrócenie prawdziwych NaN-ów (żeby Imputer wiedział co uzupełniać)
        # Traktujemy napisy 'nan', 'None', '<NA>' jako braki danych.
        y_s = y_s.replace({'nan': np.nan, 'None': np.nan, '<NA>': np.nan})
        
        # 3. Konwersja do numpy array 2D (wymóg SimpleImputera)
        y_vals = y_s.values.reshape(-1, 1)
        
        # 4. Pipeline Imputacja -> Encoding
        if fit:
            # Uczymy się najczęstszej wartości (imputer) i mapowania klas (encoder)
            y_filled = self.imputer_y.fit_transform(y_vals).ravel()
            y_encoded = self.encoder_y.fit_transform(y_filled)
        else:
            # Tylko transformujemy
            y_filled = self.imputer_y.transform(y_vals).ravel()
            try:
                y_encoded = self.encoder_y.transform(y_filled)
            except ValueError:
                # Fallback: jeśli w nowych danych pojawi się klasa, której nie było w treningu
                # (np. "Maybe" zamiast "True"/"False"), zwracamy wypełnione wartości surowe 
                # lub (lepiej) domyślną klasę 0.
                print("Uwaga: Nieznana klasa w targecie. Mapuję na 0.")
                # Opcja bezpieczna: zwracamy zera (zakładając binarną klasyfikację)
                return np.zeros(len(y_filled), dtype=int)
                
        return y_encoded

    def _fit_kmeans(self, X):
        """Logika uczenia KMeans inspirowana biblioteką MLJAR."""
        # KMeans działa tylko na liczbach. Bierzemy te, które już mamy wykryte.
        # Ważne: używamy num_cols, które są już w X (czyli po imputacji).
        valid_cols = [c for c in self.num_cols if c in X.columns]
        
        if not valid_cols or X.shape[0] < 10: # Zabezpieczenie dla małych danych
            print("Pominięto KMeans (brak kolumn numerycznych lub za mało danych).")
            self.add_kmeans_features = False
            return

        # 1. Skalowanie danych (zgodnie z MLJAR używamy StandardScalera przed KMeans)
        X_subset = X[valid_cols].values
        self.kmeans_scaler.fit(X_subset)
        X_scaled = self.kmeans_scaler.transform(X_subset)
        
        # 2. Wybór liczby klastrów (heurystyka MLJAR)
        n_clusters = int(np.log10(X.shape[0]) * 8)
        n_clusters = max(2, n_clusters)      # Minimum 2 klastry
        n_clusters = min(n_clusters, 15)     # Ograniczamy max (żeby nie zrobiło 100 kolumn)
        # MLJAR robi min(n, X.shape[1]), ale tutaj bezpieczniej dać sztywny limit górny dla wydajności
        
        # 3. Fitowanie modelu
        self.kmeans_model = MiniBatchKMeans(
            n_clusters=n_clusters, 
            init="k-means++", 
            batch_size=256,
            random_state=self.random_state,
            n_init='auto'
        )
        self.kmeans_model.fit(X_scaled)
        
        # 4. Zapamiętanie nazw nowych cech
        self.kmeans_cols = [f"Dist_Cluster_{i}" for i in range(n_clusters)] + ["Cluster"]
        print(f"--- KMeans: Wytrenowano {n_clusters} klastrów ---")

    def _transform_with_kmeans(self, X):
        """Aplikuje KMeans i dokleja nowe kolumny do X."""
        if not self.add_kmeans_features or self.kmeans_model is None:
            return X
        
        X_out = X.copy()
        valid_cols = [c for c in self.num_cols if c in X_out.columns]
        
        if not valid_cols:
            return X_out

        # Skalowanie i predykcja
        X_scaled = self.kmeans_scaler.transform(X_out[valid_cols].values)
        
        distances = self.kmeans_model.transform(X_scaled)
        clusters = self.kmeans_model.predict(X_scaled)
        
        # Dodawanie kolumn do DataFrame
        # 1. Dystanse do centroidów
        dist_cols = self.kmeans_cols[:-1] # Wszystkie oprócz ostatniego ('Cluster')
        X_out[dist_cols] = distances
        
        # 2. ID Klastra (jako kategoria/int)
        X_out["Cluster"] = clusters
        
        return X_out

    def _add_interactions(self, X):
        """Tworzy interakcje między zmiennymi numerycznymi (mnożenie)."""
        if not self.add_poly_features:
            return X
            
        # Bierzemy tylko numeryczne, żeby nie mnożyć kategorii
        valid_cols = [c for c in self.num_cols if c in X.columns]
        
        # Jeśli mamy za dużo kolumn, to PolynomialFeatures wybuchnie.
        # Ograniczmy się np. do 10-15 najważniejszych lub po prostu wszystkich jeśli jest ich mało.
        if len(valid_cols) > 20:
            # Wersja prosta: bierzemy pierwsze 20 (można tu dodać logikę wyboru np. wariancji)
            cols_to_poly = valid_cols[:20]
        else:
            cols_to_poly = valid_cols

        if not cols_to_poly:
            return X

        X_poly_in = X[cols_to_poly].values
        
        # Tworzymy transformator tylko przy fit
        if self.poly_transformer is None:
            # degree=2: tworzy A^2, A*B, B^2
            # interaction_only=True: tworzy tylko A*B (bez kwadratów), to często lepsze i lżejsze
            self.poly_transformer = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
            self.poly_transformer.fit(X_poly_in)
            
        # Transformacja
        X_poly_out = self.poly_transformer.transform(X_poly_in)
        
        # Nazwy nowych cech
        new_feature_names = self.poly_transformer.get_feature_names_out(cols_to_poly)
        
        # Tworzymy DataFrame i łączymy z oryginałem
        # Uwaga: PolynomialFeatures zwraca też oryginalne kolumny (x, y), a potem (x*y).
        # Żeby nie dublować, bierzemy tylko te nowe, które mają znak mnożenia " " (spacja w sklearn) lub "*"
        
        X_poly_df = pd.DataFrame(X_poly_out, columns=new_feature_names, index=X.index)
        
        # Filtrujemy, żeby zostawić tylko nowe interakcje (te, których nie ma w X)
        new_cols = [c for c in new_feature_names if c not in X.columns]
        
        if new_cols:
            X = pd.concat([X, X_poly_df[new_cols]], axis=1)
            
        return X

    def _remove_collinear(self, X):
        """Usuwa kolumny silnie skorelowane ze sobą."""
        if not self.remove_multicollinearity:
            return X
            
        # Obliczamy macierz korelacji (wartość bezwzględna, bo -0.99 to też silna korelacja)
        # Robimy to tylko dla kolumn numerycznych
        numeric_df = X.select_dtypes(include=['number'])
        
        if numeric_df.empty:
            return X

        corr_matrix = numeric_df.corr().abs()

        # Wybieramy górny trójkąt macierzy (żeby nie sprawdzać A z B i B z A)
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

        # Znajdujemy kolumny, które mają korelację większą niż próg
        to_drop = [column for column in upper.columns if any(upper[column] > self.multicollinearity_threshold)]
        
        # Zapisujemy do self.collinear_drop_cols, żeby w transform usunąć te same
        self.collinear_drop_cols = to_drop
        
        if to_drop:
            print(f"--- Współliniowość: Usunięto {len(to_drop)} kolumn (korelacja > {self.multicollinearity_threshold}) ---")
            # print(f"-> Usunięte: {to_drop}") # Opcjonalnie wypisz nazwy
            
        return X.drop(columns=to_drop)

    def fit(self, X, y):
        # Zabezpieczenie na wypadek gdyby fit było wołane ręcznie bez process()
        if self.target_col is None and y is not None:
             if hasattr(y, 'name'): self.target_col = y.name
             else: self.target_col = 'target'

        X = X.copy()
        y_proc = self._process_target(y, fit=True)
        
        # 1. Wykrywanie typów i daty
        self._detect_columns(X)
        X = self._process_dates_cyclical(X)
        
        current_cols = X.columns
        self.num_cols = [c for c in current_cols if c not in self.cat_cols]

        if self.num_cols:
            self.imputer_num.fit(X[self.num_cols])
            X[self.num_cols] = self.imputer_num.transform(X[self.num_cols])
        if self.cat_cols:
            self.imputer_cat.fit(X[self.cat_cols])
            X[self.cat_cols] = self.imputer_cat.transform(X[self.cat_cols])

        if self.num_cols:
            self.power_transformer.fit(X[self.num_cols])
            X[self.num_cols] = self.power_transformer.transform(X[self.num_cols])

        if self.cat_cols:
            X[self.cat_cols] = X[self.cat_cols].astype(str)
            self.cat_encoder.fit(X[self.cat_cols])
            X[self.cat_cols] = self.cat_encoder.transform(X[self.cat_cols])

        # --- 2. Generowanie Cech (Feature Engineering) ---
        
        # A. KMeans
        if self.add_kmeans_features:
            self._fit_kmeans(X)
            X = self._transform_with_kmeans(X)
            self.num_cols.extend(self.kmeans_cols)

        # B. Interakcje (na podstawie num_cols + kmeans_cols)
        if self.add_poly_features:
            self.poly_transformer = None 
            X = self._add_interactions(X)
            # Aktualizujemy listę numerycznych o nowe interakcje
            # (bierzemy różnicę zbiorów, żeby nie dodać czegoś dwa razy)
            all_num = X.select_dtypes(include=['number']).columns.tolist()
            new_poly = [c for c in all_num if c not in self.num_cols]
            self.num_cols.extend(new_poly)

        # --- 3. Czyszczenie (Współliniowość) ---
        # Robimy to PO wygenerowaniu wszystkiego, żeby usunąć np. interakcje skorelowane z oryginałem
        if self.remove_multicollinearity:
            X = self._remove_collinear(X)
            # WAŻNE: Aktualizacja self.num_cols po usunięciu
            self.num_cols = [c for c in self.num_cols if c not in self.collinear_drop_cols]

        # --- 4. Selekcja Cech (PCA / SFS) ---
        if self.feature_selection_method is not None:
            X_full = X.values
            n_current_features = X.shape[1]
            
            if isinstance(self.n_features, float):
                n_to_keep = int(n_current_features * self.n_features)
            else:
                n_to_keep = self.n_features
            n_to_keep = max(1, min(n_to_keep, n_current_features))

            if self.feature_selection_method == 'pca':
                print(f"\n--- Uruchamianie redukcji PCA ---")
                print(f"-> Cechy przed: {n_current_features}, Cel: {n_to_keep}")
                self.selector = PCA(n_components=n_to_keep, random_state=self.random_state)
                self.selector.fit(X_full)
                
            elif self.feature_selection_method == 'sfs':
                if y_proc is None:
                    print("Uwaga: Brak targetu (y), pomijam SFS.")
                else:
                    print(f"\n--- Uruchamianie selekcji SFS (Backward) ---")
                    print(f"-> Estymator bazowy: LogisticRegression (balanced)")
                    print(f"-> Cechy na wejściu: {n_current_features}")
                    
                    est = LogisticRegression(class_weight='balanced', solver='liblinear', random_state=self.random_state)
                    
                    self.selector = SequentialFeatureSelector(
                        est, 
                        direction='backward',
                        scoring='roc_auc',
                        cv=3,
                        n_jobs=-1
                        # n_features_to_select='auto' (domyślnie redukuje o połowę lub wg tolerancji)
                    )
                    
                    # 1. Fitowanie selektora
                    self.selector.fit(X_full, y_proc)
                    
                    # 2. Analiza wyników
                    support = self.selector.get_support()
                    selected_cols = X.columns[support].tolist()
                    dropped_cols = X.columns[~support].tolist()
                    n_selected = len(selected_cols)
                    
                    print(f"-> SFS zakończony sukcesem.")
                    print(f"-> Pozostawiono: {n_selected} cech (usunięto {len(dropped_cols)})")
                    
                    # 4. Wypisanie nazw (jeśli nie jest ich tysiąc)
                    if n_selected <= 50:
                        print(f"-> [LISTA] Wybrane cechy: {selected_cols}")
                    else:
                        print(f"-> [LISTA] Wybrane cechy (pierwsze 50): {selected_cols[:50]}...")
                        
                    if dropped_cols and len(dropped_cols) <= 50:
                        print(f"-> [LISTA] Odrzucone cechy: {dropped_cols}")

        self.is_fitted = True
        return self

    def transform(self, X, y=None): # <--- POPRAWKA: y=None
        if not self.is_fitted:
            raise Exception("Najpierw uruchom fit()!")
        
        X = X.copy()
        y_transformed = self._process_target(y, fit=False)
        
        X = self._process_dates_cyclical(X)
        
        # Imputacja/Skalowanie (tylko dla kolumn z num_cols, które przetrwały w fit)
        # Uwaga: valid_num musi sprawdzać self.num_cols, które w fit() zostały już "oczyszczone"
        # z kolumn współliniowych. Jednak w surowym X te kolumny wciąż są.
        # Dlatego najpierw przetwarzamy to co mamy w X, a potem dropujemy.
        
        # Musimy wiedzieć które kolumny numeryczne przetworzyć.
        # Imputery mają zapisane swoje kolumny (feature_names_in_ w nowszych sklearn),
        # ale my używamy self.num_cols. 
        # Trikiem jest to, że self.num_cols w fit() zostało pomniejszone o collinear_drop_cols.
        # Ale imputery były uczone NA PEŁNYM zestawie przed dropowaniem.
        
        # Bezpieczniej jest użyć kolumn na których imputery były uczone:
        if hasattr(self.imputer_num, "feature_names_in_"):
             cols_to_impute = self.imputer_num.feature_names_in_
        else:
             # Fallback jeśli starszy sklearn - bierzemy te z X co pasują do typów numerycznych
             # (to uproszczenie, ale zazwyczaj działa)
             cols_to_impute = X.select_dtypes(include=['number']).columns
        
        # Filtrujemy tylko te które są w obecnym X
        valid_impute = [c for c in cols_to_impute if c in X.columns]
        if len(valid_impute) > 0:
            X[valid_impute] = self.imputer_num.transform(X[valid_impute])
            X[valid_impute] = self.power_transformer.transform(X[valid_impute])

        if self.cat_cols:
            valid_cat = [c for c in self.cat_cols if c in X.columns]
            if valid_cat:
                X[valid_cat] = self.imputer_cat.transform(X[valid_cat])
                X[valid_cat] = X[valid_cat].astype(str)
                X[valid_cat] = self.cat_encoder.transform(X[valid_cat])

        # --- Generowanie Cech (Kolejność jak w fit!) ---
        
        if self.add_kmeans_features:
            X = self._transform_with_kmeans(X)

        if self.add_poly_features:
            X = self._add_interactions(X)

        # --- Czyszczenie (Współliniowość) ---
        if self.remove_multicollinearity and self.collinear_drop_cols:
            # Usuwamy te same kolumny co w fit
            cols_to_drop = [c for c in self.collinear_drop_cols if c in X.columns]
            X = X.drop(columns=cols_to_drop)

        # --- Selekcja ---
        if self.selector:
            X_values = X.values
            X_transformed = self.selector.transform(X_values)
            
            if self.feature_selection_method == 'pca':
                cols = [f"PC{i+1}" for i in range(X_transformed.shape[1])]
                X = pd.DataFrame(X_transformed, columns=cols, index=X.index)
            else:
                support = self.selector.get_support()
                cols = X.columns[support]
                X = pd.DataFrame(X_transformed, columns=cols, index=X.index)
    
        return X, y_transformed
    
    def fit_transform(self, X, y):
        """Uczy się, usuwa outliery (opcjonalnie) i zwraca gotowe dane."""
        # 1. Nauka parametrów (średnie, odchylenia, wagi SFS itd.)
        self.fit(X, y)
        
        # 2. Przygotowanie kopii roboczych
        X_proc = X.copy()
        # Zabezpieczenie: y może być None (np. w unsupervised), choć tu rzadko
        y_proc = y.copy() if y is not None else None
        
        # 3. Wykrywanie Outlierów (OPCJONALNE)
        if self.remove_outliers and self.num_cols:
            # Tworzymy TYMCZASOWĄ wersję danych tylko dla algorytmu IsolationForest.
            # Musimy zamienić daty i braki na liczby, żeby algorytm zadziałał.
            temp_X = X_proc.copy()
            temp_X = self._process_dates_cyclical(temp_X)
            
            valid_num = [c for c in self.num_cols if c in temp_X.columns]
            if valid_num:
                temp_X[valid_num] = self.imputer_num.transform(temp_X[valid_num])
                temp_X[valid_num] = self.power_transformer.transform(temp_X[valid_num])
            
            # Wykrywamy outliery
            iso = IsolationForest(contamination=0.05, random_state=42, n_jobs=-1)
            preds = iso.fit_predict(temp_X[valid_num])
            
            # Filtrujemy SUROWE dane (X_proc)
            mask = preds != -1
            print(f"Usunięto {sum(preds == -1)} wierszy (outliery).")
            
            X_proc = X_proc[mask]
            if y_proc is not None:
                # Uwaga: y musi być typu pandas Series/DataFrame lub numpy array, żeby obsłużyć maskowanie
                # Jeśli y to lista, trzeba zamienić: np.array(y_proc)[mask]
                if isinstance(y_proc, list):
                    y_proc = np.array(y_proc)[mask]
                else:
                    y_proc = y_proc[mask]

        # 4. FINALNE PRZETWARZANIE
        # Tutaj dzieje się magia: transform() bierze X_proc (który ma wciąż surowe daty),
        # przetwarza daty, imputuje braki, skaluje i selekcjonuje cechy.
        # Działa to niezależnie od tego, czy outliery były usuwane, czy nie.
        return self.transform(X_proc, y_proc)