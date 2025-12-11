import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder

from sklearn.preprocessing import PowerTransformer, PolynomialFeatures
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier

from .SimplePreprocessor import SimplePreprocessor


class AdvancedPreprocessor(SimplePreprocessor):
    def __init__(self, target_col='target', select_features=True):
        super().__init__(target_col) # Dziedziczymy logikę init z Simple
        self.select_features = select_features
        self.poly = None
        self.selector = None
        self.pt = PowerTransformer()
        self.selected_cols = None

    def fit(self, df):
        # 1. Najpierw wykonujemy podstawowy fit (średnie, mody)
        super().fit(df)
        
        # Pobieramy wstępnie przetworzone dane (aby na nich uczyć kolejne kroki)
        X_basic, y = self.transform(df) 
        
        # Jeśli nie ma y (np. unsupervised), nie możemy robić selekcji opartej na modelu
        if y is None:
             raise ValueError("Advanced preprocessing wymaga kolumny target w metodzie fit!")

        # 2. Power Transformer (Normalizacja rozkładu zmiennych numerycznych)
        if self.num_cols:
            self.pt.fit(X_basic[self.num_cols])
            # Transformujemy numeryczne do dalszych kroków
            X_basic[self.num_cols] = self.pt.transform(X_basic[self.num_cols])

        # 3. Polynomial Features (Tworzenie interakcji)
        # Ograniczamy do numerycznych, degree=2
        if self.num_cols:
            self.poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
            self.poly.fit(X_basic[self.num_cols])
            
            # Musimy wygenerować te cechy teraz, żeby selector mógł je ocenić
            poly_vals = self.poly.transform(X_basic[self.num_cols])
            poly_cols = self.poly.get_feature_names_out(self.num_cols)
            X_poly_df = pd.DataFrame(poly_vals, columns=poly_cols, index=X_basic.index)
            
            # Łączymy z resztą (zastępujemy stare numeryczne lub dodajemy - tu dodajemy)
            X_full = pd.concat([X_basic, X_poly_df], axis=1)
        else:
            X_full = X_basic

        # 4. Feature Selection (Wybór najlepszych cech)
        if self.select_features:
            rf = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42, n_jobs=-1)
            self.selector = SelectFromModel(rf, threshold="median") # Wybierz 50% najlepszych
            self.selector.fit(X_full, y)
            
            # Zapamiętujemy nazwy wybranych kolumn
            support = self.selector.get_support()
            self.selected_cols = X_full.columns[support].tolist()
            
        return self

    def transform(self, df):
        # 1. Podstawowy transform
        X_basic, y = super().transform(df)
        
        # 2. Power Transformer
        if self.num_cols:
            X_basic[self.num_cols] = self.pt.transform(X_basic[self.num_cols])
            
        # 3. Polynomial Features
        if self.num_cols and self.poly:
            poly_vals = self.poly.transform(X_basic[self.num_cols])
            poly_cols = self.poly.get_feature_names_out(self.num_cols)
            X_poly_df = pd.DataFrame(poly_vals, columns=poly_cols, index=X_basic.index)
            X_full = pd.concat([X_basic, X_poly_df], axis=1)
        else:
            X_full = X_basic
            
        # 4. Feature Selection
        if self.select_features and self.selector:
            # Zwracamy tylko wybrane kolumny
            X_final = X_full[self.selected_cols]
            return X_final, y
            
        return X_full, y