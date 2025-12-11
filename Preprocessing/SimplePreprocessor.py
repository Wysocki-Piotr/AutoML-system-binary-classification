import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder

class SimplePreprocessor:
    """
    Najprostszy processing: uzupełnianie braków, skalowanie numerycznych, kodowanie kategorycznych.
    """
    def __init__(self):
        # Magazyn wiedzy (statystyki z treningu)
        self.means = {}
        self.modes = {}
        self.cat_encoders = {} # Słownik: {kolumna: {'encoder': le, 'classes': set(...)}}
        self.y_encoder = None
        self.scaler = StandardScaler()
        
        self.cat_cols = []
        self.num_cols = []
        self.is_fitted = False

    def fit(self, X, y):
        """
        Tylko 'patrzy' na dane i zapamiętuje statystyki. Nie zwraca zmienionych danych.
        """
        # 1. Automatyczna detekcja kolumn
        self.cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
        self.num_cols = X.select_dtypes(include=['number']).columns.tolist()

        # 2. Target Encoder
        self.y_encoder = LabelEncoder()
        self.y_encoder.fit(y)

        # 3. Zmienne Numeryczne: Zapamiętaj średnie i naucz skaler
        for col in self.num_cols:
            self.means[col] = X[col].mean()
        
        # Uczymy skaler na tymczasowej kopii (żeby nie psuć X w miejscu)
        if self.num_cols:
            X_num_temp = X[self.num_cols].fillna(X[self.num_cols].mean())
            self.scaler.fit(X_num_temp)
            del X_num_temp

        # 4. Zmienne Kategoryczne: Zapamiętaj mody i naucz encodery
        for col in self.cat_cols:
            mode_val = X[col].mode()[0] if not X[col].mode().empty else 'missing'
            self.modes[col] = mode_val
            
            le = LabelEncoder()
            # Uczymy na uzupełnionych danych rzutowanych na string
            col_data = X[col].fillna(mode_val).astype(str)
            le.fit(col_data)
            
            self.cat_encoders[col] = {
                'encoder': le,
                'classes': set(le.classes_)
            }
            
        self.is_fitted = True
        return self

    def transform(self, X, y):
        """
        Aplikuje wiedzę z fit() na dowolny zbiór (treningowy lub testowy).
        Zwraca NOWE, zmodyfikowane ramki danych.
        Zrobione do użytku na zbiorze testowym.
        """
        if not self.is_fitted:
            raise Exception("Najpierw użyj fit() na zbiorze treningowym!")
        
        X_trans = X.copy()
        y_trans = y.copy()
        
        # 1. Numeryczne
        for col in self.num_cols:
            if col in X_trans.columns:
                X_trans[col] = X_trans[col].fillna(self.means[col])
        
        if self.num_cols:
            X_trans[self.num_cols] = self.scaler.transform(X_trans[self.num_cols])

        # 2. Kategoryczne
        for col in self.cat_cols:
            if col in X_trans.columns:
                mode_val = self.modes[col]
                # Uzupełnianie i rzutowanie
                X_trans[col] = X_trans[col].fillna(mode_val).astype(str)
                
                # Bezpieczne kodowanie (nieznane wartości -> moda)
                le = self.cat_encoders[col]['encoder']
                known_classes = self.cat_encoders[col]['classes']
                
                # Szybka funkcja mapująca (lambda)
                X_trans[col] = X_trans[col].apply(lambda x: x if x in known_classes else mode_val)
                X_trans[col] = le.transform(X_trans[col])

        # 3. Target
        if self.y_encoder is None:
            raise Exception("Nie nauczono encodera dla y (fit wywołano bez y).")
        y_trans = self.y_encoder.transform(y_trans)
        
        return X_trans, y_trans

    def fit_transform(self, X, y):
        """
        Uczy się I od razu zwraca zmieniony zbiór.
        Zrobione do użycia na zbiorze treningowym.
        """
        return self.fit(X, y).transform(X, y)