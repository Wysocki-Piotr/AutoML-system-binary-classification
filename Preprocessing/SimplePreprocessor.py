import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder

class SimplePreprocessor:
    def __init__(self, target_col='target'):
        self.target_col = target_col
        
        # Słowniki do przechowywania "wiedzy" z treningu
        self.means = {}           # Średnie dla zmiennych ciągłych
        self.modes = {}           # Mody dla zmiennych kategorycznych
        self.cat_encoders = {}    # Mapowania kategoryczne (LabelEncoders)
        self.scaler = StandardScaler()
        
        self.cat_cols = []
        self.num_cols = []
        self.is_fitted = False

    def fit(self, df):
        """
        Uczy się statystyk (średnie, kategorie) TYLKO na zbiorze treningowym.
        """
        X = df.drop(columns=[self.target_col])
        
        # 1. Automatyczna detekcja kolumn
        self.cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
        self.num_cols = X.select_dtypes(include=['number']).columns.tolist()

        # 2. Nauka na zmiennych numerycznych
        for col in self.num_cols:
            self.means[col] = X[col].mean()
        
        # Wstępne uzupełnienie do fitowania skalera
        X_num = X[self.num_cols].fillna(X[self.num_cols].mean())
        self.scaler.fit(X_num)

        # 3. Nauka na zmiennych kategorycznych
        for col in self.cat_cols:
            # Zapamiętujemy modę
            mode_val = X[col].mode()[0] if not X[col].mode().empty else 'missing'
            self.modes[col] = mode_val
            
            # Wypełniamy braki modą tymczasowo do nauki enkodera
            X[col] = X[col].fillna(mode_val).astype(str)
            
            # Uczymy LabelEncodera
            le = LabelEncoder()
            le.fit(X[col])
            
            # Zapisujemy klasy, żeby obsłużyć nieznane wartości w transform
            self.cat_encoders[col] = {
                'encoder': le,
                'classes': set(le.classes_)
            }
            
        self.is_fitted = True
        return self

    def transform(self, df):
        """
        Aplikuje zapamiętane statystyki na dane (treningowe lub testowe).
        """
        if not self.is_fitted:
            raise Exception("Najpierw uruchom fit() na zbiorze treningowym!")
            
        data = df.copy()
        
        # Obsługa targetu (jeśli istnieje, np. w treningu/teście, ale nie na produkcji)
        y = None
        if self.target_col in data.columns:
            y = data[self.target_col].values # Zwracamy jako numpy array
            data = data.drop(columns=[self.target_col])
        
        # 1. Numeryczne: Uzupełnianie średnią z treningu + Skalowanie
        for col in self.num_cols:
            if col in data.columns:
                data[col] = data[col].fillna(self.means[col])
        
        if self.num_cols:
            data[self.num_cols] = self.scaler.transform(data[self.num_cols])

        # 2. Kategoryczne: Uzupełnianie modą z treningu + Kodowanie
        for col in self.cat_cols:
            if col in data.columns:
                mode_val = self.modes[col]
                data[col] = data[col].fillna(mode_val).astype(str)
                
                # Bezpieczne kodowanie (handle unknown)
                le = self.cat_encoders[col]['encoder']
                known_classes = self.cat_encoders[col]['classes']
                
                # Jeśli wartość nie była w treningu, zamień na modę (lub 'other')
                data[col] = data[col].apply(lambda x: x if x in known_classes else mode_val)
                data[col] = le.transform(data[col])

        return data, y