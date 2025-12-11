import pandas as pd
import numpy as np
from fastai.tabular.all import *

class FastAIBinaryClassifier:
    def __init__(self, cat_names=None, cont_names=None, procs=None, 
                 layers=[200, 100], bs=64, emb_drop=0.1, epochs=5, lr=1e-3, ps=0.1,
                 max_card=20, n_jobs=None, random_state=None):
        """
        :param cat_names: Lista nazw kolumn kategorycznych.
        :param cont_names: Lista nazw kolumn ciągłych (liczbowych).
        :param procs: Lista procesorów FastAI (np. Categorify, Normalize).
        :param layers: Architektura sieci neuronowej (wielkości warstw ukrytych).
        :param bs: Rozmiar batcha (batch size).
        :param emb_drop: Współczynnik dropout dla warstw embeddingowych.
        :param epochs: Liczba epok treningu.
        :param lr: Współczynnik uczenia (learning rate).
        :param ps: Współczynnik dropout dla warstw ukrytych.
        :param max_card: Próg unikalnych wartości, poniżej którego kolumna liczbowa jest traktowana jako kategoryczna (jeśli automatycznie wykrywamy kolumny).
        :param n_jobs: Liczba workerów do ładowania danych (int). -1 oznacza użycie wszystkich rdzeni CPU.
        :param random_state: Ziarno losowości (int) dla powtarzalności wyników.
        """
        self.cat_names = cat_names
        self.cont_names = cont_names
        self.procs = procs if procs else [Categorify, Normalize] 
        
        # Parametry architektury
        self.layers = layers
        self.bs = bs
        self.emb_drop = emb_drop
        self.ps = ps
        
        # Parametry treningu
        self.epochs = epochs
        self.lr = lr
        
        # Parametry automatycznego wykrywania kolumn
        self.max_card = max_card

        # Ustawienia losowości i równoległości
        self.random_state = random_state
        self.n_jobs = n_jobs

        # Inicjalizacja modelu FastAI
        self.learn = None
        self.y_name = 'target'

    def fit(self, X, y):
        df = X.copy()
        df[self.y_name] = y
        
        if self.random_state is not None:
            set_seed(self.random_state, reproducible=True)


        # --- AUTOMATYCZNE WYKRYWANIE KOLUMN ---
        # Jeśli użytkownik nie podał nazw kolumn, używamy cont_cat_split
        if self.cat_names is None and self.cont_names is None:
            # cont_cat_split sprawdza typy danych i liczność (cardinality)
            self.cont_names, self.cat_names = cont_cat_split(
                df, 
                max_card=self.max_card, 
                dep_var=self.y_name
            )
            # Zabezpieczenie: upewnij się, że listy nie są None (choć funkcja zwraca listy)
            self.cont_names = list(self.cont_names)
            self.cat_names = list(self.cat_names)
            
        # Jeśli podano tylko jedną listę, drugą ustawiamy na pustą, by uniknąć błędów
        if self.cat_names is None: self.cat_names = []
        if self.cont_names is None: self.cont_names = []

        # Tworzenie TabularPandas
        # Ważne: FastAI potrzebuje zbioru walidacyjnego do monitorowania metryk.
        # W sklearn fit() zazwyczaj trenuje na 100%, ale sieci neuronowe łatwo przeuczyć bez walidacji.
        splits = RandomSplitter(valid_pct=0.2, seed=self.random_state)(range_of(df))
        
        to = TabularPandas(
            df, 
            procs=self.procs, 
            cat_names=self.cat_names, 
            cont_names=self.cont_names, 
            y_names=self.y_name,
            y_block=CategoryBlock(),
            splits=splits
        )
        
        
        # Obsługa n_jobs (num_workers)
        if self.n_jobs is None:
            workers = 0 # Domyślnie w PyTorch 0 oznacza główny proces (najbezpieczniejsze)
        elif self.n_jobs == -1:
            workers = os.cpu_count() # Użyj wszystkich rdzeni
        else:
            workers = self.n_jobs

        dls = to.dataloaders(bs=self.bs, num_workers=workers)
        
        # Parametry architektury (dropouty) pakujemy w tabular_config
        config = tabular_config(ps=self.ps, embed_p=self.emb_drop)
        
        self.learn = tabular_learner(
            dls, 
            layers=self.layers, 
            metrics=accuracy,
            config=config  # Przekazujemy config zamiast ps/embed_p bezpośrednio
        )
        
        with self.learn.no_bar(), self.learn.no_logging():
            # fit_one_cycle przyjmuje tylko parametry cyklu uczenia
            self.learn.fit_one_cycle(self.epochs, self.lr)
            
        return self

    def predict_proba(self, X):
        """
        Zwraca macierz prawdopodobieństw (N, 2).
        Kolumna 0: Prawdopodobieństwo klasy 0.
        Kolumna 1: Prawdopodobieństwo klasy 1.
        """
        if self.learn is None:
            raise Exception("Model nie został jeszcze wytrenowany!")

        # Tworzenie DataLoader dla danych testowych
        test_dl = self.learn.dls.test_dl(X)
        
        # Pobieranie predykcji. reorder=False jest kluczowe, aby zachować kolejność wierszy!
        preds, _ = self.learn.get_preds(dl=test_dl, reorder=False)
        
        # Zwracamy pełną macierz (N, 2) jako numpy array
        return preds.numpy()

    def predict(self, X):
        """
        Zwraca przewidziane klasy (0 lub 1).
        """
        # Pobieramy prawdopodobieństwa używając metody powyżej
        probs = self.predict_proba(X)
        
        # Wybieramy indeks z największym prawdopodobieństwem (argmax)
        # axis=1 oznacza, że szukamy maksimum w każdym wierszu
        return np.argmax(probs, axis=1)