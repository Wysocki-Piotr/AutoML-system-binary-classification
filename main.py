import importlib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.model_selection import train_test_split

from Preprocessing.SimplePreprocessor import SimplePreprocessor

# wczytywanie parametrów modeli
cfg = pd.read_json("models.json") # convert to list of dicts
cfg = cfg.to_dict(orient="records")

# przykładowy dataset
data = pd.read_csv("Datasets/polish_companies_bankruptcy_46950.csv")
X = data.drop(columns=["target"])
y = data["target"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# preprocessing
simple_preprocessing = SimplePreprocessor()
X_train, y_train = simple_preprocessing.fit_transform(X_train, y_train)
X_test, y_test = simple_preprocessing.transform(X_test, y_test)

scores = []

for i in range(len(cfg)):
    model_info = cfg[i]  # aktualna pozycja

    # uzyskaj fully-qualified class name (FQCN)
    fqcn = model_info.get("class")
    if not fqcn:
        raise ValueError(f"Brakuje pola 'class' w wpisie {model_info} models.json")

    # import danego modelu i całej biblioteki tego modelu
    module_name, cls_name = fqcn.rsplit(".", 1)
    module = importlib.import_module(module_name)
    Cls = getattr(module, cls_name)

    params = model_info.get("params", {}) or {}

    # spróbuj utworzyć instancję z parametrami, a w razie błędu bez nich
    try:
        model = Cls(**params)
    except Exception:
        model = Cls()
        raise Warning(f"Nie udało się utworzyć instancji {fqcn} z podanymi parametrami. Utworzono bez parametrów.")

    print("Trening modelu:", fqcn, " \n")

    # dopasuj i wykonaj predykcje
    model.fit(X_train, y_train)
    probs = model.predict_proba(X_test)[:, 1]

    # prawdopodobieństwa klasy pozytywnej (jeśli dostępne)
    # if hasattr(model, "predict_proba"):
    #     probs = model.predict_proba(X_test)[:, 1]
    # elif hasattr(model, "decision_function"):
    #     scores = model.decision_function(X_test)
    #     probs = 1.0 / (1.0 + np.exp(-scores))
    # else:
    #     probs = np.asarray(preds, dtype=float)
    
    
    # ocena modelu
    auc = roc_auc_score(y_test, probs)
    brier = brier_score_loss(y_test, probs)
    scores.append({"model": model_info, "auc": auc, "brier": brier})
    print("Sukcess:  ", i, "   ", scores[-1])
    print(50*"-", "\n\n")

# convert to DataFrame for better visualization
scores_df = pd.DataFrame(scores)
print(scores_df.sort_values(by="auc", ascending=False, inplace=True))