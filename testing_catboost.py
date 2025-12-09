import json
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
from catboost import CatBoostClassifier

with open('configs_catboost.json') as f:
    config = json.load(f)


data = pd.read_csv('airfoil_self_noise_46904.csv')  # Zmień na właściwą ścieżkę do pliku

X = data.drop(columns=['target'])
y = data['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

i = 0
for params in config.values():
    if i>5: break
    i += 1
    # delete ag_args keyword from params['hyperparameters'] if exists
    if 'ag_args' in params['hyperparameters']:
        del params['hyperparameters']['ag_args']
    model = CatBoostClassifier(**params['hyperparameters'], random_seed=42, verbose=False)
    print(model._get_params())
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    print(f'Accuracy: {accuracy}')
    print(f'ROC AUC: {roc_auc}')
