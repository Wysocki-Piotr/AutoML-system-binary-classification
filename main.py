from sklearn.metrics import balanced_accuracy_score, accuracy_score
from sklearn.model_selection import train_test_split
from Preprocessing.AutoMLPreprocessor import AutoMLPreprocessor
from wrappers.wrapper_model import ModelWrapper
import pandas as pd

pd.set_option('display.max_columns', None)
cfg = pd.read_json("models_new.json").to_dict(orient="records")


X = pd.read_csv("test_data/X.csv")
y = pd.read_csv("test_data/y.csv")

# data = pd.read_csv("Datasets/coil2000_insurance_policies_46916.csv")
# X = data.drop(columns=["target"])
# y = data["target"]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Preprocessing
preprocessing = AutoMLPreprocessor(
                 add_kmeans_features=True,
                 feature_selection= True,
                 add_poly_features=True,
                 remove_outliers=False,
                 remove_multicollinearity=True,
                 multicollinearity_threshold=0.95,
                 id_threshold=0.95,
                 random_state=42)
X_train, y_train = preprocessing.fit_transform(X_train, y_train)
X_test, y_test = preprocessing.transform(X_test, y_test)

# Collect scores in the desired format
scores = []

for model_config in cfg:
    wrapper = ModelWrapper(model_config)
    print(f"Training model: {model_config['class']}") # with parameters: {model_config.get('params', {})}")

    if model_config['class'] == 'wrappers.TorchNNClassifier.TorchNNClassifier':
        continue  # Skip TorchNNClassifier for now

    # Fit the model
    wrapper.fit(X_train, y_train)

    # Evaluate the model
    score = wrapper.evaluate(X_test, y_test)
    pred = (wrapper.predict_proba(X_test)[:, 1] >= 0.5).astype(int)
    scores.append({
        "Model Name": model_config["name"],
        "Model Class": model_config["class"],
        "Brier Score": score["brier"],
        "AUC": score["auc"],
        "Accuracy": accuracy_score(y_test, pred),
        "Balanced Accuracy": balanced_accuracy_score(y_test, pred)
    })

    # Detailed success message
    print(f"Success: Model '{model_config['name']}' achieved AUC: {score['auc']:.4f}, "
          f"Accuracy: {accuracy_score(y_test, pred):.4f}, "
          f"Balanced Accuracy: {balanced_accuracy_score(y_test, pred):.4f}")
    print("-" * 50)

# Convert to DataFrame for better visualization
scores_df = pd.DataFrame(scores)

# Sort by Balanced Accuracy in descending order
scores_df = scores_df.sort_values(by="Balanced Accuracy", ascending=False)

# Print the sorted DataFrame
print(scores_df[["Model Name", "AUC", "Balanced Accuracy"]])