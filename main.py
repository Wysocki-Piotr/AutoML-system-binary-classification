from sklearn.model_selection import train_test_split
from Preprocessing.SimplePreprocessor import SimplePreprocessor
from wrappers.wrapper_model import ModelWrapper
import pandas as pd
pd.set_option('display.max_columns', None)
# Load model configurations
cfg = pd.read_json("models.json").to_dict(orient="records")

# Load dataset
data = pd.read_csv("Datasets/diabetes_46921.csv")
X = data.drop(columns=["target"])
y = data["target"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Preprocessing
simple_preprocessing = SimplePreprocessor()
X_train, y_train = simple_preprocessing.fit_transform(X_train, y_train)
X_test, y_test = simple_preprocessing.transform(X_test, y_test)

# Collect scores in the desired format
scores = []

for model_config in cfg:
    wrapper = ModelWrapper(model_config)
    print(f"Training model: {model_config['class']} with parameters: {model_config.get('params', {})}")

    # Fit the model
    wrapper.fit(X_train, y_train)

    # Evaluate the model
    score = wrapper.evaluate(X_test, y_test)
    scores.append({
        "Model Name": model_config["name"],
        "Model Class": model_config["class"],
        "Brier Score": score["brier"],
        "AUC": score["auc"]
    })

    # Detailed success message
    print(f"Success: Model '{model_config['name']}' ({model_config['class']}) achieved AUC: {score['auc']:.4f}, "
          f"Brier Score: {score['brier']:.4f}")
    print("-" * 50)

# Convert to DataFrame for better visualization
scores_df = pd.DataFrame(scores)

# Sort the DataFrame by AUC in descending order
scores_df = scores_df.sort_values(by="AUC", ascending=False)

# Print the sorted DataFrame
print(scores_df)