import openml
import numpy as np

# 1. Get the study suite by its ID (457 corresponds to "OpenML-CC18")
suite = openml.study.get_suite(457)

print(f"Downloading {len(suite.data)} datasets from suite: {suite.name}")

# 2. Iterate through the dataset IDs listed in the suite
for dataset_id in suite.data:
    try:
        # Download the dataset
        dataset = openml.datasets.get_dataset(dataset_id, download_data=True)
        
        # Get the actual data (returns a pandas DataFrame by default)
        X, y, categorical_indicator, attribute_names = dataset.get_data(
            target=dataset.default_target_attribute
        )
        
        print(f"Successfully downloaded: {dataset.name} (ID: {dataset_id})")
        print(f" - Rows: {X.shape[0]}, Columns: {X.shape[1]}")
        
        if np.unique(y).size != 2:
            print(f"   Skipping dataset {dataset.name} (ID: {dataset_id}) as it does not have exactly 2 unique target values.")
            continue

        # Optional: Save to CSV
        output_file = f"./Datasets/{dataset.name}_{dataset_id}.csv"
        X.assign(target=y).to_csv(output_file, index=False)
        
        
    except Exception as e:
        print(f"Failed to download dataset ID {dataset_id}: {e}")