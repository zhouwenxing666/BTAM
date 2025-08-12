import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from scipy.stats import norm
from random import sample

# --- Real-world Dataset Loading ---

DATASET_CONFIG = {
    'diabetes': {'target': 'diabetes_mellitus', 'type': 'classification', 'num_classes': 2},
    'airbnb': {'target': 'price', 'type': 'regression', 'num_classes': 1},
    'har': {'target': 'Activity', 'type': 'classification', 'num_classes': 6},
    'compas': {'target': 'two_year_recid', 'type': 'classification', 'num_classes': 2},
    'MNIST': {'target': 'class_label', 'type': 'classification', 'num_classes': 10},
    'CelebA': {'target': 'gender', 'type': 'classification', 'num_classes': 2}
}

class CustomDataset(Dataset):
    def __init__(self, data, target, features=None):
        self.target = target
        self.features = features if features is not None else list(data.columns.difference([target]))
        
        self.X = torch.tensor(data[self.features].values).float()
        self.y = torch.tensor(data[self.target].values).float()

    def __len__(self):
        return self.X.shape

    def __getitem__(self, i):
        return self.X[i], self.y[i]

def get_real_dataset(data_name):
    data_path = 'data/' + data_name
    data_train_df = pd.read_csv(data_path + '/train.csv')
    data_test_df = pd.read_csv(data_path + '/test.csv')
    
    target = DATASET_CONFIG[data_name]['target']
    features = list(data_train_df.columns.difference([target]))
    
    train_dataset = CustomDataset(data_train_df, target, features=features)
    val_dataset = CustomDataset(data_test_df.sample(frac=0.5, random_state=42), target, features=features)
    test_dataset = CustomDataset(data_test_df.drop(val_dataset.indices), target, features=features)

    concept_groups, concept_names = None, None
    if data_name not in ['MNIST', 'CelebA']:
        try:
            concepts = pd.read_csv(f'{data_path}/concept_groups.csv')
            concept_groups, concept_names = [], []
            for name, group_df in concepts.groupby('concept'):
                group = [features.index(f) for f in group_df['feature']]
                concept_groups.append(group)
                concept_names.append(name)
        except FileNotFoundError:
            print(f"Concept groups file not found for {data_name}. Proceeding without concept groups.")

    return train_dataset, val_dataset, test_dataset, features, concept_groups, concept_names

# --- Synthetic Data Generation (from data_generation.py) ---
# ... (The content of data_generation.py is extensive, so I'll put a placeholder here)
# ... You would copy the full content of `data_generation.py` here, including:
# ... bsplineBasis_j, bsplinebasis, transform_splines
# ... Regression, Classfication_corrupted, Classfication_imbalance, Classfication_multi
# ... data_process, generate_regression, generate_corrupted_classification, etc.
# --- Placeholder for Synthetic Data Code ---
def get_synthetic_dataset(name, num_samples=1000, dimension=100):
    print(f"Generating synthetic dataset: {name}")
    # This is a placeholder for the logic from data_generation.py
    # In a real implementation, you'd call the respective 'generate_*' function
    X = np.random.rand(num_samples, dimension)
    y = np.random.randint(0, 2, (num_samples, 1))
    
    # Split into train/val/test
    X_train, y_train = X[:int(0.8*num_samples)], y[:int(0.8*num_samples)]
    X_val, y_val = X[int(0.8*num_samples):int(0.9*num_samples)], y[int(0.8*num_samples):int(0.9*num_samples)]
    X_test, y_test = X[int(0.9*num_samples):], y[int(0.9*num_samples):]

    train_ds = torch.utils.data.TensorDataset(torch.tensor(X_train).float(), torch.tensor(y_train).float())
    val_ds = torch.utils.data.TensorDataset(torch.tensor(X_val).float(), torch.tensor(y_val).float())
    test_ds = torch.utils.data.TensorDataset(torch.tensor(X_test).float(), torch.tensor(y_test).float())
    
    return train_ds, val_ds, test_ds, [f'f_{i}' for i in range(dimension)], None, None