import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
import torch.nn as nn
import matplotlib as mpl
import matplotlib.pyplot as plt

import torch
from torch.utils.data import Dataset, DataLoader, random_split

DATASETS = {'diabetes': {'target': 'diabetes_mellitus', 'criterion': nn.CrossEntropyLoss(), 'num_classes': 2},
            'airbnb': {'target': 'price', 'criterion': nn.MSELoss(), 'num_classes': 1},
            'har': {'target': 'Activity', 'criterion': nn.CrossEntropyLoss(), 'num_classes': 6},
            'compas': {'target': 'two_year_recid', 'criterion': nn.CrossEntropyLoss(), 'num_classes': 2},
            'MNIST': {'target': 'class_label', 'criterion': nn.CrossEntropyLoss(), 'num_classes': 10},
            'CelebA': {'target': 'gender', 'criterion': nn.CrossEntropyLoss(), 'num_classes': 2}}

class CustomDataset(Dataset):
    def __init__(self, data, target, features=None, concepts=None):
        self.target = target

        if features is None:
            self.features = list(data.columns.difference([target]))
        else:
            self.features = features
        
        self.X = torch.tensor(data[features].values).float()
        self.y = torch.tensor(data[target].values).float()

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, i):
        return self.X[i], self.y[i]

def get_datasets(data_name):
    data_path = 'data/' + data_name
    data_train = pd.read_csv(data_path + '/train.csv')
    data_test = pd.read_csv(data_path + '/test.csv')
    target = DATASETS[data_name]['target']
    features = list(data_train.columns.difference([target]))
    train = CustomDataset(data_train, target, features=features)
    val = CustomDataset(data_test.sample(frac=1.0), target, features=features)
    test = CustomDataset(data_test, target, features=features)

    if data_name not in ['MNIST', 'CelebA']:
        concepts = pd.read_csv(f'{data_path}/concept_groups.csv')
        concept_groups = []
        concept_names = []

        for name, group_df in concepts.groupby('concept'):
            group = []
            for i, row in group_df.iterrows():
                group.append(data_train[features].columns.get_loc(row['feature']))
            print(name, ':', group)
            concept_groups.append(group)
            concept_names.append(name)

        return train, val, test, target, features, concept_groups, concept_names
    else:
        return train, val, test, target, features