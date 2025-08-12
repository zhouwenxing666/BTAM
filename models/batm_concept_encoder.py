from torch import nn
import torch
from optimizer.bilevel_optimizer import MetaModule

class ExU(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(ExU, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(in_dim, out_dim))
        self.bias = nn.Parameter(torch.Tensor(in_dim))
        self._init_params()

    def _init_params(self):
        nn.init.normal_(self.weight, mean=4.0, std=0.5)
        nn.init.normal_(self.bias, std=0.5)
  
    def forward(self, x):
        out = torch.matmul((x - self.bias), torch.exp(self.weight))
        return torch.clamp(out, 0, 1)

class BATM_ConceptEncoder(MetaModule):
    def __init__(self, 
                num_mlps, 
                hidden_dims, 
                input_layer='linear', 
                concept_groups=None,
                activation=nn.ReLU(),
                order=1, 
                dropout=0.0, 
                batchnorm=False):
    
        super(BATM_ConceptEncoder, self).__init__()
        assert order > 0
        assert input_layer in ["exu", "linear"]

        self.num_mlps = num_mlps
        self.dropout = dropout
        self.batchnorm = batchnorm
        self.activation = activation
        self.use_concept_groups = bool(concept_groups)

        if self.use_concept_groups:
            print(f'Learning {len(concept_groups)} high-level concepts...')
            input_dims = [len(group) for group in concept_groups]
            self.concept_groups = concept_groups
        else:
            input_dims = [order] * num_mlps

        # Input layer
        if input_layer == "exu":
            self.input_layer = nn.ModuleList([ExU(input_dims[i], hidden_dims) for i in range(num_mlps)])
        else: # linear
            self.input_layer = nn.ModuleList([nn.Linear(input_dims[i], hidden_dims) for i in range(num_mlps)])
    
        # Hidden layers
        layers = []
        if self.batchnorm:
            layers.append(nn.BatchNorm1d(hidden_dims * num_mlps))
        if dropout > 0:
            layers.append(nn.Dropout(p=dropout))
        layers.append(self.activation)
    
        input_dim = hidden_dims
        for dim in hidden_dims[1:]:
            layers.append(nn.Conv1d(in_channels=input_dim * num_mlps,
                                    out_channels=dim * num_mlps,
                                    kernel_size=1,
                                    groups=num_mlps))
            if self.batchnorm:
                layers.append(nn.BatchNorm1d(dim * num_mlps))
            if dropout > 0:
                layers.append(nn.Dropout(p=dropout))
            layers.append(self.activation)
            input_dim = dim

        # Output layer
        layers.append(nn.Conv1d(in_channels=input_dim * num_mlps,
                                out_channels=1 * num_mlps,
                                kernel_size=1,
                                groups=num_mlps))

        self.hidden_layers = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_concept_groups:
            Xs = [self.input_layer[i](x[:, self.concept_groups[i]]) for i in range(self.num_mlps)]
            Xs = torch.cat(Xs, dim=1).unsqueeze(-1)
        else: # Assumes ExU or simple linear
            Xs = [self.input_layer[i](x[:, i].unsqueeze(1)) for i in range(self.num_mlps)]
            Xs = torch.cat(Xs, dim=1).unsqueeze(-1)
            
        z = self.hidden_layers(Xs)
        return z