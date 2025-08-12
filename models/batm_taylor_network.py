import torch
from torch import nn
import tensorly as tl
from optimizer.bilevel_optimizer import MetaModule
from models.batm_concept_encoder import BATM_ConceptEncoder
tl.set_backend('pytorch')

class BATM_Fast_Tucker_Taylor(MetaModule):
    def __init__(self, in_features: int, out_features: int, X0=None, order=2, rank=50, initial='Taylor'):
        super().__init__()
        self.order = order
        self.initial = initial
        self.X0 = X0 if X0 is not None else 0.0

        self.const = nn.Parameter(torch.empty((out_features,)))
        self.Os = nn.ParameterList([nn.Parameter(torch.empty((out_features, rank))) for _ in range(order)])
        self.Is = nn.ParameterList([nn.Parameter(torch.empty(((i + 1), rank, in_features))) for i in range(order)])
        self.Gs = nn.ParameterList([nn.Parameter(torch.empty((rank, rank**(i + 1)))) for _ in range(order)])
        
        self.reset_parameter()

    def reset_parameter(self):
        # Implementation of reset_parameter from original file
        nn.init.zeros_(self.const)
        if self.initial == 'Taylor':
            for i, O in enumerate(self.Os):
                std_dev = (1 / (O.shape))**0.5 if i == 0 else 0.0
                nn.init.normal_(O, mean=0, std=std_dev)
            
            def _get_base(in_features, current_order):
                return (1 / in_features) if current_order == 1 else (1 / (in_features * (in_features + 2 * (current_order-1))))

            for i, I in enumerate(self.Is):
                exponent = 1 / (2**(i + 1))
                base = _get_base(I.shape, i + 1)
                nn.init.normal_(I, mean=0, std=base**exponent)

            for G in self.Gs:
                nn.init.normal_(G, mean=0, std=(1 / (G.shape))**0.5)
        else:
            init_method = nn.init.xavier_uniform_ if self.initial == 'Xavier' else nn.init.kaiming_uniform_
            for param_list in [self.Os, self.Is, self.Gs]:
                for param in param_list:
                    init_method(param)

    def forward(self, X: torch.tensor):
        is_vector = X.dim() == 1
        if is_vector:
            X = X.unsqueeze(0)
        
        X = X - self.X0
        Y = self.const.unsqueeze(0).expand(X.shape, -1).clone()
        
        for i in range(self.order):
            I, O, G = self.Is[i], self.Os[i], self.Gs[i]
            Z = I @ X.T
            
            if i > 0:
                kron_res = Z
                for j in range(1, i + 1):
                    X3 = Z[j].reshape([I.shape] + * j + [X.shape])
                    kron_res = (kron_res.unsqueeze(0) * X3)
            else:
                kron_res = Z
                
            kron_res = kron_res.reshape((-1, X.shape))
            Y = Y + (O @ (G @ kron_res)).T

        return Y.squeeze(0) if is_vector else Y

class BATM_TaylorNetwork(MetaModule):
    def __init__(self,
                num_inputs,
                num_outputs,
                concept_groups=None,
                input_layer='linear',
                hidden_dims=,
                order=2,
                rank=8,
                initial='Xavier',
                concept_dropout=0.0,
                batchnorm=True,
                encode_concepts=True,
                output_penalty=0.0):
        
        super(BATM_TaylorNetwork, self).__init__()

        self.num_outputs = num_outputs
        self.output_penalty = output_penalty
        self.encode_concepts = encode_concepts

        if self.encode_concepts:
            num_concepts = len(concept_groups) if concept_groups else num_inputs
            self.concept_encoder = BATM_ConceptEncoder(num_concepts, hidden_dims,
                                                    input_layer=input_layer,
                                                    concept_groups=concept_groups,
                                                    activation=nn.LeakyReLU(),
                                                    dropout=concept_dropout,
                                                    batchnorm=batchnorm)
            taylor_in_features = num_concepts
        else:
            self.concept_encoder = None
            taylor_in_features = num_inputs
    
        self.taylor_head = BATM_Fast_Tucker_Taylor(taylor_in_features, 
                                                   num_outputs, 
                                                   X0=None, 
                                                   order=order, 
                                                   rank=rank, 
                                                   initial=initial)
    
    def regularization_loss(self, concepts):
        return self.output_penalty * (torch.pow(concepts, 2).mean())
  
    def forward(self, x):
        if self.concept_encoder:
            z = self.concept_encoder(x).squeeze(-1)
        else:
            z = x
            
        output = self.taylor_head(z)
        if self.num_outputs == 1:
            output = output.squeeze(-1)

        return output, z