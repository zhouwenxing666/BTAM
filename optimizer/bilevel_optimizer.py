import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

# --- MetaModule and Update Logic ---

def to_var(x, requires_grad=True):
    return Variable(x, requires_grad=requires_grad) if torch.cuda.is_available() else Variable(x, requires_grad=requires_grad)

class MetaModule(nn.Module):
    def params(self):
        for _, param in self.named_params(self):
            yield param

    def named_params(self, curr_module=None, memo=None, prefix=''):
        if memo is None: memo = set()
        if curr_module is None: curr_module = self
        
        for name, p in curr_module._parameters.items():
            if p is not None and p not in memo:
                memo.add(p)
                yield prefix + ('.' if prefix else '') + name, p
        
        for mname, module in curr_module.named_children():
            submodule_prefix = prefix + ('.' if prefix else '') + mname
            for name, p in self.named_params(module, memo, submodule_prefix):
                yield name, p

    def update_params(self, lr_inner, source_params=None):
        if source_params is not None:
            for (name, param), grad in zip(self.named_params(self), source_params):
                tmp = param - lr_inner * grad
                self.set_param(self, name, tmp)
        else:
            # Not used in this implementation
            pass

    def set_param(self, curr_mod, name, param):
        if '.' in name:
            n = name.split('.')
            module_name = n
            rest = '.'.join(n[1:])
            for name_child, mod in curr_mod.named_children():
                if module_name == name_child:
                    self.set_param(mod, rest, param)
                    break
        else:
            setattr(curr_mod, name, param)

# --- Meta Learner (Upper-level Model) ---

class MetaLinear(MetaModule):
    def __init__(self, *args, **kwargs):
        super().__init__()
        ignore = nn.Linear(*args, **kwargs)
        self.register_buffer('weight', to_var(ignore.weight.data, requires_grad=True))
        self.register_buffer('bias', to_var(ignore.bias.data, requires_grad=True))

    def forward(self, x):
        return F.linear(x, self.weight, self.bias)

class MetaWeightNet(MetaModule):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear1 = MetaLinear(input_size, hidden_size)
        self.relu = nn.ReLU(inplace=True)
        self.linear2 = MetaLinear(hidden_size, output_size)

    def forward(self, x):
        x = self.relu(self.linear1(x))
        return torch.sigmoid(self.linear2(x))

# --- Sparsity-inducing Proximal Updates ---

def proximal_group_lasso(w, lam, eta, group_size):
    num_groups = w.shape[-1] // group_size
    w_reshaped = w.view(num_groups, group_size)
    group_norms = torch.norm(w_reshaped, p=2, dim=1, keepdim=True)
    
    # Avoid division by zero
    group_norms[group_norms == 0] = 1.0 
    
    scale_factor = 1 - lam * eta / group_norms
    scale_factor = torch.clamp(scale_factor, min=0)
    
    v = w_reshaped * scale_factor
    w.data = v.view_as(w)

# --- Bilevel Update Steps ---

def update_lower_level_meta(lower_model, upper_model, train_batch, criterion, args, device):
    """ Step 1: Compute adapted lower-level model on training data. """
    x_train, y_train = train_batch
    x_train, y_train = x_train.to(device), y_train.to(device)

    # Create a virtual model for meta-update
    meta_model = type(lower_model)(**lower_model.init_kwargs).to(device)
    meta_model.load_state_dict(lower_model.state_dict())

    # Forward pass on virtual model
    y_pred_meta, _ = meta_model(x_train)
    cost = criterion(y_pred_meta, y_train)
    
    # Get sample weights from upper-level model
    weights = upper_model(cost.view(-1, 1).data)
    
    # Compute weighted loss and gradients
    weighted_loss = torch.sum(weights * cost) / len(cost)
    meta_model.zero_grad()
    grads = torch.autograd.grad(weighted_loss, meta_model.params(), create_graph=True)
    
    # Update virtual model parameters
    meta_model.update_params(lr_inner=args.lower_lr, source_params=grads)
    
    return meta_model, weighted_loss.item()

def update_upper_level(meta_model, optimizer_upper, val_batch, criterion, device):
    """ Step 2: Update upper-level model on validation data. """
    x_val, y_val = val_batch
    x_val, y_val = x_val.to(device), y_val.to(device)
    
    # Forward pass on the adapted model
    y_pred_val, _ = meta_model(x_val)
    val_loss = criterion(y_pred_val, y_val).mean()
    
    # Update upper-level model
    optimizer_upper.zero_grad()
    val_loss.backward()
    optimizer_upper.step()
    
    return val_loss.item()

def update_lower_level_final(lower_model, upper_model, optimizer_lower, train_batch, criterion, args, device):
    """ Step 3: Update the actual lower-level model. """
    x_train, y_train = train_batch
    x_train, y_train = x_train.to(device), y_train.to(device)
    
    y_pred, concepts = lower_model(x_train)
    cost = criterion(y_pred, y_train)
    
    with torch.no_grad():
        weights = upper_model(cost.view(-1, 1))
        
    # Normalize weights
    if torch.sum(weights) > 0:
        weights = weights / torch.sum(weights)
        
    # Compute final weighted loss, including regularization
    weighted_loss = torch.sum(weights * cost)
    regularization_loss = lower_model.regularization_loss(concepts)
    total_loss = weighted_loss + regularization_loss
    
    # Update actual lower-level model
    optimizer_lower.zero_grad()
    total_loss.backward()
    optimizer_lower.step()
    
    # Apply sparsity-inducing proximal update
    if args.penalty_coef > 0:
        # Assuming the first layer of the taylor head is the target for sparsity
        # This is a simplification; a real implementation would be more specific.
        taylor_weights = lower_model.taylor_head.Is
        proximal_group_lasso(taylor_weights, lam=args.penalty_coef, eta=args.lower_lr, group_size=1)
        
    return total_loss.item(), weighted_loss.item(), regularization_loss.item()