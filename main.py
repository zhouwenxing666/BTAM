import os
import argparse
import yaml
import time
import torch
import torch.nn as nn
from torch.optim import Adam, AdamW
from torch.utils.data import DataLoader
import wandb

from models.batm_taylor_network import BATM_TaylorNetwork
from optimizer.bilevel_optimizer import MetaWeightNet, update_lower_level_meta, update_upper_level, update_lower_level_final
from data.dataset_loader import get_real_dataset, get_synthetic_dataset, DATASET_CONFIG
from utils import macro_statistics

def parse_args():
    parser = argparse.ArgumentParser(description="Bilevel Additive Taylor Model (BATM)")
    parser.add_argument('--config', type=str, required=True, help='Path to the config file.')
    args = parser.parse_args()
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    # Convert dict to Namespace for easier access
    return argparse.Namespace(**config)

def main(args):
    # --- Setup ---
    if args.training:
        wandb.init(project=args.wandb_project, entity=args.wandb_entity, config=vars(args))
    
    device = torch.device(f"cuda:{args.gpu}" if args.use_gpu and torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed(args.seed)

    os.makedirs(args.checkpoints, exist_ok=True)
    os.makedirs(args.results, exist_ok=True)

    # --- Data Loading ---
    if 'synthetic' in args.data_name:
        train_ds, val_ds, test_ds, features, c_groups, c_names = get_synthetic_dataset(args.data_name)
        data_type = 'regression' if 'regression' in args.data_name else 'classification'
        num_classes = 1 if data_type == 'regression' else 2 # Assuming binary for synthetic
    else:
        train_ds, val_ds, test_ds, features, c_groups, c_names = get_real_dataset(args.data_name)
        data_type = DATASET_CONFIG[args.data_name]['type']
        num_classes = DATASET_CONFIG[args.data_name]['num_classes']

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    
    # --- Model & Optimizer Setup ---
    model_params = {
        'num_inputs': len(features), 'num_outputs': num_classes,
        'concept_groups': c_groups, 'input_layer': args.input_layer,
        'hidden_dims': [int(x) for x in args.hidden_dims.split(',')],
        'order': args.order, 'rank': args.rank, 'initial': args.initial,
        'concept_dropout': args.concept_dropout, 'batchnorm': args.batchnorm,
        'encode_concepts': args.encode_concepts, 'output_penalty': args.output_penalty
    }
    
    lower_level_model = BATM_TaylorNetwork(**model_params).to(device)
    lower_level_model.init_kwargs = model_params # Store for re-instantiation
    
    upper_level_model = MetaWeightNet(input_size=1, hidden_size=100, output_size=1).to(device)

    optimizer_lower = AdamW(lower_level_model.params(), lr=args.lower_lr)
    optimizer_upper = Adam(upper_level_model.params(), lr=args.upper_lr)
    
    criterion = nn.MSELoss(reduction='none') if data_type == 'regression' else nn.CrossEntropyLoss(reduction='none')

    # --- Training Loop ---
    print("Starting BATM training...")
    start_time = time.time()
    for epoch in range(1, args.num_epochs + 1):
        lower_level_model.train()
        upper_level_model.train()
        
        epoch_logs = {'train_loss': 0, 'val_loss': 0, 'weighted_loss': 0, 'reg_loss': 0}
        
        # We zip loaders, assuming they have similar length for this pseudo-code
        # A real implementation would handle differing lengths (e.g., by cycling the smaller one)
        for i, (train_batch, val_batch) in enumerate(zip(train_loader, val_loader)):
            # Step 1: Compute adapted lower-level model
            meta_model, meta_loss = update_lower_level_meta(
                lower_level_model, upper_level_model, train_batch, criterion, args, device
            )

            # Step 2: Update upper-level model
            val_loss = update_upper_level(
                meta_model, optimizer_upper, val_batch, criterion, device
            )

            # Step 3: Update actual lower-level model
            total_loss, weighted_loss, reg_loss = update_lower_level_final(
                lower_level_model, upper_level_model, optimizer_lower, train_batch, criterion, args, device
            )
            
            epoch_logs['train_loss'] += total_loss
            epoch_logs['val_loss'] += val_loss
            epoch_logs['weighted_loss'] += weighted_loss
            epoch_logs['reg_loss'] += reg_loss
            
        # --- Logging ---
        num_batches = len(train_loader)
        for key in epoch_logs:
            epoch_logs[key] /= num_batches
        
        if args.training:
            wandb.log(epoch_logs, step=epoch)

        print(f"Epoch: {epoch}/{args.num_epochs} | "
              f"Train Loss: {epoch_logs['train_loss']:.4f} | "
              f"Val Loss: {epoch_logs['val_loss']:.4f}")

    print(f"Training finished in {(time.time() - start_time)/60:.2f} minutes.")
    
    # --- Final Evaluation ---
    # ... evaluation logic on test_loader ...

if __name__ == '__main__':
    args = parse_args()
    main(args)