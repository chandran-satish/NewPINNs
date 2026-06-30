import os
import sys
from datetime import datetime
import torch
import torch.nn as nn
import torch.multiprocessing as mp
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, Callback
from pytorch_lightning.loggers import TensorBoardLogger
from torch.optim.lr_scheduler import CosineAnnealingLR

import numpy as np

from utils_cavity_ngsolve import*
from utils_cavity_unet import*

from contextlib import contextmanager
import h5py
import yaml
import argparse

torch.set_float32_matmul_precision('medium')

def parse_args():
    parser = argparse.ArgumentParser(description='Train PINN model for porous media using YAML config')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to YAML config file')
    return parser.parse_args()

def run_training(config_path):
    # Set multiprocessing start method to 'spawn'
    mp.set_start_method('spawn', force=True)

    num_threads = os.cpu_count()
    print(f"You have access to {num_threads} CPU threads.")
    
    # Load configuration from YAML file
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    # Print all contents of the YAML file
    print("\n=== YAML Configuration File Contents ===")
    with open(config_path, 'r') as file:
        yaml_contents = file.read()
        print(yaml_contents)
    print("======================================\n")
    
    # Extract configuration values
    train_num_samples = config['data']['train_num_samples']
    valid_num_samples = config['data']['valid_num_samples']
    
    ## Extract UNet model configuration if it exists
    model_config = config.get('model', {}).get('unet_config', None)

    train_file_path =  f'./data/train_data_{train_num_samples}.h5'
    valid_file_path =  f'./data/valid_data_{valid_num_samples}.h5'
    #valid_file_path =  f'./data/train_data_{valid_num_samples}.h5'

    print(f'train_file_path = {train_file_path}')
    print(f'valid_file_path = {valid_file_path}')
    
    model_type = config['model']['type']
    learning_rate = config['model']['learning_rate']
    
    save_dir = config['training']['save_dir']
    read_ckpt = config['training']['read_ckpt']
    if read_ckpt == "None":
        read_ckpt = None
    
    fem_iterations = config['training']['fem_iterations']
    Tmax = config['training']['Tmax']
    lambda_ux = config['training']['lambda_ux']
    lambda_uy = config['training']['lambda_uy']
    lambda_p = config['training']['lambda_p']
    num_epochs = config['training']['num_epochs']
    batch_size = config['training']['batch_size']
    
    # Enable progress bar if specified in config, default to disabled
    enable_progress_bar = config['training'].get('enable_progress_bar', False)
    enable_model_summary = config['training'].get('enable_model_summary', False)
    enable_validation = config['training'].get('enable_validation', True)
    
    # GPU configuration
    num_gpus = config['training'].get('num_gpus', 2)
    precision = config['training'].get('precision', '16-mixed')

    
    
    # Create PyTorch datasets
    train_dataset = CavityDataset(train_file_path)
    valid_dataset = CavityDataset(valid_file_path)
    
    # Create dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=batch_size,
        shuffle=True,
        num_workers=config['dataloader'].get('num_workers', 1),
        pin_memory=config['dataloader'].get('pin_memory', True),
        persistent_workers=config['dataloader'].get('persistent_workers', True)
    )
    
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, 
        batch_size=batch_size,
        shuffle=False,
        num_workers=config['dataloader'].get('num_workers', 1),
        pin_memory=config['dataloader'].get('pin_memory', True),
        persistent_workers=config['dataloader'].get('persistent_workers', True)
    )
    
    # Initialize the model
    model = FEMPhysicsModule(
        model_type=model_type,
        learning_rate=learning_rate,
        fem_iterations=fem_iterations,
        Tmax = Tmax,
        lambda_ux = lambda_ux,
        lambda_uy = lambda_uy,
        lambda_p = lambda_p,
        model_config = model_config
    )

    # load in a previous checkpoint/weights if specified 
    if read_ckpt is not None:
        print(f"Loading weights from checkpoint: {read_ckpt}")
        checkpoint = torch.load(read_ckpt, map_location=lambda storage, loc: storage)
        model.load_state_dict(checkpoint['state_dict'])
        print("Checkpoint loaded successfully!")
    
    # Setup checkpointing with configurable parameters
    checkpoint_callback = ModelCheckpoint(
        dirpath=save_dir,
        filename=config['checkpoint'].get('filename', 'model_{epoch}'),
        save_top_k=config['checkpoint'].get('save_top_k', -1),
        monitor=config['checkpoint'].get('monitor', 'train_loss'),
        mode=config['checkpoint'].get('mode', 'min'),
        save_last=config['checkpoint'].get('save_last', True),
        every_n_epochs=config['checkpoint'].get('every_n_epochs', 10),
    )
    
    # Setup custom output callback
    output_callback = SimplifiedOutputCallback(num_epochs)
    
    # Setup logger
    logger = TensorBoardLogger(save_dir=os.path.join(save_dir, 'logs'))
    
    # Initialize the trainer
    trainer = pl.Trainer(
        max_epochs=num_epochs,
        callbacks=[checkpoint_callback, output_callback],
        logger=logger,
        log_every_n_steps=config['training'].get('log_every_n_steps', 1),
        accelerator=config['training'].get('accelerator', 'auto'),
        devices=num_gpus,
        strategy=config['training'].get('strategy', 'ddp'),
        precision=precision,
        enable_progress_bar=enable_progress_bar,
        enable_model_summary=enable_model_summary,
        accumulate_grad_batches = 64
    )
    
    # Start training
    if enable_validation:
        trainer.fit(model, train_loader, valid_loader)
    else:
        trainer.fit(model, train_loader, None)
    
    print("Done!")

class CavityDataset(torch.utils.data.Dataset):
    def __init__(self, file_path):
        self.file_path = file_path
        with h5py.File(file_path, 'r') as f:
            self.num_samples = f['img1'].shape[0]
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        with h5py.File(self.file_path, 'r') as f:
            img1 = torch.from_numpy(f['img1'][idx])
            if 'Ueq' in f:
                Ueq = torch.from_numpy(f['Ueq'][idx])
                Veq = torch.from_numpy(f['Veq'][idx])
                Peq = torch.from_numpy(f['Peq'][idx])
                return img1, Ueq, Veq, Peq
            else:
                return img1


class SimplifiedOutputCallback(Callback):
    def __init__(self, num_epochs):
        super().__init__()
        self.num_epochs = num_epochs
    
    def on_train_epoch_end(self, trainer, pl_module):
        if trainer.global_rank != 0:
            return
        epoch = trainer.current_epoch
        train_loss = trainer.callback_metrics.get('train_loss', torch.tensor(0.0)).item()
        val_loss = trainer.callback_metrics.get('val_loss', torch.tensor(0.0)).item()

        # Print GPU memory usage
        #for i in range(torch.cuda.device_count()):
        #    print(f"GPU {i} memory: {torch.cuda.memory_allocated(i)/1e9:.2f}GB / {torch.cuda.get_device_properties(i).total_memory/1e9:.2f}GB")
        
        # Print the customized output
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f'[{current_time}] Epoch [{epoch+1}/{self.num_epochs}] Training Loss: {train_loss:.10f}, Validation Loss: {val_loss:.10f}')


@contextmanager
def suppress_stdout():
    with open(os.devnull, 'w') as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout

# Example usage:
if __name__ == "__main__":
    args = parse_args()
    run_training(args.config)