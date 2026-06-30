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

from utils_fpe_fipy import*

from contextlib import contextmanager
import h5py
import yaml
import argparse
from diffusers import UNet2DModel

torch.set_float32_matmul_precision('medium')

class ImagePredictorUNet(nn.Module):
    def __init__(self, config=None):
        super(ImagePredictorUNet, self).__init__()
        
        # Use config if provided, otherwise use defaults
        if config is None:
            config = {}
        
        # Get UNet configuration parameters
        sample_size = config.get('sample_size', (32, 32))
        in_channels = config.get('in_channels', 1)
        out_channels = config.get('out_channels', 3)
        layers_per_block = config.get('layers_per_block', 1)
        block_out_channels = config.get('block_out_channels', (8, 16, 32))
        norm_num_groups = config.get('norm_num_groups', 2)
        down_block_types = config.get('down_block_types', ("DownBlock2D", "DownBlock2D", "DownBlock2D"))
        up_block_types = config.get('up_block_types', ("UpBlock2D", "UpBlock2D", "UpBlock2D"))
        attention_head_dim = config.get('attention_head_dim', 4)
        
        # Define the UNet2D model
        self.unet = UNet2DModel(
            sample_size=sample_size,
            in_channels=in_channels,
            out_channels=out_channels,
            layers_per_block=layers_per_block,
            block_out_channels=block_out_channels,
            norm_num_groups=norm_num_groups,
            down_block_types=down_block_types,
            up_block_types=up_block_types,
            attention_head_dim=attention_head_dim,
            act_fn = "silu"
        )
        
        # Parameters for output normalization
        self.p_min = config.get('p_min', 0.00)
        self.p_max = config.get('p_max', 10.0)

        # Initialize weights using the specified method
        init_method = config.get('init_method', 'kaiming')
        if init_method == "None":
            init_method = None
        gain = config.get('init_gain', 0.02)
        if init_method is not None:
            self.initialize_weights(init_method, gain)
        
    def initialize_weights(self, method='kaiming', gain=0.02):
        """Initialize the weights of the UNet model using the specified method.
        
        Args:
            method (str): Initialization method. Options: 'kaiming', 'xavier', 'orthogonal', 
                        'normal', 'zeros', 'near_zero'
            gain (float): The gain parameter used for some initialization methods
        """
        for m in self.unet.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                if method == 'kaiming':
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                elif method == 'xavier':
                    nn.init.xavier_uniform_(m.weight, gain=gain)
                elif method == 'orthogonal':
                    nn.init.orthogonal_(m.weight, gain=gain)
                elif method == 'normal':
                    nn.init.normal_(m.weight, mean=0, std=gain)
                elif method == 'zeros':
                    nn.init.zeros_(m.weight)
                elif method == 'near_zero':
                    # Initialize with very small values close to zero
                    nn.init.normal_(m.weight, mean=0, std=gain/10)  # Using a small fraction of gain
                
                if m.bias is not None:
                    if method == 'near_zero':
                        nn.init.normal_(m.bias, mean=0, std=gain/10)
                    else:
                        nn.init.constant_(m.bias, 0)
                    
            elif isinstance(m, nn.BatchNorm2d):
                if method == 'near_zero':
                    nn.init.normal_(m.weight, mean=1, std=gain/10)
                    nn.init.normal_(m.bias, mean=0, std=gain/10)
                else:
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
                    
            elif isinstance(m, nn.Linear):
                if method == 'kaiming':
                    nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                elif method == 'xavier':
                    nn.init.xavier_uniform_(m.weight, gain=gain)
                elif method == 'orthogonal':
                    nn.init.orthogonal_(m.weight, gain=gain)
                elif method == 'normal':
                    nn.init.normal_(m.weight, mean=0, std=gain)
                elif method == 'zeros':
                    nn.init.zeros_(m.weight)
                elif method == 'near_zero':
                    nn.init.normal_(m.weight, mean=0, std=gain/10)
                
                if method == 'near_zero':
                    nn.init.normal_(m.bias, mean=0, std=gain/10)
                else:
                    nn.init.constant_(m.bias, 0)
        
        print(f"UNet weights initialized using {method} initialization" + 
            (f" with gain={gain}" if method in ['xavier', 'orthogonal', 'normal', 'near_zero'] else ""))

    def forward(self, x1):
        device = x1.device
        batch_size = x1.shape[0]
        
        x1 = x1.unsqueeze(1)  # Add channel dimension
        
        x = torch.cat([x1], dim=1)
        
        # Create dummy timesteps and encoder hidden states for UNet3DConditionModel
        timesteps = torch.zeros(batch_size, dtype=torch.long, device=device)

        # Forward pass through UNet
        output = self.unet(x, timesteps).sample

        # Normalize the density and velocities outputs
        p = self.p_min + torch.sigmoid(output[:, 0:1]) * (self.p_max - self.p_min)
        x = torch.cat([p], dim=1)
        return x


class FVMPhysicsModule(pl.LightningModule):
    def __init__(self, 
                 model_type='unet', 
                 learning_rate=1e-5, 
                 fvm_iterations=100,
                 Tmax = 50,
                 model_config=None):
        super().__init__()
        self.save_hyperparameters()
        
        if model_type == 'unet':
            self.model = ImagePredictorUNet(model_config)
        else:
            raise ValueError('No valid model!')
            
        self.criterion = nn.MSELoss()
        self.criterion_rel = self.relative_l2_loss
        self.learning_rate = learning_rate
        self.fem_iterations = fvm_iterations
        self.Tmax = Tmax
        
        # Keep track of epoch metrics for custom logging
        self.train_losses = []
        self.val_losses = []

        # Add tracking for problematic samples
        self.nan_inf_indices = []  # List to store global indices of problematic samples
        self.epoch_nan_inf_count = 0  # Count of NaN/Inf occurrences in current epoch
        self.total_nan_inf_count = 0  # Total count across all epochs

    def relative_l2_loss(self, pred, target):
        """
        Calculate relative L2 loss for each sample in the batch:
        ||pred - target||_2 / ||target||_2
        
        Args:
            pred: Predicted tensor of shape [batch_size, channels, ...]
            target: Target tensor of shape [batch_size, channels, ...]
            
        Returns:
            Mean of relative L2 losses across the batch
        """
        # Reshape tensors to [batch_size, -1] to calculate norm along all dimensions except batch
        batch_size = pred.size(0)
        pred_flat = pred.view(batch_size, -1)
        target_flat = target.view(batch_size, -1)
        
        # Calculate L2 norm of the difference for each sample (numerator)
        diff_norm = torch.norm(pred_flat - target_flat, p=2, dim=1)
        
        # Calculate L2 norm of the target for each sample (denominator)
        target_norm = torch.norm(target_flat, p=2, dim=1)
        
        # Add small epsilon to prevent division by zero
        epsilon = 1e-8
        
        # Calculate relative L2 loss for each sample
        rel_l2_loss = diff_norm / (target_norm + epsilon)
        
        # Return mean loss across the batch
        return torch.mean(rel_l2_loss)
    
    def forward(self, x1):
        return self.model(x1)
    
    def _process_train_batch(self, batch, batch_idx):
        img1 = batch
        output = self(img1)
        
        batch_target1, batch_target2, batch_target3 = [], [], []
        valid_indices = []

        # Calculate global indices for this batch
        batch_size = img1.size(0)
        global_start_idx = batch_idx * batch_size

        for i in range(output.size(0)):
            single_img1 = img1[i].cpu().numpy().squeeze()
            single_output = output[i].detach().cpu().numpy()

            alpha_unnorm = (2.0 - 1.0)*single_img1[0,0] + 1.0
            
            
            
            t1 = run_fipy_custom(alpha = alpha_unnorm,
                                    num_iter = self.fem_iterations,
                                    p0 = single_output[0],
                                    nx = 32,
                                    L = 1.0,
                                    dt = 1.0/32)
            



            # Check for NaN or Inf values
            if not (np.isnan(t1).any() or np.isinf(t1).any() ):
                t1 = torch.from_numpy(np.array(t1)).float().unsqueeze(0).unsqueeze(0).to(self.device)
                batch_target1.append(t1)
                valid_indices.append(i)
            else:
                # Track problematic samples
                global_idx = global_start_idx + i
                self.nan_inf_indices.append(global_idx)
                self.epoch_nan_inf_count += 1
                self.total_nan_inf_count += 1
                
                current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                print(f"[{current_time}], Error: NaN/Inf detected in sample {global_idx}")
            
        if len(valid_indices) > 0:
            target1 = torch.cat(batch_target1, dim=0)
            target = torch.cat([target1], dim=1)

            valid_output = output[valid_indices]
            
            # Calculate loss
            loss = self.criterion(valid_output, target)
            #loss = self.criterion_rel(valid_output, target)
            
            return loss, len(valid_indices)
        
        return None, 0
    
    def _process_val_batch(self, batch, batch_idx):
        img1, p_eq = batch
        output = self(img1)

        # Ground truth
        target1 = p_eq.unsqueeze(1).to(self.device)
       
        target = torch.cat([target1], dim=1)

        # Predicted
        pred_p = output[:, 0:1]
        true_p = target[:, 0:1]

        loss = self.criterion(pred_p, true_p)

        return loss, img1.size(0)

    def training_step(self, batch, batch_idx):
        loss, valid_count = self._process_train_batch(batch, batch_idx)
        
        if loss is not None:
            # Log metrics without progress bar
            self.log('train_loss', loss, prog_bar=False, sync_dist=True)
            self.log('train_valid_samples', valid_count, prog_bar=False, sync_dist=True)
            return loss
        
        # Return zero loss if no valid samples (will not contribute to gradients)
        return torch.tensor(0.0, requires_grad=True, device=self.device)
    
    def validation_step(self, batch, batch_idx):
        loss, valid_count = self._process_val_batch(batch, batch_idx)
        
        if loss is not None:
            # Log metrics without progress bar
            self.log('val_loss', loss, prog_bar=False, sync_dist=True)
            self.log('val_valid_samples', valid_count, prog_bar=False, sync_dist=True)
        
        # Force CUDA cache clearing
        torch.cuda.empty_cache()
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)

        scheduler = CosineAnnealingLR(
                        optimizer,
                        T_max=self.Tmax,       # total epochs
                        eta_min=1e-8    # minimum learning rate
                    )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1
            }
        }

    def on_train_epoch_start(self):
        # Reset the epoch counter
        self.epoch_nan_inf_count = 0

    def on_train_epoch_end(self):
        # Access the trainer to get logged metrics
        train_loss = self.trainer.callback_metrics.get('train_loss', torch.tensor(0.0))
        self.train_losses.append(train_loss.item())
        
        # Log NaN/Inf statistics
        current_epoch = self.trainer.current_epoch
        print(f"Epoch {current_epoch}: Found {self.epoch_nan_inf_count} samples with NaN/Inf values")
        self.log('nan_inf_count', self.epoch_nan_inf_count, prog_bar=False, sync_dist=True)
        
        # Save problematic indices to a file at regular intervals
        if current_epoch % 5 == 0 or current_epoch == self.trainer.max_epochs - 1:
            self._save_problematic_indices()

    def _save_problematic_indices(self):
        """Save the indices of problematic samples to a file."""
        save_dir = self.trainer.checkpoint_callbacks[0].dirpath  # Get the checkpoint directory
        filename = os.path.join(save_dir, f"nan_inf_indices_epoch_{self.trainer.current_epoch}.txt")
        
        with open(filename, 'w') as f:
            f.write(f"Total NaN/Inf samples: {self.total_nan_inf_count}\n")
            f.write("Global indices of problematic samples:\n")
            for idx in self.nan_inf_indices:
                f.write(f"{idx}\n")
        
        print(f"Saved problematic indices to {filename}")
    
    def on_train_epoch_end(self):
        # Access the trainer to get the logged metrics
        train_loss = self.trainer.callback_metrics.get('train_loss', torch.tensor(0.0))
        self.train_losses.append(train_loss.item())
    
    def on_validation_epoch_end(self):
        # Access the trainer to get the logged metrics
        val_loss = self.trainer.callback_metrics.get('val_loss', torch.tensor(0.0))
        self.val_losses.append(val_loss.item())


class SimplifiedOutputCallback(Callback):
    def __init__(self, num_epochs):
        super().__init__()
        self.num_epochs = num_epochs
    
    def on_train_epoch_end(self, trainer, pl_module):
        epoch = trainer.current_epoch
        train_loss = trainer.callback_metrics.get('train_loss', torch.tensor(0.0)).item()
        val_loss = trainer.callback_metrics.get('val_loss', torch.tensor(0.0)).item()

        # Print GPU memory usage
        #for i in range(torch.cuda.device_count()):
        #    print(f"GPU {i} memory: {torch.cuda.memory_allocated(i)/1e9:.2f}GB / {torch.cuda.get_device_properties(i).total_memory/1e9:.2f}GB")
        
        # Print the customized output
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f'[{current_time}] Epoch [{epoch+1}/{self.num_epochs}] Training Loss: {train_loss:.10f}, Validation Loss: {val_loss:.10f}')