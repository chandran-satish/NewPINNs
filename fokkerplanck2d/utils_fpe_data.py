import numpy as np
import torch
import random
import os
import sys
from contextlib import contextmanager
import h5py
import fipy as fp

def main():
    # Save the dataset
    # use seed = 42 for training data
    # Use seed = 43 for validation data
    

    num_samples = 256
    save_path = f'./data/train_data_{num_samples}.h5'
    myseed = 42
    utils_save_train_dataset(utils_sample_single, num_samples = num_samples, save_path = save_path, myseed = myseed)
    
    num_samples = 32
    save_path = f'./data/valid_data_{num_samples}.h5'
    myseed = 43
    utils_save_valid_dataset(utils_sample_single, num_samples = num_samples, save_path = save_path, myseed = myseed)
    


@contextmanager
def utils_suppress_stdout():
    with open(os.devnull, 'w') as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout

def utils_sample_single():
    grid_size = 32

    alpha = random.random() # normalized potential amplitude
    #alpha = 0.0

    alpha_array = alpha*np.ones((grid_size, grid_size))


    return alpha_array

def utils_save_train_dataset(get_data_func, num_samples, save_path, myseed):
    """
    Generate samples using get_data_func and save them to an HDF5 file.
    
    Parameters:
    get_data_func: Function that returns (img1, img2, target1, target2, target3)
    num_samples: Number of samples to generate
    save_path: Path to save the HDF5 file
    """
    # Create the directory if it doesn't exist
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    
    with h5py.File(save_path, 'w') as f:
        # Create datasets
        for i in range(num_samples):
            # Get one sample of Re and vlid arrays
            img1 = get_data_func()
            img1 = img1.astype(np.float32)
            
            # Create groups for first sample to setup the datasets
            if i == 0:
                f.create_dataset('img1', (num_samples, *img1.shape), dtype=img1.dtype)
            
            # Store the data
            f['img1'][i] = img1
            
            # Print progress
            if (i + 1) % 100 == 0 or i==num_samples-1:
                print(f'Saved {i + 1}/{num_samples} samples')

def utils_save_valid_dataset(get_data_func, num_samples, save_path, myseed):
    """
    Generate samples using get_data_func and save them to an HDF5 file.
    
    Parameters:
    get_data_func: Function that returns (img1, img2, target1, target2, target3)
    num_samples: Number of samples to generate
    save_path: Path to save the HDF5 file
    """
    # Create the directory if it doesn't exist
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    
    with h5py.File(save_path, 'w') as f:
        # Create datasets
        for i in range(num_samples):
            # Get one sample of alpha arrays
            img1 = get_data_func()
            img1 = img1.astype(np.float32)
            
            # get the equillibrium solutions for the density

            mesh = fp.Grid2D(nx=32, ny=32, dx=1.0/32, dy=1.0/32)
            x_coords, y_coords = mesh.cellCenters 
            V = fp.CellVariable(mesh=mesh)
            alpha = (2.0 - 1.0)*img1[0,0] + 1.0
            V.setValue(alpha*np.sin(2*np.pi*x_coords)*np.sin(2*np.pi*y_coords))
            D = 1.0
            p_analytic = np.exp(-V.value / D)
            p_analytic /= (p_analytic * mesh.cellVolumes).sum()
            p_eq = p_analytic.reshape((32, 32), order='F')
            
            
            
            
            # Create groups for first sample to setup the datasets
            if i == 0:
                f.create_dataset('img1', (num_samples, *img1.shape), dtype=img1.dtype)
                f.create_dataset('p_eq', (num_samples, *p_eq.shape), dtype=np.float32)
                
            
            # Store the data
            f['img1'][i] = img1
            f['p_eq'][i] = p_eq.astype(np.float32)
            
            # Print progress
            if (i + 1) % 100 == 0 or i==num_samples-1:
                print(f'Saved {i + 1}/{num_samples} samples')

def utils_load_dataset(file_path='dataset.h5'):
    """
    Load dataset from HDF5 file and return as a list of tuples.
    
    Parameters:
    file_path: Path to the HDF5 file
    
    Returns:
    list of tuples (img1, img2, target1, target2, target3) as torch tensors
    """
    with h5py.File(file_path, 'r') as f:
        # Get the number of samples
        num_samples = f['img1'].shape[0]
        
        # Load all data into memory
        training_data = []
        for i in range(num_samples):
            sample = (
                torch.from_numpy(f['img1'][i]),  # Add channel dimension
            )
            training_data.append(sample)
            
            # Print progress
            if (i + 1) % 100 == 0 or i==num_samples-1:
                print(f'Loaded {i + 1}/{num_samples} samples')
                
    return training_data


if __name__ == "__main__":
    main()

    
