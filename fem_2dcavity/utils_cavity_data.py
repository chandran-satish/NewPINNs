import numpy as np
import torch
import random
import os
import sys
from contextlib import contextmanager
import h5py
from utils_cavity_ngsolve import*

def main():
    # Save the dataset
    # use seed = 42 for training data
    # Use seed = 43 for validation data
    '''
    num_samples = 1
    save_path = f'./data/train_data_{num_samples}.h5'
    myseed = 42
    np.random.seed(myseed)
    utils_save_dataset(utils_sample_single, num_samples = num_samples, save_path = save_path, myseed = myseed)

    num_samples = 1
    save_path = f'./data/valid_data_{num_samples}.h5'
    myseed = 42
    np.random.seed(myseed)
    utils_save_dataset(utils_sample_single, num_samples = num_samples, save_path = save_path, myseed = myseed)
    '''

    '''
    num_samples = 128
    save_path = f'./data/train_data_{num_samples}.h5'
    myseed = 42
    utils_save_dataset(utils_sample_single, num_samples = num_samples, save_path = save_path, myseed = myseed)

    num_samples = 32
    save_path = f'./data/valid_data_{num_samples}.h5'
    myseed = 43
    utils_save_dataset(utils_sample_single, num_samples = num_samples, save_path = save_path, myseed = myseed)
    '''

    #num_samples = 32
    #save_path = f'./data/train_data_{num_samples}.h5'
    #myseed = 42
    #utils_save_train_dataset(utils_sample_single, num_samples = num_samples, save_path = save_path, myseed = myseed)
    
    num_samples = 64
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

    #Re_min = 2000
    #Re_max = 3000

    #Re = (Re_max - Re_min)*random.random() + Re_min

    Re = random.random() # normalized Reynolds number
    #Re = 0.5
    #print(f'Re = {Re}')

    Re_array = Re*np.ones((grid_size, grid_size))


    return Re_array

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
            # Get one sample of Re arrays
            img1 = get_data_func()
            img1 = img1.astype(np.float32)
            
            # get the equillibrium solutions for the x velocity, y velocity, and pressure
            
            U_initial = np.zeros((32,32))
            V_initial = np.zeros((32,32))
            P_initial = np.zeros((32,32))
            uin_max = 1.0
            tau = 0.003
            t_iter = 250000
            re_max = 3000
            re_min = 2000
            reynolds_num = (re_max - re_min)*img1[0,0] + re_min
            print(f'Sample {i}: Re = {reynolds_num}')
            Ueq, Veq, Peq, _ = run_ngsolve_custom(nu = 1.0/reynolds_num , uin_max = uin_max, tau = tau, t_iter = t_iter, U_initial = U_initial, V_initial = V_initial, P_initial = P_initial)
            
            
            
            
            # Create groups for first sample to setup the datasets
            if i == 0:
                f.create_dataset('img1', (num_samples, *img1.shape), dtype=img1.dtype)
                f.create_dataset('Ueq', (num_samples, *Ueq.shape), dtype=np.float32)
                f.create_dataset('Veq', (num_samples, *Veq.shape), dtype=np.float32)
                f.create_dataset('Peq', (num_samples, *Peq.shape), dtype=np.float32)
            
            # Store the data
            f['img1'][i] = img1
            f['Ueq'][i] = Ueq.astype(np.float32)
            f['Veq'][i] = Veq.astype(np.float32)
            f['Peq'][i] = Peq.astype(np.float32)
            
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

    
