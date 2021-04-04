import numpy as np
import xarray as xr
from itertools import product
from minisom import MiniSom, asymptotic_decay

def start_som(subsetarray, input_length, som_grid_rows, som_grid_columns, 
              sigma=0.3, learning_rate=0.5, decay_function=asymptotic_decay, neighborhood_function='gaussian', 
              topology='rectangular', activation_distance='euclidean', random_seed=1,
              num_iteration=1000, random_order=True, verbose=True, pca_weights=False):
    """
    Assemble and train SOM. 
    
    Args:
        subsetarray (xarray array): Assembled data for training in (time, new) shape.
        input_length (int): Size of "new" dimension. 
        som_grid_rows (int): Number of som rows (y-axis).
        som_grid_columns (int): Number of som columns (x-axis).
        sigma (float): Sigma for neighborhood_function initial radius of influence. Defaults to 0.3.
        learning_rate (float): Defaults to 0.5.
        decay_function (function): Defaults to "asymptotic_decay".
        neighborhood_function (str): "gaussian".
        topology (str): Defaults to "rectangular".
        activation_distance (str): Defaults to "euclidean".
        random_seed (int): Random seed for training. Defaults to 1.
        num_iteration (int): Number of training iterations. Defaults to 10000.
        random_order (boolean): Defaults to True.
        verbose (boolean): Defaults to True.
        pca_weights (boolean): Defaults to False. If True, weight initialize using pca instead of random.
        
    Returns:
        Trained som, data indices within som lattice, dictionary keys.
    """
    som = MiniSom(som_grid_rows, som_grid_columns, input_length, sigma,
                  learning_rate, decay_function, neighborhood_function,
                  topology, activation_distance, random_seed) 
    
    data = subsetarray
    
    data = data.fillna(0.0).values
    
    # weight initialization
    if not pca_weights:
        som.random_weights_init(data)   
    if pca_weights:
        som.pca_weights_init(data)
    
    # train the SOM
    som.train(
            data,
            num_iteration,
            random_order,
            verbose)
    
    # grabbing indices from SOM
    # create an empty dictionary using the rows and columns of SOM
    keys = [i for i in product(range(som_grid_rows),range(som_grid_columns))]
    winmap = {key: [] for key in keys}
    # grab the indices for the data within the SOM lattice
    for i, x in enumerate(subsetarray.values):
        winmap[som.winner(x)].append(i)
        
    # create list of the dictionary keys
    som_keys = list(winmap.keys())
    print(f"Number of composite maps: {len(som_keys)}")
    return som, winmap, som_keys
