import numpy as np
import xarray as xr
from itertools import product
from minisom import MiniSom, asymptotic_decay

"""

Function for initiating and training a self-organizing map.

Author: Maria J. Molina, NCAR (molina@ucar.edu)

"""

def start_som(data, input_length, som_grid_rows, som_grid_columns, 
              sigma=0.3, learning_rate=0.5, decay_function=asymptotic_decay, 
              neighborhood_function='gaussian', 
              topology='rectangular', activation_distance='euclidean', random_seed=1,
              num_iteration=1000, random_order=True, verbose=True, pca_weights=False):
    """
    Assemble and train SOM. 
    
    Args:
        subsetarray           (xarray array): assembled data for training in (time, new) shape.
        input_length          (int):          size of "new" dimension. 
        som_grid_rows         (int):          number of som rows (y-axis).
        som_grid_columns      (int):          number of som columns (x-axis).
        sigma                 (float):        sigma for neighborhood_function initial radius of influence. Defaults to 0.3.
        learning_rate         (float):        defaults to 0.5.
        decay_function        (function):     defaults to "asymptotic_decay".
        neighborhood_function (str):          "gaussian".
        topology              (str):          defaults to "rectangular".
        activation_distance   (str):          defaults to "euclidean".
        random_seed           (int):          random seed for training. Defaults to 1.
        num_iteration         (int):          number of training iterations. Defaults to 10000.
        random_order          (boolean):      defaults to True.
        verbose               (boolean):      defaults to True.
        pca_weights           (boolean):      defaults to False. If True, weight initialize using pca instead of random.
        
    Returns:
        Trained som, data indices within som lattice, dictionary keys.
        
    """
    som = MiniSom(som_grid_rows, som_grid_columns, input_length, sigma,
                  learning_rate, decay_function, neighborhood_function,
                  topology, activation_distance, random_seed) 
    
    if not pca_weights:
        
        som.random_weights_init(data)
        
    if pca_weights:
        
        som.pca_weights_init(data)
    
    som.train(
            data,
            num_iteration,
            random_order,
            verbose)
    
    # grabbing indices from SOM; create an empty dictionary using the rows and columns of SOM
    
    keys = [i for i in product(range(som_grid_rows), range(som_grid_columns))]
    winmap = {key: [] for key in keys}
    
    for i, x in enumerate(data): # grab the indices for the data within the SOM lattice
        
        winmap[som.winner(x)].append(i)
        
    som_keys = list(winmap.keys()) # create list of the dictionary keys
    print(f"Number of composite maps: {len(som_keys)}")
    
    return som, winmap, som_keys
