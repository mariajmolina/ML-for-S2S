import os
import fnmatch
import numpy as np
import pandas as pd
from itertools import product

def cesm2_dictionary(variable, parent_directory, ensemble, start='1999-01-01', end='2019-12-31', freq='W-MON'):
    """
    Create dictionary of variable files for use in ML training.
    
    Args:
        variable (str): Name of variable (e.g., 'zg_200').
        parent_directory (str): Directory where files are located (e.g., '/glade/scratch/$USER/s2s/').
        ensemble (str): Ensemble member of hindcast (e.g., 'm09').
        start (str): Start of hindcasts. Defaults to '1999-01-01' for CESM2.
        end (str): End of hindcasts. Defaults to '2019-12-31' for CESM2.
        freq (str): Frequency of hindcast starts. Defaults to 'W-MON' for CESM2.
    """
    # date array
    d1 = pd.date_range(start=start, end=end, freq=freq)
    # generate dictionary
    matches = {}
    for num, (yr, mo, dy) in enumerate(zip(d1.strftime("%Y"), d1.strftime("%m"), d1.strftime("%d"))):
        for root, dirnames, filenames in os.walk(f'{parent_directory}CESM2/{variable}/{yr}/{mo}/'):
            for filename in fnmatch.filter(filenames, f'*_cesm2cam6v2_{dy}*_{ensemble}.nc'):
                thefile = os.path.join(root, filename)
                if os.access(thefile, os.R_OK):
                    matches[num] = thefile
                if not os.access(thefile, os.R_OK):
                    matches[num] = np.nan
    return matches

def create_cesm2_folders(variable, parent_directory, start='1999-01-01', end='2019-12-31', freq='W-MON'):
    """
    Create folders to place new variable files in.
    
    Args:
        variable (str): Name of variable (e.g., 'sst').
        parent_directory (str): Directory to place files (e.g., '/glade/scratch/$USER/s2s/').
        start (str): Start of hindcasts. Defaults to '1999-01-01' for CESM2.
        end (str): End of hindcasts. Defaults to '2019-12-31' for CESM2.
        freq (str): Frequency of hindcast starts. Defaults to 'W-MON' for CESM2.
    """
    # date array
    d1 = pd.date_range(start=start, end=end, freq=freq)
    # generate folders for new variable
    if os.path.exists(parent_directory):
        for yr, mo in product(np.unique(d1.strftime("%Y")), np.unique(d1.strftime("%m"))):
            new_directory = 'CESM2/'+variable+'/'+yr+'/'+mo
            path = os.path.join(parent_directory, new_directory)
            try: 
                os.makedirs(path, exist_ok = True) 
                print("Directory '%s' created successfully" % new_directory) 
            except OSError as error: 
                print("Directory '%s' cannot be created" % new_directory)
    if not os.path.exists(parent_directory):
        print('Parent directory does not exist.')
