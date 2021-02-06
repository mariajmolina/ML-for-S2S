import os
import fnmatch
import numpy as np
import pandas as pd
import xarray as xr
from itertools import product

def month_num_to_string(number):
    """
    Convert number to three-letter month.
    
    Args:
        number: Month in number format.
    """
    m = {
         1: 'jan',
         2: 'feb',
         3: 'mar',
         4: 'apr',
         5: 'may',
         6: 'jun',
         7: 'jul',
         8: 'aug',
         9: 'sep',
         10: 'oct',
         11: 'nov',
         12: 'dec'
        }
    
    try:
        out = m[int(number)]
        return out
    
    except:
        raise ValueError('Not a month')

def create_cesm2_folders(variable, parent_directory, start='1999-01-01', end='2019-12-31', freq='W-MON'):
    """
    Create folders to place new variable files that were not preprocessed p1 (or other SubX priority).
    
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
        
    return

def create_cesm2_files(variable, parent_directory, ensemble, start='1999-01-01', end='2019-12-31', freq='W-MON'):
    """
    Create CESM2 variable files that were not preprocessed p1 (or other SubX priority) variables.
    Here we extract variable from daily file containing many variables to reduce memory usage.
    
    Args:
        variable (str): Name of variable in lower case (e.g., 'sst').
        parent_directory (str): Directory where files are located (e.g., '/glade/scratch/$USER/s2s/').
        ensemble (str): Two digit ensemble member of hindcast (e.g., '09').
        start (str): Start of hindcasts. Defaults to '1999-01-01' for CESM2.
        end (str): End of hindcasts. Defaults to '2019-12-31' for CESM2.
        freq (str): Frequency of hindcast starts. Defaults to 'W-MON' for CESM2.
    """
    # date array
    d1 = pd.date_range(start=start, end=end, freq=freq)
    
    for root, dirnames, filenames in os.walk(f'{parent_directory}CESM2/temp/'):

        for num, (yr, mo, dy) in enumerate(zip(d1.strftime("%Y"), d1.strftime("%m"), d1.strftime("%d"))):

            for filename in fnmatch.filter(filenames, f'cesm2cam6v2*{yr}-{mo}-{dy}.{ensemble}.cam.h2.{yr}-{mo}-{dy}-00000.nc'):

                ds = xr.open_dataset(root+filename)[variable.upper()]
                ds.to_dataset(
                    name=variable.upper()).to_netcdf(
                    f'{parent_directory}CESM2/{variable}/{yr}/{mo}/{variable}_cesm2cam6v2_{dy}{month_num_to_string(mo)}{yr}_00z_d01_d46_m{ensemble}.nc')
                
    return

def cesm2_filelist(variable, parent_directory, ensemble, start='1999-01-01', end='2019-12-31', freq='W-MON'):
    """
    Create list of variable files.
    
    Args:
        variable (str): Name of variable (e.g., 'zg_200').
        parent_directory (str): Directory where files are located (e.g., '/glade/scratch/$USER/s2s/').
        ensemble (str or list of str): Two digit ensemble member of hindcast (e.g., '09') or list (e.g., ['00', '01']).
        start (str): Start of hindcasts. Defaults to '1999-01-01' for CESM2.
        end (str): End of hindcasts. Defaults to '2019-12-31' for CESM2.
        freq (str): Frequency of hindcast starts. Defaults to 'W-MON' for CESM2.
    """
    # date array
    d1 = pd.date_range(start=start, end=end, freq=freq)
    
    # generate list
    matches = []
    for num, (yr, mo, dy) in enumerate(zip(d1.strftime("%Y"), d1.strftime("%m"), d1.strftime("%d"))):
        
        for root, dirnames, filenames in os.walk(f'{parent_directory}CESM2/{variable}/{yr}/{mo}/'):
            
            if isinstance(ensemble, str):
                for filename in fnmatch.filter(filenames, f'*_cesm2cam6v2_{dy}*_m{ensemble}.nc'):

                    thefile = os.path.join(root, filename)

                    if os.access(thefile, os.R_OK):
                        matches.append(thefile)

                    if not os.access(thefile, os.R_OK):
                        matches.append(np.nan)
                        
            if isinstance(ensemble, list):
                for ens in ensemble:
                    for filename in fnmatch.filter(filenames, f'*_cesm2cam6v2_{dy}*_m{ens}.nc'):
                        
                        thefile = os.path.join(root, filename)

                        if os.access(thefile, os.R_OK):
                            matches.append(thefile)

                        if not os.access(thefile, os.R_OK):
                            matches.append(np.nan)
                
    return matches

def cesm2_hindcast_climatology(filelist, variable, save=False, author=None, parent_directory=None):
    """
    Create CESM2 hindcast climatology. Outputs array (lon, lat, lead, 365).
    Translated from MATLAB (provided by Anne Sasha Glanville, NCAR).
    
    Args:
        filelist (list of str): List of file names and directory locations.
        variable (str): Name of variable (e.g., 'zg_200').
        save (boolean): Set to True if want to save climatology as netCDF. Defaults to False.
        author (str): Author of file. Defaults to None.
        parent_directory (str): Directory where files are located (e.g., '/glade/scratch/$USER/s2s/').
                                Defaults to None.
    """
    if save:
        assert isinstance(author, str), "Please set author for file saving."
    if save:
        assert isinstance(parent_directory, str), "Please set parent_directory to save file to."
    
    dateStrPrevious = '01jan1000'        # just a random old date that doesn't exist
    index_help = 0                       # set to 0 for the very first file date
    char_1 = "cesm2cam6v2_"              # date string help
    char_2 = "_00z_d01_d46" 
    grab_ensembles = True

    # loop through list of hindcast files
    for tline in filelist:

        fil = tline
        dateStr = fil[fil.find(char_1)+12 : fil.find(char_2)]
        starttime = pd.to_datetime(dateStr)
        doy = starttime.dayofyear

        if (starttime.year % 4) == 0 and starttime.month > 2:
            doy = doy - 1

        var = xr.open_dataset(fil)[variable].transpose('lon','lat','time').values      # (lon,lat,lead); load file and grab variable
        varChosen = var

        if varChosen.shape[2] != 46:
            varChosen = np.ones((varChosen.shape)) * np.nan

        if index_help == 0:
            climBin = np.zeros((
                varChosen.shape[0], varChosen.shape[1], varChosen.shape[2], 365))      # (lon, lat, lead, 365 days in year)
            climBinDays = np.zeros((
                varChosen.shape[0], varChosen.shape[1], varChosen.shape[2], 365))
            lon = xr.open_dataset(fil)[variable].coords['lon'].values                  # grab lon and lat arrays
            lat = xr.open_dataset(fil)[variable].coords['lat'].values
            if grab_ensembles:
                all_ensembles = []
                all_ensembles.append(tline[tline.find('_m')+2:tline.find('_m')+4])     # saving ensemble members for attrs

        if dateStr == dateStrPrevious:                           # if dates match, means you are on next ensemble member
            x += 1                                               # to compute ensemble mean
            ensAvg = (ensAvg * (x - 1) + varChosen) / x
            if grab_ensembles:
                all_ensembles.append(tline[tline.find('_m')+2:tline.find('_m')+4])
        else:
            if index_help != 0:                 # if dates don't match, but make sure we are past the first file and ensAvg has data
                if not np.all(ensAvg == 0):          
                    climBin[:,:,:,doyPrevious - 1] = climBin[:,:,:,doyPrevious - 1] + ensAvg      # doyPrevious - 1 bc 0-based index
                    climBinDays[:,:,:,doyPrevious - 1] = climBinDays[:,:,:,doyPrevious - 1] + 1
                    grab_ensembles = False
            ensAvg = varChosen
            x = 1

        dateStrPrevious = dateStr
        doyPrevious = doy
        index_help += 1

    climBin[:,:,:,doyPrevious - 1] = climBin[:,:,:,doyPrevious - 1] + ensAvg
    climBinDays[:,:,:,doyPrevious - 1] = climBinDays[:,:,:,doyPrevious - 1] + 1
    clim = climBin / climBinDays                                                         # Final climatology (sum/n)

    first_date = filelist[0][filelist[0].find(char_1)+12:filelist[0].find(char_2)]       # date strings for attrs
    final_date = filelist[-1][filelist[-1].find(char_1)+12:filelist[-1].find(char_2)]

    data_assemble = xr.Dataset({
                         'clim':(['x','y','lead','doy'],clim),
                        },
                         coords =
                        {'lead':(['lead'],np.arange(0,clim.shape[2],1)),
                         'doy':(['doy'],np.arange(1,clim.shape[3]+1,1)),
                         'lat':(['y'],lat),
                         'lon':(['x'],lon)
                        },
                        attrs = 
                        {'File Author' : author,
                         'Ensembles' : all_ensembles,
                         'First Date (lead=0)' : pd.to_datetime(first_date),
                         'Final Date (lead=0)' : pd.to_datetime(final_date)})
    
    if not save:
        return data_assemble

    if save:
        data_assemble.to_netcdf(f'{parent_directory}CESM2/{variable}_clim_cesm2cam6v2_{len(all_ensembles)}members_s2s_data.nc')
