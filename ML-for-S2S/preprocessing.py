import os
import fnmatch
import calendar
import numpy as np
import pandas as pd
import xarray as xr
from itertools import product
from util import month_num_to_string
import xesmf as xe

"""

Module contains several functions for preprocessing S2S hindcasts.

Author: Maria J. Molina, NCAR (molina@ucar.edu)

Contributions from Sasha Anne Glanville, NCAR

"""

def regrid_mask(ds, variable, reuse_weights=False):
    """
    Function to regrid onto coarser ERA5 grid (0.25-degree).

    Args:
        ds (xarray dataset): file.
        variable (str): variable.
        reuse_weights (boolean): Whether to use precomputed weights to speed up calculation.
                                 Defaults to ``False``.
    Returns:
        Regridded mask file for use with machine learning model.
    """
    ds_out = xe.util.grid_2d(lon0_b=0-0.5,   lon1_b=360-0.5, d_lon=1., 
                             lat0_b=-90-0.5, lat1_b=90,      d_lat=1.)

    regridder = xe.Regridder(ds, ds_out, method='nearest_s2d', reuse_weights=reuse_weights)

    return regridder(ds[variable])


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
    d1 = pd.date_range(start=start, end=end, freq=freq)
    
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
    Contain daily files in ``/temp/`` sub-folder.
    
    Args:
        variable (str): Name of variable in lower case (e.g., 'sst').
        parent_directory (str): Directory where files are located (e.g., '/glade/scratch/$USER/s2s/').
        ensemble (str): Two digit ensemble member of hindcast (e.g., '09').
        start (str): Start of hindcasts. Defaults to '1999-01-01' for CESM2.
        end (str): End of hindcasts. Defaults to '2019-12-31' for CESM2.
        freq (str): Frequency of hindcast starts. Defaults to 'W-MON' for CESM2.
        
    """
    d1 = pd.date_range(start=start, end=end, freq=freq)
    
    for root, dirnames, filenames in os.walk(f'{parent_directory}CESM2/temp/'):
        
        for num, (yr, mo, dy) in enumerate(zip(d1.strftime("%Y"), d1.strftime("%m"), d1.strftime("%d"))):
            
            if yr == '2016' and mo == '02' and dy == '29':
                
                dy = '28'
                
            for filename in fnmatch.filter(filenames, f'cesm2cam6v2*{yr}-{mo}-{dy}.{ensemble}.cam.h2.{yr}-{mo}-{dy}-00000.nc'):
                
                ds = xr.open_dataset(root+filename)[variable.upper()]
                
                ds.to_dataset(name=variable.upper()).to_netcdf(
                    f'{parent_directory}CESM2/{variable}/{yr}/{mo}/{variable}_cesm2cam6v2_{dy}{month_num_to_string(mo)}{yr}_00z_d01_d46_m{ensemble}.nc')
    
    return


def create_cesm2_pressure_files(filelist, variable, pressure=300.):
    """
    Create CESM2 variable files that were not preprocessed p1 (or other SubX priority) variables.
    Here we extract variables on a pressure level from files containing many pressure levels 
    to reduce memory usage.
    
    Args:
        filelist (list of str): List of file names and directory locations.
        variable (str): Name of variable in lower case (e.g., 'sst').
        pressure (float): Pressure level. Defaults to ``300.``
        
    """
    for fil in filelist:
        
        ds = xr.open_dataset(fil).sel(lev_p=pressure).drop('lev_p')
        
        ds.to_netcdf(f"{fil.split(variable)[0]}{variable}_temp{fil.split(variable)[1]}{fil.split('/')[-1]}")
        
    return


def gpcp_filelist(parent_directory, start='1999-01-01', end='2019-12-31', freq='D'):
    """
    Create list of daily GPCP Version 2.3 Combined Precipitation Data Set files.
    https://www.ncei.noaa.gov/data/global-precipitation-climatology-project-gpcp-daily/access/
    
    Args:
        parent_directory (str): Directory where files are located (e.g., '/glade/scratch/$USER/s2s/').
        start (str): Start of hindcasts. Defaults to '1999-01-01' for CESM2.
        end (str): End of hindcasts. Defaults to '2019-12-31' for CESM2.
        freq (str): Frequency of hindcast starts. Defaults to 'D' for daily.
        
    """
    d1 = pd.date_range(start=start, end=end, freq=freq)
    
    matches = []
    
    for num, (yr, mo, dy) in enumerate(zip(d1.strftime("%Y"), d1.strftime("%m"), d1.strftime("%d"))):
        
        if mo == '02' and dy == '29':
            
            continue # skip leap years
        
        for root, dirnames, filenames in os.walk(f'{parent_directory}/'):

            for filename in fnmatch.filter(filenames, f'*_daily_d{yr}{mo}{dy}_c*.nc'):

                thefile = os.path.join(root, filename)

                if os.access(thefile, os.R_OK):

                    matches.append(thefile)

                if not os.access(thefile, os.R_OK):

                    matches.append(np.nan)
                
    return matches


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
    d1 = pd.date_range(start=start, end=end, freq=freq)
    
    matches = []
    
    for num, (yr, mo, dy) in enumerate(zip(d1.strftime("%Y"), d1.strftime("%m"), d1.strftime("%d"))):
        
        if mo == '02' and dy == '29':
            
            dy = '28'
        
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


def gpcp_climatology(filelist, variable='precip', save=False, author=None, parent_directory=None):
    """
    Create GPCP Version 2.3 Combined Precipitation Data Set climatology.
    
    Args:
        filelist (list of str): List of file names and directory locations.
        save (boolean): Set to True if want to save climatology as netCDF. Defaults to False.
        author (str): Author of file. Defaults to None.
        parent_directory (str): Directory where files are located (e.g., '/glade/scratch/$USER/s2s/').
                                Defaults to None.
                                
    """
    if save:
        
        assert isinstance(author, str), "Please set author for file saving."
        assert isinstance(parent_directory, str), "Please set parent_directory to save file to."

    clim = np.zeros((int(len(filelist)/365), 365, 180, 360))

    doy = 0
    yr = 0
    dates = []
    years = []

    for num, file in enumerate(filelist):

        ds = xr.open_dataset(file)
        ds = ds[variable].isel(time=0)

        dates.append(pd.Timestamp(ds.time.values))
        
        ds = ds.where(ds>=0.,0.)  # valid range: [0.,100.]
        ds = ds.where(ds<=100.,100.)

        if num == 0:

            lats = ds.latitude.values
            lons = ds.longitude.values

        clim[yr,doy,:,:] = ds.values

        doy += 1

        if doy == 365:

            doy = 0
            yr += 1
            
            years.append(int(ds.time.dt.strftime('%Y').values))

    data_assemble = xr.Dataset({
                         'clim': (['time','lat','lon'], np.nanmean(clim, axis=0)),
                        },
                         coords =
                        {'date_range': (['date_range'], pd.to_datetime(dates)),
                         'time': (['time'], np.arange(1,365 + 1,1)),
                         'lat' : (['lat'], lats),
                         'lon' : (['lon'], lons)
                        },
                        attrs = 
                        {'File Author' : author,
                         'Years' : np.array(years)})

    if not save:

        return data_assemble

    if save:

        data_assemble.to_netcdf(f'{parent_directory}CESM2_OBS/{variable.lower()}_clim_gpcp_data.nc')


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
        assert isinstance(parent_directory, str), "Please set parent_directory to save file to."
    
    dateStrPrevious = '01jan1000' # just a random date
    index_help = 0
    char_1 = "cesm2cam6v2_"
    char_2 = "_00z_d01_d46"
    grab_ensembles = True
    
    for fil in filelist:

        dateStr = fil[fil.find(char_1)+12 : fil.find(char_2)]
        starttime = pd.to_datetime(dateStr)
        doy = starttime.dayofyear

        if (starttime.year % 4) == 0 and starttime.month > 2:
            
            doy = doy - 1

        var = xr.open_dataset(fil)[variable].transpose('lon','lat','time').values # (lon,lat,lead); load file and grab variable
        
        varChosen = var
        
        if variable == 'pr' or variable == 'pr_sfc':
            
            varChosen = varChosen * 84600 # convert kg/m2/s to mm/day
            
        if variable == 'tas_2m':
            
            varChosen = varChosen - 273.15 # convert K to C

        if varChosen.shape[2] != 46:
            
            varChosen = np.ones((ensAvg.shape)) * np.nan

        if index_help == 0:
            
            climBin = np.zeros((varChosen.shape[0], varChosen.shape[1], varChosen.shape[2], 365)) # (lon, lat, lead, 365 days)
            
            climBinDays = np.zeros((varChosen.shape[0], varChosen.shape[1], varChosen.shape[2], 365))
            
            lon = xr.open_dataset(fil)[variable].coords['lon'].values # grab lon and lat arrays
            lat = xr.open_dataset(fil)[variable].coords['lat'].values
            
            if grab_ensembles:
                
                all_ensembles = []
                all_ensembles.append(fil[fil.find('_m')+2:fil.find('_m')+4]) # saving ensemble members for attrs

        if dateStr == dateStrPrevious: # if dates match, means you are on next ensemble member
            
            x += 1 # to compute ensemble mean
            ensAvg = (ensAvg * (x - 1) + varChosen) / x
            
            if grab_ensembles:
                
                all_ensembles.append(fil[fil.find('_m')+2:fil.find('_m')+4])
                
        else:
            
            if index_help != 0: # if dates don't match, but make sure we are past the first file and ensAvg has data
                
                if not np.all(ensAvg == 0):
                    
                    climBin[:,:,:,doyPrevious - 1] = climBin[:,:,:,doyPrevious - 1] + ensAvg # doyPrevious - 1 bc 0-based index
                    climBinDays[:,:,:,doyPrevious - 1] = climBinDays[:,:,:,doyPrevious - 1] + 1
                    grab_ensembles = False
                    
            ensAvg = varChosen
            x = 1

        dateStrPrevious = dateStr
        doyPrevious = doy
        index_help += 1

    climBin[:,:,:,doyPrevious - 1] = climBin[:,:,:,doyPrevious - 1] + ensAvg
    climBinDays[:,:,:,doyPrevious - 1] = climBinDays[:,:,:,doyPrevious - 1] + 1
    clim = climBin / climBinDays
    
    dates_array = pd.to_datetime(np.array([file[file.find(char_1)+12:file.find(char_2)] for file in filelist])).unique()

    data_assemble = xr.Dataset({
                         'clim': (['lon','lat','lead','time'], clim),
                         'date_range': (['date_range'], dates_array),
                        },
                         coords =
                        {'lead': (['lead'], np.arange(0,clim.shape[2],1)),
                         'time': (['time'], np.arange(1,clim.shape[3]+1,1)),
                         'lat' : (['lat'], lat),
                         'lon' : (['lon'], lon)
                        },
                        attrs = 
                        {'File Author' : author,
                         'Ensembles' : all_ensembles})
    
    if not save:
        
        return data_assemble

    if save:
        
        if len(all_ensembles) > 1:
            
            data_assemble.to_netcdf(
                f'{parent_directory}CESM2/{variable.lower()}_clim_cesm2cam6v2_{str(len(all_ensembles))}members_s2s_data.nc')
            
        if len(all_ensembles) == 1:
            
            data_assemble.to_netcdf(
                f'{parent_directory}CESM2/{variable.lower()}_clim_cesm2cam6v2_{str(all_ensembles[0])}member_s2s_data.nc')

            
def cesm2_total_ensemble(filelist):
    """
    Extract the total number of ensembles contained in the list of CESM2 hindcast files.
    Returns an integer type scalar.
    
    Args:
        filelist (list of str): List of file names and directory locations.
        
    """
    dateStrPrevious = '01jan1000' # just a random date
    index_help = 0
    char_1 = "cesm2cam6v2_"
    char_2 = "_00z_d01_d46" 
    grab_ensembles = True
    
    for fil in filelist:
        
        dateStr = fil[fil.find(char_1)+12 : fil.find(char_2)]

        if index_help == 0:
            
            if grab_ensembles:
                
                all_ensembles = []
                all_ensembles.append(fil[fil.find('_m')+2:fil.find('_m')+4])
                
        if dateStr == dateStrPrevious:
            
            if grab_ensembles:
                
                all_ensembles.append(fil[fil.find('_m')+2:fil.find('_m')+4])
                
        else:
            
            if index_help != 0:
                
                if not np.all(ensAvg == 0):
                    
                    grab_ensembles = False
                    
            ensAvg = 1

        dateStrPrevious = dateStr
        index_help += 1

        if not grab_ensembles:
            
            return int(len(all_ensembles))
        

def cesm2_hindcast_anomalies(filelist, variable, parent_directory, save=False, author=None):
    """
    Create CESM2 hindcast anomalies. Outputs array (lon, lat, lead, number of forecasts).
    Number of forecasts is equal to the length of ``filelist`` divided by ``total ensembles``.
    Translated from MATLAB (provided by Anne Sasha Glanville, NCAR).
    
    Args:
        filelist (list of str): List of file names and directory locations.
        variable (str): Name of variable (e.g., 'zg_200').
        parent_directory (str): Directory where climatology is located and where to save anomalies 
                                (e.g., '/glade/scratch/$USER/s2s/').
        save (boolean): Set to True if want to save climatology as netCDF. Defaults to False.
        author (str): Author of file. Defaults to None.
        
    """
    if save:
        
        assert isinstance(author, str), "Please set author for file saving."

    assert isinstance(parent_directory, str), "Please set parent_directory to save file to."
    
    if str(cesm2_total_ensemble(filelist)) != str(11):
        
        import warnings
        warnings.warn("Using climatology computed from 11 ensemble members!")
    
    clima = xr.open_dataset(
        f'{parent_directory}CESM2/{variable.lower()}_clim_cesm2cam6v2_11members_s2s_data.nc') # open climo
    
    climCyclical = xr.concat([clima['clim'], clima['clim'], clima['clim']], dim='time') # stack 3x's time for smoothing
    
    # smooth time with 31 days 2x's (31 day window to copy Lantao, but maybe it should be 16)
    climSmooth = climCyclical.rolling(time=31, min_periods=1, center=True).mean(skipna=True).rolling(
                                      time=31, min_periods=1, center=True).mean(skipna=True)
    
    climSmooth = climSmooth.isel(time=slice(365,365 * 2)) # choose the middle year (smoothed)
    
    climSmooth = climSmooth.transpose('lon','lat','lead','time').values # extract array for loop
    
    del climCyclical # delete previous arrays
    del clima
    
    dateStrPrevious = '01jan1000' # just a random date
    index_help = 0 
    forecastCounter = 0
    char_1 = "cesm2cam6v2_"
    char_2 = "_00z_d01_d46" 
    grab_ensembles = True
    
    for fil in filelist: # loop through list of hindcast files

        dateStr = fil[fil.find(char_1)+12 : fil.find(char_2)]
        starttime = pd.to_datetime(dateStr)
        doy = starttime.dayofyear

        if (starttime.year % 4) == 0 and starttime.month > 2:
            
            doy = doy - 1

        var = xr.open_dataset(fil)[variable].transpose('lon','lat','time').values # (lon,lat,lead); load file and grab variable
        varChosen = var
        
        if variable == 'pr' or variable == 'pr_sfc':
            
            varChosen = varChosen * 84600 # convert kg/m2/s to mm/day
            
        if variable == 'tas_2m':
            
            varChosen = varChosen - 273.15 # convert K to C

        if varChosen.shape[2] != 46:
            
            varChosen = np.ones((ensAvg.shape)) * np.nan

        if index_help == 0:
            
            dim_last = int(len(filelist)/cesm2_total_ensemble(filelist))
            
            anom = np.empty((varChosen.shape[0], varChosen.shape[1], varChosen.shape[2], dim_last))
            
            starttimeBin = np.empty((dim_last), dtype="S10") # (lon, lat, lead, num of forecasts)
            lon = xr.open_dataset(fil)[variable].coords['lon'].values # grab lon and lat arrays
            lat = xr.open_dataset(fil)[variable].coords['lat'].values
            
            if grab_ensembles:
                
                all_ensembles = []
                all_ensembles.append(fil[fil.find('_m')+2:fil.find('_m')+4]) # saving ensemble members for attrs

        if dateStr == dateStrPrevious: # if dates match, means you are on next ensemble member
            
            x += 1  # to compute ensemble mean
            ensAvg = (ensAvg * (x - 1) + varChosen) / x
            
            if grab_ensembles:
                
                all_ensembles.append(fil[fil.find('_m')+2:fil.find('_m')+4])
                
        else:
            
            if index_help != 0: # if dates don't match, but make sure we are past the first file and ensAvg has data
                
                if not np.all(ensAvg == 0):
                    
                    forecastCounter += 1
                    anom[:,:,:,forecastCounter - 1] = ensAvg - np.squeeze(climSmooth[:,:,:,doyPrevious - 1])
                    starttimeBin[forecastCounter - 1] = str(starttimePrevious)
                    grab_ensembles = False
                    
            ensAvg = varChosen
            x = 1

        dateStrPrevious = dateStr
        starttimePrevious = starttime
        doyPrevious = doy
        index_help += 1

    forecastCounter += 1
    anom[:,:,:,forecastCounter - 1] = ensAvg - np.squeeze(climSmooth[:,:,:,doyPrevious - 1])
    starttimeBin[forecastCounter - 1] = starttimePrevious
    
    dates_array = pd.to_datetime(np.array([file[file.find(char_1)+12:file.find(char_2)] for file in filelist])).unique()
    
    data_assemble = xr.Dataset({
                         'anom': (['lon','lat','lead','time'], anom),
                         'fcst': (['time'], starttimeBin),
                         'date_range': (['date_range'], dates_array),
                        },
                         coords =
                        {'lead': (['lead'], np.arange(0,anom.shape[2],1)),
                         'time': (['time'], np.arange(1,anom.shape[3]+1,1)),
                         'lat' : (['lat'], lat),
                         'lon' : (['lon'], lon)
                        },
                        attrs = 
                        {'File Author' : author,
                         'Ensembles' : all_ensembles})
    
    if not save:
        
        return data_assemble
    
    if save:
        
        if len(all_ensembles) > 1:
            
            data_assemble.to_netcdf(
                f'{parent_directory}CESM2/{variable.lower()}_anom_cesm2cam6v2_{str(len(all_ensembles))}members_s2s_data.nc')
            
        if len(all_ensembles) == 1:
            
            data_assemble.to_netcdf(
                f'{parent_directory}CESM2/{variable.lower()}_anom_cesm2cam6v2_{str(all_ensembles[0])}member_s2s_data.nc')

            
def gpcp_hindcast_anomalies(parent_directory, variable='precip',
                            start_range='1999-01-01', end_range='2019-12-31',
                            save=False, author=None,):
    """
    Create GPCP Version 2.3 Combined Precipitation Data Set anomalies.
    
    Args:
        parent_directory (str): Directory where climatology is located and where to save anomalies 
                                (e.g., '/glade/scratch/$USER/s2s/').
        variable (str): Name of variable. Defaults to precip for GPCP.
        start_range (str): Start range of analysis. Defaults to '1999-01-01'.
        end_range (str): End range of analysis. Defaults to '2019-12-31'.
        save (boolean): Set to True if want to save climatology as netCDF. Defaults to False.
        author (str): Author of file. Defaults to None.
        
    """
    if save:
        
        assert isinstance(author, str), "Please set author for file saving."

    assert isinstance(parent_directory, str), "Please set parent_directory to save file to."
    
    # -- open and smooth obs climo

    clima = xr.open_dataset(f'{parent_directory}CESM2_OBS/{variable.lower()}_clim_gpcp_data.nc')

    climCyclical = xr.concat([clima['clim'], clima['clim'], clima['clim']], dim='time')

    climSmooth = climCyclical.rolling(time=31, min_periods=1, center=True).mean(skipna=True).rolling(
                                      time=31, min_periods=1, center=True).mean(skipna=True)

    climSmooth = climSmooth.isel(time=slice(365,365 * 2))
    climSmooth = climSmooth.transpose('time','lat','lon')

    # -- reduce mem usage

    del climCyclical
    del clima

    # -- add lead time to climo

    climCyclicalObs = xr.concat([climSmooth, climSmooth, climSmooth], dim='time')

    climFinal = np.zeros((climSmooth.shape[0],46,climSmooth.shape[1],climSmooth.shape[2]))

    for i in range(365):

        climFinal[i,:,:,:] = climCyclicalObs[365+i:365+i+46,:,:]

    # -- create time arrays for subsequent indexing

    d_mon = pd.date_range(start=start_range, end=end_range, freq='W-MON')
    d_dly = pd.date_range(start=start_range, end=end_range, freq='D')

    for num, (yr, mo, day) in enumerate(zip(d_dly.strftime("%Y"),d_dly.strftime("%m"),d_dly.strftime("%d"))):

        if calendar.isleap(int(yr)):

            if mo == '02' and day == '29':

                d_dly = d_dly.drop(f'{yr}-02-29')

    for num, (yr, mo, day) in enumerate(zip(d_mon.strftime("%Y"),d_mon.strftime("%m"),d_mon.strftime("%d"))):

        if calendar.isleap(int(yr)):

            if mo == '02' and day == '29':

                d_mon = d_mon.drop(f'{yr}-02-29')

    # -- create daily obs for final anom computation
    
    filelist2 = gpcp_filelist(parent_directory='/glade/work/molina/GPCP',
                              start=start_range, 
                              end=str(int((end_range)[:4])+1)+'-12-31')

    varObs = np.zeros((len(filelist2), 180, 360))

    for num, file in enumerate(filelist2):

        ds = xr.open_dataset(file)
        ds = ds[variable].isel(time=0)
        ds = ds.where(ds>=0.,0.)  # valid range: [0.,100.]
        ds = ds.where(ds<=100.,100.)
        
        if num == 0:
            
            lats = ds.latitude.values
            lons = ds.longitude.values

        varObs[num,:,:] = ds.values

    # -- add lead time to daily obs

    varFinal = np.zeros((int(len(d_mon)), 46, 180, 360))

    for num, i in enumerate(d_mon):

        varFinal[num,:,:,:] = varObs[int(
            np.argwhere(d_dly==np.datetime64(i))[0]):int(np.argwhere(d_dly==np.datetime64(i))[0])+46,:,:]

    # -- compute obs anomalies

    anom = np.zeros((int(len(d_mon)), 46, 180, 360))

    for num, i in enumerate(d_mon):

        doy_indx = i.dayofyear - 1

        if calendar.isleap(int(i.year)) and i.month > 2:

            doy_indx = doy_indx - 1

        anom[num,:,:,:] = varFinal[num,:,:,:] - climFinal[doy_indx,:,:,:]

    # --
    
    data_assemble = xr.Dataset({
                         'anom': (['time','lead','lat','lon'], anom),
                         'date_range': (['date_range'], d_mon),
                        },
                         coords =
                        {'lead': (['lead'], np.arange(0,anom.shape[1],1)),
                         'time': (['time'], np.arange(1,anom.shape[0]+1,1)),
                         'lat' : (['lat'], lats),
                         'lon' : (['lon'], lons)
                        },
                        attrs = 
                        {'File Author' : author})
    
    if not save:
        
        return data_assemble
    
    if save:
            
        data_assemble.to_netcdf(f'{parent_directory}CESM2_OBS/{variable.lower()}_anom_gpcp_data.nc')
        
        
def era5_variable_regrid(obs_directory, variable, start_range='1999-01-01', end_range='2020-12-31'):
    """
    Regridding of ERA5 temperatures.
    
    Args:
        obs_directory (str): Directory where files are located.
        start_range (str): Start of hindcasts. Defaults to '1999-01-01'.
        end_range (str): End of hindcasts. Defaults to '2020-12-31'.
        
    """
    d_daily = pd.date_range(start=start_range, end=end_range, freq='D')
    d_daily = d_daily[~((d_daily.day==29)&(d_daily.month==2))]

    if variable == "mx2t":
        var = "MX2T"; filename = "e5.oper.fc.sfc.minmax.128_201_mx2t.ll025sc"; constant=1
        
    if variable == "mn2t":
        var = "MN2T"; filename = "e5.oper.fc.sfc.minmax.128_202_mn2t.ll025sc"; constant=1
        
    if variable == "sstk":
        var = "SSTK"; filename = "e5.oper.an.sfc.128_034_sstk.ll025sc"; constant=1
    
    if variable == "ua200" or variable == "ua850":
        var = "U"; filename = "e5.oper.an.pl.128_131_u.ll025uv"; constant=1
        
    if variable == "va200" or variable == "va850":
        var = "V"; filename = "e5.oper.an.pl.128_132_v.ll025uv"; constant=1
        
    if variable == "z500":
        var = "Z"; filename = "e5.oper.an.pl.128_129_z.ll025sc"; constant=1/9.80665
        
    if variable == "ttrc":
        var = "TTRC"; filename = "e5.oper.fc.sfc.accumu.128_209_ttrc.ll025sc"; constant=1/86400
        
    if variable == "ttr":
        var = "TTR"; filename = "e5.oper.fc.sfc.accumu.128_179_ttr.ll025sc"; constant=1/86400
        
    if variable == "tp":
        var = "TP"; filename = "e5.oper.fc.sfc.accumu.128_142_tp.ll025sc"; constant=1000 # convert m to mm
        
    for num, t in enumerate(d_daily):
        
        ds_ = xr.open_dataset(f"{obs_directory}/era5_{variable}/{filename}.{t.strftime('%Y%m%d')}.nc")
        
        if variable != 'sstk':
            ds_ = regrid_mask(ds_ * constant, var)
            
        if variable == 'sstk':
            ds_ = regrid_mask(ds_ - 273.15, var)
            
        ds_.to_dataset(
            name=var).to_netcdf(
            f"{obs_directory}/era5_{variable}_regrid/{filename}.{t.strftime('%Y%m%d')}.nc")


def era5_variable_climatology(obs_directory, save_directory, variable, start='1999-01-01', end='2020-12-31', 
                              save=False, author=None):
    """
    Create ERA5 variable hindcast climatology. Outputs array (365, lat, lon).
    
    Args:
        obs_directory (str): Directory where files are located.
        save_directory (str): Directory where to save files.
        start (str): Start of hindcasts. Defaults to '1999-01-01'.
        end (str): End of hindcasts. Defaults to '2020-12-31'.
        save (boolean): Set to True if want to save climatology as netCDF. Defaults to False.
        author (str): Author of file. Defaults to None.
                                
    """
    if variable == "ua200" or variable == "ua850":
        var = "U"; filename = "e5.oper.an.pl.128_131_u.ll025uv"
        
    if variable == "va200" or variable == "va850":
        var = "V"; filename = "e5.oper.an.pl.128_132_v.ll025uv"
        
    if variable == "z500":
        var = "Z"; filename = "e5.oper.an.pl.128_129_z.ll025sc"
        
    if variable == "ttrc":
        var = "TTRC"; filename = "e5.oper.fc.sfc.accumu.128_209_ttrc.ll025sc"
        
    if variable == "ttr":
        var = "TTR"; filename = "e5.oper.fc.sfc.accumu.128_179_ttr.ll025sc"
        
    if variable == "tp":
        var = "TP"; filename = "e5.oper.fc.sfc.accumu.128_142_tp.ll025sc"
        
    if variable == "sstk":
        var = "SSTK"; filename = "e5.oper.an.sfc.128_034_sstk.ll025sc"
        
    td = pd.date_range(start=start, end=end, freq='D')
    td = td[~((td.day==29)&(td.month==2))]

    doy = 0
    yr = 0
    dates = []
    years = []

    for num, t in enumerate(td):

        ds_ = xr.open_dataset(
            f"{obs_directory}/era5_{variable}_regrid/{filename}.{t.strftime('%Y%m%d')}.nc")[var].transpose('y','x')

        dates.append(pd.Timestamp(t.strftime('%Y%m%d')))

        if num == 0:

            clim = np.zeros((td.year.unique().shape[0],365,ds_.shape[0],ds_.shape[1]))

            lats = ds_.lat[:,0].values
            lons = ds_.lon[0,:].values

        clim[yr,doy,:,:] = ds_.values

        doy += 1

        if doy == 365:

            doy = 0
            yr += 1

            years.append(int(t.strftime('%Y')))

    data_assemble = xr.Dataset({
                         'clim': (['time','lat','lon'], np.nanmean(clim, axis=0)),
                        },
                         coords =
                        {'date_range': (['date_range'], pd.to_datetime(dates)),
                         'time': (['time'], np.arange(1,365 + 1,1)),
                         'lat' : (['lat'], lats),
                         'lon' : (['lon'], lons)
                        },
                        attrs = 
                        {'File Author' : author,
                         'Years' : np.array(years)})

    if not save:

        return data_assemble

    if save:

        data_assemble.to_netcdf(f'{save_directory}/era5_{variable}_clim_data.nc')


def era5_temp_climatology(obs_directory, save_directory, start='1999-01-01', end='2020-12-31', 
                          save=False, author=None):
    """
    Create ERA5 temperature hindcast climatology. Outputs array (365, lat, lon).
    
    Args:
        obs_directory (str): Directory where files are located.
        save_directory (str): Directory where to save files.
        start (str): Start of hindcasts. Defaults to '1999-01-01'.
        end (str): End of hindcasts. Defaults to '2020-12-31'.
        save (boolean): Set to True if want to save climatology as netCDF. Defaults to False.
        author (str): Author of file. Defaults to None.
                                
    """
    td = pd.date_range(start=start, end=end, freq='D')
    td = td[~((td.day==29)&(td.month==2))]

    doy = 0
    yr = 0
    dates = []
    years = []

    for num, t in enumerate(td):

        tmax = xr.open_dataset(
            f"{obs_directory}era5_mx2t_regrid/e5.oper.fc.sfc.minmax.128_201_mx2t.ll025sc.{t.strftime('%Y%m%d')}.nc"
        )['MX2T'] - 273.15 # convert K to C

        tmin = xr.open_dataset(
            f"{obs_directory}era5_mn2t_regrid/e5.oper.fc.sfc.minmax.128_202_mn2t.ll025sc.{t.strftime('%Y%m%d')}.nc"
        )['MN2T'] - 273.15 # convert K to C

        avg_temp = (tmin + tmax) / 2

        dates.append(pd.Timestamp(t.strftime('%Y%m%d')))

        if num == 0:

            clim = np.zeros((td.year.unique().shape[0],365,avg_temp.shape[0],avg_temp.shape[1]))

            lats = tmax.lat[:,0].values
            lons = tmax.lon[0,:].values

        clim[yr,doy,:,:] = avg_temp

        doy += 1

        if doy == 365:

            doy = 0
            yr += 1

            years.append(int(t.strftime('%Y')))

    data_assemble = xr.Dataset({
                         'clim': (['time','lat','lon'], np.nanmean(clim, axis=0)),
                        },
                         coords =
                        {'date_range': (['date_range'], pd.to_datetime(dates)),
                         'time': (['time'], np.arange(1,365 + 1,1)),
                         'lat' : (['lat'], lats),
                         'lon' : (['lon'], lons)
                        },
                        attrs = 
                        {'File Author' : author,
                         'Years' : np.array(years)})

    if not save:

        return data_assemble

    if save:

        data_assemble.to_netcdf(f'{save_directory}era5_temp_clim_data.nc')


def era5_variable_anomalies(obs_directory, save_directory, variable, 
                            start_range='1999-01-01', end_range='2019-12-31', save=False, author=None):
    """
    Create ERA5 temperature hindcast anomalies.
    
    Args:
        obs_directory (str): Directory where files are located.
        save_directory (str): Directory where to save files.
        start (str): Start of hindcasts. Defaults to '1999-01-01'.
        end (str): End of hindcasts. Defaults to '2020-12-31'.
        save (boolean): Set to True if want to save climatology as netCDF. Defaults to False.
        author (str): Author of file. Defaults to None.
        
    """
    if variable == "ua200" or variable == "ua850":
        var = "U"; filename = "e5.oper.an.pl.128_131_u.ll025uv"
        
    if variable == "va200" or variable == "va850":
        var = "V"; filename = "e5.oper.an.pl.128_132_v.ll025uv"
        
    if variable == "z500":
        var = "Z"; filename = "e5.oper.an.pl.128_129_z.ll025sc"
        
    if variable == "ttrc":
        var = "TTRC"; filename = "e5.oper.fc.sfc.accumu.128_209_ttrc.ll025sc"
        
    if variable == "ttr":
        var = "TTR"; filename = "e5.oper.fc.sfc.accumu.128_179_ttr.ll025sc"
        
    if variable == "tp":
        var = "TP"; filename = "e5.oper.fc.sfc.accumu.128_142_tp.ll025sc"
        
    if variable == "sstk":
        var = "SSTK"; filename = "e5.oper.an.sfc.128_034_sstk.ll025sc"
        
    # -- open and smooth obs climo

    clima = xr.open_dataset(f'{save_directory}/era5_{variable}_clim_data.nc')

    climCyclical = xr.concat([clima['clim'], clima['clim'], clima['clim']], dim='time')

    climSmooth = climCyclical.rolling(time=31, min_periods=1, center=True).mean(skipna=True).rolling(
                                      time=31, min_periods=1, center=True).mean(skipna=True)

    climSmooth = climSmooth.isel(time=slice(365,365 * 2))
    climSmooth = climSmooth.transpose('time','lat','lon')

    # -- reduce mem usage

    del climCyclical
    del clima

    # -- add lead time to climo

    climCyclicalObs = xr.concat([climSmooth, climSmooth, climSmooth], dim='time')

    climFinal = np.zeros((climSmooth.shape[0],46,climSmooth.shape[1],climSmooth.shape[2]))

    for i in range(365):

        climFinal[i,:,:,:] = climCyclicalObs[365+i:365+i+46,:,:]

    # -- create time arrays for subsequent indexing

    d_mon = pd.date_range(start=start_range, end=end_range, freq='W-MON')
    d_dly = pd.date_range(start=start_range, end=end_range, freq='D')
    
    d_mon = d_mon[~((d_mon.day==29)&(d_mon.month==2))]
    d_dly = d_dly[~((d_dly.day==29)&(d_dly.month==2))]

    # -- create daily obs for final anom computation
    
    d_daily = pd.date_range(start=start_range, end=str(int((end_range)[:4])+1)+'-12-31', freq='D')
    d_daily = d_daily[~((d_daily.day==29)&(d_daily.month==2))]
    
    for num, t in enumerate(d_daily):

        ds_ = xr.open_dataset(
            f"{obs_directory}/era5_{variable}_regrid/{filename}.{t.strftime('%Y%m%d')}.nc")[var].transpose('y','x')

        if num == 0:

            varObs = np.zeros((len(d_daily),climSmooth.shape[1],climSmooth.shape[2]))

            lats = ds_.lat[:,0].values
            lons = ds_.lon[0,:].values

        varObs[num,:,:] = ds_.values

    # -- add lead time to daily obs

    varFinal = np.zeros((int(len(d_mon)),46,climSmooth.shape[1],climSmooth.shape[2]))

    for num, i in enumerate(d_mon):

        varFinal[num,:,:,:] = varObs[int(
            np.argwhere(d_dly==np.datetime64(i))[0]):int(np.argwhere(d_dly==np.datetime64(i))[0])+46,:,:]

    # -- compute obs anomalies

    anom = np.zeros((int(len(d_mon)),46,climSmooth.shape[1],climSmooth.shape[2]))

    for num, i in enumerate(d_mon):

        doy_indx = i.dayofyear - 1

        if calendar.isleap(int(i.year)) and i.month > 2:

            doy_indx = doy_indx - 1

        anom[num,:,:,:] = varFinal[num,:,:,:] - climFinal[doy_indx,:,:,:]

    # --
    
    data_assemble = xr.Dataset({
                         'anom': (['time','lead','lat','lon'], anom),
                         'date_range': (['date_range'], d_mon),
                        },
                         coords =
                        {'lead': (['lead'], np.arange(0,anom.shape[1],1)),
                         'time': (['time'], np.arange(1,anom.shape[0]+1,1)),
                         'lat' : (['lat'], lats),
                         'lon' : (['lon'], lons)
                        },
                        attrs = 
                        {'File Author' : author})
    
    if not save:
        
        return data_assemble
    
    if save:
            
        data_assemble.to_netcdf(f'{save_directory}/era5_{variable}_anom_data.nc')
        
        
def era5_temp_anomalies(obs_directory, save_directory, start_range='1999-01-01', end_range='2019-12-31',
                        save=False, author=None):
    """
    Create ERA5 temperature hindcast anomalies.
    
    Args:
        obs_directory (str): Directory where files are located.
        save_directory (str): Directory where to save files.
        start (str): Start of hindcasts. Defaults to '1999-01-01'.
        end (str): End of hindcasts. Defaults to '2020-12-31'.
        save (boolean): Set to True if want to save climatology as netCDF. Defaults to False.
        author (str): Author of file. Defaults to None.
        
    """
    # -- open and smooth obs climo

    clima = xr.open_dataset(f'{save_directory}era5_temp_clim_data.nc')

    climCyclical = xr.concat([clima['clim'], clima['clim'], clima['clim']], dim='time')

    climSmooth = climCyclical.rolling(time=31, min_periods=1, center=True).mean(skipna=True).rolling(
                                      time=31, min_periods=1, center=True).mean(skipna=True)

    climSmooth = climSmooth.isel(time=slice(365,365 * 2))
    climSmooth = climSmooth.transpose('time','lat','lon')

    # -- reduce mem usage

    del climCyclical
    del clima

    # -- add lead time to climo

    climCyclicalObs = xr.concat([climSmooth, climSmooth, climSmooth], dim='time')

    climFinal = np.zeros((climSmooth.shape[0],46,climSmooth.shape[1],climSmooth.shape[2]))

    for i in range(365):

        climFinal[i,:,:,:] = climCyclicalObs[365+i:365+i+46,:,:]

    # -- create time arrays for subsequent indexing

    d_mon = pd.date_range(start=start_range, end=end_range, freq='W-MON')
    d_dly = pd.date_range(start=start_range, end=end_range, freq='D')
    
    d_mon = d_mon[~((d_mon.day==29)&(d_mon.month==2))]
    d_dly = d_dly[~((d_dly.day==29)&(d_dly.month==2))]

    # -- create daily obs for final anom computation
    
    d_daily = pd.date_range(start=start_range, end=str(int((end_range)[:4])+1)+'-12-31', freq='D')
    d_daily = d_daily[~((d_daily.day==29)&(d_daily.month==2))]
    
    for num, t in enumerate(d_daily):

        tmax = xr.open_dataset(
            f"{obs_directory}era5_mx2t_regrid/e5.oper.fc.sfc.minmax.128_201_mx2t.ll025sc.{t.strftime('%Y%m%d')}.nc"
        )['MX2T'] - 273.15 # convert K to C
        
        tmin = xr.open_dataset(
            f"{obs_directory}era5_mn2t_regrid/e5.oper.fc.sfc.minmax.128_202_mn2t.ll025sc.{t.strftime('%Y%m%d')}.nc"
        )['MN2T'] - 273.15 # convert K to C
        
        avg_temp = (tmin + tmax) / 2

        if num == 0:

            varObs = np.zeros((len(d_daily),climSmooth.shape[1],climSmooth.shape[2]))

            lats = tmin.lat[:,0].values
            lons = tmin.lon[0,:].values

        varObs[num,:,:] = avg_temp

    # -- add lead time to daily obs

    varFinal = np.zeros((int(len(d_mon)),46,climSmooth.shape[1],climSmooth.shape[2]))

    for num, i in enumerate(d_mon):

        varFinal[num,:,:,:] = varObs[int(
            np.argwhere(d_dly==np.datetime64(i))[0]):int(np.argwhere(d_dly==np.datetime64(i))[0])+46,:,:]

    # -- compute obs anomalies

    anom = np.zeros((int(len(d_mon)),46,climSmooth.shape[1],climSmooth.shape[2]))

    for num, i in enumerate(d_mon):

        doy_indx = i.dayofyear - 1

        if calendar.isleap(int(i.year)) and i.month > 2:

            doy_indx = doy_indx - 1

        anom[num,:,:,:] = varFinal[num,:,:,:] - climFinal[doy_indx,:,:,:]

    # --
    
    data_assemble = xr.Dataset({
                         'anom': (['time','lead','lat','lon'], anom),
                         'date_range': (['date_range'], d_mon),
                        },
                         coords =
                        {'lead': (['lead'], np.arange(0,anom.shape[1],1)),
                         'time': (['time'], np.arange(1,anom.shape[0]+1,1)),
                         'lat' : (['lat'], lats),
                         'lon' : (['lon'], lons)
                        },
                        attrs = 
                        {'File Author' : author})
    
    if not save:
        
        return data_assemble
    
    if save:
            
        data_assemble.to_netcdf(f'{save_directory}era5_temp_anom_data.nc')

        
def noaa_cpc_regrid(obs_directory, variable, start_range='1999-01-01', end_range='2020-12-31'):
    """
    Regridding of NOAA CPC data.
    
    Args:
        obs_directory (str): Directory where files are located.
        variable (str): Name of variable.
        start_range (str): Start of hindcasts. Defaults to '1999-01-01'.
        end_range (str): End of hindcasts. Defaults to '2020-12-31'.
        
    """
    d_yearly = pd.date_range(start=start_range, end=end_range, freq='AS')
    
    for num, t in enumerate(d_yearly):
        
        file = xr.open_dataset(f"{obs_directory}/{variable}.{t.strftime('%Y')}.nc")
        file = regrid_mask(file, variable)
        file.to_dataset(name=variable).to_netcdf(f"{obs_directory}/{variable}_regrid.{t.strftime('%Y')}.nc")


def noaa_cpc_filelist(parent_directory, variable, start='1999-01-01', end='2020-12-31', freq='AS'):
    """
    Create list of yearly NOAA CPC files.
    https://www.ncei.noaa.gov/data/global-precipitation-climatology-project-gpcp-daily/access/
    
    Args:
        parent_directory (str): Directory where files are located (e.g., '/glade/scratch/$USER/s2s/').
        start (str): Start of hindcasts. Defaults to '1999-01-01' for CESM2.
        end (str): End of hindcasts. Defaults to '2019-12-31' for CESM2.
        freq (str): Frequency of hindcast starts. Defaults to 'AS' for yearly.
        
    """
    d1 = pd.date_range(start=start, end=end, freq=freq)
    
    matches = []
    
    for num, yr in enumerate(zip(d1.strftime("%Y"))):
        
        for root, dirnames, filenames in os.walk(f'{parent_directory}/'):

            for filename in fnmatch.filter(filenames, f'{variable}_regrid.{yr[0]}.nc'):

                thefile = os.path.join(root, filename)

                if os.access(thefile, os.R_OK):

                    matches.append(thefile)

                if not os.access(thefile, os.R_OK):

                    matches.append(np.nan)
                
    return matches


def ncpc_precip_climatology(filelist, variable='precip', save=False, author=None, parent_directory=None):
    """
    Create NOAA CPC precipitation climatology.
    
    Args:
        filelist (list of str): List of file names and directory locations.
        variable (str): Name of variable. Defaults to precip.
        save (boolean): Set to True if want to save climatology as netCDF. Defaults to False.
        author (str): Author of file. Defaults to None.
        parent_directory (str): Directory where files are located (e.g., '/glade/scratch/$USER/s2s/').
                                Defaults to None.
                                
    """
    if save:
        
        assert isinstance(author, str), "Please set author for file saving."
        assert isinstance(parent_directory, str), "Please set parent_directory to save file to."

    clim = np.zeros((int(len(filelist)), 365, 181, 360))

    years = []

    for num, file in enumerate(filelist):

        ds = xr.open_dataset(file)
        dates = pd.to_datetime(ds.time.values)
        
        if len(dates) == 365:
            
            clim[num,:,:,:] = ds[variable].values
            
        if len(dates) > 365:
        
            dates = dates[dates!=f"{dates[0].strftime('%Y')}-02-29"]
            ds = ds.sel(time=dates)
            
            clim[num,:,:,:] = ds[variable].values
        
        if num == 0:

            lats = ds.lat.values
            lons = ds.lon.values

        years.append(int(dates[0].strftime('%Y')))

    data_assemble = xr.Dataset({
                         'clim': (['time','y','x'], np.nanmean(clim, axis=0)),
                        },
                         coords =
                        {'time': (['time'], np.arange(1,365 + 1,1)),
                         'lat' : (['y','x'], lats),
                         'lon' : (['y','x'], lons)
                        },
                        attrs = 
                        {'File Author' : author,
                         'Years' : np.array(years)})

    if not save:

        return data_assemble

    if save:

        data_assemble.to_netcdf(f'{parent_directory}CESM2_OBS/{variable.lower()}_clim_ncpc_data.nc')


def ncpc_temp_climatology(filelist_max, filelist_min, save=False, author=None, parent_directory=None):
    """
    Create NOAA CPC temperature climatology.
    
    Args:
        filelist_max (list of str): List of file names and directory locations for maximum temps.
        filelist_min (list of str): List of file names and directory locations for minimum temps.
        save (boolean): Set to True if want to save climatology as netCDF. Defaults to False.
        author (str): Author of file. Defaults to None.
        parent_directory (str): Directory where files are located (e.g., '/glade/scratch/$USER/s2s/').
                                Defaults to None.
                                
    """
    if save:
        
        assert isinstance(author, str), "Please set author for file saving."
        assert isinstance(parent_directory, str), "Please set parent_directory to save file to."
        assert len(filelist_max) == len(filelist_min), "Tmax and Tmin lists do not match in length."

    clim = np.zeros((int(len(filelist_max)), 365, 181, 360))

    years = []

    for num, (filemax, filemin) in enumerate(zip(filelist_max, filelist_min)):

        ds_max = xr.open_dataset(filemax)
        ds_min = xr.open_dataset(filemin)
        dates = pd.to_datetime(ds_max.time.values)
        
        if len(dates) == 365:
            
            clim[num,:,:,:] = (ds_min['tmin'].values + ds_max['tmax'].values) / 2
            
        if len(dates) > 365:
        
            dates = dates[dates!=f"{dates[0].strftime('%Y')}-02-29"]
            
            ds_min = ds_min.sel(time=dates)
            ds_max = ds_max.sel(time=dates)
            
            clim[num,:,:,:] = (ds_min['tmin'].values + ds_max['tmax'].values) / 2
        
        if num == 0:

            lats = ds_max.lat.values
            lons = ds_max.lon.values

        years.append(int(dates[0].strftime('%Y')))

    data_assemble = xr.Dataset({
                         'clim': (['time','y','x'], np.nanmean(clim, axis=0)),
                        },
                         coords =
                        {'time': (['time'], np.arange(1,365 + 1,1)),
                         'lat' : (['y','x'], lats),
                         'lon' : (['y','x'], lons)
                        },
                        attrs = 
                        {'File Author' : author,
                         'Years' : np.array(years)})

    if not save:

        return data_assemble

    if save:

        data_assemble.to_netcdf(f'{parent_directory}CESM2_OBS/temp_clim_ncpc_data.nc')


def ncpc_temp_anomalies(obs_directory, save_directory, start_range='1999-01-01', end_range='2019-12-31',
                        save=False, author=None):
    """
    Create NOAA CPC temperature anomalies.
    
    Args:
        obs_directory (str): Directory where files are located.
        save_directory (str): Directory where to save files.
        start (str): Start of hindcasts. Defaults to '1999-01-01'.
        end (str): End of hindcasts. Defaults to '2020-12-31'.
        save (boolean): Set to True if want to save climatology as netCDF. Defaults to False.
        author (str): Author of file. Defaults to None.
        
    """
    # -- open and smooth obs climo

    clima = xr.open_dataset(f'{save_directory}CESM2_OBS/temp_clim_ncpc_data.nc')

    climCyclical = xr.concat([clima['clim'], clima['clim'], clima['clim']], dim='time')

    climSmooth = climCyclical.rolling(time=31, min_periods=1, center=True).mean(skipna=True).rolling(
                                      time=31, min_periods=1, center=True).mean(skipna=True)

    climSmooth = climSmooth.isel(time=slice(365, 365 * 2))
    climSmooth = climSmooth.transpose('time','y','x')

    # -- reduce mem usage

    del climCyclical
    del clima

    # -- add lead time to climo

    climCyclicalObs = xr.concat([climSmooth, climSmooth, climSmooth], dim='time')

    climFinal = np.zeros((climSmooth.shape[0], 46, climSmooth.shape[1], climSmooth.shape[2]))

    for i in range(365):

        climFinal[i,:,:,:] = climCyclicalObs[365+i:365+i+46,:,:]

    # -- create time arrays for subsequent indexing

    d_mon = pd.date_range(start=start_range, end=end_range, freq='W-MON')
    d_dly = pd.date_range(start=start_range, end=end_range, freq='D')
    
    d_mon = d_mon[~((d_mon.day==29)&(d_mon.month==2))]
    d_dly = d_dly[~((d_dly.day==29)&(d_dly.month==2))]

    # -- create daily obs for final anom computation
    
    d_yearly = pd.date_range(start=start_range, end=str(int((end_range)[:4]) + 1)+'-12-31', freq='AS')
    years_for_files = np.unique(d_yearly.year)
    
    varObs = np.zeros((int(len(years_for_files)) * 365, 181, 360))
    
    for num, yr_for_file in enumerate(years_for_files):
        
        assert (((num + 1) * 365)) - (num * 365) == 365

        tmax = xr.open_dataset(f"{obs_directory}/tmax_regrid.{yr_for_file}.nc")['tmax']
        tmin = xr.open_dataset(f"{obs_directory}/tmin_regrid.{yr_for_file}.nc")['tmin']
        
        if not calendar.isleap(yr_for_file):
            
            avgtemp = (tmin.values + tmax.values) / 2
            
            varObs[num * 365:((num + 1) * 365),:,:] = avgtemp
            
        if calendar.isleap(yr_for_file):
        
            first_half = np.arange(0,366,1)[:31+29-1]
            second_half = np.arange(0,366,1)[31+29:]
            
            avgtemp = (tmin.values + tmax.values) / 2
            
            varObs[num * 365:((num + 1) * 365),:,:] = avgtemp[np.hstack([first_half,second_half])]

        if num == 0:

            lats = tmax.lat.values
            lons = tmax.lon.values

    # -- add lead time to daily obs

    varFinal = np.zeros((int(len(d_mon)),46,climSmooth.shape[1],climSmooth.shape[2]))

    for num, i in enumerate(d_mon):

        varFinal[num,:,:,:] = varObs[int(
            np.argwhere(d_dly==np.datetime64(i))[0]):int(np.argwhere(d_dly==np.datetime64(i))[0])+46,:,:]

    # -- compute obs anomalies

    anom = np.zeros((int(len(d_mon)),46,climSmooth.shape[1],climSmooth.shape[2]))

    for num, i in enumerate(d_mon):

        doy_indx = i.dayofyear - 1

        if calendar.isleap(int(i.year)) and i.month > 2:

            doy_indx = doy_indx - 1

        anom[num,:,:,:] = varFinal[num,:,:,:] - climFinal[doy_indx,:,:,:]

    # --
    
    data_assemble = xr.Dataset({
                         'anom': (['time','lead','y','x'], anom),
                         'date_range': (['date_range'], d_mon),
                        },
                         coords =
                        {'lead': (['lead'], np.arange(0,anom.shape[1],1)),
                         'time': (['time'], np.arange(1,anom.shape[0]+1,1)),
                         'lat' : (['y','x'], lats),
                         'lon' : (['y','x'], lons)
                        },
                        attrs = 
                        {'File Author' : author})
    
    if not save:
        
        return data_assemble
    
    if save:
            
        data_assemble.to_netcdf(f'{save_directory}CESM2_OBS/temp_anom_ncpc_data.nc')


def ncpc_precip_anomalies(obs_directory, save_directory, start_range='1999-01-01', end_range='2019-12-31',
                          save=False, author=None):
    """
    Create NOAA CPC precipitation anomalies.
    
    Args:
        obs_directory (str): Directory where files are located.
        save_directory (str): Directory where to save files.
        start (str): Start of hindcasts. Defaults to '1999-01-01'.
        end (str): End of hindcasts. Defaults to '2020-12-31'.
        save (boolean): Set to True if want to save climatology as netCDF. Defaults to False.
        author (str): Author of file. Defaults to None.
        
    """
    # -- open and smooth obs climo

    clima = xr.open_dataset(f'{save_directory}CESM2_OBS/precip_clim_ncpc_data.nc')

    climCyclical = xr.concat([clima['clim'], clima['clim'], clima['clim']], dim='time')

    climSmooth = climCyclical.rolling(time=31, min_periods=1, center=True).mean(skipna=True).rolling(
                                      time=31, min_periods=1, center=True).mean(skipna=True)

    climSmooth = climSmooth.isel(time=slice(365, 365 * 2))
    climSmooth = climSmooth.transpose('time','y','x')

    # -- reduce mem usage

    del climCyclical
    del clima

    # -- add lead time to climo

    climCyclicalObs = xr.concat([climSmooth, climSmooth, climSmooth], dim='time')

    climFinal = np.zeros((climSmooth.shape[0], 46, climSmooth.shape[1], climSmooth.shape[2]))

    for i in range(365):

        climFinal[i,:,:,:] = climCyclicalObs[365+i:365+i+46,:,:]

    # -- create time arrays for subsequent indexing

    d_mon = pd.date_range(start=start_range, end=end_range, freq='W-MON')
    d_dly = pd.date_range(start=start_range, end=end_range, freq='D')
    
    d_mon = d_mon[~((d_mon.day==29)&(d_mon.month==2))]
    d_dly = d_dly[~((d_dly.day==29)&(d_dly.month==2))]

    # -- create daily obs for final anom computation
    
    d_yearly = pd.date_range(start=start_range, end=str(int((end_range)[:4]) + 1)+'-12-31', freq='AS')
    years_for_files = np.unique(d_yearly.year)
    
    varObs = np.zeros((int(len(years_for_files)) * 365, 181, 360))
    
    for num, yr_for_file in enumerate(years_for_files):
        
        assert (((num + 1) * 365)) - (num * 365) == 365

        precip = xr.open_dataset(f"{obs_directory}/precip_regrid.{yr_for_file}.nc")['precip']
        
        if not calendar.isleap(yr_for_file):
            
            varObs[num * 365:((num + 1) * 365),:,:] = precip.values
            
        if calendar.isleap(yr_for_file):
        
            first_half = np.arange(0,366,1)[:31+29-1]
            second_half = np.arange(0,366,1)[31+29:]
            
            varObs[num * 365:((num + 1) * 365),:,:] = precip.values[np.hstack([first_half,second_half])]

        if num == 0:

            lats = precip.lat.values
            lons = precip.lon.values

    # -- add lead time to daily obs

    varFinal = np.zeros((int(len(d_mon)),46,climSmooth.shape[1],climSmooth.shape[2]))

    for num, i in enumerate(d_mon):

        varFinal[num,:,:,:] = varObs[int(
            np.argwhere(d_dly==np.datetime64(i))[0]):int(np.argwhere(d_dly==np.datetime64(i))[0])+46,:,:]

    # -- compute obs anomalies

    anom = np.zeros((int(len(d_mon)),46,climSmooth.shape[1],climSmooth.shape[2]))

    for num, i in enumerate(d_mon):

        doy_indx = i.dayofyear - 1

        if calendar.isleap(int(i.year)) and i.month > 2:

            doy_indx = doy_indx - 1

        anom[num,:,:,:] = varFinal[num,:,:,:] - climFinal[doy_indx,:,:,:]

    # --
    
    data_assemble = xr.Dataset({
                         'anom': (['time','lead','y','x'], anom),
                         'date_range': (['date_range'], d_mon),
                        },
                         coords =
                        {'lead': (['lead'], np.arange(0,anom.shape[1],1)),
                         'time': (['time'], np.arange(1,anom.shape[0]+1,1)),
                         'lat' : (['y','x'], lats),
                         'lon' : (['y','x'], lons)
                        },
                        attrs = 
                        {'File Author' : author})
    
    if not save:
        
        return data_assemble
    
    if save:
            
        data_assemble.to_netcdf(f'{save_directory}CESM2_OBS/precip_anom_ncpc_data.nc')
        
        