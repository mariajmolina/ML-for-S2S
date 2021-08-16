import os
import numpy as np
import xarray as xr
import pandas as pd
import torch
from pylab import *
import itertools


def pacific_lon(ds, lon_name='lon'):
    """
    Help converting Pacific longitudes to 180 or 360 coordinates.
    
    Args:
        ds: xarray dataset.
        lon_name (str): name of longitude coordinate in ds.
        
    """
    return xr.where(ds[lon_name] > 180,
                    ds[lon_name] - 360,
                    ds[lon_name])


def compute_lat_weights(ds):
    """
    Computation of weights for latitude/longitude grid mean.
    Weights are cosine of the latitude.
    
    Args:
        ds: xarray dataset.
        
    """
    weights = np.cos(np.deg2rad(ds.lat))
    _, weights = xr.broadcast(ds, weights)
    weights = weights.isel(time=0)
    
    return weights


def matlab_to_python_time(ds):
    """
    Conversion of matlab time to python time (human understandable).
    
    Args:
        ds: xarray dataset.
        
    """
    datenums = ds.coords['time'].values
    timestamps = pd.to_datetime(datenums-719529, unit='D')
    
    return timestamps


def select_region(ds, region=None):
    """
    Select region based on latitude for S2S Challenge or other.
    
    Args:
        ds: xarray dataset.
        region (str): string representation of region. Defaults to None.
        
    """
    dict_region = {'NH': (30, 90),
                   'TP': (-29, 29),
                   'SH': (-60, -30),
                   'NOANT': (-60, 90),
                   'FIRST': (61, 90),
                   'SECOND': (31, 60),
                   'THIRD': (1, 30),
                   'FOURTH': (-29, 0),
                   'FIFTH': (-60, -30),
                  }
    
    if region:
        return ds.sel(lat=slice(dict_region[region][0], dict_region[region][1]))
    
    if not region:
        return ds


def select_biweekly(ds, week='34'):
    """
    Select week 3-4 (``34``) or 5-6 (``56``).
    
    Args:
        ds: xarray dataset.
        week (str): string representation of bi-weekly period. Defaults to `34`.
        
    """
    if week == '34': 
        return ds.isel(lead=slice(15, 29)).mean('lead', skipna=True)
        
    if week == '56':
        return ds.isel(lead=slice(29, 43)).mean('lead', skipna=True)


def create_landmask(path, region=None):
    """
    Create land mask based on observations (CPC).
    Outputs a binary mask. Values of 1. represent land. Values of 0. represent ocean.
    
    Args:
        path (str): directory path to masked data.
        region (str): string representation of region. Defaults to None.
        
    """
    ds = xr.open_dataset(f'{path}/pr_anom_CPC_Mon_data.nc').isel(lead=0,time=0)
    ds = select_region(ds, region=region)
    ds = ds['anom'].values
    ds = np.where(~np.isnan(ds), 1., np.nan)
    
    return ds


def process_ensemble_members(path, week='34', region=None):
    """
    Open and process the 11 ensemble members for training.
    
    Args:
        path (str): directory path to data.
        week (str): string representation of bi-weekly period. Defaults to `34`.
        region (str): string representation of region. Defaults to None.
    
    """
    ds00 = xr.open_dataset(f'{path}/pr_sfc_anom_cesm2cam6v2_00member_s2s_data.nc').isel(time=slice(0,887),date_range=slice(0,887))
    ds01 = xr.open_dataset(f'{path}/pr_sfc_anom_cesm2cam6v2_01member_s2s_data.nc').isel(time=slice(0,887),date_range=slice(0,887))
    ds02 = xr.open_dataset(f'{path}/pr_sfc_anom_cesm2cam6v2_02member_s2s_data.nc').isel(time=slice(0,887),date_range=slice(0,887))
    ds03 = xr.open_dataset(f'{path}/pr_sfc_anom_cesm2cam6v2_03member_s2s_data.nc').isel(time=slice(0,887),date_range=slice(0,887))
    ds04 = xr.open_dataset(f'{path}/pr_sfc_anom_cesm2cam6v2_04member_s2s_data.nc').isel(time=slice(0,887),date_range=slice(0,887))
    ds05 = xr.open_dataset(f'{path}/pr_sfc_anom_cesm2cam6v2_05member_s2s_data.nc').isel(time=slice(0,887),date_range=slice(0,887))
    ds06 = xr.open_dataset(f'{path}/pr_sfc_anom_cesm2cam6v2_06member_s2s_data.nc').isel(time=slice(0,887),date_range=slice(0,887))
    ds07 = xr.open_dataset(f'{path}/pr_sfc_anom_cesm2cam6v2_07member_s2s_data.nc').isel(time=slice(0,887),date_range=slice(0,887))
    ds08 = xr.open_dataset(f'{path}/pr_sfc_anom_cesm2cam6v2_08member_s2s_data.nc').isel(time=slice(0,887),date_range=slice(0,887))
    ds09 = xr.open_dataset(f'{path}/pr_sfc_anom_cesm2cam6v2_09member_s2s_data.nc').isel(time=slice(0,887),date_range=slice(0,887))
    ds10 = xr.open_dataset(f'{path}/pr_sfc_anom_cesm2cam6v2_10member_s2s_data.nc').isel(time=slice(0,887),date_range=slice(0,887))
    
    ds00 = select_biweekly(ds00, week=week)
    ds01 = select_biweekly(ds01, week=week)
    ds02 = select_biweekly(ds02, week=week)
    ds03 = select_biweekly(ds03, week=week)
    ds04 = select_biweekly(ds04, week=week)
    ds05 = select_biweekly(ds05, week=week)
    ds06 = select_biweekly(ds06, week=week)
    ds07 = select_biweekly(ds07, week=week)
    ds08 = select_biweekly(ds08, week=week)
    ds09 = select_biweekly(ds09, week=week)
    ds10 = select_biweekly(ds10, week=week)
    
    ds00 = select_region(ds00, region=region)
    ds01 = select_region(ds01, region=region)
    ds02 = select_region(ds02, region=region)
    ds03 = select_region(ds03, region=region)
    ds04 = select_region(ds04, region=region)
    ds05 = select_region(ds05, region=region)
    ds06 = select_region(ds06, region=region)
    ds07 = select_region(ds07, region=region)
    ds08 = select_region(ds08, region=region)
    ds09 = select_region(ds09, region=region)
    ds10 = select_region(ds10, region=region)
        
    dstotal = xr.concat([
                ds00['anom'], ds01['anom'], ds02['anom'], ds03['anom'], 
                ds04['anom'], ds05['anom'], ds06['anom'], ds07['anom'], 
                ds08['anom'], ds09['anom'], ds10['anom']], dim='time')
    
    return dstotal.to_dataset().transpose('time','lat','lon')


def process_cpc_testdata(path, week='34', region=None):
    """
    Open and process the CPC data for 11 ensemble members for training.
    
    Args:
        path (str): directory path to data.
        week (str): string representation of bi-weekly period. Defaults to `34`.
        region (str): string representation of region. Defaults to None.
        
    """
    ds00 = xr.open_dataset(f'{path}/pr_anom_CPC_Mon_data.nc').isel(time=slice(0,887))
    ds00 = select_biweekly(ds00, week=week)
    ds00 = select_region(ds00, region=region)
    
    dstotal = xr.concat([
                ds00['anom'], ds00['anom'], ds00['anom'], ds00['anom'], 
                ds00['anom'], ds00['anom'], ds00['anom'], ds00['anom'], 
                ds00['anom'], ds00['anom'], ds00['anom']], dim='time')
    
    return dstotal.to_dataset().transpose('time','lat','lon')


def process_single_member(path, member, week='34', region=None):
    """
    Open and process single ensemble member for training.
    
    Args:
        path (str): directory path to data.
        week (str): string representation of bi-weekly period. Defaults to `34`.
        region (str): string representation of region. Defaults to None.
        
    """
    ds00 = xr.open_dataset(f'{path}/pr_sfc_anom_cesm2cam6v2_{member}member_s2s_data.nc').isel(time=slice(0,887),date_range=slice(0,887))
    ds00 = select_biweekly(ds00, week=week)
    ds00 = select_region(ds00, region=region)
    dstotal = ds00['anom']
    
    return dstotal.to_dataset().transpose('time','lat','lon')


def process_single_cpc(path, week='34', region=None):
    """
    Open and process the CPC data for single ensemble member for training.
    
    Args:
        path (str): directory path to data.
        week (str): string representation of bi-weekly period. Defaults to `34`.
        region (str): string representation of region. Defaults to None.
        
    """
    ds00 = xr.open_dataset(f'{path}/pr_anom_CPC_Mon_data.nc').isel(time=slice(0,887))
    ds00 = select_biweekly(ds00, week=week)
    ds00 = select_region(ds00, region=region)
    dstotal = ds00['anom']
    
    return dstotal.to_dataset().transpose('time','lat','lon')


def minmax_compute(ds):
    """
    Min max computation.
    
    Args:
        ds: ndarray
        
    """
    return (ds - np.nanmin(ds, axis=0)) / (np.nanmax(ds, axis=0) - np.nanmin(ds, axis=0))


def dual_norm_minmax(fct, obs):
    """"
    Dual dataset min max normalization.
    
    Args:
        fct (ndarray): forecast data
        obs (ndarray): observation data
        
    """
    fmin = np.min(fct, axis=0)
    omin = np.min(obs, axis=0)
    fmax = np.max(fct, axis=0)
    omax = np.max(obs, axis=0)
    
    maxval = np.max(np.vstack([abs(fmax), abs(omax)]), axis=0)
    minval = np.max(np.vstack([abs(fmin), abs(omin)]), axis=0)
    
    fct = (fct - (-minval)) / (maxval - (-minval))
    obs = (obs - (-minval)) / (maxval - (-minval))
    
    return fct, obs, -minval, maxval


def preset_minmax_compute(ds, MIN, MAX):
    """
    Min max computation.
    
    Args:
        ds: ndarray.
        MIN (ndarray): minimum data grid.
        MAX (ndarray): maximum data grid.
        
    """
    return (ds - MIN) / (MAX - MIN)


def mask_and_flatten_data(ds, land_mask):
    """
    Preprocessing of CESM2 data for deep learning model.
    
    Args:
        ds: xarray dataset.
        land_mask (ndarray): land mask.
        
    """ 
    ds_ = ds['anom']
    ds_ = xr.where(np.isnan(land_mask), np.nan, ds_.transpose('time','lat','lon'))
    ds_ = ds_.stack(dim_0=['lat','lon']).reset_index('dim_0').drop(['lat','lon'])
    ds_ = ds_.where(np.isfinite(ds_), drop=True)
    
    return ds_.values.astype(np.float32)


def remove_any_nans(fct, obs, return_indx=True):
    """
    Final round of nan removal.
    
    Args:
        fct (ndarray): forecast data
        obs (ndarray): observation data
        return_indx (boolean): whether to return indices for later 
                               data parsing (e.g., validation, testing).
                               Defaults to True.
        
    """
    if return_indx:
        index_return = np.arange(obs.shape[0])
    
    indx_remove = np.unique(np.argwhere(np.isnan(obs))[:,0])
    fct = np.delete(fct, indx_remove, axis=0)
    obs = np.delete(obs, indx_remove, axis=0)
    
    if return_indx:
        index_return= np.delete(index_return, indx_remove, axis=0)
        
        return fct, obs, index_return
    
    if not return_indx:
        
        return fct, obs
    
    
def random_test_split(array, seed=0, train=0.8, val=0.1):
    """
    Generate indices for train, validation, evaluation randomly.
    Index split based on percentages.
    
    Args:
        array (ndarray): indices from ``remove_any_nans``
        seed (int): integer for set random seed
        train (float): training percentage of data
        val (float): validation percentage of data
        
    Note: Remaining percent of data goes to evaluation (testing).
    
    """
    assert train + val < 1.0, "decrease train and val percentages so their sum is below 1.0"
    
    np.random.seed(seed)
    rand_ = np.random.permutation(array)

    train_ = rand_[:int(rand_.shape[0] * train)]
    valid_ = rand_[int(rand_.shape[0] * train):int(rand_.shape[0] * (train + val))]
    evalu_ = rand_[int(rand_.shape[0] * (train + val)):]

    return train_, valid_, evalu_


def year_test_split(dates, indx, yrs_valid, yrs_evalu):
    """
    Generate indices for train, validation, evaluation based on selected years.
    
    Args:
        dates (ndarray): array of dates from ``matlab_to_python_time``
        indx (ndarray): indices from ``remove_any_nans``
        yrs_valid (list): list of years (int) for validation
        yrs_evalu (list): list of years (int) for evaluation
        
    Note: Remaining years go to training.
    
    """
    assert isinstance(yrs_valid, list), "yrs_valid and yrs_evalu should be lists"
    assert isinstance(yrs_evalu, list), "yrs_valid and yrs_evalu should be lists"
    
    newindx = np.arange(indx.shape[0]) # generate new indices for indx array
    
    train_ = newindx[np.isin(dates[indx].year, yrs_valid + yrs_evalu, assume_unique=True, invert=True)]
    valid_ = newindx[np.isin(dates[indx].year, yrs_valid, assume_unique=True, invert=False)]
    evalu_ = newindx[np.isin(dates[indx].year, yrs_evalu, assume_unique=True, invert=False)]
    
    return train_, valid_, evalu_


def split_into_sets(array, tindx, vindx, eindx):
    """
    Split data into train, validation, and evaluation set with indices.
    
    Args:
        array (ndarray): processed without nans
        tindx (array): train data indices
        vindx (array): validation data indices
        eindx (array): evaluation data indices
    
    """
    array_t = array[tindx, :]
    array_v = array[vindx, :]
    array_e = array[eindx, :]
    
    return array_t, array_v, array_e


def inverse_minmax(dl_output, MIN, MAX):
    """
    Inverse of min max.
    
    Args:
        dl_output (ndarray): trained model prediction.
        MIN (ndarray): minimum data grid.
        MAX (ndarray): maximum data grid.
        
    """
    return (dl_output * (MAX - MIN)) + MIN

        
def reconstruct_grid(mask, ds_dl):
    """
    Reconstruction of 2d grid. Run inverse_minmax first.
    
    Args:
        mask (ndarray): land mask used.
        ds_dl (ndarray): trained model prediction.
        
    """
    landmask = np.argwhere(np.isnan(mask))
    empty    = np.zeros((ds_dl.shape[0], mask.shape[0], mask.shape[1]))
    
    counter = 0
    
    for i, j in itertools.product(list(range(mask.shape[0])),list(range(mask.shape[1]))):
        
        if np.argwhere(np.logical_and(np.isin(landmask[:,0], i), np.isin(landmask[:,1], j))).shape[0] > 0:
            empty[:, i, j] = np.nan
        
        else:
            empty[:, i, j] = ds_dl[:, counter]
            counter += 1
            
    return empty


def save_decoded_image(img, name=None):
    """
    Saving of decoded image per BATCH.
    
    Args:
        img (ndarray): input, output, or label.
        name (str): casename
    
    """
    number_of_subplots = int(img.shape[0])
    img = np.squeeze(img)
    
    plt.figure(figsize=(6,10))
    
    for i, v in enumerate(range(number_of_subplots)):
        ax = subplot(number_of_subplots, 2, v + 1)
        ax.pcolormesh(img[i,:,:], vmin=-10, vmax=10, cmap='BrBG')
        ax.tick_params(bottom=False, labelbottom=False, left=False, labelleft=False)

    if name:
        plt.tight_layout()
        plt.savefig(name, bbox_inches='tight', dpi=200)
        plt.close()
    
    if not name:
        return plt.show()


def get_device():
    """
    Grab GPU.
    
    """
    if torch.cuda.is_available():
        device = 'cuda:0'
        
    else:
        device = 'cpu'
        
    return device


def make_directory(path, folder='Saved_Images'):
    """
    Directory for saving progress images, models, or csvs.
    
    Args:
        path (str): directory for image folder.
        folder (str): name of folder. Defaults to ``Saved_Images``.
        
    """
    if not os.path.exists(path+folder):
        
        os.makedirs(path+folder)
