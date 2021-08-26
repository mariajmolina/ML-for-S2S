import os
import itertools
import numpy as np
import pandas as pd
import xarray as xr
from pylab import *
import torch
from torch.autograd import Variable

"""

Functions for use with Pytorch.

This module contains utility functions for the S2S machine learning project.

Author: Maria J. Molina, NCAR

"""

def pacific_lon(array):
    """
    Help converting pacific 360 longitudes to 180.
    
    Args:
        array: longitude array.
        
    Returns:
        converted longitudes array.
        
    """
    return xr.where(array > 180,
                    array - 360,
                    array)


def compute_lat_weights(ds):
    """
    Computation of weights for latitude/longitude grid mean.
    Weights are cosine of the latitude.
    
    Args:
        ds: xarray dataset.
    
    Returns:
        latitude weights.
        
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
        
    Returns:
        pandas datetime timestamps.
        
    """
    datenums = ds.coords['time'].values
    timestamps = pd.to_datetime(datenums-719529, unit='D')
    
    return timestamps


def select_region(ds, region=None):
    """
    Select region based on latitude or other.
    
    Args:
        ds: xarray dataset.
        region (str): string representation of region. defaults to None.
        
    Returns:
        subset array based on selected region.
        
    """
    dict_region = {'NH':     (30, 90),
                   'TP':     (-29, 29),
                   'SH':     (-60, -30),
                   'NOANT':  (-60, 90),
                   'FIRST':  (61, 90),
                   'SECOND': (31, 60),
                   'THIRD':  (1, 30),
                   'FOURTH': (-29, 0),
                   'FIFTH':  (-60, -30),
                  }
    
    if region:
        
        return ds.sel(lat=slice(dict_region[region][0], dict_region[region][1]))
    
    if not region:
        
        return ds


def select_biweekly(ds, day_init=15, day_end=21):
    """
    Time selection for weekly subsets.
    
    Args:
        ds: xarray dataset.
        day_init (int): first lead day selection. Defaults to 15.
        day_end (int): last lead day selection. Defaults to 21.
        
    Returns:
        xarray dataset subset based on time selection.
    
    ::Lead time indices for reference::

    Week 1:  1,  2,  3,  4,  5,  6,  7
    Week 2:  8,  9, 10, 11, 12, 13, 14
    Week 3: 15, 16, 17, 18, 19, 20, 21
    Week 4: 22, 23, 24, 25, 26, 27, 28
    Week 5: 29, 30, 31, 32, 33, 34, 35
    Week 6: 36, 37, 38, 39, 40, 41, 42
        
    """
    return ds.isel(lead=slice(day_init, day_end + 1)).mean('lead', skipna=True)


def select_daily(ds, day_init=15, day_end=21):
    """
    Select lead time days.
    
    Args:
        ds: xarray dataset.
        day_init (int): first lead day selection. Defaults to 15.
        day_end (int): last lead day selection. Defaults to 21.
        
    Returns:
        xarray dataset subset based on time selection.
    
    ::Lead time indices for reference::

    Week 1:  1,  2,  3,  4,  5,  6,  7
    Week 2:  8,  9, 10, 11, 12, 13, 14
    Week 3: 15, 16, 17, 18, 19, 20, 21
    Week 4: 22, 23, 24, 25, 26, 27, 28
    Week 5: 29, 30, 31, 32, 33, 34, 35
    Week 6: 36, 37, 38, 39, 40, 41, 42
        
    """
    return ds.isel(lead=slice(day_init, day_end + 1))


def create_landmask(path, variable='pr', region=None):
    """
    Create binary land mask. Values of 1 represent land and 0 represent ocean.
    
    Args:
        path (str): directory path to masked data.
        variable **
        region (str): string representation of region. defaults to None.
        
    Returns:
        mask (array), latitudes (array), and longitudes (array).
    
    ::Regions for reference::
    
    - Northern Hemisphere (NH):      30N to 90N
    - Tropics (TP):                  29S to 29N
    - Southern Hemisphere (SH):      60S to 30S
    - No Antarctica (NOANT):         60S to 90N
    - Five global sections (FIRST):  61N to 90N
    - Five global sections (SECOND): 31N to 60N
    - Five global sections (THIRD):   1N to 30N
    - Five global sections (FOURTH): 29S to  0N
    - Five global sections (FIFTH):  60S to 30S
    
    """
    ds = xr.open_dataset(f'{path}/{variable}_anom_CPC_Mon_data.nc').isel(lead=0,time=0)
    ds = select_region(ds, region=region)
    
    # grab lat lons for evaluations later
    lt = ds.coords['lat'].values
    ln = ds.coords['lon'].values
    
    # grab mask
    ds = ds['anom'].values
    ds = np.where(~np.isnan(ds), 1., np.nan)
    
    return ds, lt, ln


def process_ensemble_members(path, variable='pr_sfc', leadopt='weekly', day_init=15, day_end=21, region=None):
    """
    Open and process the 11 ensemble members for training.
    
    Args:
        path (str): directory path to data.
        variable (str): variable choice. defaults to pr_sfc.
        leadopt (str): choice for time selection. defaults to ``weekly``. Other: ``daily``.
        day_init (int): first lead day selection. defaults to 15.
        day_end (int): last lead day selection. defaults to 21.
        region (str): string representation of region. defaults to None.
    
    """
    ds00 = xr.open_dataset(f'{path}/{variable}_anom_cesm2cam6v2_00member_s2s_data.nc').isel(time=slice(0,887),date_range=slice(0,887)) # edit these once cesm2 files > 2015
    ds01 = xr.open_dataset(f'{path}/{variable}_anom_cesm2cam6v2_01member_s2s_data.nc').isel(time=slice(0,887),date_range=slice(0,887))
    ds02 = xr.open_dataset(f'{path}/{variable}_anom_cesm2cam6v2_02member_s2s_data.nc').isel(time=slice(0,887),date_range=slice(0,887))
    ds03 = xr.open_dataset(f'{path}/{variable}_anom_cesm2cam6v2_03member_s2s_data.nc').isel(time=slice(0,887),date_range=slice(0,887))
    ds04 = xr.open_dataset(f'{path}/{variable}_anom_cesm2cam6v2_04member_s2s_data.nc').isel(time=slice(0,887),date_range=slice(0,887))
    ds05 = xr.open_dataset(f'{path}/{variable}_anom_cesm2cam6v2_05member_s2s_data.nc').isel(time=slice(0,887),date_range=slice(0,887))
    ds06 = xr.open_dataset(f'{path}/{variable}_anom_cesm2cam6v2_06member_s2s_data.nc').isel(time=slice(0,887),date_range=slice(0,887))
    ds07 = xr.open_dataset(f'{path}/{variable}_anom_cesm2cam6v2_07member_s2s_data.nc').isel(time=slice(0,887),date_range=slice(0,887))
    ds08 = xr.open_dataset(f'{path}/{variable}_anom_cesm2cam6v2_08member_s2s_data.nc').isel(time=slice(0,887),date_range=slice(0,887))
    ds09 = xr.open_dataset(f'{path}/{variable}_anom_cesm2cam6v2_09member_s2s_data.nc').isel(time=slice(0,887),date_range=slice(0,887))
    ds10 = xr.open_dataset(f'{path}/{variable}_anom_cesm2cam6v2_10member_s2s_data.nc').isel(time=slice(0,887),date_range=slice(0,887))
    
    if leadopt == 'weekly':
        
        ds00 = select_biweekly(ds00, day_init, day_end)
        ds01 = select_biweekly(ds01, day_init, day_end)
        ds02 = select_biweekly(ds02, day_init, day_end)
        ds03 = select_biweekly(ds03, day_init, day_end)
        ds04 = select_biweekly(ds04, day_init, day_end)
        ds05 = select_biweekly(ds05, day_init, day_end)
        ds06 = select_biweekly(ds06, day_init, day_end)
        ds07 = select_biweekly(ds07, day_init, day_end)
        ds08 = select_biweekly(ds08, day_init, day_end)
        ds09 = select_biweekly(ds09, day_init, day_end)
        ds10 = select_biweekly(ds10, day_init, day_end)
    
    if leadopt == 'daily':
        
        ds00 = select_daily(ds00, day_init, day_end)
        ds01 = select_daily(ds01, day_init, day_end)
        ds02 = select_daily(ds02, day_init, day_end)
        ds03 = select_daily(ds03, day_init, day_end)
        ds04 = select_daily(ds04, day_init, day_end)
        ds05 = select_daily(ds05, day_init, day_end)
        ds06 = select_daily(ds06, day_init, day_end)
        ds07 = select_daily(ds07, day_init, day_end)
        ds08 = select_daily(ds08, day_init, day_end)
        ds09 = select_daily(ds09, day_init, day_end)
        ds10 = select_daily(ds10, day_init, day_end)
    
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
    
    if leadopt == 'weekly':
        
        return dstotal.to_dataset().transpose('time','lat','lon')
    
    if leadopt == 'daily':
        
        return dstotal.to_dataset().transpose('time','lead','lat','lon')


def process_cpc_testdata(path, variable='pr', leadopt='weekly', day_init=15, day_end=21, region=None):
    """
    Open and process the CPC data for 11 ensemble members for training.
    
    Args:
        path (str): directory path to data.
        variable (str): variable for analysis.
        leadopt (str): choice for time selection. Defaults to ``weekly``. Other: ``daily``.
        day_init (int): first lead day selection. Defaults to 15.
        day_end (int): last lead day selection. Defaults to 21.
        region (str): string representation of region. Defaults to None.
        
    """
    ds00 = xr.open_dataset(f'{path}/{variable}_anom_CPC_Mon_data.nc').isel(time=slice(0,887)) # edit this once cesm2 files > 2015
    
    if leadopt == 'weekly':
        
        ds00 = select_biweekly(ds00, day_init, day_end)
        
    if leadopt == 'daily':
        
        ds00 = select_daily(ds00, day_init, day_end)
    
    ds00 = select_region(ds00, region=region)
    
    dstotal = xr.concat([
                ds00['anom'], ds00['anom'], ds00['anom'], ds00['anom'], 
                ds00['anom'], ds00['anom'], ds00['anom'], ds00['anom'], 
                ds00['anom'], ds00['anom'], ds00['anom']], dim='time')
    
    if leadopt == 'weekly':
        
        return dstotal.to_dataset().transpose('time','lat','lon')
    
    if leadopt == 'daily':
        
        return dstotal.to_dataset().transpose('time','lead','lat','lon')


def process_single_member(path, member, variable='pr_sfc', leadopt='weekly', day_init=15, day_end=21, region=None):
    """
    Open and process single ensemble member for training.
    
    Args:
        path (str): directory path to data.
        member (str): ensemble member.
        variable (str): variable for analysis. defaults to pr_sfc.
        leadopt (str): choice for time selection. defaults to weekly. other is daily.
        day_init (int): first lead day selection. defaults to 15.
        day_end (int): last lead day selection. defaults to 21.
        region (str): string representation of region. defaults to None.
        
    """
    ds00 = xr.open_dataset(
        f'{path}/{variable}_anom_cesm2cam6v2_{member}member_s2s_data.nc').isel(time=slice(0,887),date_range=slice(0,887)) 
    # edit this once cesm2 files > 2015

    if leadopt == 'weekly':
        
        ds00 = select_biweekly(ds00, day_init, day_end)
        
    if leadopt == 'daily':
        
        ds00 = select_daily(ds00, day_init, day_end)
    
    ds00 = select_region(ds00, region=region)
    dstotal = ds00['anom']
    
    if leadopt == 'weekly':
        
        return dstotal.to_dataset().transpose('time','lat','lon')

    if leadopt == 'daily':
        
        return dstotal.to_dataset().transpose('time','lead','lat','lon')


def process_single_cpc(path, variable='pr', leadopt='weekly', day_init=15, day_end=21, region=None):
    """
    Open and process the CPC data for single ensemble member for training.
    
    Args:
        path (str): directory path to data.
        variable (str): variable for analysis. defaults to pr.
        leadopt (str): choice for time selection. defaults to weekly. other is daily.
        day_init (int): first lead day selection. defaults to 15.
        day_end (int): last lead day selection. defaults to 21.
        region (str): string representation of region. defaults to None.
        
    """
    ds00 = xr.open_dataset(f'{path}/{variable}_anom_CPC_Mon_data.nc').isel(time=slice(0,887)) # edit this once cesm2 files > 2015
    
    if leadopt == 'weekly':
        
        ds00 = select_biweekly(ds00, day_init, day_end)
        
    if leadopt == 'daily':
        
        ds00 = select_daily(ds00, day_init, day_end)
        
    ds00 = select_region(ds00, region=region)
    dstotal = ds00['anom']
    
    if leadopt == 'weekly':
        
        return dstotal.to_dataset().transpose('time','lat','lon')

    if leadopt == 'daily':
        
        return dstotal.to_dataset().transpose('time','lead','lat','lon')
    

def mask_and_flatten_data(ds, land_mask):
    """
    Preprocessing of model data for deep learning model.
    
    Args:
        ds: xarray dataset.
        land_mask (ndarray): land mask.
        
    """ 
    ds_ = ds['anom']
    
    try:
        
        ds_ = xr.where(np.isnan(land_mask), np.nan, ds_.transpose('time','lead','lat','lon'))
        
    except ValueError:
        
        ds_ = xr.where(np.isnan(land_mask), np.nan, ds_.transpose('time','lat','lon'))
        
    ds_ = ds_.stack(dim_0=['lat','lon']).reset_index('dim_0').drop(['lat','lon'])
    
    try:
        
        ds_.coords['lead']
        ds_ = ds_.stack(dim_1=['dim_0','lead']).reset_index('dim_1').drop(['dim_0','lead'])
        ds_ = ds_.rename({'dim_1':'dim_0'})
        ds_ = ds_.where(np.isfinite(ds_), drop=True)
        
    except KeyError:
        
        ds_ = ds_.where(np.isfinite(ds_), drop=True)
    
    return ds_.values.astype(np.float32)


def mask_and_flatten_latweights(ds_, land_mask):
    """
    Preprocessing of latitude weights for training loss.
    
    Args:
        ds_ (xarray array): latitude weights.
        land_mask (ndarray): land mask.
        
    """ 
    ds_ = xr.where(np.isnan(land_mask), np.nan, ds_.transpose('lat','lon'))
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
                               fefaults to True.
    
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


def random_test_split(array, seed=0, train=0.8):
    """
    Generate indices for training set and validation (or evaluation) randomly.
    Index split based on percentages.
    
    Args:
        array (ndarray): indices
        seed (int): integer for set random seed
        train (float): training percentage of data
        
    Note: remaining percent of data goes to validation/evaluation (testing).
    
    """
    assert train < 1.0, "decrease train percentage to below 1.0"
    
    np.random.seed(seed)
    rand_ = np.random.permutation(array)

    train_ = rand_[:int(rand_.shape[0] * train)]
    evalu_ = rand_[int(rand_.shape[0] * train):]

    return train_, evalu_


def year_test_split(dates, indx, yrs_subset):
    """
    Generate indices for training set and validation (or evaluation) based on selected years.
    
    Args:
        dates (ndarray): array of dates from `matlab_to_python_time`
        indx (ndarray): indices from `remove_any_nans`
        yrs_subset (list): list of years (int) to set aside for validation/evaluation
        
    Note: femaining years go to training.
    
    """
    assert isinstance(yrs_subset, list), "yrs_subset should be a list"
    
    newindx = np.arange(indx.shape[0]) # generate new indices for indx array
    
    train_ = newindx[np.isin(dates[indx].year, yrs_subset, assume_unique=True, invert=True)]
    evalu_ = newindx[np.isin(dates[indx].year, yrs_subset, assume_unique=True, invert=False)]
    
    return train_, evalu_


def split_into_sets(array, tindx, eindx):
    """
    Split data into train and validation (or evaluation) set with indices.
    
    Args:
        array (ndarray): processed without nans
        tindx (array): train data indices
        eindx (array): validation/evaluation data indices
    
    """
    array_t = array[tindx, ...]
    array_e = array[eindx, ...]
    
    return array_t, array_e


def split_time_sets(array, dsindx, tindx, eindx):
    """
    Split time array into train and validation (or evaluation) set with indices.
    
    Args:
        array (ndarray): processed without nans
        dsindx (array): data indices post nan removal
        tindx (array): train data indices
        eindx (array): validation/evaluation data indices
    
    """
    array_t = array[dsindx][tindx]
    array_e = array[dsindx][eindx]
    
    return array_t, array_e


def reconstruct_grid(mask, ds_dl):
    """
    Reconstruction of 2d grid.
    
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
    Saving of decoded image per batch.
    limit to 16 images.
    
    Args:
        img (ndarray): input, output, or label.
        name (str): filename
    
    """
    number_of_subplots = int(img.shape[0])
    
    if number_of_subplots > 16:
        
        number_of_subplots = 16
    
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
    Grab GPU (cuda).
    
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
        folder (str): name of folder. defaults to `Saved_Images`.
        
    """
    if not os.path.exists(path+folder):
        
        os.makedirs(path+folder)

        
def RMSELoss(pred, target):
    """
    Torch implementation of RMSE loss function.
    
    Args:
        pred (torch): torch model prediction
        target (torch): torch model label
        
    """
    return torch.sqrt(torch.mean((pred - target)**2))


def corrcoef(pred, target):
    """
    Torch implementation of pearson correlation coefficient.
    
    Args:
        pred (torch): torch model prediction
        target (torch): torch model label
    
    ::References::
    # np.corrcoef in torch from @mdo
    # https://forum.numer.ai/t/custom-loss-functions-for-xgboost-using-pytorch/960
    
    """
    pred_n = pred - pred.mean()
    target_n = target - target.mean()
    pred_n = pred_n / pred_n.norm()
    target_n = target_n / target_n.norm()
    
    return (pred_n * target_n).sum()


def weighted_mse_loss(output, label, lat_weights, reduction='sum'):
    """
    Latitude weighted MSE loss function for pytorch model.
    
    Args:
        output: torch model prediction
        label: torch dataloader label
        lat_weights: numpy array of latitude weights
        reduction (str): how to reduce weighted MSE. defaults to sum. 
                         other reduction option: mean.
                         
    """
    device = get_device() # get cpu
    
    weights = Variable(torch.from_numpy(lat_weights)).to(device)
    
    pct_var = (output - label)**2
    out = pct_var * weights.expand_as(pct_var)
    out = torch.div(out, torch.sum(torch.from_numpy(lat_weights)))
    
    if reduction == 'mean':
        
        loss = out.mean()
        
    if reduction == 'sum':
        
        loss = out.sum()
    
    return loss


def weighted_rmse_loss(output, label, lat_weights):
    """
    Latitude weighted RMSE loss function for pytorch model.
    
    Args:
        output: torch model prediction
        label: torch dataloader label
        lat_weights: numpy array of latitude weights
        
    """
    device = get_device() # get cpu
    
    weights = Variable(torch.from_numpy(lat_weights)).to(device)
    
    pct_var = (output - label)**2
    out = pct_var * weights.expand_as(label)
    out = torch.div(out, torch.sum(torch.from_numpy(lat_weights)))
    loss = torch.sqrt(out.mean())
    
    return loss
