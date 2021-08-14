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
        longitude (array): Longitude values.
    """
    return xr.where(
                ds[lon_name] > 180,
                ds[lon_name] - 360,
                ds[lon_name])


def compute_lat_weights(ds):
    
    """Computation of weights for latitude/longitude grid mean"""
    
    weights = np.cos(np.deg2rad(ds.lat))    # make weights as cosine of the latitude and broadcast
    _, weights = xr.broadcast(ds, weights)
    weights = weights.isel(time=0)          # remove the time dimension from weights
    
    return weights


def matlab_to_python_time(ds):
    
    """Conversion of matlab time to python time (human understandable)"""
    
    datenums = ds.coords['time'].values
    timestamps = pd.to_datetime(datenums-719529, unit='D')
    
    return timestamps


def select_region(ds, region=None):
    
    """Select region based on latitude for S2S Challenge"""
    
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
    
    """Select week 3-4 (``34``) or 5-6 (``56``)"""
    
    if week=='34':
        
        return ds.isel(lead=slice(15, 29)).mean('lead', skipna=True)
        
    if week=='56':
        
        return ds.isel(lead=slice(29, 43)).mean('lead', skipna=True)


def create_landmask(path, region=None):
    
    """
    Create land mask based on observations (CPC).
    Outputs a binary mask. 
    Values of 1. represent land. Values of 0. represent ocean.
    """
    
    ds = xr.open_dataset(f'{path}/pr_anom_CPC_Mon_data.nc').isel(lead=0,time=0)
    ds = select_region(ds, region=region)
    ds = ds['anom'].values
    ds = np.where(~np.isnan(ds), 1., np.nan)
    
    return ds


def process_ensemble_members(path, week='34', region=None):
    
    """Open and process the 11 ensemble members for training"""
    
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
    
    """Open and process the 11 ensemble members for training"""
    
    ds00 = xr.open_dataset(f'{path}/pr_anom_CPC_Mon_data.nc').isel(time=slice(0,887))
    ds00 = select_biweekly(ds00, week=week)
    ds00 = select_region(ds00, region=region)
    
    dstotal = xr.concat([
                ds00['anom'], ds00['anom'], ds00['anom'], ds00['anom'], 
                ds00['anom'], ds00['anom'], ds00['anom'], ds00['anom'], 
                ds00['anom'], ds00['anom'], ds00['anom']], dim='time')
    
    return dstotal.to_dataset().transpose('time','lat','lon')


def minmax_compute(ds):
    
    """Min max computation"""
    
    return (ds - np.nanmin(ds, axis=0)) / (np.nanmax(ds, axis=0) - np.nanmin(ds, axis=0))


def min_max_val_pos(ds):
    
    """Min and max values for positive anomalies"""
    
    tmp_pos = ds.where(ds > 0, np.nan)
    
    return np.nanmin(tmp_pos, axis=0), np.nanmax(tmp_pos, axis=0)


def min_max_val_neg(ds):
    
    """Min and max values for negative anomalies"""
    
    tmp_neg = ds.where(ds < 0, np.nan)
    
    return np.nanmin(tmp_neg, axis=0), np.nanmax(tmp_neg, axis=0)


def min_max_anom(ds):
    
    """Application of min max individually to + and - anomalies individually"""
    
    # min-max for negative anomalies
    tmp_neg = ds.where(ds < 0, np.nan)
    ds = ds.where(ds >= 0, -minmax_compute(tmp_neg))
    
    # min-max for positive anomalies
    tmp_pos = ds.where(ds > 0, np.nan)
    ds = ds.where(ds <= 0,  minmax_compute(tmp_pos))
    
    return ds


def preprocess_cesm(ds, land_mask):
    
    """Preprocessing of CESM2 data for deep learning model"""
     
    ds_ = ds['anom']
    
    ds_ = xr.where(np.isnan(land_mask), np.nan, ds_.transpose('time','lat','lon'))
    ds_ = ds_.stack(dim_0=['lat','lon']).reset_index('dim_0').drop(['lat','lon'])
    ds_ = ds_.where(np.isfinite(ds_), drop=True)
    
    return ds_.values.astype(np.float32)


def preprocess_cpclabel(ds, land_mask):
    
    """Preprocessing of CPC data for deep learning model"""
        
    ds_ = ds['anom']
    
    ds_ = xr.where(np.isnan(land_mask), np.nan, ds_.transpose('time','lat','lon'))
    ds_ = ds_.stack(dim_0=['lat','lon']).reset_index('dim_0').drop(['lat','lon'])
    ds_ = ds_.where(np.isfinite(ds_), drop=True)
    
    return ds_.values.astype(np.float32)


def remove_any_nans(fct, obs):
    
    """Final round of nan removal"""
    
    indx_remove = np.unique(np.argwhere(np.isnan(obs))[:,0])
    fct = np.delete(fct, indx_remove, axis=0)
    obs = np.delete(obs, indx_remove, axis=0)
    
    return fct, obs


def dual_norm_minmax(fct, obs):
    
    """"Dual dataset min max normalization"""
    
    fmin = np.min(fct, axis=0)
    omin = np.min(obs, axis=0)
    fmax = np.max(fct, axis=0)
    omax = np.max(obs, axis=0)
    
    maxval = np.max(np.vstack([abs(fmax), abs(omax)]), axis=0)
    minval = np.max(np.vstack([abs(fmin), abs(omin)]), axis=0)
    
    fct = (fct - (-minval)) / (maxval - (-minval))
    obs = (obs - (-minval)) / (maxval - (-minval))
    
    return fct, obs, -minval, maxval


def inverse_minmax(dl_output, MIN, MAX):
    
    """Inverse of min max"""
    
    return (dl_output * (MAX - MIN)) + MIN

        
def reconstruct_grid(mask, ds_dl):
    
    """
    Reconstruction of 2d grid. Run inverse_minmax first.
    """
    
    landmask = np.argwhere(np.isnan(mask))
    
    empty = np.zeros((ds_dl.shape[0], mask.shape[0], mask.shape[1]))
    
    counter = 0
    
    for i, j in itertools.product(list(range(mask.shape[0])),list(range(mask.shape[1]))):
        
        if np.argwhere(np.logical_and(np.isin(landmask[:,0], i), np.isin(landmask[:,1], j))).shape[0] > 0:
            
            empty[:, i, j] = np.nan
        
        else:
            
            empty[:, i, j] = ds_dl[:, counter]
            counter += 1
            
    return empty


def save_decoded_image(img, name=None):
    
    """Saving of decoded image per BATCH"""
    
    number_of_subplots = int(img.shape[0])
    
    img = np.squeeze(img)
    
    plt.figure(figsize=(6,20))
    
    for i, v in enumerate(range(number_of_subplots)):
        ax = subplot(number_of_subplots, 2, v + 1)
        ax.pcolormesh(img[i,:,:], vmin=-10, vmax=10, cmap='BrBG')
        ax.tick_params(bottom=False, labelbottom=False, left=False, labelleft=False)

    if name:
        plt.savefig(name, bbox_inches='tight', dpi=200)
    
    plt.close()


def get_device():
    
    """Grab GPU"""
    
    if torch.cuda.is_available():
        device = 'cuda:0'
        
    else:
        device = 'cpu'
        
    return device


def make_img_dir(fullpath):
    
    """Directory for saving progress images"""
    
    image_dir = 'Saved_Images'
    
    if not os.path.exists(fullpath+image_dir):
        
        os.makedirs(fullpath+image_dir)
        

def make_model_dir(fullpath):
    
    """Directory for saving progress images"""
    
    image_dir = 'Saved_Models'
    
    if not os.path.exists(fullpath+image_dir):
        
        os.makedirs(fullpath+image_dir)
        

def make_csv_dir(fullpath):
    
    """Directory for saving progress images"""
    
    image_dir = 'Saved_CSV'
    
    if not os.path.exists(fullpath+image_dir):
        
        os.makedirs(fullpath+image_dir)