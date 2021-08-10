import os
import numpy as np
import xarray as xr
import pandas as pd
from itertools import product
import matplotlib.pyplot as plt
from pylab import *

import torch
#import torchvision
import torch.nn as nn
#import torchvision.transforms as transforms
#import torch.optim as optim
#import torch.nn.functional as F
#from torchvision import datasets
#from torch.utils.data import DataLoader
#from torchvision.utils import save_image
#from torch.utils.data import Dataset
#from torchvision.transforms import ToTensor
#from torchvision.io import read_image


"""NEED TO ADD SAVE_DECODED_IMAGE!!!"""


def matlab_to_python_time(ds):
    
    """Conversion of matlab time to python time (human understandable)"""
    
    datenums = ds.coords['time'].values
    timestamps = pd.to_datetime(datenums-719529, unit='D')
    return timestamps


def select_region(ds, region=None):
    
    """Select region based on latitude for S2S Challenge"""
    
    dict_region = {'NH': (30, 90),
                   'SH': (-60, -30),
                   'TP': (-29, 29)}
    
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


def custom_scale(ds_s2s, negative=False, return_scaler=True):
    
    """Custom scaling based on anomaly sign"""
    
    if not negative:
        
        shifted_ds = ds_s2s[:,:]
        
    if negative:
        
        shifted_ds = np.abs(ds_s2s[:,:])
        
    scaled_ds = np.nan_to_num((shifted_ds - np.nanmin(shifted_ds, axis=0)) / (
                np.nanmax(shifted_ds, axis=0) - np.nanmin(shifted_ds, axis=0)), 
                nan=0, posinf=1, neginf=0)
    
    if not return_scaler:
        
        return scaled_ds
    
    if return_scaler:
        
        return scaled_ds, np.nanmax(shifted_ds, axis=0), np.nanmin(shifted_ds, axis=0)


def test_scale(ds_s2s, ds_MAX, ds_MIN, negative=False):
    
    """Custom scaling based on anomaly sign for testing data"""
    
    if not negative:
        
        shifted_ds = ds_s2s[:,:]
        
    if negative:
        
        shifted_ds = np.abs(ds_s2s[:,:])
        
    return np.nan_to_num((shifted_ds - ds_MIN) / (ds_MAX - ds_MIN), 
                          nan=0, posinf=1, neginf=0)


def preprocess_cesm(ds, land_mask, anomaly_sign):
    
    """Preprocessing of CESM2 data for deep learning model"""
    
    if anomaly_sign == 'positive':
        
        ds_ = (ds['anom']).where(ds['anom']>=0, 0.0)
        negative_statement = False
        
    if anomaly_sign == 'negative':
        
        ds_ = (ds['anom']).where(ds['anom']<=0, 0.0)
        negative_statement = True
        
    ds_ = xr.where(np.isnan(land_mask), np.nan, ds_.transpose('time','lat','lon'))
    ds_ = ds_.stack(dim_0=['lat','lon']).reset_index('dim_0').drop(['lat','lon'])
    ds_ = ds_.where(np.isfinite(ds_), drop=True)
    
    ds_, MAX, MIN = custom_scale(ds_.transpose('time','dim_0').values, 
                                 negative=negative_statement, return_scaler=True)
    
    return ds_.astype(np.float32), MAX, MIN


def preprocess_cpclabel(ds, land_mask, anomaly_sign, ds_MAX, ds_MIN):
    
    """Preprocessing of CPC data for deep learning model"""
    
    if anomaly_sign == 'positive':
        
        ds_ = (ds['anom']).where(ds['anom']>=0, 0.0)
        negative_statement = False
        
    if anomaly_sign == 'negative':
        
        ds_ = (ds['anom']).where(ds['anom']<=0, 0.0)
        negative_statement = True
        
    ds_ = xr.where(np.isnan(land_mask), np.nan, ds_.transpose('time','lat','lon'))
    ds_ = ds_.stack(dim_0=['lat','lon']).reset_index('dim_0').drop(['lat','lon'])
    ds_ = ds_.where(np.isfinite(ds_), drop=True)
    
    ds_ = test_scale(ds_.transpose('time','dim_0').values, ds_MAX, ds_MIN, 
                     negative=negative_statement)
    
    return ds_.astype(np.float32)


def inverse_minmax(dl_output, ds_MAX, ds_MIN, negative=False):
    
    """Inverse of custom scaling based on anomaly sign"""
    
    tmp_ds = (dl_output * (ds_MAX - ds_MIN)) + ds_MIN
    
    if not negative:
        
        return tmp_ds
    
    if negative:
        
        return -tmp_ds


def get_device():
    
    """Grab GPU"""
    
    if torch.cuda.is_available():
        device = 'cuda:0'
        
    else:
        device = 'cpu'
        
    return device


def make_dir():
    
    """Directory for saving progress images"""
    
    image_dir = 'Saved_Images'
    
    if not os.path.exists(image_dir):
        
        os.makedirs(image_dir)
