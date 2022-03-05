import warnings
from datetime import timedelta
import numpy as np
import pandas as pd
import xarray as xr
import sammon


def fxn():
    warnings.warn("future", FutureWarning)
    

def get_cold_indx(ds):
    """
    Extract indices for cold season.
    Grabbing Sept thru February init, for Oct thru March predictions.
    """
    dt_array = pd.to_datetime(ds['date_range'])
    return xr.where((dt_array.month>=9) | (dt_array.month<=2), True, False)


def preprocess_data(da, lat0=15, lat1=60, lon0=-140, lon1=-55, 
                    leadday0=1, leadday1=42, observations=False, roll_days=5):
    """
    Function to preprocess opened files.
    
    Week 1:  1,  2,  3,  4,  5,  6,  7
    Week 2:  8,  9, 10, 11, 12, 13, 14
    Week 3: 15, 16, 17, 18, 19, 20, 21
    Week 4: 22, 23, 24, 25, 26, 27, 28
    Week 5: 29, 30, 31, 32, 33, 34, 35
    Week 6: 36, 37, 38, 39, 40, 41, 42
    """
    if not observations:
        
        da = da.sel(lat=slice(lat0, lat1), lon=slice(lon0 + 360, lon1 + 360))
        da = da.rolling(lead=roll_days, min_periods=1, center=True).mean(skipna=True)
        
        return da.isel(lead=slice(leadday0, leadday1 + 1))['anom'].transpose('time','lead','lat','lon')
    
    if observations:
        
        da = da.where((da['lat']<=lat1) & (da['lat']>=lat0) & (
             da['lon']>=lon0+360) & (da['lon']<=lon1+360),drop=True)
        da = da.rolling(lead=roll_days, min_periods=1, center=True).mean(skipna=True)
        
        return da.isel(lead=slice(leadday0, leadday1 + 1))['anom'].transpose('time','lead','lat','lon')      


def open_era5_files(variable, return_time=False, date_time=True, 
                    lat0=15, lat1=60, lon0=-140, lon1=-55, 
                    leadday0=1, leadday1=42, rolldays=5):
    """
    Open ERA5 files. Input using variable.
    """
    ds = xr.open_dataset(f'/glade/scratch/molina/s2s/CESM2_OBS/era5_{variable}_anom_data.nc')
    
    if date_time:
        ds = ds.assign_coords(time=("time", ds['date_range'].data))
        ds = ds.assign_coords(
            doy=("time", 
                 np.array(
                     [pd.to_datetime(str(
                         d.dt.strftime('%m-%d').values+'-1999')).dayofyear for d in ds.time])))
    
    da = preprocess_data(ds, lat0, lat1, lon0, lon1, leadday0, leadday1,
                         roll_days=rolldays)[get_cold_indx(ds), ...]
    
    if not return_time:
        return da
    
    if return_time:
        return da, ds['date_range'][get_cold_indx(ds)].values


def open_cesm_files(variable, return_time=False, date_time=True,                    
                    lat0=15, lat1=60, lon0=-140, lon1=-55, 
                    leadday0=1, leadday1=42, rolldays=5):
    
    ds = xr.open_dataset(
        f'/glade/scratch/molina/s2s/CESM2/{variable}_anom_cesm2cam6v2_11members_s2s_data.nc')
    
    if date_time:
        ds = ds.assign_coords(time=("time", ds['date_range'].data))
        ds = ds.assign_coords(
            doy=("time", 
                 np.array(
                     [pd.to_datetime(str(
                         d.dt.strftime('%m-%d').values+'-1999')).dayofyear for d in ds.time])))
    
    da = preprocess_data(ds, lat0, lat1, lon0, lon1, leadday0, leadday1, 
                         roll_days=rolldays)[get_cold_indx(ds), ...]
    
    ta = ds['date_range'][get_cold_indx(ds)].values
    da = da[~np.isin(ta, np.array('2016-02-28T00:00:00', dtype='datetime64[ns]')), ...]
    ta = ta[~np.isin(ta, np.array('2016-02-28T00:00:00', dtype='datetime64[ns]'))]
    
    if not return_time:
        return da
    
    if return_time:
        return da, ta
    
    
def open_noaa_files(variable, return_time=False, date_time=True,                    
                    lat0=15, lat1=60, lon0=-140, lon1=-55, 
                    leadday0=1, leadday1=42, rolldays=5):
    
    ds = xr.open_dataset(
        f'/glade/scratch/molina/s2s/CESM2_OBS/{variable}_anom_ncpc_data.nc')
    
    ds = ds.assign_coords(x=("x", ds['lon'].data[0]))
    ds = ds.assign_coords(y=("y", ds['lat'].data[:,0]))
    ds = ds.drop('lat').drop('lon')
    ds = ds.rename({'x':'lon','y':'lat'})
    
    if date_time:
        ds = ds.assign_coords(time=("time", ds['date_range'].data))
        ds = ds.assign_coords(
            doy=("time", 
                 np.array(
                     [pd.to_datetime(str(
                         d.dt.strftime('%m-%d').values+'-1999')).dayofyear for d in ds.time])))
    
    da = preprocess_data(ds, lat0, lat1, lon0, lon1, leadday0, leadday1, 
                         roll_days=rolldays)[get_cold_indx(ds), ...]
    
    ta = ds['date_range'][get_cold_indx(ds)].values
    da = da[~np.isin(ta, np.array('2016-02-28T00:00:00', dtype='datetime64[ns]')), ...]
    ta = ta[~np.isin(ta, np.array('2016-02-28T00:00:00', dtype='datetime64[ns]'))]
    
    if not return_time:
        return da
    
    if return_time:
        return da, ta
    
    
def open_cesm_ensembles(variable, return_time=False, date_time=True, return_ens=True, 
                        lat0=15, lat1=60, lon0=-140, lon1=-55, 
                        leadday0=1, leadday1=42, rolldays=5):
    
    dict_ens = {}
    ensembles = ['00','01','02','03','04','05','06','07','08','09','10']
    
    for num, ens in enumerate(ensembles):
        
        ds = xr.open_dataset(
            f'/glade/scratch/molina/s2s/CESM2/{variable}_anom_cesm2cam6v2_{ens}member_s2s_data.nc')
        
        if date_time:
            ds = ds.assign_coords(time=("time", ds['date_range'].data))
            ds = ds.assign_coords(
                doy=("time", 
                     np.array(
                         [pd.to_datetime(str(
                             d.dt.strftime('%m-%d').values+'-1999')).dayofyear for d in ds.time])))
        
        dict_ens[num] = preprocess_data(ds, lat0, lat1, lon0, lon1, leadday0, leadday1, 
                                        roll_days=rolldays)[get_cold_indx(ds), ...]
    
    ca = xr.concat([dict_ens[0],dict_ens[1],dict_ens[2],dict_ens[3],dict_ens[4],dict_ens[5],
                    dict_ens[6],dict_ens[7],dict_ens[8],dict_ens[9],dict_ens[10]], 
                    dim='ensemble').transpose('time','ensemble','lead','lat','lon')
    
    ta = ds['date_range'][get_cold_indx(ds)].values
    
    ca = ca[~np.isin(ta, np.array('2016-02-28T00:00:00', dtype='datetime64[ns]')), ...]
    ta = ta[~np.isin(ta, np.array('2016-02-28T00:00:00', dtype='datetime64[ns]'))]
    
    if not return_time:
        return ca
    
    if return_time and not return_ens:
        return ca, ta
    
    if return_time and return_ens:
        ea = np.repeat(ensembles, len(ta)/len(ensembles))
        return ca, ta, ea


def open_ncpc_files(variable, model_data, return_time=False, date_time=True, 
                    lat0=15, lat1=60, lon0=-140, lon1=-55, 
                    leadday0=1, leadday1=42, rolldays=5):
    
    ds = xr.open_dataset(f'/glade/scratch/molina/s2s/CESM2_OBS/{variable}_anom_ncpc_data.nc')
    
    if date_time:
        ds = ds.assign_coords(time=("time", ds['date_range'].data))
        ds = ds.assign_coords(
            doy=("time", 
                 np.array(
                     [pd.to_datetime(str(
                         d.dt.strftime('%m-%d').values+'-1999')).dayofyear for d in ds.time])))
    
    da = preprocess_data(ds, lat0, lat1, lon0, lon1, leadday0, leadday1, 
                         observations=True, roll_days=rolldays)[get_cold_indx(ds), ...]
    
    ca = xr.concat([da,da,da,da,da,da,da,da,da,da,da], dim='time')
    
    ca = xr.DataArray(data=ca.values, 
                      dims=["lat", "lon", "time"], 
                      coords=dict(lon=(["lon"], model_data.lon.values),
                                  lat=(["lat"], model_data.lat.values),
                                  time=model_data.time.values)).rename('anom')
    
    if not return_time:
        return ca
    
    if return_time:
        ta = ds['date_range'][get_cold_indx(ds)].values
        ta = pd.to_datetime(np.hstack([ta,ta,ta,ta,ta,ta,ta,ta,ta,ta,ta]))
        return ca, ta
    
    
def standardize_vals(data):
    """
    Output mean and standard deviation for standardizing data.
    """
    if type(data) == xr.core.dataarray.DataArray:
        data = data.values
    mu = np.nanmean(data)
    std = np.nanstd(data)
    return mu, std


def standardize_apply(data, mu, std):
    """
    Output standardized data using input mean and std.
    """
    return (data - mu) / std


def monthly_mean(ds):
    """
    Output monthly climatology.
    """
    return ds.groupby("time.month").mean("time", skipna=True)


def monthly_std(ds):
    """
    Output monthly climatology.
    """
    return ds.groupby("time.month").std("time", skipna=True)


def weekly_mean(ds):
    """
    Output weekly climatology.
    """
    return ds.groupby("time.week").mean("time", skipna=True)


def weekly_std(ds):
    """
    Output weekly climatology.
    """
    return ds.groupby("time.week").std("time", skipna=True)


def daily_mean(ds):
    """
    Output daily climatology.
    """
    return ds.groupby("doy").mean("time", skipna=True)


def daily_std(ds):
    """
    Output daily climatology.
    """
    return ds.groupby("doy").std("time", skipna=True)


def monthly_standard_anomalies(ds, ds_ob=None):
    """
    Output standardized monthly anomalies, climatology, and std.
    """
    if ds_ob is None:
        ds_ob = ds
    climatology_mean = monthly_mean(ds_ob)
    climatology_std  = monthly_std(ds_ob)
    stand_anomalies = xr.apply_ufunc(
                                lambda x, m, s: (x - m) / s,
                                ds.groupby("time.month"),
                                climatology_mean,
                                climatology_std,
                            )
    return stand_anomalies, climatology_mean, climatology_std


def weekly_standard_anomalies(ds, ds_ob=None):
    """
    Output standardized weekly anomalies, climatology, and std.
    """
    if ds_ob is None:
        ds_ob = ds
    climatology_mean = weekly_mean(ds_ob)
    climatology_std  = weekly_std(ds_ob)
    stand_anomalies = xr.apply_ufunc(
                                lambda x, m, s: (x - m) / s,
                                ds.groupby("time.week"),
                                climatology_mean,
                                climatology_std,
                            )
    return stand_anomalies, climatology_mean, climatology_std


def daily_standard_anomalies(ds, ds_ob=None):
    """
    Output standardized daily anomalies, climatology, and std.
    """
    if ds_ob is None:
        ds_ob = ds
    climatology_mean = daily_mean(ds_ob)
    climatology_std  = daily_std(ds_ob)
    stand_anomalies = xr.apply_ufunc(
                                lambda x, m, s: (x - m) / s,
                                ds.groupby("doy"),
                                climatology_mean,
                                climatology_std,
                            )
    return stand_anomalies, climatology_mean, climatology_std


def node_assignment(ds, trained_som, ensemble=None):
    
    if ensemble is None:
        ensemble = ['00']
    
    som_era5_ = np.zeros((len(ensemble),
                          np.unique(ds.time).shape[0], 
                          np.unique(ds.lead).shape[0]))
    
    som_time_ = np.empty((len(ensemble),
                          np.unique(ds.time).shape[0], 
                          np.unique(ds.lead).shape[0]), dtype='datetime64[s]')
        
    for time_ix, di_ in enumerate(np.unique(ds.time)):
        
        for lead_ix, dl_ in enumerate(np.unique(ds.lead)):

            data_ = ds[((ds.time==di_) & (ds.lead==dl_)), :].values
            
            ensm_ix = 0
            
            for d_ in data_:
                
                freq = trained_som.activation_response(np.expand_dims(d_, axis=0))
                
                som_era5_[ensm_ix, time_ix, lead_ix] = np.argwhere(freq.flatten())[0][0]

                som_time_[ensm_ix, time_ix, lead_ix] = (pd.to_datetime(ds.time)[((
                    ds.time==di_) & (ds.lead==dl_))] + timedelta(days=lead_ix + 1))[0]
                
                ensm_ix += 1
        
        if time_ix == np.unique(ds.time).shape[0]:
            ensm_ix += 1
            
    return som_era5_, som_time_


def node_percentages(node_, lead_time=None):
    """
    np.arange( 0,14,1)
    np.arange(13,27,1)
    np.arange(27,41,1)
    """
    if lead_time is not None:
        node_ = node_[:,:,lead_time]
    
    unique_ = {}; counts_ = {}
    
    for i in range(node_.shape[0]):
    
        unique_[i], counts_[i] = np.unique(node_[i,:,:], return_counts=True)
        counts_[i] = counts_[i]/np.sum(counts_[i])
    
    if node_.shape[0] == 1:
        return unique_[0], counts_[0]
    
    if node_.shape[0] > 1:
        
        stds_ = {}
        
        for r_ in range(len(counts_[0])):
            new_std = []
            
            for d_ in counts_.values():
                new_std.append(d_[r_])
                
            stds_[r_] = np.std(new_std)
        
        return unique_, counts_, np.asarray(list(stds_.values()))
    
    