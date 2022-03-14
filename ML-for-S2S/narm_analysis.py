import numpy as np
import pandas as pd
import xarray as xr
from preprocessing import cesm2_filelist


def era5_z500(lat0=10, lat1=70, lon0=-150, lon1=-40):
    """
    Assemble ERA5 climatology for weather regime study.
    Region selected based on Robertson et al. 2020.
    """
    start='1999-01-01'; end='2019-12-31'
    var = "Z"; filename = "e5.oper.an.pl.128_129_z.ll025sc"

    td = pd.date_range(start=start, end=end, freq='D')
    td = td[~((td.day==29)&(td.month==2))]

    doy = 0; yr = 0; dates = []; years = []

    for num, t in enumerate(td):

        ds_ = xr.open_dataset(
            f"/glade/scratch/molina/era5_z500_regrid/{filename}.{t.strftime('%Y%m%d')}.nc")[var].transpose('y','x')

        dates.append(pd.Timestamp(t.strftime('%Y%m%d')))

        if num == 0:
            clim = np.zeros((td.year.unique().shape[0] * 365, ds_.shape[0], ds_.shape[1]))
            lats = ds_.lat[:,0].values
            lons = ds_.lon[0,:].values

        clim[num,:,:] = ds_.values

    data_assemble = xr.Dataset({
                         'clim': (['time','lat','lon'], clim),
                        },
                         coords =
                        {'time': (['time'], pd.to_datetime(dates)),
                         'lat' : (['lat'], lats),
                         'lon' : (['lon'], lons)
                        },
                        attrs = 
                        {'File Author' : 'Maria J. Molina',})
    
    data_assemble = data_assemble.sel(lat=slice(lat0, lat1), lon=slice(lon0 + 360, lon1 + 360))
    
    return data_assemble


def cesm_z500(ensemble_str):
    """
    Preprocessing of hindcasts following weather regime framework.
    
    Example:
    # ds00 = cesm_z500(ensemble_str='00')
    # ds00.to_netcdf('/glade/scratch/molina/s2s/CESM2/cesm2_z500_00.nc')
    """
    var = 'zg_500'
    char_1 = "cesm2cam6v2_"
    char_2 = "_00z_d01_d46"

    filelist00 = cesm2_filelist(variable=var, 
                                parent_directory='/glade/scratch/molina/s2s/',
                                ensemble=[ensemble_str])

    for i in range(len(filelist00)):

        thetime = pd.to_datetime(
            datetime.strptime(
                filelist00[i][filelist00[i].find(char_1)+12:filelist00[i].find(char_2)], 
                '%d%b%Y'))

        if i == 0:

            ds00 = xr.open_dataset(filelist00[i])[var].rename(
                {'time':'lead'}).transpose('lead','lat','lon').assign_coords(
                {'time':thetime}).expand_dims('time').assign_coords({'lead':np.arange(0,46,1)}).assign_coords(
                {'ensemble':int(ensemble_str)+1}).expand_dims('ensemble')

            ds_t = ds00

        if i > 0:

            ds00 = xr.open_dataset(filelist00[i])[var].rename(
                {'time':'lead'}).transpose('lead','lat','lon').assign_coords(
                {'time':thetime}).expand_dims('time').assign_coords({'lead':np.arange(0,46,1)}).assign_coords(
                {'ensemble':int(ensemble_str)+1}).expand_dims('ensemble')

            ds_t = xr.concat([ds_t,ds00],dim='time')
            
    return ds_t


def era5_climo_wrs(ds_era5, rolling_days=5, variable='clim'):
    """
    Preprocessing for subseasonal analysis.
    """
    # smooth lead time dim
    fiveday_run_mean = ds_era5[variable].rolling(
        time=rolling_days, min_periods=1, center=True).mean(skipna=True)

    # daily climo mean
    ds_era5_clim = fiveday_run_mean.groupby("time.dayofyear").mean(skipna=True)

    # anom
    fiveday_run_anom = fiveday_run_mean.groupby("time.dayofyear") - ds_era5_clim
    
    return fiveday_run_anom


def open_cesm_climo_wrs(lat0=10, lat1=70, lon0=-150, lon1=-40,
                        directory='/glade/scratch/molina/s2s/CESM2/'):
    """
    Open and concat cesm files for weather regime analysis.
    """
    ds00 = xr.open_dataset(f'{directory}cesm2_z500_00.nc').sel(
        lat=slice(lat0, lat1), lon=slice(lon0 + 360, lon1 + 360))
    ds01 = xr.open_dataset(f'{directory}cesm2_z500_01.nc').sel(
        lat=slice(lat0, lat1), lon=slice(lon0 + 360, lon1 + 360))
    ds02 = xr.open_dataset(f'{directory}cesm2_z500_02.nc').sel(
        lat=slice(lat0, lat1), lon=slice(lon0 + 360, lon1 + 360))
    ds03 = xr.open_dataset(f'{directory}cesm2_z500_03.nc').sel(
        lat=slice(lat0, lat1), lon=slice(lon0 + 360, lon1 + 360))
    ds04 = xr.open_dataset(f'{directory}cesm2_z500_04.nc').sel(
        lat=slice(lat0, lat1), lon=slice(lon0 + 360, lon1 + 360))
    ds05 = xr.open_dataset(f'{directory}cesm2_z500_05.nc').sel(
        lat=slice(lat0, lat1), lon=slice(lon0 + 360, lon1 + 360))
    ds06 = xr.open_dataset(f'{directory}cesm2_z500_06.nc').sel(
        lat=slice(lat0, lat1), lon=slice(lon0 + 360, lon1 + 360))
    ds07 = xr.open_dataset(f'{directory}cesm2_z500_07.nc').sel(
        lat=slice(lat0, lat1), lon=slice(lon0 + 360, lon1 + 360))
    ds08 = xr.open_dataset(f'{directory}cesm2_z500_08.nc').sel(
        lat=slice(lat0, lat1), lon=slice(lon0 + 360, lon1 + 360))
    ds09 = xr.open_dataset(f'{directory}cesm2_z500_09.nc').sel(
        lat=slice(lat0, lat1), lon=slice(lon0 + 360, lon1 + 360))
    ds10 = xr.open_dataset(f'{directory}cesm2_z500_10.nc').sel(
        lat=slice(lat0, lat1), lon=slice(lon0 + 360, lon1 + 360))
    
    ds_cesm = xr.concat([ds00,ds01,ds02,ds03,ds04,ds05,ds06,ds07,ds08,ds09,ds10],
                    dim='ensemble')
    
    return ds_cesm
    
    
def cesm_climo_wrs(ds_cesm, rolling_days=5, variable='zg_500'):
    """
    Preprocessing for subseasonal analysis.
    """
    # smooth lead time dim
    step_1 = ds_cesm[variable].rolling(
        lead=rolling_days,min_periods=1,center=True).mean(skipna=True)

    # ensemble mean
    step_2 = step_1.mean('ensemble',skipna=True)

    # daily climo mean
    step_3 = step_2.groupby("time.dayofyear").mean(skipna=True)

    # anom
    step_4 = step_2.groupby("time.dayofyear") - step_3
    
    return step_4

