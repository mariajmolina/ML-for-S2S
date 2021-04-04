import numpy as np
import pandas as pd
import xarray as xr

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
        
def normalize_data(data):
    """
    Function for normalizing data prior to training using z-score.
    """
    return (data - np.nanmean(data)) / np.nanstd(data)
        
def datenum_to_datetime(datenums):
    """
    Convert Matlab datenum into Python datetime.
    
    Args:
        datenums (list or array): Date(s) in datenum format.
        
    Returns:
        Datetime objects corresponding to datenums.
    """
    from datetime import datetime, timedelta
    
    new_datenums = []
    
    for datenum in datenums:
        days = datenum % 1
        new_datenums.append(datetime.fromordinal(int(datenum)) + timedelta(days=days) - timedelta(days=365))
        
    return pd.to_datetime(new_datenums)

def regridder(ds, variable, method='nearest_s2d', offset=0.5, dcoord=1.0, reuse_weights=False,
              lat_coord='lat', lon_coord='lon'):
        """
        Function to regrid netcdf onto different grid.
        
        Args:
            ds (xarray dataset): Variable file.
            variable (str): String for variable name in ``ds``.
            method (str): Regrid method. Defaults to ``nearest_s2d``. Options include 
                          'bilinear', 'conservative', 'nearest_s2d', 'nearest_d2s', 'patch'.
            offset (float): Value to recenter grid by. Defaults to ``0.5`` for 1.0-degree grid.
            dcoord (float): Distance between lat/lons. Defaults to ``1.0``.
            reuse_weights (boolean): Whether to use precomputed weights to speed up calculation.
                                     Defaults to ``False``.
            lat_coord (str): Latitude coordinate. Defaults to ``lat``.
            lon_coord (str): Longitude coordinate. Defaults to ``lon``.
            
        Returns:
            Regridded file.
        """
        import xesmf as xe
        
        lat0_bnd = int(np.around(ds[lat_coord].min(skipna=True).values))
        lat1_bnd = int(np.around(ds[lat_coord].max(skipna=True).values))
        lon0_bnd = int(np.around(ds[lon_coord].min(skipna=True).values))
        lon1_bnd = int(np.around(ds[lon_coord].max(skipna=True).values))
        
        ds_out = xe.util.grid_2d(lon0_b=lon0_bnd-offset, lon1_b=lon1_bnd+offset, d_lon=dcoord, 
                                 lat0_b=lat0_bnd-offset, lat1_b=lat1_bnd+offset, d_lat=dcoord)
        
        if method == 'conservative':
            latb = np.hstack([(ds[lat_coord]-((ds[lat_coord][1]-ds[lat_coord][0])/2)),ds[lat_coord][-1]+((ds[lat_coord][1]-ds[lat_coord][0])/2)])
            lonb = np.hstack([(ds[lon_coord]-((ds[lon_coord][1]-ds[lon_coord][0])/2)),ds[lon_coord][-1]+((ds[lon_coord][1]-ds[lon_coord][0])/2)])
            ds = ds.assign_coords({'lat_b':(latb),
                                   'lon_b':(lonb)})
            
        regridder = xe.Regridder(ds, ds_out, method, reuse_weights=reuse_weights)
        
        dr_out = regridder(ds[variable], keep_attrs=True)
        dr_out = dr_out.assign_coords(lon=('x', dr_out.coords[lon_coord][0,:].values), lat=('y', dr_out.coords[lat_coord][:,0].values))
        dr_out = dr_out.rename(y=lat_coord, x=lon_coord)
        
        return dr_out

def compute_rws(ds_u, ds_v, lat_coord='lat', lon_coord='lon', time_coord='time'):
    """
    Computation of absolute vorticity, divergence, and Rossby wave source.
    Outputs xarray datasets of each.
    
    Args:
        ds_u (xarray data array): Zonal (u) wind (m/s).
        ds_v (xarray data array): Meridional (v) wind (m/s).
        lat_coord (str): Latitude coordinate. Defaults to ``lat``.
        lon_coord (str): Longitude coordinate. Defaults to ``lon``.
        time_coord (str): Time coordinate. Defaults to ``time``.
        
    Returns:
        Xarray datasets for absolute vorticity, divergence, and Rossby wave source.
    """
    from windspharm.standard import VectorWind
    from windspharm.tools import prep_data, recover_data, order_latdim
    
    # grab lat and lon coords
    lats = ds_u.coords[lat_coord].values
    lons = ds_u.coords[lon_coord].values
    time = ds_u.coords[time_coord].values

    _, wnd_info = prep_data(ds_u.values, 'tyx')

    # reorder dims into lat, lon, time
    uwnd = ds_u.transpose(lat_coord, lon_coord, time_coord).values
    vwnd = ds_v.transpose(lat_coord, lon_coord, time_coord).values

    # reorder lats to north-south direction
    lats, uwnd, vwnd = order_latdim(lats, uwnd, vwnd)

    # initialize wind vector instance
    w = VectorWind(uwnd, vwnd)

    # Absolute vorticity (sum of relative and planetary vorticity).
    eta = w.absolutevorticity()

    # Horizontal divergence.
    div = w.divergence()

    # Irrotational (divergent) component of the vector wind.
    uchi, vchi = w.irrotationalcomponent()

    # Computes the vector gradient of a scalar field on the sphere.
    etax, etay = w.gradient(eta)

    # Compute rossby wave source
    S = -eta * div - (uchi * etax + vchi * etay)

    # recover data shape
    S = recover_data(S, wnd_info)
    div = recover_data(div, wnd_info)
    eta = recover_data(eta, wnd_info)
    
    # assemble xarray datasets
    data_rws = xr.Dataset({
                         'rws': (['time', 'lat', 'lon'], S),},
                         coords =
                        {'time': (['time'], time),
                         'lat' : (['lat'], lats),
                         'lon' : (['lon'], lons)},
                        attrs = {'long_name' : 'Rossby wave source'})
    data_div = xr.Dataset({
                         'div': (['time', 'lat', 'lon'], div),},
                         coords =
                        {'time': (['time'], time),
                         'lat' : (['lat'], lats),
                         'lon' : (['lon'], lons)},
                        attrs = {'long_name' : 'Horizontal divergence (300-mb)'})
    data_eta = xr.Dataset({
                         'eta': (['time', 'lat', 'lon'], eta),},
                         coords =
                        {'time': (['time'], time),
                         'lat' : (['lat'], lats),
                         'lon' : (['lon'], lons)},
                        attrs = {'long_name' : 'Absolute vorticity (sum of relative and planetary vorticity)'})
    
    return data_eta, data_div, data_rws
