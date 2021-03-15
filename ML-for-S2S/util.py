import xesmf as xe
import numpy as np
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

def regridder(ds, variable, method='nearest_s2d', offset=0.5, dcoord=1.0, reuse_weights=False):
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
        Returns:
            Regridded file.
        """
        lat_coord='lat'
        lon_coord='lon'
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
