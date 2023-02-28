from datetime import timedelta
import numpy as np
import pandas as pd
import xarray as xr
import torch
from torch.utils.data import Dataset


class S2SDataset(Dataset):
    
    """
    Class instantiation for file lists of cesm/era5 train, validate, and test data.
    
    Args:
        week (int): lead time week (1, 2, 3, 4, 5, or 6).
        variable (str): variable (tas2m, prsfc, tas2m_anom, or prsfc_anom).
        norm (str): normalization. Defaults to zscore. Also use None, minmax, or negone.
        region (str): region method used. Defaults 'fixed' uses one region. 'random' changes regions.
        minv (float): minimum value for normalization. Defaults to None.
        maxv (float): maximum value for normalization. Defaults to None.
        mnv (float): mean value for normalization. Defaults to None.
        stdv (float): standard deviation value for normalization. Defaults to None.
        lon0 (float): bottom left corner of 'fixed' region (0 to 360). Defaults to None.
        lat0 (float): bottom left corner of 'fixed' region (-90 to 90). Defaults to None.
        dxdy (float): number of grid cells for 'fixed' or 'random' region. Defaults to 32.
        feat_topo (boolean): use terrian heights (era5) as feature. Defaults True.
        feat_lats (boolean): use latitudes as feature. Defaults True.
        feat_lons (boolean): use longitudes as feature. Defaults True.
        startdt (str): datetime start. Defaults to 1999-02-01.
        enddt (str): datetime end. Defaults to 2021-12-31.
        homedir (str): home directory. Defaults to /glade/scratch/molina/.
        
    Returns:
        list of cesm and era5 files for train/val/test sets.
    
    """
    
    def __init__(self, week, variable, norm='zscore', region='fixed',
                 minv=None, maxv=None, mnv=None, stdv=None, lon0=None, lat0=None, dxdy=32,
                 feat_topo=True, feat_lats=True, feat_lons=True, 
                 startdt='1999-02-01', enddt='2021-12-31', homedir='/glade/scratch/molina/'):
        
        self.week = week
        self.day_init, self.day_end = self.leadtime_help()
        
        self.variable_ = variable
        self.startdt = startdt
        self.enddt = enddt
        
        self.homedir = homedir
        self.cesm_dir = f'{self.homedir}cesm_{self.variable_}_week{self.week}/'
        self.era5_dir = f'{self.homedir}era5_{self.variable_}_week{self.week}/'
        
        self.filelists()
        
        self.norm = norm
        if self.norm == 'zscore':
            self.zscore_values(mnv, stdv)
        if self.norm == 'minmax':
            self.minmax_values(minv, maxv)
        if self.norm == 'negone':
            self.minmax_values(minv, maxv)
            
        self.region_ = region
        
        if self.region_ == 'fixed':
            
            self.lon0=lon0
            self.lat0=lat0
            self.dxdy=dxdy
            
        if self.region_ == 'random':
            
            self.dxdy=dxdy
            assert lon0 != None and lat0 != None, 'please set lat0 and lon0 to None'
            
        self.feat_topo=feat_topo
        self.feat_lats=feat_lats
        self.feat_lons=feat_lons
            
        
    def __len__(self):
        
        return len(self.list_of_cesm)

    
    def __getitem__(self, idx):
        """
        assembles input and label data
        """
        # create files using random indices
        self.create_files(idx)
        
        image = self.img_train
        label = self.img_label
        
        # need to convert precip cesm file
        if self.variable_ == 'prsfc':
            image = image * 84600 # convert kg/m2/s to mm/day
        
        # normalization options applied here
        if self.norm == 'zscore':
            image, label = self.zscore_compute(image), self.zscore_compute(label)
        if self.norm == 'minmax':
            image, label = self.minmax_compute(image), self.minmax_compute(label)
        if self.norm == 'negone':
            image, label = self.negone_compute(image), self.negone_compute(label)
        
        # add the spatial variable to coordinate data
        self.coord_data["cesm"]=(['sample','x','y'], 
                                 image.transpose('sample','lon','lat').values)
        self.coord_data["era5"]=(['sample','x','y'], 
                                 label.transpose('sample','x','y').values)
        
        # features including terrain, lats, and lons
        if self.feat_topo and self.feat_lats and self.feat_lons:
            # input features
            img = xr.concat([self.coord_data['top'],
                             self.coord_data['lat'],
                             self.coord_data['lon'],
                             self.coord_data['cesm']],dim='feature')
            
        # features including terrain
        if self.feat_topo and not self.feat_lats and not self.feat_lons:
            # input features
            img = xr.concat([self.coord_data['top'],
                             self.coord_data['cesm']],dim='feature')
            
        # features including lats and lons
        if not self.feat_topo and self.feat_lats and self.feat_lons:
            # input features
            img = xr.concat([self.coord_data['lat'],
                             self.coord_data['lon'],
                             self.coord_data['cesm']],dim='feature')
            
        # features including terrain and lat
        if self.feat_topo and self.feat_lats and not self.feat_lons:
            # input features
            img = xr.concat([self.coord_data['top'],
                             self.coord_data['lat'],
                             self.coord_data['cesm']],dim='feature')
            
        # features including terrain and lon
        if self.feat_topo and not self.feat_lats and self.feat_lons:
            # input features
            img = xr.concat([self.coord_data['top'],
                             self.coord_data['lon'],
                             self.coord_data['cesm']],dim='feature')
            
        # no extra features
        if not self.feat_topo and not self.feat_lats and not self.feat_lons:
            # input features
            img = xr.concat([self.coord_data['cesm']],dim='feature') 
        
        # label
        lbl = xr.concat([self.coord_data['era5']],dim='feature')
        
        # slice region
        img, lbl = self.box_cutter(img, lbl)
            
        return {'input': img.transpose('feature','sample','x','y').values, 
                'label': lbl.transpose('feature','sample','x','y').values}
    
    
    def leadtime_help(self):
        """
        helps with lead time start and end period
        """
        # start dict
        weekdict_init = {
            1: 1,
            2: 8,
            3: 15,
            4: 22,
            5: 29,
            6: 36,
        }
        
        # end dict
        weekdict_end = {
            1: 7,
            2: 14,
            3: 21,
            4: 28,
            5: 35,
            6: 42,
        }
        
        return weekdict_init[self.week], weekdict_end[self.week]
    
    
    def zscore_compute(self, data):
        """
        Function for normalizing data prior to training using z-score.
        """
        return (data - self.mean_val) / self.std_val


    def minmax_compute(self, data):
        """
        Min max computation.
        """
        return (data - self.min_val) / (self.max_val - self.min_val)
    
    
    def negone_compute(self, data):
        """
        Scale between negative 1 and positive 1.
        """
        return (2 * (data - self.min_val) / (self.max_val - self.min_val)) - 1
    
    
    def datetime_range(self):
        """
        helps with creating datetime range for data assembly
        """
        dt_cesm = pd.date_range(start=self.startdt, end=self.enddt, freq='W-MON')
        
        # remove missing july date/file
        dt_cesm = dt_cesm[~((dt_cesm.month==7)&(dt_cesm.day==26)&(dt_cesm.year==2021))]
        
        # anomalies have a lead time issue, this resolves it
        if self.variable_ == 'prsfc_anom' or self.variable_ == 'tas2m_anom':
            dt_cesm = dt_cesm[~((dt_cesm.month==2)&(dt_cesm.day==29)&(dt_cesm.year==2016))]
        
        # list containing datetime array
        matches = []

        # loop through datetimes
        for num, (yr, mo, dy) in enumerate(zip(
            dt_cesm.strftime("%Y"), dt_cesm.strftime("%m"), dt_cesm.strftime("%d"))):

            # time adjustment for leap year
            if mo == '02' and dy == '29':
                matches.append(dt_cesm[num] - timedelta(days=1))
            else:
                matches.append(dt_cesm[num])
        
        self.dt_cesm = pd.to_datetime(matches)
        
        
    def filelists(self):
        """
        creates list of filelists from the datetime range
        """
        # run related method
        self.datetime_range()

        # lists to be populated with filenames
        self.list_of_cesm = []
        self.list_of_era5 = []

        # loop through datetimes
        for i in self.dt_cesm:
            
            # convert datetime to string
            dt_string = str(i.strftime("%Y")+i.strftime("%m")+i.strftime("%d"))
            
            self.list_of_cesm.append(
                f'{self.cesm_dir}cm_{self.variable_}_'+dt_string+'.nc') # cesm2 list
            self.list_of_era5.append(
                f'{self.era5_dir}e5_{self.variable_}_'+dt_string+'.nc') # era5 list
            
            
    def zscore_values(self, mnv, stdv):
        """
        compute zscore values
        """
        # help with variable names inside files
        if self.variable_ == 'tas2m':
            var = 'tas_2m'
        if self.variable_ == 'prsfc':
            var = 'pr_sfc'
        if self.variable_ == 'prsfc_anom' or self.variable_ == 'tas2m_anom':
            var = 'anom'
        
        # if mean and standard deviation are NOT provided do this (only era5)
        if mnv == None or stdv == None:
        
            self.mean_val = xr.open_mfdataset(
                self.list_of_era5, concat_dim='sample', combine='nested')[var].mean(
                skipna=True).values

            self.std_val = xr.open_mfdataset(
                self.list_of_era5, concat_dim='sample', combine='nested')[var].std(
                skipna=True).values
            
        # if mean and standard deviation ARE provided do this
        if mnv != None and stdv != None:
            
            self.mean_val = mnv
            self.std_val = stdv
        

    def minmax_values(self, minv, maxv):
        """
        compute minmax values
        """
        # help with variable names inside files
        if self.variable_ == 'tas2m':
            var = 'tas_2m'
        if self.variable_ == 'prsfc':
            var = 'pr_sfc'
        if self.variable_ == 'prsfc_anom' or self.variable_ == 'tas2m_anom':
            var = 'anom'
        
        # if min and max are NOT provided do this (only era5)
        if minv == None or maxv == None:
            
            self.max_val = xr.open_mfdataset(
                self.list_of_era5, concat_dim='sample', combine='nested')[var].max(
                skipna=True).values
        
            self.min_val = xr.open_mfdataset(
                self.list_of_era5, concat_dim='sample', combine='nested')[var].min(
                skipna=True).values
            
        # if min and max ARE provided do this
        if minv != None and maxv != None:
            
            self.max_val = maxv
            self.min_val = minv
        
        
    def create_files(self, indx):
        """
        create input and label data using file list and indx from sampling
        """
        # help with variable names inside files
        if self.variable_ == 'tas2m':
            var = 'tas_2m'
        if self.variable_ == 'prsfc':
            var = 'pr_sfc'
        if self.variable_ == 'prsfc_anom' or self.variable_ == 'tas2m_anom':
            var = 'anom'
            
        # coordinates (terrain and lat/lon features)
        self.coord_data = xr.open_dataset(
            self.homedir+'/ml_coords.nc').expand_dims('sample')
        
        # open files using lists and indices
        self.img_train = xr.open_mfdataset(self.list_of_cesm[indx], 
                                           concat_dim='sample', 
                                           combine='nested')[var]
        self.img_label = xr.open_mfdataset(self.list_of_era5[indx], 
                                           concat_dim='sample', 
                                           combine='nested')[var]
        
        
    def box_cutter(self, ds1, ds2):
        """
        help slicing region
        """
        # if random/moving region is desired, do this
        if self.region_ == 'random':
            
            # this needs to be double checked (recent change)
            range_x = np.arange(0., 358. + 1 - self.dxdy, 1)
            range_y = np.arange(-90., 89. + 1 - self.dxdy, 1)

            ax = np.random.choice(range_x, replace=False)
            by = np.random.choice(range_y, replace=False)
        
        # if a fixed region is desired, do this
        if self.region_ == 'fixed':
            
            ax = self.lon0
            by = self.lat0
        
        # slicing occurs here using data above
        ds1 = ds1.sel(y=slice(by, by + self.dxdy), x=slice(ax, ax + self.dxdy))
        ds2 = ds2.sel(y=slice(by, by + self.dxdy), x=slice(ax, ax + self.dxdy))
        
        return ds1, ds2