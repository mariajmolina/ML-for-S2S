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
        startdt (str): datetime start. Defaults to 1999-02-01.
        enddt (str): datetime end. Defaults to 2021-12-31.
        homedir (str): home directory. Defaults to /glade/scratch/molina/.
        
    Returns:
        list of cesm and era5 files for train/val/test sets.
    
    """
    
    def __init__(self, week, variable, norm='zscore', 
                 minv=None, maxv=None, mnv=None, stdv=None,
                 startdt='1999-02-01', enddt='2021-12-31', 
                 homedir='/glade/scratch/molina/'):
        
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
        
        
    def __len__(self):
        
        return len(self.list_of_cesm)

    
    def __getitem__(self, idx):
        
        self.create_files(idx)
        
        image = self.img_train
        label = self.img_label
        
        if self.norm == 'zscore':
            
            image, label = self.zscore_compute(image), self.zscore_compute(label)
            
        if self.norm == 'minmax':
            
            image, label = self.minmax_compute(image), self.minmax_compute(label)
            
        if self.norm == 'negone':
            
            image, label = self.negone_compute(image), self.negone_compute(label)
        
        self.coord_data["cesm"]=(['sample','x','y'], 
                                 image.transpose('sample','lon','lat').values)
        
        self.coord_data["era5"]=(['sample','x','y'], 
                                 label.transpose('sample','x','y').values)
        
        img = xr.concat([self.coord_data['top'],
                         self.coord_data['lat'],
                         self.coord_data['lon'],
                         self.coord_data['cesm']],dim='feature')
        
        lbl = xr.concat([self.coord_data['era5']],dim='feature')
        
        img, lbl = self.box_cutter(img, lbl)
            
        return {'input': img.transpose('feature','sample','x','y').values, 
                'label': lbl.transpose('feature','sample','x','y').values}
    
    
    def leadtime_help(self):

        weekdict_init = {
            1: 1,
            2: 8,
            3: 15,
            4: 22,
            5: 29,
            6: 36,
        }

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
        
        dt_cesm = pd.date_range(start=self.startdt, end=self.enddt, freq='W-MON')
        
        # remove missing july date/file
        dt_cesm = dt_cesm[~((dt_cesm.month==7)&(dt_cesm.day==26)&(dt_cesm.year==2021))]
        
        if self.variable_ == 'prsfc_anom' or self.variable_ == 'tas2m_anom':
            dt_cesm = dt_cesm[~((dt_cesm.month==2)&(dt_cesm.day==29)&(dt_cesm.year==2016))]
        
        matches = []

        for num, (yr, mo, dy) in enumerate(zip(
            dt_cesm.strftime("%Y"), dt_cesm.strftime("%m"), dt_cesm.strftime("%d"))):

            if mo == '02' and dy == '29':
                matches.append(dt_cesm[num] - timedelta(days=1))
            else:
                matches.append(dt_cesm[num])
        
        self.dt_cesm = pd.to_datetime(matches)
        
        
    def filelists(self):
        
        self.datetime_range()

        self.list_of_cesm = []
        self.list_of_era5 = []

        for i in self.dt_cesm:
            
            dt_string = str(i.strftime("%Y")+i.strftime("%m")+i.strftime("%d"))
            
            self.list_of_cesm.append(
                f'{self.cesm_dir}cm_{self.variable_}_'+dt_string+'.nc')

            self.list_of_era5.append(
                f'{self.era5_dir}e5_{self.variable_}_'+dt_string+'.nc')
            
            
    def zscore_values(self, mnv, stdv):
        
        if self.variable_ == 'tas2m':
            var = 'tas_2m'
            
        if self.variable_ == 'prsfc':
            var = 'pr_sfc'
            
        if self.variable_ == 'prsfc_anom' or self.variable_ == 'tas2m_anom':
            var = 'anom'
        
        if mnv == None or stdv == None:
        
            self.mean_val = xr.open_mfdataset(
                self.list_of_era5, concat_dim='sample', combine='nested')[var].mean(
                skipna=True).values

            self.std_val = xr.open_mfdataset(
                self.list_of_era5, concat_dim='sample', combine='nested')[var].std(
                skipna=True).values
            
        if mnv != None and stdv != None:
            
            self.mean_val = mnv
            self.std_val = stdv
        

    def minmax_values(self, minv, maxv):
        
        if self.variable_ == 'tas2m':
            var = 'tas_2m'
            
        if self.variable_ == 'prsfc':
            var = 'pr_sfc'
            
        if self.variable_ == 'prsfc_anom' or self.variable_ == 'tas2m_anom':
            var = 'anom'
        
        if minv == None or maxv == None:
            
            self.max_val = xr.open_mfdataset(
                self.list_of_era5, concat_dim='sample', combine='nested')[var].max(
                skipna=True).values
        
            self.min_val = xr.open_mfdataset(
                self.list_of_era5, concat_dim='sample', combine='nested')[var].min(
                skipna=True).values
            
        if minv != None and maxv != None:
            
            self.max_val = maxv
            self.min_val = minv
        
        
    def create_files(self, indx):
        
        if self.variable_ == 'tas2m':
            var = 'tas_2m'
            
        if self.variable_ == 'prsfc':
            var = 'pr_sfc'
            
        if self.variable_ == 'prsfc_anom' or self.variable_ == 'tas2m_anom':
            var = 'anom'
            
        self.coord_data = xr.open_dataset(
            '/glade/scratch/molina/s2s/CESM2_OBS/ml_coords.nc').expand_dims(
            'sample')
        
        self.img_train = xr.open_mfdataset(self.list_of_cesm[indx], 
                                           concat_dim='sample', 
                                           combine='nested')[var]
        
        self.img_label = xr.open_mfdataset(self.list_of_era5[indx], 
                                           concat_dim='sample', 
                                           combine='nested')[var]
        
        
    def box_cutter(self, ds1, ds2):

        range_y = np.arange(-90., 59.+1, np.random.choice([1,2]))
        range_x = np.arange(0., 328.+1, np.random.choice([1,2,3,4,5]))

        ax = np.random.choice(range_x, replace=False)
        by = np.random.choice(range_y, replace=False)
        
        ds1 = ds1.sel(y=slice(by, by + 31), x=slice(ax, ax + 31))
        ds2 = ds2.sel(y=slice(by, by + 31), x=slice(ax, ax + 31))
        
        return ds1, ds2