import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
from pylab import *
import xskillscore as xs
import torch_funcs

"""

Module contains functions for extracting data for training.

Author: Maria J. Molina, NCAR (molina@ucar.edu)

"""

def extract_all_the_members_era5t(
                            model_path, obs_path, evalution_yrs, 
                            variable='tas_2m', train_percent=0.95, lead_option='weekly', 
                            day_init=15, day_end=21, region=None, season=None):
    """
    Data extraction and preprocessing (all cesm members). For CNN.
    
    Args:
        model_path     (str):         directory path to model data.
        obs_path       (str):         directory path to cpc data.
        evaluation_yrs (list of int): years for evaluation set.
        variable       (str):         variable to process. defaults to pr_sfc.
        train_percent  (float):       percent of data for training. defaults to 0.95.
        lead_option    (str):         lead time option. defaults to weekly. other is daily.
        day_init       (int):         initial day index. defaults to 21.
        day_end        (int):         end day index. defaults to 15.
        region         (str):         region for training. defaults to None.
    
    Returns:
        cesm_t  (numpy array): cesm training data.
        cesm_v  (numpy array): cesm validation data.
        cesm_e  (numpy array): cesm evaluation (test) data.
        ncpc_t  (numpy array): noaa cpc training data.
        ncpc_v  (numpy array): noaa cpc validation data.
        ncpc_e  (numpy array): noaa cpc evaluation (test) data.
        date_t  (numpy array): noaa cpc training data dates.
        date_v  (numpy array): noaa cpc validation data dates.
        date_e  (numpy array): noaa cpc evaluation data dates.
        l_wght  (numpy array): latitude weights.
    
    ::Lead time indices for reference::
    
    Week 1:  1,  2,  3,  4,  5,  6,  7
    Week 2:  8,  9, 10, 11, 12, 13, 14
    Week 3: 15, 16, 17, 18, 19, 20, 21
    Week 4: 22, 23, 24, 25, 26, 27, 28
    Week 5: 29, 30, 31, 32, 33, 34, 35
    Week 6: 36, 37, 38, 39, 40, 41, 42
    
    """
    ############################ extract noaa cpc data
    
    if variable == 'tas_2m':
        
        var = 'tas_2m'
        varobs = 'temp'
    
    obs_, date = torch_funcs.process_era5_testdata(obs_path, 'weekly', day_init, day_end, region, season=season)
    
    l_wght = torch_funcs.compute_lat_weights(obs_) # get lat weights for loss func
    l_wght = torch_funcs.visual_data_process_latweights(l_wght)
    
    obs_ = torch_funcs.visual_data_process(obs_)
    
    ############################ extract model ensembles
    
    cesm = torch_funcs.process_ensemble_members(model_path, variable, lead_option, 
                                                day_init, day_end, region, season=season)
    
    cesm = torch_funcs.visual_data_process(cesm)
    ds_indx = torch_funcs.rematch_indices(cesm, obs_)
    
    ############################ grab train and evaluation data
    
    t_indx, e_indx = torch_funcs.year_test_split(pd.to_datetime(date), ds_indx, evalution_yrs)
    
    cesm_t, cesm_e = torch_funcs.split_into_sets(cesm, t_indx, e_indx)
    obs__t, obs__e = torch_funcs.split_into_sets(obs_, t_indx, e_indx)
    
    date_t, date_e = torch_funcs.split_time_sets(pd.to_datetime(date), ds_indx, t_indx, e_indx)
    
    ############################ grab new indices (ds_indx; there are no more nans)
    
    ds_indx = torch_funcs.rematch_indices(cesm_t, obs__t)
    
    ############################ now create validation data
    
    t_indx, v_indx = torch_funcs.random_test_split(ds_indx, seed=0, train=train_percent)
    cesm_t, cesm_v = torch_funcs.split_into_sets(cesm_t, t_indx, v_indx)
    obs__t, obs__v = torch_funcs.split_into_sets(obs__t, t_indx, v_indx)
    date_t, date_v = torch_funcs.split_time_sets(date_t, ds_indx, t_indx, v_indx)
    
    ############################ check data sizes match
    
    assert cesm_t.shape[0] == obs__t.shape[0], "training input and label shapes should match!"
    assert cesm_e.shape[0] == obs__e.shape[0], "evaluation input and label shapes should match!"
    assert cesm_v.shape[0] == obs__v.shape[0], "validation input and label shapes should match!"
    assert cesm_t.shape[0] == date_t.shape[0], "issue with time array (training)!"
    assert cesm_v.shape[0] == date_v.shape[0], "issue with time array (validation)!"
    assert cesm_e.shape[0] == date_e.shape[0], "issue with time array (evaluation)!"
    assert obs__t.shape[1] == l_wght.shape[0], "issue with lat weights (training)!"
    assert obs__v.shape[1] == l_wght.shape[0], "issue with lat weights (validation)!"
    
    print(f"training length: {cesm_t.shape[0]}, validation length: {cesm_v.shape[0]}, evaluation length: {cesm_e.shape[0]}")

    ############################ return data
    
    return cesm_t, cesm_v, cesm_e, obs__t, obs__v, obs__e, date_t, date_v, date_e, l_wght


def extract_all_the_members_wgpcp(
                            model_path, obs_path, evalution_yrs, 
                            variable='pr_sfc', train_percent=0.95, lead_option='weekly', 
                            day_init=15, day_end=21, region=None, season=None):
    """
    Data extraction and preprocessing (all cesm members). For CNN.
    
    Args:
        model_path     (str):         directory path to model data.
        obs_path       (str):         directory path to cpc data.
        evaluation_yrs (list of int): years for evaluation set.
        variable       (str):         variable to process. defaults to pr_sfc.
        train_percent  (float):       percent of data for training. defaults to 0.95.
        lead_option    (str):         lead time option. defaults to weekly. other is daily.
        day_init       (int):         initial day index. defaults to 21.
        day_end        (int):         end day index. defaults to 15.
        region         (str):         region for training. defaults to None.
    
    Returns:
        cesm_t  (numpy array): cesm training data.
        cesm_v  (numpy array): cesm validation data.
        cesm_e  (numpy array): cesm evaluation (test) data.
        ncpc_t  (numpy array): noaa cpc training data.
        ncpc_v  (numpy array): noaa cpc validation data.
        ncpc_e  (numpy array): noaa cpc evaluation (test) data.
        date_t  (numpy array): noaa cpc training data dates.
        date_v  (numpy array): noaa cpc validation data dates.
        date_e  (numpy array): noaa cpc evaluation data dates.
        l_wght  (numpy array): latitude weights.
    
    ::Lead time indices for reference::
    
    Week 1:  1,  2,  3,  4,  5,  6,  7
    Week 2:  8,  9, 10, 11, 12, 13, 14
    Week 3: 15, 16, 17, 18, 19, 20, 21
    Week 4: 22, 23, 24, 25, 26, 27, 28
    Week 5: 29, 30, 31, 32, 33, 34, 35
    Week 6: 36, 37, 38, 39, 40, 41, 42
    
    """
    ############################ extract noaa cpc data
    
    if variable == 'pr_sfc':
        
        var = 'pr'
        varobs = 'precip'
    
    gpcp, date = torch_funcs.process_gpcp_testdata(obs_path, 'weekly', day_init, day_end, region, season=season)
    
    gpcp = torch_funcs.visual_data_process(gpcp)
    
    ############################ extract model ensembles
    
    cesm = torch_funcs.process_ensemble_members(model_path, variable, lead_option, 
                                                day_init, day_end, region, season=season)
    
    l_wght = torch_funcs.compute_lat_weights(cesm) # get lat weights for loss func
    l_wght = torch_funcs.visual_data_process_latweights(l_wght)
    
    cesm = torch_funcs.visual_data_process(cesm)
    ds_indx = torch_funcs.rematch_indices(cesm, gpcp)
    
    ############################ grab train and evaluation data
    
    t_indx, e_indx = torch_funcs.year_test_split(pd.to_datetime(date), ds_indx, evalution_yrs)
    
    cesm_t, cesm_e = torch_funcs.split_into_sets(cesm, t_indx, e_indx)
    gpcp_t, gpcp_e = torch_funcs.split_into_sets(gpcp, t_indx, e_indx)
    
    date_t, date_e = torch_funcs.split_time_sets(pd.to_datetime(date), ds_indx, t_indx, e_indx)
    
    ############################ grab new indices (ds_indx; there are no more nans)
    
    ds_indx = torch_funcs.rematch_indices(cesm_t, gpcp_t)
    
    ############################ now create validation data
    
    t_indx, v_indx = torch_funcs.random_test_split(ds_indx, seed=0, train=train_percent)
    cesm_t, cesm_v = torch_funcs.split_into_sets(cesm_t, t_indx, v_indx)
    gpcp_t, gpcp_v = torch_funcs.split_into_sets(gpcp_t, t_indx, v_indx)
    date_t, date_v = torch_funcs.split_time_sets(date_t, ds_indx, t_indx, v_indx)
    
    ############################ check data sizes match
    
    assert cesm_t.shape[0] == gpcp_t.shape[0], "training input and label shapes should match!"
    assert cesm_e.shape[0] == gpcp_e.shape[0], "evaluation input and label shapes should match!"
    assert cesm_v.shape[0] == gpcp_v.shape[0], "validation input and label shapes should match!"
    assert cesm_t.shape[0] == date_t.shape[0], "issue with time array (training)!"
    assert cesm_v.shape[0] == date_v.shape[0], "issue with time array (validation)!"
    assert cesm_e.shape[0] == date_e.shape[0], "issue with time array (evaluation)!"
    assert cesm_t.shape[1] == l_wght.shape[0], "issue with lat weights (training)!"
    assert cesm_v.shape[1] == l_wght.shape[0], "issue with lat weights (validation)!"
    
    print(f"training length: {cesm_t.shape[0]}, validation length: {cesm_v.shape[0]}, evaluation length: {cesm_e.shape[0]}")

    ############################ return data
    
    return cesm_t, cesm_v, cesm_e, gpcp_t, gpcp_v, gpcp_e, date_t, date_v, date_e, l_wght


def extract_all_the_members(model_path, obs_path, evalution_yrs, 
                            variable='pr_sfc', train_percent=0.95, lead_option='weekly', 
                            day_init=15, day_end=21, region=None, mask=None, ocean=False):
    """
    Data extraction and preprocessing (all cesm members). For ANN.
    
    Args:
        model_path     (str):         directory path to model data.
        obs_path       (str):         directory path to cpc data.
        evaluation_yrs (list of int): years for evaluation set.
        variable       (str):         variable to process. defaults to pr_sfc.
        train_percent  (float):       percent of data for training. defaults to 0.95.
        lead_option    (str):         lead time option. defaults to weekly. other is daily.
        day_init       (int):         initial day index. defaults to 21.
        day_end        (int):         end day index. defaults to 15.
        region         (str):         region for training. defaults to None.
        mask           (array):       land mask. defaults to None.
        ocean          (boolean):     include ocean points in input data. defaults to False.
    
    Returns:
        cesm_t  (numpy array): cesm training data.
        cesm_v  (numpy array): cesm validation data.
        cesm_e  (numpy array): cesm evaluation (test) data.
        ncpc_t  (numpy array): noaa cpc training data.
        ncpc_v  (numpy array): noaa cpc validation data.
        ncpc_e  (numpy array): noaa cpc evaluation (test) data.
        date_t  (numpy array): noaa cpc training data dates.
        date_v  (numpy array): noaa cpc validation data dates.
        date_e  (numpy array): noaa cpc evaluation data dates.
        l_wght  (numpy array): latitude weights.
    
    ::Lead time indices for reference::
    
    Week 1:  1,  2,  3,  4,  5,  6,  7
    Week 2:  8,  9, 10, 11, 12, 13, 14
    Week 3: 15, 16, 17, 18, 19, 20, 21
    Week 4: 22, 23, 24, 25, 26, 27, 28
    Week 5: 29, 30, 31, 32, 33, 34, 35
    Week 6: 36, 37, 38, 39, 40, 41, 42
    
    """
    ############################ extract noaa cpc data
    
    if variable == 'pr_sfc':
        
        var = 'pr'
    
    ncpc = torch_funcs.process_cpc_testdata(obs_path, var, 'weekly', day_init, day_end, region)
    
    date = torch_funcs.matlab_to_python_time(ncpc) # get dates
    
    l_wght = torch_funcs.compute_lat_weights(ncpc) # get lat weights for loss func
    l_wght = torch_funcs.mask_and_flatten_latweights(l_wght, mask)
    
    ncpc = torch_funcs.mask_and_flatten_data(ncpc, mask)
    
    ############################ extract model ensembles
    
    cesm = torch_funcs.process_ensemble_members(model_path, variable, lead_option, 
                                                day_init, day_end, region)  
    
    if not ocean:
        
        cesm = torch_funcs.mask_and_flatten_data(cesm, mask)
        
    if ocean:
        
        cesm = torch_funcs.only_flatten_data(cesm)
    
    ############################ remove nans
    
    cesm, ncpc, ds_indx = torch_funcs.remove_any_nans(cesm, ncpc, return_indx=True)
    
    ############################ grab train and evaluation data
    
    t_indx, e_indx = torch_funcs.year_test_split(date, ds_indx, evalution_yrs)
    cesm_t, cesm_e = torch_funcs.split_into_sets(cesm, t_indx, e_indx)
    ncpc_t, ncpc_e = torch_funcs.split_into_sets(ncpc, t_indx, e_indx)
    date_t, date_e = torch_funcs.split_time_sets(date, ds_indx, t_indx, e_indx)
    
    ############################ grab new indices (ds_indx; there are no more nans)
    
    cesm_t, ncpc_t, ds_indx = torch_funcs.remove_any_nans(cesm_t, ncpc_t, return_indx=True)
    
    ############################ now create validation data
    
    t_indx, v_indx = torch_funcs.random_test_split(ds_indx, seed=0, train=train_percent)
    cesm_t, cesm_v = torch_funcs.split_into_sets(cesm_t, t_indx, v_indx)
    ncpc_t, ncpc_v = torch_funcs.split_into_sets(ncpc_t, t_indx, v_indx)
    date_t, date_v = torch_funcs.split_time_sets(date_t, ds_indx, t_indx, v_indx)
    
     ############################ check data sizes match
    
    assert cesm_t.shape[0] == ncpc_t.shape[0], "training input and label shapes should match!"
    assert cesm_e.shape[0] == ncpc_e.shape[0], "evaluation input and label shapes should match!"
    assert cesm_v.shape[0] == ncpc_v.shape[0], "validation input and label shapes should match!"
    assert cesm_t.shape[0] == date_t.shape[0], "issue with time array (training)!"
    assert cesm_v.shape[0] == date_v.shape[0], "issue with time array (validation)!"
    assert cesm_e.shape[0] == date_e.shape[0], "issue with time array (evaluation)!"
    assert ncpc_t.shape[1] == l_wght.shape[0], "issue with lat weights (training)!"
    assert ncpc_v.shape[1] == l_wght.shape[0], "issue with lat weights (validation)!"
    
    print(f"training length: {cesm_t.shape[0]}, validation length: {cesm_v.shape[0]}, evaluation length: {cesm_e.shape[0]}")

    ############################ return data
    
    return cesm_t, cesm_v, cesm_e, ncpc_t, ncpc_v, ncpc_e, date_t, date_v, date_e, l_wght
    
    
def extract_single_member(model_path, obs_path, evaluation_yrs, 
                          variable='pr_sfc', train_percent=0.9, member='00', lead_option='weekly', 
                          day_init=15, day_end=21, region=None, mask=None, ocean=False):
    """
    Data extraction and preprocessing (single cesm members).
    
    Args:
        model_path     (str):         directory path to model data.
        obs_path       (str):         directory path to cpc data.
        evaluation_yrs (list of int): years for evaluation set.
        variable       (str):         variable to process. defaults to pr_sfc.
        train_percent  (float):       percent of data for training. defaults to 0.95.
        member         (str):         ensemble member to process. defaults to `00`.
        lead_option    (str):         lead time option. defaults to weekly. other is daily.
        day_init       (int):         initial day index. defaults to 21.
        day_end        (int):         end day index. defaults to 15.
        region         (str):         region for training. defaults to None.
        mask           (array):       land mask. defaults to None.
        ocean          (boolean):     include ocean points in input data. defaults to False.
    
    Returns:
        cesm_t  (numpy array): cesm training data.
        cesm_v  (numpy array): cesm validation data.
        cesm_e  (numpy array): cesm evaluation (test) data.
        ncpc_t  (numpy array): noaa cpc training data.
        ncpc_v  (numpy array): noaa cpc validation data.
        ncpc_e  (numpy array): noaa cpc evaluation (test) data.
        date_t  (numpy array): noaa cpc training data dates.
        date_v  (numpy array): noaa cpc validation data dates.
        date_e  (numpy array): noaa cpc evaluation data dates.
        l_wght  (numpy array): latitude weights.
    
    ::Lead time indices for reference::
    
    Week 1:  1,  2,  3,  4,  5,  6,  7
    Week 2:  8,  9, 10, 11, 12, 13, 14
    Week 3: 15, 16, 17, 18, 19, 20, 21
    Week 4: 22, 23, 24, 25, 26, 27, 28
    Week 5: 29, 30, 31, 32, 33, 34, 35
    Week 6: 36, 37, 38, 39, 40, 41, 42
        
    """
    ############################ noaa cpc data
    
    ncpc = torch_funcs.process_single_cpc(obs_path, lead_option='weekly', 
                                          day_init=day_init, day_end=day_end, region=region)

    date = torch_funcs.matlab_to_python_time(ncpc) # get dates
    
    l_wght = torch_funcs.compute_lat_weights(ncpc) # get lat weights for loss func
    l_wght = torch_funcs.mask_and_flatten_latweights(l_wght, mask)
    
    ncpc = torch_funcs.mask_and_flatten_data(ncpc, mask)
    
    ############################ each model ensemble
    
    cesm = torch_funcs.process_single_member(model_path, member=member, leadopt=lead_option, 
                                             day_init=day_init, day_end=day_end, region=region)
    
    if not ocean:
        
        cesm = torch_funcs.mask_and_flatten_data(cesm, mask)
    
    ############################ remove nans
    
    cesm, ncpc, ds_indx = torch_funcs.remove_any_nans(cesm, ncpc, return_indx=True)
    
    ############################ grab train and evaluation data
    
    t_indx, e_indx = torch_funcs.year_test_split(date, ds_indx, evaluation_yrs)
    cesm_t, cesm_e = torch_funcs.split_into_sets(cesm, t_indx, e_indx)
    ncpc_t, ncpc_e = torch_funcs.split_into_sets(ncpc, t_indx, e_indx)
    date_t, date_e = torch_funcs.split_time_sets(date, t_indx, e_indx)

    ############################ remove nans
    
    cesm_t, ncpc_t, ds_indx = torch_funcs.remove_any_nans(cesm_t, ncpc_t, return_indx=True)
    
    ############################ now create validation data
    
    t_indx, v_indx = torch_funcs.random_test_split(ds_indx, seed=0, train=train_percent)
    cesm_t, cesm_v = torch_funcs.split_into_sets(cesm_t, t_indx, v_indx)
    ncpc_t, ncpc_v = torch_funcs.split_into_sets(ncpc_t, t_indx, v_indx)
    date_t, date_v = torch_funcs.split_time_sets(date_t, t_indx, v_indx)
    
    ############################
    
    assert cesm_t.shape[0] == ncpc_t.shape[0], "training input and label shapes should match!"
    assert cesm_e.shape[0] == ncpc_e.shape[0], "evaluation input and label shapes should match!"
    assert cesm_v.shape[0] == ncpc_v.shape[0], "validation input and label shapes should match!"
    assert cesm_t.shape[0] == date_t.shape[0], "issue with time array (training)!"
    assert cesm_v.shape[0] == date_v.shape[0], "issue with time array (validation)!"
    assert cesm_e.shape[0] == date_e.shape[0], "issue with time array (evaluation)!"
    assert ncpc_t.shape[1] == l_wght.shape[0], "issue with lat weights (training)!"
    assert ncpc_v.shape[1] == l_wght.shape[0], "issue with lat weights (validation)!"
    
    print(f"member: {member}, training length: {cesm_t.shape[0]}, validation length: {cesm_v.shape[0]}, evaluation length: {cesm_e.shape[0]}")
    
    ############################
    
    return cesm_t, cesm_v, cesm_e, ncpc_t, ncpc_v, ncpc_e, date_t, date_v, date_e, l_wght
