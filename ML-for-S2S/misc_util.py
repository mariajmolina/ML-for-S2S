import os
import itertools
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from pylab import *
import torch
from torch.autograd import Variable
import xskillscore as xs
import torch_funcs

"""

Miscellaneous utility functions.

This module contains functions written during the S2S project ideation and research phase.

Author: Maria J. Molina, NCAR

"""

def extract_all_the_members(modelpath, obspath, yrs_evalu, train_perc=0.9, leadopt='weekly', day_init=15, day_end=21, 
                            region=None, mask=None, standardize=False):
    """
    Time selection for weekly subsets.
    
    Args:
        ds: xarray dataset.
        day_init (int): first lead day selection. Defaults to 15.
        day_end (int): last lead day selection. Defaults to 21.
    
    Returns:
        
    
    ::Lead time indices for reference::
    
    Week 1: 1, 2, 3, 4, 5, 6, 7
    Week 2: 8, 9, 10, 11, 12, 13, 14
    Week 3: 15, 16, 17, 18, 19, 20, 21
    Week 4: 22, 23, 24, 25, 26, 27, 28
    Week 5: 29, 30, 31, 32, 33, 34, 35
    Week 6: 36, 37, 38, 39, 40, 41, 42
    
    """
    ############################ extract noaa cpc data
    
    ds_ncpc = torch_funcs.process_cpc_testdata(path=obspath, leadopt='weekly', day_init=day_init, day_end=day_end, region=region)
    
    cpcdate = torch_funcs.matlab_to_python_time(ds_ncpc)        # get dates
    
    l_wghts = torch_funcs.compute_lat_weights(ds_ncpc)          # get lat weights for loss func
    l_wghts = torch_funcs.mask_and_flatten_latweights(l_wghts, mask)
    
    ds_ncpc = torch_funcs.mask_and_flatten_data(ds_ncpc, mask)
    
    ############################ extract model ensembles
    
    ds_cesm = torch_funcs.process_ensemble_members(path=modelpath, leadopt=leadopt, day_init=day_init, day_end=day_end, region=region)  
    ds_cesm = torch_funcs.mask_and_flatten_data(ds_cesm, mask)
    
    ############################ remove nans
    
    ds_cesm, ds_ncpc, ds_indx = torch_funcs.remove_any_nans(ds_cesm, ds_ncpc, return_indx=True)
    
    ############################ grab train and evaluation data
    
    t_indx, e_indx = torch_funcs.year_test_split(cpcdate, ds_indx, yrs_evalu)
    ds_cesm_t, ds_cesm_e = torch_funcs.split_into_sets(ds_cesm, t_indx, e_indx)
    ds_ncpc_t, ds_ncpc_e = torch_funcs.split_into_sets(ds_ncpc, t_indx, e_indx)
    cpcdate_t, cpcdate_e = torch_funcs.split_time_sets(cpcdate, t_indx, e_indx)
    
    # checking input and labels match
    assert ds_cesm_t.shape[0] == ds_ncpc_t.shape[0], "training input and label shapes should match!"
    assert ds_cesm_e.shape[0] == ds_ncpc_e.shape[0], "evaluation input and label shapes should match!"
    
    if standardize:
        
        # normalize using only training data
        ds_cesm_t, ds_ncpc_t, MIN, MAX = torch_funcs.dual_norm_minmax(ds_cesm_t, ds_ncpc_t)
        ds_cesm_e = torch_funcs.preset_minmax_compute(ds_cesm_e, MIN, MAX)
        ds_ncpc_e = torch_funcs.preset_minmax_compute(ds_ncpc_e, MIN, MAX)
    
    # remove nans
    ds_cesm_t, ds_ncpc_t, ds_indx = torch_funcs.remove_any_nans(ds_cesm_t, ds_ncpc_t, return_indx=True)

    # now create validation data
    
    t_indx, v_indx = torch_funcs.random_test_split(ds_indx, seed=0, train=train_perc)
    ds_cesm_t, ds_cesm_v = torch_funcs.split_into_sets(ds_cesm_t, t_indx, v_indx)
    ds_ncpc_t, ds_ncpc_v = torch_funcs.split_into_sets(ds_ncpc_t, t_indx, v_indx)
    cpcdate_t, cpcdate_v = torch_funcs.split_time_sets(cpcdate_t, t_indx, v_indx)
    
    assert ds_cesm_v.shape[0] == ds_ncpc_v.shape[0], "validation input and label shapes should match!"
    
    assert ds_cesm_t.shape[0] == cpcdate_t.shape[0], "issue with time array (training)!"
    assert ds_cesm_v.shape[0] == cpcdate_v.shape[0], "issue with time array (validation)!"
    assert ds_cesm_e.shape[0] == cpcdate_e.shape[0], "issue with time array (evaluation)!"
    
    assert ds_ncpc_t.shape[1] == l_wghts.shape[0], "issue with lat weights (training)!"
    assert ds_ncpc_v.shape[1] == l_wghts.shape[0], "issue with lat weights (validation)!"
    
    print(f"training length: {ds_cesm_t.shape[0]}, validation length: {ds_cesm_v.shape[0]}, evaluation length: {ds_cesm_e.shape[0]}")
    
    if standardize:
        
        return ds_cesm_t, ds_cesm_v, ds_cesm_e, ds_ncpc_t, ds_ncpc_v, ds_ncpc_e, cpcdate_t, cpcdate_v, cpcdate_e, l_wghts, MIN, MAX
    
    if not standardize:
        
        return ds_cesm_t, ds_cesm_v, ds_cesm_e, ds_ncpc_t, ds_ncpc_v, ds_ncpc_e, cpcdate_t, cpcdate_v, cpcdate_e, l_wghts
    
    
def extract_single_member(modelpath, obspath, yrs_evalu, train_perc=0.9, member='00', leadopt='weekly', day_init=15, day_end=21, 
                          region=None, mask=None, standardize=False, MIN=None, MAX=None):
    """
    Time selection for weekly subsets.
    
    Args:
        ds: xarray dataset.
        day_init (int): first lead day selection. Defaults to 15.
        day_end (int): last lead day selection. Defaults to 21.
    
    ::Lead time indices for reference::

    Week 1: 1, 2, 3, 4, 5, 6, 7
    Week 2: 8, 9, 10, 11, 12, 13, 14
    Week 3: 15, 16, 17, 18, 19, 20, 21
    Week 4: 22, 23, 24, 25, 26, 27, 28
    Week 5: 29, 30, 31, 32, 33, 34, 35
    Week 6: 36, 37, 38, 39, 40, 41, 42
        
    """
    # noaa cpc data
    
    ds_ncpc_ = torch_funcs.process_single_cpc(path=obspath, leadopt='weekly', day_init=day_init, day_end=day_end, region=region)

    cpcdate = torch_funcs.matlab_to_python_time(ds_ncpc)        # get dates
    
    l_wghts = torch_funcs.compute_lat_weights(ds_ncpc)          # get lat weights for loss func
    l_wghts = torch_funcs.mask_and_flatten_latweights(l_wghts, mask)
    
    ds_ncpc = torch_funcs.mask_and_flatten_data(ds_ncpc, mask)
    
    # each model ensemble
    
    ds_cesm_ = torch_funcs.process_single_member(path=modelpath, member=member, leadopt=leadopt, day_init=day_init, day_end=day_end, region=region)  
    ds_cesm_ = torch_funcs.mask_and_flatten_data(ds_cesm_, mask)
    
    # remove nans
    ds_cesm_, ds_ncpc_, ds_indx_ = torch_funcs.remove_any_nans(ds_cesm_, ds_ncpc_, return_indx=True)
    
    # grab train and evaluation data
    
    t_indx_, e_indx_ = torch_funcs.year_test_split(cpcdates, ds_indx_, yrs_evalu)
    ds_cesm_t_, ds_cesm_e_ = torch_funcs.split_into_sets(ds_cesm_, t_indx_, e_indx_)
    ds_ncpc_t_, ds_ncpc_e_ = torch_funcs.split_into_sets(ds_ncpc_, t_indx_, e_indx_)
    cpcdates_t, cpcdates_e = torch_funcs.split_time_sets(cpcdates, t_indx_, e_indx_)
    
    # checking input and labels match
    assert ds_cesm_t_.shape[0] == ds_ncpc_t_.shape[0], "training input and label shapes should match!"
    assert ds_cesm_e_.shape[0] == ds_ncpc_e_.shape[0], "evaluation input and label shapes should match!"
    
    if standardize:
        
        # normalize ensemble members
        ds_cesm_t_ = torch_funcs.preset_minmax_compute(ds_cesm_t_, MIN, MAX)
        ds_cesm_e_ = torch_funcs.preset_minmax_compute(ds_cesm_e_, MIN, MAX)

        # normalize noaa cpc
        ds_ncpc_t_ = torch_funcs.preset_minmax_compute(ds_ncpc_t_, MIN, MAX)
        ds_ncpc_e_ = torch_funcs.preset_minmax_compute(ds_ncpc_e_, MIN, MAX)
    
    # remove nans
    ds_cesm_t_, ds_ncpc_t_, ds_indx_ = torch_funcs.remove_any_nans(ds_cesm_t_, ds_ncpc_t_, return_indx=True)
    
    # now create validation data
    
    t_indx_, v_indx_ = torch_funcs.random_test_split(ds_indx_, seed=0, train=train_perc)
    ds_cesm_t_, ds_cesm_v_ = torch_funcs.split_into_sets(ds_cesm_t_, t_indx_, v_indx_)
    ds_ncpc_t_, ds_ncpc_v_ = torch_funcs.split_into_sets(ds_ncpc_t_, t_indx_, v_indx_)
    cpcdates_t, cpcdates_v = torch_funcs.split_time_sets(cpcdates_t, t_indx_, v_indx_)
    
    assert ds_cesm_v_.shape[0] == ds_ncpc_v_.shape[0], "validation input and label shapes should match!"
    
    assert ds_cesm_t_.shape[0] == cpcdates_t.shape[0], "issue with time array (training)!"
    assert ds_cesm_v_.shape[0] == cpcdates_v.shape[0], "issue with time array (validation)!"
    assert ds_cesm_e_.shape[0] == cpcdates_e.shape[0], "issue with time array (evaluation)!"
    
    assert ds_ncpc_t_.shape[1] == l_wghts.shape[0], "issue with lat weights (training)!"
    assert ds_ncpc_v_.shape[1] == l_wghts.shape[0], "issue with lat weights (validation)!"
    
    print(f"member: {member}, training length: {ds_cesm_t_.shape[0]}, validation length: {ds_cesm_v_.shape[0]}, evaluation length: {ds_cesm_e_.shape[0]}")
    
    return ds_cesm_t_, ds_cesm_v_, ds_cesm_e_, ds_ncpc_t_, ds_ncpc_v_, ds_ncpc_e_, cpcdates_t, cpcdates_v, cpcdates_e, l_wghts


def train(model, dataloader, weights=None):
    """
    Training function.
    
    Args:
        model (torch): pytorch neural network.
        dataloader (torch): pytorch dataloader.
        weights (boolean): array of latitude weights for custom loss.
    
    """
    model.train()
    
    running_loss = 0.0
    corrcoef_loss = 0.0
    
    for data in dataloader:
        
        img_noisy = data['train'].unsqueeze(dim=1)
        img_noisy = img_noisy.to(device)
        
        img_label = data['test'].unsqueeze(dim=1)
        img_label = img_label.to(device)

        optimizer.zero_grad()
        outputs = model(img_noisy)
        
        try:
            if not weights:
                loss = criterion(outputs, img_label)
                
        except ValueError:
            loss = criterion(outputs, img_label, weights)
                
        closs = torch_funcs.corrcoef(outputs, img_label)
        
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        
        corrcoef_loss += closs.item()
    
    train_loss = running_loss / len(dataloader)
    coef_loss = corrcoef_loss / len(dataloader)
    
    return train_loss, coef_loss


def validate(model, dataloader, epoch, weights=None, MIN=None, MAX=None, MASK=None, NAME=None):
    """
    Validation function.
    
    Args:
        model: pytorch neural network.
        dataloader: pytorch dataloader.
        epoch (int): epoch iteration for figure saving.
        weights (boolean): array of latitude weights for custom loss.
        MIN (ndarray): minimum value array for inverse min max. Defaults to None.
        MAX (ndarray): maximum value array for inverse min max. Defaults to None.
        MASK (ndarray): 2d mask for reconstruction figure. Defaults to None.
        NAME (str): directory and filename for image saving. Defaults to None.
    
    """
    model.eval()
    
    running_loss = 0.0
    corrcoef_loss = 0.0
    
    with torch.no_grad():
        
        for i, data in enumerate(dataloader):

            img_noisy = data['train'].unsqueeze(dim=1)
            img_noisy = img_noisy.to(device)

            img_label = data['test'].unsqueeze(dim=1)
            img_label = img_label.to(device)

            outputs = model(img_noisy)

            try:
                if not weights:
                    loss = criterion(outputs, img_label)

            except ValueError:
                loss = criterion(outputs, img_label, weights)

            closs = torch_funcs.corrcoef(outputs, img_label)
            running_loss += loss.item()
            
            corrcoef_loss += closs.item()

            if i==0:

                img_noisy_revert = img_noisy.cpu().detach().numpy()
                
                if MIN and MAX:
                    img_noisy_revert = torch_funcs.inverse_minmax(np.squeeze(img_noisy_revert), MIN, MAX)
                    
                #img_noisy_revert = torch_funcs.reconstruct_grid(MASK, np.squeeze(img_noisy_revert))
                #torch_funcs.save_decoded_image(img_noisy_revert, name=NAME+'noisy_{}.png'.format(epoch + 1))

                outputs_revert = outputs.cpu().detach().numpy()
                
                if MIN and MAX:
                    outputs_revert = torch_funcs.inverse_minmax(np.squeeze(outputs_revert), MIN, MAX)
                    
                outputs_revert = torch_funcs.reconstruct_grid(MASK, np.squeeze(outputs_revert))
                torch_funcs.save_decoded_image(outputs_revert, name=NAME+'denoised_{}.png'.format(epoch + 1))

                img_label_revert = img_label.cpu().detach().numpy()
                
                if MIN and MAX:
                    img_label_revert = torch_funcs.inverse_minmax(np.squeeze(img_label_revert), MIN, MAX)
                    
                img_label_revert = torch_funcs.reconstruct_grid(MASK, np.squeeze(img_label_revert))
                torch_funcs.save_decoded_image(img_label_revert, name=NAME+'label_{}.png'.format(epoch + 1))
        
    val_loss = running_loss / len(dataloader)
    coef_loss = corrcoef_loss / len(dataloader)
    
    return val_loss, coef_loss
