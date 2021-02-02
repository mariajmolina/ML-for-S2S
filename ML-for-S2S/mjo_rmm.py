import xarray as xr
import numpy as np
from eofs.xarray import Eof

def calc_RMM_phase(RMM1, RMM2, amp_thresh=0):
    """
    Given RMM1 and RMM2 indices, calculate MJO phase.
    Provided by Zane K. Martin (CSU).
    
    Args:
        RMM1: EOF for MJO phase space diagram.
        RMM2: EOF for MJO phase space diagram.
        amp_thresh (int): MJO amplitude threshold. Defaults to 0.
    """
    if (RMM1**2 + RMM2**2 > amp_thresh) and (RMM2 < 0) and (RMM2 > RMM1):
        phase = 1
    elif (RMM1**2 + RMM2**2 > amp_thresh) and (RMM1 < 0) and (RMM2 < RMM1):
        phase = 2
    elif (RMM1**2 + RMM2**2 > amp_thresh) and (RMM1 > 0) and (RMM2 < -RMM1):
        phase = 3
    elif (RMM1**2 + RMM2**2 > amp_thresh) and (RMM2 < 0) and (RMM2 > -RMM1):
        phase = 4
    elif (RMM1**2 + RMM2**2 > amp_thresh) and (RMM2 > 0) and (RMM2 < RMM1):
        phase = 5
    elif (RMM1**2 + RMM2**2 > amp_thresh) and (RMM1 > 0) and (RMM2 > RMM1):
        phase = 6
    elif (RMM1**2 + RMM2**2 > amp_thresh) and (RMM1 < 0) and (RMM2 > -RMM1):
        phase = 7
    elif (RMM1**2 + RMM2**2 > amp_thresh) and (RMM2 > 0) and (RMM2 < -RMM1):
        phase = 8  
    else:
        phase = 0
    return(phase)
