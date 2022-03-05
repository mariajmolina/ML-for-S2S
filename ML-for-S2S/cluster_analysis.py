import numpy as np


def single_clusters(ds_, kmeans, pca=None, use_pca=False):
    """
    Return nearest k-means cluster for each ensemble member.
    
    Args:
        ds_: preprocessed array containing ensemble members.
        kmeans: trained k-means object.
    """
    d00_ = np.zeros((1,544,42))

    for nl in range(1,43):

        ds_train = ds_.sel(lead=nl).values
        if use_pca:
            ds_train = pca.transform(ds_train)
        d00_[0,:,nl-1] = kmeans.predict(ds_train)

    return d00_


def ensemble_clusters(ds_, kmeans, pca=None, use_pca=False):
    """
    Return nearest k-means cluster for each ensemble member.
    
    Args:
        ds_: preprocessed array containing ensemble members.
        kmeans: trained k-means object.
    """
    d00_ = np.zeros((544,42))
    d01_ = np.zeros((544,42))
    d02_ = np.zeros((544,42))
    d03_ = np.zeros((544,42))
    d04_ = np.zeros((544,42))
    d05_ = np.zeros((544,42))
    d06_ = np.zeros((544,42))
    d07_ = np.zeros((544,42))
    d08_ = np.zeros((544,42))
    d09_ = np.zeros((544,42))
    d10_ = np.zeros((544,42))

    for nl in range(1,43):

        ds_train = ds_.sel(ensemble= 0, lead=nl).values
        if use_pca:
            ds_train = pca.transform(ds_train)
        d00_[:,nl-1] = kmeans.predict(ds_train)
        
        ds_train = ds_.sel(ensemble= 1, lead=nl).values
        if use_pca:
            ds_train = pca.transform(ds_train)
        d01_[:,nl-1] = kmeans.predict(ds_train)
        
        ds_train = ds_.sel(ensemble= 2, lead=nl).values
        if use_pca:
            ds_train = pca.transform(ds_train)
        d02_[:,nl-1] = kmeans.predict(ds_train)
        
        ds_train = ds_.sel(ensemble= 3, lead=nl).values
        if use_pca:
            ds_train = pca.transform(ds_train)
        d03_[:,nl-1] = kmeans.predict(ds_train)
        
        ds_train = ds_.sel(ensemble= 4, lead=nl).values
        if use_pca:
            ds_train = pca.transform(ds_train)
        d04_[:,nl-1] = kmeans.predict(ds_train)
        
        ds_train = ds_.sel(ensemble= 5, lead=nl).values
        if use_pca:
            ds_train = pca.transform(ds_train)
        d05_[:,nl-1] = kmeans.predict(ds_train)
        
        ds_train = ds_.sel(ensemble= 6, lead=nl).values
        if use_pca:
            ds_train = pca.transform(ds_train)
        d06_[:,nl-1] = kmeans.predict(ds_train)
        
        ds_train = ds_.sel(ensemble= 7, lead=nl).values
        if use_pca:
            ds_train = pca.transform(ds_train)
        d07_[:,nl-1] = kmeans.predict(ds_train)
        
        ds_train = ds_.sel(ensemble= 8, lead=nl).values
        if use_pca:
            ds_train = pca.transform(ds_train)
        d08_[:,nl-1] = kmeans.predict(ds_train)
        
        ds_train = ds_.sel(ensemble= 9, lead=nl).values
        if use_pca:
            ds_train = pca.transform(ds_train)
        d09_[:,nl-1] = kmeans.predict(ds_train)
        
        ds_train = ds_.sel(ensemble=10, lead=nl).values
        if use_pca:
            ds_train = pca.transform(ds_train)
        d10_[:,nl-1] = kmeans.predict(ds_train)

    return np.stack([d00_,d01_,d02_,d03_,d04_,d05_,d06_,d07_,d08_,d09_,d10_])


def composite_clusters(ds_, kmeans, pca=None, use_pca=False):
    """
    Return composite mean for each cluster.
    
    Args:
        ds_: preprocessed array containing ensemble members.
        kmeans: trained k-means object.
    """
    if use_pca:
        ds_train = pca.transform(ds_.values)
        
    if not use_pca:
        ds_train = ds_.values
    
    labs_ = kmeans.predict(ds_train)
    
    c_01 = ds_[np.argwhere(labs_==0)[:,0], :].unstack('flat')
    c_02 = ds_[np.argwhere(labs_==1)[:,0], :].unstack('flat')
    c_03 = ds_[np.argwhere(labs_==2)[:,0], :].unstack('flat')
    c_04 = ds_[np.argwhere(labs_==3)[:,0], :].unstack('flat')
    
    return c_01, c_02, c_03, c_04


def composite_clusters_indx(ds_, kmeans, pca=None, use_pca=False):
    """
    Return composite mean for each cluster.
    
    Args:
        ds_: preprocessed array containing ensemble members.
        kmeans: trained k-means object.
    """
    if use_pca:
        ds_train = pca.transform(ds_.values)
        
    if not use_pca:
        ds_train = ds_.values
    
    labs_ = kmeans.predict(ds_train)
    
    c_01 = np.argwhere(labs_==0)[:,0]
    c_02 = np.argwhere(labs_==1)[:,0]
    c_03 = np.argwhere(labs_==2)[:,0]
    c_04 = np.argwhere(labs_==3)[:,0]
    
    return c_01, c_02, c_03, c_04


def cluster_percentages(array_, lead_time=None):
    """
    np.arange( 0,14,1)
    np.arange(13,27,1)
    np.arange(27,41,1)
    """
    if lead_time is not None:
        
        array_ = array_[:,:,lead_time]
    
    unique_ = {}; counts_ = {}
    
    for i in range(array_.shape[0]):
    
        unique_[i], counts_[i] = np.unique(array_[i,:,:], return_counts=True)
        counts_[i] = counts_[i]/np.sum(counts_[i])
    
    if array_.shape[0] == 1:
        
        return unique_[0], counts_[0]
    
    if array_.shape[0] > 1:
        
        stds_ = {}
        
        for r_ in range(len(counts_[0])):
            
            new_std = []
            
            for d_ in counts_.values():
                new_std.append(d_[r_])
                
            stds_[r_] = np.std(new_std)
        
        return unique_, counts_, np.asarray(list(stds_.values()))
    
