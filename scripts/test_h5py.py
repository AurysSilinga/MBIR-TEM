# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 17:35:37 2013

@author: Jan
"""


import h5py
import numpy as np


with h5py.File('testfile.hdf5', 'w') as f:
    dset = f.create_dataset('testdataset', (100,), dtype='i')
    print 'dset.shape:', dset.shape
    print 'dset.dtype:', dset.dtype
    dset[...] = np.arange(100)
    print 'dset[0]:', dset[0]
    print 'dset[10]:', dset[10]
    print 'dset[0:100:10]:', dset[0:100:10]
    print 'dset.name:', dset.name
    print 'f.name:', f.name
    grp = f.create_group('subgroup')
    dset_big = grp.create_dataset('another_set', (1000, 1000, 1000), dtype='f')
    for i in range(dset_big.shape[0]):
        print 'i:', i
        dset_big[i, ...] = np.ones(dset_big.shape[1:])
