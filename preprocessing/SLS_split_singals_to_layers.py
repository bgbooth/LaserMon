#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Split signals into individual layers

Created on Fri Aug 30 13:49:53 2024

@author: bbooth
"""

import h5py
import numpy as np
import matplotlib.pyplot as plt


def load_signals(f_loc):
    
    with h5py.File(f_loc, 'r') as f:
        diode_filt = np.array(list(f['diode_filtered']))
        diode_non = np.array(list(f['diode_unfiltered']))
        lstat = np.array(list(f['laser_status']))
    
    M = np.zeros([diode_filt.shape[1], 3])
    M[:,0] = diode_filt
    M[:,1] = diode_non
    M[:,2] = lstat
    
    return M


def find_on_off_times(lstat):
    
    eid = np.nonzero((1-lstat[1:]) * lstat[:-1])[0]
    sid = np.nonzero(lstat[1:] * (1-lstat[:-1]))[0]+1
    eid = eid[1:]
    
    # Compute the time each laser is on
    dt = 1e-5 * (eid - sid + 1)
    
    return sid, eid, dt


def find_layer_splits(sigs):
    
    sid, eid, dt = find_on_off_times(sigs[:,2])
    
    idx = np.where(sid[1:] - eid[:-1] > 1e6)[0]
    #print(idx.shape)
    
    # plt.plot(sid[1:] - eid[:-1])
    # plt.show()
    # plt.clf()

    # print(eid[idx[0]-1])
    # print(eid[idx[0]])
    # print(eid[idx[0]+1])
    
    # print(sid[idx[0]-1])
    # print(sid[idx[0]])
    # print(sid[idx[0]+1])
    
    return eid[idx] + 100


def save_sig_layer(sigs, layerID):
    
    out_loc = '/scratch/bbooth/P3AI/signals/layer_%02d.h5'
    
    #plt.plot(sigs[:,2])
    #plt.show()
    #plt.clf()
    
    hf = h5py.File(out_loc % layerID, 'w')
    hf.create_dataset('diode_filtered', data=sigs[:,0])
    hf.create_dataset('diode_unfiltered', data=sigs[:,1])
    hf.create_dataset('laser_status', data=sigs[:,2])
    hf.close()


def main():
    
    signal_loc = '/scratch/bbooth/P3AI/signals/run1.h5'
    
    M = load_signals(signal_loc)
    
    layer_eid = find_layer_splits(M)
    
    for ii in range(len(layer_eid)):
        if ii == 0:
            layerII_sig = M[0:layer_eid[ii],:]    
        else:
            layerII_sig = M[layer_eid[ii-1]:layer_eid[ii],:]
        save_sig_layer(layerII_sig, ii)
        plt.plot(layerII_sig[:,2])
        plt.show()
        plt.clf()
        print(ii)


if __name__ == '__main__':
    main()