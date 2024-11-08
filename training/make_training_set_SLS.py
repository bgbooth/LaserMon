#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Make training set for SLS data

Created on Mon Sep 23 10:05:19 2024

@author: bbooth
"""
import h5py
import os.path
import numpy as np
import matplotlib.pyplot as plt

def load_layer(layerID):
    
    f_loc = '/scratch/bbooth/P3AI/signals/final_layer_%02d.h5'
    #print(f_loc % layerID)
    
    with h5py.File(f_loc % layerID, 'r') as f:
        diode_filt = np.array(list(f['diode_filtered']))
        diode_non = np.array(list(f['diode_unfiltered']))
        lstat = np.array(list(f['laser_status']))
        lx = np.array(list(f['laser_x']))
        ly = np.array(list(f['laser_y']))
        px = np.array(list(f['pixel_x']))
        py = np.array(list(f['pixel_y']))
        idx = np.array(list(f['img_idx']))
        vid = np.array(list(f['images']))
        f.close()
    
    #print(diode_filt.shape)
    M = np.zeros([diode_filt.shape[0], 8])
    M[:,0] = diode_filt
    M[:,1] = diode_non
    M[:,2] = lstat
    M[:,3] = lx
    M[:,4] = ly
    M[:,5] = px
    M[:,6] = py
    M[:,7] = idx
    
    return M, vid


def make_samples(sigs, vid, nSamples):
    
    inputs = np.zeros([nSamples, 100])
    aux_in = np.zeros([nSamples, 3])
    outputs = np.zeros([nSamples, 2])
    
    # Find the range within which we should sample
    sidx = np.min(np.where(sigs[:,2] > 0)[0]) + 49
    eidx = np.max(np.where(sigs[:,2] > 0)[0])

    pts = np.linspace(sidx, eidx, nSamples, dtype=np.int32)
    #print(pts[0:25])
    for ii in range(len(pts)):
        #print(pts[ii])
        #print(sigs[pts[ii]-19:pts[ii]+1,0])
        inputs[ii, 0:50] = sigs[pts[ii]-49:pts[ii]+1,0]
        inputs[ii, 50:100] = sigs[pts[ii]-49:pts[ii]+1,1]
        #inputs[ii,100] = 1
        
        #print(np.floor(sigs[ii,7]))
        idx0 = int(np.floor(sigs[pts[ii], 7]))
        idx1 = int(np.ceil(sigs[pts[ii], 7]))
        frac = sigs[pts[ii], 7] - idx0
        ii_img = frac * vid[:,:,idx0] + (1-frac) * vid[:,:,idx1]
        
        xx = int(sigs[pts[ii], 6])
        yy = int(sigs[pts[ii], 5])
        #print(xx)
        #print(yy)
        
        aux_in[ii,0] = vid[xx,yy,idx0]
        aux_in[ii,1] = (vid[xx,yy,idx1] - vid[xx,yy,idx0]) 
        aux_in[ii,2] = int(80 * frac)
        
        outputs[ii,0] = ii_img[xx,yy]
        outputs[ii,1] = (ii_img[xx,yy] - vid[xx,yy,idx0]) / np.max(np.array([frac, 1e-8]))
    
    return inputs, aux_in, outputs


def save_dataset(inputs, aux_in, outputs, f_loc):
    
    hf = h5py.File(f_loc, 'w')
    hf.create_dataset('inputs', data=inputs)
    hf.create_dataset('aux_inputs', data=aux_in)
    hf.create_dataset('outputs', data=outputs)
    hf.close()


def main():
    
    f_loc = '/scratch/bbooth/P3AI/signals/final_layer_%02d.h5'
    
    inputs = np.zeros([0, 100])
    aux_in = np.zeros([0, 3])
    outputs = np.zeros([0, 2])
    
    for ii in range(1,45,3):
        if os.path.isfile(f_loc % ii):
            sigs, vid = load_layer(ii)
            
            # print(vid.shape)
            # plt.hist(sigs[:,6])
            # plt.show()
            # return
            
            layer_in, layer_aux, layer_out = make_samples(sigs, vid, 5000)
            inputs = np.concatenate((inputs, layer_in), axis=0)
            aux_in = np.concatenate((aux_in, layer_aux), axis=0)
            outputs = np.concatenate((outputs, layer_out), axis=0)
            print(ii)
    
    out_loc = '/scratch/bbooth/P3AI/training_set.h5'
    
    save_dataset(inputs, aux_in, outputs, out_loc)
    

if __name__ == '__main__':
    main()