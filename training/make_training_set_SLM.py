#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Make the SLM training set

Created on Fri Sep 27 14:37:44 2024

@author: bbooth
"""
import h5py
import numpy as np
import matplotlib.pyplot as plt


def load_layer(cylinderID, layerID):
    if cylinderID == 13:
        f_loc = '/scratch/bbooth/FAIR/Cyl%d/layer%04d.h5' % (cylinderID, layerID)
    else:
        f_loc = '/scratch/bbooth/FAIR/Cyl%d_Processed/layer%04d.h5' % (cylinderID, layerID)

    fp = h5py.File(f_loc, 'r')
    fp.visit(print)
    #vid = np.array(list(fp['frames']))
    diode1 = np.array(list(fp['photodiode0']))
    diode2 = np.array(list(fp['photodiode1']))
    mapping = np.array(list(fp['mapping']), dtype=np.int32)
    xx = np.array(list(fp['setpoint_x']))
    yy = np.array(list(fp['setpoint_y']))
    lstat = np.array(list(fp['laser_status']))
    features = np.array(list(fp['features']))
    fp.close()
    
    sigs = np.zeros([len(diode1), 5])
    sigs[:,0] = diode1
    sigs[:,1] = diode2
    sigs[:,2] = lstat
    sigs[:,3] = xx
    sigs[:,4] = yy
    
    # fidx = [5, 11, 17]
    # if features.shape[1] > 3:
    #     features = features[:,fidx]
    #     features[:,0] *= 16
    # else:
    #     features[:,0] /= 255
    
    return sigs, mapping, features


def process_layer(sigs, idx, features, layerID):
    
    #print(idx.shape)
    #print(sigs.shape)
    #print(features.shape)
    
    inputs = np.zeros([idx.shape[0]-1, 40])
    aux_in = np.zeros([idx.shape[0]-1, 6])
    outputs = np.zeros([idx.shape[0]-1, 3])
    log_data = np.zeros([idx.shape[0]-1,3])
    
    k=np.ones(1, dtype=np.int32)
    N = 0
    
    for ii in range(1, idx.shape[0]):
        if sigs[idx[ii],2] == 0:
            continue
        
        if idx[ii] < 19:
            inputs[N,19-idx[ii]:20] = sigs[0:idx[ii]+1,0]
            inputs[N,-(idx[ii]+1):] = sigs[0:idx[ii]+1,1]
        else:
            inputs[N,0:20] = sigs[idx[ii]-19:idx[ii]+1,0]
            inputs[N,20:] = sigs[idx[ii]-19:idx[ii]+1,1]
        
        if features.shape[0] == sigs.shape[0]:
            outputs[N,:] = features[idx[ii], :]
            aux_in[N,0:3] = features[idx[ii]-k[0], :]
        else:
            outputs[N,:] = features[ii, :]
            aux_in[N,0:3] = features[ii-k[0], :]
        
        aux_in[N,3] = k[0] * 5
        k = np.maximum((k+1) % 5, np.ones(1, dtype=np.int32))
        
        u = np.array([sigs[idx[ii],3], sigs[idx[ii], 4]])
        d = np.linalg.norm(u)
        aux_in[N,4] = u[0] / np.maximum(d, 1e-8)
        aux_in[N,5] = u[1] / np.maximum(d, 1e-8)
        
        log_data[N,0] = sigs[idx[ii],3]
        log_data[N,1] = sigs[idx[ii],4]
        log_data[N,2] = layerID
        N += 1
        
        
    return inputs[0:N,:], aux_in[0:N,:], outputs[0:N,:], log_data[0:N,:]
        

def save_layer(inputs, aux_in, outputs, coords, objectID, layerID):
    out_loc = '/scratch/bbooth/FAIR/datasets/Cyl%d_layer%03d.h5' % (objectID, layerID)

    fpOUT = h5py.File(out_loc, 'w')
    fpOUT.create_dataset('inputs', data=inputs)
    fpOUT.create_dataset('aux_inputs', data=aux_in)
    fpOUT.create_dataset('outputs', data=outputs)
    fpOUT.create_dataset('coords', data=coords)
    fpOUT.close()


def main():
    
    train_layers = np.array([60, 61, 62, 90, 99, 100, 101, 120, 150, 180, 210, 240])
    test_layers = np.array([270, 300, 330, 349, 350, 351, 360, 390, 420, 450, 470, 471, 472])
    
    for ii in train_layers:   
        #sigs, idx, features = load_layer(13, ii)
        #inputs, aux_in, outputs, coords = process_layer(sigs, idx, features, ii)
        #save_layer(inputs, aux_in, outputs, coords, 13, ii)
        
        sigs, idx, features = load_layer(14, ii)
        inputs, aux_in, outputs, coords = process_layer(sigs, idx, features, ii)
        save_layer(inputs, aux_in, outputs, coords, 14, ii)
        print(ii)
    
        
    for ii in test_layers:   
        #sigs, idx, features = load_layer(13, ii)
        #inputs, aux_in, outputs, coords = process_layer(sigs, idx, features, ii)
        #save_layer(inputs, aux_in, outputs, coords, 13, ii)
        
        sigs, idx, features = load_layer(14, ii)
        inputs, aux_in, outputs, coords = process_layer(sigs, idx, features, ii)
        save_layer(inputs, aux_in, outputs, coords, 14, ii)
        print(ii)
    

if __name__ =='__main__':
    main()