#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Group training and testing datasets

Created on Tue Oct 15 10:25:46 2024

@author: bbooth
"""
import h5py
import numpy as np

def load_layer(cylinderID, layerID):
    in_loc = '/scratch/bbooth/FAIR/datasets/Cyl%d_layer%03d.h5' % (cylinderID, layerID)

    fp = h5py.File(in_loc, 'r')
    #fp.visit(print)
    #vid = np.array(list(fp['frames']))
    inputs = np.array(list(fp['inputs']))
    aux_in = np.array(list(fp['aux_inputs']))
    outputs = np.array(list(fp['outputs']))
    coords = np.array(list(fp['coords']))
    fp.close()
    
    return inputs, aux_in, outputs, coords 


def save_layer(inputs, aux_in, outputs, coords, out_loc):

    fpOUT = h5py.File(out_loc, 'w')
    fpOUT.create_dataset('inputs', data=inputs)
    fpOUT.create_dataset('aux_inputs', data=aux_in)
    fpOUT.create_dataset('outputs', data=outputs)
    fpOUT.create_dataset('coords', data=coords)
    fpOUT.close()
    
    
def make_datasets(layers, objectID, out_loc):
    
    all_inputs = np.zeros([0,40])
    all_aux = np.zeros([0,6])
    all_outputs = np.zeros([0,3])
    all_coords = np.zeros([0,3])
    
    for ii in range(len(layers)):
        inputs, aux_in, outputs, coords = load_layer(objectID, layers[ii])
        all_inputs = np.concatenate((all_inputs, inputs), axis=0)
        all_aux = np.concatenate((all_aux, aux_in), axis=0)
        all_outputs = np.concatenate((all_outputs, outputs), axis=0)
        all_coords = np.concatenate((all_coords, coords), axis=0)
        
    save_layer(all_inputs, all_aux, all_outputs, all_coords, out_loc)
    print(out_loc)
    
    
def main():
    
    train_layers = np.array([60, 61, 62, 90, 99, 100, 101, 120, 150, 180, 210, 240])
    test_layers = np.array([270, 300, 330, 349, 350, 351, 360, 390, 420, 450, 470, 471, 472])
    
    make_datasets(train_layers, 13, '/scratch/bbooth/FAIR/datasets/training_data.h5')
    make_datasets(test_layers, 13, '/scratch/bbooth/FAIR/datasets/test_data_A.h5')
    make_datasets(train_layers, 14, '/scratch/bbooth/FAIR/datasets/test_data_B.h5')
    make_datasets(test_layers, 14, '/scratch/bbooth/FAIR/datasets/test_data_C.h5')


if __name__ == '__main__':
    main()