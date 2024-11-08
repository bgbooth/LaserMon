#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Split job file into layers

Created on Tue Sep  3 15:40:21 2024

@author: bbooth
"""
import copy
import h5py
import numpy as np
import matplotlib.pyplot as plt

def is_odd(num):
    return num & 0x1


def get_scan_lines():
    
    in_file = '/scratch/bbooth/P3AI/job_file/RS+10np-10.hdf5'
    with h5py.File(in_file, 'r') as f:
        #f.visit(print)
        speeds = f['Speeds']
        powers = f['Powers']
        types = f['Types']
        points = f['Points']
        layer_nums = f['Heights']

        n_lines = len(layer_nums)
        lines = np.zeros([n_lines, 9])
        for ii in range(n_lines):
            lines[ii,0] = points[ii][0][0]
            lines[ii,1] = points[ii][0][1]
            lines[ii,2] = layer_nums[ii]
            lines[ii,3] = points[ii][1][0]
            lines[ii,4] = points[ii][1][1]
            lines[ii,5] = layer_nums[ii]
            lines[ii,6] = speeds[ii]
            lines[ii,7] = powers[ii]
            lines[ii,8] = types[ii]
    
    layer_heights = np.unique(lines[:,2])
    new_lines = copy.copy(lines)
    for ii in range(len(layer_heights)):
        idx = np.where(lines[:,2] == layer_heights[ii])[0]
        new_lines[idx,5] = ii
        new_lines[idx,2] = ii
        
    return new_lines


def split_border_bulk(lines):
    
    bulk_idx = np.zeros([6,2], dtype=np.int32)
    bulk_idx[:,0] = np.nonzero((lines[1:,8] == 1) * (lines[:-1,8] == 2))[0]+1
    bulk_idx[:,1] = np.nonzero((lines[1:,8] == 2) * (lines[:-1,8] == 1))[0]
    
    border_idx = np.zeros([6,2], dtype=np.int32)
    for ii in range(6):
        if ii > 0:
            border_idx[ii,0] = bulk_idx[ii-1,1]+1
        border_idx[ii,1] = bulk_idx[ii,0]-1
    
    return border_idx, bulk_idx


def get_object_idx(lines, objectID, flip_order=False):
    
    x_min = np.array([280, 280, 240, 240, 210, 210])
    x_max = np.array([310, 310, 280, 280, 240, 240])
    y_min = np.array([230, 260, 230, 260, 230, 260])
    y_max = np.array([260, 290, 260, 290, 260, 290])
    
    fid = np.array([4, 5, 2, 3, 0, 1])
    
    if flip_order:
        objectID = fid[objectID]
    
    obj_idx = np.where( (lines[:,0] > x_min[objectID]) * (lines[:,0] < x_max[objectID]) *
                        (lines[:,1] > y_min[objectID]) * (lines[:,1] < y_max[objectID]) )[0]

    obj_lines = lines[obj_idx,:]
    
    bulk_obj_idx = np.where(obj_lines[:,8] == 1)[0]
    bulk_idx = obj_idx[bulk_obj_idx]
    
    border_obj_idx = np.where(obj_lines[:,8] == 2)[0]
    border_idx = obj_idx[border_obj_idx]
    
    return bulk_idx, border_idx
    

def reorganize_layer(lines, flip_order):
    
    lines_fixed = np.zeros([0,9])
    
    for ii in range(6):
        bulk_idx, border_idx = get_object_idx(lines, ii, flip_order)
        border_lines = lines[border_idx,:]
        bulk_lines = lines[bulk_idx,:]
        object_lines = np.concatenate((border_lines, bulk_lines, border_lines), axis=0)
        lines_fixed = np.concatenate((lines_fixed, object_lines), axis=0)
        
    return lines_fixed


def save_layer(lines, layerID):
    out_loc = '/scratch/bbooth/P3AI/signals/scan_lines_layer_%02d.h5'
    
    hf = h5py.File(out_loc % layerID, 'w')
    hf.create_dataset('sid_x', data=lines[:,0])
    hf.create_dataset('sid_y', data=lines[:,1])
    hf.create_dataset('eid_x', data=lines[:,3])
    hf.create_dataset('eid_y', data=lines[:,4])
    hf.create_dataset('speed', data=lines[:,6])
    hf.create_dataset('is_border', data=lines[:,8]-1)
    hf.close()
    

def main():
    
    lines = get_scan_lines()
    #print(np.unique(lines[:,6]))
    
    for ii in range(45):
        idx = np.where(lines[:,5] == ii)[0]
    
        layerII_lines = lines[idx,:]
    
        #border_idx, bulk_idx = split_border_bulk(layerII_lines)
        
        layerII_lines_fixed = reorganize_layer(layerII_lines, is_odd(ii))
        
        # plt.plot(layerII_lines_fixed[:,8])
        # plt.show()
        # plt.clf()
        
        save_layer(layerII_lines_fixed, ii)
        print(ii)
        #break


if __name__ == '__main__':
    main()