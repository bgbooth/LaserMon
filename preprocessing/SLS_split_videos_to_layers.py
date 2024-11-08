#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Split videos into separate layers

Created on Wed Sep  4 11:16:53 2024

@author: bbooth
"""
import cv2
import h5py
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt

def load_video_file(file_id):
    
    imgs_loc = '/scratch/bbooth/P3AI/raw_data/Thermal/Nifti/recording%d.nii'
    
    nii = nib.load(imgs_loc % file_id)
    vol = nii.get_fdata()
    
    cropped_vol = vol[205:253,310:383]
    
    return cropped_vol


def detect_layer_idx(vid, startLayer=False, endLayer=False):
    
    offset = 50
    
    avg_temp = np.mean(vid, axis=(0,1))
    
    #plt.plot(avg_temp)
    #plt.show()
    
    recoat_idx = np.nonzero(np.diff(avg_temp < 150))[0] + 1
    #print(recoat_idx)
    
    layer_idx_start = list()
    layer_idx_end = list()
    
    # Add start layer if exists
    if startLayer:
        layer_idx_start.append(0)
        layer_idx_end.append(400)
    
    for ii in range(len(recoat_idx)-1):
        if recoat_idx[ii+1] - recoat_idx[ii] > 100:
            layer_idx_start.append(recoat_idx[ii] + offset)
            layer_idx_end.append(recoat_idx[ii+1] - offset)

    # Add end layer if exists
    if endLayer:
        layer_idx_end.append(vid.shape[2]-1)
        layer_idx_start.append(vid.shape[2]-400)

    return np.array(layer_idx_start), np.array(layer_idx_end)


def align_frames(vid, tform, imSz):
    
    aligned_frames_array = np.zeros([imSz[0], imSz[1], vid.shape[2]])
    for ii in range(vid.shape[2]):
        aligned_frames_array[:,:,ii] = cv2.warpPerspective(vid[:,:,ii], tform, np.flip(imSz))

    return aligned_frames_array


def save_layer(vol, layerID):
    
    out_loc = '/scratch/bbooth/P3AI/signals/video_layer%02d.nii'

    nii = nib.Nifti1Image(vol, np.eye(4))
    nib.save(nii, out_loc % layerID)
    

def main():
    
    tform = np.load('/scratch/bbooth/P3AI/job_file/img_transform.npy')
    
    # Align the frames with lasering to the job file's coordinate system
    timestamps_loc = '/scratch/bbooth/P3AI/job_file/scan_line_timestamps.nii.gz'
    timestamp_nii = nib.load(timestamps_loc)
    time_img = timestamp_nii.get_fdata()[:,:,0]
    time_img = np.flip(time_img.T, axis=0)
    imSz = time_img.shape
    
    layerID = [0, 1, 2, 3, 4, 5, 7, 8, 10, 11, 12, 13, 14, 15, 17, 18, 20,
               21, 22, 23, 24, 25, 27, 28, 30, 31, 32, 33, 34, 35, 37, 38,
               40, 41, 42, 43, 44]
    
    N = 0
    for ii in range(32,46):
        vol = load_video_file(ii)
        
        if ii == 33 or ii == 36 or ii == 39 or ii == 42 or ii == 45:
            sidx, eidx = detect_layer_idx(vol, True, False)
        elif ii == 35 or ii == 38 or ii == 41 or ii == 44: 
            sidx, eidx = detect_layer_idx(vol, False, True)
        else:
            sidx, eidx = detect_layer_idx(vol, False, False)
        
        print(sidx)
        print(eidx)
        
        for jj in range(len(sidx)):
            layer_vol = vol[:,:,sidx[jj]:eidx[jj]]
            aligned_layer_vol = align_frames(layer_vol, tform, imSz)
            save_layer(aligned_layer_vol, layerID[N])
            N += 1
            print(N)


if __name__ == '__main__':
    main()