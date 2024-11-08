#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Detect print layers in the video.

Created on Tue Jul 30 14:32:33 2024

@author: bbooth
"""

import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt

def load_video_file(file_id):
    
    imgs_loc = '/scratch/bbooth/P3AI/raw_data/Thermal/Nifti/recording%d.nii'
    
    nii = nib.load(imgs_loc % file_id)
    vol = nii.get_fdata()
    
    cropped_vol = vol[205:253,310:383]
    
    return cropped_vol


def detect_layer_idx(vid):
    
    offset = 50
    
    avg_temp = np.mean(vid, axis=(0,1))
    
    #plt.plot(avg_temp)
    #plt.show()
    
    recoat_idx = np.nonzero(np.diff(avg_temp < 150))[0] + 1
    #print(recoat_idx)
    
    layer_idx_start = list()
    layer_idx_end = list()
    for ii in range(len(recoat_idx)-1):
        if recoat_idx[ii+1] - recoat_idx[ii] > 100:
            layer_idx_start.append(recoat_idx[ii] + offset)
            layer_idx_end.append(recoat_idx[ii+1] - offset)

    return np.array(layer_idx_start), np.array(layer_idx_end)


def main():
    
    vid = load_video_file(32)
    
    idx_start, idx_end = detect_layer_idx(vid)
    #print(idx_start)
    #print(idx_end)
    
    layer1 = vid[:,:,idx_start[0]:idx_end[0]]
    
    max_img = np.max(layer1, axis=2)
    arg_max_img = np.argmax(layer1, axis=2)
    
    save_loc = '/scratch/bbooth/P3AI/layer1_imgs/max_temps.npy'
    save_loc2 = '/scratch/bbooth/P3AI/layer1_imgs/max_temps_idx.npy'
    
    plt.imshow(max_img)
    plt.colorbar()
    #plt.savefig(save_loc)
    plt.show()
    plt.clf()
    
    plt.imshow(arg_max_img)
    plt.colorbar()
    plt.show()
    plt.clf()
    
    plt.imshow(arg_max_img * (max_img > 185))
    plt.colorbar()
    #plt.savefig(save_loc2)
    plt.show()
    plt.clf()
    
    np.save(save_loc, max_img)
    max_img_idx = arg_max_img * (max_img > 185)
    np.save(save_loc2, max_img_idx)
    
    # layer1[0,0,:] = 255
    # layer1[-1,-1,:] = 127
    
    # save_loc = '/scratch/bbooth/P3AI/layer1_imgs/frame%03d.png'
    
    # for ii in range(layer1.shape[2]):
    #     plt.imshow(layer1[:,:,ii])
    #     #if ii == 0:
    #     plt.colorbar()
    #     plt.savefig(save_loc % ii)
    #     plt.clf()
    #     print(ii)

    # save_loc2 = '/scratch/bbooth/P3AI/layer1_imgs/diff_img%03d.png'

    # for ii in range(layer1.shape[2]-1):
    #     diff_img = layer1[:,:,ii+1] - layer1[:,:,ii]
    #     diff_img[0,0] = -70
    #     diff_img[-1,-1] = 70
    #     plt.imshow(diff_img)
    #     plt.colorbar()
    #     plt.savefig(save_loc2 % ii)
    #     plt.clf()
    #     print(ii)
    

if __name__ == '__main__':
    main()