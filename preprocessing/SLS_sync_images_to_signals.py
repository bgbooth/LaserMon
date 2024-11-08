#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sync images to signals

Created on Wed Sep 11 13:13:00 2024

@author: bbooth
"""

import h5py
import os.path
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors

def load_signals(layerID):
    
    f_loc = '/scratch/bbooth/P3AI/signals/synced_layer_%02d.h5'
    print(f_loc % layerID)
    
    with h5py.File(f_loc % layerID, 'r') as f:
        diode_filt = np.array(list(f['diode_filtered']))
        diode_non = np.array(list(f['diode_unfiltered']))
        lstat = np.array(list(f['laser_status']))
        lx = np.array(list(f['laser_x']))
        ly = np.array(list(f['laser_y']))
        px = np.array(list(f['pixel_x']))
        py = np.array(list(f['pixel_y']))
        f.close()
    
    print(diode_filt.shape)
    M = np.zeros([diode_filt.shape[0], 7])
    M[:,0] = diode_filt
    M[:,1] = diode_non
    M[:,2] = lstat
    M[:,3] = lx
    M[:,4] = ly
    M[:,5] = px
    M[:,6] = py
    
    return M


def save_layer(sigs, cam_idx, imgs, layerID):
    out_loc = '/scratch/bbooth/P3AI/signals/final_layer_%02d.h5'
    
    hf = h5py.File(out_loc % layerID, 'w')
    hf.create_dataset('diode_filtered', data=sigs[:,0])
    hf.create_dataset('diode_unfiltered', data=sigs[:,1])
    hf.create_dataset('laser_status', data=sigs[:,2])
    hf.create_dataset('laser_x', data=sigs[:,3])
    hf.create_dataset('laser_y', data=sigs[:,4])
    hf.create_dataset('pixel_x', data=sigs[:,5])
    hf.create_dataset('pixel_y', data=sigs[:,6])
    hf.create_dataset('img_idx', data=cam_idx)
    hf.create_dataset('images', data=imgs)
    hf.close()
    
    
def load_images(layerID):
    
    imgs_loc = '/scratch/bbooth/P3AI/signals/video_layer%02d.nii'
    
    nii = nib.load(imgs_loc % layerID)
    vol = nii.get_fdata()
    
    return vol


def find_signal_splits(sigs):
    
    #diff_idx = np.where(sigs[1:,2] != sigs[:-1,2])[0]
    
    #sidx = np.where((sigs[1:,2] == 1) * (sigs[:-1,2] == 0))[0] - 1
    #eidx = np.where((sigs[1:,2] == 0) * (sigs[:-1,2] == 1))[0]
    sigs[0,2] = 0 # Correction for first layer
    
    on_idx = np.nonzero(sigs[:,2])[0]
    dd = np.diff(on_idx)
    gap_idx = np.where(np.abs(dd) > 20000)[0]
    
    #print(len(gap_idx)+1)
    
    sidx = np.zeros(len(gap_idx)+1, dtype=np.int32)
    eidx = np.zeros(len(gap_idx)+1, dtype=np.int32)
    for ii in range(len(gap_idx)+1):
        if ii == 0:
            sidx[ii] = on_idx[0]
        else:
            sidx[ii] = on_idx[gap_idx[ii-1]+1]
        if ii == len(gap_idx):
            eidx[ii] = on_idx[-1]
        else:
            eidx[ii] = on_idx[gap_idx[ii]]
        #print([ii, sidx[ii], eidx[ii]])
    
    good_idx = np.where(((eidx-sidx) * 1e-2) > 15)[0]
    
    return sidx[good_idx], eidx[good_idx]
    

def fix_signal_splits(sidx, eidx, layerID):
    
    new_sidx = np.zeros(13, dtype=np.int32)
    new_eidx = np.zeros(13, dtype=np.int32)
    
    print(layerID)
    print(sidx)
    print(eidx)
    
    if layerID == 0:
        new_sidx[0:2] = sidx[0:2]
        new_eidx[0:2] = eidx[0:2]
        new_sidx[2] = sidx[2]
        new_eidx[2] = eidx[3]
        new_sidx[3] = sidx[4]
        new_eidx[3] = eidx[4]
        new_sidx[4] = sidx[5]
        new_eidx[4] = eidx[6]
        new_sidx[5:8] = sidx[7:10]
        new_eidx[5:8] = eidx[7:10]
        new_sidx[8] = sidx[10]
        new_eidx[8] = eidx[11]
        new_sidx[9:] = sidx[12:]
        new_eidx[9:] = eidx[12:]
    else:
        new_sidx[0:2] = sidx[0:2]
        new_eidx[0:2] = eidx[0:2]
        new_sidx[2] = sidx[2]
        new_eidx[2] = eidx[3]
        new_sidx[3:6] = sidx[4:7]
        new_eidx[3:6] = eidx[4:7]
        new_sidx[6] = sidx[7]
        new_eidx[6] = eidx[8]
        new_sidx[7:10] = sidx[9:12]
        new_eidx[7:10] = eidx[9:12]
        new_sidx[10] = sidx[12]
        new_eidx[10] = eidx[13]
        new_sidx[11:] = sidx[14:]
        new_eidx[11:] = eidx[14:]
    
    return new_sidx, new_eidx


def find_image_splits(vid):
    
    diff_imgs = np.maximum(0, np.diff(vid, axis=2))
    lasered_pix = np.sum(diff_imgs > 3, axis=(0,1))

    on_idx = np.where(lasered_pix > 750)[0]+1
    dd = np.diff(on_idx)
    gap_idx = np.where(np.abs(dd) > 3)[0]

    sidx = np.zeros(len(gap_idx)+1, dtype=np.int32)
    eidx = np.zeros(len(gap_idx)+1, dtype=np.int32)
    for ii in range(len(gap_idx)+1):
        if ii == 0:
            sidx[ii] = on_idx[0]
        else:
            sidx[ii] = on_idx[gap_idx[ii-1]+1]
        if ii == len(gap_idx):
            eidx[ii] = on_idx[-1]
        else:
            eidx[ii] = on_idx[gap_idx[ii]]    
    
    return sidx, eidx


def sync_imgs_to_sigs(vid, sigs, sidx, eidx, sidx2, eidx2):
    
    stage2_vid = vid[:,:,sidx2[1]-1:eidx2[1]+2]
    stage2_sigs = sigs[sidx[1]:eidx[1]+1,:]
    
    diff_imgs = np.maximum(0, np.diff(stage2_vid, axis=2))
    lasered_pix = np.sum(diff_imgs > 3, axis=(0,1))
    lasered_pix = np.float64(lasered_pix * (lasered_pix > 750))
    N = np.sum(lasered_pix)
    lasered_pix = np.cumsum(lasered_pix) / N
    
    fraqs = np.cumsum(stage2_sigs[:,2])
    M = np.sum(stage2_sigs[:,2])
    fraqs /= M
    
    vid_idx = np.nonzero(lasered_pix)[0][0]
    vid_fraq = lasered_pix[vid_idx]
    
    nnSig = NearestNeighbors(n_neighbors=1)
    nnSig.fit(fraqs.reshape(-1, 1))
    _, idx_sig = nnSig.kneighbors(vid_fraq.reshape(-1, 1))
    idx_sig = idx_sig.flatten()[0]
    
    cam_idx = np.zeros(sigs.shape[0])
    cam_idx[sidx[1]+idx_sig] = vid_idx + sidx[1] + 1
    
    offsets = np.arange(0, len(cam_idx), dtype=np.float64)
    offsets /= 4000 # difference in sampling frequencies
    offsets = offsets - offsets[sidx[1]+idx_sig]
    
    cam_idx = cam_idx + offsets
    cam_idx = np.maximum(0, cam_idx)
    
    #print(sidx[1]+idx_sig)
    
    #plt.plot(cam_idx)
    #plt.show()
    print(cam_idx.flatten().shape)
    print(vid.shape)
    
    return cam_idx.flatten()


def crop_sigs(sigs, cam_idx):
    
    sidx = np.max(np.where(cam_idx ==0)[0])
    
    sigs = sigs[sidx:,:]
    cam_idx = cam_idx[sidx:]
    
    return sigs, cam_idx
    

def crop_video(vids, cam_idx):
    
    eidx = np.int32(np.ceil(np.max(cam_idx))+2)
    print(eidx)
    
    vids = vids[:,:,:eidx]
    
    return vids


def main():
    
    imgs_loc = '/scratch/bbooth/P3AI/signals/video_layer%02d.nii'
    
    for ii in range(45):
        if os.path.isfile(imgs_loc % ii):
            sigs = load_signals(ii)
            imgs = load_images(ii)
            
            sidx, eidx = find_signal_splits(sigs)
            sidx2, eidx2 = find_image_splits(imgs)
            #print([ii, len(sidx2)])
            
            #print(sidx2)
            #print(eidx2)
            
            if ii < 2:
                sidx, eidx = fix_signal_splits(sidx, eidx, ii)
                sidx2, eidx2 = fix_signal_splits(sidx2, eidx2, ii)
                
            cam_idx = sync_imgs_to_sigs(imgs, sigs, sidx, eidx, sidx2, eidx2)
            
            print(len(cam_idx))
            
            sigs, cam_idx = crop_sigs(sigs, cam_idx)
            
            print(len(cam_idx))
            
            imgs = crop_video(imgs, cam_idx)
            
            save_layer(sigs, cam_idx, imgs, ii)
            print(ii)
            #return
            
            
if __name__ == '__main__':
    main()
