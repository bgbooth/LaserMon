#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 15:09:20 2024

@author: bbooth
"""
import cv2
import h5py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors

def load_sync_sigs(layerID):
    in_loc = '/scratch/bbooth/FAIR/Cyl14/layer%04d.h5' % layerID
    
    # Load timestamps and laser status signals for synchronization, and
    # video for image feature calculation
    fp = h5py.File(in_loc, 'r')
    #fp.visit(print)
    lstat1 = np.array(list(fp['mcp/QIL.Laser Status']))
    tstamp1 = np.array(list(fp['mcp/Timestamp']))
    vid = np.array(list(fp['onaxis/frames']))
    lstat2 = np.array(list(fp['onaxis/laser']))
    tstamp2 = np.array(list(fp['onaxis/timestamps']))
    fp.close()
    
    sigs_mcp = np.zeros([lstat1.shape[0], 2])
    sigs_mcp[:,0] = lstat1
    sigs_mcp[:,1] = tstamp1
    sigs_cam = np.zeros([lstat2.shape[0], 2])
    sigs_cam[:,0] = lstat2
    sigs_cam[:,1] = tstamp2

    return sigs_mcp, sigs_cam, vid


def assign_idx(sidx_cam, eidx_cam, sidx_mcp, eidx_mcp):
    
    # Linear interpolation
    N = eidx_cam - sidx_cam
    idx = np.round(np.linspace(sidx_mcp, eidx_mcp-1, num=N))
    return idx


def align_laser_status(lstat_mcp, lstat_cam):
    
    # Get start and end points of each scan line
    grad_mcp = np.diff(lstat_mcp)
    sidx_mcp = np.where(grad_mcp > 0.5)[0]+1
    eidx_mcp = np.where(grad_mcp < -0.5)[0]
    
    grad_cam = np.diff(lstat_cam)
    sidx_cam = np.where(grad_cam > 0.5)[0]+1
    eidx_cam = np.where(grad_cam < -0.5)[0]

    t_on_mcp = eidx_mcp - sidx_mcp + 1
    mcp_frac = np.cumsum(t_on_mcp) / np.sum(t_on_mcp)
    t_on_cam = eidx_cam - sidx_cam + 1
    cam_frac = np.cumsum(t_on_cam) / np.sum(t_on_cam)
    
    # Match starts and ends of scan lines across signals and images
    mapping = np.zeros(len(cam_frac), dtype=np.int32)
    nn_mcp = NearestNeighbors(n_neighbors=1)
    nn_mcp.fit(mcp_frac.reshape(-1, 1))
    for ii in range(len(cam_frac)):
        _, nn_idx = nn_mcp.kneighbors(cam_frac[ii].reshape(1, -1))
        mapping[ii] = nn_idx.flatten()[0]
    
    # Linearly interpolate each scan line separately, and the gaps between them
    cam_idx = np.zeros(len(lstat_cam), dtype=np.int32)
    cam_idx[0:sidx_cam[0]] = assign_idx(0, sidx_cam[0], 0, sidx_mcp[0])
    for ii in range(len(mapping)-1):
        cam_idx[sidx_cam[ii]:eidx_cam[ii]+1] = assign_idx(sidx_cam[ii], eidx_cam[ii]+1, 
                                                          sidx_mcp[mapping[ii]], 
                                                          eidx_mcp[mapping[ii]]+1)
        cam_idx[eidx_cam[ii]+1:sidx_cam[ii+1]] = assign_idx(eidx_cam[ii]+1, sidx_cam[ii+1], 
                                                          eidx_mcp[mapping[ii]]+1, 
                                                          sidx_mcp[mapping[ii+1]])
    cam_idx[sidx_cam[-1]:eidx_cam[-1]+1] = assign_idx(sidx_cam[-1], eidx_cam[-1]+1, 
                                                      sidx_mcp[mapping[-1]], 
                                                      eidx_mcp[mapping[-1]]+1)
    cam_idx[eidx_cam[-1]+1:] = assign_idx(eidx_cam[-1]+1, len(lstat_cam), 
                                          eidx_mcp[mapping[-1]]+1, len(lstat_mcp))
    
    return cam_idx


def calc_meltpool_features(vid):
    
    #print(vid.shape)
    features = np.zeros([vid.shape[0], 3])
    
    for ii in range(vid.shape[0]):
        # Copmute connected components
        seg = cv2.threshold(np.squeeze(vid[ii,:,:]), 38, 255, cv2.THRESH_BINARY)[1]
        analysis = cv2.connectedComponentsWithStats(seg, 4, cv2.CV_32S) 
        (totalLabels, label_img, stats, centroid) = analysis 
        
        #print(stats)
        # Number of spatters (ignore background and meltpool)
        features[:,2] = np.maximum(0, totalLabels-2)
        
        # Size of meltpool
        if totalLabels > 1:
            areas = np.sort(stats[:,-1])
            features[ii,1] = areas[-2]
        
            # Intensity of the meltpool
            idx = np.where(stats[:,-1] == features[ii,1])[0]
            #print(idx)
            #print(label_img.shape)
            mp_seg = np.float32(label_img == idx[0])
            features[ii,0] = np.sum(mp_seg * np.squeeze(vid[ii,:,:])) / features[ii,1]
            features[ii,0] /= 255

    return features


def save_layer(layerID, mapping, features):
    
    out_loc = '/scratch/bbooth/FAIR/Cyl14_Processed/layer%04d.h5' % layerID
    in_loc = '/scratch/bbooth/FAIR/Cyl14/layer%04d.h5' % layerID
    
    # Load all signals
    fpIN = h5py.File(in_loc, 'r')
    diode1 = np.array(list(fpIN['mcp/TR1Mod4.AI0']))
    diode2 = np.array(list(fpIN['mcp/TR1Mod4.AI1']))
    xx = np.array(list(fpIN['mcp/QIL.X Setpoint']))
    yy = np.array(list(fpIN['mcp/QIL.Y Setpoint']))
    lstat = np.array(list(fpIN['mcp/QIL.Laser Status']))
    vid = np.array(list(fpIN['onaxis/frames']))
    fpIN.close()
    
    # Save signals with new mapping and features
    fpOUT = h5py.File(out_loc, 'w')
    fpOUT.create_dataset('frames', data=vid)
    fpOUT.create_dataset('photodiode0', data=diode1)
    fpOUT.create_dataset('photodiode1', data=diode2)
    fpOUT.create_dataset('mapping', data=mapping)
    fpOUT.create_dataset('setpoint_x', data=xx)
    fpOUT.create_dataset('setpoint_y', data=yy)
    fpOUT.create_dataset('laser_status', data=lstat)
    fpOUT.create_dataset('features', data=features)
    fpOUT.close()
    

def main():
    
    idx = [240, 270, 300, 330, 349, 350, 351, 360, 390, 420, 450, 470, 471, 472]
    
    for ii in idx:
        sigs_mcp, sigs_cam, vid = load_sync_sigs(ii)
        mapping = align_laser_status(sigs_mcp[:,0], sigs_cam[:,0])
        features = calc_meltpool_features(vid)
        save_layer(ii, mapping, features)
        print(ii)
    
    
    
if __name__ == '__main__':
    main()