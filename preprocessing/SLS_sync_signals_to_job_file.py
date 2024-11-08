#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analyze the signals and try to match them to the job file.

Created on Wed Aug 14 14:07:08 2024

@author: bbooth
"""

import h5py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors

def load_signals(layerID):
    
    f_loc = '/scratch/bbooth/P3AI/signals/layer_%02d.h5'
    #print(f_loc % layerID)
    
    with h5py.File(f_loc % layerID, 'r') as f:
        diode_filt = np.array(list(f['diode_filtered']))
        diode_non = np.array(list(f['diode_unfiltered']))
        lstat = np.array(list(f['laser_status']))
        f.close()
    
    #print(diode_filt.shape)
    M = np.zeros([diode_filt.shape[0], 3])
    M[:,0] = diode_filt
    M[:,1] = diode_non
    M[:,2] = lstat
    
    return M


def load_scan_lines(layerID):
    
    f_loc = '/scratch/bbooth/P3AI/signals/scan_lines_layer_%02d.h5'
    #print(f_loc % layerID)
    
    with h5py.File(f_loc % layerID, 'r') as f:
        x_start = np.array(list(f['sid_x']))
        y_start = np.array(list(f['sid_y']))
        speed = np.array(list(f['speed']))
        x_end = np.array(list(f['eid_x']))
        y_end = np.array(list(f['eid_y']))
        isBorder = np.array(list(f['is_border']))
        f.close()
    
    #print(x_start.shape)
    M = np.zeros([x_start.shape[0], 6])
    M[:,0] = x_start
    M[:,1] = y_start
    M[:,2] = x_end
    M[:,3] = y_end
    M[:,4] = speed
    M[:,5] = isBorder
    
    return M

def find_scan_line_splits(lines):
    
    db = np.nonzero(np.diff(lines[:,5]))[0]
    sidx = np.zeros(len(db)+1, dtype=np.int32)
    eidx = np.zeros(len(db)+1, dtype=np.int32)
    
    #plt.plot(lines[:,5])
    #plt.show()
    
    for ii in range(len(db)+1):
        if ii == 0:
            eidx[ii] = db[ii]
        elif ii == len(db):
            sidx[ii] = db[ii-1]+1
            eidx[ii] = lines.shape[0]-1
        else:
            sidx[ii] = db[ii-1]+1
            eidx[ii] = db[ii]
    
    return sidx, eidx+1


def get_scan_line_timesteps(stage_lines):
    
    start_stamps = np.zeros(stage_lines.shape[0])
    end_stamps = np.zeros(stage_lines.shape[0])
    D = 0
    
    for ii in range(stage_lines.shape[0]):
        start_stamps[ii] = D
        line_length = np.sqrt((stage_lines[ii,2] - stage_lines[ii,0]) ** 2 + 
                              (stage_lines[ii,3] - stage_lines[ii,1]) ** 2)
        D += line_length
        end_stamps[ii] = start_stamps[ii] + line_length
    
    start_stamps /= D
    end_stamps /= D
    
    return start_stamps, end_stamps


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


def get_stage_timesteps(sigs):
    
    tstamps = np.zeros(sigs.shape[0])
    
    N = np.sum(sigs[:,2])
    idx = np.nonzero(sigs[:,2])[0]
    for jj in range(len(idx)):
        tstamps[idx[jj]] = jj / (N-1)
    
    return tstamps


def look_up_job_coordinates(lines, sig_fraqs, sfraqs, efraqs):
    
    # FIXME: the order of scan lines is not verified.
    
    job_x = np.zeros(len(sig_fraqs))
    job_y = np.zeros(len(sig_fraqs))

    nnStart = NearestNeighbors(n_neighbors=1)
    nnStart.fit(sfraqs.reshape(-1, 1))
    nnEnd = NearestNeighbors(n_neighbors=1)
    nnEnd.fit(efraqs.reshape(-1, 1))

    for ii in range(len(sig_fraqs)):
        if sig_fraqs[ii] == 0:
            continue
        dS, idxS = nnStart.kneighbors(sig_fraqs[ii].reshape(-1, 1))
        dE, idxE = nnEnd.kneighbors(sig_fraqs[ii].reshape(-1, 1))
        idxS = idxS.flatten()[0]
        idxE = idxE.flatten()[0]
        dS = dS.flatten()[0]
        dE = dE.flatten()[0]
        #print([idxS, idxE])
        #print([dS, dE])
        fraq = dS / np.maximum(1e8, dS + dE)
        job_x[ii] = fraq * lines[idxS, 0] + (1-fraq) * lines[idxE, 2]
        job_y[ii] = fraq * lines[idxS, 1] + (1-fraq) * lines[idxE, 3]
    
    return job_x, job_y


def get_job_file_LUT_params(lines):
    
    imSz = np.zeros(2, dtype=np.int32)
    x_max = np.maximum(np.max(lines[:,2], axis=0), np.max(lines[:,0], axis=0))
    x_min = np.minimum(np.min(lines[:,2], axis=0), np.min(lines[:,0], axis=0))
    y_max = np.maximum(np.max(lines[:,3], axis=0), np.max(lines[:,1], axis=0))
    y_min = np.minimum(np.min(lines[:,3], axis=0), np.min(lines[:,1], axis=0))
    x_cc = (x_max + x_min) / 2.0
    y_cc = (y_max + y_min) / 2.0
    
    x_sz = round(x_max - x_min + 2)
    y_sz = round(y_max - y_min + 2)+2
    voxel_scale = 10.0
    
    imSz[0] = int(voxel_scale * x_sz)
    imSz[1] = int(voxel_scale * y_sz)
    x_start = round(imSz[0] / 2.0)
    y_start = round(imSz[1] / 2.0)
    
    #print(imSz)
    
    return np.array([x_cc, y_cc, voxel_scale, x_start, y_start])


def map_to_LUT_coords(xx, yy, params):
    
    pix_x = np.zeros(len(xx))
    pix_y = np.zeros(len(yy))
    
    #print(params)
    for ii in range(len(xx)):
        #print([xx[ii], yy[ii]])
        pix_x[ii] = np.maximum(0, ((xx[ii] - params[0]) * params[2]) + params[3])
        pix_y[ii] = np.maximum(0, ((yy[ii] - params[1]) * params[2]) + params[4])
        #print([pix_x[ii], pix_y[ii]])
        #return

    return pix_x, pix_y


def save_layer(sigs, coords, layerID):
    out_loc = '/scratch/bbooth/P3AI/signals/synced_layer_%02d.h5'
    
    hf = h5py.File(out_loc % layerID, 'w')
    hf.create_dataset('diode_filtered', data=sigs[:,0])
    hf.create_dataset('diode_unfiltered', data=sigs[:,1])
    hf.create_dataset('laser_status', data=sigs[:,2])
    hf.create_dataset('laser_x', data=coords[:,0])
    hf.create_dataset('laser_y', data=coords[:,1])
    hf.create_dataset('pixel_x', data=coords[:,2])
    hf.create_dataset('pixel_y', data=coords[:,3])
    hf.close()


def main():
    
    # Load job file to image transform parameters
    lines = load_scan_lines(0)
    params = get_job_file_LUT_params(lines)
    
    # Sync each layer
    for ii in range(45):
        sigs = load_signals(ii)
        lines = load_scan_lines(ii)
        
        sidx, eidx = find_signal_splits(sigs)
        
        if ii < 2:
            sidx, eidx = fix_signal_splits(sidx, eidx, ii)
        
        sidx2, eidx2 = find_scan_line_splits(lines) 
        
        all_coords = np.zeros([sigs.shape[0], 4])
        
        for jj in range(len(sidx2)):
            stage_sigs = sigs[sidx[jj]:eidx[jj],:]
            stage_lines = lines[sidx2[jj]:eidx2[jj],:]
            sig_fraqs = get_stage_timesteps(stage_sigs)
            line_starts, line_ends = get_scan_line_timesteps(stage_lines)
            job_x, job_y = look_up_job_coordinates(stage_lines, sig_fraqs, line_starts, line_ends)
            pixel_x, pixel_y = map_to_LUT_coords(job_x, job_y, params)
            all_coords[sidx[jj]:eidx[jj],0] = job_x
            all_coords[sidx[jj]:eidx[jj],1] = job_y
            all_coords[sidx[jj]:eidx[jj],2] = pixel_x
            all_coords[sidx[jj]:eidx[jj],3] = pixel_y
            
        
        #plt.plot(all_coords[:,2], all_coords[:,3], '.')
        #plt.show()
        #return
    
        save_layer(sigs, all_coords, ii)
        print(ii)
        

if __name__ == '__main__':
    main()