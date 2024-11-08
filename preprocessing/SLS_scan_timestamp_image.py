#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Make an image that shows the time the laser crosses a specific pixel

Created on Tue Jul 30 16:47:00 2024

@author: bbooth
"""

import h5py
import nibabel as nib
import numpy as np
from skimage import morphology
from sklearn.neighbors import NearestNeighbors


def get_points(filename):
    
    with h5py.File(filename, 'r') as f:
        points = f['Points']
        layer_nums = f['Heights']

        n_lines = len(layer_nums)
        lines = np.zeros([n_lines, 6])
        for ii in range(n_lines):
            lines[ii,0] = points[ii][0][0]
            lines[ii,1] = points[ii][0][1]
            lines[ii,2] = layer_nums[ii]
            lines[ii,3] = points[ii][1][0]
            lines[ii,4] = points[ii][1][1]
            lines[ii,5] = layer_nums[ii]
    return lines


def make_scan_line_timestamp_image(points, imSz):

    mask = np.zeros([imSz[0], imSz[1]])
    n_samples = 200
    # Plot scan lines onto an image with the scan line length as pixel intensity
    for ii in range(len(points[:,0])):
        line_pts = np.zeros([n_samples+1, 2])
        #d = np.sqrt((points[ii][3] - points[ii][0]) * (points[ii][3] - points[ii][0]) + \
        #            (points[ii][4] - points[ii][1]) * (points[ii][4] - points[ii][1]))
        for jj in range(n_samples+1):
            w = jj / n_samples
            line_pts[jj][0] = points[ii][0] + w * (points[ii][3] - points[ii][0])
            line_pts[jj][1] = points[ii][1] + w * (points[ii][4] - points[ii][1])
        line_pts = np.rint(line_pts)
        line_pts = np.unique(line_pts, axis=0)
        for jj in range(len(line_pts[:,0])):
            mask[int(line_pts[jj,0]), int(line_pts[jj,1])] = ii

    # Fill holes in the object using mathematical morphology
    mask_full = morphology.remove_small_holes(mask > 0.01, 200)
    strel = morphology.disk(4)     
    mask_full = morphology.binary_closing(mask_full, strel)
    
    # plt.imshow(mask)
    # plt.show()
    # plt.imshow(mask_full)
    # plt.show()
    
    # Use nearest neighbour calculations to propagate scan line length information
    # to pixels between scan lines
    pts_on_lines = np.where(mask > 0.5)
    train_pts = np.zeros([len(pts_on_lines[0]), 2])
    train_pts[:,0] = np.asarray(pts_on_lines[0])
    train_pts[:,1] = np.asarray(pts_on_lines[1])

    pts_in_obj = np.where(mask_full)
    test_pts = np.zeros([len(pts_in_obj[0]), 2])
    test_pts[:,0] = np.asarray(pts_in_obj[0])
    test_pts[:,1] = np.asarray(pts_in_obj[1])

    mask2 = np.zeros([imSz[0], imSz[1]])
    if len(train_pts) > 0:
        knn = NearestNeighbors(n_neighbors=1)
        knn.fit(train_pts)
        test_neighbours = np.asarray(knn.kneighbors(test_pts, 1, return_distance=False))
        for ii in range(len(test_pts[:,0])):
            mask2[int(test_pts[ii,0]), int(test_pts[ii,1])] = \
                mask[int(train_pts[test_neighbours[ii],0]), int(train_pts[test_neighbours[ii],1])]
    
    return mask2


def make_coord_images(timestamp_img, x_cc, y_cc, voxel_scale, x_start, y_start):
    
    x_coord_img = np.zeros(timestamp_img.shape)
    y_coord_img = np.zeros(timestamp_img.shape)
    
    for ii in range(x_coord_img.shape[0]):
        tmp_x = ((ii - x_start) / voxel_scale) + x_cc
        x_coord_img[ii,:] = tmp_x
    
    for ii in range(x_coord_img.shape[1]):
        tmp_y = ((ii - y_start) / voxel_scale) + y_cc
        y_coord_img[:,ii] = tmp_y

    x_coord_img = x_coord_img * (timestamp_img > 1e-8)
    y_coord_img = y_coord_img * (timestamp_img > 1e-8)

    return x_coord_img, y_coord_img


def main():
    
    in_file = '/scratch/bbooth/P3AI/job_file/RS+10np-10.hdf5'

    # Parse the simplified job file
    lines = get_points(in_file)

    # Get the unique heights for each scan line so that we
    # can identify each layer
    layer_heights = np.unique(lines[:,2])
    for ii in range(len(layer_heights)):
        idx = np.where(lines[:,2] == layer_heights[ii])
        lines[idx[0],5] = ii

    # Get the size of the look-up table
    imSz = np.zeros(3, dtype=np.int32)
    imSz[2] = len(layer_heights)+1 # Number of layers
    x_max = np.maximum(np.max(lines[:,3], axis=0), np.max(lines[:,0], axis=0))
    x_min = np.minimum(np.min(lines[:,3], axis=0), np.min(lines[:,0], axis=0))
    y_max = np.maximum(np.max(lines[:,4], axis=0), np.max(lines[:,1], axis=0))
    y_min = np.minimum(np.min(lines[:,4], axis=0), np.min(lines[:,1], axis=0))
    x_cc = (x_max + x_min) / 2.0
    y_cc = (y_max + y_min) / 2.0
    
    x_sz = round(x_max - x_min + 2)
    y_sz = round(y_max - y_min + 2)
    voxel_scale = 10.0
    
    imSz[0] = int(voxel_scale * x_sz)
    imSz[1] = int(voxel_scale * y_sz)
    x_start = round(imSz[0] / 2.0)
    y_start = round(imSz[1] / 2.0)
    
    coord_sz = np.zeros(4, dtype=np.int32)
    coord_sz[0:3] = imSz
    coord_sz[3] = 2
    
    time_vol = np.zeros(imSz)
    coord_vol = np.zeros(coord_sz)
    
    for ii in range(len(layer_heights)):
        idx_layer = np.where(lines[:,5] == ii)
        layer_pts = lines[idx_layer[0],:]
        for jj in range(len(layer_pts[:,0])):
            layer_pts[jj,0] = ((layer_pts[jj,0] - x_cc) * voxel_scale) + x_start
            layer_pts[jj,1] = ((layer_pts[jj,1] - y_cc) * voxel_scale) + y_start
            layer_pts[jj,3] = ((layer_pts[jj,3] - x_cc) * voxel_scale) + x_start
            layer_pts[jj,4] = ((layer_pts[jj,4] - y_cc) * voxel_scale) + y_start
        time_vol[:,:,ii] = make_scan_line_timestamp_image(layer_pts, time_vol.shape)
        coord_vol[:,:,ii,0], coord_vol[:,:,ii,1] = make_coord_images(time_vol[:,:,ii], x_cc, y_cc, \
                                                                     voxel_scale, x_start, y_start)
        print("Scan Line Timestamps, Layer " + str(ii))

    scan_line_nii = nib.Nifti1Image(time_vol, np.eye(4))
    nib.save(scan_line_nii, '/scratch/bbooth/P3AI/job_file/scan_line_timestamps.nii.gz')
    
    coords_nii = nib.Nifti1Image(coord_vol, np.eye(4))
    nib.save(coords_nii, '/scratch/bbooth/P3AI/job_file/buildplate_coords.nii.gz')
    

if __name__ == "__main__":
    main()
