#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Align coordinates to the thermal images

Created on Mon Aug  5 10:34:03 2024

@author: bbooth
"""

import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import SimpleITK as sitk
import cv2
import skimage.morphology as morph

def command_iteration(method):
    print(
        f"{method.GetOptimizerIteration():3} "
        + f"= {method.GetMetricValue():10.5f} "
        + f": {method.GetOptimizerPosition()}"
    )
    

def align_images(temp_img, tstamp_img):
    
    temp_simg = sitk.GetImageFromArray(temp_img)
    tstamp_simg = sitk.GetImageFromArray(tstamp_img)
    
    R = sitk.ImageRegistrationMethod()
    R.SetMetricAsMattesMutualInformation(32)
    R.SetOptimizerAsRegularStepGradientDescent(0.5, 0.00005, 200)
    R.SetInitialTransform(sitk.AffineTransform(temp_simg.GetDimension()))
    R.SetInterpolator(sitk.sitkLinear)
    
    R.AddCommand(sitk.sitkIterationEvent, lambda: command_iteration(R))
    
    tform = R.Execute(tstamp_simg, temp_simg)
    print(tform)
    
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(tstamp_simg)
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetDefaultPixelValue(0)
    resampler.SetTransform(tform)

    aligned_simg = resampler.Execute(temp_simg)
    aligned_img = sitk.GetArrayFromImage(aligned_simg)
    
    return aligned_img


def get_object_centroids(img):
    

    mask = np.uint8(cv2.threshold(img, 127, 3500, cv2.THRESH_BINARY)[1])
    
    threshold = np.uint8(morph.remove_small_holes(mask, 5000))
    
    analysis = cv2.connectedComponentsWithStats(threshold, 4, cv2.CV_32S) 
    (totalLabels, label_ids, values, centroid) = analysis 

    # plt.imshow(threshold)
    # plt.colorbar()
    # plt.plot(centroid[1:,0], centroid[1:,1], 'xr')
    # plt.show()
    # plt.clf()

    return centroid[1:,:]


def align_homography(pts_move, pts_fixed, img_move, imSz):
    
    H, status = cv2.findHomography(pts_move, pts_fixed)
    
    #print(H)
    #print(status)
    
    img_aligned = cv2.warpPerspective(img_move, H, np.flip(imSz))
    
    return H, img_aligned


def main():
    
    img_loc = '/scratch/bbooth/P3AI/layer1_imgs/max_temps_idx.npy'
    timestamps_loc = '/scratch/bbooth/P3AI/job_file/scan_line_timestamps.nii.gz'
    coords_loc = '/scratch/bbooth/P3AI/job_file/buildplate_coords.nii.gz'
    
    img = np.float64(np.load(img_loc))
    newSz = np.array(img.shape) * 10
    scaled_img = cv2.resize(img, dsize=(newSz[1], newSz[0]), interpolation=cv2.INTER_CUBIC)
    
    timestamp_nii = nib.load(timestamps_loc)
    time_img = timestamp_nii.get_fdata()[:,:,0]
    time_img = np.flip(time_img.T, axis=0)
    
    coords_nii = nib.load(coords_loc)
    coords_imgs = np.squeeze(coords_nii.get_fdata()[:,:,0,:])
    coords_imgs = np.flip(np.transpose(coords_imgs, (1,0,2)), axis=0)
    
    plt.imshow(img)
    plt.colorbar()
    plt.show()
    plt.clf()
    
    plt.imshow(time_img)
    plt.colorbar()
    plt.show()
    plt.clf()

    temp_cc = get_object_centroids(scaled_img) / 10
    time_cc = get_object_centroids(time_img)
    
    H, aligned_img = align_homography(temp_cc, time_cc, img, time_img.shape)
    
    plt.imshow(aligned_img)
    plt.colorbar()
    plt.show()
    plt.clf()
    
    save_loc = '/scratch/bbooth/P3AI/job_file/img_transform.npy'
    
    np.save(save_loc, H)
    
    
    # plt.imshow(coords_imgs[:,:,0])
    # plt.colorbar()
    # plt.show()
    # plt.clf()
    
    # plt.imshow(coords_imgs[:,:,1])
    # plt.colorbar()
    # plt.show()
    # plt.clf()
    
    # aligned_temps = align_images(10 * scaled_img, time_img)
    
    # plt.imshow(aligned_temps)
    # plt.colorbar()
    # plt.show()
    # plt.clf()
    
    # plt.imshow(time_img)
    # plt.colorbar()
    # plt.show()
    # plt.clf()
    

if __name__ == '__main__':
    main()