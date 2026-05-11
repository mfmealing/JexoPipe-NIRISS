#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 18 18:24:36 2025

@author: c1341133
"""

import numpy as np
import matplotlib.pyplot as plt
import glob
import re
from astropy.io import fits
from scipy.optimize import least_squares
from scipy.optimize import minimize_scalar
from scipy.interpolate import interp1d
from scipy import interpolate
from scipy.ndimage import shift
from scipy.optimize import minimize

import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate, signal
from tqdm import tqdm


# --- Optimized helpers ---
from scipy.ndimage import fourier_shift

from scipy import signal

import os
# Set the STPSF_PATH environment variable
os.environ['STPSF_PATH'] = os.path.expanduser('/Users/c24050258/data/stpsf-data')

# import stpsf


# ni=stpsf.NIRISS()
# ni.filter='CLEAR'
# ni.pupil_mask='GR700XD'
 

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy import interpolate, signal
from scipy.ndimage import fourier_shift

# --- helper functions ---
def fft_shift2(arr, shift):
    return np.real(np.fft.ifft(fourier_shift(np.fft.fft(arr), shift)))

def cross_corr2(slice_, new_psf):
    correlation = signal.fftconvolve(new_psf[::-1], slice_, mode='full')
    max_index = np.argmax(correlation) - len(new_psf) + 1
    new_psf = np.roll(new_psf, max_index)
    return new_psf, max_index

def adjust_height_32(slice_, new_psf, y, show_fig=0, pre_y=100, post_y=100, method='leastsq'):
    """Scale `new_psf` to match `slice_` in the window [y-pre_y : y+post_y].

    method:
      - 'median' : replicate original code (median of ratios)
      - 'ls'     : least-squares closed-form (fast)
    Returns (scaled_psf, scale)
    """
    idx = np.arange(max(0, y-pre_y), min(len(slice_), y+post_y))
    if idx.size == 0:
        return new_psf, 0.0

    mask = new_psf[idx] != 0
    if not np.any(mask):
        return new_psf, 0.0

    if method == 'median':
        # original behaviour: median of ratios (robust to outliers)
        ratios = slice_[idx][mask] / new_psf[idx][mask]
        scale = np.median(ratios)
    else:
        # least-squares closed-form (faster)
        num = np.dot(slice_[idx][mask], new_psf[idx][mask])
        den = np.dot(new_psf[idx][mask], new_psf[idx][mask])
        scale = num / den if den > 0 else 0.0

    scaled_psf = new_psf * scale

    if show_fig == 1:
        plt.figure('adjust height 3')
        plt.plot(slice_, 'b-', label="slice")
        plt.plot(new_psf, 'r--', label="original psf")
        plt.plot(scaled_psf, 'g-', label="scaled psf")
        # highlight the indices used for scaling
        plt.plot(idx, slice_[idx], 'bo')
        plt.plot(idx, new_psf[idx], 'ro')
        plt.legend()

    return scaled_psf, scale
# --- main function ---
def run_stage3_v2(new_psf_array, sci, jexo_dic, o1_psf_array, o2_psf_array, show_fig=0):
    mag = 10

    # Load wavelength solutions
    aa = np.load(f'./niriss_wav_sol_{jexo_dic["obs_number"]}_{jexo_dic["tag"]}.npy')
    xpos, ypos, wl = aa
    xpos_, ypos_, wl_ = xpos, mag*ypos, wl

    aa2 = np.load(f'./niriss_wav_sol2_{jexo_dic["obs_number"]}_{jexo_dic["tag"]}.npy')
    xpos2, ypos2, wl2 = aa2
    xpos_2, ypos_2, wl2 = xpos2, mag*ypos2, wl2

    sci_arr_o1 = np.zeros_like(sci)
    sci_arr_o2 = np.zeros_like(sci)

    seq = np.arange(sci.shape[0]).astype(int)
    for ii in tqdm(seq):
        img = sci[ii][:200]

        if show_fig == 1:
            plt.figure(1)
            plt.imshow(img, aspect='auto', vmax=20)

        img_array = img.copy()

        yin = np.linspace(-1.0, 1.0, img_array.shape[1])
        xin = np.linspace(-1.0, 1.0, img_array.shape[0])
        yout = np.linspace(-1.0, 1.0, img_array.shape[1])
        xout = np.linspace(-1.0, 1.0, mag*img_array.shape[0])

        f = interpolate.RectBivariateSpline(xin, yin, img_array)
        new_img_array = f(xout, yout)

        buffer0 = 400*mag
        buffer = np.zeros((buffer0, new_img_array.shape[1]))
        new_img_array = np.vstack((buffer, new_img_array, buffer))

        if show_fig == 1:
            plt.figure(2)
            plt.imshow(new_img_array, aspect='auto')

        o1_slice_array = np.zeros_like(new_img_array)
        o2_slice_array = np.zeros_like(new_img_array)

        box = 50
        bbox = np.ones(box)/box

        for col in range(0, 1000):
            # col =900
            
            idx_o1 = max(col-4, 4)
            idx_o2 = col

            y2 = int(np.round(ypos_2[idx_o2])) + buffer0
            y = int(np.round(ypos_[idx_o1])) + buffer0

            wav = wl2[idx_o2]

            slice_ = new_img_array[:, col].copy()

            if show_fig == 1:
                plt.figure(f'stage 3 initial slice col = {col}')
                plt.plot(slice_)
                plt.plot([y2],[0],'ro')
                plt.plot([y],[0],'bo')
                
                
            # Reverse slice to estimate o1 wing
            PSF = slice_[y-400:y+400]
            idx = np.argwhere(PSF >= 0.5*PSF.max()).T[0] + (y-400)
            idx1 = np.arange(idx[0]-10, idx[0])
            idx2 = np.arange(idx[-1], idx[-1]+10)
            mid_point = (np.mean(idx2)+np.mean(idx1))/2

            slicex = slice_[::-1]
            y_rev = len(slicex)-1-y
            PSF = slicex[y_rev-400:y_rev+400]
            idx = np.argwhere(PSF >= 0.5*PSF.max()).T[0] + (y_rev-400)
            idx1 = np.arange(idx[0]-10, idx[0])
            idx2 = np.arange(idx[-1], idx[-1]+10)
            mid_point_2 = (np.mean(idx2)+np.mean(idx1))/2

            shift = mid_point-mid_point_2
            slicex = fft_shift2(slicex, shift)

            if show_fig == 1:
                plt.figure('reversed slice')
                plt.plot(slice_, '-')
                plt.plot(slicex, '-')
                plt.plot([y2],[0],'ro')

            diff = slice_-slicex
            diff_orig = diff.copy()

            if show_fig == 1:
                plt.figure('stage 3 diff (before smoothing)')
                plt.plot(diff)

            # Smooth diff with boxcar
            diff = np.convolve(diff, bbox, 'same')
            diff = np.where(diff < 0, 0, diff)

            if show_fig == 1:
                plt.figure('stage 3 diff (smoothed)')
                plt.plot(diff)
                plt.plot([y2],[0],'ro')

            # Prepare PSF
            mid = int(np.round(ypos_[1000-4]+buffer0))
            shift = y2-mid
            new_psf = np.copy(new_psf_array[:, idx_o2])
            new_psf_orig = new_psf.copy()

            # Smooth PSF for fitting
            new_psf = np.convolve(new_psf, bbox, 'same')
            new_psf = np.roll(new_psf, shift)
            new_psf_orig = np.roll(new_psf_orig, shift)

            if col >=700:
                new_psf[:y2-300]=0
                new_psf[y2+300:]=0
            elif col>100:
                new_psf[:y2]=0
                new_psf[y2+200:]=0
            else:
                new_psf[:y2]=0
                new_psf[y2+200:]=0

            if show_fig == 1:
                plt.figure('new psf pre scaling')
                plt.plot(new_psf,'g--')
                plt.plot(diff)

            if col >= 700:
                pre_y, post_y = 100, 100
            elif col > 100:
                pre_y, post_y = 0, 100
            else:
                pre_y, post_y = 0, 50
                
                
            new_psf, scale = adjust_height_32(diff, new_psf, y2, show_fig, pre_y=pre_y, post_y=post_y)

            if show_fig == 1:
                plt.figure('stage 3 psf after scaling')
                plt.plot(new_psf,'g--')
                plt.plot(diff)

            # if col >=750:#< 750 we can get errors in cross correlation with the o2 psf misplaced
            if col >=1000:
                new_psf, shift = cross_corr2(diff, new_psf)
                if show_fig == 1:
                    plt.figure('stage 3 further psf shift')
                    plt.plot(new_psf,'y--')
                new_psf_orig = np.roll(new_psf_orig, shift)

            new_psf = new_psf_orig * scale

            if show_fig == 1:
                plt.figure('stage 3 final scaled psf')
                plt.plot(new_psf,'g--')
                plt.plot(diff_orig)
                plt.plot([y2],[0],'ro')

            slice_ = new_img_array[:, col].copy()
            model = slice_.copy()
            model[y+100:y+800] = slicex[y+100:y+800]
            slice_[y2+250:] = 0
            model[y2+250:] = 0

            if show_fig == 1:
                plt.figure('stage 3 slice 0 and slice - both smoothed')
                plt.plot(model, label='model')
                plt.plot(slice_, '--', label='original data')
                plt.legend()

            o1_slice = new_img_array[:, col]-new_psf
            o2_slice = new_psf

            if show_fig == 1:
                plt.figure('model vs o1 slice (subtracted data)')
                plt.plot(model,label='model')
                plt.plot(o1_slice,label='o1 slice')
                plt.legend()

                plt.figure('o1 psf and o2 psf separated')
                plt.plot(o1_slice)
                plt.plot(o2_slice)
                plt.plot(new_img_array[:,col],'--')
                
                ccccc

            o1_slice_array[:, col] = o1_slice
            o2_slice_array[:, col] = o2_slice

        o1_slice_array[:,1000:] = new_img_array[:,1000:]
        o2_slice_array[:,1000:] = new_img_array[:,1000:]

        sci2_1 = np.mean(o1_slice_array.reshape(-1,10,o1_slice_array.shape[1]),axis=1)
        sci2_1 = sci2_1[int(buffer0/mag):-int(buffer0/mag)]
        sci_arr_o1[ii][:200] = sci2_1

        sci2_2 = np.mean(o2_slice_array.reshape(-1,10,o2_slice_array.shape[1]),axis=1)
        sci2_2 = sci2_2[int(buffer0/mag):-int(buffer0/mag)]
        sci_arr_o2[ii][:200] = sci2_2

    return sci_arr_o1, sci_arr_o2



# =============================================================================
# adjust fwhm of the sample psf to the target psf
# =============================================================================
def stretch_array(arr, factor, midx):
    new_arr = np.zeros_like(arr)
    # center  = ((len(arr) - 1) / 2) 
    center  = midx
    for i in range(len(arr)):
        old_idx = center + (i - center) / factor
        if old_idx >= 0 and old_idx < len(arr):
            new_arr[i] = np.interp(old_idx, np.arange(len(arr)), arr)
    return new_arr

# =============================================================================
# add functions
# =============================================================================
def adjust_fwhm(slice, av_psf):
    x = np.arange(len(slice))
     
    idx0_2 = np.argwhere(slice>=slice.max()/2).T[0]
    fwhm2 = x[idx0_2[-1]]-x[idx0_2[0]]
    midx_2  = int( (x[idx0_2[-1]]+x[idx0_2[0]]) /2)
    print (fwhm2)  
    idx0_1 = np.argwhere(av_psf>=av_psf.max()/2).T[0]
    fwhm1 = x[idx0_1[-1]]-x[idx0_1[0]]
    midx  = int( (x[idx0_1[-1]]+x[idx0_1[0]]) /2)
    print (fwhm1)

    stretch = fwhm2/fwhm1
    # print (stretch)
    # xnew = np.linspace(x[0], x[-1], int(np.round(len(x)*stretch)) )
    # aa = np.interp(xnew, x, av_psf)
    # Example usage
    arr = av_psf # Example shape
    factor = stretch
    new_psf= stretch_array(arr, factor, midx)
    
    return new_psf, midx, midx_2, fwhm1, fwhm2, factor



def adjust_height(slice, new_psf,  show_fig=0, N=20):
    # max_n= 100
    # max_n= 10
    max_n= N
    idx = np.argsort(slice)[-max_n:]
    if show_fig==1:
        plt.figure('adjust height 1')
        plt.plot(idx, slice[idx], 'bo')
         
    idx1 = np.argsort(new_psf)[-max_n:]
    if show_fig==1:
        plt.plot(idx1, new_psf[idx1], 'ro')
         
    # av_max_s = np.mean(slice[idx])
    
    scale =np.median(slice[idx] /  new_psf[idx1]) 
   
    new_psf *= scale

    return new_psf, scale

def adjust_height_2(slice, new_psf, y, show_fig=0):
    idx = np.arange(y-100,y+100)
    if show_fig==1:
        plt.figure('adjust height 2')
        plt.plot(slice, 'b-')
  
        plt.plot(new_psf, 'r-')
        plt.plot(idx, slice[idx], 'bo')
  
        plt.plot(idx, new_psf[idx], 'ro')
         
    # av_max_s = np.mean(slice[idx])
    
    scale =np.median(slice[idx] /  new_psf[idx]) 
   
    new_psf *= scale

    return new_psf, scale

def adjust_height_3(slice, new_psf, y, show_fig=0, pre_y=100, post_y=100):
    idx = np.arange(y-pre_y,y+post_y)
    if show_fig==1:
        plt.figure('adjust height 2')
        plt.plot(slice, 'b-')
  
        plt.plot(new_psf, 'r-')
        plt.plot(idx, slice[idx], 'bo')
  
        plt.plot(idx, new_psf[idx], 'ro')
         
    # av_max_s = np.mean(slice[idx])
    
    scale =np.median(slice[idx] /  new_psf[idx]) 
   
    new_psf *= scale

    return new_psf, scale
    
    
 

def cross_corr(slice, new_psf):
    # Calculate cross-correlation using FFT
    correlation = signal.fftconvolve(new_psf[::-1], slice, mode='full')
    
    # Find the lag that maximizes the cross-correlation
    max_index = np.argmax(correlation) - len(new_psf) + 1
    
    # Roll new_psf to maximize cross-correlation
    new_psf = np.roll(new_psf, max_index)
    
    return new_psf, max_index

 
def scale_array(array_to_scale, target_array):
    def loss(scale_factor):
        scaled_array = array_to_scale * scale_factor
        return np.sum((scaled_array - target_array) ** 2)

    res = minimize_scalar(loss)
    return res.x, res.fun
 

def scale_wings(slice, new_psf, midx_2, fwhm2, q=1, show_fig=0):
    
      slice0 = slice*1
      new_psf0 = new_psf*1
      
      if show_fig==1:
          plt.figure('wing scaling - pre scaling')
          plt.plot(slice0)
          plt.plot(new_psf0)
          plt.plot([midx_2],[0],'ro')
   
      
      print (midx_2)
 
      x = np.arange(len(slice0))
      slice0[ int(midx_2-fwhm2*q): int(midx_2+fwhm2*q)] = np.nan
      new_psf0[int(midx_2-fwhm2*q): int(midx_2+fwhm2*q)] = np.nan
      
      slice0[np.argwhere(np.isnan(slice0))]=0
      new_psf0[np.argwhere(np.isnan(new_psf0))]=0
      new_psf0[np.argwhere(slice0==0)]=0

      if show_fig==1:
          plt.figure('wing scaling - pre scaling - post masking')
          plt.plot(slice0)
          plt.plot(new_psf0)
 
      scale_factor, loss = scale_array(new_psf0,slice0)
      print("Scale factor:", scale_factor)
      print("Loss:", loss)
      
      if show_fig==1:
          plt.figure('wing scaling - pre scaling - post scaling')
          plt.plot(slice0)
          plt.plot(new_psf0*scale_factor)
      
      # scaled_array = new_psf0 * scale_factor
      # # print("Scaled array:", scaled_array)
      
      # if show_fig==1:
      #     plt.plot(scaled_array, '--')
           
      new_psf =  new_psf*scale_factor
      
      if show_fig==1:
           plt.figure('wing scaling - pre scaling - post scaling 2')
           plt.plot(new_psf0*scale_factor)
           plt.plot(new_psf)

      
      # create a slice with the new wings from scaled psf but the old core - unsure if we need this
      new_slice = new_psf*1
      val = slice[int(midx_2-fwhm2*q)]
      idx = np.argwhere(slice >=val).T[0]
      # new_slice[ int(midx_2-fwhm2*q): int(midx_2+fwhm2*q)] = slice[int(midx_2-fwhm2*q): int(midx_2+fwhm2*q)]
      new_slice[ int(midx_2-fwhm2*q): idx[-1]] = slice[int(midx_2-fwhm2*q): idx[-1]]

      #not sure we need new_slice - better for wing subtraction to use the scaled PSF as the core subtraction is not important for the adjacent order
      
      # replace_idx = np.arange(int(midx_2-fwhm2*q), int(midx_2+fwhm2*q),1)
      replace_idx = np.arange(int(midx_2-fwhm2*q), idx[-1],1)

      
      if show_fig==1:
          plt.figure('wing scaling - scaled psf and new slice')
          plt.plot(new_psf, '--')   
          plt.plot(new_slice, '-')  
          
          
     
      return new_psf, new_slice, scale_factor, replace_idx
  
    
def fft_shift(arr, shift):
    
    freq = np.fft.fftfreq(len(arr))
    shifted_arr = np.real(np.fft.ifft(np.fft.fft(arr) * np.exp(-2j * np.pi * freq * shift)))
    return shifted_arr
    
def interpolate_psf(psf, wav, new_wav):
    f = interp1d(wav, psf, kind='cubic', fill_value="extrapolate")
    return f(new_wav)

def chi_sq(x, slice0, slice_, new_psf):
    return np.sum((slice0 - (slice_ - new_psf * x))**2)

def chi_sq2(x, slice, new_psf):
    return np.sum((slice - (slice - new_psf * x))**2)

def process_slice_new(slice, col, ypos_, ypos_2, av_psf, buffer0, new_img_array, order='o1', show_fig=0, scale_x=1) :

    y2 =  int(np.round(ypos_2[col]))+buffer0
    y =  int(np.round(ypos_[col]))+buffer0
    
    box = 50; bbox= np.ones(box)/box
    slice = np.convolve(slice, bbox,'same')
    
    model_psf = np.copy(av_psf)
    smoothed_model_psf = np.convolve(av_psf, bbox,'same')
   
    if order == 'o1':
        yb = y2
        ya = y
    else:
        yb = y
        ya = y2
    
    if show_fig==1:
        plt.figure('initial state %s'%(order))
        plt.plot(slice)
        plt.plot([y],[0] ,'ro')
        plt.plot([y2],[0], 'ro')
    

    slice [yb-400: yb+400 ] = np.nan
    
    if order == 'o1':
        slice [ya+400: ] = np.nan
    else:
        slice [:ya-400 ] = np.nan
        
    print (ya, yb)
    
    slice[np.argwhere(np.isnan(slice))]=0
    
    print (show_fig)
    
    if show_fig==1:
        plt.figure('intial state after masking %s'%(order))
        plt.plot(slice, label='slice for fitting')
 
    # ============================================================================
    #       adjust fwhm
    # =============================================================================
    smoothed_model_psf, midx, midx_2, fwhm1, fwhm2, factor = adjust_fwhm(slice, smoothed_model_psf)
    model_psf= stretch_array(model_psf, factor, midx)
    smoothed_model_psf = np.convolve(model_psf, bbox, 'same')
 
    if show_fig==1:
        plt.figure('post fwhm adjust %s'%(order))
        plt.plot(slice)
        plt.plot(smoothed_model_psf)

    # =============================================================================
    #       adjust height  - first adjustment to allow cross-corr
    # =============================================================================
    new_psf, scale = adjust_height(slice, smoothed_model_psf,  show_fig=show_fig)
    new_model_psf = model_psf*scale
    new_psf = np.convolve(new_model_psf, bbox, 'same')

 
    if show_fig==1:
        plt.figure('post height adjust 1 %s'%(order))
        plt.plot(slice)
        plt.plot(new_psf)
 
   
    # =============================================================================
    #  now use cross correlation to adjust position of the new sample psf
    # =============================================================================
    new_psf, shift = cross_corr(slice, new_psf)
    new_model_psf = np.roll(new_model_psf, shift)
    new_psf = np.convolve(new_model_psf, bbox, 'same')

  
    if show_fig==1:
        plt.figure('post cross corr %s'%(order))
        plt.plot(slice)
        plt.plot(new_psf)
        plt.plot(new_model_psf)
        plt.legend()
        plt.show() 
        
    
            
    # based on the same points    
    new_psf, scale = adjust_height_2(slice, new_psf, ya, show_fig=show_fig)
    new_model_psf *scale
    new_psf = np.convolve(new_model_psf, bbox, 'same')


    if show_fig==1:
        plt.figure('post height adjust 2 %s'%(order))
        plt.plot(slice)
        plt.plot(new_psf)
        plt.plot(new_model_psf)
        plt.legend()
        plt.show() 
        
    
      
    # =============================================================================
    #       mask the core and scale the wings separately
    # =============================================================================
 
    # we look to see if the scaling works for the leading wing
    if order =='o1':
        slice_ = slice*1
        psf_ = new_psf*1
        slice_[ya:] =0
        idx = np.argwhere((slice_ < 0.025*slice.max()) & (slice_>0)).T[0]
        
        # print (idx)
        
        # xxxx
        slice_ = slice_[idx]
        psf_ = psf_[idx]
        # idx = np.argwhere(slice_==0)
        # psf_[idx] =0
        
        # plt.figure('slice_%s'%(order))
        # plt.plot(slice_)
        # plt.plot(psf_)
        
        # slice_[np.isnan(slice_)] =0
        # psf_[np.isnan(psf_)] =0
        
        if show_fig==1:
            plt.figure('height adjustment 3 a %s'%(order))
            plt.plot(slice)
            plt.plot(new_psf)
            plt.plot(idx, slice_, '.-')
            plt.plot(idx, psf_, '.-')
            
   
        scale = slice_/psf_
    
        idx = np.argwhere(scale <1)
        if len(idx) > 0.5*len(scale):
            scale_ = np.mean(scale)
            new_psf *=scale_
            new_model_psf *= scale_

              
            if show_fig==1:
                plt.figure('height adjustment 3 b %s'%(order))
                plt.plot(slice)
                plt.plot(new_psf)
                plt.plot(new_model_psf)
                
    if order =='o2':
         slice_ = slice*1
         psf_ = new_psf*1
         slice_[:ya] =0
         idx = np.argwhere((slice_ < 0.025*slice.max()) & (slice_>0)).T[0]   
         idx0 = int(ya+fwhm1)
         arr = idx
         x = idx0
         left = next((i for i, v in enumerate(arr) if v > x), len(arr))
         idx = arr[:left]
         
         slice_ = slice_[idx]
         psf_ = psf_[idx]
         
         if show_fig==1:
             plt.figure('height adjustment 3 a %s'%(order))
             plt.plot(slice)
             plt.plot(new_psf)
             plt.plot(idx, slice_, '.-')
             plt.plot(idx, psf_, '.-')

         scale = slice_/psf_
     
         idx = np.argwhere(scale <1)
         if len(idx) > 0.5*len(scale):
             scale_ = np.mean(scale)
             new_psf *=scale_
             new_model_psf *= scale_
               
             if show_fig==1:
                 plt.figure('height adjustment 3 b %s'%(order))
                 plt.plot(slice)
                 plt.plot(new_psf)
 
    # =============================================================================
    #       now subtract off new slice
    # =============================================================================
    # replace slice wings with model psf wings - this allows most of core to be subtracted off
    # wings are convolved but I don't think this matters.  Original slice should not be convolved.
    slice = np.copy(new_img_array[:,col])
    
    new_slice = slice*1  # to be subtracted
    if order == 'o1':
        # d = 1.5
        d = 1.0
    if order == 'o2':
        # d = 0.5
        d = 0.7
    new_slice[:int(ya-fwhm1*d)]= new_model_psf[:int(ya-fwhm1*d)]
    new_slice[int(ya+fwhm1*d):]= new_model_psf[int(ya+fwhm1*d):]
    
    if show_fig==1:
        plt.figure('trial subtraction %s'%(order))
        plt.plot(new_slice)
        plt.plot(slice, '--')
        
    if show_fig==1:
        plt.figure('trial subtraction -new slice %s'%(order))
 
        plt.plot(slice -new_slice, '-')

    if show_fig==1:
        plt.figure('trial subtraction -psf  %s'%(order)) # subttracting off slice is probably better
 
        plt.plot(slice - new_model_psf, '-')
        
        
    
    if order =='o1':  # to deal with oversubtraction on the inner wing - not really issue for o2
        sec = np.arange(int(ya+fwhm1*0.8), int(ya+fwhm1*1.4))
        slice_test = np.copy(slice)
        slice_test = np.convolve(slice_test, bbox, 'same')
        new_psf = np.convolve(new_model_psf, bbox, 'same')
        
        if show_fig==1:
            plt.figure('height adjustment 4 test samples')
            plt.plot(slice_test)
            plt.plot(new_psf)
            plt.plot(sec, slice_test[sec], 'r.')
            
        scale = slice_test[sec]/new_psf[sec]
        print (scale)
        
        idx = np.argwhere(scale < 1).T[0]
        if len(idx)/len(scale) > 0.5:
   
            scale_ = scale[idx]
            print (scale_)
            scale_ = np.mean(scale_)
            new_psf *=scale_
            new_model_psf *=scale_
            
            slice2  = slice - new_model_psf
            
            if show_fig==1:
    
                plt.figure('slice and new psf after further scaling')
    
                plt.plot(slice, 'r-', label='original slice %s'%(order))
                plt.plot(new_model_psf, 'b--', label='new psf')
                plt.legend()
                plt.show()  
    
    #redefine and subtract
    slice = np.copy(new_img_array[:,col])       
    new_slice = np.copy(slice)
    new_slice[:int(ya-fwhm1*d)]= new_model_psf[:int(ya-fwhm1*d)]
    new_slice[int(ya+fwhm1*d):]= new_model_psf[int(ya+fwhm1*d):]      
    slice2  = slice - new_slice   # not convolved but has the subtraction of the convolved wing - I think this is okay since the core is not be subtracted and convolved wing is about same as non-convolved
 
    if show_fig==1:
        plt.figure('final slice2 %s'%(order))
        plt.plot(slice2)
        plt.plot(slice, '--') 
 
   
    return slice2
  
       
def process_slice_orig(slice, col, ypos_, ypos_2, av_psf, buffer0, new_img_array, order='o1', show_fig=0, scale_x=1) :

    y2 =  int(np.round(ypos_2[col]))+buffer0
    y =  int(np.round(ypos_[col]))+buffer0
    
    box = 50; bbox= np.ones(box)/box
    slice = np.convolve(slice, bbox,'same')
    
    av_psf = np.convolve(av_psf, bbox,'same')
 

    if order == 'o1':
        yb = y2
        ya = y
    else:
        yb = y
        ya = y2
    
    if show_fig==1:
        plt.figure('initial state %s'%(order))
        plt.plot(slice)
        plt.plot([y],[0] ,'ro')
        plt.plot([y2],[0], 'ro')
 
    slice [yb-400: yb+400 ] = np.nan
    
    if order == 'o1':
        slice [ya+400: ] = np.nan
    else:
        slice [:ya-400 ] = np.nan
        
    print (ya, yb)
    
    slice[np.argwhere(np.isnan(slice))]=0
    
    print (show_fig)
    
    if show_fig==1:
        plt.figure('intial state after masking %s'%(order))
        plt.plot(slice, label='slice for fitting')
 
    # ============================================================================
    #       adjust fwhm
    # =============================================================================
    new_psf, midx, midx_2, fwhm1, fwhm2,factor = adjust_fwhm(slice, av_psf)
 
    if show_fig==1:
        plt.figure('post fwhm adjust %s'%(order))
        plt.plot(slice)
        plt.plot(new_psf)
        
       
    # =============================================================================
    #       adjust height  - first adjustment to allow cross-corr
    # =============================================================================
    new_psf, scale = adjust_height(slice, new_psf,  show_fig=show_fig)
   
 
    if show_fig==1:
        plt.figure('post height adjust 1 %s'%(order))
        plt.plot(slice)
        plt.plot(new_psf)
        
 
    # =============================================================================
    #  now use cross correlation to adjust position of the new sample psf
    # =============================================================================
    new_psf, shift = cross_corr(slice, new_psf)
  
    if show_fig==1:
        plt.figure('post cross corr %s'%(order))
        plt.plot(slice)
        plt.plot(new_psf)
        plt.legend()
        plt.show() 
            
        
    new_psf, scale = adjust_height_2(slice, new_psf, ya, show_fig=show_fig)

    if show_fig==1:
        plt.figure('post height adjust 2 %s'%(order))
        plt.plot(slice)
        plt.plot(new_psf)
        
    
   
           
    # =============================================================================
    #       mask the core and scale the wings separately
    # =============================================================================
  
    if order =='o1':
        slice_ = slice*1
        psf_ = new_psf*1
        slice_[ya:] =0
        idx = np.argwhere((slice_ < 0.025*slice.max()) & (slice_>0)).T[0]
        
        # print (idx)
        
        # xxxx
        slice_ = slice_[idx]
        psf_ = psf_[idx]
        # idx = np.argwhere(slice_==0)
        # psf_[idx] =0
        
        # plt.figure('slice_%s'%(order))
        # plt.plot(slice_)
        # plt.plot(psf_)
        
        # slice_[np.isnan(slice_)] =0
        # psf_[np.isnan(psf_)] =0
        
        if show_fig==1:
            plt.figure('height adjustment 3 a %s'%(order))
            plt.plot(slice)
            plt.plot(new_psf)
            plt.plot(idx, slice_, '.-')
            plt.plot(idx, psf_, '.-')
            
   
        scale = slice_/psf_
    
        idx = np.argwhere(scale <1)
        if len(idx) > 0.5*len(scale):
            scale_ = np.mean(scale)
            new_psf *=scale_
              
            if show_fig==1:
                plt.figure('height adjustment 3 b %s'%(order))
                plt.plot(slice)
                plt.plot(new_psf)
                
    if order =='o2':
         slice_ = slice*1
         psf_ = new_psf*1
         slice_[:ya] =0
         idx = np.argwhere((slice_ < 0.025*slice.max()) & (slice_>0)).T[0]   
         idx0 = int(ya+fwhm1)
         arr = idx
         x = idx0
         left = next((i for i, v in enumerate(arr) if v > x), len(arr))
         idx = arr[:left]
         
         slice_ = slice_[idx]
         psf_ = psf_[idx]
         
         if show_fig==1:
             plt.figure('height adjustment 3 a %s'%(order))
             plt.plot(slice)
             plt.plot(new_psf)
             plt.plot(idx, slice_, '.-')
             plt.plot(idx, psf_, '.-')

         scale = slice_/psf_
     
         idx = np.argwhere(scale <1)
         if len(idx) > 0.5*len(scale):
             scale_ = np.mean(scale)
             new_psf *=scale_
               
             if show_fig==1:
                 plt.figure('height adjustment 3 b %s'%(order))
                 plt.plot(slice)
                 plt.plot(new_psf)
                 
 
    # =============================================================================
    #       now subtract off new slice
    # =============================================================================
    # replace slice wings with model psf wings - this allows most of core to be subtracted off
    # wings are convolved but I don't think this matters.  Original slice should not be convolved.
    slice = np.copy(new_img_array[:,col])
    
    new_slice = slice*1  # to be subtracted
    if order == 'o1':
        # d = 1.5
        d = 1.0
    if order == 'o2':
        # d = 0.5
        d = 0.7
    new_slice[:int(ya-fwhm1*d)]= new_psf[:int(ya-fwhm1*d)]
    new_slice[int(ya+fwhm1*d):]= new_psf[int(ya+fwhm1*d):]
    
    if show_fig==1:
        plt.figure('trial subtraction %s'%(order))
        plt.plot(new_slice)
        plt.plot(slice, '--')
        
    if show_fig==1:
        plt.figure('trial subtraction -new slice %s'%(order))
 
        plt.plot(slice -new_slice, '-')

    if show_fig==1:
        plt.figure('trial subtraction -psf  %s'%(order)) # subttracting off slice is probably better
 
        plt.plot(slice - new_psf, '-')
        
    
    if order =='o1':  # to deal with oversubtraction on the inner wing - not really issue for o2
        sec = np.arange(int(ya+fwhm1*0.8), int(ya+fwhm1*1.4))
        slice_test = np.copy(slice)
        slice_test = np.convolve(slice_test, bbox, 'same')
        
        if show_fig==1:
            plt.figure('height adjustment 4')
            plt.plot(slice_test)
            plt.plot(new_psf)
            plt.plot(sec, slice_test[sec], 'r.')
            
        scale = slice_test[sec]/new_psf[sec]
        print (scale)
        
        idx = np.argwhere(scale < 1).T[0]
        if len(idx)/len(scale) > 0.5:
   
            scale_ = scale[idx]
            print (scale_)
            scale_ = np.mean(scale_)
            new_psf *=scale_
            
            slice2  = slice - new_psf
            
            if show_fig==1:
    
                plt.figure('slice and new psf after further scaling')
    
                plt.plot(slice, 'r-', label='original slice %s'%(order))
                plt.plot(new_psf, 'b--', label='new psf')
                plt.legend()
                plt.show()  
    
    slice = np.copy(new_img_array[:,col])       
    new_slice = np.copy(slice)
    new_slice[:int(ya-fwhm1*d)]= new_psf[:int(ya-fwhm1*d)]
    new_slice[int(ya+fwhm1*d):]= new_psf[int(ya+fwhm1*d):]      
    slice2  = slice - new_slice   # not convolved but has the subtraction of the convolved wing - I think this is okay since the core is not be subtracted and convolved wing is about same as non-convolved
    # slice2  = slice - new_psf
 
    if show_fig==1:
        plt.figure('final slice2 %s'%(order))
        plt.plot(slice2)
        plt.plot(slice, '--') 
 
   
    return slice2
  
    
  
# =============================================================================
#      for columns > 1000   some tweaking needed for masks but generally working
# =============================================================================
 
def run_stage1(sci, jexo_dic, show_fig=0):
    
    
    # aa='/Users/c1341133/Desktop/Subi_jwst_pipeline/fits_files/niriss/jw03557007001_04101_00001-COMBINED_division1_order9_nis_calints__18_7_25.fits'

    # hdul = fits.open(aa)
    # sci = hdul[1].data
    med = np.nanmedian(sci, axis=0)[:200]
    
 
    
    # print (np.nansum(med))


    # med = np.load('./med.npy')[:200]
    
    # print (np.nansum(med))
    
    # xxxx
    plt.figure('sci median')
    plt.imshow(med, aspect='auto', vmax = 20)
    
    
     
    img_array=med
    # F_y
    # mag = F_y/F_x 
    mag = 10
    # new_img_array = np.zeros((int(np.round(img_array.shape[1]*mag)), img_array.shape[1], img_array.shape[2]))
    yin = np.linspace(-1.0, 1.0, img_array.shape[1])
    xin = np.linspace(-1.0, 1.0, img_array.shape[0])
    yout = np.linspace(-1.0, 1.0, img_array.shape[1])
    xout = np.linspace(-1.0, 1.0, mag*img_array.shape[0])
     
    
    f = interpolate.RectBivariateSpline(xin, yin, img_array)
    redata = f(xout, yout)
    # redata /= redata.sum()
    new_img_array = redata


   

    # plt.figure(3)
    # col = 1300
    # plt.plot(new_img_array[:,col])
    # plt.plot(np.linspace(0,new_img_array.shape[0], med.shape[0]), med[:,int(col/mag)], 'r--')

    # col = 9000
    # plt.plot(new_img_array[:,col])
    # plt.plot(np.linspace(0,new_img_array.shape[0], med.shape[0]), med[:,int(col/mag)], 'r--')

    mask = np.ones_like(med)
    aa = np.load('./niriss_wav_sol_%s_%s.npy'%(jexo_dic['obs_number'], jexo_dic['tag'])) #order 1
    xpos = aa[0]; ypos=aa[1];wl=aa[2]  
    xpos_ = aa[0]; ypos_=mag*aa[1];wl_=aa[2]  

    aa2 = np.load('./niriss_wav_sol2_%s_%s.npy'%(jexo_dic['obs_number'], jexo_dic['tag'])) #order 2
    xpos2 = aa2[0]; ypos2=aa2[1];wl2=aa2[2]  
    xpos_2 = aa2[0]; ypos_2=mag*aa2[1];wl2=aa2[2] 

       
 
        
        
    plt.figure('new img array =')
    plt.imshow(new_img_array, aspect='auto')
        
    plt.plot(xpos_, ypos_, 'r--')
    idx = np.argwhere(wl<=1)
    # idx = np.argwhere(xpos_>12500)
    plt.plot(xpos_[idx], ypos_[idx], 'bx-')
     
    for i in range(0, len(wl), 150):
        # if np.round(wl[i] * 1) % 1 == 0:  # Check if wavelength is a multiple of 0.1
            plt.annotate(f"{wl[i]:.3f}", (xpos_[i], ypos_[i]), textcoords="offset points", xytext=(0, 10), ha='center', color='w', rotation=90)

    plt.plot(xpos_2, ypos_2, 'r--')
    for i in range(0, len(wl2), 150):
        # if np.round(wl[i] * 1) % 1 == 0:  # Check if wavelength is a multiple of 0.1
            plt.annotate(f"{wl2[i]:.3f}", (xpos_2[i], ypos_2[i]), textcoords="offset points", xytext=(0, 10), ha='center', color='w', rotation=90)
 


    plt.xlabel('X')
    plt.ylabel('Y')
 
    
    make_new_av_psf = 1
    
    if make_new_av_psf == 1:
  
        
        ct=0
        slice_array=[]
        buffer0 =  400*mag
        buffer00 = buffer0*1
        buffer = np.zeros((buffer0, new_img_array.shape[1])) 
        new_img_array = np.vstack((buffer, new_img_array, buffer))
    
        #loop through columns
    
        for ii in idx[:-10]:
            print ('ppp', ii)
            i = int(ii)
            col = int(xpos_[i])
            y =  int(np.round(ypos_[i]))+buffer0
            slice = new_img_array[:,col]
            
            # just plot the psf unshifted (some will have incomplete wings)
            # plt.figure('56')
            # plt.plot(slice)
            
            # slice[y-250:y+250] = np.nan
            
            # slice = np.roll(slice,-y)
            # slice  = np.where(slice==0, np.nan, slice)
            
            
            #shift them so they are all centred over the same point
            # shift = y-int(np.round(ypos_[0])+buffer0
                          
            shift = y-int(np.round(ypos_[0])+buffer0)
    
          
            
            slice = np.roll(slice,-shift)
    
         
            #show shifted psf slice
            # plt.figure('57')
            # plt.plot(slice)
           
            #stack these
            if ii == idx[0]:
                slice_array = slice
            else:
                slice_array = np.vstack((slice_array,slice))
            ct+=1
            
        # get the mean of the psfs
        av_psf = np.nanmean(slice_array, axis=0)    
        
        plt.plot(av_psf, 'k-')
        plt.figure('av_psf')
        plt.plot(av_psf, 'k-')
        
        # np.save('./av_psf_3.npy', av_psf)
        
        # ccccc
        
       
    
        
        ct=0
        slice_array=[]
        buffer0 =  200*mag
        # buffer0 =  400*mag
    
        new_img_array = redata
        buffer = np.zeros((buffer0, new_img_array.shape[1])) 
        new_img_array = np.vstack((buffer, new_img_array, buffer))
        
        #loop through columns again
        for ii in idx[:-10]:
        # for ii in [idx[0], idx[-10]]:
            i = int(ii)
            col = int(xpos_[i])
            y =  int(np.round(ypos_[i]))+buffer0
            slice = new_img_array[:,col]
            
            #plot the psf of the column
            # plt.figure('59')
            # plt.plot(slice)
            
            #exclude the core region
            slice[y-250:y+250] = np.nan
            
            #plot the new slice
            plt.plot(slice)
            
            #shift so that the psfs 
            
            slice = np.roll(slice,-y)
            
            # shift = y-int(np.round(ypos_[0]))+buffer0
            
            # slice = np.roll(slice,-shift)
    
            
            
            
            slice  = np.where(slice==0, np.nan, slice)
    
        
            # plt.figure('60')
            # plt.plot(slice)
             
             
             
            if ii == idx[0]:
                slice_array = slice
            else:
                slice_array = np.vstack((slice_array,slice))
            ct+=1
            
        av_slice = np.nanmean(slice_array, axis=0)    
        plt.plot(av_slice, 'k-')
        
        plt.figure('av_slice')
        plt.plot(av_slice, 'k-')
        
        av_slice_psf = av_psf*1
        y_mid  = int(np.round(ypos_[0]))+buffer0
        av_slice_psf[y_mid-250:y_mid+250] =np.nan
        
        plt.plot(av_slice_psf, 'b-')
        
       
    
        
        
        
        
         
        plt.figure('wings')
    
        plt.plot(av_slice, 'k-')
       
        rw = av_slice[:1220]
        lw = av_slice[5469:]
        plt.plot(rw, 'r-')
        plt.plot(lw, 'b-')
    
        plt.figure('wings 2')
        buffer = 25
        rw = np.hstack((np.zeros(buffer), rw))
        plt.plot(rw, 'r-')
        plt.plot(lw[::-1], 'b-')
        aa = len(rw)-len(lw)
        lw = np.hstack((lw[::-1], np.zeros(aa))) 
        plt.plot(lw, 'g-')
        lw = np.where(lw==0,np.nan, lw)
        rw = np.where(rw==0,np.nan, rw)
        aw = np.nanmean(np.vstack((lw,rw)), axis=0)
         
        plt.figure('wings 3')
        plt.plot(rw, 'r-')
        plt.plot(lw, 'b-')
        plt.plot(aw, 'g.-')
        
        # #for toi1231b    
        # # Indices to remove and interpolate
        # idx = np.arange(530, 580)
        # x = np.delete(np.arange(len(aw)), idx)
        # # y: corresponding data values excluding the range to be interpolated
        # y = np.delete(aw, idx)
        # # Points where we want interpolated values (indices 530 to 559)
        # x_new = idx
        # # Perform linear interpolation
        # interpolated_values = np.interp(x_new, x, y)
        # # Replace the original values with interpolated values
        # aw[idx] = interpolated_values    
        # aw = np.delete(aw, np.arange(1150, len(aw)))
            
        
        
        box =10; bbox = np.ones(box)/box
        aw = np.convolve(aw,bbox, 'same')[int(box/2):-int(box/2)]
        aw = np.convolve(aw,bbox, 'same')[int(box/2):-int(box/2)]
        
        plt.plot(aw, 'm-')
        
        
        plt.figure('wings 4')
        plt.plot(aw, 'm-')
       
        aw[np.argwhere(np.isnan(aw))]=0
        
        print (aw.max())
        
    # =============================================================================
    # 
    # =============================================================================
        idx = np.argwhere(av_psf>=aw.max()).T[0]
        idx1 = idx[0]; idx2 =idx[-1]
        print (idx)
        
        rw = aw[~np.isnan(aw)] 
        rw = aw[~(aw==0)]
        
        plt.plot(rw, 'r-')
        
        lw = rw[::-1]
        plt.plot(lw)
        
         
        plt.figure('av psf 3')
        plt.plot(av_psf)
        
        av_psf[idx2+1:idx2+1+len(rw)] = rw
        
        lw = av_psf[idx2+1:idx2+1+2000]
        lw = lw[::-1]
        
        av_psf[idx1-len(lw):idx1] = lw
        
        plt.figure('av psf 3')
        plt.plot(av_psf, '--')
        
        
        plt.figure('av psf final')
        
        plt.plot(av_psf, '-')
        
    else:
        #use average psf from 3 obs and with left wing extended.
        av_psf = np.load('./psf_av.npy')
    
    
 
    # =============================================================================
    # we have now created an average psf with wings
    # =============================================================================
     
    # =============================================================================
    # 1. subtract off 1st order wings for columns >1000 to clean 2nd order
    # =============================================================================

    
    new_img_array = redata
    buffer0 =  400*mag
    buffer = np.zeros((buffer0, new_img_array.shape[1])) 
    new_img_array = np.vstack((buffer, new_img_array, buffer))

     


 
    # idx2 = np.argwhere((wl2>0.65 )&(wl2<=1.1))
    # idx2 = np.argwhere((wl2>0.65 )&(wl2<=0.9))
    # idx2 = np.argwhere((wl2>0.65 )&(wl2<=0.93))
    # idx = np.argwhere(xpos_>12500)

    idx2 = np.arange(1000, xpos[-1]).astype(int)

    start_col = 1000      
    # end_col = 1750
    end_col = 1600
     
    slice = 1*new_img_array[:,start_col]
    plt.figure(); plt.plot(slice)
     
    n_col = end_col -start_col
    
    o1_psf_array = np.zeros((len(slice), n_col))
    o2_psf_array = np.zeros((len(slice), n_col))
    
    
    
    #xxxx
    ct=-1
    for col in range(start_col,start_col+n_col):
        # col =1600
        
        # col = 1250
        o1idx = col-4
        o2idx=col
        o1_wl = wl[o1idx]
        o2_wl = wl2[o2idx]
        
        print ('wavelength o1: %s'%(o1_wl))
        print ('wavelength o2: %s'%(o2_wl))
        
   

        new_img_array = redata
        buffer0 =  400*mag
        buffer = np.zeros((buffer0, new_img_array.shape[1])) 
        new_img_array = np.vstack((buffer, new_img_array, buffer))
 
        ct+=1
        print ('======col %s======='%(col))
    
        slice = np.copy(new_img_array[:,col])
 
        av_psf1 =av_psf2 = av_psf
        
        
        process_slice = process_slice_new 
        # process_slice = process_slice_orig
  
        
        scale_x=""
        #input slice is the original column; output is the o1 wing-subtracted o2 psf (with noise over o1)
        slice  = process_slice(slice, col,ypos_, ypos_2, av_psf1, buffer0, new_img_array,  order ='o1', show_fig=0, scale_x= scale_x)
    
        #input slice is the o1 wing-subtracted o2 psf (with noise over o1); output is the o2 wing-subtracted o1 psf (with noise over o2)
        slice  = process_slice(slice, col, ypos_, ypos_2, av_psf2, buffer0, new_img_array, order ='o2', show_fig=0, scale_x= scale_x)
        
        #input slice o2 wing-subtracted o1 psf (with noise over o2); output is the o1 wing-subtracted o2 psf (with noise over o1)
        slice  = process_slice(slice, col, ypos_, ypos_2, av_psf, buffer0, new_img_array, order ='o1',  show_fig=0, scale_x= scale_x)
        slice_o2 = slice*1  
        
        
 
        slice  = process_slice(slice, col, ypos_, ypos_2, av_psf, buffer0, new_img_array, order ='o2',  show_fig=0, scale_x= scale_x)
        slice_o1 = slice*1
        
    
        
        o1_psf_array[:,ct] =slice_o1 
        o2_psf_array[:,ct] =slice_o2 
      
        # show_fig=1
        if show_fig==1:
        
            plt.figure('slice final')
            plt.plot(slice_o1)
            plt.plot(slice_o2)
            plt.plot(new_img_array[:,col], '--')

            # plt.plot(slice_o1+slice_o2, '--')
            scale_x=""
            plt.figure('slice2 final %s'%(scale_x))
            plt.plot(new_img_array[:,col])
            plt.plot(slice_o1+slice_o2, '--')
            
         
    return  o1_psf_array, o2_psf_array


def run_stage2(o1_psf_array, sci, jexo_dic, show_fig=0):  
    
    mag = 10
    
    
    aa = np.load('./niriss_wav_sol_%s_%s.npy'%(jexo_dic['obs_number'], jexo_dic['tag'])) #order 1
    xpos = aa[0]; ypos=aa[1];wl=aa[2]  
    xpos_ = aa[0]; ypos_=mag*aa[1];wl_=aa[2]  
    
    aa2 = np.load('./niriss_wav_sol2_%s_%s.npy'%(jexo_dic['obs_number'], jexo_dic['tag'])) #order 1
    xpos2 = aa2[0]; ypos2=aa2[1];wl2=aa2[2]  
    xpos_2 = aa2[0]; ypos_2=mag*aa2[1];wl2=aa2[2]  
    
    med = np.nanmedian(sci, axis=0)[:200]
    
  
    plt.figure(1)
    plt.imshow(med, aspect='auto', vmax = 20)


    img_array=med
 
    # new_img_array = np.zeros((int(np.round(img_array.shape[1]*mag)), img_array.shape[1], img_array.shape[2]))
    yin = np.linspace(-1.0, 1.0, img_array.shape[1])
    xin = np.linspace(-1.0, 1.0, img_array.shape[0])
    yout = np.linspace(-1.0, 1.0, img_array.shape[1])
    xout = np.linspace(-1.0, 1.0, mag*img_array.shape[0])
     
    
    f = interpolate.RectBivariateSpline(xin, yin, img_array)
    redata = f(xout, yout)
    # redata /= redata.sum()
    new_img_array = redata
 
    buffer0 =  400*mag
    buffer00 = buffer0*1
    buffer = np.zeros((buffer0, new_img_array.shape[1])) 
    new_img_array = np.vstack((buffer, new_img_array, buffer))

    
    col = 1000
    i = col-1000
    plt.figure('100')
    plt.plot(o1_psf_array[:,i])
    idx = col-4
    y =  ypos_[idx]+buffer0
    plt.plot([y],[0], 'o')
    
    
    

    # Define the target y value
    target_y = ypos_[1000-4]
    
    # Shift each PSF
    shifted_o1_psf_array = np.zeros((new_img_array.shape[0],1000))
    
 
    # =============================================================================
    #  1. shift psfs to a common centre
    # =============================================================================
    for i in range(0,1000,1):
        
        # i= 595

        col = 1000+i
        idx = col-4
        
        # if i <750:
        if i <600:
            psf = o1_psf_array[:,i]
        else:
            psf = new_img_array[:,col]
            psf[5500:]=0
        shift =  (ypos_[idx]+buffer0)- (ypos_[1000-4]+buffer0)
        
        y =  ypos_[idx]+buffer0
        
        plt.figure('unshifted psf')
        plt.plot(psf)
        
         
        # print (y, shift)
     
       
        shifted_psf = fft_shift(psf, -shift)  
        
        plt.figure('101')
        # plt.plot(psf)
        plt.plot(shifted_psf, '-')
        shifted_o1_psf_array[:,i] = shifted_psf 
        plt.plot([y],[0], 'o')
        
         
    psf_array = shifted_o1_psf_array 
    
    
    

    # =============================================================================
    #  1. interpolate to the o2 wavelength grid
    # =============================================================================
    
    new_wav = wl2[0:1000]
    wav = wl[1000-4: 2000-4]
    
    new_psf_array = np.apply_along_axis(interpolate_psf, 1, psf_array, wav, new_wav)
    
    for i in range(0,1000,1):
        plt.figure('interpoloated psf array for o2 in <1000 col region')
        # plt.plot(psf)
        plt.plot(new_psf_array[:,i], '-')
    
    # np.save('./new_psf_array_o2.npy', new_psf_array)
 
    return new_psf_array

  


def run_stage3(new_psf_array, sci, jexo_dic, o1_psf_array, o2_psf_array, show_fig=0):
    
    mag = 10
    
    #load the wavelength solutions
    aa = np.load('./niriss_wav_sol_%s_%s.npy'%(jexo_dic['obs_number'], jexo_dic['tag'])) #order 1
    xpos = aa[0]; ypos=aa[1];wl=aa[2]  
    xpos_ = aa[0]; ypos_=mag*aa[1];wl_=aa[2] 
    
    aa2 = np.load('./niriss_wav_sol2_%s_%s.npy'%(jexo_dic['obs_number'], jexo_dic['tag'])) #order 1
    xpos2 = aa2[0]; ypos2=aa2[1];wl2=aa2[2]  
    xpos_2 = aa2[0]; ypos_2=mag*aa2[1];wl2=aa2[2] 
    
    
    
 

   
    # med = np.nanmedian(sci, axis=0)
    
    #make the final image arrays
    sci_arr_o1 = np.zeros_like((sci))
    sci_arr_o2 = np.zeros_like((sci))
    
    
    
    # loop though all images
    from tqdm import tqdm
    seq = np.arange(sci.shape[0]).astype(int)
              
    for ii in tqdm(seq):
        # print (ii)
        img = sci[ii][:200]
        
        if show_fig==1:
            plt.figure(1)
            plt.imshow(img, aspect='auto', vmax = 20)
        
        img_array=img*1
         
        # oversample the image in the y direction and add buffers 
        
        # new_img_array = np.zeros((int(np.round(img_array.shape[1]*mag)), img_array.shape[1], img_array.shape[2]))
        yin = np.linspace(-1.0, 1.0, img_array.shape[1])
        xin = np.linspace(-1.0, 1.0, img_array.shape[0])
        yout = np.linspace(-1.0, 1.0, img_array.shape[1])
        xout = np.linspace(-1.0, 1.0, mag*img_array.shape[0])
        
        f = interpolate.RectBivariateSpline(xin, yin, img_array)
        redata = f(xout, yout)
        # redata /= redata.sum()
        new_img_array = redata
        
        buffer0 =  400*mag
        buffer = np.zeros((buffer0, new_img_array.shape[1])) 
        
        new_img_array = np.vstack((buffer, new_img_array, buffer))
        
        if show_fig==1:
            plt.figure(2)
            plt.imshow(new_img_array, aspect='auto')
         
        #prepare empty arrays to hold separated order images 
        o1_slice_array = np.zeros_like(new_img_array)
        o2_slice_array = np.zeros_like(new_img_array)
        
       
        # loop though first 1000 columns
        
        for col in range(0,1000):
            
            # print (col)
            
            #columsn with contaminants will fail : mask afterwards
            # col =900
            
            #obtain y position of trace for o1 and o2 in this column
            idx_o1=col-4
            idx_o2=col*1
            # since no solution for o1 in first 4 colums will estrimate by using the position for col[3]
            if idx_o1<0: idx_o1=4
            
            y2 =  int(np.round(ypos_2[idx_o2]))+buffer0  #for o2
            y =  int(np.round(ypos_[idx_o1]))+buffer0  # for o1
            
            
            # obtain the wavelength for the column for o2
            wav = wl2[idx_o2]
            # print (wl[idx1], wl2[idx2])

            # obtain the column slice 
            slice = new_img_array[:, col]*1
            
            if show_fig==1:
                plt.figure('stage 3 initial slice col = %s'%(col))
                plt.plot(slice)
                plt.plot([y2],[0], 'ro')
                plt.plot([y],[0], 'bo')
                
                 
                
            # smooth slice with a box function

            box = 50
            bbox = np.ones(box)/box
            # slice_new = np.convolve(slice, bbox, 'same')
            slice_new = slice*1
            if show_fig==1:
                plt.figure('stage 3 initial slice smoothed col=%s'%(col))
                plt.plot(slice)
                # plt.plot(o1_slice)
                plt.plot(slice_new, '--')
                plt.plot([y2],[0], 'ro')
                plt.plot([y],[0], 'bo')
            slice = slice_new *1
            
             
            # =============================================================================
            #  reverse o1 psf           
            # =============================================================================

            
            PSF = slice[y-400:y+400]
            idx = np.argwhere(PSF>=0.5*PSF.max()).T[0]
            idx = idx + y-400
            
            idx1 = np.arange(idx[0]-10,idx[0])
            idx2 = np.arange(idx[-1],idx[-1]+10)
            
   
            mid_point = (np.mean(idx2) + np.mean(idx1))/2
         
 
            # get a reversed version of the core 
            slicex = slice[::-1]
            y_rev = len(slicex )-1 - y
            PSF = slicex[y_rev-400:y_rev+400]
            idx = np.argwhere(PSF>=0.5*PSF.max()).T[0]
            idx = idx + y_rev-400
            
            idx1 = np.arange(idx[0]-10,idx[0])
            idx2 = np.arange(idx[-1],idx[-1]+10)
            
       
            mid_point_2 = (np.mean(idx2) + np.mean(idx1))/2
         
       
            shift = mid_point - mid_point_2
            
            shifted_slice = fft_shift(slicex, shift)  
            slicex = shifted_slice
            
            if show_fig==1:
                plt.figure('reversed slice')

                plt.plot(slice,'-')
                plt.plot(shifted_slice,'-')
                plt.plot([y2],[0], 'ro')    
                
        
 
            # we now subtract the reversed slice from the original slice; in the region of o2, this acts to subtract off the wing of o1 
            diff = slice-slicex
            diff_orig  = diff*1
            
            # convolve to prepare for PSF fitting
            diff = np.convolve(diff, bbox, 'same')
            
            diff = np.where(diff<0,0,diff)
            
            if show_fig==1:
                plt.figure('stage 3 diff')
                plt.plot(diff)
                plt.plot([y2],[0], 'ro')    
            
            # we mask off the regions around o2 in prep for fitting the matched intep0olated o1 psf
            if col >=700:
                diff[:y2-300]=0
                # diff[:y2]=0
                diff[y2+300:]=0
                pre_y = 100; post_y=100
            elif col>100:
                diff[:y2-0]=0
                # diff[y2+100:]=0
                diff[y2+200:]=0
                pre_y = 0; post_y=100
            else:
                diff[:y2-0]=0
                # diff[y2+100:]=0
                diff[y2+200:]=0
                pre_y = 0; post_y=50
            
            if show_fig==1:
                plt.figure('stage 3 order 2 after o1 wing subtraction')
                plt.plot(diff)
                plt.plot([y2],[0], 'ro')
                
                
                
                 
            # =============================================================================
            #     fit psf to o2     (convolved) convolution used for fitting psf but then use unconvolved for subtraction
            # =============================================================================
                
            # take the pre-made o1 PSF and find shift to get it the o2 position; apply the shift
            mid  = int(np.round(ypos_[1000-4]+buffer0))
            shift = y2-mid
             
            new_psf  = np.copy(new_psf_array[:,idx_o2])
            
            #apply the same smoothing function as to slice
            # new_psf = np.convolve(new_psf, bbox, 'same')
           
            new_psf_orig = np.copy(new_psf_array[:,idx_o2])
            
            new_psf = np.convolve(new_psf, bbox, 'same')
            
            
            new_psf = np.roll(new_psf, shift)
            new_psf_orig =  np.roll(new_psf_orig, shift)
                 
            # mask off to match the masked o2 in the slice; and match the height
            # new_psf[np.argwhere(diff==0)]=0
            
            if col >=700:
                new_psf[:y2-300]=0
                new_psf[y2+300:]=0
            elif col>100:
                new_psf[:y2-0]=0
                new_psf[y2+200:]=0
            else:
                new_psf[:y2-0]=0
                new_psf[y2+200:]=0
            
            
            if show_fig==1:
                plt.figure('new psf pre scaling')
                plt.plot(new_psf, 'g--')
                plt.plot(diff)
                
             
            new_psf, scale = adjust_height_3(diff, new_psf, y2, show_fig=1, pre_y=pre_y, post_y=post_y)

            if show_fig==1:
                plt.figure('stage 3 psf after scaling')
                plt.plot(new_psf, 'g--')
                plt.plot(diff)
                 
              
                
            #  further shift in postion for full psfs at cols>700
            if col >=700:
                new_psf, shift = cross_corr(diff, new_psf)
                
                if show_fig==1:
                    plt.figure('stage 3 further psf shift')
                    plt.plot(new_psf, 'y--')
            
                new_psf_orig =  np.roll(new_psf_orig, shift)
                     
            # =============================================================================
            #   now scale the full model psf and remove smoothing
            # =============================================================================
            new_psf  = new_psf_orig*scale
            
            if show_fig==1:
                plt.figure('stage 3 final scaled psf')
                plt.plot(new_psf, 'g--')
                plt.plot(diff_orig)
                plt.plot([y2],[0], 'ro')
              
            # =============================================================================
            #      subtract o2 wing from o1           
            # =============================================================================

            slice = new_img_array[:, col]*1
            model = slice*1
            if show_fig==1:
                plt.figure('stage 3 slice col =%s'%(col))
                plt.plot(slice)
                
            # slice0 = np.convolve(slice0, bbox, 'same')
            # slice = np.convolve(slice, bbox, 'same')
                
              
            # replace the wing with the reversed wing for this scaling approach = desired outcome slice0
            model[y+100:y+800] = slicex[y+100:y+800]
            
            # remove other bits 
            slice[y2+250:] = 0
            model[y2+250:] = 0
            
            #model is now the model of where the non-reversed slice - o2 PSF should be
            
            if show_fig==1:
                plt.figure('stage 3 slice 0 and slice - both smoothed')
                plt.plot(model, label = 'model to aim for')
                plt.plot(slice, '--', label = 'original data')
                plt.legend()
               
                
            #scale the model psf so that the original slice -  psf matches the model
            #ai version
            
            # it tries to make slice-psf as close to slice0 as possible by scaling psf
            res = minimize_scalar(lambda x: chi_sq(x, model, slice, new_psf), bounds=(0.001, 1.3), method='bounded')
            x_optimal = res.x
            # print(f"Optimal x: {x_optimal}")
            
            new_psf *= x_optimal
            
            # # restore the bits masked on original slice now that fitting done
            # slice[y2+250:]=new_img_array[:, col][y2+250:]*1
            
            # if show_fig==1:
            #     plt.figure('111h')
            #     plt.plot(slice)
            #     plt.plot(new_psf)
            #     plt.plot(new_psf+slice0, '--')
                
            #     plt.figure('slice - rescaled psf')
            #     plt.plot(slice-new_psf)
            #     plt.plot(slice, '--')
                
              
            slice  = new_img_array[:, col]*1  
            # subtract off the o2 psf
            o1_slice = slice-new_psf  # store this
            o2_slice = new_psf # not used
            
            if show_fig==1:
                plt.figure('model vs o1 slice (subtracted data)')
                plt.plot(model, label='model')
                plt.plot(o1_slice, label = 'o1 slice')
                plt.legend()
                
                plt.figure('o1 psf and o2 psf separated')
                plt.plot(o1_slice)
                plt.plot(o2_slice)
                plt.plot(slice, '--')
                
                plt.plot([y2],[0], 'ro')
                plt.plot([y],[0], 'ro')
                plt.plot([y+200],[0], 'bs')
                plt.plot([y-200],[0], 'bs')
                
                plt.figure('o1 psf and o2 psf added')
                plt.plot(o1_slice+o2_slice, '--')
                plt.plot(slice, '-')
                
                cccc
                

            o1_slice_array[:,col] = o1_slice
            o2_slice_array[:,col] = o2_slice
            
           
        o1_slice_array[:,1000:] = new_img_array[:,1000:]
        o2_slice_array[:,1000:] = new_img_array[:,1000:]
        
        sci2_1 = np.mean(o1_slice_array.reshape(-1, 10, o1_slice_array.shape[1]), axis=1)
        sci2_1  = sci2_1[int(buffer0/mag):-int(buffer0/mag)]
        sci_arr_o1[ii][:200]= sci2_1
    
        sci2_2 = np.mean(o2_slice_array.reshape(-1, 10, o2_slice_array.shape[1]), axis=1)
        sci2_2  = sci2_2[int(buffer0/mag):-int(buffer0/mag)]
        sci_arr_o2[ii][:200]= sci2_2
 
                
    return sci_arr_o1, sci_arr_o2
    

      

# =============================================================================
#     
# =============================================================================

# aa='/Users/c1341133/Desktop/Subi_jwst_pipeline/fits_files/niriss/jw03557007001_04101_00001-COMBINED_division1_order9_nis_calints__18_7_25.fits'

# hdul = fits.open(aa)
# sci = hdul[1].data

# print (np.nansum(sci))
 
def sep_spectra(sci, jexo_dic):
    
    med  = np.median(sci[:100],axis=0)
    
    plt.figure()
    plt.imshow(med, aspect='auto')
    

  
    o1_psf_array, o2_psf_array = run_stage1(sci, jexo_dic, show_fig=0)
    
     
    
    # plt.figure('after stage 1 o1')
    # plt.imshow(o1_psf_array, aspect='auto')
    # plt.figure('after stage 1 o2')
    # plt.imshow(o2_psf_array, aspect='auto')
    
    np.save('./o1_psf_array_%s_%s'%(jexo_dic['obs_number'], jexo_dic['tag']), o1_psf_array)
    np.save('./o2_psf_array_%s_%s'%(jexo_dic['obs_number'], jexo_dic['tag']), o2_psf_array)
    
    # xxxx
     

    
    
    # o1_psf_array = np.load('./o1_psf_array_%s_%s.npy'%(jexo_dic['obs_number'], jexo_dic['tag']))
    # o2_psf_array = np.load('./o2_psf_array_%s_%s.npy'%(jexo_dic['obs_number'], jexo_dic['tag']))
    
   
    # produce the interpolated psf array for use in stage 3  qqqq
    new_psf_array = run_stage2(o1_psf_array, sci, jexo_dic, show_fig=0)
    

    
    sci_o1, sci_o2 = run_stage3_v2(new_psf_array, sci, jexo_dic, o1_psf_array, o2_psf_array, show_fig=0)
    
    # sci_o1, sci_o2 = run_stage3(new_psf_array, sci, jexo_dic, o1_psf_array, o2_psf_array, show_fig=1)

 
     
    np.save('./sci_o1_%s_%s.npy'%(jexo_dic['obs_number'], jexo_dic['tag']), sci_o1)
    np.save('./sci_o2_%s_%s.npy'%(jexo_dic['obs_number'], jexo_dic['tag']), sci_o2)
    
    
    # sci_o1 = np.load('./sci_o1_%s_%s.npy'%(jexo_dic['obs_number'], jexo_dic['tag']))
    # sci_o2 = np.load('./sci_o2_%s_%s.npy'%(jexo_dic['obs_number'], jexo_dic['tag']))
  
    
   
    plt.figure('sci1')
    plt.imshow(sci_o1[0], aspect='auto')
    
    plt.figure('sci2')
    plt.imshow(sci_o2[0], aspect='auto')
 
    return sci_o1, sci_o2 
    
    

# sci2_o1, sci2_o2 = run(sci)

# print (np.nansum(sci2_o1), np.nansum(sci2_o2))


def zeroth_sub(sci, jexo_dic, f277w_calints_file):
    #load the f277 calints file and obtain its median f2
   
    # bb= '/Users/c1341133/Desktop/Subi_jwst_pipeline/fits_files/niriss/jw01353101001_04102_00001-COMBINED_order0_nis_calints__11_9_25_ssc.fits'

    
    # bb= '/Users/c1341133/Desktop/Subi_jwst_pipeline/fits_files/niriss/jw01353101001_04102_00001-COMBINED_order0_nis_calints__11_9_25_ssc.fits'
    bb= f277w_calints_file
    hdul = fits.open(bb)
    f2 = np.nanmedian(hdul[1].data, axis =0)
    

         
    
    plt.figure('f277 median')
    plt.imshow(f2, aspect='auto', vmin=0, vmax=20)
    
    
    
    
    #load the main calints file
    # bb = '/Users/c1341133/Desktop/Subi_jwst_pipeline/fits_files/niriss/jw01353101001_04101_00001-COMBINED_order4_nis_calints__11_9_25_ssc.fits'
    # hdul = fits.open(bb)
    # sci = hdul[1].data
    

    med = np.nanmedian(sci, axis =0)
    
    # np.save('/Users/c1341133/Desktop/Subi_jwst_pipeline/med1.npy', med)
    # ccc
    
    
    # plt.figure()
    # plt.imshow(data[0], aspect='auto', vmin=0, vmax=50)
    
    # cc = '/Users/c1341133/Desktop/Subi_jwst_pipeline/med1.npy'
    # med = np.load(cc)
    
    plt.figure('median of science images')
    plt.imshow(med, aspect='auto', vmin=0, vmax=20)
    
    # =============================================================================
    # define example section to be used that has a clean 0th order image in the science median=sec - must be changed per observation
    # =============================================================================
    
    # toi1231
    # xmin = 1050
    # xmax = 1090
    # ymin = 150
    # ymax = 200
    
    
    #wasp 17
    xmin = 1062
    xmax = 1095
    ymin = 144
    ymax = 190
    
    # lhs 1140
    # xmin = 1352
    # xmax = 1367
    # ymin = 212
    # ymax = 250
    
    
    sec = med[ymin:ymax,xmin:xmax]
    plt.figure('sec')
    plt.imshow(sec, aspect='auto', vmin=0, vmax=10)
    print (np.argmax(sec))
    
    
        

    
    
  
  
    
    # remove some more background
    bkg = med[ymin-5:ymin, xmin:xmax]
    bkg_med = np.nanmedian(bkg,axis=0)
    sec = sec-bkg_med
    
    # bkg = np.hstack((sec[:,0:5], sec[:,-5:]))
    # print (bkg.shape)
    # plt.figure('bkg')
    # plt.imshow(bkg, aspect='auto', vmin=0, vmax=50)
    # bkg = np.median(bkg, axis=1)
    # sec= sec-bkg[:, np.newaxis] 
    plt.figure('sec after bkg sub')
    plt.imshow(sec, aspect='auto', vmin=0, vmax=10)
    
    
    
     
    # get the corresponding example 0th order image in the f2 image=fsec
     
    fsec = f2[ymin:ymax,xmin:xmax]
    plt.figure('fsec')
    plt.imshow(fsec, aspect='auto', vmin=0, vmax=10)
    
    # for toi-1231b
    # fsec [10:42,9:14] = fsec[10:42,0:5]
    # fsec [35:42,14:15] = fsec[35:42,0:1]
    
    # fsec [31:42,14:16] = fsec[31:42,0:2]


    
    plt.figure('fsec2')
    plt.imshow(fsec, aspect='auto', vmin=0, vmax=10)
    
    
       
     
    # find the x and y maximum position on this, and its x and y shapes 
    a =np.unravel_index(np.argmax(fsec), fsec.shape)[0]
    b =np.unravel_index(np.argmax(fsec), fsec.shape)[1]
    c = fsec.shape[0]
    d = fsec.shape[1]
    
    # find the correction factor
    corr = sec/fsec
    

 
   
    
   
    plt.figure('scaled fsec')
    xx = corr*fsec
    plt.imshow(xx, aspect='auto', vmin=0, vmax=10)

    # produce a duplicate of f2 but with the first 500 columns masked
    f0 = f2*1
    f0[:,0:500]=0 # may have to change to higher values for some data sets (standard 500)
    f0[np.isnan(f0)]=0
    plt.figure('f0 before hot spot removal')
    plt.imshow(f0, aspect='auto', vmin=0, vmax=10)
    
    
    # remove hot spots on the f0 image
    for i in range(500, f0.shape[1]): # update for number of columns masked (standard 500)
          col = f0[:,i]
          # col = med[:,1243]
          # plt.plot(col, 'bo-') 
          med0 =[]; idx=[]
          box = 5
          for j in range(box,len(col)-box):
              rm = np.median(col[j-box:j+box+1])
              med0.append(rm)
              idx.append(j)
          med0 = np.array(med0)
          med0 = np.hstack((np.array([med0[0]]*box), med0, np.array([med0[-1]]*box)))
          # plt.plot(med0, 'g-')
          # idx = np.argwhere((col>(col2+col2*0.9)) & (col>0.5))
          idx = np.argwhere((col>(med0+med0*5)) & (col>0.5))
          # plt.plot(idx,col[idx], 'ro')  
          col[idx] = med0[idx]
          # plt.plot(idx,col[idx], 'yo')
          f0[:,i]=col
    plt.figure('f0 after hot spot removal')
    plt.imshow(f0, aspect='auto', vmin=0, vmax =5) 
     
    
    
    # we add a buffer to f0 to allow for 0th order images close to the edge to be dealt with 
    buffer = np.zeros((50,f0.shape[1]))
    f0 = np.vstack((buffer, f0, buffer))
    buffer2 = np.zeros((50,f0.shape[0])).T
    f0 = np.hstack((buffer2, f0, buffer2))
    
    plt.figure('f0 w buffer')
    plt.imshow(f0, aspect='auto', vmin=0, vmax=10)
    
     
    # we create a zero value array of the same size as f0 to hold scaled 0th order images from the science median = f2 blank
    f2_blank= np.zeros_like(f0)
    
    # we create a duplicate of f0 from which we can subtract off the previous 0th order image, leaving a new maximum 
    f0_copy = f0*1 
    
    
    # # # test removal of the example image
    # f2_blank_test = np.copy(f2_blank[50:-50, 50:-50])*1
    # max_scale = np.max(f0_copy[50:-50, 50:-50][ymin:ymax,xmin:xmax])/np.max(fsec) # the scale factor is this times the correction factor
    # f2_blank_test[ymin:ymax,xmin:xmax]+= fsec*max_scale*corr
    # img = sci[1]
    # plt.figure('blank test')
    # plt.imshow(f2_blank_test, aspect='auto', vmin=0, vmax=50)   
    
    # plt.figure('sci image before0')
    # plt.imshow(img, aspect='auto', vmin=0, vmax=10)   
    # img = img -f2_blank_test*1
    # plt.figure('sci image after0')
    # plt.imshow(img, aspect='auto', vmin=0, vmax=10)  
 
    fudge = 1
    # we loop thorough the 0th order images in order of brightness; the image is removed from f0_copy and the scaled 0th order image added to f2_blank
    # for i in range(40):  # estimate the number from f2
    # for i in range(50):  # estimate the number from f2 - toi1231b
    # for i in range(30):  # estimate the number from f2 - wasp 17
    
    for i in range(30):  # estimate the number from f2


        
        max_ = np.unravel_index(np.argmax(f0_copy), f0_copy.shape)
    
        max_scale = np.max(f0_copy)/np.max(fsec) # the scale factor is this times the correction factor
        
        # max_scale = np.median(f0_copy[max_[0]-a:  max_[0]-a+c,  max_[1]-b:max_[1]-b+d] / fsec)
        
        f0_copy[max_[0]-a:  max_[0]-a+c,  max_[1]-b:max_[1]-b+d] = 0
        
        f2_blank[max_[0]-a:  max_[0]-a+c,  max_[1]-b:max_[1]-b+d] += fsec*max_scale*corr*fudge
        
        # plt.figure('example')
        # plt.imshow(fsec*max_scale*corr, vmin=0, vmax=10)
         
        
    plt.figure('f2_blank populated')
    plt.imshow(f2_blank, aspect='auto', vmin=0, vmax=10)
 
    #  remove buffer  
    f2_blank = f2_blank[50:-50]
    f2_blank = f2_blank[:,50:-50]
    
    #  test on a science image
    img = sci[1]*1
    
    plt.figure('sci image before')
    plt.imshow(img, aspect='auto', vmin=0, vmax=20)   
    
    img = img -f2_blank*1
    
    plt.figure('sci image after')
    plt.imshow(img, aspect='auto', vmin=0, vmax=20) 

    
    
    
    
    for i in range(sci.shape[0]):
        sci[i] = sci[i]-f2_blank*1
    
    return sci
