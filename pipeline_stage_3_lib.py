#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  5 14:11:39 2024

@author: c1341133
"""

import pylightcurve as plc
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits

import pandas as pd

from astropy import units as u
import os
import matplotlib.cm as cm
import matplotlib.pyplot as plt

from exotic_ld import StellarLimbDarkening
import gp
import emcee
import copy
import pipeline_lib  
from pipeline_lib import bin_spectrum



from scipy import optimize 
 


def transit_model2(t, rat, t0, gamma0, gamma1, per, ars, inc, w, ecc, a, b, c, d, ldc_type='quad'):

    lc = plc.transit([gamma0, gamma1], rat, per, ars, ecc, inc, w, t0, t, method=ldc_type)
    
    # syst = A*np.exp(-B*t)+ a*t + b
    
    syst = a*(1+ b*t + c*np.exp(-t/d))

 
    lc = lc * syst

    return lc


def load_file(data_dic, f, channel='miri'):
    

    if 'miri' in f:
        channel = 'miri'   
    if 'nrs' in f:
        channel = 'nirspec'
    if 'nis' in f:
            channel = 'niriss'
    
    hdul = fits.open(f)
 
    int_times = hdul[1].data
    slc = hdul[2].data
    wav = hdul[3].data
    var = (hdul[4].data)**2
    
    bjd = int_times['int_mid_BJD_TDB']
    mjd= int_times['int_mid_MJD_UTC']
 
  
    aa= []
    for i in range(len(bjd)):
        if bjd[i] in aa:
            pass
        else:
            aa.append(bjd[i])
    bjd = np.sort(aa)
 
 
    data_dic['slc']=slc; data_dic['var']=var; data_dic['bjd']=bjd
    data_dic['channel'] =  channel
    data_dic['wav'] =wav
    data_dic['mjd'] = mjd
    data_dic['bjd_orig'] = bjd*1
    
    
    return data_dic





def outliers(data_dic, alpha):
    slc = data_dic['slc']; var = data_dic['var']; bjd = data_dic['bjd']
    wlc = np.nansum(slc,axis=1)
    
    x = np.arange(len(wlc))
    plt.figure('outliers')
    plt.plot(x, wlc, 'bo')
    
    median = np.zeros_like(wlc)
    std = np.zeros_like(wlc)
    for i in range(len(wlc)):
            median[i] = np.median(wlc[i-10:i+10])
            std[i] = np.std(wlc[i-10:i+10])
    idx1 = np.where(wlc>median+std*alpha)
    idx2 = np.where(wlc<median-std*alpha)
    idx = np.hstack((idx1,idx2)).T
    plt.plot(x[idx], wlc[idx], 'r.')
    
    for i in idx:
        i = i.item()
        # print (i, slc.shape[0])
        if i < slc.shape[0]-1: # in case final case is an outlier
            slc[i] = (slc[i-1]+slc[i+1])/2
            var[i] = (var[i-1]+var[i+1])/2
    wlc = np.nansum(slc,axis=1)
    plt.plot(x, wlc, 'g.')
    data_dic['slc'] = slc; data_dic['var'] = var; data_dic['bjd'] = bjd
     
    return data_dic


# def cutoff(data_dic, cutoff):
#     if cutoff is not None:
#         slc = data_dic['slc']; var = data_dic['var']; bjd = data_dic['bjd']
#         x  = np.arange(0,slc.shape[0])
#         wlc =  slc.sum(axis=1)
#         plt.figure('cutoff')
#         plt.plot(x, wlc, 'g.')
         
#         if cutoff[1] == 0:
#             cutoff[1] = None
#         slc = slc[cutoff[0]:cutoff[1]]
#         var = var[cutoff[0]:cutoff[1]]
#         bjd = bjd[cutoff[0]:cutoff[1]]
#         wlc = np.sum(slc,axis=1)
#         x = x[cutoff[0]:cutoff[1]]
#         plt.figure('cutoff')
#         plt.plot(x, wlc, '.')
#         data_dic['slc'] = slc; data_dic['var'] = var; data_dic['bjd'] = bjd
#     return data_dic

def cutoff2(data_dic, idx):
    
    data_dic['slc_pre_cut'] =data_dic['slc']*1
    data_dic['var_pre_cut'] =data_dic['var']*1
    data_dic['bjd_pre_cut'] =data_dic['bjd']*1
    
    if idx is not None:
        slc = data_dic['slc']; var = data_dic['var']; bjd = data_dic['bjd']

        x  = np.arange(0,slc.shape[0])
        wlc = np.nansum(slc, axis=1)
        plt.figure('cutoff')
        plt.plot(x, wlc, 'g.')
        
        slc = np.delete(slc, idx, axis =0)
        var = np.delete(var, idx, axis =0)
        bjd = np.delete(bjd, idx)
        wlc = np.nansum(slc, axis=1)
        
      
        x = np.delete(x, idx)
        plt.figure('cutoff')
        plt.plot(x, wlc, '.')
        data_dic['slc'] = slc; data_dic['var'] = var; data_dic['bjd'] = bjd
        
     
    return data_dic


def time_bin(data_dic, time_bin):
    
    slc = data_dic['slc']; var = data_dic['var']; bjd = data_dic['bjd']
    plt.figure('wlc after time bin')
    wlc  =  np.nansum(slc, axis=1)
    plt.plot(bjd, wlc*time_bin, 'b.')
    idx  =  np.arange(0, slc.shape[0], time_bin)
    bjd =   (np.add.reduceat(bjd, idx)/  time_bin) [:-1]
    # mjd_binned =   (np.add.reduceat(mjd_mid_time, idx)/ time_bin) [:-1]
    slc  =  np.add.reduceat(slc, idx, axis=0)[:-1]
    var  =  np.add.reduceat(var, idx, axis=0)[:-1]
    wlc  =  np.nansum(slc, axis=1)
    wlc_var  = np.nansum(var, axis=1)
    print ('time_step (s): ', np.diff(bjd)[0]*24*60*60)   
    plt.errorbar(bjd, wlc, wlc_var**0.5, fmt='ro')
    data_dic['slc'] = slc; data_dic['var'] = var; data_dic['bjd'] = bjd
    return data_dic

def oot_section(data_dic, oot_times):
    slc = data_dic['slc']; var = data_dic['var']; bjd = data_dic['bjd']
    oot1 = np.argwhere(bjd>=oot_times[0])[0].item()
    oot2 = np.argwhere(bjd>=oot_times[1])[0].item()  
    idx_oot1 = np.arange(oot1)
    idx_oot2 = np.arange(oot2, slc.shape[0],1)
    idx_oot = np.hstack((idx_oot1,idx_oot2))
    data_dic['idx_oot'] = idx_oot
    return data_dic

def get_wlc(data_dic, wav_targ=None):
    slc = data_dic['slc']; var = data_dic['var']; bjd = data_dic['bjd']
    slc_pre_cut = data_dic['slc_pre_cut']; var_pre_cut = data_dic['var_pre_cut']; bjd_pre_cut = data_dic['bjd_pre_cut']
    
    wav = data_dic['wav']
    if wav_targ == None:
        wlc = np.nansum(slc, axis=1)
        wlc_var = np.nansum(var,axis=1)
        
        wlc_pre_cut = np.nansum(slc_pre_cut,axis=1)
        wlc_var_pre_cut = np.nansum(var_pre_cut,axis=1)

    else:
        idx = np.argwhere((wav<wav_targ[1]) &(wav>=wav_targ[0])).T[0]
        slc = slc[:,idx]
        print(var.shape)
        wav = wav[idx]
        var = var[:,idx]
        
        slc_pre_cut = slc_pre_cut[:,idx]
        var_pre_cut = var_pre_cut[:,idx]
        
        wlc = np.nansum(slc,axis=1)
        wlc_var = np.nansum(var,axis=1)
        
        wlc_pre_cut = np.nansum(slc_pre_cut,axis=1)
        wlc_var_pre_cut = np.nansum(var_pre_cut,axis=1)
        
    plt.figure('wlc after wavelength selection')
    plt.plot(bjd, wlc, '.')
    data_dic['wlc'] = wlc
    data_dic['wlc_var'] = wlc_var
    
    data_dic['wlc_pre_cut'] = wlc_pre_cut
    data_dic['wlc_var_pre_cut'] = wlc_var_pre_cut
    
    return data_dic

def wavelength_range(data_dic, wav_targ):
    slc = data_dic['slc']; var = data_dic['var']; bjd = data_dic['bjd']
    wav = data_dic['wav']

    idx = np.argwhere((wav<wav_targ[1]) &(wav>=wav_targ[0])).T[0]
    slc = slc[:,idx]
    wav = wav[idx]
    var = var[:,idx]
    wlc = np.nansum(slc,axis=1)
    plt.figure('wlc after wavelength selection')
    plt.plot(bjd, wlc, '.')
    data_dic['slc'] = slc; data_dic['var'] = var; data_dic['bjd'] = bjd
    data_dic['wav'] = wav
    return data_dic


def wavelength_bin(data_dic, bin_type='None', R=100, bin_size=10, wavgrid=None, colgrid=None):
    slc = data_dic['slc']; var = data_dic['var']; bjd = data_dic['bjd']
    wav = data_dic['wav']
    
    slc_pre_cut = data_dic['slc_pre_cut']; var_pre_cut = data_dic['var_pre_cut'];
    
    # data_dic['slc_orig'] = slc*1; data_dic['var_orig'] = var*1;data_dic['wav_orig'] = wav*1
    
    # data_dic['slc_orig'] =  slc_pre_cut*1; data_dic['var_orig'] = var_pre_cut*1;data_dic['wav_orig'] = wav*1

 
    if bin_type != 'None':
        
        var_pre_cut,_,_,_ = bin_spectrum(var_pre_cut, wav, R, bin_size, bin_type=bin_type, wavgrid = wavgrid, colgrid=colgrid)
        slc_pre_cut, _,_,_ = bin_spectrum(slc_pre_cut, wav, R, bin_size, bin_type=bin_type, wavgrid = wavgrid, colgrid=colgrid)
        
 
        var,_,_,_ = bin_spectrum(var, wav, R, bin_size, bin_type=bin_type, wavgrid = wavgrid, colgrid=colgrid)
        slc, wav, edges, idx = bin_spectrum(slc, wav, R, bin_size, bin_type=bin_type, wavgrid = wavgrid, colgrid=colgrid)
        

        
        if colgrid is not None:
            print (wav)
            
            
        
        print ('number of wavelength bins',  slc.shape[1])
        data_dic['slc'] = slc; data_dic['var'] = var; data_dic['bjd'] = bjd
        data_dic['wav'] = wav
        data_dic['wav_edges'] = edges
        data_dic['wav_idx'] = idx
        data_dic['slc_pre_cut'] = slc_pre_cut; data_dic['var_pre_cut'] = var_pre_cut


           
    else:
        print ('no wavelength binning...')
    return data_dic 


def wavelength_bin_random(slc, wav, var, bin_type='None', R=100, bin_size=10, wavgrid=None):

    if bin_type != 'None':
 
        var,_,_,_ = bin_spectrum(var, wav, R, bin_size, bin_type=bin_type, wavgrid = wavgrid)
        slc, wav, edges, idx = bin_spectrum(slc, wav, R, bin_size, bin_type=bin_type, wavgrid = wavgrid)

        return slc, wav, var
 
    else:
        print ('no wavelength binning...')
     

 
def load_ldc_file(data_dic, ldc_file):
    # slc = data_dic['slc']; var = data_dic['var']; bjd = data_dic['bjd']
    wav = data_dic['wav']
        
    if ldc_file !=None:
        print ('loading ldc file...')
        df = pd.read_fwf(ldc_file)
        ldc_wav = np.array(df['wave_eff'][1:]).astype(float)
        c1 =  (np.array(df['c1'][1:]).astype(float))
        c2 =  (np.array(df['c2'][1:]).astype(float)) 
        ldc = np.vstack((c1,c2))
 
    
        print ('binning ldc...')
        ldc_new = np.zeros((2, len(wav))) 
        plt.figure('ldc')
        for i in range(ldc.shape[0]):
            ldc_new[i]= (np.interp(wav, ldc_wav, ldc[i]))
            plt.plot(ldc_wav, ldc[i], '.-')
            plt.plot(wav, ldc_new[i], 'o-')
            
        data_dic['ldc'] = ldc_new ; data_dic['ldc_wav'] = wav


    return data_dic 



def get_ldc_exotic(data_dic, M_H, Teff, logg, mode):
    wav = data_dic['wav']
    
     
    # temp fix
    # if data_dic['channel'] == 'nirspec':
    #     idx  = np.argwhere((wav>=2.87)&(wav<=5.1769)).T[0]
    #     wav = wav[idx]
        
    if mode == 'JWST_NIRSpec_G395H': 

            idx  = np.argwhere((wav>=2.87)&(wav<=5.1769)).T[0]
            wav = wav[idx]
    
    idx = np.argwhere(wav<=2.81).T[0]
    wav = wav[idx]    
    
    wav_edges = (wav[1:]+ wav[:-1]) /2
    
    # quick solution for now... may be make more accurate
  
    edge1 = wav_edges[0]- np.gradient(wav_edges)[0]
    edge2 = wav_edges[-1]+ np.gradient(wav_edges)[-1]

    wav_edges = np.hstack((edge1, wav_edges, edge2))
    
    ld_data_path = 'exotic_ld_data'
 
    ld_model = 'kurucz'


    sld = StellarLimbDarkening(M_H=M_H, Teff=Teff, logg=logg,
                               ld_model=ld_model,
                               ld_data_path=ld_data_path,
                               interpolate_type="nearest")
    u1_list = []
    u2_list = []
    for i in range(len(wav)):
     
        wavelength_range = [wav_edges[i],   wav_edges[i+1]]*u.um
        wavelength_range = wavelength_range.to(u.Angstrom).value
        u1,u2 = sld.compute_quadratic_ld_coeffs(wavelength_range, mode)
        u1_list.append(u1)
        u2_list.append(u2)
 
    ldc_new = np.vstack((u1_list, u2_list))
     
    plt.figure('ldc')
 
    plt.plot(wav, ldc_new[0], 'o-')
    plt.plot(wav, ldc_new[1], 'o-')

        
    data_dic['ldc'] = ldc_new ; data_dic['ldc_wav'] = wav
    
    #wlc
    wavelength_range = [wav_edges[0],   wav_edges[-1]]*u.um
    wavelength_range = wavelength_range.to(u.Angstrom).value
    u1,u2 = sld.compute_quadratic_ld_coeffs(wavelength_range, mode)
    data_dic['wlc_ldc'] =[u1, u2]
    print ('wlc ldc', u1, u2)
    
    if mode == 'JWST_NIRSpec_G395H' or mode == 'JWST_NIRSpec_G235H':
        if data_dic['combine_wlc'] ==1 and mode == 'JWST_NIRSpec_G395H':
            wavelength_range =  [2.87,   5.1769]*u.um
            wavelength_range = wavelength_range.to(u.Angstrom).value
            u1,u2 = sld.compute_quadratic_ld_coeffs(wavelength_range, mode)
            data_dic['wlc_ldc'] =[u1, u2]
            print ('wlc combined ldc', u1, u2)
          
    return data_dic 





def proc_spec_file(root,label='', c='k', fmt='o', lm=0, alpha=1):
    
    if lm==1:
        aa =  np.load('%s_lm_fit.npy'%(root))
        
        
        wav =aa[0]
        rp = aa[1]
        rp_err =aa[2]
        depth = rp**2
        depth_plus = (rp+rp_err)**2
        depth_minus = (rp-rp_err)**2
        depth_err_plus = depth_plus- depth
        depth_err_minus =   depth- depth_minus
        
        plt.figure('pppp2')
        plt.plot( depth_err_plus)
        plt.plot( depth_err_minus)
        
        idx1  = np.where(depth_err_plus < 0)[0]
        idx2  = np.where(depth_err_minus < 0)[0]
        
        
        wav = np.delete(wav, idx1); wav = np.delete(wav, idx2)
        rp = np.delete(rp, idx1); rp = np.delete(rp, idx2)
        rp_err = np.delete(rp_err, idx1); rp_err = np.delete(rp_err, idx2)
        depth = np.delete(depth, idx1); depth = np.delete(depth, idx2)
        depth_err_plus = np.delete(depth_err_plus, idx1); depth_err_plus = np.delete(depth_err_plus, idx2)
        depth_err_minus = np.delete(depth_err_minus, idx1); depth_err_minus = np.delete(depth_err_minus, idx2)

  
        
        yerr = np.array(list(zip(depth_err_minus, depth_err_plus))).T
        plt.figure('spectrum', figsize=(19,8))
        plt.errorbar(wav, depth, yerr, fmt = fmt, color=c, label = label, alpha = alpha)
        
    
        R = wav/np.gradient(wav)
        print ('R min, max, av', R.min(), R.max(), R.mean())
        
        av_error = (depth_err_plus +depth_err_minus)/2
        half_bin_width  = np.gradient(wav)/2
        
        aa = np.vstack((wav, half_bin_width, depth, av_error)).T
        
        dic={}
        dic['wav'] = wav
        dic['spec'] = depth
        dic['yerr'] = yerr
        dic['half_bin_width'] = half_bin_width
        dic['av_error'] = av_error
        dic['all'] = aa

        
    else:
        wav =  np.load('%s_wav.npy'%(root))
        idx = np.argsort(wav)
        wav = wav[idx]
        R = wav/np.gradient(wav)
        print ('R min, max, av', R.min(), R.max(), R.mean())
        
        params = np.load('%s_params.npy'%(root))
        ex  = np.load('%s_exp_sampler_results.npy'%(root))[:,:,0]
        # spec = ex[:,1]
        # err_plus =  ex[:,2]- ex[:,1]
        # err_minus =  ex[:,1] - ex[:,0]
        # spec = spec[idx]
        # err_plus=err_plus[idx]
        # err_minus=err_minus[idx]
        spec = ex[:,1]**2
        err_plus =  ex[:,2]**2- ex[:,1]**2
        err_minus =  ex[:,1]**2 - ex[:,0]**2
        spec = spec[idx]
        err_plus=err_plus[idx]
        err_minus=err_minus[idx]
      
        yerr = np.array(list(zip(err_minus, err_plus))).T
        av_error = (err_plus +err_minus)/2
        half_bin_width  = np.gradient(wav)/2
        
        plt.figure('spectrum', figsize=(19,8))
        plt.errorbar(wav, spec, yerr, fmt = fmt, color=c, label = label, alpha = alpha)
        plt.ylabel('Transit depth')
        plt.xlabel('Wavelength (um)')
        aa = np.vstack((wav, half_bin_width, spec, av_error)).T
        
        dic={}
        dic['wav'] = wav
        dic['spec'] = spec
        dic['yerr'] = yerr
        dic['half_bin_width'] = half_bin_width
        dic['av_error'] = av_error
        dic['all'] = aa
        
        
      
        
        if 'u0' in params:
            plt.figure('empirical ldcs')
            ex  = np.load('%s_exp_sampler_results.npy'%(root))[:,:,1]
            u0spec = ex[:,1]
            u0err_plus =  ex[:,2]- ex[:,1]
            u0err_minus =  ex[:,1] - ex[:,0]
            u0spec = u0spec[idx]
            u0err_plus=u0err_plus[idx]
            u0err_minus=u0err_minus[idx]
            u0yerr = np.array(list(zip(u0err_minus, u0err_plus))).T
            plt.errorbar(wav, u0spec, u0yerr, fmt = 'ro-', label = 'u0')
            dic['u0'] = u0spec
            dic['u0err'] = u0yerr
            
        if 'u1' in params:
            plt.figure('empirical ldcs')
            ex  = np.load('%s_exp_sampler_results.npy'%(root))[:,:,2]
            u1spec = ex[:,1]
            u1err_plus =  ex[:,2]- ex[:,1]
            u1err_minus =  ex[:,1] - ex[:,0]
            u1spec = u1spec[idx]
            u1err_plus=u1err_plus[idx]
            u1err_minus=u1err_minus[idx]
            u1yerr = np.array(list(zip(u1err_minus, u1err_plus))).T
            plt.errorbar(wav, u1spec, u1yerr, fmt = 'bo-', label = 'u1')
            plt.legend()
            # ldc = np.vstack((wav, u0spec, u1spec))
            dic['u1'] = u1spec
            dic['u1err'] = u1yerr
        
        
    return dic
    

# =============================================================================
# PREVIOUSLY IN STAGE3_PIPELINE_LIB
# =============================================================================
global free_params_names 
global fixed_params_dic 
global c_sign, b_sign, a_sign       

def linear_func(t, a, b):
      syst0 = a * (1 + b* (t))
      return syst0 
def quadratic_func(t, a, b, c):
      syst0 = a * (1 + b* (t) + c*t**2)
      return syst0 
def lin_exp_func(t, a, b, c, d):
    syst = a*(1+ b*t + c*np.exp(-t/d))
    return syst

def fit_function(qq, q1, q2, q3, q4, q5, q6, data_dic0):

      data_dic = copy.deepcopy(data_dic0)

      idx_oot = data_dic['idx_oot']
      # shift_idx  = data_dic['shift_idx'] 
      # grating_correction = data_dic['grating_correction'] 
      t = data_dic['t']
      channel = data_dic['channel']
      if 'nrs' in data_dic['f']:
          nrs = data_dic['nrs']
      
      f = data_dic['f']
      ldc_type = data_dic['ldc_type']
      wav = data_dic['wav']
      slc = data_dic['slc']
      var = data_dic['var']
      wlc = data_dic['wlc'] 
      wlc_var = data_dic['wlc_var']  
      
      fit_wlc= data_dic['fit_wlc']
      ldc =  data_dic['ldc']
      mp = data_dic['mp']
      mcmc_option = data_dic['mcmc_option']
      kipping = data_dic['kipping']
      folder = data_dic['folder']
      t_start = data_dic['t_start']

      tag = data_dic['tag']
      
      lc_model_params = data_dic['lc_model_params'] 
      syst_model_params = data_dic['syst_model_params'] 
      
      syst_model_type = data_dic['syst_model_type'] 
      
      print ('syst model type', syst_model_type)
      
      free_params_names=[]
      for key in lc_model_params.keys():
           if lc_model_params[key]['fit'] == True:
               free_params_names.append(key)
      for key in syst_model_params.keys():
           if syst_model_params[key]['fit'] == True:
               free_params_names.append(key)
 
      if kipping ==1: # convert normal ldcs to kipping formulation
          print ('converting to kipping ldc formulation...')
          
          q1_kipping = (ldc[0]+ldc[1])**2
          q2_kipping = ldc[0]/(2*(ldc[0]+ldc[1]))
          ldc[0] = q1_kipping 
          ldc[1] = q2_kipping
       
          u0_normal = lc_model_params['u0']['init'] 
          u1_normal = lc_model_params['u1']['init'] 
          q1_kipping = (u0_normal+u1_normal)**2
          q2_kipping = u0_normal/(2*(u0_normal+u1_normal))   
           
          print (lc_model_params['u0']['init'], lc_model_params['u1']['init'])
           
          lc_model_params['u0']['init'] = q1_kipping
          lc_model_params['u1']['init'] = q2_kipping
          
      print (lc_model_params['u0']['init'], lc_model_params['u1']['init'])
   
      bjd = t   

      if fit_wlc==1:
          data_lc = wlc
          data_lc_var = wlc_var
          mp =0
          gamma0 = [lc_model_params['u0']['init'], lc_model_params['u1']['init']  ]
          
      else:
 
          data_lc = slc[:,qq]
          data_lc_var = var[:,qq]
          gamma0 = ldc[:,qq]
          print ('+++++++++  wav,qq, ldc', wav[qq], qq, gamma0)
 
      if mp == 0:    
          plt.figure('final data_lc')
          plt.plot(data_lc, 'o')

          plt.figure('final data_lc2')
          plt.errorbar(t, data_lc, data_lc_var**0.5, fmt='o')

      use_c_ratio = 1
      

      y= data_lc
      
      save_fig=1
      show_fig=1
      
      oot_section = data_lc[idx_oot]
     
      if syst_model_type == 'linear':
            
            a_est = np.mean(y[0:10])
            grad_est =  (y[idx_oot][-1] -  y[idx_oot][0]) / (t[idx_oot][-1] - t[idx_oot][0])
            b_est = grad_est/a_est
            d_est = 1
            c_est = 0            
            popt, pcov = optimize.curve_fit(linear_func, t[idx_oot], y[idx_oot], sigma=None,
                                          p0=(a_est, b_est), method="lm",  maxfev=50000)
          
            yy_oot = linear_func(t[idx_oot], popt[0], popt[1])
            a,b,c,d   = popt[0], popt[1], 0,1
            print (popt)
            
      if syst_model_type == 'quadratic' or syst_model_type == 'poly':
          # fit the oot points, to get initial values for c,b and a and yerr 
            a_est = np.mean(y[0:10])
            grad_est =  (y[idx_oot][-1] -  y[idx_oot][0]) / (t[idx_oot][-1] - t[idx_oot][0])
            b_est = grad_est/a_est
            c_est = 0
            popt, pcov = optimize.curve_fit(quadratic_func, t[idx_oot], y[idx_oot], sigma=None,
                                          p0=(a_est, b_est, c_est), method="lm",  maxfev=50000)
            
            yy_oot = quadratic_func(t[idx_oot], popt[0], popt[1], popt[2])
            a,b,c,d   = popt[0], popt[1], popt[2],1
            print (popt)
         
      if syst_model_type == 'lin_exp' or  syst_model_type == 'exp':
          # Initial guess for parameters (a, b, c, d)
          p0 = [np.mean( y[idx_oot]), 0.0, 0.01, 0.1]
          # Fit the curve
          popt, pcov = optimize.curve_fit(lin_exp_func, t[idx_oot], y[idx_oot], p0=p0, sigma=None,method="lm", maxfev=50000)
          # Extract parameters
          a_fit, b_fit, c_fit, d_fit = popt
          print(f"Fitted parameters:\n a={a_fit:.6g}\n b={b_fit:.6g}\n c={c_fit:.6g}\n d={d_fit:.6g}")
          yy_oot = lin_exp_func(t[idx_oot], *popt)
          # Plot the fit
          plt.figure(figsize=(10,6))
          plt.scatter(t, y, s=5, label='Data', color='blue')
          plt.plot(t[idx_oot], yy_oot , 'r-', label='Fitted model', linewidth=2)
          plt.xlabel("Time from 0 (days)")
          plt.ylabel("Flux (arbitrary units)")
          plt.title("Fit: a*(1 + b*t + c*exp(-t/d))")
          plt.legend()
          plt.grid(True)
          plt.show()
          a,b,c,d   = popt[0], 0,0,0
                   
      # if show_fig ==1:   
      plt.figure('oot 2'); plt.plot(t[idx_oot], oot_section, 'o'); plt.plot(t[idx_oot], yy_oot, '-')
      plt.figure('oot 3'); plt.plot(t[idx_oot], yy_oot - oot_section, 'o')  
        
      yerr = np.std(yy_oot - oot_section)
      
      print ('yerr estimate', yerr)
        
      error_bar = data_lc_var**0.5
      error_bar_orig = error_bar*1
      print (np.mean(error_bar))
      
      scale_err = yerr/(np.nansum(error_bar)/len(error_bar))
      
      print ('scale err', scale_err)
      
      error_bar *=scale_err
      print (np.mean(error_bar))
      
      if mp==0:
          plt.figure('final data_lc2 scaled err bars')
          plt.errorbar(bjd, data_lc, error_bar, fmt='o')

      yerr = error_bar
      
    
# =============================================================================
#    intial guess
# =============================================================================   
      model = pipeline_lib.TransitModel()
 
      parameter_names = []
      parameter_bounds = []
      parameter_vector =[]
      parameter_signs =[]
      prior_type =[]
      log_value =[]
      
      fixed_params_dic ={}
      fixed_params_dic['t']=t
      free_params_dic={}
      
      if syst_model_params['a']['fit']==True:
          syst_model_params['a']['init'] = a  = np.mean( y[idx_oot])
      else:
          a = syst_model_params['a']['init']
      
      a_range = syst_model_params['a']['range']
    
      syst_model_params['a']['range'] = [a - abs(a)*a_range, a + abs(a)*a_range]
                
        

      # now set signs for each parameter (matters if using log in fit)
        
      for key in lc_model_params.keys():
           if lc_model_params[key]['init'] < 0:
               lc_model_params[key]['sign'] = -1
           else:
               lc_model_params[key]['sign'] = 1
              
      for key in syst_model_params.keys():
           if syst_model_params[key]['init'] < 0:
                 syst_model_params[key]['sign'] = -1
           else:
                 syst_model_params[key]['sign'] = 1
                       
      # if not fitting ldcs, assign them the model values. 
      if lc_model_params['u0']['fit'] == False:
          lc_model_params['u0']['init'] = gamma0[0]
          
      if lc_model_params['u1']['fit'] == False:
          lc_model_params['u1']['init'] = gamma0[1]
          
           
      # set log values in the model param vector ; o/w set normal values in vector
      for key in lc_model_params.keys():
            if lc_model_params[key]['fit']== True:
                
                prior_type.append(lc_model_params[key]['prior_type'])
                log_value.append(lc_model_params[key]['log'])


                if lc_model_params[key]['log'] == True:
                    xx = 'log_%s'%(key)      
                    # print (xx)
                    # print (np.log(abs(lc_model_params[key]['init'])))
                    # model.__dict__[xx] = 1
                    parameter_vector.append(np.log(abs(lc_model_params[key]['init'])))
                    parameter_names.append(xx)
                    parameter_signs.append(lc_model_params[key]['sign'])
                    free_params_dic[xx] = np.log(abs(lc_model_params[key]['init']))
                    
                    if lc_model_params[key]['prior_type'] == 'uniform':
                        parameter_bounds.append(lc_model_params[key]['log_range'])
                    elif lc_model_params[key]['prior_type'] == 'gaussian':
                        parameter_bounds.append([lc_model_params[key]['mu'], lc_model_params[key]['sigma']])


                else:
                    # model.__dict__[key] = lc_model_params[key]['init']
                    parameter_names.append(key)
                    parameter_vector.append(lc_model_params[key]['init'])
                    free_params_dic[key] = lc_model_params[key]['init']
                    parameter_signs.append(lc_model_params[key]['sign'])
                    
                    if lc_model_params[key]['prior_type'] == 'uniform':
                        parameter_bounds.append(lc_model_params[key]['range'])
                    elif lc_model_params[key]['prior_type'] == 'gaussian':
                        parameter_bounds.append([lc_model_params[key]['mu'], lc_model_params[key]['sigma']])

            else:
                fixed_params_dic[key] = lc_model_params[key]['init']
     
         
      for key in syst_model_params.keys():
             if syst_model_params[key]['fit']== True:
                 
                 prior_type.append(syst_model_params[key]['prior_type'])
                 log_value.append(syst_model_params[key]['log'])


                 if syst_model_params[key]['log'] == True:
                     
                     xx = 'log_%s'%(key)      
                     print (xx)
                     print (np.log(abs(syst_model_params[key]['init'])))
                     # model.__dict__[xx] = 1
                     parameter_vector.append(np.log(abs(syst_model_params[key]['init'])))
                     parameter_names.append(xx)
                     free_params_dic[xx] = np.log(abs(syst_model_params[key]['init']))
                     parameter_signs.append(syst_model_params[key]['sign'])
                     
                     
                     if syst_model_params[key]['prior_type'] == 'uniform':
                         parameter_bounds.append(syst_model_params[key]['log_range'])
                     elif syst_model_params[key]['prior_type'] == 'gaussian':
                         parameter_bounds.append([syst_model_params[key]['mu'], syst_model_params[key]['sigma']])
                     


                 else:
                     # model.__dict__[key] = lc_model_params[key]['init']
                     parameter_names.append(key)
                     parameter_vector.append(syst_model_params[key]['init'])
                     free_params_dic[key] = syst_model_params[key]['init']
                     parameter_signs.append(syst_model_params[key]['sign'])
                     
                     if syst_model_params[key]['prior_type'] == 'uniform':
                         parameter_bounds.append(syst_model_params[key]['range'])
                     elif syst_model_params[key]['prior_type'] == 'gaussian':
                         parameter_bounds.append([syst_model_params[key]['mu'], syst_model_params[key]['sigma']])
                    

             else:
                 fixed_params_dic[key] = syst_model_params[key]['init']
                
                
 
      model.parameter_names = parameter_names
         
      for i in range(len(parameter_names)):
            model.__dict__[parameter_names[i]]= parameter_vector[i]

      model.set_parameter_vector(parameter_vector, include_frozen=True)
      
      model.parameter_signs = parameter_signs
      
      for key in fixed_params_dic.keys():
          model.__dict__[key] = fixed_params_dic[key]
          
          
      model.__dict__['syst_model_type'] = syst_model_type
      model.__dict__['kipping'] = kipping

      # print (model.__dict__)
      # print (model.model_type)

      model.__dict__['fit_spot'] = data_dic['fit_spot']  # now defunct but needed transit functions at the moment
  
# =============================================================================
#       intial guess
# =============================================================================
      super_dict ={}
      super_dict['lc_model_params'] =  lc_model_params
      super_dict['syst_model_params'] =  syst_model_params
      super_dict['syst_model_type'] =  syst_model_type

      super_dict['prior_ranges']  = parameter_bounds
      super_dict['prior_type']  = prior_type
      super_dict['log_value']  = log_value
      super_dict['init_values']  = parameter_vector
      super_dict['parameter_names']  = parameter_names

      super_dict['fixed_params_dic']  = fixed_params_dic

      super_dict['ldc_type'] = ldc_type 
      super_dict['kipping'] = kipping
      super_dict['use_c_ratio'] =  use_c_ratio
      
      super_dict['t'] = t
 
      lc_fit = model.get_value(t)
      
      plt.figure('initial guess')
      plt.plot(t, data_lc, 'o')
      plt.plot(t, lc_fit, 'r-', lw=10)
      
      print (syst_model_params['a'], syst_model_params['b'])
          
# =============================================================================
#             mcmc
# =============================================================================
      free_log_params_prior =  super_dict['prior_ranges'] 
      fit_log_init = super_dict['init_values'] 
      prior_type =  super_dict['prior_type'] 
      log_value =  super_dict['log_value'] 
      parameter_names =  super_dict['parameter_names'] 
      
      for ww in range(3):
     
        
          ndim = len(free_params_dic.keys())
          nwalkers = 16
          nwalkers = 32
        
          if mcmc_option == 1:
              nwalkers = 64
          
          pos=[]
          for key in free_params_dic.keys():
              pos.append(free_params_dic[key])
          pos = np.array(pos)
          
          pos = pos + 1e-4*np.random.randn(nwalkers, ndim)
  
          sampler = emcee.EnsembleSampler(nwalkers, ndim, pipeline_lib.simple_log_prob,
                                     args=(model,  t, y, yerr, super_dict))
       
          # burn in
          if mcmc_option ==3:
            burn_in = 500
          if mcmc_option==0:
            burn_in = 1000
          if mcmc_option ==2:
           burn_in = 100
          if mcmc_option == 1:
           burn_in = 4000
          if mcmc_option == 4:
            burn_in = data_dic['custom_burn_in']
          pos, _, _ = sampler.run_mcmc(pos, burn_in, progress=True, tune=True)
 
          if kipping:
              try:
                idx1 = parameter_names.index('u0')
               
                q1_kipping = sampler.flatchain[:,idx1]
                idx2 = parameter_names.index('u1')
             
                q2_kipping = sampler.flatchain[:,idx2]
                u0_normal = 2*np.sqrt(q1_kipping)*q2_kipping
                u1_normal = np.sqrt(q1_kipping)*(1-2*q2_kipping)
                sampler.flatchain[:,idx1] = u0_normal
                sampler.flatchain[:,idx2] = u1_normal
              except:
                pass

          gp.corner(sampler.flatchain, 
                  labels= parameter_names)
 
          sampler.reset()

          # production run  
          if mcmc_option ==3:
            production = 1000
          if mcmc_option ==2:
            production = 100
          if mcmc_option == 0:
            production = 2000
          if mcmc_option == 1:
            production = 4000 
          if mcmc_option == 4:
            production = data_dic['custom_production']
          sampler.run_mcmc(pos, production,progress=True, tune=True);

          if kipping:
             try:
 
                idx1 = parameter_names.index('u0')
            
                q1_kipping = sampler.flatchain[:,idx1]
                idx2 = parameter_names.index('u1')
            
                q2_kipping = sampler.flatchain[:,idx2]
                u0_normal = 2*np.sqrt(q1_kipping)*q2_kipping
                u1_normal = np.sqrt(q1_kipping)*(1-2*q2_kipping)
                sampler.flatchain[:,idx1] = u0_normal
                sampler.flatchain[:,idx2] = u1_normal
             except:
                pass

          gp.corner(sampler.flatchain, 
                  labels= parameter_names)
 
          percentiles = [16,50,84]
          
          sampler_results_w_logs = np.zeros((len(percentiles),len(parameter_names)))
          
          sampler_results_no_logs = np.zeros((len(percentiles),len(parameter_names)))
          
          for i in range(len(percentiles)):
              sampler_results_w_logs[i] = np.percentile(sampler.flatchain, percentiles[i], axis=0)
    
          for i in range(len(parameter_names)):
               if 'log' in  parameter_names[i]:
                   sampler_results_no_logs[:,i] = np.exp(sampler_results_w_logs[:,i])
               else:
                   sampler_results_no_logs[:,i] = sampler_results_w_logs[:,i]              
           
          print ("================================")
          for i in range(len(parameter_names)):
                  
              print ( parameter_names[i],  np.round(sampler_results_w_logs[:,i][1], 6), '-', 
                      np.round(sampler_results_w_logs[:,i][1]- sampler_results_w_logs[:,i][0],6), '+', 
                      np.round(sampler_results_w_logs[:,i][2]- sampler_results_w_logs[:,i][1], 6))
          
          print ("================================")
          for i in range(len(parameter_names)):
             if 'log' in  parameter_names[i]:
                   xx = parameter_names[i][4:]
             else:
                 xx = parameter_names[i]
    
             print ( xx,  np.round(sampler_results_no_logs[:,i][1], 6), '-', 
                     np.round(sampler_results_no_logs[:,i][1]- sampler_results_no_logs[:,i][0],6), '+', 
                     np.round(sampler_results_no_logs[:,i][2]- sampler_results_no_logs[:,i][1], 6)) 
        
          print ("================================")
              
          print ('Rp/Rs',  np.round((1e-6*sampler_results_no_logs[:,0][1])**0.5, 6), '-', 
                     np.round((1e-6*sampler_results_no_logs[:,0][1])**0.5- 
                              (1e-6*sampler_results_no_logs[:,0][0])**0.5,   6), '+', 
                     np.round((1e-6*sampler_results_no_logs[:,0][2])**0.5- 
                              (1e-6*sampler_results_no_logs[:,0][1])**0.5,  6)) 
          
          try:
              print ('t0',  np.round(sampler_results_no_logs[:,1][1] +t_start, 6), '-', 
                         np.round((sampler_results_no_logs[:,1][1] +t_start)- (sampler_results_no_logs[:,1][0] +t_start),6), '+', 
                         np.round((sampler_results_no_logs[:,1][2] +t_start)- (sampler_results_no_logs[:,1][1] +t_start), 6))
          except:
              pass
          print ('sampler acceptance fraction', sampler.acceptance_fraction) 
          # ideally close to 0.4-0.5
          
          # print ('sampler autocorrelation times', sampler.get_autocorr_time())

    # =============================================================================
    #    final fit      
    # =============================================================================
          parameter_vector =[]
          for i in range(len(parameter_names)):
              parameter_vector.append(sampler_results_w_logs[:, i][1])
    
     
          for i in range(len(parameter_names)):
                model.__dict__[parameter_names[i]]= parameter_vector[i]
     
          model.set_parameter_vector(parameter_vector, include_frozen=True)
          
 
          model.__dict__['kipping'] = False
          
          fitted_model = model.get_value(t)
    
          plt.figure('fitted model final')
          plt.plot(t, data_lc, 'o')
          plt.plot(t, fitted_model, 'r-', lw=2)
          plt.ylabel('Flux (MJy)')
          plt.ylabel('Time (BJD UTC)')
          
          plt.figure('residuals')
          plt.plot(t, data_lc-fitted_model, 'ko', lw=2)
          plt.ylabel('Flux (MJy)')
          plt.ylabel('Time (BJD UTC)')
   
      
          for i in range (len(model.parameter_names)):
            if 'log' in model.parameter_names[i]:
                xx_val = np.exp(model.parameter_vector[i])*model.parameter_signs[i]
                xx = model.parameter_names[i][4:]
                model.__dict__[xx] = xx_val 
          
          lc = pipeline_lib.transit_model(np.sqrt(model.depth*1e-6), 
                             model.t0, [model.u0, model.u1], model.per, 
                             model.ars, model.inc,
                             model.w, model.ecc, t, ldc_type = ldc_type)
  
          parameter_names_no_logs = []
          for i in range(len(parameter_names)):
               if 'log' in parameter_names[i]:
                   xx = parameter_names[i][4:]
               else:
                   xx = parameter_names[i]
               parameter_names_no_logs.append(xx)
            
          return_lc_model_params = lc_model_params
          for key in return_lc_model_params:
               for i in range(len(parameter_names_no_logs)):
                   if parameter_names_no_logs[i] == key:
                       xx = sampler_results_no_logs[:,i][1]
                       return_lc_model_params[key]['init'] = xx
                               
          res = y - fitted_model
   
          red_chi_sq = np.nansum((fitted_model-y)**2 / (res.std())**2)/(len(y)-len(free_params_names))
      
          print ('reduced chi squared:>>>>>> ', red_chi_sq) 
          
          if mp ==1:
              plt.close()
          
          if  red_chi_sq < 1.05:  #safety in case of a really bad fit
              break
       
 
       
      if fit_wlc ==1:
          
          folder_wlc = './wlc_data/%s'%(tag)
          if 'nrs' in data_dic['f']:
              folder_wlc = '%s/%s/nrs%s'%(folder_wlc, channel, nrs)
          elif 'miri' in data_dic['f']:
              folder_wlc = '%s/%s/mirimage'%(folder_wlc, channel)
          elif 'nis' in data_dic['f']:
              folder_wlc = '%s/%s/nis_o%s'%(folder_wlc, channel, data_dic['order'])

          folder_mcmc = '%s/mcmc'%(folder_wlc)
          folder_lmfit = '%s/lmfit'%(folder_wlc)
          
          if os.path.exists('./%s'%folder_mcmc) == False:
               os.makedirs('./%s'%folder_mcmc)
               
          if os.path.exists('./%s'%folder_lmfit) == False:
               os.makedirs('./%s'%folder_lmfit)
               
          np.save('./%s/t_start.npy'%(folder_mcmc), t_start)
          np.save('./%s/t.npy'%(folder_mcmc), t+t_start)
          np.save('./%s/wlc_transit_fit.npy'%(folder_mcmc), lc)
          np.save('./%s/wlc_full_fit.npy'%(folder_mcmc), fitted_model)
          np.save('./%s/wlc.npy'%(folder_mcmc), wlc)
          np.save('./%s/wlc_err.npy'%(folder_mcmc), yerr)
          np.save('./%s/wlc_sampler_flatchain.npy'%(folder_mcmc), sampler.flatchain)
          np.save('./%s/wlc_sampler_acceptance_fraction.npy'%(folder_mcmc), sampler.acceptance_fraction)
          np.save('./%s/signs.npy'%(folder_mcmc), np.array(parameter_signs)) 
          
          res = y - fitted_model
    
          print ('yerr final', res.std())
           
          red_chi_sq = np.nansum((fitted_model-y)**2 / (res.std())**2)/(len(y)-len(free_params_names))
        
          print ('reduced chi squared:>>>>>> ', red_chi_sq) 
           
          gp.corner(sampler.flatchain, 
                   labels= parameter_names)
          plt.savefig('%s/corner_%s.png'%(folder_mcmc, np.round(wav[qq],5)))
          plt.close()
          
          plt.figure('fit %s chsq:%s'%(wav[qq], np.round(red_chi_sq,2)))
          plt.plot(t, data_lc, 'ko', alpha = 0.5)
          plt.plot(t, fitted_model, 'r-')
          plt.ylabel('Flux (DN/s)')
          plt.xlabel('Time (BJD UTC)')
          plt.savefig('%s/fit_%s_%s.png'%(folder_mcmc, np.round(wav[qq],5), np.round(red_chi_sq,2)))
 
          plt.figure('fit wlc residuals')
          plt.plot(t, res, 'ko', alpha = 0.5)
          plt.ylabel('Flux (DN/s)')
          plt.xlabel('Time (BJD UTC)') 
          plt.savefig('%s/res_%s_%s.png'%(folder_mcmc, np.round(wav[qq],5), np.round(red_chi_sq,2)))          



      if fit_wlc ==0 and mp ==0: # single spectral light curve
        
          folder_slc = 'slc_data/'
          
          if 'nrs' in data_dic['f']:
   
              folder_slc = '%s%s_nrs%s_%s'%(folder_slc, channel, nrs, wav[qq])
              
          elif 'miri' in data_dic['f']:
              folder_slc = '%s%s_mirimage_%s'%(folder_slc, channel, wav[qq])
               
          folder_mcmc = '%s/mcmc'%(folder_slc)
          folder_lmfit = '%s/lmfit'%(folder_slc)
      
          if os.path.exists('./%s'%folder_mcmc) == False:
                 os.makedirs('./%s'%folder_mcmc)
                 
          if os.path.exists('./%s'%folder_lmfit) == False:
                 os.makedirs('./%s'%folder_lmfit)        

          np.save('./%s/t_start.npy'%(folder_mcmc), t_start)
          np.save('./%s/t.npy'%(folder_mcmc), t+t_start)
          np.save('./%s/wlc_transit_fit.npy'%(folder_mcmc), lc)
          np.save('./%s/wlc_full_fit.npy'%(folder_mcmc), fitted_model)
          np.save('./%s/wlc.npy'%(folder_mcmc), wlc)
          np.save('./%s/wlc_err.npy'%(folder_mcmc), yerr)
          np.save('./%s/wlc_sampler_flatchain.npy'%(folder_mcmc), sampler.flatchain)
          np.save('./%s/wlc_sampler_acceptance_fraction.npy'%(folder_mcmc), sampler.acceptance_fraction)
        
          res = y - fitted_model
  
          print ('yerr final', res.std())
         
          red_chi_sq = np.nansum((fitted_model-y)**2 / (res.std())**2)/(len(y)-len(free_params_names))
      
          print ('reduced chi squared:>>>>>> ', red_chi_sq) 
         
          gp.corner(sampler.flatchain, 
                 labels= parameter_names)
          plt.savefig('%s/corner_%s.png'%(folder_mcmc, np.round(wav[qq],5)))
          plt.close()
        
          plt.figure('fit %s chsq:%s'%(wav[qq], np.round(red_chi_sq,2)))
          plt.plot(t, data_lc, 'ko', alpha = 0.5)
          plt.plot(t, fitted_model, 'r-')
          plt.ylabel('Flux (DN/s)')
          plt.xlabel('Time (BJD UTC)')
          plt.savefig('%s/fit_%s_%s.png'%(folder_mcmc, np.round(wav[qq],5), np.round(red_chi_sq,2)))
  
          plt.figure('fit wlc residuals')
          plt.plot(t, res, 'ko', alpha = 0.5)
          plt.ylabel('Flux (DN/s)')
          plt.xlabel('Time (BJD UTC)') 
          plt.savefig('%s/res_%s_%s.png'%(folder_mcmc, np.round(wav[qq],5), np.round(red_chi_sq,2)))          
    
 
        
      if fit_wlc ==0 and mp ==1: # multiple spectral light curve
      
           folder_mcmc = '%s/mcmc'%(folder)
           folder_lmfit = '%s/lmfit'%(folder)
              
           if os.path.exists('./%s'%folder_mcmc) == False:
               os.makedirs('./%s'%folder_mcmc)
           if os.path.exists('./%s/data'%folder_mcmc) == False:
               os.makedirs('./%s/data'%folder_mcmc)      
           if os.path.exists('./%s/corners'%folder_mcmc) == False:
                os.makedirs('./%s/corners'%folder_mcmc)   
           if os.path.exists('./%s/res'%folder_mcmc) == False:
                os.makedirs('./%s/res'%folder_mcmc)   
           if os.path.exists('./%s/fits'%folder_mcmc) == False:
                os.makedirs('./%s/fits'%folder_mcmc)              
                       
           if os.path.exists('./%s'%folder_lmfit) == False:
               os.makedirs('./%s'%folder_lmfit)      
           if os.path.exists('./%s/data'%folder_lmfit) == False:
                os.makedirs('./%s/data'%folder_lmfit)   
           if os.path.exists('./%s/res'%folder_lmfit) == False:
                os.makedirs('./%s/res'%folder_lmfit)   
           if os.path.exists('./%s/fits'%folder_lmfit) == False:
                os.makedirs('./%s/fits'%folder_lmfit)   


           res = y - fitted_model
           print ('yerr final', res.std())
            
           red_chi_sq = np.nansum((fitted_model-y)**2 / (res.std())**2)/(len(y)-len(free_params_names))     
           print ('reduced chi squared:>>>>>> ', red_chi_sq) 
            
           gp.corner(sampler.flatchain, 
                    labels= parameter_names)
           plt.savefig('%s/corners/corner_%s.png'%(folder_mcmc, np.round(wav[qq],5)))
           plt.close()
           
           plt.figure('fit %s chsq:%s'%(wav[qq], np.round(red_chi_sq,2)))
           plt.plot(t, data_lc, 'ko', alpha = 0.5)
           plt.plot(t, fitted_model, 'r-')
           plt.ylabel('Flux (MJy)')
           plt.xlabel('Time (BJD UTC)')
           plt.savefig('%s/fits/fit_%s_%s.png'%(folder_mcmc, np.round(wav[qq],5), np.round(red_chi_sq,2)))
           plt.close()
         
           plt.figure('fit wlc residuals')
           plt.plot(t, res, 'ko', alpha = 0.5)
           plt.ylabel('Flux (MJy)')
           plt.xlabel('Time (BJD UTC)') 
           plt.savefig('%s/res/res_%s_%s.png'%(folder_mcmc, np.round(wav[qq],5), np.round(red_chi_sq,2)))          
           plt.close()
 
      if mp==True:
          q1.put(wav[qq])
          q2.put(np.vstack(( parameter_names,free_params_names)))
          q3.put(sampler_results_w_logs)
          q4.put(sampler_results_no_logs)
          q5.put(parameter_signs)
          # q6.put([(result.params['depth'].value*1e-6)**0.5, plus_err])
          q6.put(1)

      else:
          return return_lc_model_params
 

