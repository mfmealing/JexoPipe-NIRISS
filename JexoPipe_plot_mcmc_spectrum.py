#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 10 18:26:34 2024

@author: c1341133
"""

# import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.style
import matplotlib as mpl
from uncertainties import ufloat
# mpl.style.use('classic')

 
def extract(root, root2, label, c='k'):
    wav =  np.load('%s_wav.npy'%(root))
  
    idx = np.argsort(wav)
    wav = wav[idx]
    
    # print (wav/np.gradient(wav))
 
    params = np.load('%s_params.npy'%(root))
    ex  = np.load('%s_exp_sampler_results.npy'%(root))[:,:,0]
    print(ex.shape)
    
    
    
    spec = ex[:,1]
    err_plus =  np.abs(ex[:,2]- ex[:,1])
    err_minus =  np.abs(ex[:,1] - ex[:,0])
 
    spec = spec[idx]
    err_plus=err_plus[idx]
    err_minus=err_minus[idx]
 
    spec = ex[:,1]**2
    err_plus =  np.abs(ex[:,2]**2- ex[:,1]**2)
    err_minus =  np.abs(ex[:,1]**2 - ex[:,0]**2)
    
    spec = spec[idx]
    err_plus=err_plus[idx]
    err_minus=err_minus[idx]
   
    yerr = np.array(list(zip(err_minus, err_plus))).T
    print(yerr)
    
    half_bin_width  = np.gradient(wav)/2
    # av_error = (err_plus +err_minus)/2
    
    xerr = half_bin_width
    
    # data = np.vstack((wav,xerr,spec,yerr)).T
    # np.save('%sfinal_data'%(root2), data)
    
    
    
    # if label == 'O1: LHS 1140 b':
    #     wav = wav[:-3]
    #     spec = spec[:-3]
    #     yerr = yerr[:, :-3]
    #     xerr = xerr[:-3]
    

    

    plt.figure('spectrum', figsize=(19,8))
    
    plt.errorbar(wav, spec, yerr=yerr, xerr=xerr, fmt = 'o', color=c, label = label, capsize=4)
 
    plt.legend()
 
    if 'u0' in params:
        plt.figure('empirical ldcs')
        ex  = np.load('%s_exp_sampler_results.npy'%(root))[:,:,1]
     
        spec = ex[:,1]
        err_plus =  ex[:,2]- ex[:,1]
        err_minus =  ex[:,1] - ex[:,0]
        spec = spec[idx]
        spec1= spec*1
        err_plus=err_plus[idx]
        err_minus=err_minus[idx]
        yerr = np.array(list(zip(err_minus, err_plus))).T
        plt.errorbar(wav, spec, yerr, fmt = 'ro-', label = label)
        
    if 'u1' in params:
        plt.figure('empirical ldcs')
        ex  = np.load('%s_exp_sampler_results.npy'%(root))[:,:,2]
        spec = ex[:,1]
        err_plus =  ex[:,2]- ex[:,1]
        err_minus =  ex[:,1] - ex[:,0]
        spec = spec[idx]
        spec2= spec*1
        err_plus=err_plus[idx]
        err_minus=err_minus[idx]
        yerr = np.array(list(zip(err_minus, err_plus))).T
        plt.errorbar(wav, spec, yerr, fmt = 'bo-', label = label)
        
        aa = np.vstack((wav, spec1, spec2))
         
        np.save('./%s_ldc'%(label), aa)
# =============================================================================
#         
# =============================================================================
 
 

# label = 'O1: WASP-17 b'
# root = '/Volumes/Crucial X9/JexoPipe-NIRISS/spectrum/niriss_jw01353101001_04101_00001_box_extract_o1_WASP_17_linear/order1/spectrum'
# extract (root, label, c='g')

# label = 'O2: WASP-17 b'
# root = '/Volumes/Crucial X9/JexoPipe-NIRISS/spectrum/niriss_jw01353101001_04101_00001_box_extract_o2_WASP_17_linear/order2/spectrum'
# extract (root, label, c='b')


# ----- K2-18b -----

# label = 'O1: Main'
# root = '/Volumes/Crucial X9/JexoPipe-NIRISS/spectrum/niriss_jw02722003001_04101_00001_box_extract_o1_K2_18_linear/order1/spectrum'
# extract (root,label, c='black')

# label = 'O2: Main'
# root = '/Volumes/Crucial X9/JexoPipe-NIRISS/spectrum/niriss_jw02722003001_04101_00001_box_extract_o2_K2_18_linear/order2/spectrum'
# extract (root, label, c='dimgrey')

# label = 'O1: Unseparated'
# root = '/Volumes/Crucial X9/JexoPipe-NIRISS/spectrum/niriss_jw02722003001_04101_00001_box_extract_o1_K2_18_no_sep_spec_linear/order1/spectrum'
# extract (root, label, c='forestgreen')

# label = 'O2: Unseparated'
# root = '/Volumes/Crucial X9/JexoPipe-NIRISS/spectrum/niriss_jw02722003001_04101_00001_box_extract_o2_K2_18_no_sep_spec_linear/order2/spectrum'
# extract (root, label, c='limegreen')

# label = 'O1: ATOCA'
# root = '/Volumes/Crucial X9/JexoPipe-NIRISS/spectrum/niriss_jw02722003001_04101_00001_atoca_extract_o1_K2_18_atoca_linear/order1/spectrum'
# extract (root, label, c='mediumblue')

# label = 'O2: ATOCA'
# root = '/Volumes/Crucial X9/JexoPipe-NIRISS/spectrum/niriss_jw02722003001_04101_00001_atoca_extract_o2_K2_18_atoca_linear/order2/spectrum'
# extract (root, label, c='dodgerblue')

# ----------

# label = 'O1: WASP-17 b'
# root = '/Volumes/Crucial X9/JexoPipe-NIRISS/spectrum/niriss_jw01353101001_04101_00001_box_extract_o1_WASP_17_col_2_bin_linear/order1/spectrum'
# root2 = '/Volumes/Crucial X9/JexoPipe-NIRISS/spectrum/niriss_jw01353101001_04101_00001_box_extract_o1_WASP_17_col_2_bin_linear/order1/'
# extract (root, root2, label, c='k')

# label = 'O2: WASP-17 b'
# root = '/Volumes/Crucial X9/JexoPipe-NIRISS/spectrum/niriss_jw01353101001_04101_00001_box_extract_o2_WASP_17_linear/order2/spectrum'
# root2 = '/Volumes/Crucial X9/JexoPipe-NIRISS/spectrum/niriss_jw01353101001_04101_00001_box_extract_o2_WASP_17_linear/order2/'
# extract (root, root2, label, c='b')

# df = pd.read_csv('/Volumes/Crucial X9/JexoPipe_NIRISS_raw_files/WASP-17b_NIRISS_SOSS_transmission_spectrum_pixel_o1.csv')
# lit_wav, lit_xerr, lit_spec, lit_yerr = df['wave'], df['wave_err'], df['dppm'], df['dppm_err']
# plt.errorbar(lit_wav, lit_spec/1e6, lit_yerr/1e6, fmt='o')



# ----------

label = 'O1: GJ 9827 d (visit 1)'
root = '/Volumes/Crucial X9/JexoPipe-NIRISS/spectrum/niriss_jw04098007001_04101_00001_box_extract_o1_GJ_9827_lit_comp_linear/order1/spectrum'
root2 = '/Volumes/Crucial X9/JexoPipe-NIRISS/spectrum/niriss_jw04098007001_04101_00001_box_extract_o1_GJ_9827_lit_comp_linear/order1/'
extract (root, root2, label, c='k')

# label = 'O1: GJ 9827 d test'
# root = '/Volumes/Crucial X9/JexoPipe-NIRISS/spectrum/niriss_jw04098007001_04101_00001_box_extract_o1_GJ_9827_col_15_bin_linear/order1/spectrum'
# root2 = '/Volumes/Crucial X9/JexoPipe-NIRISS/spectrum/niriss_jw04098007001_04101_00001_box_extract_o1_GJ_9827_col_15_bin_linear/order1/'
# extract (root, root2, label, c='b')

# label = 'O2: GJ 9827 d (visit 1)'
# root = '/Volumes/Crucial X9/JexoPipe-NIRISS/spectrum/niriss_jw04098007001_04101_00001_box_extract_o2_GJ_9827_linear/order2/spectrum'
# root2 = '/Volumes/Crucial X9/JexoPipe-NIRISS/spectrum/niriss_jw04098007001_04101_00001_box_extract_o2_GJ_9827_linear/order2/'
# extract (root, root2, label, c='dodgerblue')

label = 'O1: GJ 9827 d (visit 2)'
root = '/Volumes/Crucial X9/JexoPipe-NIRISS/spectrum/niriss_jw04098008001_04101_00001_box_extract_o1_GJ_9827_V2_lit_comp_linear/order1/spectrum'
root2 = '/Volumes/Crucial X9/JexoPipe-NIRISS/spectrum/niriss_jw04098008001_04101_00001_box_extract_o1_GJ_9827_V2_lit_comp_linear/order1/'
extract (root, root2, label, c='r')

# label = 'O2: GJ 9827 d (visit 2)'
# root = '/Volumes/Crucial X9/JexoPipe-NIRISS/spectrum/niriss_jw04098008001_04101_00001_box_extract_o2_GJ_9827_V2_linear/order2/spectrum'
# root2 = '/Volumes/Crucial X9/JexoPipe-NIRISS/spectrum/niriss_jw04098008001_04101_00001_box_extract_o2_GJ_9827_V2_linear/order2/'
# extract (root, root2, label, c='pink')

# label = 'O1: GJ 9827 d test V2'
# root = '/Volumes/Crucial X9/JexoPipe-NIRISS/spectrum/niriss_jw04098008001_04101_00001_box_extract_o1_GJ_9827_V2_col_15_bin_linear/order1/spectrum'
# root2 = '/Volumes/Crucial X9/JexoPipe-NIRISS/spectrum/niriss_jw04098008001_04101_00001_box_extract_o1_GJ_9827_V2_col_15_bin_linear/order1/'
# extract (root, root2, label, c='r')





fontsize =20
fontsize2 =20

plt.figure('spectrum', figsize=(19,8))
ax = plt.gca()
plt.xlabel('Wavelength (μm)')
plt.ylabel('Transit depth')
# plt.ylim(0.0052, 0.0059)

#set off Scientific notation from Y-axis
ax.get_yaxis().get_major_formatter().set_useOffset(False)
# ax.get_yaxis().get_major_formatter().set_scientific(False)
ax.get_xaxis().get_major_formatter().set_useOffset(False)

 
ax = plt.gca()
legend = ax.legend(loc='upper center', ncol=2, shadow=False, fontsize=fontsize, frameon=False)
ax.get_legend().get_title().set_fontsize('18')
ax.get_legend().get_title().set_fontweight('bold')

frame = legend.get_frame()
frame.set_facecolor('0.90')
for label in legend.get_texts():
    label.set_fontsize('medium')
    label.set_fontsize(fontsize)
    label.set_fontweight('bold')
for label in legend.get_lines():
    label.set_linewidth(1.5)  # the legend line width
for item in (ax.get_xticklabels() + ax.get_yticklabels()):
                  item.set_fontsize(fontsize)      
for item in (ax.get_xticklabels() + ax.get_yticklabels()):
                  item.set_fontweight('bold')       
for item in ([ax.title, ax.xaxis.label, ax.yaxis.label]):item.set_fontsize(fontsize2)     
for item in ([ax.title, ax.xaxis.label, ax.yaxis.label]):item.set_fontweight('bold')         
    
                           
[x.set_linewidth(5) for x in ax.spines.values()]
ax.xaxis.set_tick_params(width=3, length=10)
ax.yaxis.set_tick_params(width=3, length=10)

# plt.subplots_adjust(left=0.11, right=0.95, top=0.95, bottom=0.2)
# plt.savefig('/Users/c24050258/Desktop/Plots/Paper_graphs/spectrum_comp_no_sep_spec.png', dpi=150)
