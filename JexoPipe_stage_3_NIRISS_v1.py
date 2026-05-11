
from multiprocessing import Process, Queue
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os
from scipy import optimize 
matplotlib.style.use('classic')
import matplotlib.pyplot as plt
from astropy.io import fits
import pandas as pd
import pipeline_lib as pipeline_lib
global free_params_names 
global fixed_params_dic 
global c_sign, b_sign, a_sign   
import pipeline_stage_3_lib as pipeline_stage_3_lib
# from stage_3_pipeline_lib_6 import fit_function
from pipeline_stage_3_lib import fit_function

from lmfit.model import load_modelresult
import sys

# import lc_fit
import copy


from uncertainties import ufloat


def func0(t, a, b, c, d):
      syst0 = a * (1 + b * np.exp(-(t) / c) + d* (t))
      return syst0
  
    
 
  

# =============================================================================
# 1. load file(s)
# =============================================================================

#wasp 17b double pass
# f='/Volumes/Crucial X9/JexoPipe-NIRISS/fits_files/niriss/jw01353101001_04101_00001-COMBINED_order4_nis_1Dspec_box_extract_o1_WASP_17_col_2_bin.fits'
# f='/Volumes/Crucial X9/JexoPipe-NIRISS/fits_files/niriss/jw01353101001_04101_00001-COMBINED_order4_nis_1Dspec_box_extract_o2_WASP_17.fits'

# k2 18b
# f = '/Volumes/Crucial X9/JexoPipe-NIRISS/fits_files/niriss/jw02722003001_04101_00001-COMBINED_order3_nis_1Dspec_box_extract_o1_K2_18_test.fits'
# f = '/Volumes/Crucial X9/JexoPipe-NIRISS/fits_files/niriss/jw02722003001_04101_00001-COMBINED_order3_nis_1Dspec_box_extract_o2_K2_18.fits'

# k2 18b no order separation
# f = '/Volumes/Crucial X9/JexoPipe-NIRISS/fits_files/niriss/jw02722003001_04101_00001-COMBINED_order3_nis_1Dspec_box_extract_o1_K2_18_no_sep_spec.fits'
# f = '/Volumes/Crucial X9/JexoPipe-NIRISS/fits_files/niriss/jw02722003001_04101_00001-COMBINED_order3_nis_1Dspec_box_extract_o2_K2_18_no_sep_spec.fits'

# k2 18b atoca
# f = '/Volumes/Crucial X9/JexoPipe-NIRISS/fits_files/niriss/jw02722003001_04101_00001-COMBINED_order3_nis_1Dspec_atoca_extract_o1_K2_18_atoca.fits'
# f = '/Volumes/Crucial X9/JexoPipe-NIRISS/fits_files/niriss/jw02722003001_04101_00001-COMBINED_order3_nis_1Dspec_atoca_extract_o2_K2_18_atoca.fits'

# lhs 1140b
# f = '/Volumes/Crucial X9/JexoPipe-NIRISS/fits_files/niriss/jw06543001001_04101_00001-COMBINED_order4_nis_1Dspec_box_extract_o1_LHS_1140.fits'
# f = '/Volumes/Crucial X9/JexoPipe-NIRISS/fits_files/niriss/jw06543001001_04101_00001-COMBINED_order4_nis_1Dspec_box_extract_o2_LHS_1140.fits'

# lhs 1140b no order separation
# f = '/Volumes/Crucial X9/JexoPipe-NIRISS/fits_files/niriss/jw06543001001_04101_00001-COMBINED_order4_nis_1Dspec_box_extract_o1_LHS_1140_no_sep_spec.fits'
# f = '/Volumes/Crucial X9/JexoPipe-NIRISS/fits_files/niriss/jw06543001001_04101_00001-COMBINED_order4_nis_1Dspec_box_extract_o2_LHS_1140_no_sep_spec.fits'

# lhs 1140b manually changed wavelength solution
# f = '/Volumes/Crucial X9/JexoPipe-NIRISS/fits_files/niriss/jw06543001001_04101_00001-COMBINED_order4_nis_1Dspec_opt_extract_o1_LHS_1140_manual.fits'
# f = '/Volumes/Crucial X9/JexoPipe-NIRISS/fits_files/niriss/jw06543001001_04101_00001-COMBINED_order4_nis_1Dspec_opt_extract_o2_LHS_1140_manual.fits'

# gj 9827d visit 1
# f = '/Volumes/Crucial X9/JexoPipe-NIRISS/fits_files/niriss/jw04098007001_04101_00001-COMBINED_order4_nis_1Dspec_box_extract_o1_GJ_9827_lit_comp.fits'
f = '/Volumes/Crucial X9/JexoPipe-NIRISS/fits_files/niriss/jw04098007001_04101_00001-COMBINED_order4_nis_1Dspec_box_extract_o2_GJ_9827_lit_comp.fits'

# gj 9827d visit 2
# f = '/Volumes/Crucial X9/JexoPipe-NIRISS/fits_files/niriss/jw04098008001_04101_00001-COMBINED_order4_nis_1Dspec_box_extract_o1_GJ_9827_V2_lit_comp.fits'
# f = '/Volumes/Crucial X9/JexoPipe-NIRISS/fits_files/niriss/jw04098008001_04101_00001-COMBINED_order4_nis_1Dspec_box_extract_o2_GJ_9827_V2.fits'




data_dic={}

# data_dic['first_run'] = 1

if 'o2' in f:
    data_dic['order'] = 2
    # data_dic['first_run'] = 0 # can't have first run in data set be for 2nd order, do 1st order runthrough first to get wlc data
else:
    data_dic['order'] = 1

 
data_dic = pipeline_stage_3_lib.load_file(data_dic, f)
print ('\n loading file...')
print (data_dic.keys())
 
# =============================================================================
# 2. deal with residual outliers
# =============================================================================
alpha = 2.5
data_dic = pipeline_stage_3_lib.outliers(data_dic, alpha)
print ('\n residual outliers...')
print (data_dic.keys())

# =============================================================================
# 3 . cut out some data points
# =============================================================================
# example
idx  = np.arange(1500,2000)
idx2  = np.arange(2150,2300)
idx_cut  = np.hstack((idx,idx2))

#default
idx_cut  = None
 
data_dic = pipeline_stage_3_lib.cutoff2(data_dic, idx_cut)
print ('\n cutoff...')
print (data_dic.keys())


wlc = np.nansum(data_dic['slc'], axis=1)

plt.figure('raw wlc')
plt.plot(wlc, '-')

# =============================================================================
# 4. time-bin
# ============================================================================= 
time_bin = 10 #cob-k
time_bin = 1#cob-k
# data_dic = pipeline_stage_3_lib.time_bin(data_dic, time_bin)
# =============================================================================
# 5. i.d. oot section
# ============================================================================= 
plt.figure('oot section')
plt.plot(data_dic['bjd'], np.nansum(data_dic['slc'],axis=1), 'bo')

flux = np.nansum(data_dic['slc'],axis=1)
time = data_dic['bjd']
plt.figure('flux')
plt.plot(flux)

# box = 100
box = int(len(flux)/100)
print ('box size', box)
 
bbox =np.ones(box)/box
flux = np.convolve(flux, bbox, 'same')
flux = np.convolve(flux, bbox, 'same')

derivative = np.gradient(flux, time)
derivative[:box+1] =0; derivative[-box-1:]=0

plt.figure('derivative')
plt.plot(derivative)

idx1 = np.argmin(derivative)
idx2 = np.argmax(derivative)
diff = int(0.3*(idx2-idx1) / 2)
 
oot1 = data_dic['bjd'][idx1-diff]
oot2 = data_dic['bjd'][idx2+diff]

oot_times = [oot1, oot2]
oot_times = [60258.3732928989, 60258.431889700936]
     
t0_est = np.mean((oot_times[0],oot_times[1]))
data_dic = pipeline_stage_3_lib.oot_section(data_dic, oot_times)
print ('\n oot section...')
print (data_dic.keys())

data_dic['oot_times']=oot_times 

plt.figure('oot section')
plt.plot(data_dic['bjd'][data_dic['idx_oot']], np.nansum(data_dic['slc'],axis=1)[data_dic['idx_oot']], 'ro')


# =============================================================================
# 6. get wlc and wavelength selection for wlc (optional) if we want the wlc to be within a given range different from the full slc
# wav_targ_wlc = None
wav_targ_wlc =[0.8, 2.0]
data_dic = pipeline_stage_3_lib.get_wlc(data_dic, wav_targ=wav_targ_wlc) # returns  data_dic['wlc'] and data_dic['wlc_var']

print ('\n get_wlc..')
print (data_dic.keys())
       


# 7. define t_start and set first time to 0;  make copies of slc, var and wav, before any binning.
# =============================================================================
bjd = data_dic['bjd']
time = data_dic['t'] = data_dic['bjd']- data_dic['bjd_orig'][0]
data_dic['t_start'] =  data_dic['bjd_orig'][0]
t0_est = t0_est - data_dic['bjd_orig'][0]

 
slc_orig = np.copy( data_dic['slc'])
var_orig = np.copy( data_dic['var'])
wav_orig = np.copy( data_dic['wav'])


# =============================================================================
# 8. wavelength selection for slc  : generally no needed for o1, but use for o2 which has a small usable range
# ============================================================================= 
if 'o2' in f:
    wav_targ =[0.63, 0.85] # 
    data_dic = pipeline_stage_3_lib.wavelength_range(data_dic, wav_targ)
    
# # =============================================================================
# 9. wavelength binning


bin_size = 50 # R varies from 20 to 60 - currently 50 but change to 2 for k2 18b native
R = 50
# bin_type = 'col'
# bin_type = 'R-bin'
bin_type = 'wavgrid'
# wavgrid= None
# bin_type = 'col_grid_R' #change back to this

lit_vals = np.loadtxt('/Volumes/Crucial X9/JexoPipe_NIRISS_raw_files/published_spec_gj9827d.txt')
lit_wav = lit_vals[:,0]
wavgrid = lit_wav[10:]
if 'o2' in f:
    wavgrid = lit_wav[:10]

# df = pd.read_csv('/Volumes/Crucial X9/JexoPipe_NIRISS_raw_files/GJ_9827d_lit.csv', header=None)
# wavgrid_lit = np.array(df.iloc[:,0])[::3]
# wavgrid = wavgrid_lit

data_dic = pipeline_stage_3_lib.wavelength_bin(data_dic, bin_type=bin_type, R=R, bin_size=bin_size, wavgrid=wavgrid, colgrid=None)
xx
print ('\n wavelength_bin...')
print (data_dic.keys())


R_ = data_dic['wav']/ np.gradient(data_dic['wav'])
print (R_)
print (data_dic['wav'])

if bin_type== 'col_grid_R':
    data_dic['wav'] = data_dic['wav'][1:]
    data_dic['wav_edges'] = data_dic['wav_edges'][1:]
    data_dic['slc'] = data_dic['slc'][:,1:]
    data_dic['var'] = data_dic['var'][:,1:]
 
R_ = data_dic['wav']/ np.gradient(data_dic['wav'])
print (R_)
print (data_dic['wav'])
print (data_dic['wav_edges'])


# =============================================================================
# 
# =============================================================================
# 9a. wavelength binning to allow for binned hi res spectra 
bin_type = 'col'
bin_size = 2
slc0, wav0, var0  = pipeline_stage_3_lib.wavelength_bin_random(slc_orig, wav_orig, var_orig, bin_type=bin_type, R=R, bin_size=bin_size, wavgrid=None)

print ('\n hi res wavelength_bin...')
print (data_dic.keys())


R_ = wav0/np.gradient(wav0)
 
data_dic['wav_orig'] = wav0
data_dic['slc_orig'] = slc0
data_dic['var_orig'] = var0


# =============================================================================
# 10. load and bin ldcs
# ============================================================================= 
ldc_type = 'quad'
ldc_file = None
# ldc_file = '/Users/c1341133/Downloads/ExoCTK_L1689_MIRI.txt'
# ldc_file = '/Users/c1341133/Downloads/ExoCTK_wasp39b_miri.txt'
# data_dic = pipeline_stage_3_lib.load_ldc_file(data_dic, ldc_file)
data_dic['ldc_type'] = ldc_type

#WASP107 Dyrek:
# M_H = 0.020; Teff =4425; logg = 4.633

# #WASP39b Powell:
# M_H =	0.01; Teff =5485; logg = 	4.453

# cob-K  :
# M_H =	0.12; Teff =3457; logg = 	4.7864000

#toi 1231
# M_H =	0.0410; Teff =3553; logg = 	4.767

#k2-18 b
# M_H =		0.12; Teff =3457; logg = 	4.7864

#wasp-96 b
# M_H =			0.14; Teff =	5500; logg = 		4.42


#wasp-52 b
# M_H =			0.030; Teff =	5000; logg = 		4.5

#wasp-17 b
# M_H = -0.25; Teff = 6550; logg = 4.149

# lhs 1140 b
# M_H = -0.15; Teff = 3096; logg = 5.041

# gj 9827 d
M_H = -0.29; Teff = 4236; logg = 4.70

# mode = 'JWST_MIRI_LRS'   
# mode = 'JWST_NIRSpec_G395H' 
mode = 'JWST_NIRISS_SOSSo1' 
if 'o2' in f:
    mode = 'JWST_NIRISS_SOSSo2' 

data_dic = pipeline_stage_3_lib.get_ldc_exotic(data_dic, M_H, Teff,logg, mode)

print ('\n get_ldc_exotic...')
print (data_dic.keys())


# =============================================================================
#  prepare for fitting
# =============================================================================
# wlc =   data_dic['wlc']
# wlc_var = data_dic['wlc_var']
slc = data_dic['slc'] 
channel = data_dic['channel']


if data_dic['wav'].shape[0] != data_dic['ldc_wav'].shape[0]:
    ldc_ = np.zeros((2, data_dic['wav'].shape[0]))
    ldc_[:,0:data_dic['ldc_wav'].shape[0]] = data_dic['ldc']
    ldc_[:,data_dic['ldc_wav'].shape[0]:] = data_dic['ldc'][:,-1][:, np.newaxis]
    data_dic['ldc'] = ldc_
    
# k2-18b
# depth = 0.05375; ars = 83.83; inc = 89.5785; per = 32.940045

# # toi-1468c
# depth = 0.0526; ars =49.8; inc= 89.220; per= 15.532477

# wasp 39-b
# depth = 0.1457; ars =11.37; inc= 87.75; per= 4.05528043

# wasp 96-b
# depth = 0.1186; ars =9.255; inc= 85.6; per=3.42525674

# # wasp 52-b
# depth = 0.1646; ars =	7.38; inc= 85.35; per=	1.74978117

# wasp 17-b
# depth = 0.12472; ars = 7.110; inc = 87.217; per = 3.73548546

#toi-1231 b
# depth = 0.0701; ars =	58.1; inc= 89.73; per= 24.245586

# lhs 1140b
# depth = 0.07439; ars = 95.2; inc = 89.86; per = 24.73723

# gj 9827 d
depth = 0.03073; ars = 19.739; inc = 87.41; per = 6.201830



data_dic['f'] =  f

# =============================================================================
# light curve fitting
# --- in general if has spots use dynesty --- if no spots use MCMC or LMFit
# =============================================================================

# =============================================================================
data_dic['trend'] ='linear'

lc_model_params ={}
lc_model_params['depth']= {'init':depth, 'fit': True, 'log': False, 'range': 0.1, 'log_range': 4, 'mu': depth, 'sigma': 0.01,'prior_type': 'uniform'}
lc_model_params['t0']= {'init': t0_est, 'fit': True, 'log': False, 'range': 0.1, 'log_range': None, 'prior_type': 'uniform'}
lc_model_params['ars']= {'init': ars, 'fit': True, 'log': False, 'range': 3, 'log_range': 5,  'prior_type': 'uniform'}
lc_model_params['inc']= {'init':	inc, 'fit': True, 'log': False, 'range': 1.0, 'log_range': 5,  'prior_type': 'uniform'}
# lc_model_params['u0']= {'init': data_dic['wlc_ldc'][0], 'fit':  True, 'log': False, 'range': [0,1], 'log_range': None, 'prior_type': 'uniform'}
# lc_model_params['u1']= {'init': data_dic['wlc_ldc'][1], 'fit': True, 'log': False, 'range': [0,1], 'log_range': None, 'prior_type': 'uniform'}
lc_model_params['u0']= {'init': data_dic['wlc_ldc'][0], 'fit': False, 'log': None, 'range': None, 'log_range': None, 'prior_type': 'uniform'}
lc_model_params['u1']= {'init': data_dic['wlc_ldc'][1], 'fit': False, 'log': None, 'range': None, 'log_range': None, 'prior_type': 'uniform'}
lc_model_params['w']= {'init': 90, 'fit': False, 'log': None, 'range': None, 'log_range': None, 'prior_type': 'uniform'}
lc_model_params['ecc']= {'init': 0, 'fit': False, 'log': None, 'range': None, 'log_range': None, 'prior_type': 'uniform'}
lc_model_params['per']= {'init': per, 'fit': False, 'log': None, 'range': None, 'log_range': None, 'prior_type': 'uniform'}

if data_dic['trend'] == 'quadratic':
    syst_model_params ={}
    syst_model_params['a']= {'init':1, 'fit': True, 'log': True, 'range': None, 'log_range': 8, 'prior_type': 'uniform'}
    syst_model_params['b']= {'init':1, 'fit': True, 'log': True, 'range': None, 'log_range': 8, 'prior_type': 'uniform'}
    syst_model_params['c']= {'init':1, 'fit': True, 'log': True, 'range': None, 'log_range': 8, 'prior_type': 'uniform'}
    syst_model_params['d']= {'init':0, 'fit': False, 'log': True, 'range': None, 'log_range': 8, 'prior_type': 'uniform'}

if data_dic['trend'] == 'linear':
    syst_model_params ={}
    syst_model_params['a']= {'init':1, 'fit': 1, 'log':  0, 'range': 0.025, 'log_range': 10, 'prior_type': 'uniform'}
    syst_model_params['b']= {'init':0, 'fit': 1, 'log': 0, 'range':  [-0.025, 0.025], 'log_range': 10, 'prior_type': 'uniform'}
    syst_model_params['c']= {'init':0, 'fit': 0, 'log':  0, 'range': [-0.025, 0.025], 'log_range': 10, 'prior_type': 'uniform'}
    syst_model_params['d']= {'init':0, 'fit': 0, 'log':  0, 'range': [0, 0.08333], 'log_range': 10, 'prior_type': 'uniform'}

if data_dic['trend'] == 'lin_exp':
    syst_model_params ={}
    syst_model_params['a']= {'init':1, 'fit': 1, 'log':  0, 'range': 0.025, 'log_range': 10, 'prior_type': 'uniform'}
    syst_model_params['b']= {'init':0, 'fit': 1, 'log': 0, 'range':  [-0.025, 0.025], 'log_range': 10, 'prior_type': 'uniform'}
    syst_model_params['c']= {'init':0, 'fit': 1, 'log':  0, 'range': [-0.025, 0.025], 'log_range': 10, 'prior_type': 'uniform'}
    syst_model_params['d']= {'init':0, 'fit': 1, 'log':  0, 'range': [0, 0.08333], 'log_range': 10, 'prior_type': 'uniform'}


data_dic['lc_model_params'] = dict(lc_model_params)
data_dic['syst_model_params'] = dict(syst_model_params)

# tag__ = f[f.find('COMBINED')+9 : f.find('.fits')]
tag__ = f[f.find('COMBINED')+27 : f.find('.fits')]
obs = f[f.find('jw0'): f.find('COMBINED')-1]
tag__ = '%s_%s'%(obs, tag__)


# extra_tag ='quadratic'
extra_tag ='linear'


total_tag = '%s_%s'%(tag__, extra_tag)

print (total_tag)
 
data_dic['extra_tag'] = extra_tag
data_dic['tag'] = total_tag
data_dic['fit_wlc'] = 1 
data_dic['fit_spot'] = 0
data_dic['mp'] =0
data_dic['hi_fidel'] =0
data_dic['kipping'] =True
# data_dic['kipping'] =False

data_dic['syst_model_type'] = data_dic['trend']

folder_spec = "%s_%s/order%s"%(channel, total_tag, data_dic['order'])
if not os.path.exists('./spectrum/%s'%(folder_spec)):
    os.makedirs('./spectrum/%s'%(folder_spec))
folder_spec = './spectrum/%s'%(folder_spec)
data_dic['folder'] = None

 
folder_wlc = "%s/%s/order%s"%(total_tag, channel, data_dic['order'])

data_dic['folder_wlc'] = folder_wlc 

data_dic['slc_spot_fit'] =0
# =============================================================================
# the following lines are used for LMFits with spots for slc in combination with a previous dynesty run 
# =============================================================================

# np.save('./slc.npy', data_dic['slc'])    
# np.save('./wav.npy',  data_dic['wav'])    
# np.save('./t.npy',  data_dic['t'])     
# np.save('./var.npy', data_dic['var'])   

# np.save('./slc2.npy', data_dic['slc'])    
# np.save('./wav2.npy',  data_dic['wav'])    
# np.save('./t2.npy',  data_dic['t'])     
# np.save('./var2.npy', data_dic['var'])
 

if   data_dic['slc_spot_fit'] ==1:

    slc = data_dic['slc'] 
    var = data_dic['var'] 
    t = data_dic['t'] 
    wav= data_dic['wav']
    
    # res_ = 'hi'
    res_ ='lo'
     
    # set folder name for the low res spectra 
    folder = 'spec_results_niriss_R=50'
    
    import spotfit7_wLMfit_niriss_run as spfit
    
    
    if res_== 'lo':
        spfit.fit_spec(slc, var, t, wav, folder, marker ='gx')
     
    if res_== 'hi':
        extra_tag = '_2pix'
        # extra_tag = '_R=300__'
         
        hi_res = spfit.fit_spec(data_dic['slc_orig'], data_dic['var_orig'], t, data_dic['wav_orig'], folder, marker ='gx', res='hi', wav_edges=data_dic['wav_edges'], wav_lo_res=wav, 
                                order=data_dic['order'], extra_tag=extra_tag)
    
     

# aa = np.vstack((data_dic['t'], data_dic['wlc'], data_dic['wlc_var']**0.5)).T
# np.savetxt('./wlc_final_toi_1231b_niriss.txt', aa)
# xxxx

# =============================================================================
# main wlc mcmc
# # =============================================================================


# data_dic['mcmc_option']=0
# data_dic['mcmc_option']=4
# data_dic['custom_burn_in']=1000
# data_dic['custom_production']=2000

# data_dic['wlc_fit'] = 1
# data_dic['lm_fit'] = 0
# # con_dic['spot_fit'] = 1
# data_dic['spot_fit'] = data_dic['fit_spot'] = 0
 
# q1,q2,q3,q4,q5,q6,  mp =0,0,0,0,0,0,0
# return_dic  = fit_function(0, q1, q2, q3, q4, q5, q6, data_dic)


if data_dic['order'] == 2:
    total_tag = total_tag.replace('o2', 'o1')

flatchain = np.load('./wlc_data/%s/niriss/nis_o%s/mcmc/wlc_sampler_flatchain.npy'%(total_tag, '1'))
# flatchain = np.load('./wlc_data/%s/niriss/nis_o%s/mcmc/wlc_sampler_flatchain.npy'%(total_tag, data_dic['order']))

try:
    # depth,t0, ars, inc, u0, u1, a, b = np.median(flatchain, axis=0)
    depth,t0, ars, inc, a, b = np.median(flatchain, axis=0)
except:
    # depth,t0, ars, inc, u0, u1, a, b, c = np.median(flatchain, axis=0)
    depth,t0, ars, inc, a, b, c = np.median(flatchain, axis=0)
    
print (depth, t0, ars, inc)

# ars = 18.69654401374393
# inc = 87.2372167621478

data_dic['lc_model_params']['t0']= {'init': t0, 'fit': 0, 'log': False, 'range': 0.1, 'log_range': None, 'prior_type': 'uniform'}
data_dic['lc_model_params']['ars']= {'init': ars, 'fit': 0, 'log': False, 'range': 10, 'log_range': 5,  'prior_type': 'uniform'}
data_dic['lc_model_params']['inc']= {'init': inc, 'fit': 0, 'log': False, 'range': 0.1, 'log_range': 5,  'prior_type': 'uniform'}


# # =============================================================================
# #slc mcmc at low res and collect ldcs
# # =============================================================================

data_dic['mcmc_option']=4
data_dic['custom_burn_in'] =1000
data_dic['custom_production'] =2000
data_dic['lm_fit'] = 0
data_dic['wlc_fit'] = data_dic['fit_wlc'] = 0 
data_dic['spot_fit'] = data_dic['fit_spot'] = 0
mp = data_dic['mp'] = 1
data_dic['folder'] = folder_spec

  
if __name__ == "__main__":
                
                 div_list = np.arange(data_dic['slc'].shape[1]) [::-1].tolist() 
                 div=0
                 if len(div_list ) >500:
                    div = 1
                 div_list_list = []
                 if div == 1:
                    seglength = int(len(div_list)/4)
                    remainder = len(div_list)%4
                    idx = np.arange(0, len(div_list)+1, seglength)
                    for i in range(4):
                        div_list_list.append(np.arange(idx[i], idx[i+1], 1))
                 else:
                    div_list_list = [div_list]
   
                 folder = data_dic['folder'] 
                 
                 
                 if data_dic['mp']==1:
                     ct=-1
                     
                     for div_list in div_list_list:
                      
                         ct+=1
                         q1 = Queue()
                         q2 = Queue()
                         q3 = Queue()
                         q4 = Queue()
                         q5 = Queue()
                         q6 = Queue()              
                         processes = [Process(target=fit_function, args=(xx, 
                                                                    q1, q2, q3, q4, q5, q6,
                                                                    data_dic)) for xx in div_list]
                         
                         for p in processes:
                             p.start()

                         wav_list = [q1.get() for p in processes]
                         params_names_list = [q2.get() for p in processes]
                         sampler_results_list = [q3.get() for p in processes]
                         exp_sampler_results_list = [q4.get() for p in processes]
                         signs_list = [q5.get() for p in processes]
                         lmfit_list = [q6.get() for p in processes]
    
                         for p in processes:
                             p.join()
                         
                         if ct==0:
                             wav_stack = wav_list
                             params_names_stack  =  params_names_list[0]
                             sampler_results_stack  =  sampler_results_list
                             exp_sampler_results_stack  = exp_sampler_results_list
                             signs_stack  = signs_list
                             lmfit_stack = lmfit_list
             
                         else:
                             wav_stack = wav_stack + wav_list
                             params_names_stack  =  params_names_stack 
                             sampler_results_stack  =  sampler_results_stack + sampler_results_list
                             exp_sampler_results_stack  = exp_sampler_results_stack + exp_sampler_results_list
                             signs_stack  = signs_stack + signs_list
                             lmfit_stack = lmfit_stack + lmfit_list
             
                         
                         np.save('%s/spectrum_wav.npy'%(folder), np.array(wav_stack))
                         np.save('%s/spectrum_params.npy'%(folder), np.array(params_names_stack))
                         np.save('%s/spectrum_sampler_results.npy'%(folder), np.array(sampler_results_stack)) 
                         np.save('%s/spectrum_exp_sampler_results.npy'%(folder), np.array(exp_sampler_results_stack))
                         np.save('%s/spectrum_signs.npy'%(folder), np.array(signs_stack))
                         np.save('%s/spectrum_lmfit.npy'%(folder), np.array(lmfit_stack))
    
                         idx = np.argsort(wav_stack)
                         wav = np.array(wav_stack)[idx]; sampler_results=np.array(sampler_results_stack)[idx]; exp_sampler_results=np.array(exp_sampler_results_stack)[idx]

                         dic={}
                         for i in range(params_names_stack.shape[1]):
                             param = params_names_stack[1][i]
                             median_list = []
                             pc16_list =[]
                             pc84_list =[]
                             for j in range(exp_sampler_results.shape[0]):
                         
                                 pc16 = exp_sampler_results[j][0][i]
                                 median = exp_sampler_results[j][1][i]
                                 pc84 = exp_sampler_results[j][2][i]
                                 
                                 median_list.append(median)
                                 pc16_list.append(pc16)
                                 pc84_list.append(pc84)
                             dic[param]={}
                             dic[param]['median'] = np.array(median_list)
                             dic[param]['pc16'] =  np.array(pc16_list)
                             dic[param]['pc84'] =  np.array(pc84_list)
                      
                         
                         x = 'depth'
                          
                         median = dic[x]['median']
                         err_minus = dic[x]['median']-dic[x]['pc16']
                         err_plus = dic[x]['pc84']- dic[x]['median']
                         yerr = (err_minus,err_plus)
                         yerr0 = (err_minus/1e6,err_plus/1e6)
             
                 else:
                    
                    qq = 3
                    q1,q2,q3, q4, q5, q6, mp =0,0,0,0, 0,0,0
                    fit_function(qq, q1, q2, q3, q4, q5, q6, data_dic)
