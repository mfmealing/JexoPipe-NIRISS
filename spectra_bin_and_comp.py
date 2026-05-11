import numpy as np
import matplotlib; matplotlib.style.use('classic')
import matplotlib.pyplot as plt
from scipy import interpolate
from  scipy.integrate import cumulative_trapezoid as cumtrapz
import pandas as pd
from matplotlib import gridspec


def rebin(spec, plus, minus, wav, wav2=None):
    spec = spec
    wav = wav
    plus = plus
    minus = minus
    
    var_plus = plus**2
    var_minus = minus**2
    
    if wav2.all()==None or len(wav2) ==0:
        wav2=wav
    fp = spec
    xp = wav
    x = wav2
# ============================================================================
    fp[np.isnan(fp)] = 0
  # first select non-zero elements
    idx_x = np.argwhere(x>0).T[0]
    idx_xp = np.argwhere(xp>0).T[0]

    x_ = x[idx_x]
    xp_ = xp[idx_xp]
    fp_ = fp[idx_xp]

    var_plus_ = var_plus[idx_xp]
    var_minus_ = var_minus[idx_xp]

   # select elements in input grid within wl range of new grid
    idx = np.where(np.logical_and(xp_ > 0.9*x_.min(), xp_ < 1.1*x_.max()))[0]
    xp = xp_[idx]
    fp = fp_[idx]
    var_plus = var_plus_[idx]
    var_minus = var_minus_[idx]
 
   #To check that the binning or interp is applied correctly uncomment following
   # print (abs(np.diff(xp_)).min() ,  abs(np.diff(x_)).min())

  
    diff = np.diff(x)
    diff_pos = np.array((diff/2).tolist() +[diff[-1]/2]) # edge bins fix
    diff_neg = np.array([diff[0]/2] +(diff/2).tolist()) # edge bins fix
    delta = diff_pos+diff_neg
    
    c = cumtrapz(fp, x=xp)
    # c = fp*np.gradient(wav)
    # c = np.cumsum(c)

    xpc = xp[1:]
    new_c_1 = np.interp(x-diff_neg, xpc, c, 
                       left=0.0, right=0.0)
    new_c_2 = np.interp(x+diff_pos, xpc, c, 
                       left=0.0, right=0.0)
    # new_f = ((new_c_2 - new_c_1)/np.gradient(wav2))  
    new_f = ((new_c_2 - new_c_1)/delta)  

    
    
    # c_err = cumtrapz(var_plus, x=xp)
    c_err = var_plus*np.gradient(wav)**2  #apply weights to variances (use the square of the wavelength gradient - should return original err if unbinned or close to it)
    c_err = np.cumsum(c_err)
    xpc = xp
    new_c_1 = np.interp(x-diff_neg, xpc, c_err, 
                       left=0.0, right=0.0)
    new_c_2 = np.interp(x+diff_pos, xpc, c_err, 
                       left=0.0, right=0.0)
    # print ('qq', (new_c_2 - new_c_1)[10])
    # new_err1 = ((new_c_2 - new_c_1)**0.5)/np.gradient(wav2)
    new_err1 = ((new_c_2 - new_c_1)**0.5)/delta
    
    c_err = var_minus*np.gradient(wav)**2
    c_err = np.cumsum(c_err)
    xpc = xp
    new_c_1 = np.interp(x-diff_neg, xpc, c_err, 
                       left=0.0, right=0.0)
    new_c_2 = np.interp(x+diff_pos, xpc, c_err, 
                       left=0.0, right=0.0)
    # print ('qq', (new_c_2 - new_c_1)[10])
    # new_err1 = ((new_c_2 - new_c_1)**0.5)/np.gradient(wav2)
    new_err2 = ((new_c_2 - new_c_1)**0.5)/delta   
    

    idx  = np.argwhere(new_f>0).T[0]


    x = x[idx]
    new_f = new_f[idx]
    new_err1 = new_err1[idx]
    new_err2 = new_err2[idx]

    
    idx  = np.argwhere(new_f>np.median(new_f)*0.5).T[0]

    x = x[idx]
    new_f = new_f[idx]
    new_err1 = new_err1[idx]
    new_err2 = new_err2[idx]
    

    return [x, new_f, new_err1]


def extract(wav, ex):
    idx = np.argsort(wav)
    wav = wav[idx]

    print(np.average(wav/np.gradient(wav)))

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
       

    half_bin_width  = np.gradient(wav)/2
    av_err = (err_plus + err_minus)/2
    xerr = half_bin_width
    
    return [wav, spec, xerr, av_err]
    

wav = np.load('/Volumes/Crucial X9/JexoPipe-NIRISS/spectrum/niriss_jw01353101001_04101_00001_box_extract_o1_WASP_17_col_2_bin_linear/order1/spectrum_wav.npy')
ex = np.load('/Volumes/Crucial X9/JexoPipe-NIRISS/spectrum/niriss_jw01353101001_04101_00001_box_extract_o1_WASP_17_col_2_bin_linear/order1/spectrum_exp_sampler_results.npy')[:,:,0]

wav, spec, xerr, yerr = extract(wav, ex)

# print(np.average(wav/np.gradient(wav)))
wav_bin = np.array_split(wav, 54)
wav2 = np.array([bin.mean() for bin in wav_bin])
# print(np.average(wav2/np.gradient(wav2)))

wav_new, spec_new, yerr_new = rebin(spec, yerr, yerr, wav, wav2)

# plt.figure(figsize=(19,8))
# plt.errorbar(wav_new, spec_new, yerr_new, fmt='o', capsize=4, color='r', label='JexoPipe')


df = pd.read_csv('/Volumes/Crucial X9/JexoPipe_NIRISS_raw_files/WASP-17b_NIRISS_SOSS_transmission_spectrum_pixel_o1.csv')
lit_wav, lit_xerr, lit_spec, lit_yerr = df['wave'], df['wave_err'], df['dppm']/1e6, df['dppm_err']/1e6


lit_wav_new, lit_spec_new, lit_yerr_new = rebin(lit_spec, lit_yerr, lit_yerr, lit_wav, wav2)
# plt.errorbar(lit_wav_new, lit_spec_new, lit_yerr_new, fmt='o', capsize=4, color='k', label='lit')
# plt.legend()

res = spec_new - lit_spec_new
res_err = ((yerr_new)**2 + lit_yerr_new**2)**0.5


# ---------- GJ 9827 d -----------
# # plt.figure('test')
# df = pd.read_csv('/Volumes/Crucial X9/JexoPipe_NIRISS_raw_files/GJ_9827d_lit.csv', header=None)
# lit_wav = np.array(df.iloc[:,0])[::3]
# av_wav = ((lit_wav + np.roll(lit_wav,1))/2.0)[1::2]
# lit_spec = np.array(df.iloc[:,1])[::3] / 1e6
# av_spec = ((lit_spec + np.roll(lit_spec,1))/2.0)[1::2]
# up_quart = np.array(df.iloc[:,1])[1::3] / 1e6
# av_up = ((up_quart + np.roll(lit_spec,1))/2.0)[1::2]
# low_quart = np.array(df.iloc[:,1])[2::3] / 1e6
# av_low = ((low_quart + np.roll(lit_spec,1))/2.0)[1::2]
# up_err = av_up - av_spec
# low_err = av_spec - av_low
# # up_err = up_quart - lit_spec
# # low_err = lit_spec - low_quart
# lit_yerr = (up_err + low_err)/2

# av_wav = av_wav[3:]
# av_spec = av_spec[3:]
# lit_yerr = lit_yerr[3:]

# # plt.errorbar(av_wav, av_spec, lit_yerr, fmt='o', capsize=4, color='k', label='Piaulet-Ghorayeb et al.')


# wav = np.load('/Volumes/Crucial X9/JexoPipe-NIRISS/spectrum/niriss_jw04098007001_04101_00001_box_extract_o1_GJ_9827_linear/order1/spectrum_wav.npy')
# ex = np.load('/Volumes/Crucial X9/JexoPipe-NIRISS/spectrum/niriss_jw04098007001_04101_00001_box_extract_o1_GJ_9827_linear/order1/spectrum_exp_sampler_results.npy')[:,:,0]
# wav, spec, xerr, yerr = extract(wav, ex)
# wav_new, spec_new, yerr_new = rebin(spec, yerr, yerr, wav, av_wav)
# # plt.errorbar(wav_new, spec_new, yerr_new, fmt='o', capsize=4, color='r', label='visit 1', alpha=0.4)

# wav = np.load('/Volumes/Crucial X9/JexoPipe-NIRISS/spectrum/niriss_jw04098008001_04101_00001_box_extract_o1_GJ_9827_V2_linear/order1/spectrum_wav.npy')
# ex = np.load('/Volumes/Crucial X9/JexoPipe-NIRISS/spectrum/niriss_jw04098008001_04101_00001_box_extract_o1_GJ_9827_V2_linear/order1/spectrum_exp_sampler_results.npy')[:,:,0]
# wav2, spec2, xerr2, yerr2 = extract(wav, ex)
# wav_new2, spec_new2, yerr_new2 = rebin(spec2, yerr2, yerr2, wav2, av_wav)
# # plt.errorbar(wav_new, spec_new2, yerr_new2, fmt='o', capsize=4, color='b', label='visit 2', alpha=0.4)


# spec_final = (spec_new + spec_new2)/2
# yerr_final = ((yerr_new + yerr_new2)/2)/(np.sqrt(2))


# # plt.errorbar(wav_new, spec_final, yerr_final, fmt='o', capsize=4, color='b', label='JexoPipe')
# # plt.legend()

# res = spec_final - av_spec
# res_err = ((yerr_final)**2 + lit_yerr**2)**0.5

# ---------------k2 18 b-------------------------------

# wav = np.load('/Volumes/Crucial X9/JexoPipe-NIRISS/spectrum/niriss_jw02722003001_04101_00001_box_extract_o1_K2_18_col_2_bin_linear/order1/spectrum_wav.npy')
# ex = np.load('/Volumes/Crucial X9/JexoPipe-NIRISS/spectrum/niriss_jw02722003001_04101_00001_box_extract_o1_K2_18_col_2_bin_linear/order1/spectrum_exp_sampler_results.npy')[:,:,0]

# wav, spec, xerr, yerr = extract(wav, ex)

# print(np.average(wav/np.gradient(wav)))
# wav_bin = np.array_split(wav, 55)
# wav2 = np.array([bin.mean() for bin in wav_bin])

# wav_new, spec_new, yerr_new = rebin(spec, yerr, yerr, wav, wav2)

# # plt.figure(figsize=(19,8))
# # plt.errorbar(wav_new, spec_new, yerr_new, fmt='o', capsize=4, color='r', label='JexoPipe')

# with open('/Volumes/Crucial X9/NIRISS_Pipeline/Data/K2_18b_NIRISS/K2-18b_niriss_soss_native.txt', 'r') as f:
#     header_line = f.readline().strip()

# column_names = header_line.split(',')
# df = pd.read_csv('/Volumes/Crucial X9/NIRISS_Pipeline/Data/K2_18b_NIRISS/K2-18b_niriss_soss_native.txt', sep='\s+', skiprows=1, names=column_names)

# lit_wav = df[column_names[0]].values
# lit_depth = df[column_names[2]].values
# lit_depth_err = df[column_names[3]].values

# lit_wav_new, lit_spec_new, lit_yerr_new = rebin(lit_depth, lit_depth_err, lit_depth_err, lit_wav, wav2)

# res = spec_new - lit_spec_new
# res_err = ((yerr_new)**2 + lit_yerr_new**2)**0.5



plt.figure(figsize=(19,6.5))
plt.errorbar(wav_new, spec_new, yerr_new, fmt='o', capsize=4, color='g', label='JexoPipe')
# plt.errorbar(av_wav, spec_final, yerr_final, fmt='o', capsize=4, color='b', label='JexoPipe') # for GJ 9827 d
# plt.errorbar(lit_wav_new, lit_spec_new, lit_yerr_new, fmt='s', capsize=4, color='k', label='Madhusudhan et al. 2023')
# plt.errorbar(av_wav, av_spec, lit_yerr, fmt='s', capsize=4, color='k', label='Piaulet-Ghorayeb et al. 2024')
plt.errorbar(lit_wav_new, lit_spec_new, lit_yerr_new, fmt='s', capsize=4, color='k', label='Louie et al. 2025')
plt.xlim(0.8,2.9)
plt.ylim(0.0145, 0.0161)
plt.xticks(fontsize=18, fontweight='bold')
plt.yticks(fontsize=18, fontweight='bold')
plt.ylabel('Transit depth (%)', labelpad=20, fontsize=18, fontweight='bold')
plt.xlabel('Wavelength (µm)', fontsize=18, fontweight='bold')
plt.legend(loc='upper left', fontsize=18, prop={'weight': 'bold'})
plt.savefig('/Users/c24050258/Library/CloudStorage/OneDrive-CardiffUniversity/General/Conferences/UKExoM 2026/K2_18b_spectrum_test.png')

xx


# =========== final plot code ===========
gs = gridspec.GridSpec(3, 1, height_ratios=[3,1,1], left=0.17, bottom=0.1, right=0.99, top=0.97, wspace=None, hspace=0.28)
ax0 = plt.subplot(gs[0])

plt.errorbar(wav_new, spec_new, yerr=yerr_new, 
             fmt = 'o', color='r', label = 'JexoPipe', capsize=4)
plt.errorbar(wav_new, av_spec, yerr=lit_yerr, 
             fmt = 's', color='k', label = 'Piaulet-Ghorayeb et al.', capsize=4)
plt.xlim(0.8,2.9)
plt.xticks(fontsize=18, fontweight='bold')
plt.yticks(fontsize=18, fontweight='bold')
plt.ylabel('Transit depth (%)', labelpad=20, fontsize=18, fontweight='bold')
plt.legend(loc='best', fontsize=18, prop={'weight': 'bold'})


ax1 = plt.subplot(gs[1])

plt.errorbar(wav_new, res, res_err, fmt='o', markeredgecolor='none',
              markerfacecolor= 'k', ecolor ='k', ls='none',
              capsize=0, elinewidth=2, alpha = 0.6)
plt.xlim(0.8,2.9)
plt.ylim(-0.0005,0.0005)
plt.locator_params(axis='y', nbins=4)
plt.ylabel('Residuals', fontsize=18, fontweight='bold')
plt.plot([0,6],[0,0], ':', linewidth=2, color='k')
plt.xticks(fontsize=18, fontweight='bold')
plt.yticks(fontsize=18, fontweight='bold')



ax2 = plt.subplot(gs[2])

plt.plot(wav_new, yerr_new, '.', color='r', linewidth =2)
plt.plot(wav_new, lit_yerr, '.', color='k', linewidth =2)
plt.ylabel('Average\nerror', fontsize=18, fontweight='bold') 
plt.xlabel('Wavelength (µm)', fontsize=18, fontweight='bold')
plt.xlim(0.8,2.9)
plt.locator_params(axis='y', nbins=4)
plt.xticks(fontsize=18, fontweight='bold')
plt.yticks(fontsize=18, fontweight='bold')
