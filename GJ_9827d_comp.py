import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

data_v1 = np.load('/Volumes/Crucial X9/JexoPipe-NIRISS/spectrum/niriss_jw04098007001_04101_00001_box_extract_o1_GJ_9827_lit_comp_linear/order1/final_data.npy')
data_v2 = np.load('/Volumes/Crucial X9/JexoPipe-NIRISS/spectrum/niriss_jw04098008001_04101_00001_box_extract_o1_GJ_9827_V2_lit_comp_linear/order1/final_data.npy')

wav_v1, xerr_v1, spec_v1, yerr_v1 = data_v1[:,0], data_v1[:,1], data_v1[:,2], data_v1[:,3]
wav_v2, xerr_v2, spec_v2, yerr_v2 = data_v2[:,0], data_v2[:,1], data_v2[:,2], data_v2[:,3]

spec_av = (spec_v1+spec_v2)/2
yerr_av = ((yerr_v1+yerr_v2)/2)/(np.sqrt(2))

plt.figure(figsize=(19,6.5))
# plt.errorbar(wav_v1,spec_v1,xerr=xerr_v1,yerr=yerr_v1, fmt='o', capsize=4, alpha=0.5, color='red')
# plt.errorbar(wav_v2,spec_v2,xerr=xerr_v2,yerr=yerr_v2, fmt='o', capsize=4, alpha=0.5, color='blue')
plt.errorbar(wav_v2,spec_av,yerr=yerr_av, fmt='o', capsize=4, color='b', label='JexoPipe')


# df = pd.read_csv('/Volumes/Crucial X9/JexoPipe_NIRISS_raw_files/GJ_9827d_lit.csv', header=None)
# wav = np.array(df.iloc[:,0])[::3]
# av_wav = ((wav + np.roll(wav,1))/2.0)[1::2]
# spec = np.array(df.iloc[:,1])[::3] / 1e6
# av_spec = ((spec + np.roll(spec,1))/2.0)[1::2]
# up_quart = np.array(df.iloc[:,1])[1::3] / 1e6
# av_up = ((up_quart + np.roll(spec,1))/2.0)[1::2]
# low_quart = np.array(df.iloc[:,1])[2::3] / 1e6
# av_low = ((low_quart + np.roll(spec,1))/2.0)[1::2]
# up_err = av_up - av_spec
# low_err = av_spec - av_low
# av_err = ((up_err+low_err)/2)/(np.sqrt(2))

lit_vals = np.loadtxt('/Volumes/Crucial X9/JexoPipe_NIRISS_raw_files/published_spec_gj9827d.txt')
lit_wav = lit_vals[:,0]
lit_spec = lit_vals[:,1]
lit_err = lit_vals[:,2]


plt.errorbar(lit_wav[10:],lit_spec[10:], lit_err[10:],fmt='s', capsize=4, color='k', label='Piaulet-Ghorayeb et al. 2024')
plt.xlim(0.8,2.9)
plt.xticks(fontsize=18, fontweight='bold')
plt.yticks(fontsize=18, fontweight='bold')
plt.ylabel('Transit depth (%)', labelpad=20, fontsize=18, fontweight='bold')
plt.xlabel('Wavelength (µm)', fontsize=18, fontweight='bold')
plt.legend(loc='upper left', fontsize=18, prop={'weight': 'bold'})