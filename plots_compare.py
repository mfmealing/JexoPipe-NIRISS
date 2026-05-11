import netCDF4

 
import numpy as np
import matplotlib; matplotlib.style.use('classic')
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
from scipy.integrate import cumtrapz
from matplotlib import gridspec


def rebin2(dic, wav, wav2=None):

    x = 'depth'   
    median  =  dic[x]['median'] /1e6
    pc_84 = dic[x]['pc84'] /1e6
    pc_16 = dic[x]['pc16'] /1e6
    wav = wav
   
 
    # median = np.load('%s/spectrum_rprs2_median.npy'%(aa))
    # wav = np.load('%s/spectrum_wav.npy'%(aa))
    # pc_84 = np.load('%s/spectrum_rprs2_pc_84.npy'%(aa))
    # pc_16 = np.load('%s/spectrum_rprs2_pc_16.npy'%(aa))
    
    # median = np.load('%s/spectrum_rprs2_median.npy'%(aa))
    # wav = np.load('%s/spectrum_wav.npy'%(aa))
    # pc_84 = np.load('%s/spectrum_rprs2_pc_84.npy'%(aa))
    # pc_16 = np.load('%s/spectrum_rprs2_pc_16.npy'%(aa))
    
    plus = pc_84 - median
    minus = median - pc_16
    
    var_plus = plus**2
    var_minus = minus**2
    
    spec = median
    
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
    

    return [x, new_f, new_err1, new_err2]


def plott(aa,  ct, pipe, col='k'):

    spec = np.load('%s/spectrum_rprs.npy'%(aa))
    wav = np.load('%s/spectrum_wav.npy'%(aa))
    err1 = np.load('%s/spectrum_plus.npy'%(aa))
    err2 = np.load('%s/spectrum_minus.npy'%(aa))
    
    idx =  np.argwhere((wav>=2.87) & (wav <=5.14)).T[0]
    spec =spec[idx]
    wav= wav[idx]
    err1=err1[idx]
    err2=err2[idx]
    
    spec2 = spec**2
    upper = spec+err1
    lower = spec-err2
    upper2= upper**2
    lower2= lower**2
    err2plus = upper2-spec2
    err2minus = spec2-lower2
    


    # plt.figure('no err')
    # plt.plot(wav, spec2, '%s-'%(col))
    # plt.figure('w err')
    # plt.errorbar(wav, spec2, yerr=(err2minus,err2plus), fmt='%s'%(col))
    
    binsize = 5
    if 'BINNED' in aa:
        binsize=1
        col = 'b'
    idx = np.arange(1,len(wav), binsize)
   
    wav = (np.add.reduceat(wav,idx)/binsize)[:-1]
   
    spec2 = (np.add.reduceat(spec2,idx)/binsize)[:-1] *100
    err2plus = ((np.add.reduceat(err2plus**2,idx)**0.5)/binsize)[:-1] *100
    err2minus = ((np.add.reduceat(err2minus**2,idx)**0.5)/binsize)[:-1] *100
   
    
    # plt.figure('no err')
    # plt.plot(wav, spec2, '%s-'%(col))
    
    plt.figure('Grating comparison %s'%(pipe), dpi=100, figsize=(16,5), )
    plt.subplots_adjust(left=0.1, bottom=0.13, right=0.9, top=0.9, wspace=0, hspace=0)

    if 'nrs1' in aa:
        plt.errorbar(wav, spec2, yerr=(err2minus,err2plus), fmt='%so'%(col), ls='none',
                 capsize=0, elinewidth=2, label ='JexoPipe')
    else:
        plt.errorbar(wav, spec2, yerr=(err2minus,err2plus), fmt='%so'%(col), ls='none',
                 capsize=0, elinewidth=2)
    # plt.errorbar(wav, spec2, yerr=(err2minus,err2plus), marker='o', fmt='None', label ='JexoPipe')



filenames = ['/Users/c1341133/Downloads/G395H_paper_data/3_TRANSMISSION_SPECTRA/transit-spectrum-W39b-G395H-10pix_weighted-average.nc'
]

col = ['orange', 'c', 'm', 'g', 'r', 'b', 'y', 'purple', '0.5', 'pink', 'olive', 'rosybrown']
pipe = ['Grant', 'Tiberius', 'Alam', 'Espinoza-transitspectroscopy', 'Roy-Eureka-ExoTEP', 'Barat', 'Wakeford-Damiano', 'Inglis', 'Wallack', 'Evans', 'Flag-Eureka',
        'Alderson et al. 2023: weighted mean'
        ]
 

params = {"xtick.top": False, "ytick.right": False, "xtick.direction": "out", "ytick.direction": "out"}
plt.rcParams.update(params)
ax = plt.gca()

fontsize = 22
fontsize2 = 26
for no in range(1,13):
# for no in range(1,11):
# for no in [12]:
    
    # aa1 = "/Users/c1341133/Desktop/Subi_jwst_pipeline/spectrum/nrs1_1Dspec_uncorrected_for_step_X_run_Alphagrating_X_ALL/"

    # aa2 = "/Users/c1341133/Desktop/Subi_jwst_pipeline/spectrum/nrs2_1Dspec_uncorrected_for_step_X_run_Alphagrating_X_ALL/"

    # aa1= '/Users/c1341133/Desktop/Subi_jwst_pipeline/spectrum/nrs1_1Dspec_cr_rej_15_definitive_31_5_23_8_7_23_full_res_ALL/'
     
    # aa2= '/Users/c1341133/Desktop/Subi_jwst_pipeline/spectrum/nrs2_1Dspec_cr_rej_15_definitive_31_5_23_8_7_23_full_res_ALL/'

    
    # f1= '/Users/c1341133/Desktop/Subi_jwst_pipeline/spectrum/grating__division3_order8_nrs1_1Dspec_box_extract_5_1_24_mean_bkg6_1_24_box_definitive'
    
    f1 = "/Users/c1341133/Desktop/Subi_jwst_pipeline/spectrum/grating__division3_order8_nrs1_1Dspec_box_extract_mean_bkg_14_1_24_baseline/",
    
    
    aa= ['%s/spectrum_wav.npy'%(f1), '%s/spectrum_exp_sampler_results.npy'%(f1), '%s/spectrum_sampler_results.npy'%(f1), '%s/spectrum_params.npy'%(f1)]
    
    wav1 = np.load(aa[0])
    sampler_results1 = np.load(aa[2])
    exp_sampler_results1 = np.load(aa[1])
    params1 = np.load(aa[3])
    
    idx = np.argsort(wav1)
    wav1 = wav1[idx]; sampler_results1=sampler_results1[idx]; exp_sampler_results1=exp_sampler_results1[idx]

    dic1={}
    for i in range(params1.shape[1]):
        param = params1[1][i]
        median_list = []
        pc16_list =[]
        pc84_list =[]
        for j in range(exp_sampler_results1.shape[0]):
    
            pc16 = exp_sampler_results1[j][0][i]
            median = exp_sampler_results1[j][1][i]
            pc84 = exp_sampler_results1[j][2][i]
            
            median_list.append(median)
            pc16_list.append(pc16)
            pc84_list.append(pc84)
        dic1[param]={}
        dic1[param]['median'] = np.array(median_list)
        dic1[param]['pc16'] =  np.array(pc16_list)
        dic1[param]['pc84'] =  np.array(pc84_list)
        

    # f2= '/Users/c1341133/Desktop/Subi_jwst_pipeline/spectrum/grating__division3_order8_nrs2_1Dspec_box_extract_5_1_24_mean_bkg6_1_24_box_definitive'
    
    f2= "/Users/c1341133/Desktop/Subi_jwst_pipeline/spectrum/grating__division3_order8_nrs2_1Dspec_box_extract_mean_bkg_14_1_24_baseline/"

    aa= ['%s/spectrum_wav.npy'%(f2), '%s/spectrum_exp_sampler_results.npy'%(f2), '%s/spectrum_sampler_results.npy'%(f2), '%s/spectrum_params.npy'%(f2)]
    
    wav2 = np.load(aa[0])
    sampler_results2 = np.load(aa[2])
    exp_sampler_results2 = np.load(aa[1])
    params2 = np.load(aa[3])
   
    
    
    idx = np.argsort(wav2)
    wav2 = wav2[idx]; sampler_results2=sampler_results2[idx]; exp_sampler_results2=exp_sampler_results2[idx]
    
    idx  = np.argwhere(wav2>5.1).T[0][0]
    wav2=wav2[:idx]
    sampler_results2=sampler_results2[:idx]; exp_sampler_results2=exp_sampler_results2[:idx]

    dic2={}
    for i in range(params2.shape[1]):
        param = params2[1][i]
        median_list = []
        pc16_list =[]
        pc84_list =[]
        for j in range(exp_sampler_results2.shape[0]):
    
            pc16 = exp_sampler_results2[j][0][i]
            median = exp_sampler_results2[j][1][i]
            pc84 = exp_sampler_results2[j][2][i]
            
            median_list.append(median)
            pc16_list.append(pc16)
            pc84_list.append(pc84)
        dic2[param]={}
        dic2[param]['median'] = np.array(median_list)
        dic2[param]['pc16'] =  np.array(pc16_list)
        dic2[param]['pc84'] =  np.array(pc84_list)
 
    ct=no-1
    
    if no == 3:
        filename = '/Users/c1341133/Downloads/G395H_paper_data/3_TRANSMISSION_SPECTRA/transit-spectrum-W39b-G395H-10pix_reduction%s.xc'%(no)
    
    elif no == 12:
        filename = '/Users/c1341133/Downloads/G395H_paper_data/3_TRANSMISSION_SPECTRA/transit-spectrum-W39b-G395H-10pix_weighted-average.nc'


    else:
        filename = '/Users/c1341133/Downloads/G395H_paper_data/3_TRANSMISSION_SPECTRA/transit-spectrum-W39b-G395H-10pix_reduction%s.nc'%(no)
   

    x = 'depth'   
    median  =  dic1[x]['median'] /1e6
    pc_84 = dic1[x]['pc84'] /1e6
    pc_16 = dic1[x]['pc16'] /1e6
    plus = pc_84 - median
    minus = median - pc_16
    err= (minus, plus)
   

    plt.figure('test1')
    plt.errorbar(wav1, median, err)
    
    x = 'depth'   
    median  =  dic2[x]['median'] /1e6
    pc_84 = dic2[x]['pc84'] /1e6
    pc_16 = dic2[x]['pc16'] /1e6
    plus = pc_84 - median
    minus = median - pc_16
    err= (minus, plus)
   

    plt.figure('test1')
    plt.errorbar(wav2, median, err)
    
   
    f = netCDF4.Dataset(filename,'r')
    
    print ("")
    print (no, f)
     
 
    
    # print (f)
    # print ('xxxxx')
    # print (f.variables['central_wavelength'])
 
    wl = np.array(f.variables['central_wavelength'][:])
    p2 = np.array(f.variables['transit_depth'][:])
    if no ==11:
        err_pos = np.array(f.variables['transit_depth_error_pos'])
        err_neg = np.array(f.variables['transit_depth_error_neg'])
        err = (err_pos + err_neg)/2
        
    else:
        err = np.array(f.variables['transit_depth_error'][:])
        

   
    idx =  np.argwhere((wl>=2.87) & (wl <=5.14)).T[0]
    wl = wl[idx]
    # if no ==11:
    #     err_pos=err_pos[idx]*100
    #     err_neg = err_neg[idx]*100
    # else:
    #     err= err[idx]*100
        
    err= err[idx]*100
    p2=p2[idx]*100
    
    # plt.figure('Grating comparison %s'%(pipe[ct]), dpi=100, figsize=(16,9), )
    
    plt.figure('Grating comparison %s'%(pipe[ct]), dpi=100, figsize=(13,9), )

    
        
        
    gs = gridspec.GridSpec(3, 1, height_ratios=[3, 1,1],   left=0.17, bottom=0.1, right=0.99, top=0.97, wspace=None, hspace=0.28)
    ax0 = plt.subplot(gs[0])
    
    # plt.subplots_adjust(left=0.1, bottom=0.13, right=0.9, top=0.9, wspace=0, hspace=0)

     
    f.close()
 
    
    idx  =  np.argwhere(wl>=3.75).T[0][0]
    
    wl1 = wl[:idx]
    wl2 = wl[idx:]
    

    x = rebin2(dic1, wav1, wav2=wl1)
    spec = x[1]
    wav = x[0]
    err1 = x[2]
    err2 = x[3]
    
    upper = spec+err1
    lower = spec-err2
    spec2 = (upper+lower)/2
    err2 = (upper-lower)/ 2
    
    wav1, spec2_1, err2_1 = wav*1, spec2*1, err2*1
    
    plt.figure('Grating comparison %s'%(pipe[ct]))

    # plt.errorbar(wav, spec2*100 , err2*100, fmt='o', markeredgecolor='none',
    #          markerfacecolor= 'k', ecolor ='k', ls='none',
    #          capsize=0, elinewidth=2, label ='JexoPipe', alpha = 0.6)
    
    
    x = rebin2(dic2, wav2, wav2=wl2)
   
    spec = x[1]
    wav = x[0]
    err1 = x[2]
    err2 = x[3]
    
    
    upper = spec+err1
    lower = spec-err2
    spec2 = (upper+lower)/2
    err2 = (upper-lower)/ 2
    
    
    plt.figure('Grating comparison %s'%(pipe[ct]))
    
    

    # plt.errorbar(wav, spec2*100 , err2*100, fmt='o', markeredgecolor='none',
    #          markerfacecolor= 'k', ecolor ='k', ls='none',
    #          capsize=0, elinewidth=2, alpha = 0.6)
    
    wav = np.hstack((wav1, wav))
    spec2 = np.hstack((spec2_1, spec2))
    err2 = np.hstack((err2_1, err2))
    
     
    idx =[]
    for i in range(len(wl)):
        if wl[i] in wav:
            idx.append(i)
    idx = np.array(idx)

          
   
    idx = np.delete(idx,0)
    wav, spec2, err2 = wav[1:], spec2[1:], err2[1:]
    
    if no == 11:
        idx = np.delete(idx,-2)
        wav, spec2, err2 = np.delete(wav,-2), np.delete(spec2,-2), np.delete(err2,-2)
    
    wl, p2, err = wl[idx], p2[idx], err[idx]


    plt.errorbar(wav, spec2*100 , err2*100, fmt='o', markeredgecolor='none',
            markerfacecolor= 'k', ecolor ='k', ls='none',
            capsize=0, elinewidth=2, label ='JexoPipe', alpha = 0.6)
  
 
    # if no ==11:
    #     plt.errorbar(wl, p2, yerr=(err_neg,err_pos), fmt='o', markeredgecolor='none',
    #                  markerfacecolor= col[ct], ecolor =col[ct], ls='none',
    #                  capsize=0, elinewidth=2, label =pipe[ct], alpha = 0.6)
    # else:
    plt.errorbar(wl, p2, err, fmt='o', markeredgecolor='none',
             markerfacecolor= col[ct], ecolor =col[ct], ls='none',
             capsize=0, elinewidth=2, label =pipe[ct], alpha = 0.6)
        
        
    plt.legend(loc='upper left')
    plt.ylim(1.95, 2.4)
    plt.xlim(2.8, 5.2)

    # plt.figure('no err')
    # plt.plot(wl, p2)
    

    # plt.xlabel('Wavelength ($\mu m$)')
    plt.ylabel('Transit depth (%)', labelpad=20)
    
    axes = plt.gca()
    ymin, ymax = axes.get_ylim()
    y = np.array([ymin, ymax])
    plt.fill_betweenx(y,  0.688, 1.909, alpha=0.1, color='0.1')
    
    ax= plt.gca()
    #set off Scientific notation from Y-axis
    ax.get_yaxis().get_major_formatter().set_useOffset(False)
    # ax.get_yaxis().get_major_formatter().set_scientific(False)
    ax.get_xaxis().get_major_formatter().set_useOffset(False)

     
    ax = plt.gca()
    legend = ax.legend(loc='upper left', shadow=False, fontsize=fontsize, frameon=False)
    ax.get_legend().get_title().set_fontsize('18')
    ax.get_legend().get_title().set_fontweight('bold')

    frame = legend.get_frame()
    frame.set_facecolor('0.90')
    for label in legend.get_texts():
        label.set_fontsize('medium')
        label.set_fontsize(fontsize2)
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

 


    
 
    
    ax1 = plt.subplot(gs[1])
    # res = p2/100-spec2
    
    res = spec2-p2/100
    
    print ('www', len(res), len(wl))
    
    res_err =    ((err/100)**2 + err2**2)**0.5
    plt.errorbar(wl, res*1e6, res_err*1e6, fmt='o', markeredgecolor='none',
                  markerfacecolor= 'k', ecolor ='k', ls='none',
                  capsize=0, elinewidth=2, alpha = 0.6)
    axes = plt.gca()
    ymin, ymax = axes.get_ylim()
    y = np.array([ymin, ymax])
 
    plt.plot([0,6],[0,0], ':', linewidth=2, color='k')
    plt.ylabel('Residuals\n(ppm)')
   
    max = res*1e6+res_err*1e6
    min = res*1e6-res_err*1e6
    outlier = []
    inlier = []
    for i in range(len(max)):
        if max[i]>=0 and min[i]<=0:
            inlier.append(i)
        else:
            outlier.append(i)       
    # print(inlier)
    # print(outlier)
    print ('=====')
    print ('res match %s'%(pipe[ct]),  len(inlier)/len(max))
    print ('=====')    
    
    res_mean = 1e6* np.mean(res)
    res_mean_err = 1e6* np.sqrt(np.mean(res_err**2)/ len(res_err))
    
    print ('res mean', res_mean, res_mean_err)
    print ('=====')   
  
     
    plt.xlim(2.8, 5.2)
    ax= plt.gca()
    #set off Scientific notation from Y-axis
    ax.get_yaxis().get_major_formatter().set_useOffset(False)
    # ax.get_yaxis().get_major_formatter().set_scientific(False)
    ax.get_xaxis().get_major_formatter().set_useOffset(False)

  
    ax = plt.gca()
    legend = ax.legend(loc='upper left', shadow=False, fontsize=fontsize, frameon=False)
    ax.get_legend().get_title().set_fontsize('18')
    ax.get_legend().get_title().set_fontweight('bold')

    frame = legend.get_frame()
    frame.set_facecolor('0.90')
    for label in legend.get_texts():
        label.set_fontsize('medium')
        label.set_fontsize(fontsize2)
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
    
  
    
    # aa = np.arange(0,  y.max()+400, 400)
    # bb = np.arange(0,  y.min()-400, -400)[::-1]
    
    # aa = np.arange(0,  y.max()+500, 500)
    # bb = np.arange(0,  y.min()-500, -500)[::-1]
    
    
    dd= 500
    if pipe[ct] =='Inglis' or pipe[ct]== 'Wakeford-Damiano':
        dd=1000
    
    aa = np.arange(0,  y.max()+500, dd)
    bb = np.arange(0,  y.min()-500, -dd)[::-1]
    cc  = np.hstack((bb[:-1], aa))
    
    
    
    # print (aa)
    # print (bb)
    # print (cc)
    
     
    plt.yticks(cc)

    # ax.xaxis.labelpad = 2
    # plt.xlabel('Wavelength (µm)')
    
    
    ax1 = plt.subplot(gs[2])
    # plt.figure('err', figsize=(19,3))
    
 
    
    plt.plot(wav, 1e6*err/100, '.', color= col[ct], linewidth =2)
    plt.plot(wav, 1e6* err2, 'k.', linewidth =2)
    plt.ylabel('Average\nerror\n(ppm)') 
    plt.xlabel('Wavelength (µm)')
    
    axes = plt.gca()
    ymin, ymax = axes.get_ylim()
    y = np.array([ymin, ymax])
    plt.fill_betweenx(y,  0.688, 1.909, alpha=0.1, color='0.1')
    
    
    jex_err = 1e6* err2
    alt_err  = 1e6*err/100
    diff_err = jex_err-alt_err
    
    print ('error difference  ', np.mean(diff_err ), np.std(diff_err ))
    print ('=====')    

     
    ax = plt.gca()
    
    #set off Scientific notation from Y-axis
    ax.get_yaxis().get_major_formatter().set_useOffset(False)
    # ax.get_yaxis().get_major_formatter().set_scientific(False)
    ax.get_xaxis().get_major_formatter().set_useOffset(False)
    
    dd= 100
    if pipe[ct] =='Wallack' or pipe[ct]== 'Wakeford-Damiano' or pipe[ct]=='Barat':
        dd=200
    
    aa = np.arange(0,  y.max()+100, dd)
    bb = np.arange(0,  y.min()-100, -dd)[::-1]
    cc  = np.hstack((bb[:-1], aa))
    
    
    
    # print (aa)
    # print (bb)
    # print (cc)
    
     
    plt.yticks(cc)
    
    
    ax = plt.gca()
    # legend = ax.legend(loc='best', shadow=False, fontsize=fontsize, frameon=False)
    # ax.get_legend().get_title().set_fontsize('18')
    # ax.get_legend().get_title().set_fontweight('bold')
    
    
    frame = legend.get_frame()
    frame.set_facecolor('0.90')
    for label in legend.get_texts():
        label.set_fontsize('medium')
        label.set_fontsize(fontsize2)
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
    
    plt.subplots_adjust(left=0.1, right=0.9, top=0.95, bottom=0.3)
    plt.xlim(2.8, 5.2)
    
    
    if no == 12:
        plt.savefig('/Users/c1341133/Pictures/Cobra_3_revision/Grating comparison %s'%('weighted mean'))
    else:
        plt.savefig('/Users/c1341133/Pictures/Cobra_3_revision/Grating comparison %s'%(pipe[ct]))

 
    
 
 
