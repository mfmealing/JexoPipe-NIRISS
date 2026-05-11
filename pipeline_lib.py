#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 22 21:31:32 2023

@author: user1
"""


import os
#set environmental variables
# os.environ={'TERM_PROGRAM': 'Apple_Terminal', 'SHELL': '/bin/bash', 'TERM': 'xterm-256color', 'TMPDIR': '/var/folders/pf/gnssyvt522z1t_8__nkt0b3h0000gn/T/', 'CONDA_SHLVL': '2', 'CONDA_PROMPT_MODIFIER': '(jwst) ', 'TERM_PROGRAM_VERSION': '433', 'TERM_SESSION_ID': '2B5D8DCE-4860-432F-9195-A01C7724EA43', 'CRDS_PATH': '/Users/user1/crds_cache', 'USER': 'user1', 'CONDA_EXE': '/opt/anaconda3/bin/conda', 'SSH_AUTH_SOCK': '/private/tmp/com.apple.launchd.lxWzo7h8iV/Listeners', '_CE_CONDA': '', 'CONDA_PREFIX_1': '/opt/anaconda3', 'PATH': '/opt/anaconda3/envs/jwst/bin:/opt/anaconda3/condabin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin:/opt/local/bin:/opt/X11/bin:/Library/Apple/usr/bin:/opt/local/bin', 'LaunchInstanceID': '525FDCF2-4C0B-4BD2-B781-01AAD16AC30F', 'CONDA_PREFIX': '/opt/anaconda3/envs/jwst', 'PWD': '/Users/user1', 'LANG': 'en_GB.UTF-8', 'XPC_FLAGS': '0x0', '_CE_M': '', 'XPC_SERVICE_NAME': '0', 'SHLVL': '1', 'HOME': '/Users/user1', 'CONDA_PYTHON_EXE': '/opt/anaconda3/bin/python', 'LOGNAME': 'user1', 'CRDS_SERVER_URL': 'https://jwst-crds.stsci.edu', 'CONDA_DEFAULT_ENV': 'jwst', 'DISPLAY': '/private/tmp/com.apple.launchd.PpDoQsg2xs/org.xquartz:0', 'SECURITYSESSIONID': '186a8', '_': '/opt/anaconda3/envs/jwst/bin/python', 'CRDS_FITS_IGNORE_MISSING_END': 'false', 'CRDS_FITS_VERIFY_CHECKSUM': 'true', 'CRDS_ADD_LOG_MSG_COUNTER': 'false', 'CRDS_ALLOW_BAD_REFERENCES': 'false', 'CRDS_ALLOW_BAD_RULES': 'false', 'PASS_INVALID_VALUES': 'false', 'CRDS_ALLOW_SCHEMA_VIOLATIONS': 'false', 'CRDS_ALLOW_BAD_PARKEY_VALUES': 'false', 'CRDS_ALLOW_BAD_USEAFTER': 'false', 'CRDS_USE_PICKLED_CONTEXTS': 'false', 'CRDS_AUTO_PICKLE_CONTEXTS': 'false', 'CRDS_FORCE_COMPLETE_LOAD': 'false', 'CRDS_EXPLICIT_GARBAGE_COLLECTION': 'true', 'CRDS_MODE': 'auto', 'CRDS_IGNORE_MAPPING_CHECKSUM': 'false', 'CRDS_DOWNLOAD_MODE': 'http', 'CRDS_CONFIG_URI': 'None', 'CRDS_MAPPING_URI': 'None', 'CRDS_REFERENCE_URI': 'None', 'CRDS_PICKLE_URI': 'None', 'CRDS_DOWNLOAD_CHECKSUMS': 'true', 'CRDS_DOWNLOAD_LENGTHS': 'true', 'CRDS_CLIENT_RETRY_COUNT': '1', 'CRDS_CLIENT_RETRY_DELAY_SECONDS': '0', 'CRDS_LOCK_PATH': '/tmp', 'CRDS_USE_LOCKING': 'true', 'CRDS_LOCKING_MODE': 'multiprocessing', 'CRDS_S3_ENABLED': 'false', 'CRDS_S3_RETURN_URI': 'false', 'CRDS_OBSERVATORY': 'None'}
#this must come BEFORE jwst is imported

# import asdf
import copy
import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
from scipy.integrate import cumulative_trapezoid as cumtrapz
from scipy import interpolate


# jwst imports 
import jwst
print(jwst.__version__)

# Individual steps that make up calwebb_detector1
from jwst.dq_init import DQInitStep
from jwst.saturation import SaturationStep

from jwst import datamodels
from jwst.group_scale import GroupScaleStep


output_dir = './output'
if not os.path.exists(output_dir ): 
    os.makedirs(output_dir )
    

from jwst.stpipe import Step 

def get_dq_stack(file_list, sp_factor):
    
    for uncal_file in file_list:
 
        result = uncal_file 
        
        step = GroupScaleStep()
        result = step.run(result)
 
        step = DQInitStep()
        result = step.run(result)
     
        step = SaturationStep()
        step.n_pix_grow_sat = sp_factor # important.... 
        result = step.run(result)
        
        dq = result.groupdq
        if uncal_file == file_list[0]:
            dq_stack = dq
        else:
            dq_stack = np.vstack((dq_stack,dq))
            
    return dq_stack

def get_custom_superbias(file_list, channel, nrs):
    
    for uncal_file in file_list:
 
        result = uncal_file 
        
        step = GroupScaleStep()
        result = step.run(result)
        
        step = DQInitStep()
        result = step.run(result)
        
        data = result.data[:,0,:,:]
        if uncal_file == file_list[0]:
            data_stack = data
        else:
            data_stack = np.vstack((data_stack,data))
            
    print (data_stack.shape)
    custom_superbias = np.median(data_stack, axis =0)
    plt.figure('sss')
    plt.imshow(custom_superbias)
    
    #0229 = prism
    #0405 = prism
    #0427 = nrs1
    #0429 =nrs2
    
    if channel == 'prism':  
        f = '/Users/c1341133/crds_cache/references/jwst/nirspec/jwst_nirspec_superbias_0299 copy.fits'
    if channel == 'grating':
        if nrs ==1 :
            f = '/Users/c1341133/crds_cache/references/jwst/nirspec/jwst_nirspec_superbias_0427 copy.fits'
        if nrs ==2 :
            f = '/Users/c1341133/crds_cache/references/jwst/nirspec/jwst_nirspec_superbias_0429 copy.fits'
    f2 = '/Users/c1341133/crds_cache/references/jwst/nirspec/jwst_nirspec_superbias_custom_%s_nrs_%s.fits'%(channel,nrs)
    hdul = fits.open(f)            
    hdul[1].data = custom_superbias
    hdul.writeto(f2, overwrite=True)
    hdul.close()
                     
    return f2




def fix_groups(dq_stack, result):
  
        # dq = result.groupdq  
        dq= copy.deepcopy(dq_stack)
        sat_dq = (dq & 2) > 0

        # data = np.load('/Users/user1/Desktop/prism_data.npy')
        # gpdq = np.load('/Users/user1/Desktop/prism_dq.npy')
        
        gpdq = result.groupdq
        data = result.data
         
        # dq_stack  = np.load('/Users/user1/Desktop/dq_stack_prism.npy')
        
        # A) find min number of unsat gps per intg (per pixel)
        sat_dq = (dq_stack & 2) > 0  # ones and zeros (boolean)
        unsat_gps = np.min(sat_dq.shape[1]- sat_dq.sum(axis=1), axis=0)
        plt.figure('min number of unsat gps in any intg')
        plt.imshow(unsat_gps)
        
        sat_dq_gponly = (gpdq & 2) > 0
        unsat_gps_gponly = np.min(sat_dq_gponly.shape[1]- sat_dq_gponly.sum(axis=1), axis=0)
        plt.figure('min number of unsat gps in any intg in this segment')
        plt.imshow(unsat_gps_gponly)
        
       
       
        
        
        
        
        
        # B) find pixels that have >10% of intgs with at least one saturated gp
        dq= sat_dq
        x_ = dq[:, dq.shape[1]-1,:,:] # picks off last group in each intg; one slice per intg
        satd_intgs = np.sum(np.where(x_>0,1,0), axis=0) # if sat flag in last group flag as 1 : means at least one satd grp in that intg
        plt.figure('perm sat : no of intgs with at least one sat flag') # number of integrations where at least one group (i.e. last) is saturated
        plt.imshow(satd_intgs)
        satd_intgs_10pc = np.where(satd_intgs >= 0.1* dq.shape[0], satd_intgs, 0)
        satd_pix_10pc = np.where(satd_intgs_10pc>0,1,0)
        plt.figure('perm sat 3 : permanently satd pixels ') # "permanently satd pixels"
        plt.imshow(satd_pix_10pc)
        
        # C) select out the perm sat region and the remainder and treat differently
        mask_perm_sat = satd_pix_10pc 
        mask_un_sat = np.where(mask_perm_sat==1,0,1)
        plt.figure('mask perm sat') 
        plt.imshow(mask_perm_sat)
        plt.figure('mask un sat') 
        plt.imshow(mask_un_sat)
        
       
        
        data_perm_sat = data*mask_perm_sat
        data_un_sat = data*mask_un_sat
        
        gpdq_perm_sat = gpdq*mask_perm_sat
        gpdq_un_sat = gpdq*mask_un_sat
        
        
        dq_perm_sat = dq_stack*mask_perm_sat
        dq_un_sat = dq_stack*mask_un_sat
        
        unsat_gps_perm_sat = unsat_gps*np.where(mask_perm_sat==1,1,np.nan)
        unsat_gps_un_sat = unsat_gps*np.where(mask_un_sat==1,1,np.nan)
        
        plt.figure('cccc')
        plt.imshow(unsat_gps_perm_sat)
        
        plt.figure('cccczz')
        plt.imshow(unsat_gps_un_sat)
        
      
        
        # C) apply group control to perm sat region
        
        new_method = 1
# =============================================================================
#         old method
# =============================================================================
        if new_method ==0:
        # for t in [0]:
        
            # Deal first with single unsat groups
            # if < 10% of timeline has single groups, we will remove these 
            # first find number of intgs per pixel where the 2nd grp is saturated
            
            # sat_2nd_gp_intgs_perm_sat = np.sum(np.where(dq_perm_sat[:, 1,:,:]>0,1,0), axis=0) # 2 d array 
            sat_2nd_gp_intgs_perm_sat = np.sum(np.where((dq_perm_sat[:, 1,:,:]&2)>0,1,0), axis=0) # 2 d array 
    
            
            plt.figure('no of 1 groups') # no of intgs per pixel with a 1 gp (assuming 1st gp is is unsat)
            plt.imshow(sat_2nd_gp_intgs_perm_sat)
            
            plt.figure('no of 1 groups row')
            plt.plot(sat_2nd_gp_intgs_perm_sat[16], 'o-')
            
            plt.figure('dq flags in 12 105')
            for i in range(dq_perm_sat.shape[0]):
                    x = np.arange(dq_perm_sat.shape[1])+dq_perm_sat.shape[1]*i
               
                    plt.plot(x,dq_perm_sat[i,:,12,105], 'o-')
            
            
            # where this is <10% of the timeline, make the unsat group such that group control to 1 gp is not applied and instead 2 gps is applied below
            unsat_gps_perm_sat = np.where((sat_2nd_gp_intgs_perm_sat/dq_perm_sat.shape[0] < 0.1) & (sat_2nd_gp_intgs_perm_sat>0) , 2, unsat_gps_perm_sat)
            plt.figure('gp min 2')
            plt.imshow(unsat_gps_perm_sat) 
            
        
            # must apply nans to the 2nd group in the data for all these pixels, so it will result in a nan ramp when attempts to fit
            # and must change the 2nd group flag in the gorup dq array to unsat so it will attempt to fit a ramp to 2 grps
            mask = np.where((sat_2nd_gp_intgs_perm_sat/dq_perm_sat.shape[0] < 0.1) & (sat_2nd_gp_intgs_perm_sat>0) , 1, 0) # a 2-d mask for  pixels that have < 10% 1 groups 
            
            # mask out all but those small no 1 gp pixels from the gp dq array
            x_ = mask*gpdq_perm_sat
            x_[:,2:,:,:]=0 # masks out groups > 2 in x_ giving them zeros
            # mask now ids the first 2 gps in pixels that have small no of 1 grps
            
            print (np.nansum(gpdq_perm_sat))
            
            #remove any sat flags from the first two gps of pixels ided using the mask so that ramp fit proceeds with 2 gps
            gpdq_perm_sat= np.where(x_>0, gpdq_perm_sat-2, gpdq_perm_sat)
            
            print (np.nansum(gpdq_perm_sat))
            # now use mask to apply nans to first two groups ensuring a nan in the ramp. 
            data_perm_sat = data_perm_sat*np.where(x_>0, np.nan, 1)
            
            
            gpdq_perm_sat_method0= gpdq_perm_sat*1

            
            # now apply group control
            # mask_in = np.where(ee==1,1 ,0)
            # mask_out = np.where(ee==1,0 ,1)
            
            
            # plt.figure('mask2') # masks in perm sat region
            # plt.imshow(mask_in)
            
            # plt.figure('mask1') # masks out perm sat region
            # plt.imshow(mask_out)
             
            # # now modify group dq
            
            # dq = gpdq
            # dq2 = dq*mask_in # in region 
            # dq1 = dq*mask_out # not in region  # dq1 not touched
        
        
# =============================================================================
#         new method - first deal with <10% 1 groups
# =============================================================================
        if new_method==1:
            

            
            sat_dq = (dq_perm_sat & 2) > 0  # ones and zeros (boolean)
            unsat_gps = (sat_dq.shape[1]- sat_dq.sum(axis=1))
            
            # plt.imshow(unsat_gps)
            onegroupint = np.where(unsat_gps==1,1,0)
            all_one = np.sum(onegroupint, axis=0)
            
            # plt.figure("1")
            # plt.imshow(all_one)
            
            mask = np.where( (all_one/dq_perm_sat.shape[0] < 0.1) & (all_one>0),1,0 )
            # mask = np.where((all_one/dq_perm_sat.shape[0] < 0.05) & (all_one>0),1,0 )

            
            #ADDED ONLY NEEDED IN PERM SAT REGION FOR GP CONTROL; ALL INTS (for pix with <0.1 1 gp) CONTROLLED TO 2 GPS
            unsat_gps_perm_sat = np.where(mask==1 , 2, unsat_gps_perm_sat)

            
            # plt.figure("2")
            # plt.imshow(mask)
            
            sat_dq_gp = (gpdq_perm_sat & 2) > 0
            unsat_gps_gp = (gpdq_perm_sat.shape[1]- sat_dq_gp.sum(axis=1))
            onegroupint_gp = np.where(unsat_gps_gp==1,1,0)
            aa = onegroupint_gp* mask
            
            # plt.figure("2")
            # plt.imshow(aa.sum(axis=0))
            
            aa = np.expand_dims(aa, axis=1)
            aa = aa.repeat(gpdq_perm_sat.shape[1],axis=1)
    
            
            idx=[]
            for i in range(aa.shape[0]):
                # print (aa[i,...].sum())
                if aa[i,...].sum()>0:
                    idx.append(i)
            print (idx)
            # we add a 1 (bitwise) to the dq array for all groups in the affected intg and nan the whole group in the data
            gpdq_perm_sat = np.where(aa==1, gpdq_perm_sat|1, gpdq_perm_sat)
            data_perm_sat = np.where(aa==1, np.nan, data_perm_sat)
            
           
        

# =============================================================================
#         
# =============================================================================
           
        # # remove all saturation flags in the region
        dq_no_sat0 = np.where((gpdq_perm_sat & 2) > 0, gpdq_perm_sat-2 , gpdq_perm_sat)
        
        # add back but only so that min no oh unsat groups are flagged 
        aa =  np.zeros((gpdq_perm_sat.shape[1], gpdq_perm_sat.shape[2], gpdq_perm_sat.shape[3]))
        for i in range(aa.shape[0]):
            bb = np.zeros_like(aa[i])
            bb = np.where(unsat_gps_perm_sat <= i, 2, bb)
            aa[i] = bb
        gpdq_perm_sat  = dq_no_sat0 +aa
        

        # =============================================================================
        # old method for unsat region
        # =============================================================================  
        # # D) un sat region
        # # no group control
        # # we must however manage any stray single groups in same way as above, otherwise will be read and lead to noise
   
        # # sat_2nd_gp_intgs_un_sat = np.sum(np.where(dq_un_sat[:, 1,:,:]>0,1,0), axis=0)        
        # sat_2nd_gp_intgs_un_sat = np.sum(np.where((dq_un_sat[:, 1,:,:]&2)>0,1,0), axis=0) # 2 d array        
        
        # plt.figure('no of 1 groups (un sat') # no of intgs per pixel with a 1 gp (assuming 1st gp is is unsat)
        # plt.imshow(sat_2nd_gp_intgs_un_sat, aspect='auto')

         
        # # # where this is <10% of the timeline, make the unsat group such that it thinks it needs to fit minumum of 2 groups
        # # unsat_gps_perm_sat = np.where((sat_2nd_gp_intgs_perm_sat/dq_perm_sat.shape[0] < 0.1) & (sat_2nd_gp_intgs_perm_sat>0) , 2, unsat_gps_perm_sat)
        # # plt.figure('gp min 2')
        # # plt.imshow(unsat_gps_perm_sat) 
        # # must apply nans to the 2nd group in the data for all these pixels, so it will result in a nan ramp when attempts to fit
        # # and must change the 2nd group flag in the gorup dq array to unsat so it will attempt to fit a ramp to 2 grps
        # mask = np.where(sat_2nd_gp_intgs_un_sat>0 , 1, 0) # a mask for  pixels that have < 10% 1 groups 
        
        # plt.figure('mask xx') # no of intgs per pixel with a 1 gp (assuming 1st gp is is unsat)
        # plt.imshow(mask, aspect='auto')
        
        
        
        # x_ = mask*gpdq_un_sat
        # x_[:,2:,:,:]=0 # masks out groups > 2 in ww dq array giving them zeros
        
        # gpdq_un_sat= np.where(x_>0, gpdq_un_sat-2, gpdq_un_sat)
        # data_un_sat = data_un_sat*np.where(x_>0, np.nan, 1)
        # # xxxx
        
        # # print ('gggg')
        # # print (np.nansum(data_un_sat[:,:,:,200]))
        # # print (np.nansum(gpdq_un_sat[:,:,:,200]))
        
        # # xxx
 
         
# =============================================================================
#   new method for unsat region
# =============================================================================
 
        sat_dq = (dq_un_sat & 2) > 0  # ones and zeros (boolean)
        unsat_gps = (sat_dq.shape[1]- sat_dq.sum(axis=1))
        
        # plt.imshow(unsat_gps)
        onegroupint = np.where(unsat_gps==1,1,0)
        all_one = np.sum(onegroupint, axis=0)
        
        # plt.figure("1")
        # plt.imshow(all_one)
        
        mask = np.where( (all_one/dq_un_sat.shape[0] < 0.1) & (all_one>0),1,0 )
        # mask = np.where( (all_one/dq_un_sat.shape[0] < 0.05) & (all_one>0),1,0 )

        
        # plt.figure("2")
        # plt.imshow(mask)
        
        sat_dq_gp = (gpdq_un_sat & 2) > 0
        unsat_gps_gp = (gpdq_un_sat.shape[1]- sat_dq_gp.sum(axis=1))
        onegroupint_gp = np.where(unsat_gps_gp==1,1,0)
        aa = onegroupint_gp* mask
        
        # plt.figure("2")
        # plt.imshow(aa.sum(axis=0))
        
        aa = np.expand_dims(aa, axis=1)
        aa = aa.repeat(gpdq_un_sat.shape[1],axis=1)

        
        idx=[]
        for i in range(aa.shape[0]):
            # print (aa[i,...].sum())
            if aa[i,...].sum()>0:
                idx.append(i)
        print (idx)
        gpdq_un_sat = np.where(aa==1, gpdq_un_sat|1, gpdq_un_sat)
        data_un_sat = np.where(aa==1, np.nan, data_un_sat)


        # =============================================================================
        # add things back together
        # =============================================================================
        data = data_un_sat + data_perm_sat
        gpdq = gpdq_un_sat + gpdq_perm_sat
        
         
        return data, gpdq
                
    

def fix_groups_new(super_dq, result):
    
    # try to explain better...
     
        section_dq  = result.groupdq
        data = result.data
         
        
        # A)  min_gps: 2-d array: min number of (unsat) gps (per intg) (per pixel) over whole pixel timeline.  
        # The no of gps will in principle be fixed to this
        super_dq_sat = (super_dq & 2) > 0  #4-d array; gives 1 to satd gps and 0 to unsatd gps
        
        super_no_gps_per_intg  = super_dq_sat.shape[1]- super_dq_sat.sum(axis=1) #3-d array of gps per integration per pixel
        
        min_gps = np.min(super_dq_sat.shape[1]- super_dq_sat.sum(axis=1), axis=0 )  #2-d array of min no of gps per pixel
        
        # plt.figure('min number of unsat gps in any intg')
        # plt.imshow(min_gps)
        
       
        # s = super_no_gps_per_intg[:,6,128]
        # plt.figure('kkk')
        # plt.plot(s,'o-')
        
        
# =============================================================================
#         determine the persistently saturated region
# =============================================================================
        
        # B) find pixels that have >10% of intgs with at least one saturated gp
      
        last_gp_dq_sat = super_dq_sat[:, super_dq_sat.shape[1]-1,:,:] # 3-d array, picks off last group in each intg; one slice per intg
        
        
        no_satd_intgs_per_pixel = np.sum(np.where(last_gp_dq_sat>0,1,0), axis=0) # 2-day array of no of intgs per pixel
        mask_in = np.where(no_satd_intgs_per_pixel >= 0.1* super_dq_sat.shape[0], 1, 0) #2-day array of pixels with >10% satd intgs: masks the perm sat region
        mask_out = np.where(mask_in==1,0,1)
        
        

        # plt.figure('no satd ingts per pixel')
        # plt.imshow(no_satd_intgs_per_pixel)
        # plt.figure('mask in') 
        # plt.imshow(mask_in)
        # plt.figure('mask out') 
        # plt.imshow(mask_out)
        
   
        
        
# =============================================================================
#         For ease we divde the data and dq arrrays into two regions
# =============================================================================
        data_1 = data*mask_in
        data_2 = data*mask_out
        
        section_dq_1 = section_dq*mask_in
        section_dq_2 = section_dq*mask_out

        
        
        section_dq_sat = (section_dq & 2) > 0  #4-d array; gives 1 to satd gps and 0 to unsatd gps
        section_dq_sat_1 = section_dq_sat * mask_in
        section_dq_sat_2 = section_dq_sat * mask_out
        
        section_no_gps_per_intg = section_dq_sat.shape[1]- section_dq_sat.sum(axis=1)
         
         
        
# =============================================================================
#       deal with single group integrations  
# =============================================================================
        # 1. if a pixel timeline has < 10% single groups, then do not make whole timeline single groups
        # instead: for intgs in timeline that have 2 groups, make no change.
        # for ings in timeline with a single group, make the whole ramp nan, so that after ramp fit,
        # the intg is returned as a nan and can be "filled in" later.
        
        # A. ID pixels with 1 gp intgs in a mask
        single_gp_intg_mask = np.where(super_no_gps_per_intg==1, 1, 0) #3-d array
        
        
        # print (single_gp_intg_mask.shape)
        
        plt.figure('sum')
        plt.imshow(single_gp_intg_mask.sum(axis=0))
        
        
        
        # plt.figure('qqqq')
        # plt.imshow(super_no_gps_per_intg[11] )
        
        # plt.figure('qqq')
        # plt.imshow(single_gp_intg_mask[11] )
        
       
        # B. find those pixels where <10% 1 gps occur
        pixels_w_small_no_1_gps = np.where((single_gp_intg_mask.sum(axis=0)/super_dq.shape[0]< 0.1)&(single_gp_intg_mask.sum(axis=0)>0), 1,0) 
        #2-d array
        
        plt.figure('qqqqqqq')
        plt.imshow(pixels_w_small_no_1_gps, vmin =0, vmax=1 ) 
        
        
        
        # plt.figure('qqqqqqqaaa')
        # plt.imshow(single_gp_intg_mask.sum(axis=0)/data.shape[0])
        
         
        # plt.figure('qqqqqqqwwwsss')
        # plt.imshow(single_gp_intg_mask.sum(axis=0), vmin=0,vmax=1)  
    
        single_gp_intg_mask =  single_gp_intg_mask * pixels_w_small_no_1_gps #3-d mask Iding all 1 gp intgs where they take up <10% of pixel timeline
        
        # plt.figure('qqqqqqqwww')
        # plt.imshow(single_gp_intg_mask.sum(axis=0), vmin=0,vmax=1)  
        
        #c. having found pixels_w_small_no_1_gps, over whole timeline,can now apply that mask to the current section
        
        
        section_single_gp_intg_mask = np.where(section_no_gps_per_intg==1, 1, 0) #3-d array
        
        # plt.figure('qqqqqqqwwwsss-sec')
        # plt.imshow(section_single_gp_intg_mask.sum(axis=0), vmin=0,vmax=1)  
        
        section_single_gp_intg_mask =  section_single_gp_intg_mask * pixels_w_small_no_1_gps #3-d mask Iding all 1 gp intgs where they take up <10% of pixel timeline


        # plt.figure('qqqqqqqwww-sec')
        # plt.imshow(section_single_gp_intg_mask.sum(axis=0), vmin=0,vmax=1) 
        
      
        #at end use idx of mask to add sat flags to final dq
# =============================================================================
#         
# =============================================================================
        # where there are small numbers of single groups fixe min gps to 2
        
        # plt.figure('min_gps raw')
        # plt.imshow(min_gps)   
        
        
        min_gps = np.where(pixels_w_small_no_1_gps==1, 2, min_gps)
        
        # plt.figure('min_gps modified')
        # plt.imshow(min_gps)   
        
        
        min_gps_1 = min_gps*np.where(mask_in==1,1,np.nan)  #2-d array
        min_gps_2 = min_gps*np.where(mask_out==1,1,np.nan)
        # nans needed at the group control stage to exclude the rest of the array
        
        # plt.figure('min_gps 1')
        # plt.imshow(min_gps_1)   
        
        
        # plt.figure('min_gps 2')
        # plt.imshow(min_gps_2)  
        
         
        
# =============================================================================
#       now group control using min_gps
# =============================================================================      
          
         
      
         

        # # remove all saturation flags in the region
        dq_no_sat0 = np.where((section_dq_1 & 2) > 0, section_dq_1-2 , section_dq_1)
        
        # add back but only so that min no oh unsat groups are flagged 
        aa =  np.zeros((section_dq_1.shape[1], section_dq_1.shape[2], section_dq_1.shape[3]))
        for i in range(aa.shape[0]):
            bb = np.zeros_like(aa[i]) #2-d array
            bb = np.where(min_gps_1 <= i, 2, bb)
            aa[i] = bb
        section_dq_1  = dq_no_sat0 +aa
        
                  
        # idx =1975
            
        # plt.figure('aaaa')
        # plt.imshow((section_dq_2[idx][1]))
            
        # plt.figure('aaab')
        # plt.imshow((section_dq_1[idx][1]))
            
            
        section_dq  = section_dq_2 + section_dq_1
            
        # plt.figure('aaac')
        # plt.imshow((section_dq[idx][1]),vmin= 0, vmax=2)
        
     
        
        #what happens to the pix with min gp zero?? they are not gp controlled so okay
      
  

        section_single_gp_intg_mask0 = np.zeros_like(section_dq)
        for i in range(section_single_gp_intg_mask0.shape[0]):
            for j in range(section_single_gp_intg_mask0.shape[1]):
                section_single_gp_intg_mask0[i][j] = section_single_gp_intg_mask[i]
                
                
        # finally in small single group intgs override and make whole ramp satd to induce a nan later
        section_dq  = np.where(section_single_gp_intg_mask0==1, 2, section_dq )
        

        # plt.figure('ssss')
        # plt.imshow((section_single_gp_intg_mask[idx]),vmin= 0, vmax=1)
        
        # s=[]
        # for i in range(section_single_gp_intg_mask.shape[0]):
        #     s.append(section_single_gp_intg_mask[i].sum())
        # plt.figure('pppppp')
        # plt.plot(s)
        
     
        
        # plt.figure('aaad')
        # plt.imshow((section_dq[idx][1]),vmin= 0, vmax=2)
        
        return data, section_dq
        
        
      
                

class CustomSuperBiasStep(Step):
    
    class_alias = "custom_superbias"

    spec = """ """
    # reference_file_types = ['superbias']

 
    def process(self, input):

        # Open the input data model
        with datamodels.RampModel(input) as input_model:
            
            result = self.custom_superbias(input_model)
            input_model.close()
            result.meta.cal_step.custom_bkg = 'COMPLETE'

        return result
 
    
    def custom_superbias(self, input_model):
        
         bm = input_model
         data = bm.data
    
         img_stack = np.zeros((data.shape[0],data.shape[2],data.shape[3]))
         for i in range(data.shape[0]):
             intg = data[i]
             img = intg[0]
             img_stack[i] = img
         
         gp1 = np.median(img_stack, axis =0)
         
         
         plt.figure('test 1a')
         plt.imshow(data[0][0], aspect='auto')
         
         data = data-gp1
         
         plt.figure('gp1 image median')
         plt.imshow(gp1, aspect='auto')
         
         
         plt.figure('test 1b')
         plt.imshow(data[0][0], aspect='auto')
          
         bm.data = data   
         return bm
         
     

   

 
class CustomBkgStage1_grating(Step):
     
    class_alias = "custom_bkg"

    spec = """ """
    # reference_file_types = ['superbias']

 
    def process(self, input, nrs, bkg_type):

        # Open the input data model
        with datamodels.RampModel(input) as input_model:
            
            result = self.custom_bkg_subtraction(input_model, nrs, bkg_type)
            input_model.close()
            result.meta.cal_step.custom_bkg = 'COMPLETE'

        return result
    
    
    def custom_bkg_subtraction(self, input_model, nrs, bkg_type):
        
        bm = input_model
        data = bm.data
        bkg_av= []
        
        
        if nrs == 1:
            cond = 1
        else:
            cond = 2
        
        x = data.shape[1]
        sup_img = np.nanmedian(data[:,x-1,:,:], axis=0)
        
        
        plt.figure('psf')
        plt.plot(sup_img[:,1996])
       

        
        # sup_img = data.sum(axis=1)
        # sup_img = np.nanmedian(sup_img,axis=0)
               
        plt.figure('sup img')
        plt.imshow(sup_img, aspect='auto', vmin =0, vmax = np.nanpercentile(sup_img, 99)) 
        

        max_list =[]
        x_list =[]
        
        if cond == 1:
            start =500
        else: 
            start = 5
        binsize=20
        for i in range(start,sup_img.shape[1]-binsize-5, binsize):
            slice = sup_img[:,i:i+binsize]
            slice = np.median(slice,axis=1)
            idx = np.nanargmax(slice)
            for j in range(binsize):
                max_list.append(idx)
        x_list = np.arange(start, start + len(max_list))
        plt.plot(x_list, max_list, 'ko')    
         
        
        
        # for i in range(1,len(max_list)):
        #     if max_list[i] > max_list[i-1]+1 or max_list[i] < max_list[i-1]-1:
        #         max_list[i] = max_list[i-1]
        # plt.plot(x_list, max_list, 'yo')
        
       
        r = 4
        
        max_list = np.array(max_list)
        x_list = np.array(x_list)
        z = np.polyfit(x_list,max_list, r)
       
        x = np.arange(sup_img.shape[1])
 
        y =0
        for i in range (0,r+1):
            y = y + z[i]*x**(r-i) 
        
        plt.plot(x, y, '-', color='g', linewidth=2) 
        
       
        max_list = (np.round(y,0)).astype(int)
        plt.figure('sup img 2')
        plt.imshow(sup_img, aspect='auto', vmin =0, vmax = np.nanpercentile(sup_img, 99)) 
        plt.plot(x, max_list, 'o-', color='y' ) 
        
       
        buffer = 10
        sup_buffer = np.zeros((sup_img.shape[0]+2*buffer, sup_img.shape[1]))
        sup_buffer[buffer:-buffer] =  sup_img
        sup_img =sup_buffer
        max_list = max_list + buffer
        
   
        plt.figure('sup img 3')
        plt.imshow(sup_img, aspect='auto', vmin =0, vmax = np.nanpercentile(sup_img, 99)) 
        
        mask = np.ones_like(sup_img)
        bbox=10
        # bbox=8
       
        ct=-1
        for i in range(5,mask.shape[1]-5):
            ct+=1
            mask[:,i][max_list[ct]-bbox: max_list[ct]+bbox+1] = np.nan
            
        # for i in range(5):
        #     mask[:,i] =  mask[:,5]
        #     mask[:,-i] =  mask[:,-6]
            
       
        
        plt.figure('sup + mask bb')
        plt.imshow(mask*sup_img,  aspect='auto',  vmin =0, vmax = np.nanpercentile((sup_img*mask), 99)) 
        
        sup_img_masked = (mask*sup_img)[buffer:-buffer]

        plt.figure('sup masked')
        plt.imshow(sup_img_masked,  aspect='auto',  vmin =0, vmax = np.nanpercentile(sup_img_masked, 99))
        
        plt.figure('mask 0')
        plt.imshow(mask,  aspect='auto',  vmin =0, vmax = np.nanpercentile(sup_img_masked, 99))
        
        
# =============================================================================
#         
# =============================================================================
        data_masked = data*1        


        # plt.figure('bkg test 1')
        # plt.imshow(data_masked[100][2])
           
        gpdq = bm.groupdq
        pixdq = bm.pixeldq
        
        gpdq_mask  =  np.where(gpdq >0, np.nan, 1)
        pixdq_mask  =  np.where(pixdq>0, np.nan, 1)
        
        data_masked*=gpdq_mask
        
        plt.figure('bkg test 2')
        plt.imshow(data_masked[50][1], aspect='auto')
        
        data_masked*=pixdq_mask
        
        plt.figure('bpix mask')
        plt.imshow(pixdq_mask)
        
        
        plt.figure('bkg test 3')
        plt.imshow(data_masked[50][1], aspect='auto')
        
    
       
        for intg in range(data.shape[0]):
        # for intg in [50]:

     
       
           for gp in range(data.shape[1]):
               
                img = data_masked[intg][gp]
               
               
                plt.figure('img')
                plt.imshow(img, aspect='auto', vmin =0, vmax = np.nanpercentile(img, 99))   
                
                plt.figure('img2')
                plt.imshow(data[intg][gp], aspect='auto')   
                
            
                
                img_buffer  = np.zeros((img.shape[0]+2*buffer, img.shape[1]))
                img_buffer[buffer:-buffer] =  img
                
                # plt.figure('mask')
                # plt.imshow(mask, aspect='auto')
                
                # plt.figure('img_buffer')
                # plt.imshow(img_buffer , aspect='auto')
                
                img_masked = img_buffer*mask
                
                # plt.figure('img masked')
                # plt.imshow(img_masked , aspect='auto')
                
                img_masked = img_masked[buffer:-buffer]
                
                plt.figure('img masked 2')
                plt.imshow(img_masked , aspect='auto')
              
                
                # =============================================================================
                #     filter as per referree
                # =============================================================================
            
                # bkg_median = np.nanmedian(img_masked, axis=0)
                
                # plt.figure('bkg median')
                # plt.plot(bkg_median, 'b-')
                
                # bkg_pc16 = np.nanpercentile(img_masked, 16, axis=0)
                # bkg_pc84 = np.nanpercentile(img_masked, 84, axis=0)
                
                # bkg_median_stack = mask[buffer:-buffer]* np.tile(bkg_median, (img_masked.shape[0], 1))
                # bkg_pc16_stack = mask[buffer:-buffer]* np.tile(bkg_pc16, (img_masked.shape[0], 1))
                # bkg_pc84_stack = mask[buffer:-buffer]* np.tile(bkg_pc84, (img_masked.shape[0], 1))
                # bkg_sigma = (bkg_pc84_stack - bkg_pc16_stack )/2
                
                
                # plt.figure('bkg median_stack')
                # plt.imshow(bkg_median_stack, aspect='auto')
                # plt.figure('bkg sigma_stack')
                # plt.imshow(bkg_sigma, aspect='auto')
                
               
                # alpha = 5
                # img_masked = np.where(img_masked> bkg_median_stack + bkg_sigma*alpha, np.nan, img_masked) 
                # img_masked = np.where(img_masked< bkg_median_stack - bkg_sigma*alpha, np.nan, img_masked) 
                
                # =============================================================================

                img_masked_0 = img_masked*1
                
                # import time
                # aa = time.time()


                if bkg_type == 'median':
                    
                    bkg0 = np.nanmean(img_masked, axis=0)
                    alpha = 5
                    
                    median_2d = np.nanmedian(img_masked)
                    pc_16 =  np.nanpercentile(img_masked, 16)
                    pc_84 =  np.nanpercentile(img_masked, 84)
                    sigma = (pc_84-pc_16 ) /2
                    img_masked = np.where(img_masked > median_2d + alpha*sigma, np.nan, img_masked)
                    img_masked = np.where(img_masked < median_2d - alpha*sigma, np.nan, img_masked)
                    
                  
                    bkg =  np.nanmean(img_masked, axis=0)
                    
                
                if bkg_type == 'mean':

                    bkg0 = np.nanmean(img_masked, axis=0)
                    
                    use_median_for_outliers =1
                    
                    alpha = 10
                    for i in range(3):
                        bkg_std = np.nanstd(img_masked, axis=0)
                        
                        if use_median_for_outliers == 0:
                            
                            bkg_mean = np.nanmean(img_masked, axis=0)
                        
                            bkg_mean_stack = mask[buffer:-buffer] * np.tile(bkg_mean, (img_masked.shape[0], 1))
                            bkg_std_stack = mask[buffer:-buffer]* np.tile(std, (img_masked.shape[0], 1))
                            
                            img_masked = np.where(img_masked> bkg_mean_stack + bkg_std_stack*alpha, np.nan, img_masked) 
                            img_masked = np.where(img_masked< bkg_mean_stack -  bkg_std_stack*alpha, np.nan, img_masked) 
                            
                        else:
                            bkg_median = np.nanmedian(img_masked, axis=0)
                            
                            bkg_pc16 = np.nanpercentile(img_masked, 16, axis=0)
                            bkg_pc84 = np.nanpercentile(img_masked, 84, axis=0)
                            
                            bkg_median_stack = mask[buffer:-buffer] * np.tile(bkg_median, (img_masked.shape[0], 1))
                            bkg_pc16_stack = mask[buffer:-buffer]* np.tile(bkg_pc16, (img_masked.shape[0], 1))
                            bkg_pc84_stack = mask[buffer:-buffer]* np.tile(bkg_pc84, (img_masked.shape[0], 1))
                            bkg_sigma = (bkg_pc84_stack - bkg_pc16_stack )/2

                            
                            img_masked = np.where(img_masked> bkg_median_stack + bkg_sigma*alpha, np.nan, img_masked) 
                            img_masked = np.where(img_masked< bkg_median_stack - bkg_sigma*alpha, np.nan, img_masked) 
                            
                            # img_masked = np.where(img_masked> bkg_pc84_stack*alpha, np.nan, img_masked) 
                            # img_masked = np.where(img_masked< bkg_pc16_stack*alpha, np.nan, img_masked) 
                        
                    # bkg = np.nanmean(img_masked, axis=0)    
                    # bkg2 = bkg*1
        
                    # alpha2 = 3
                 
                    # for iter in range(3):
                    #     box = 10; bbox = np.ones(box)/box
                    #     bkg_smoothed = np.convolve(bkg, bbox, 'same')
                    #     bkg_smoothed[:box] = np.nanmean(bkg[:box])
                    #     bkg_smoothed[-box:] = np.nanmean(bkg[-box:])
                    #     bkg_std = np.nanstd(bkg)
                    #     if nrs ==1:
                    #         bkg_std = np.nanstd(bkg[1000:1500])
                    #     # if nrs ==2:
                    #     #     bkg_std = np.nanstd(bkg[1000:1500])
                    
                    #     # print (bkg_std)
                    #     # bkg = np.where(bkg> bkg_smoothed + alpha2*bkg_std, bkg_smoothed, bkg)
                    #     # bkg = np.where(bkg< bkg_smoothed - alpha2*bkg_std, bkg_smoothed, bkg)
                    #     x_ = np.arange(len(bkg))
                    #     idx1  =  np.argwhere(bkg> bkg_smoothed + alpha2*bkg_std).T[0]
                    #     idx2  =  np.argwhere(bkg< bkg_smoothed - alpha2*bkg_std).T[0]
                    #     idx  = np.hstack((idx1,idx2))
                        
                    #     bkg = np.delete(bkg, idx)
                    #     x0_ = np.delete(x_, idx)
                    #     bkg = np.interp(x_, x0_, bkg)
                    
                    # bkg_smoothed = np.convolve(bkg, bbox, 'same')
                    # bkg_smoothed[:box] = np.nanmean(bkg[:box])
                    # bkg_smoothed[-box:] = np.nanmean(bkg[-box:])
                    # bkg_std = np.nanstd(bkg) 
                    
                    # # add back points that are in the right region (some inappropriate taken out due to smoothed func)
                    # idx  =  np.argwhere((bkg2 <= bkg_smoothed + alpha2*bkg_std)  & (bkg2 >= bkg_smoothed - alpha2*bkg_std) ).T[0]
  
                    # bkg[idx] = bkg2[idx]
                    
                 
                        
                     
                   
    
                if intg == 0 and gp ==0:
                    
                # if np.sum(bkg0[-80:]) != np.sum(bkg[-80:]):
                #     pass

                    
                    print ('filtered...', intg, gp, np.nansum(img_masked_0), np.nansum(img_masked)  )
                    plt.figure('filter')
                    plt.plot(bkg0, 'k-')
                    # plt.plot(bkg1, 'r-')
                    plt.plot(bkg, 'g-')
                    
                    plt.figure('filter_a')
         
                    plt.plot(bkg, 'g-')
                    
                  
                    
                    # print (time.time()-aa)
                    
                     

                    
                    # plt.figure('filter2')
                    # plt.plot(img_masked[:,1200])
                    
                    # # plt.figure('filter3')
                    # # plt.plot(img_masked[:,1883])
                    
                    # plt.figure('filter3')
                    # plt.plot(img_masked_0[:,1535],'o-')
                    # plt.plot(img_masked[:,1535], 'o-')
                    
                    # plt.figure('filter3a')
                    # plt.plot(img_masked_0[:,352],'o-')
                    # plt.plot(img_masked[:,352], 'o-')
                    
                    
                    # print (img_masked_0[:,-1])
                    
                    
                    # # plt.figure('filter3')
                    # # plt.plot(img_masked[:,352])
                    
                    
                    # plt.figure('filter4')
                    # plt.plot(bkg, 'g-')
                    
                    # plt.figure('filter5')
                    # plt.plot(bkg0[-80:], 'ko-')
                    # # plt.plot(bkg1, 'r-')
                    # plt.plot(bkg[-80:], 'go-')
                    
                    # plt.figure('filter6')
                    # plt.plot(bkg2, 'ro-')
                    # # plt.plot(bkg1, 'r-')
                    # plt.plot(bkg, 'go-')
                    
                    
                    # plt.figure('pppp')
                    # plt.plot(bkg_smoothed + alpha2*bkg_std, 'r--')
                    # plt.plot(bkg_smoothed - alpha2*bkg_std, 'r--')
                    # plt.plot(bkg2, 'b-')
                    
                 
                    # plt.plot(bkg2[idx], 'mo')
                 
         
                    # plt.figure('final')
                    # plt.plot(bkg2, 'ro-')
                    #   # plt.plot(bkg1, 'r-')
                    # plt.plot(bkg, 'go-')
                    
                    
 
                # =============================================================================
                # 
                # =============================================================================
                
                img0 = img - bkg0
            
                img = img - bkg
                
                # if gp == data.shape[1]-1:
                #     plt.figure('dddd')
                #     plt.plot(np.nansum(img0,axis=0), 'b.-')
                #     plt.plot(np.nansum(img,axis=0), 'r.-')
                    
                #     plt.figure('ddddddd')
                #     plt.plot(bkg0, 'b.-')
                #     plt.plot(bkg, 'r.-')
 
 
                data[intg][gp] = img
                
                # if gp==4:    
                #     plt.figure('img 4')
                #     # plt.imshow(img, aspect='auto', vmin =0, vmax = np.nanpercentile(img+bkg, 99))
                #     plt.imshow(img, aspect='auto', vmin =0, vmax = np.max(bkg_median))
          
        bm.data = data   
        return bm


 
class CustomBkgStage1_prism(Step):
    # SHOULD ERR array be changed?
     
    class_alias = "custom_bkg"

    spec = """ """
    # reference_file_types = ['superbias']

    def process(self, input, bkg_type):

        # Open the input data model
        with datamodels.RampModel(input) as input_model:
            
            result = self.custom_bkg_subtraction(input_model, bkg_type)
            input_model.close()
            result.meta.cal_step.custom_bkg = 'COMPLETE'

        return result
    
    def custom_bkg_subtraction(self, input_model, bkg_type):
        
        bm = input_model
        data = bm.data
        
         
        
        # sup_img = data.sum(axis=1)
        # sup_img = np.nanmedian(sup_img,axis=0)
        
        
        x = data.shape[1]
        sup_img = np.nanmedian(data[:,x-1,:,:], axis=0)

        
        plt.figure('sup img')
        plt.imshow(sup_img, aspect='auto', vmin =0, vmax = np.nanpercentile(sup_img, 99)) 
       
        
        y_profile = np.sum(sup_img, axis=1)
        y_max = np.argmax(y_profile)
        
        plt.figure()
        plt.plot(y_profile)
        
        mask = np.ones_like(sup_img)
        bbox=10
        
        
        mask[y_max-bbox: y_max+bbox+1] = np.nan
        
        #alternative
        mask[5:-5] = np.nan
 
     
        sup_img_masked = (mask*sup_img)

        plt.figure('sup masked')
        plt.imshow(sup_img_masked,  aspect='auto',  vmin =0, vmax = np.nanpercentile(sup_img_masked, 99))
    
    
        plt.figure('MASK')
        plt.imshow(mask,  aspect='auto')
        
        
        
# =============================================================================
#         
# =============================================================================
        data_masked = data*1        


        # plt.figure('bkg test 1')
        # plt.imshow(data_masked[100][2])
           
        gpdq = bm.groupdq
        pixdq = bm.pixeldq
        
        gpdq_mask  =  np.where(gpdq >0, np.nan, 1)
        pixdq_mask  =  np.where(pixdq>0, np.nan, 1)
        
        data_masked*=gpdq_mask
        
        # plt.figure('bkg test 2')
        # plt.imshow(data_masked[100][1])
        
        data_masked*=pixdq_mask
        
        plt.figure('bpix mask')
        plt.imshow(pixdq_mask)
        

        
        for intg in range(data.shape[0]):
        
            
            for gp in range(data.shape[1]):
                img = data_masked[intg][gp]
            
                
                img_masked = img*mask
                img_masked_0 = img_masked*1
                
                # plt.figure('img 1')
                # plt.imshow(img, aspect='auto')
                
                # plt.figure('img masked')
                # plt.imshow(img_masked, aspect='auto')
                
                # =============================================================================
                #     filter as per referree
                # =============================================================================
            
                if bkg_type == 'median':
                    
                    bkg0 = np.nanmean(img_masked, axis=0)
                    alpha = 5
                    
                    median_2d = np.nanmedian(img_masked)
                    pc_16 =  np.nanpercentile(img_masked, 16)
                    pc_84 =  np.nanpercentile(img_masked, 84)
                    sigma = (pc_84-pc_16 ) /2
                    img_masked = np.where(img_masked > median_2d + alpha*sigma, np.nan, img_masked)
                    img_masked = np.where(img_masked < median_2d - alpha*sigma, np.nan, img_masked)
                    
                     
  
                    bkg =  np.nanmean(img_masked, axis=0)
                    
                 
                if bkg_type == 'mean':

                    bkg0 = np.nanmean(img_masked, axis=0)
                    
                    use_median_for_outliers =1
                    
                    alpha = 10
                    for i in range(3):
                        bkg_std = np.nanstd(img_masked, axis=0)
                        
                        if use_median_for_outliers == 0:
                            
                            bkg_mean = np.nanmean(img_masked, axis=0)
                        
                            bkg_mean_stack = mask * np.tile(bkg_mean, (img_masked.shape[0], 1))
                            bkg_std_stack = mask* np.tile(std, (img_masked.shape[0], 1))
                            
                            img_masked = np.where(img_masked> bkg_mean_stack + bkg_std_stack*alpha, np.nan, img_masked) 
                            img_masked = np.where(img_masked< bkg_mean_stack -  bkg_std_stack*alpha, np.nan, img_masked) 
                            
                        else:
                            bkg_median = np.nanmedian(img_masked, axis=0)
                            
                            bkg_pc16 = np.nanpercentile(img_masked, 16, axis=0)
                            bkg_pc84 = np.nanpercentile(img_masked, 84, axis=0)
                            
                            bkg_median_stack = mask * np.tile(bkg_median, (img_masked.shape[0], 1))
                            bkg_pc16_stack = mask* np.tile(bkg_pc16, (img_masked.shape[0], 1))
                            bkg_pc84_stack = mask* np.tile(bkg_pc84, (img_masked.shape[0], 1))
                            bkg_sigma = (bkg_pc84_stack - bkg_pc16_stack )/2

                            
                            img_masked = np.where(img_masked> bkg_median_stack + bkg_sigma*alpha, np.nan, img_masked) 
                            img_masked = np.where(img_masked< bkg_median_stack - bkg_sigma*alpha, np.nan, img_masked) 
                            
                            # img_masked = np.where(img_masked> bkg_pc84_stack*alpha, np.nan, img_masked) 
                            # img_masked = np.where(img_masked< bkg_pc16_stack*alpha, np.nan, img_masked) 
                        
                    bkg = np.nanmean(img_masked, axis=0)    
                    bkg2 = bkg*1
        
                    alpha2 = 3
                 
                    for iter in range(3):
                        box = 10; bbox = np.ones(box)/box
                        bkg_smoothed = np.convolve(bkg, bbox, 'same')
                        bkg_smoothed[:box] = np.nanmean(bkg[:box])
                        bkg_smoothed[-box:] = np.nanmean(bkg[-box:])
                        bkg_std = np.nanstd(bkg)
                        # if nrs ==1:
                        #     bkg_std = np.nanstd(bkg[1000:1500])
                        # # if nrs ==2:
                        # #     bkg_std = np.nanstd(bkg[1000:1500])
                    
                        # print (bkg_std)
                        # bkg = np.where(bkg> bkg_smoothed + alpha2*bkg_std, bkg_smoothed, bkg)
                        # bkg = np.where(bkg< bkg_smoothed - alpha2*bkg_std, bkg_smoothed, bkg)
                        x_ = np.arange(len(bkg))
                        idx1  =  np.argwhere(bkg> bkg_smoothed + alpha2*bkg_std).T[0]
                        idx2  =  np.argwhere(bkg< bkg_smoothed - alpha2*bkg_std).T[0]
                        idx  = np.hstack((idx1,idx2))
                        
                        bkg = np.delete(bkg, idx)
                        x0_ = np.delete(x_, idx)
                        bkg = np.interp(x_, x0_, bkg)
                    
                    bkg_smoothed = np.convolve(bkg, bbox, 'same')
                    bkg_smoothed[:box] = np.nanmean(bkg[:box])
                    bkg_smoothed[-box:] = np.nanmean(bkg[-box:])
                    bkg_std = np.nanstd(bkg) 
                    
                    # add back points that are in the right region (some inappropriate taken out due to smoothed func)
                    idx  =  np.argwhere((bkg2 <= bkg_smoothed + alpha2*bkg_std)  & (bkg2 >= bkg_smoothed - alpha2*bkg_std) ).T[0]
  
                    bkg[idx] = bkg2[idx]
                    
 
                # if intg == 50 and gp == 4:
                    
                #     print ('filtered...', intg, gp, np.nansum(img_masked_0), np.nansum(img_masked)  )
                #     plt.figure('filter')
                #     plt.plot(bkg0, 'k-')
                #     # plt.plot(bkg1, 'r-')
                #     plt.plot(bkg, 'g-')
                    
                #     print ('filtered...', intg, gp, np.nansum(img_masked_0), np.nansum(img_masked)  )
                #     plt.figure('filter_a')
         
                #     plt.plot(bkg, 'g-')
            
                  
                    
                #     plt.figure('filter3a')
                #     plt.plot(img_masked_0[:,352],'o-')
                #     plt.plot(img_masked[:,352], 'o-')
                    
                  
                    
                #     plt.figure('filter4')
                #     plt.plot(bkg, 'g-')
                    
                #     plt.figure('filter5')
                #     plt.plot(bkg0[-80:], 'ko-')
                #     # plt.plot(bkg1, 'r-')
                #     plt.plot(bkg[-80:], 'go-')
                    
                #     plt.figure('filter6')
                #     plt.plot(bkg2, 'ro-')
                #     # plt.plot(bkg1, 'r-')
                #     plt.plot(bkg, 'go-')
                    
                    
                #     plt.figure('pppp')
                #     plt.plot(bkg_smoothed + alpha2*bkg_std, 'r--')
                #     plt.plot(bkg_smoothed - alpha2*bkg_std, 'r--')
                #     plt.plot(bkg2, 'b-')
                    
                  
                #     plt.plot(bkg2[idx], 'mo')
                    
        
                    
                #     plt.figure('final')
                #     plt.plot(bkg2, 'ro-')
                #       # plt.plot(bkg1, 'r-')
                #     plt.plot(bkg, 'go-')
                    
                    
                   
                            
                        #     xxxx
                    
                     
                # =============================================================================
                #               
                # =============================================================================
                img = img - bkg 
                    
             
               
                data[intg][gp] = img
  
                # plt.figure()
                # plt.imshow(img, aspect = 'auto')
            
            
        bm.data = data   
        return bm
 
            



 
class CustomBadPixelFlagging(Step):
     
    class_alias = "custom_bad_corr"

    spec = """ """
    # reference_file_types = ['superbias']

    def process(self, input):

        # Open the input data model
        with datamodels.CubeModel(input) as input_model:      
            result = self.custom_bad_pixel_correction(input_model)
            input_model.close()
            result.meta.cal_step.custom_bad_corr = 'COMPLETE'
        return result
    
    def custom_bad_pixel_correction(self, input_model):
        
        sci = input_model.data
        dq_ = input_model.dq


        # exclude pure saturation flags  and jumps from DQ
        # dq_ =  np.where(dq_==2, 0, dq_)
        dq_ = np.where(dq_ & 2 > 0, dq_ -2, dq_)
        dq_ = np.where(dq_ & 4 > 0, dq_-4, dq_)
        
        
       
        
        
        # plt.figure(33)
        # plt.imshow(dq_[0], vmax=1, interpolation='None')
        
        #new way
        bad_flag = np.where(dq_==0,1,np.nan)
        sci *= bad_flag  # bad flag applies nans to bad pixel locations
        plt.figure(34)
        plt.imshow(sci[0], interpolation='None')
        

        
      # """should be okay to get median images before img flagging for outliers
      # as median will tend not to have these"""
      
        # # now get median images
        # median_sci = np.zeros_like(sci)
        # bbox = 2 #range of the median +/-
        # for i in range(bbox, sci.shape[0]-bbox):
        #     slice = sci[i-bbox:i+bbox+1]
        #     median = np.nanmedian(slice, axis=0)
        #     # plt.figure()
        #     # plt.imshow(median)
        #     median_sci[i] = median 
        # # fill edge values with nearest median
        # median_sci[0:bbox] = np.tile(median_sci[bbox], (2, 1, 1))
        # median_sci[-bbox:None] = np.tile(median_sci[-bbox-1], (2, 1, 1)) 
      
        print ("flagging bad pixels...")
        from tqdm import tqdm
        seq = np.arange(sci.shape[0])
        for intg in tqdm(seq):

        # for intg in range(sci.shape[0]):
            
            # print ('bad pixel correction for image %s.....'%(intg))
         
            img = sci[intg]
            dq = dq_[intg]
            
            # print (dq[9][712])
             
            '''
            1) create a bad_flag array for each integration 
            '''
            # # not this ignores saturated flags as these are managed in stage 1 
            # print ('creating a bad_flag array: integration %s'%(intg))
            # aa = np.arange(31)  # include all bad flags except saturation
            # bit_num = 2**aa
            # bit_num = np.delete(bit_num, 1) #exclude saturation flag as no purpose here (dealt with in stage 1)
            # bad_flag = np.ones((dq.shape[0],dq.shape[1]))
            
            # # print (bit_num)
            
            # # find idx of all bad pixels
            # # based on analysis of pixels that looks bad vs don't use the following
            # # for bitnum in [8,1024, 2048,4096, 8192, 262144, 1073741824]:
            # for bitnum in bit_num:
            #      bitarray = np.zeros_like((dq))+bitnum
            #      idx = np.where(dq & bitarray>0) # idx of all bad pixels
            #      for j in range(len(idx[0])):
            #             bad_flag[idx[0][j]][idx[1][j]] = np.nan # flag these with nans
            # # plt.figure('bad flag')
            # # plt.imshow(bad_flag)
            # img *= bad_flag  # bad flag applies nans to bad pixel locations
            # plt.figure('img after bad flag')
            # plt.imshow(img,aspect='auto')
      
            '''
            2) find outliers in the image, based on rolling line median, line sigma and rolling local line sigma
            '''
            
            alpha =  5 #sigma clip level
            iter = 1 # number of iterations
            
            # for idx in range(img.shape[0]) :
            #     xprofile = img[idx]
            #     # plt.figure('image row %s'%(idx))
            #     # plt.plot(xprofile, 'bo-', alpha=0.5, label = 'pre-clip') 
            
            for ii in range(iter):
                bbox = 5 # rolling median of 5 pixels centred on the column i
                local_median = np.zeros((img.shape[0], img.shape[1]))
                local_std = np.zeros((img.shape[0], img.shape[1]))     
                for i in range(bbox, img.shape[1]- bbox):
                    slice = img[:,i-bbox:i+bbox+1]
                    # slice = np.delete(slice, bbox, 1) 
                    local_median[:,i] = np.nanmedian(slice, axis=1)
                    local_std[:,i] = np.nanstd(slice, axis=1)
                # fill edge values with closest column values
                local_median[:,0:bbox] = np.tile(local_median[:,bbox],(bbox, 1)).T
                local_median[:,-bbox:None] = np.tile(local_median[:,-bbox-1],(bbox, 1)).T      
                local_std[:,0:bbox] = np.tile(local_std[:,bbox],(bbox, 1)).T
                local_std[:,-bbox:None] = np.tile(local_std[:,-bbox-1],(bbox, 1)).T        
                # find a line sigma based on the 16-84 pc range    
                line_pc16 = np.nanpercentile(img,16, axis=1) #  
                line_pc84 = np.nanpercentile(img,84, axis=1) #   
                line_sigma =    (line_pc84-line_pc16) /2
                line_sigma = np.tile(line_sigma, (img.shape[1] , 1)).T
                # now clip outliers and replace with nans : based on line sigma
                idx = np.argwhere(img > local_median + alpha*line_sigma).T
                for j in range(len(idx[0])):
                    img[idx[0][j]][idx[1][j]] = np.nan
                idx = np.argwhere(img < local_median - alpha*line_sigma).T
                for j in range(len(idx[0])):
                    img[idx[0][j]][idx[1][j]] = np.nan
                # clip again based on the local std
                idx = np.argwhere(img > local_median + alpha*local_std).T
                for j in range(len(idx[0])):
                    img[idx[0][j]][idx[1][j]] = np.nan
                idx = np.argwhere(img < local_median -  alpha*local_std).T
                for j in range(len(idx[0])):
                    img[idx[0][j]][idx[1][j]] = np.nan
                    
                
            # for idx in range(img.shape[0]) :
            #     xprofile = img[idx]
            #     plt.figure('image row %s'%(idx))
            #     plt.plot(xprofile, 'ro-', alpha=0.5, label = 'post-clip')
            #     plt.legend()
                
               
            '''
            3) produce a median image for each integration (based on the median of surrounding integrations)
            '''
            # a further step to remove CRs in particular
            
       
            # 
            # =============================================================================
            
             
            # median_img = median_sci[intg] 
                
            # # now sigma clip the median image
            # alpha =  3 #sigma clip level (a little tighter than previously)
            # iter = 1
             
            # for ii in range(iter):
            #     bbox = 5 # rolling median of 5 pixels
            #     local_median = np.zeros((median_img.shape[0], median_img.shape[1]))
            #     local_std = np.zeros((median_img.shape[0], median_img.shape[1]))     
            
            #     for i in range(bbox, median_img.shape[1]- bbox):
            #         slice = median_img[:,i-bbox:i+bbox+1]
            #         slice = np.delete(slice, bbox, 1)
            #         local_median[:,i] = np.nanmedian(slice, axis=1)
            #         local_std[:,i] = np.nanstd(slice, axis=1)
            #     local_median[:,0:bbox] = np.tile(local_median[:,bbox],(bbox, 1)).T
            #     local_median[:,-bbox:None] = np.tile(local_median[:,-bbox-1],(bbox, 1)).T      
            #     local_std[:,0:bbox] = np.tile(local_std[:,bbox],(bbox, 1)).T
            #     local_std[:,-bbox:None] = np.tile(local_std[:,-bbox-1],(bbox, 1)).T        
                    
            #     line_pc16 = np.nanpercentile(median_img,16, axis=1) #  
            #     line_pc84 = np.nanpercentile(median_img,84, axis=1) #   
            #     line_sigma =    (line_pc84-line_pc16) /2
            #     line_sigma = np.tile(line_sigma, (median_img.shape[1] , 1)).T
                 
            #     idx = np.argwhere(median_img > local_median + alpha*line_sigma).T
            #     for j in range(len(idx[0])):
            #         median_img[idx[0][j]][idx[1][j]] = np.nan
            #     idx = np.argwhere(median_img < local_median - alpha*line_sigma).T
            #     for j in range(len(idx[0])):
            #         median_img[idx[0][j]][idx[1][j]] = np.nan
                    
            #     idx = np.argwhere(median_img > local_median + alpha*local_std).T
            #     for j in range(len(idx[0])):
            #         median_img[idx[0][j]][idx[1][j]] = np.nan
            #     idx = np.argwhere(median_img < local_median -  alpha*local_std).T
            #     for j in range(len(idx[0])):
            #         median_img[idx[0][j]][idx[1][j]] = np.nan
            
            # median_img0 = median_img*1  
              
            # for idx in range(img.shape[0]) :
            #     medprofile = median_img[idx]
            #     plt.figure('median image after clipping row %s'%(idx))
            #     plt.plot(medprofile, 'b-') 
                
            # # now fill in clipped values using linear interpolation    
            # for i in range(median_img.shape[0]):
            #     median0 = median_img[i]
            #     pos = np.arange(len(median0)) 
            #     idx = np.argwhere(np.isnan(median0)).T[0]
            #     pos2 = np.delete(pos, idx)
            #     median2 = np.delete(median0, idx)
            #     median_img[i] = np.interp(pos, pos2, median2)
            
            # for idx in range(median_img.shape[0]) :
            #     medprofile = median_img[idx]
            #     medprofile0 = median_img0[idx]
            #     plt.figure('median image after clipping row %s'%(idx))
            #     plt.plot(medprofile, 'ro-') 
            #     plt.plot(medprofile0, 'bo-') 
                
            '''
            4) Now clip image again by comparing to the median image
            '''
            # alpha = 3
            # iter= 1
            
            # # we resuse the local_std from the median image clipping
            # for ii in range(iter):
            #     idx = np.argwhere(img > median_img + alpha*local_std).T
            #     for j in range(len(idx[0])):
            #         img[idx[0][j]][idx[1][j]] = np.nan
            #     idx = np.argwhere(img < median_img -  alpha*local_std).T
            #     for j in range(len(idx[0])):
            #         img[idx[0][j]][idx[1][j]] = np.nan
            
            # for idx in range(median_img.shape[0]) :
            #     xprofile = img[idx]
            #     plt.figure('final clipped image row %s'%(idx))
            #     plt.plot(xprofile, 'go-', alpha=0.5) 
             
            '''
            4) Now fill in all clipped locations (all nans in the image)
            '''
            # fill nans with median_img values
            # alternative is to fill with interpolated values
                
            # idx = np.argwhere(np.isnan(img)).T
            # for j in range(len(idx[0])):
            #     img[idx[0][j]][idx[1][j]] =  median_img[idx[0][j]][idx[1][j]]
                
            # for idx in range(median_img.shape[0]) :
            #     xprofile = img[idx]
            #     medprofile = median_img[idx]
                
                
                
                # plt.figure('final filled image row %s'%(idx))
                # plt.plot(xprofile, 'go-', alpha=0.5) 
            
            # plt.figure('final cleaned image: integration %s'%(intg))
            # plt.imshow(img,aspect='auto')
            # # plt.figure('final x profile of cleaned image: integration %s'%(intg))
            # plt.figure('final x profile of cleaned image')
            # plt.plot(img.sum(axis=0))
            
            # xxxx
        
            sci[intg] = img
            
            # plt.figure('spec')
            # plt.plot(img.sum(axis=0))
            
            
            # xxxxx
            
            
            # xxxx
    

        input_model.data = sci   
        return input_model



class CustomBadPixelCorrection3(Step):
     
    class_alias = "custom_bad_corr"

    spec = """ """
    # reference_file_types = ['superbias']

    def process(self, input):

        # Open the input data model
        with datamodels.CubeModel(input) as input_model:      
            result = self.custom_bad_pixel_correction(input_model)
            input_model.close()
            result.meta.cal_step.custom_bad_corr = 'COMPLETE'
        return result
    
    def custom_bad_pixel_correction(self, input_model):
        
        sci = input_model.data
        dq_ = input_model.dq


        # exclude pure saturation flags  and jumps from DQ
        # dq_ =  np.where(dq_==2, 0, dq_)
        dq_ = np.where(dq_ & 2 > 0, dq_ -2, dq_)
        dq_ = np.where(dq_ & 4 > 0, dq_-4, dq_)

        # plt.figure(33)
        # plt.imshow(dq_[0], vmax=1, interpolation='None')
        
        #new way
        bad_flag = np.where(dq_==0,1,np.nan)
        sci *= bad_flag  # bad flag applies nans to bad pixel locations
         
        input_model.data = sci   
        return input_model
    
    
    
    
 
def polyfitw(x, y, w, ndegree, return_fit=0):
   """
   Performs a weighted least-squares polynomial fit with optional error estimates.

   Inputs:
      x: 
         The independent variable vector.

      y: 
         The dependent variable vector.  This vector should be the same 
         length as X.

      w: 
         The vector of weights.  This vector should be same length as 
         X and Y.

      ndegree: 
         The degree of polynomial to fit.

   Outputs:
      If return_fit==0 (the default) then polyfitw returns only C, a vector of 
      coefficients of length ndegree+1.
      If return_fit!=0 then polyfitw returns a tuple (c, yfit, yband, sigma, a)
         yfit:  
            The vector of calculated Y's.  Has an error of + or - Yband.

         yband: 
            Error estimate for each point = 1 sigma.

         sigma: 
            The standard deviation in Y units.

         a: 
            Correlation matrix of the coefficients.

   Written by:   George Lawrence, LASP, University of Colorado,
                 December, 1981 in IDL.
                 Weights added, April, 1987,  G. Lawrence
                 Fixed bug with checking number of params, November, 1998, 
                 Mark Rivers.  
                 Python version, May 2002, Mark Rivers
   """
   n = min(len(x), len(y)) # size = smaller of x,y
   m = ndegree + 1         # number of elements in coeff vector
   a = np.zeros((m,m))  # least square matrix, weighted matrix
   b = np.zeros(m)    # will contain sum w*y*x^j
   z = np.ones(n)     # basis vector for constant term

   a[0,0] = np.sum(w)
   b[0] = np.sum(w*y)

   for p in range(1, 2*ndegree+1):     # power loop
      z = z*x   # z is now x^p
      if (p < m):  b[p] = np.sum(w*y*z)   # b is sum w*y*x^j
      sum = np.sum(w*z)
      for j in range(max(0,(p-ndegree)), min(ndegree,p)+1):
         a[j,p-j] = sum

   a = np.linalg.inv(a)
   c = np.matmul(b, a)
   if (return_fit == 0):
      return c     # exit if only fit coefficients are wanted

   # compute optional output parameters.
   yfit = np.zeros(n)+c[0]   # one-sigma error estimates, init
   for k in range(1, ndegree +1):
      yfit = yfit + c[k]*(x**k)  # sum basis vectors
   var = np.sum((yfit-y)**2 )/(n-m)  # variance estimate, unbiased
   sigma = np.sqrt(var)
   yband = np.zeros(n) + a[0,0]
   z = np.ones(n)
   for p in range(1,2*ndegree+1):     # compute correlated error estimates on y
      z = z*x		# z is now x^p
      sum = 0.
      for j in range(max(0, (p - ndegree)), min(ndegree, p)+1):
         sum = sum + a[j,p-j]
      yband = yband + sum * z      # add in all the error sources
   yband = yband*var
   yband = np.sqrt(yband)
   return c, yfit, yband, sigma, a



def wmean(a, w, axis=None, reterr=False):
    """wmean(a, w, axis=None)

    Perform a weighted mean along the specified axis.

    :INPUTS:
      a : sequence or Numpy array
        data for which weighted mean is computed

      w : sequence or Numpy array
        weights of data -- e.g., 1./sigma^2

      reterr : bool
        If True, return the tuple (mean, err_on_mean), where
        err_on_mean is the unbiased estimator of the sample standard
        deviation.

    :SEE ALSO:  :func:`wstd`
    """
    # 2008-07-30 12:44 IJC: Created this from ...
    # 2012-02-28 20:31 IJMC: Added a bit of documentation
    # 2012-03-07 10:58 IJMC: Added reterr option

    newdata    = np.array(a, subok=True, copy=True)
    newweights = np.array(w, subok=True, copy=True)

    if axis==None:
        newdata    = newdata.ravel()
        newweights = newweights.ravel()
        axis = 0

    ash  = list(newdata.shape)
    wsh  = list(newweights.shape)

    nsh = list(ash)
    nsh[axis] = 1

    if ash<wsh or ash >wsh:
        print('Data and weight must be arrays of same shape.')
        return []
    
    wsum = newweights.sum(axis=axis).reshape(nsh) 
    
    weightedmean = (a * newweights).sum(axis=axis).reshape(nsh) / wsum
    if reterr:
        # Biased estimator:
        #e_weightedmean = sqrt((newweights * (a - weightedmean)**2).sum(axis=axis) / wsum)

        # Unbiased estimator:
        #e_weightedmean = sqrt((wsum / (wsum**2 - (newweights**2).sum(axis=axis))) * (newweights * (a - weightedmean)**2).sum(axis=axis))
        
        # Standard estimator:
        e_weightedmean = np.sqrt(1./newweights.sum(axis=axis))

        ret = weightedmean, e_weightedmean
    else:
        ret = weightedmean

    return ret


def polyfitr(x, y, N, s, fev=100, w=None, diag=False, clip='both', \
                 verbose=False, plotfit=False, plotall=False, eps=1e-13, catchLinAlgError=False):
    """Matplotlib's polyfit with weights and sigma-clipping rejection.

    :DESCRIPTION:
      Do a best fit polynomial of order N of y to x.  Points whose fit
      residuals exeed s standard deviations are rejected and the fit is
      recalculated.  Return value is a vector of polynomial
      coefficients [pk ... p1 p0].

    :OPTIONS:
        w:   a set of weights for the data; uses CARSMath's weighted polynomial 
             fitting routine instead of numpy's standard polyfit.

        fev:  number of function evaluations to call before stopping

        'diag'nostic flag:  Return the tuple (p, chisq, n_iter)

        clip: 'both' -- remove outliers +/- 's' sigma from fit
              'above' -- remove outliers 's' sigma above fit
              'below' -- remove outliers 's' sigma below fit

        catchLinAlgError : bool
          If True, don't bomb on LinAlgError; instead, return [0, 0, ... 0].

    :REQUIREMENTS:
       :doc:`CARSMath`

    :NOTES:
       Iterates so long as n_newrejections>0 AND n_iter<fev. 


     """
    # 2008-10-01 13:01 IJC: Created & completed
    # 2009-10-01 10:23 IJC: 1 year later! Moved "import" statements within func.
    # 2009-10-22 14:01 IJC: Added 'clip' options for continuum fitting
    # 2009-12-08 15:35 IJC: Automatically clip all non-finite points
    # 2010-10-29 09:09 IJC: Moved pylab imports inside this function
    # 2012-08-20 16:47 IJMC: Major change: now only reject one point per iteration!
    # 2012-08-27 10:44 IJMC: Verbose < 0 now resets to 0
    # 2013-05-21 23:15 IJMC: Added catchLinAlgError

    from numpy import polyfit, polyval, isfinite, ones
    from numpy.linalg import LinAlgError
    from pylab import plot, legend, title, figure

    if verbose < 0:
        verbose = 0

    xx = np.array(x, copy=False)
    yy = np.array(y, copy=False)
    noweights = (w==None)
    
  
     
     
    if  noweights.any():
        ww = np.ones(xx.shape[0])
    else:
        ww = np.array(w, copy=False)

    ii = 0
    nrej = 1

    if noweights.any():
        goodind = isfinite(xx)*isfinite(yy)
    else:
        goodind = isfinite(xx)*isfinite(yy)*isfinite(ww)
    
    xx2 = xx[goodind]
    yy2 = yy[goodind]
    ww2 = ww[goodind]

    while (ii<fev and (nrej<0 or nrej>0)):
        if noweights.any():
            p = polyfit(xx2,yy2,N)
            residual = yy2 - polyval(p,xx2)
            stdResidual = np.std(residual)
            clipmetric = s * stdResidual
        else:
            if catchLinAlgError:
                try:
                    p = polyfitw(xx2,yy2, ww2, N)
                except LinAlgError:
                    p = np.zeros(N+1, dtype=float)
            else:
                p = polyfitw(xx2,yy2, ww2, N)

            p = p[::-1]  # polyfitw uses reverse coefficient ordering
            residual = (yy2 - polyval(p,xx2)) * np.sqrt(ww2)
            clipmetric = s

        if clip=='both':
            worstOffender = abs(residual).max()
            #pdb.set_trace()
            if worstOffender <= clipmetric or worstOffender < eps:
                ind = ones(residual.shape, dtype=bool)
            else:
                ind = abs(residual) < worstOffender
        elif clip=='above':
            worstOffender = residual.max()
            if worstOffender <= clipmetric:
                ind = ones(residual.shape, dtype=bool)
            else:
                ind = residual < worstOffender
        elif clip=='below':
            worstOffender = residual.min()
            if worstOffender >= -clipmetric:
                ind = ones(residual.shape, dtype=bool)
            else:
                ind = residual > worstOffender
        else:
            ind = ones(residual.shape, dtype=bool)
    
        xx2 = xx2[ind]
        yy2 = yy2[ind]
        if (not noweights.any()):
            ww2 = ww2[ind]
        ii = ii + 1
        nrej = len(residual) - len(xx2)
        if plotall:
            figure()
            plot(x,y, '.', xx2,yy2, 'x', x, polyval(p, x), '--')
            legend(['data', 'fit data', 'fit'])
            title('Iter. #' + str(ii) + ' -- Close all windows to continue....')

        if verbose:
            print (str(len(x)-len(xx2)) + ' points rejected on iteration #' + str(ii))

    if (plotfit or plotall):
        figure()
        plot(x,y, '.', xx2,yy2, 'x', x, polyval(p, x), '--')
        legend(['data', 'fit data', 'fit'])
        title('Close window to continue....')

    if diag:
        chisq = ( (residual)**2 / yy2 ).sum()
        p = (p, chisq, ii)

    return p



# def opt_extract_old(img, err_img, median, csigma=5, nreject = 10):
def opt_extract_old(img, err_img, dq_img, median, intg, csigma=5, nreject = 10):

    
    iter = 5   
    # gain = 1
    # readnoise = 29
    nreject = 10
    
    data = img
    profile = median
    profile = profile/profile.sum(axis=0)
    profile  = np.where( profile==0,1e-30, profile)
    
    var = err_img**2
    var = np.where(var==0,1e30, var)
    idx = np.isnan(var)
    var[idx] = 1e30
    
 
    gpm = np.ones_like(img)
    
    newBadPixels = True
    # for j in range(iter):
    while newBadPixels:
    
        X = data/profile
        var_X = var/profile**2
        wt_X = gpm*1/var_X
        
        weighted_average_X  =  np.sum((wt_X*X*gpm), axis=0)/ np.sum(wt_X*gpm, axis=0)
        
        spec_opt = weighted_average_X
        var_opt = 1/np.sum(wt_X, axis =0)
        
        modelData = spec_opt * profile
        outlierSigmas = (img - modelData)**2/var
        
        if outlierSigmas.max() > csigma**2:
            maxRejectedValue = max(csigma**2, np.sort(outlierSigmas.ravel())[-nreject])
            worstOutliers = (outlierSigmas>=maxRejectedValue).nonzero()
            gpm[worstOutliers] = False
            newBadPixels = True
            numberRejected = len(worstOutliers[0])
        else:
            newBadPixels = False
            numberRejected = 0
        print ("Rejected %i pixels on this iteration " % numberRejected)
        newBadPixels = False

    # spec_opt = weighted_average_X
    # var_opt = 1/np.sum(wt_X, axis =0)
    
    plt.figure('op1')
    plt.plot(spec_opt, 'b-')
    plt.plot(img.sum(axis=0), 'r-')
    
    plt.figure('op2')
    plt.plot(spec_opt, 'g-')
    
    plt.figure('op3')
    plt.plot(var_opt, 'g+')
    plt.plot((err_img**2).sum(axis=0), 'r-')  
   
    
    return spec_opt, var_opt
   

def opt_extract(img, err_img, dq_img, median, intg, csigma=5, nreject = 10):
    
    iter = 500   
    # gain = 1
    # readnoise = 29
    nreject =10
    
    data = img
    profile = median

    
    
    profile = profile/profile.sum(axis=0)
    profile  = np.where( profile==0,1e-30, profile)
    
    
    
    
    
    var = err_img**2
    var = np.where(var==0, np.nan, var)
    # make all bad var values nan then fix later
 
    gpm = np.ones_like(data)   
    gpm = np.where(np.isnan(data), 0, gpm)  
    gpm = np.where(np.isnan(var), 0, gpm)

    # can't have nans in the calculation, make snr as low as possible, beware of infs
    var = np.where(np.isnan(var), 1e30, var)
    data = np.where(np.isnan(data), 1e-30, data)

    newBadPixels = True
    # for j in range(iter):
        
    ct = 0
    while newBadPixels:
    
        X = data/profile
        var_X = var/(profile**2)
        wt_X = gpm*1/var_X
        
        weighted_average_X  =  np.sum((wt_X*X*gpm), axis=0)/ np.sum(wt_X*gpm, axis=0)
        weighted_average_X2  =  np.nansum((wt_X*X*gpm), axis=0)/ np.nansum(wt_X*gpm, axis=0)

        # aa = 374
        # bb = 373
        
        # print (gpm[:,aa])
        
        # print (img[:,aa])
        
        # print (data[:,aa])
        
        # print (var[:,aa])
        
        # print (X[:,aa])
        
        # print (wt_X[:,aa])
        
        
         
        # plt.figure('5555')
        # plt.plot(X[:,aa], 'bo-')
        # plt.plot(X[:,bb], 'ro-')
        
        # plt.figure('5556')
        # plt.plot(wt_X[:,aa], 'bo-')
        # plt.plot(wt_X[:,bb], 'ro-')
        
        
        # plt.figure('5557')
        # plt.plot(data[:,aa], 'bo-')
        # plt.plot(data[:,bb], 'ro-')
        
        # plt.figure('5558')
        # plt.imshow(data, aspect = 'auto')
        
        # plt.figure('5559')
        # plt.imshow(dq_img, aspect = 'auto')
        
        # print ( wt_X[:,aa]/ np.sum(wt_X[:,aa]) )
        
        # print  (X[:,aa]* wt_X[:,aa]/ np.sum(wt_X[:,aa] ))
        
        # print (np.sum(      X[:,aa]* wt_X[:,aa]/ np.sum(wt_X[:,aa] )))
        
         
         
        
        spec_opt = weighted_average_X
        spec_opt2 = weighted_average_X2

        
    
        
        var_opt = 1/np.sum(wt_X, axis =0)
        var_opt2 = 1/np.nansum(wt_X, axis =0)

        
        modelData = spec_opt * profile
        outlierSigmas = (data - modelData)**2/var
        
        
            
        red_chi_sq = np.sum(outlierSigmas) / profile.size
    
        print ('red_ch_sq...', red_chi_sq )
        
     
        
        # print (outlierSigmas[:,aa])
        # print (outlierSigmas[:,bb])
        if ct > iter:
            newBadPixels = False
            numberRejected = 0
    
        elif outlierSigmas.max() > csigma**2:
            maxRejectedValue = max(csigma**2, np.sort(outlierSigmas.ravel())[-nreject])
            worstOutliers = (outlierSigmas>=maxRejectedValue).nonzero()
             
            gpm0 = gpm*1
            gpm[worstOutliers] = False
            if gpm0.sum() == gpm.sum():
                newBadPixels = False
                numberRejected = 0
            else:
                newBadPixels = True
                numberRejected = len(worstOutliers[0])
                # print (worstOutliers)
                ct+=1
        else:
            newBadPixels = False
            numberRejected = 0
        print ("Rejected %i pixels on this iteration " % numberRejected)
        # newBadPixels = False

    # spec_opt = weighted_average_X
    # var_opt = 1/np.sum(wt_X, axis =0)
     
    
    plt.figure('op1')
    plt.plot(spec_opt, 'g-')
    plt.plot(spec_opt2, 'b-')
    plt.plot(np.nansum(img,axis=0), 'r-')
    
    plt.figure('op2')
    plt.plot(spec_opt, 'g-')
    plt.plot(spec_opt2, 'b-')
    
    plt.figure('op3')
    plt.plot(var_opt, 'g-')
    plt.plot(var_opt2, 'b-')
    plt.plot((err_img**2).sum(axis=0), 'r-') 
    
    
    plt.plot()
    
    
    
    
    
    # if len(np.argwhere(np.isnan(spec_opt)))>0: #rare situation where entire column is bad and no estimators are produced
    #     # print (np.argwhere(np.isnan(spec_opt)).T[0])
        
    #     x_ = np.arange(len(spec_opt))
    #     idx  = np.argwhere(np.isnan(spec_opt))
    #     x_new = np.delete(x_,idx)
        
    #     spec_opt_old = spec_opt*1
    #     var_opt_old = var_opt*1
        
    #     spec_opt_new = np.delete(spec_opt, idx)
    #     var_opt_new = np.delete(var_opt, idx)

    #     spec_opt_new_new = np.interp(x_, x_new, spec_opt_new)
    #     var_opt_new_new = np.interp(x_, x_new, var_opt_new)

    #     spec_opt = spec_opt_new_new
    #     var_opt = var_opt_new_new
        
    #     print ('REQUIRED FILLING IN OF NANS IN FINAL SPECTRUM>>>>>>> integration %s'%(intg))
        
    #     plt.figure('op4')
    #     plt.plot(spec_opt, 'bo-')    
            
    #     plt.figure('op4')
    #     plt.plot(spec_opt_old, 'ro-')
        
    #     plt.figure('op5')
    #     plt.plot(var_opt, 'bo-')    
            
    #     plt.figure('op5')
    #     plt.plot(var_opt_old, 'ro-')
        
        
    #     xxxx
   

    return spec_opt, var_opt, red_chi_sq
   
        
        




def opt_extract_DONOTUSE(frame, variance, gain, readnoise, verbose=True):

    # img_stack =np.load('/Users/c1341133/Downloads/test_img.npy')
    # img = img_stack[0]
    # plt.figure(0)
    # plt.imshow(img, aspect='auto') 

    # # img = np.where(img<0, 0, img)

     
    # var = np.sqrt(img)
    # # idx = np.isnan(var)
    # # var[idx] = 0


    # args = [img,var,1,10]
    # args = [img,var,gain,readnoise]

    kw ={}

 
    """
    Extract spectrum, following Horne 1986.

    :INPUTS:
       data : 2D Numpy array
         Appropriately calibrated frame from which to extract
         spectrum.  Should be in units of ADU, not electrons!

       variance : 2D Numpy array
         Variances of pixel values in 'data'.

       gain : scalar
         Detector gain, in electrons per ADU

       readnoise : scalar
         Detector readnoise, in electrons.

    :OPTIONS:
       goodpixelmask : 2D numpy array
         Equals 0 for bad pixels, 1 for good pixels

       bkg_radii : 2- or 4-sequence
         If length 2: inner and outer radii to use in computing
         background. Note that for this to be effective, the spectral
         trace should be positions in the center of 'data.'
         
         If length 4: start and end indices of both apertures for
         background fitting, of the form [b1_start, b1_end, b2_start,
         b2_end] where b1 and b2 are the two background apertures, and
         the elements are arranged in strictly ascending order.

       extract_radius : int or 2-sequence
         radius to use for both flux normalization and extraction.  If
         a sequence, the first and last indices of the array to use
         for spectral normalization and extraction.


       dispaxis : bool
         0 for horizontal spectrum, 1 for vertical spectrum

       bord : int >= 0
         Degree of polynomial background fit.

       bsigma : int >= 0
         Sigma-clipping threshold for computing background.

       pord : int >= 0
         Degree of polynomial fit to construct profile.

       psigma : int >= 0
         Sigma-clipping threshold for computing profile.

       csigma : int >= 0
         Sigma-clipping threshold for cleaning & cosmic-ray rejection.

       finite : bool
         If true, mask all non-finite values as bad pixels.

       nreject : int > 0
         Number of pixels to reject in each iteration.
             
    :RETURNS:
       3-tuple:
          [0] -- spectrum flux (in electrons)

          [1] -- uncertainty on spectrum flux

          [1] -- background flux


    :EXAMPLE:
      ::


    :SEE_ALSO:
      :func:`superExtract`.

    :NOTES:
      Horne's classic optimal extraction algorithm is optimal only so
      long as the spectral traces are very nearly aligned with
      detector rows or columns.  It is *not* well-suited for
      extracting substantially tilted or curved traces, for the
      reasons described by Marsh 1989, Mukai 1990.  For extracting
      such spectra, see :func:`superExtract`.
    """

    # 2012-08-20 08:24 IJMC: Created from previous, low-quality version.
    # 2012-09-03 11:37 IJMC: Renamed to replace previous, low-quality
    #                        version. Now bkg_radii and extract_radius
    #                        can refer to either a trace-centered
    #                        coordinate system, or the specific
    #                        indices of all aperture edges. Added nreject.


    # from scipy import signal
    
 
    # Parse inputs:
    # frame, variance, gain, readnoise = args[0:4]
    
    # frame = np.where(frame<0,0,frame)
    # variance = np.sqrt(frame)
    
     
    # Parse options:
    if 'goodpixelmask' in kw.keys():
        goodpixelmask = np.array(kw['goodpixelmask'], copy=True).astype(bool)
    else:
        goodpixelmask = np.ones(frame.shape, dtype=bool)
  
    if 'dispaxis' in kw.keys():
        if kw['dispaxis']==1:
            frame = frame.transpose()
            variance = variance.transpose()
            goodpixelmask = goodpixelmask.transpose()
        
    frame = frame.transpose()
    variance = variance.transpose()
    # readnoise = readnoise.transpose
    
    goodpixelmask = goodpixelmask.transpose()

    # if 'verbose' in kw.keys():
    #     verbose = kw['verbose']
    # else:
    #     # verbose = False
    #     verbose = True
        
    # plt.figure(99)
    # plt.imshow(frame, aspect='auto')
    
    # plt.figure(98)
    # plt.plot(frame.sum(axis = 0))
  
    

    
    if 'bkg_radii' in kw.keys():
        bkg_radii = kw['bkg_radii']
    else:
        bkg_radii = [15, 20]
        bkg_radii = [0, 0, 400,432]
        bkg_radii = [0, 5, 18,21]


        if verbose: print("Setting option 'bkg_radii' to: " + str(bkg_radii))


    if 'extract_radius' in kw.keys():
        extract_radius = kw['extract_radius']
    else:
        # extract_radius = 10
        # extract_radius = [5, 17]
        extract_radius = [5, 18]

        if verbose: print("Setting option 'extract_radius' to: " + str(extract_radius))

    if 'bord' in kw.keys():
        bord = kw['bord']
    else:
        bord = 2
        if verbose: print("Setting option 'bord' to: " + str(bord))

    if 'bsigma' in kw.keys():
        bsigma = kw['bsigma']
    else:
        bsigma = 3
        if verbose: print("Setting option 'bsigma' to: " + str(bsigma))

    if 'pord' in kw.keys():
        pord = kw['pord']
    else:
        pord = 31  # this high gives a good match to the box extraction shape
        if verbose: print("Setting option 'pord' to: " + str(pord))

    if 'psigma' in kw.keys():
        psigma = kw['psigma']
    else:
        psigma = 5
        if verbose: print("Setting option 'psigma' to: " + str(psigma))

    if 'csigma' in kw.keys():
        csigma = kw['csigma']
    else:
        csigma = 5
        if verbose: print("Setting option 'csigma' to: " + str(csigma))
        
  
    if 'finite' in kw.keys():
        finite = kw['finite']
    else:
        finite = True
        if verbose: print("Setting option 'finite' to: " + str(finite))

    if 'nreject' in kw.keys():
        nreject = kw['nreject']
    else:
        nreject = 10
        if verbose: print("Setting option 'nreject' to: " + str(nreject))
    
    if finite:
        goodpixelmask *= (np.isfinite(frame) * np.isfinite(variance))

    variance[True^goodpixelmask] = frame[goodpixelmask].max() * 1e9

    nlam, fitwidth = frame.shape
    
    xxx = np.arange(-fitwidth/2, fitwidth/2)
    xxx0 = np.arange(fitwidth)
    
    
    if len(bkg_radii)==4: # Set all borders of background aperture:
      
        backgroundAperture = ((xxx0 > bkg_radii[0]) * (xxx0 <= bkg_radii[1])) + \
            ((xxx0 > bkg_radii[2]) * (xxx0 <= bkg_radii[3]))
    else: # Assume trace is centered, and use only radii.
        backgroundAperture = (np.abs(xxx) > bkg_radii[0]) * (np.abs(xxx) <= bkg_radii[1])

    if hasattr(extract_radius, '__iter__'):
        extractionAperture = (xxx0 > extract_radius[0]) * (xxx0 <= extract_radius[1])
    else:
        extractionAperture = np.abs(xxx) < extract_radius

     
    nextract = extractionAperture.sum()
    xb = xxx[backgroundAperture]
    
   

    #Step3: Sky Subtraction
    
    bord =0
    if bord==0: # faster to take weighted mean:
        background = wmean(frame[:, backgroundAperture], (goodpixelmask/variance)[:, backgroundAperture], axis=1)
    else:
        background = 0. * frame
        for ii in range(nlam):
            print (ii)
            plt.figure(99999)
            if ii >= 267:
                plt.plot(frame[ii, backgroundAperture], 'o-')
            fit = polyfitr(xb, frame[ii, backgroundAperture], bord, bsigma, w=(goodpixelmask/variance)[ii, backgroundAperture], verbose=verbose-1)
            background[ii, :] = np.polyval(fit, xxx)
            
    
    
    # (my 3a: mask any bad values)
    
    # background[10][10]= np.nan #test
    
    badBackground = True^np.isfinite(background) # = True - ...
    
    background[badBackground] = 0.
    if verbose and badBackground.any():
        print ("Found bad background values at: ", badBackground.nonzero())
   
    # skysubFrame = frame - background  #background subtraction row by row, frame by frame
    
    skysubFrame = frame   #background subtraction row by row, frame by frame

   

    #Step4: Extract 'standard' spectrum and its variance
    
    
    standardSpectrum = nextract * wmean(skysubFrame[:, extractionAperture], goodpixelmask[:, extractionAperture], axis=1)
    varStandardSpectrum = nextract * wmean(variance[:, extractionAperture], goodpixelmask[:, extractionAperture], axis=1)
    
    plt.figure(111)
    plt.plot(standardSpectrum)
    plt.plot(varStandardSpectrum)
    
    plt.plot(frame.sum(axis=1))
    
 

    
     

    # (my 4a: mask any bad values)
    badSpectrum = True^(np.isfinite(standardSpectrum))
       
    standardSpectrum[badSpectrum] = 1.
    varStandardSpectrum[badSpectrum] = varStandardSpectrum[True^badSpectrum].max() * 1e9
    
    
    # plt.figure(111)
    # plt.plot(standardSpectrum)
    # plt.plot(varStandardSpectrum)
  

    #Step5: Construct spatial profile; enforce positivity & normalization
    normData = skysubFrame / standardSpectrum
    varNormData = variance / standardSpectrum**2
    
    
    plt.figure(112)
    plt.imshow(normData, aspect ='auto')
    # plt.imshow(varNormData, aspect ='auto')
    plt.figure(113)
    # plt.imshow(normData, aspect ='auto')
    plt.plot(normData[:,12])
    
    
    # Iteratively clip outliers
    newBadPixels = True
    
    iter = -1
    if verbose: print ("Looking for bad pixel outliers in profile construction.")
    
    xl = np.linspace(-1., 1., nlam)
    
   
    while newBadPixels:
        
        iter += 1
        if verbose:
            print (',,,,,,,,,,,,,,', iter)


        if pord==0: # faster to take weighted mean:
            profile = np.tile(wmean(normData, (goodpixelmask/varNormData), axis=0), (nlam,1))
        else:
            profile = 0. * frame
            for ii in range(fitwidth):
                fit = polyfitr(xl, normData[:, ii], pord, np.inf, w=(goodpixelmask/varNormData)[:, ii], verbose=verbose-1)
                profile[:, ii] = np.polyval(fit, xl)  #profile changes as deweighting bad pixels at each iteration
        
        if profile.min() < 0:
            profile[profile < 0] = 0.
        profile /= profile.sum(1).reshape(nlam, 1)

        #Step6: Revise variance estimates 
        # modelData = standardSpectrum * profile + background
        modelData = standardSpectrum * profile  

        variance = (np.abs(modelData)/gain + (readnoise/gain)**2) / \
            (goodpixelmask + 1e-9) # Avoid infinite variance

        # plt.figure(11111)
        # plt.imshow(modelData, aspect='auto')
        # plt.figure(11112)
        # plt.imshow(frame, aspect='auto')
        
        # plt.figure(1111)
        # plt.plot(modelData[:,5])
        # plt.figure(1111)
        # plt.plot(frame[:,5])
        
       
        
        outlierSigmas = (frame - modelData)**2/variance  # find data pixels that deviate from the "model"; Model is made up from the standard spectrum
        if outlierSigmas.max() > psigma**2:
            maxRejectedValue = max(psigma**2, np.sort(outlierSigmas[:, extractionAperture].ravel())[-nreject])
  
            worstOutliers = (outlierSigmas>=maxRejectedValue).nonzero()
            
            goodpixelmask[worstOutliers] = False
            newBadPixels = True
            numberRejected = len(worstOutliers[0])
        else:
            newBadPixels = False
            numberRejected = 0

        if verbose: print ("Rejected %i pixels on this iteration " % numberRejected)

        #Step5: Construct spatial profile; enforce positivity & normalization
        varNormData = variance / standardSpectrum**2
        
        if iter ==1:
            newBadPixels = False

    if verbose: print ("%i bad pixels found" % iter)
    
        
   

    # Iteratively clip Cosmic Rays
    newBadPixels = True
    iter = -1
    if verbose: print ("Looking for bad pixel outliers in optimal extraction.")
    while newBadPixels:
        iter += 1

        #Step 8: Extract optimal spectrum and its variance
        gp = goodpixelmask * profile
        denom = (gp * profile / variance)[:, extractionAperture].sum(1)
        spectrum = ((gp * skysubFrame  / variance)[:, extractionAperture].sum(1) / denom).reshape(nlam, 1)
        varSpectrum = (gp[:, extractionAperture].sum(1) / denom).reshape(nlam, 1)


        #Step6: Revise variance estimates 
        modelData = spectrum * profile
        # modelData = spectrum * profile + background

        variance = (np.abs(modelData)/gain + (readnoise/gain)**2) / \
            (goodpixelmask + 1e-9) # Avoid infinite variance


        #Iterate until worse outliers are all identified:
        outlierSigmas = (frame - modelData)**2/variance
        if outlierSigmas.max() > csigma**2:
            maxRejectedValue = max(csigma**2, np.sort(outlierSigmas[:, extractionAperture].ravel())[-nreject])
            worstOutliers = (outlierSigmas>=maxRejectedValue).nonzero()
            goodpixelmask[worstOutliers] = False
            newBadPixels = True
            numberRejected = len(worstOutliers[0])
        else:
            newBadPixels = False
            numberRejected = 0

        # if verbose: print ("Rejected %i pixels on this iteration " % numberRejected)
        newBadPixels = False


    if verbose: print ("%i bad pixels found" % iter)

    ret = (spectrum, varSpectrum, profile, background, goodpixelmask)
    return spectrum.T[0], varSpectrum.T[0]

# =============================================================================
# spot rod cpnverted
# =============================================================================


def planet_orbit(period, sma_over_rs, eccentricity, inclination, periastron, mid_time, time_array, ww=0):

     
    inclination = inclination * np.pi / 180.0
    

    
    periastron = periastron * np.pi / 180.0
    ww = ww * np.pi / 180.0

    if eccentricity == 0 and ww == 0:
        vv = 2 * np.pi * (time_array - mid_time) / period
        
        
        bb = sma_over_rs * np.cos(vv)
 
        
     
        # added by me
        y = bb * np.cos(inclination)
        x = sma_over_rs * np.sin(vv)
        z =  (y**2+x**2)**0.5
        
        
        # xxxxx
        
        return x,y,z
    
        # looks wrong
        # return [bb * np.sin(inclination), sma_over_rs * np.sin(vv), - bb * np.cos(inclination)]
    
    if periastron < np.pi / 2:
        aa = 1.0 * np.pi / 2 - periastron
    else:
        aa = 5.0 * np.pi / 2 - periastron
    bb = 2 * np.arctan(np.sqrt((1 - eccentricity) / (1 + eccentricity)) * np.tan(aa / 2))
    if bb < 0:
        bb += 2 * np.pi
    mid_time = float(mid_time) - (period / 2.0 / np.pi) * (bb - eccentricity * np.sin(bb))
    m = (time_array - mid_time - np.int_((time_array - mid_time) / period) * period) * 2.0 * np.pi / period
    u0 = m
    stop = False
    u1 = 0
    for ii in range(10000):  # setting a limit of 1k iterations - arbitrary limit
        u1 = u0 - (u0 - eccentricity * np.sin(u0) - m) / (1 - eccentricity * np.cos(u0))
        stop = (np.abs(u1 - u0) < 10 ** (-7)).all()
        if stop:
            break
        else:
            u0 = u1
    if not stop:
        raise RuntimeError('Failed to find a solution in 10000 loops')

    vv = 2 * np.arctan(np.sqrt((1 + eccentricity) / (1 - eccentricity)) * np.tan(u1 / 2))
    #
    rr = sma_over_rs * (1 - (eccentricity ** 2)) / (np.ones_like(vv) + eccentricity * np.cos(vv))
    aa = np.cos(vv + periastron)
    bb = np.sin(vv + periastron)
    x = rr * bb * np.sin(inclination)
    y = rr * (-aa * np.cos(ww) + bb * np.sin(ww) * np.cos(inclination))
    z = rr * (-aa * np.sin(ww) - bb * np.cos(ww) * np.cos(inclination))
    
    z = (y**2 +z**2)**0.5
    
    return x,y,z


def circleangle4(r, p, z):
    
 
    """
    Calculates the angle between the line from the origin to each point in r
    and the line from the origin to the point (p, z).
    
    Parameters:
    r (numpy.ndarray): Array of radii.
    p (float): x-coordinate of the point.
    z (float): y-coordinate of the point.
    
    Returns:
    numpy.ndarray: Array of angles in radians.
    """
    r_squared = r ** 2
    p_squared = p ** 2
    z_squared = z ** 2
    
    angles = np.zeros((len(z), len(r)))
    
    r_squared = np.reshape(r_squared, (1, len(r)))
    r_squared = np.repeat(r_squared, len(z), axis =0)
    
    z_squared = np.reshape(z_squared, (len(z), 1))
    z_squared = np.repeat(z_squared, len(r), axis =1)
    
    
    r_ = np.reshape(r, (1, len(r)))
    r_ = np.repeat(r_, len(z), axis =0) 
    
    z_ = np.reshape(z, (len(z), 1))
    z_ = np.repeat(z_, len(r), axis =1)
 
    # Calculate the angle using the law of cosines
    angles = np.arccos((r_squared + z_squared - p_squared) / (2 * r_ * z_))

    angles[r_ < (p-z_) ]= np.pi

    angles[r_ > (z_+p) ] = 0
    angles[r_ < (z_-p) ] = 0

    
    # plt.figure('angles')
    # plt.plot(angles[targ], '+')
    
    # plt.figure('rr')
    # plt.plot(r_[targ], '.-', label = 'r')
    # plt.plot([p]*len(r), '.-', label ='p')
    # plt.plot(z_[targ], '.-', label ='z')
    # plt.plot(p- z_[targ], '.-', label ='p-z')
    # plt.plot(p +z_[targ], '.-', label ='p+z')
    # plt.plot(z_[targ]-p, '.-', label ='z-p')

    # plt.legend()
 
    return angles


from numba import jit, prange


# @jit(nopython=True, parallel=True)

# @numba.njit(parallel=True)
# @jit(nopython=True, parallel=True)
import numba
@numba.njit(parallel=True)
def integratetransit_numba(m,n,k, planetx, planety, z, p, r, f, spotx, spoty, spotradius, spotcontrast, planetangle):
    
    spotx = np.array(spotx)
    spoty = np.array(spoty)
    spotradius = np.array(spotradius)
    spotcontrast = np.array(spotcontrast)
    spotangle = np.zeros((k,n))
    
    
  
    if k == 0:
        ootflux = 0.0
        aa  =  r*f*np.pi
        ootflux = np.sum(aa)
     
        idx  = np.argwhere(z>=1.0+p)
        aa = (np.pi - planetangle)*r*f
        aa = aa.sum(axis = 1)
        aa = aa/ootflux
    
        idx = np.argwhere(z >=1.0+p).T[0]
        aa[idx]=1.0
        answer_ = aa
        
    else:
        spotcenterdistancesquared  = (spotx**2 + spoty**2) * (1.0 - spotradius**2)
        spotcenterdistance = np.sqrt(spotcenterdistancesquared)
        
        # spotangle = ellipseangle_new2(r, spotradius, spotcenterdistance, n)
        
        
      
       
        a_ = spotradius
        z_ = spotcenterdistance
        
        
        a_ = np.reshape(a_, (len(a_), 1))
        a_ = a_ * np.ones((1, n))
                
        z_ = np.reshape(z_, (len(z_),1))
        z_ = z_ * np.ones((1, n))
        
        
        b = a_ * np.sqrt(1.0 - z_**2 / (1 - a_**2))
        
         
        bminusz = b - z_
        
        zsquared = z_**2
        asquared = a_**2
        A = (a_ / b)**2.0 - 1.0
        
        
        # # Vectorize the calculations
        ri = r
        halfD = zsquared - A * (ri**2 - zsquared - asquared)
        yp = (-z_ + np.sqrt(np.maximum(0, halfD))) / A
        answer__ = np.where(ri <= bminusz, np.pi, np.where((-bminusz < ri) & (yp > -b), np.arccos((z_ - yp) / ri), 0.0))
        
        # # Handle special cases
        answer__ = np.where(a_ == 0, 0.0, answer__)
        mask = (z_ == 0) & (r < a_)
        answer__ = np.where(mask, np.pi, answer__)
        mask = (z_ == 0) & (r >= a_)
        answer__ = np.where(mask, 0.0, answer__)
        spotangle = answer__
        
        trapeze =  np.pi + np.sum((spotcontrast - 1.0) * spotangle.T, axis =1)
        ootflux =  np.sum(trapeze * r * f)
 
        # aa = np.array(planetx*k)
        # bb = np.array([planety]*k)

        
    
        # aa = np.zeros((k, len(planetx)))
        # for i in range (k):
        #     aa[i]= planetx
        bb = np.zeros((k, len(planety)))
        for i in prange (k):
            bb[i]= planety    
            
        aa = np.broadcast_to(planetx, (k, len(planetx)))
        bb = np.broadcast_to(planety, (k, len(planety)))

      
        # planetx = planetx.reshape((len(planetx), 1))
        
        A = (spotx*np.sqrt(1.0 - spotradius** 2)).reshape(k, 1)
        B =(spoty*np.sqrt(1.0 - spotradius** 2)).reshape(k, 1)
        
        d_ = ((aa-A)**2 + (bb-B)**2)**0.5
        
        # z_ = np.zeros((k, len(z)))
        # for i in range (k):
        #     z_[i]= z
            
        z_ = np.broadcast_to(z, (k, len(z)))
# 

        # print (aa.shape, A.shape)
        # xxx

       
        # d_ = ((aa - (spotx*np.sqrt(1.0 - spotradius** 2)).reshape(k, 1))**2 + (bb - (spoty*np.sqrt(1.0 - spotradius** 2)).reshape(k, 1))**2)**0.5
        # z_ = np.array([z]*k)
        
        aa = (z_**2 + (spotcenterdistance.reshape(k,1))**2 - d_**2)/(2.0 * z * spotcenterdistance.reshape(k,1))

        
        planetspotangle_ = np.arccos(aa)       
        planetspotangle_ = np.where(np.isnan(planetspotangle_), np.arccos(1), planetspotangle_)
        # spotcenterdistance_ = np.tile( spotcenterdistance.reshape(k,1), m)
        spotcenterdistance_ = spotcenterdistance.reshape(k, 1) * np.ones((k, m))

        planetspotangle_ = np.where( (spotcenterdistance_ == 0) &  (z_==0), 0, planetspotangle_)
        planetspotangle_ = planetspotangle_.T
        # planetspotangle_ = np.reshape(planetspotangle_, (m,k,1))
        # planetspotangle_ = np.repeat(planetspotangle_, n, axis=2)
        
        planetspotangle_ = planetspotangle_.reshape(m, k, 1)
        planetspotangle_ = np.broadcast_to(planetspotangle_, (m, k, n))
        
        # planetangle_ = np.reshape(planetangle, (m,1,n))
        # planetangle_ = np.repeat(planetangle_, k, axis=1)
        
        planetangle_ = planetangle.reshape(m, 1, n)
        planetangle_ = np.broadcast_to(planetangle_, (m, k, n))     
        
        
        # spotangle_ = np.reshape(spotangle, (1,k,n))
        # spotangle_ = np.repeat(spotangle_, m, axis=0)   
        
        spotangle_ = spotangle.reshape(1, k, n)
        spotangle_ = np.broadcast_to(spotangle_, (m, k, n))                   
        
        values_ = np.zeros_like((planetangle_))
        values_[:,0,:] = np.pi - planetangle_[:,0,:]
        values_orig = values_*1

        # spotcontrast_ = np.reshape(spotcontrast, (1,k,1))
        # spotcontrast_ = np.repeat(spotcontrast_, m, axis=0)
        # spotcontrast_ = np.repeat(spotcontrast_, n, axis=2)
        
        spotcontrast_ = np.broadcast_to(spotcontrast.reshape(1, k, 1), (m, k, n))

 
        values_ = np.where(planetangle_ <= planetspotangle_ + spotangle_, 
                          values_orig + ((spotcontrast_ - 1.0) * (np.pi - planetangle_)), 
                          values_)
        values_ = np.where((planetangle_ <= planetspotangle_ + spotangle_) & (2 * np.pi - planetspotangle_ >= planetangle_ + spotangle_),
                            values_orig + (0.5 * (spotcontrast_ - 1.0) * (spotangle_ + planetspotangle_ - planetangle_)),
                            values_)          
        values_ = np.where(spotangle_ > planetspotangle_ + planetangle_, 
                            values_orig + (spotcontrast_ - 1.0) * (spotangle_ - planetangle_),
                            values_)
        values_ = np.where(planetspotangle_ >  planetangle_ + spotangle_,  
                            values_orig + (spotcontrast_ - 1.0) * spotangle_,
                            values_)
        values_ = np.sum(values_, axis =1)
        answer_ = np.sum(r* f* values_, axis =1)
        answer_ /=ootflux
 
        idx = np.argwhere(z_[0] >= 1.0+p).T[0] # might not be needed 
        answer_[idx] = 1 # might not be needed
    
    
    return answer_

import numba
@numba.njit(parallel=True)
def integratetransit_numba2(m, n, k, planetx, planety, z, p, r, f, spotx, spoty, spotradius, spotcontrast, planetangle):
    
    spotx = np.array(spotx)
    spoty = np.array(spoty)
    spotradius = np.array(spotradius)
    spotcontrast = np.array(spotcontrast)
    spotangle = np.zeros((k,n))
    
    if k == 0:
        ootflux = 0.0
        aa  =  r*f*np.pi
        ootflux = np.sum(aa)
     
        idx  = np.where(z>=1.0+p)[0]
        aa = (np.pi - planetangle)*r*f
        aa = aa.sum(axis = 1)
        aa = aa/ootflux
    
        idx = np.where(z >=1.0+p)[0]
        aa[idx]=1.0
        answer_ = aa
        
    else:
        spotcenterdistancesquared  = (spotx**2 + spoty**2) * (1.0 - spotradius**2)
        spotcenterdistance = np.sqrt(spotcenterdistancesquared)
        
        a_ = spotradius
        z_ = spotcenterdistance
        
        a_ = a_[:, np.newaxis] * np.ones((k, n))
        z_ = z_[:, np.newaxis] * np.ones((k, n))
        
        b = a_ * np.sqrt(1.0 - z_**2 / (1 - a_**2))
        
        bminusz = b - z_
        
        zsquared = z_**2
        asquared = a_**2
        A = (a_ / b)**2.0 - 1.0
        
        ri = r
        halfD = zsquared - A * (ri**2 - zsquared - asquared)
        yp = (-z_ + np.sqrt(np.maximum(0, halfD))) / A
        answer__ = np.where(ri <= bminusz, np.pi, np.where((-bminusz < ri) & (yp > -b), np.arccos((z_ - yp) / ri), 0.0))
        
        answer__ = np.where(a_ == 0, 0.0, answer__)
        mask = (z_ == 0) & (r < a_)
        answer__ = np.where(mask, np.pi, answer__)
        mask = (z_ == 0) & (r >= a_)
        answer__ = np.where(mask, 0.0, answer__)
        spotangle = answer__
        
        
        

def integratetransit(m, n, k, planetx, planety, z, p, r, f, spotx, spoty, spotradius, spotcontrast, planetangle):
 
    spotx = np.array(spotx)
    spoty = np.array(spoty)
    spotradius = np.array(spotradius)
    spotcontrast = np.array(spotcontrast)
    spotangle = np.zeros((k,n))

    if k == 0:
        ootflux = 0.0
        aa  =  r*f*np.pi
        ootflux = np.sum(aa)
     
        idx  = np.argwhere(z>=1.0+p)
        aa = (np.pi - planetangle)*r*f
        aa = aa.sum(axis = 1)
        aa = aa/ootflux
    
        idx = np.argwhere(z >=1.0+p).T[0]
        aa[idx]=1.0
        answer_ = aa
      
    else:
        spotcenterdistancesquared  = (spotx**2 + spoty**2) * (1.0 - spotradius**2)
        spotcenterdistance = np.sqrt(spotcenterdistancesquared)
   
        # for K in range(k):
        #     spotangle[K] = ellipseangle(r, spotradius[K], spotcenterdistance[K], n)
        #     print ('k...1', np.std(spotangle[K]))
            
        #     spotangle[K] = ellipseangle_new(r, spotradius[K], spotcenterdistance[K], n)
        #     print ('k...2', np.std(spotangle[K]))
            
        #     spotangle[K] = ellipseangle_original(r, spotradius[K], spotcenterdistance[K], n)
        #     print ('k...3', np.std(spotangle[K]))
            
        
        spotangle = ellipseangle_new2(r, spotradius, spotcenterdistance, n)
        # for K in range(k):
        #     print ('k...alt', np.std(spotangle[K]))
            
        

        trapeze =  np.pi + np.sum((spotcontrast - 1.0) * spotangle.T, axis =1)
        ootflux =  np.sum(trapeze * r * f)
 
        aa = np.array([planetx]*k)
        bb = np.array([planety]*k)
        d_ = ((aa - (spotx*np.sqrt(1.0 - spotradius** 2)).reshape(k, 1))**2 +
        (bb - (spoty*np.sqrt(1.0 - spotradius** 2)).reshape(k, 1))**2)**0.5
        z_ = np.array([z]*k)

        aa = (z_**2 + (spotcenterdistance.reshape(k,1))**2 - d_**2)/(2.0 * z * spotcenterdistance.reshape(k,1))
 
        planetspotangle_ = np.arccos(aa)       
        planetspotangle_ = np.where(np.isnan(planetspotangle_), np.arccos(1), planetspotangle_)
        spotcenterdistance_ = np.tile( spotcenterdistance.reshape(k,1), m)
        planetspotangle_ = np.where( (spotcenterdistance_ == 0) &  (z_==0), 0, planetspotangle_)
        planetspotangle_ = planetspotangle_.T
        planetspotangle_ = np.reshape(planetspotangle_, (m,k,1))
        planetspotangle_ = np.repeat(planetspotangle_, n, axis=2)
        planetangle_ = np.reshape(planetangle, (m,1,n))
        planetangle_ = np.repeat(planetangle_, k, axis=1)
        spotangle_ = np.reshape(spotangle, (1,k,n))
        spotangle_ = np.repeat(spotangle_, m, axis=0)
      
        values_ = np.zeros_like((planetangle_))
        values_[:,0,:] = np.pi - planetangle_[:,0,:]
        values_orig = values_*1

        spotcontrast_ = np.reshape(spotcontrast, (1,k,1))
        spotcontrast_ = np.repeat(spotcontrast_, m, axis=0)
        spotcontrast_ = np.repeat(spotcontrast_, n, axis=2)
 
        values_ = np.where(planetangle_ <= planetspotangle_ + spotangle_, 
                          values_orig + ((spotcontrast_ - 1.0) * (np.pi - planetangle_)), 
                          values_)
        values_ = np.where((planetangle_ <= planetspotangle_ + spotangle_) & (2 * np.pi - planetspotangle_ >= planetangle_ + spotangle_),
                            values_orig + (0.5 * (spotcontrast_ - 1.0) * (spotangle_ + planetspotangle_ - planetangle_)),
                            values_)          
        values_ = np.where(spotangle_ > planetspotangle_ + planetangle_, 
                            values_orig + (spotcontrast_ - 1.0) * (spotangle_ - planetangle_),
                            values_)
        values_ = np.where(planetspotangle_ >  planetangle_ + spotangle_,  
                            values_orig + (spotcontrast_ - 1.0) * spotangle_,
                            values_)
        values_ = np.sum(values_, axis =1)
        answer_ = np.sum(r* f* values_, axis =1)
        answer_ /=ootflux
 
        idx = np.argwhere(z_[0] >= 1.0+p).T[0] # might not be needed 
        answer_[idx] = 1 # might not be needed
       
    return answer_
 


def ellipseangle_new2(r, a, z, n):
     
    answer = np.zeros((len(a), n))
    
    
    a = np.reshape(a, (len(a),1))
    a = np.repeat(a, n, axis = 1)
    
    z = np.reshape(z, (len(z),1))
    z = np.repeat(z, n, axis = 1)
    
 
    b = a * np.sqrt(1.0 - z**2 / (1 - a**2))
    
     
    bminusz = b - z
    
    zsquared = z**2
    asquared = a**2
    A = (a / b)**2.0 - 1.0
    
 

    # Vectorize the calculations
    ri = r
    halfD = zsquared - A * (ri**2 - zsquared - asquared)
    yp = (-z + np.sqrt(np.maximum(0, halfD))) / A
    answer = np.where(ri <= bminusz, np.pi, np.where((-bminusz < ri) & (yp > -b), np.arccos((z - yp) / ri), 0.0))

    # Handle special cases
    answer[a == 0] = 0.0
    answer[(z == 0) & (r < a)] = np.pi
    answer[(z == 0) & (r >= a)] = 0.0
    

    return answer


def quadraticlimbdarkening(r, u1, u2):
  answer = np.zeros_like(r);
  mask = (r<=1.0);
  oneminusmu = 1.0 - np.sqrt(1.0 - np.power(r[mask],2));
  answer[mask] = 1.0 - u1 * oneminusmu - u2 * np.power(oneminusmu,2);
  return answer;

 

 
# =============================================================================
# light curve fitting
# =============================================================================
import pylightcurve as plc 
from gp.modeling import Model

def transit_model(rat, t0, gamma, per, ars, inc, w, ecc, t, ldc_type='quad'):
    lc = plc.transit(gamma, rat, per, ars, ecc, inc, w, t0, t, method=ldc_type)
    return lc

class TransitModel_old(Model):

      def get_value(self, super_dict):
          
          fixed_params_dic = super_dict['fixed_params_dic']
          # b_sign = super_dict['b_sign']
          # c_sign = super_dict['c_sign']
          grating_correction = super_dict['grating_correction']
          # free_params_names = super_dict['free_params_names']
          ldc_type = super_dict['ldc_type']
          kipping = super_dict['kipping']
          use_c_ratio = super_dict['use_c_ratio']
          shift_idx = super_dict['shift_idx']
          
          
          free_params_names = super_dict['parameter_names']
     
          lc_model_params = super_dict['lc_model_params']
          syst_model_params = super_dict['syst_model_params']
          syst_model_type = super_dict['syst_model_type']





          try: t= fixed_params_dic['t']
          except: t = self.t
          try: t0= fixed_params_dic['t0']
          except: t0 = self.t0
          try:  u0= fixed_params_dic['u0']   
          except:u0 = self.u0        
          try:  u1= fixed_params_dic['u1']   
          except: u1 = self.u1
          try:  per= fixed_params_dic['per']  
          except: per = self.per 
          try: w = fixed_params_dic['w'] 
          except:w = self.w
          try:ecc = fixed_params_dic['ecc']   
          except: ecc = self.ecc
          try:log_ars = np.log(fixed_params_dic['ars'])
          except:log_ars = self.log_ars
          try: log_inc = np.log(fixed_params_dic['inc'])
          except:log_inc = self.log_inc   
          
          if syst_model_type == 'poly':
              
                        
              b_sign = syst_model_params['b']['sign']
              c_sign = syst_model_params['c']['sign']
     
          
              try: log_a = np.log(fixed_params_dic['a'])
              except:log_a = self.log_a
              try: log_b = np.log(fixed_params_dic['b'])
              except:log_b = self.log_b
              try:log_c = np.log(fixed_params_dic['c'])
              except:log_c = self.log_c
              
              
              
          if syst_model_type == 'exp':
              try: a = fixed_params_dic['a']
              except:a = self.a
              try: b = fixed_params_dic['b']
              except:b = self.b
        
              try: A = fixed_params_dic['A']
              except:A = self.A
              try: B = fixed_params_dic['B']
              except:B = self.B
              
              
              
                  
          try: shift = fixed_params_dic['shift']
          except:shift = self.shift
          
          if ldc_type == 'claret':
              try:  u2= fixed_params_dic['u2']   
              except:u2 = self.u2        
              try:  u3= fixed_params_dic['u3']   
              except: u3 = self.u3   
             
          rat = np.sqrt(np.exp(self.log_depth)*1e-6)
          # rat = np.sqrt(self.depth*1e-6)

          
          # t0 = self.t0
          ars =  np.exp(log_ars)  
          inc = np.exp(log_inc)
          
          
          
          # kipping formulation
          if kipping ==1:
              g0 = float(2*np.sqrt(u0)*u1)
              g1 = float(np.sqrt(u0)*(1-2*u1))
              gamma = [g0,g1]
          else:
              gamma = [u0,u1]
              
          if ldc_type == 'claret':
              gamma = [u0, u1, u2, u3]
          
          
 
          lc = plc.transit(gamma, rat, per, ars, ecc, inc, w, t0, t, method=ldc_type)
          
          if grating_correction==1:  lc[shift_idx:] = lc[shift_idx:] - shift
     
          if syst_model_type ==  'poly':
              if use_c_ratio == 1:
                  if "log_b" not in free_params_names and "log_a" in free_params_names:
                         syst =  np.exp(log_a) 
                  if "log_c" not in free_params_names and "log_b" in free_params_names:
                         syst = (np.exp(log_a) +(np.exp(log_a)* b_sign*np.exp(log_b)*t) )   
                  if "log_c" in free_params_names:
                         syst = (np.exp(log_a) +(np.exp(log_a)*b_sign*np.exp(log_b)*t) 
                                    +(np.exp(log_a) * c_sign*np.exp(log_c)*t**2  ))
              else:     
                 if "log_b" not in free_params_names and "log_a" in free_params_names:
                      syst =  np.exp(log_a) 
                 if "log_c" not in free_params_names and "log_b" in free_params_names:
                       syst = (np.exp(log_a) +b_sign*np.exp(log_b)*t)    
                 if "log_c" in free_params_names:
                       syst = (np.exp(log_a) +b_sign*np.exp(log_b)*t +c_sign*np.exp(log_c)*t**2  )
                       
          if syst_model_type ==  'exp':    
              syst = A*np.exp(-B*t)+ b*t + a              
                       

          lc = lc*syst
          
          print (syst)
          xxx
          
          return lc
      
        
      
class TransitModel(Model):

      def get_value(self, t):
          
          # fixed_params_dic = super_dict['fixed_params_dic']
          # # b_sign = super_dict['b_sign']
          # # c_sign = super_dict['c_sign']
          # # grating_correction = super_dict['grating_correction']
          # free_params_names = super_dict['free_params_names']
          # ldc_type = super_dict['ldc_type']
          # kipping = super_dict['kipping']
          # # use_c_ratio = super_dict['use_c_ratio']
          # shift_idx = super_dict['shift_idx']
          
          
         
          for i in range (len(self.parameter_names)):
              if 'log' in self.parameter_names[i]:
                  xx_val = np.exp(self.parameter_vector[i])*self.parameter_signs[i]
                  xx = self.parameter_names[i][4:]
                  self.__dict__[xx] = xx_val
       
     
          u0 = self.u0        
          u1 = self.u1
          per = self.per 
          w = self.w
          ecc = self.ecc
          
          ars = self.ars
          inc = self.inc  
          
          if self.syst_model_type == 'linear':
              a = self.a
              b = self.b
            
          
          if self.syst_model_type == 'quadratic' or self.syst_model_type == 'poly':
              a = self.a
              b = self.b
              c = self.c
          
          elif self.syst_model_type == 'lin_exp' or self.syst_model_type == 'exp':       
              a = self.a
              b = self.b
              c = self.c
              d = self.d
 
 
          # rat = np.sqrt(self.depth*1e-6)
          # rat = np.sqrt(self.depth*1e-6)
          
          rat =self.depth

          t0 = self.t0
          
          
          # print (self.u0   ,     
          #           self.u1,
          #           self.per ,
          #           self.w,
          #           self.ecc,
                    
          #           self.ars,
          #           self.inc  ,
                    
          #           self.a,
          #           self.b,
          #           self.A,
          #           self.B, self.t0)
 
          # xxx
          # kipping formulation
          if self.kipping ==1:
               g0 = float(2*np.sqrt(u0)*u1)
               g1 = float(np.sqrt(u0)*(1-2*u1))
               gamma = [g0,g1]
          else:
               gamma = [u0,u1]
              
          # if ldc_type == 'claret':
          #     gamma = [u0, u1, u2, u3]
 
  
 
          lc = plc.transit(gamma, rat, per, ars, ecc, inc, w, t0, t, method='quad')
          
          # if grating_correction==1:  lc[shift_idx:] = lc[shift_idx:] - shift
          
          if self.syst_model_type == 'quadratic' or self.syst_model_type == 'poly':
              syst =  a*(1+b*t+c*t**2)
             
          if self.syst_model_type == 'lin_exp' or self.syst_model_type == 'exp':        
              # syst = A*np.exp(-B*t)+ b*t + a
              syst = a*(1+ b*t + c*np.exp(-t/d))

          if self.syst_model_type == 'linear':        
              # syst = A*np.exp(-B*t)+ b*t + a
              syst = a*(1+ b*t)          
         
          
          # planetangle = self.spot_planetangle
          # x_,y_,z = self.spot_xyz
          # spot_x,spot_y,spot_size,spot_contrast = self.spot_params
                  
          # n=1000
          # n_spot = len(spot_x)
          # r = np.linspace(1.0/(2*n), 1.0-1.0/(2*n), n)
          # f = 2.0 * quadraticlimbdarkening(r, gamma[0], gamma[1]) / n;
          
          
          

          # spotrod_lc_spot = integratetransit_numba(len(t), n, n_spot, x_, y_, z, rat, r, f, spot_x.tolist(), spot_y.tolist(), spot_size.tolist(), spot_contrast.tolist(), planetangle) 
          # spotrod_lc_no_spot = integratetransit_numba(len(t), n, 0, x_, y_, z, rat, r, f, spot_x.tolist(), spot_y.tolist(), spot_size.tolist(), spot_contrast.tolist(), planetangle) 


          # # spotrod_lc_spot = integratetransit_numba(len(t), n, n_spot, x_, y_, z, rat, r, f, spot_x, spot_y, spot_size, spot_contrast, planetangle) 
          # # spotrod_lc_no_spot = integratetransit_numba(len(t), n, 0, x_, y_, z, rat, r, f, spot_x, spot_y, spot_size, spot_contrast, planetangle) 


           
          # corr = spotrod_lc_spot/spotrod_lc_no_spot
          
          
          if self.fit_spot ==1:
              corr = self.spot_corr

              lc = lc*corr
            
        
        
          lc = lc * syst

          

          return lc
  

class TransitModel_mans(Model):

      def get_value(self, t):
          
          # fixed_params_dic = super_dict['fixed_params_dic']
          # # b_sign = super_dict['b_sign']
          # # c_sign = super_dict['c_sign']
          # # grating_correction = super_dict['grating_correction']
          # free_params_names = super_dict['free_params_names']
          # ldc_type = super_dict['ldc_type']
          # kipping = super_dict['kipping']
          # # use_c_ratio = super_dict['use_c_ratio']
          # shift_idx = super_dict['shift_idx']
          
          
         
          for i in range (len(self.parameter_names)):
              if 'log' in self.parameter_names[i]:
                  xx_val = np.exp(self.parameter_vector[i])*self.parameter_signs[i]
                  xx = self.parameter_names[i][4:]
                  self.__dict__[xx] = xx_val
       
     
          u0 = self.u0        
          u1 = self.u1
          per = self.per 
          w = self.w
          ecc = self.ecc
          
          ars = self.ars
          inc = self.inc  
          
          if self.syst_model_type == 'poly':
              a = self.a
              b = self.b
              c = self.c
          
          elif self.syst_model_type == 'exp':        
              a = self.a
              b = self.b
              c = self.c
              d = self.d
              
              
              # print (a,b,c,d)
              # cccc
              
              
          p0 = a
          p1 = b
          p2 = c
          p3 = d

 
         
 
          if self.use_depth == 1:
              rat = np.sqrt(self.depth*1e-6)
          else:
              rat = self.depth

          

          t0 = self.t0
          
          
          # print (self.u0   ,     
          #           self.u1,
          #           self.per ,
          #           self.w,
          #           self.ecc,
                    
          #           self.ars,
          #           self.inc  ,
                    
          #           self.a,
          #           self.b,
          #           self.A,
          #           self.B, self.t0)
 
          # xxx
          # kipping formulation
          if self.kipping ==1:
               g0 = float(2*np.sqrt(u0)*u1)
               g1 = float(np.sqrt(u0)*(1-2*u1))
               gamma = [g0,g1]
          else:
               gamma = [u0,u1]
              
          # if ldc_type == 'claret':
          #     gamma = [u0, u1, u2, u3]
 
  
 
          lc = plc.transit(gamma, rat, per, ars, ecc, inc, w, t0, t, method='quad')
          
          # if self.spot ==True and 1==1:
          #  n = 1000
           
       
          #    # Midpoint rule for integration.
          #    # Integration annulii radii.
          #  r = np.linspace(1.0/(2*n), 1.0-1.0/(2*n), n)
          #  x_,_y_,z = planet_orbit(per, ars, ecc, inc, w, t0, t, ww=0)
          #  planetangle = self.planetangle
          #  p = [0.1, z.min()-0.05, 0.07, 0.124]
          #  p = [0.1, z.min()-0.05, 0.07, 0.124, 0.5, z.min(), 0.04, 0.124, -0.3, z.min()-0.05, 0.07, 0.124]
          #  y_ = z.min()
          #  x_ = (z**2-y_**2)**0.5
          #  idx = np.argmin(x_)
          #  x_[:idx] = -x_[:idx]
          #  # plt.figure('x')
          #  # plt.plot(timebkjd, x_,'o' )
          #  y_ = x_*0 + y_
          #  f = 2.0 * quadraticlimbdarkening(r, gamma[0], gamma[1]) / n;
          #  spotrod_lc_spot = integratetransit_numba(len(t), n, 1, x_, y_, z, self.depth, r, f, [0.1], [0.1], [0.1], [0.9], planetangle) 

          #  # spotrod_lc_spot = integratetransit_numba(len(t), n, 1, x_, y_, z, self.depth, r, f, [self.xspot], [self.yspot], [self.spot_size], [self.spot_contrast], planetangle) 
          #  # plt.plot(timebkjd, spotrod_lc_spot,'b+')
          #  # spotrod_lc_no_spot = integratetransit(len(t), n, 0, x_, y_, z, self.depth, r, f, [self.xspot], [self.yspot], [self.spot_size], [self.spot_contrast], planetangle) 
          #  # corr = spotrod_lc_spot/spotrod_lc_no_spot
          #  # lc_spot = lc*corr
          #  # lc = lc_spot
          #  # plt.figure('lc')
          #  # plt.plot(timebkjd, lc, '.:', label = 'unspotted')
          #  # plt.plot(timebkjd, lc_spot, '.:', label = 'spotted')

           
           
          
          # if grating_correction==1:  lc[shift_idx:] = lc[shift_idx:] - shift
          
          if self.syst_model_type == 'poly':
              syst =  a*(1+b*t+c*t**2)
             
          if self.syst_model_type == 'exp':        
              # syst = A*np.exp(-B*t)+ b*t + a
              # syst = a*(1+ b*t + c*np.exp(-t/d))
              
              syst = p0 * (1 + p1 * np.exp(-(t) / p2) + p3* (t))


                   
          lc = lc * syst
          

          return lc


  


def simple_log_prob(params, model, t, y, yerr, super_dict):
    #params is a vector of values for free params
    
    # fixed_params_dic =  super_dict['fixed_params_dic'] 
    # free_log_params_prior =  super_dict['free_log_params_prior'] 
    # fit_log_init = super_dict['fit_log_init'] 
    
    free_log_params_prior =  super_dict['prior_ranges'] 
    fit_log_init = super_dict['init_values'] 
    prior_type =  super_dict['prior_type'] 
    log_value =  super_dict['log_value'] 
    parameter_names =  super_dict['parameter_names'] 



    
    # print ('pppp', params[:3])
    model.set_parameter_vector(params, include_frozen=True)
    # resid = y - (model.get_value(super_dict)  )
    resid = y - (model.get_value(t)  )
    
    m = model.get_value(t)

    
    # resid = y - model.get_value(fixed_params_dic)
    # return (-0.5 * np.sum((resid/yerr)**2))  
    log_prob = (-0.5 * np.sum((resid/yerr)**2)) + lnprior(params, free_log_params_prior, fit_log_init, prior_type, log_value, parameter_names)


    # when using kipping, we sometimes get ldc which are very negative giving a nan model and a nan log likelhood.
    #  these seem always associated with -np.inf in the log prior
    # therefore I think we can treat them in same way as other states where the log prior is -inf, and return a value for
    # log likelhood as -inf.  The code will crash if nan is returned but not -inf.  We are saying that nan+-inf = -inf.
    
    # print ('====================================', log_prob)
    
    if np.isnan(log_prob):
        # print ((-0.5 * np.sum((resid/yerr)**2)))
        # print (lnprior(params, free_log_params_prior, fit_log_init, prior_type, log_value, parameter_names))
        # print (np.sum(m))
        # plt.figure('lll')
        # plt.plot(m, 'o')
        # print (m)
        # print (parameter_names)
        # print (params)
        
        if lnprior(params, free_log_params_prior, fit_log_init, prior_type, log_value, parameter_names) == -np.inf:
            log_prob = -np.inf
            # yyyyy
        
        # xxxxx
    
    return log_prob

def lnprior(params, free_log_params_prior, fit_log_init, prior_type, log_value, parameter_names):
    X  = params   
    
    marker = 1
    for i in range(len(X)):
        if prior_type[i]=='uniform':
            prior_range = free_log_params_prior[i]
            init_value = fit_log_init[i]
            if type(prior_range)==list: 
                lower_bound = prior_range[0]
                upper_bound = prior_range[1]
            else:
                lower_bound = init_value - prior_range
                upper_bound = init_value + prior_range
            if  lower_bound < X[i] < upper_bound:
                marker = marker*1
            else:
                marker = marker*0
    if marker == 0:
       return -np.inf
    else:
       lnprior_sum=0
       for i in range(len(X)):
           if prior_type[i]=='gaussian':
               if log_value[i]==True:
                   value = np.exp(X[i])
                   
                   # print ('=======================================', parameter_names[i])
                   # xxxxx
               else:
                   value = X[i]
               mu = free_log_params_prior[i][0]
               sigma = free_log_params_prior[i][1]
               log_prior_prob = np.log(1.0/(np.sqrt(2*np.pi)*sigma))-0.5*(value-mu)**2/sigma**2
               lnprior_sum += log_prior_prob
        
       return lnprior_sum
  
    return -np.inf


def setup_conditional_sampler(model, super_dict):
    # Pre-compute a bunch of things for speed.
    # Compute the predictive mean.
    # model_xs = model.get_value(super_dict)
    
    model_xs = model.get_value(super_dict['t'])

    total = model_xs  
    return  total

 
def get_clean_order_1 (data, wav_sol, col_min, col_max):
    
    import matplotlib.pyplot as plt
    
    aa = np.load('./niriss_wav_sol.npy') #order 1
    
    aa = np.load(wav_sol)
    xpos = aa[0]; ypos=aa[1];wl=aa[2]  
    # print (wl, len(wl))
    level =20
    ap_hw =12
    # ap_hw =17
    ap_hw =21
    start =int(xpos[0])
    mask2 = np.zeros((256,2048), dtype=bool)
    for i in range(0, len(ypos)): #needs to be different for o2
        mask2[:,i+start][int(np.round(ypos[i]))-ap_hw: int(np.round(ypos[i]))+ap_hw+1] = 1
    # plt.figure()
    # plt.imshow(mask, aspect='auto')
 
     
    # data = np.load('/Users/c1341133/Desktop/Subi_jwst_pipeline/res_data.npy')
    data = data
    
    for ii in range(data.shape[0]):
        
        print (ii)
    
        img = data[ii]
    
        # img = np.load('/Users/c1341133/Desktop/Subi_jwst_pipeline/med.npy')
        img[np.isnan(img)] =0
        
       
        # plt.figure('oo')
        # plt.imshow(img, aspect='auto', vmin =0, vmax=2000)
        plt.plot(xpos,ypos, 'r--')
        ypos_max = ypos-20
        ypos_min = ypos-4
        # plt.plot(xpos,ypos_max, 'w--')
        # plt.plot(xpos,ypos_min, 'w--')
        ypos_max = np.round(ypos_max)
        ypos_min = np.round(ypos_min)
        # plt.plot(xpos,ypos_max, 'w--')
        # plt.plot(xpos,ypos_min, 'w--')
        start =int(xpos[0])
        mask = np.zeros((256,2048), dtype=bool)
        for i in range(0, len(ypos)): #needs to be different for o2
            mask[:,i+start][int(ypos_max[i]): int(ypos_min[i])+1] = 1
        img_mask = img*mask
        # plt.figure('pp')
        # plt.imshow(img_mask, aspect='auto', vmin =0, vmax=1000)
        wing_spec = np.sum(img_mask, axis=0)
        # plt.figure('wing_spec')
        # plt.plot(wing_spec)
        
        img_mask2= img*mask2
        # =============================================================================
        # pick a column  #410-840 pix col   
        # =============================================================================
        col = 500
        for col in range(col_min,col_max):
        # for col in range(410,840):
            # plt.figure('psf')
            col_psf_wing = img_mask[:,col]
            col_psf = img_mask2[:,col]
            # plt.plot(col_psf_wing)
            # plt.plot(col_psf)
            # # =============================================================================
            # #get a sample clean psf
            # =============================================================================
            # plt.figure('psf1')
            med_psf = np.median(img[:,1195:1200], axis=1)
            med_psf = med_psf[:85]
            med_psf_pure = med_psf*1
            med_psf_ = np.zeros(img.shape[0])
            med_psf_[:85] = med_psf
            med_psf = med_psf_
            # plt.plot(med_psf)
            med_psf_full = med_psf*1
            med_psf[39:]=0
            med_psf_wing = med_psf
            med_psf = med_psf_full
            
            ##############################
            
            # plt.figure('psf2')
            # plt.plot(col_psf_wing)
            # plt.plot(med_psf_wing)
            
            # =============================================================================
            # adjust fwhem of the sample psf to the target psf
            # =============================================================================
            
            x = np.arange(len(col_psf))
            xnew= np.linspace(0, len(col_psf), len(col_psf)*100 )
            col_psf_hi = np.interp(xnew, x, col_psf) 
            # plt.figure('psf3')
            # plt.plot(x, col_psf)
            # plt.plot(xnew, col_psf_hi, '.')
            idx = np.argwhere(col_psf_hi>=col_psf_hi.max()/2).T[0]
            fwhm2 = xnew[idx[-1]]-xnew[idx[0]]
            # print (fwhm2)
            
            x = np.arange(len(med_psf_pure))
            xnew= np.linspace(0, len(med_psf_pure), len(med_psf_pure)*100 )
            med_psf_pure_hi = np.interp(xnew, x, med_psf_pure) 
            # plt.figure('psf3')
            # plt.plot(x, med_psf_pure)
            # plt.plot(xnew, med_psf_pure_hi, '.')
            idx = np.argwhere(med_psf_pure_hi>=med_psf_pure_hi.max()/2).T[0]
            fwhm1 = xnew[idx[-1]]-xnew[idx[0]]
            # print (fwhm1)
            
            stretch = fwhm2/fwhm1
            # print (stretch)
            xnewnew = np.linspace(xnew[0], xnew[-1], int(np.round(len(xnew)*stretch)) )
            aa = np.interp(xnewnew, xnew, med_psf_pure_hi)
            # plt.figure('psf4')
            
            # plt.plot(med_psf_pure_hi, 'r-')
            # plt.plot(aa, 'b-')
            block_size = 100
            array =aa
            aaa = np.array([np.mean(array[i:i+block_size]) for i in range(0, len(array), block_size)])
            
            # plt.figure('psf5')
            # plt.plot(med_psf_pure, 'r.-')
            # plt.plot(aaa, 'b.-')
            
            # =============================================================================
            #  now use cross correlation to adjust position of the new sample psf
            # =============================================================================
            med_psf_new = aaa
            med_psf_new_wing = med_psf_new*1
            med_psf_new_wing[int(len(med_psf_new_wing)/2):]=0
            aa = med_psf_new_wing
            idx = np.argmax(aa)
            med_psf_new_wing[idx+4:]=0
            med_psf_new_wing[:20]=0
             
            # plt.figure('psf6')
            # plt.plot(med_psf_new_wing)
            aa = np.zeros(256) ; bb=np.zeros(256)
            aa[:len(med_psf_new_wing)] = med_psf_new_wing
            bb[:len(med_psf_new)] = med_psf_new
            med_psf_new_wing = aa
            # plt.figure('psf6')
            # plt.plot(med_psf_new_wing)
            
            cross_correlations = []
            for i in range(len(med_psf_new_wing)):
                rolled_array1 = np.roll(med_psf_new_wing, i)
                cross_correlation = np.dot(rolled_array1, col_psf_wing)
                cross_correlations.append(cross_correlation)
            # Find the roll that maximizes the cross-correlation
            max_index = np.argmax(cross_correlations)
            max_cross_correlation = cross_correlations[max_index]
            # print(f"Roll that maximizes similarity: {max_index}")
            # print(f"Max cross-correlation: {max_cross_correlation}")
            # Plot the arrays and the rolled array with max cross-correlation (optional)
            import matplotlib.pyplot as plt
            # will introduce -1 here as osbervationally looks offset
            # max_index = max_index-1
            rolled_array1 = np.roll(med_psf_new_wing, max_index)
            # plt.plot(med_psf_new_wing, label='Array 1')
            # plt.plot(col_psf_wing, 'o-', label='Array 2')
            # plt.plot(rolled_array1, label='Rolled Array 1 with max cross-correlation')
            # plt.legend()
            # plt.show() 
            
            aa = rolled_array1
            
            # =============================================================================
            # now find scale factor
            # =============================================================================
            idx = np.argsort(col_psf_wing)[::-1]
            idx_ = idx[:7]
            # plt.plot(idx_,col_psf[idx_], 'bo')
            scale = np.mean(rolled_array1[idx_]/col_psf[idx_])
             
            # =============================================================================
            # now scale down the model psf
            # =============================================================================
            
            med_psf_new = np.roll(bb, max_index)
            # plt.plot(med_psf_new, label='Array 1')
            mod_psf = med_psf_new/scale
            # plt.figure('psf7')
            # plt.plot(col_psf, 'b-') 
            # plt.plot(mod_psf, 'r-') 
            
            img[:,col] = mod_psf
    
        # plt.figure('final')
        # plt.imshow(img, aspect='auto', vmax =20)
        
        data[ii] = img
    
    return data
    


def rebin(x, xp, fp):
  ''' Resample a function fp(xp) over the new grid x, rebinning if necessary, 
    otherwise interpolates
    Parameters
    ----------
    x	: 	array like
	New coordinates
    fp 	:	array like
	y-coordinates to be resampled
    xp 	:	array like
	x-coordinates at which fp are sampled
	
    Returns
    -------
    out	: 	array like
	new samples  
  '''
  
  # if (x.unit != xp.unit):
  #   print (x.unit, xp.unit)
  #   exosim_n_error('Units mismatch')
  
  idx = np.where(np.logical_and(xp > 0.9*x.min(), xp < 1.1*x.max()))[0]

  xp = xp[idx]
  fp = fp[idx]
  

  if np.diff(xp).min() < np.diff(x).min():
   
    # print ('binning')    # Binning!
    fp[np.isnan(fp)] = 0
    
    # old method
    # c = cumtrapz(fp, x=xp)
    # xpc = xp[1:]    
    # delta = np.gradient(x)
    # new_c_1 = np.interp(x-0.5*delta, xpc, c, 
    #                     left=0.0, right=0.0)
    # new_c_2 = np.interp(x+0.5*delta, xpc, c, 
    #                     left=0.0, right=0.0)
    # new_f = (new_c_2 - new_c_1)/delta
    
    # I think this is a bit more correct - not much difference in reality
    c = cumtrapz(fp, x=xp)
    xpc = xp[1:]
    # x = x.value
    diff = np.diff(x)
    diff_pos = np.array((diff/2).tolist() +[diff[-1]/2]) # edge bins fix
    diff_neg = np.array([diff[0]/2] +(diff/2).tolist()) # edge bins fix
    delta = diff_pos+diff_neg
    new_c_1 = np.interp(x-diff_neg, xpc, c, 
                        left=0.0, right=0.0)
    new_c_2 = np.interp(x+diff_pos, xpc, c, 
                        left=0.0, right=0.0)
    # new_f = ((new_c_2 - new_c_1)/delta)*fp.unit
    new_f = ((new_c_2 - new_c_1)/delta)
  
    # x = x*xp.unit
        
  else:
    # Interpolate !
    # print ('interpolating')
    new_f = np.interp(x, xp, fp, left=0.0, right=0.0)
    
  # new_f = (new_f.value)*fp.unit
  return new_f
 
    
# =============================================================================
# bin spectrum
# =============================================================================
    
def bin_spectrum(slc, wav, R, bin_size, bin_type = 'R-bin', wavgrid = None, colgrid=None):
    # =============================================================================
    #  find minimum R-power
    # =============================================================================
    # R = wav/ abs(np.gradient(wav))
    # fig = plt.figure('ex_spec')
    # ax = fig.add_subplot(2,2,1)
    # ax.plot(R)
    # plt.figure('Native R')
    # plt.plot(R)
    
    spec = slc
    
    if bin_type == 'R-bin' or  bin_type == 'R' or bin_type == 'R-bin' or bin_type == 'R_min' or bin_type == 'R-min':
    
        # print ('R min and max:', R.min(), R.max())
        if bin_type == 'R_min' or bin_type == 'R-min':
            print ('picking min R')
            R = np.int(R.min())
            print ('binning to R =', R)

        # =============================================================================
        #  R-bin
        # # =============================================================================
 
        wav_range = [wav[0], wav[-1]]
        print ((wav[0], wav[-1]))
        if wav[-1]<wav[0]:
            wav_range = wav_range[::-1]
 
        pixSize = 18 # any number will do , doesn't make a difference
        spec_stack = spec
     
        print('binning to R power of %s...' % (R))

        wl = wav
        x_wav = wav
        x_pix = np.arange(len(wl))
        
        # remove zeros from wl solution ends
        for i in range(len(wl)):
            if wl[i] > 0:
                idx0 = i
                break
        for i in range(len(wl)-1, 0, -1):
            if wl[i] > 0:
                idx1 = i+1
                break
        
        wl0 = wl[idx0:idx1]
        x_pix = x_pix[idx0:idx1]
        
        # b) find w0, the starting wavelength
        if wl0[-1] < wl0[0]:
            w0 = wl0[-1]
        else:
            w0 = wl0[0]
         
        # c) calculate the size of each bin in microns of wavelength
        dw = w0/(R-0.5)
        bin_sizes = [dw]
        for i in range(1000):
            dw2 = (1+1/(R-0.5))*dw
            bin_sizes.append(dw2)
            dw = dw2
            if np.sum(bin_sizes) > wav_range[1]-w0:
                break
        bin_sizes = np.array(bin_sizes)
        
        # d) find the edges of each bin in wavelength space
        wavcen = w0+np.cumsum(bin_sizes)  # the central wavelength of each bin
        wavedge1 = wavcen-bin_sizes/2.  # front edges
        wavedge2 = wavcen+bin_sizes/2.  # back edges
        # obtain an average value for each edge
        wavedge = np.hstack(
            (wavedge1[0], ((wavedge1[1:]+wavedge2[:-1])/2.), wavedge1[-1]))
        
        # e)  find the bin edges in spatial units, in microns where 0 is at the left edge and centre of pixel is pixsize/2
        # e1) translate wl to x (microns)
        wl_osr = x_wav
        x_osr = np.arange(pixSize/2., (pixSize)*(len(x_wav)), pixSize)
        # e2) convert wavedge to xedge (bin edges in spatial units)
        
        xedge = interpolate.interp1d(
            wl_osr, x_osr, kind='linear', bounds_error=False)(wavedge)
        # e3) invert depending on wavelength solution
        if wl0[-1] < wl0[0]:
            xedge = xedge[::-1]
            wavcen = wavcen[::-1]
            wavedge = wavedge[::-1]

        qq = np.round(xedge/pixSize)[2:].astype(int)
        diff = np.diff(qq)
    
        wav0 = np.add.reduceat(x_wav, qq)[:-1]/diff
    
        xedge = xedge[1:]  # make len(xedge) = len(wavcen)
        # e4) remove nans
        idx = np.argwhere(np.isnan(xedge))
        xedge0 = np.delete(xedge, idx)
        wavcen0 = np.delete(wavcen, idx)
        # So now we have A) edges of bins in wavelength units, B) edges of bins distance units : xedge0

        for intg in range(spec_stack.shape[0]):
 
            spec = spec_stack[intg]  # pick a 1 D spectrum
            ct = 0
            count = []
            xedge1 = xedge0
            wavcen1 = wavcen0
            xedgepix = xedge1/pixSize

            for j in range(len(wavcen1)-1):
        
                # selects if next bin edge is NOT in the same pixel
                if int(xedgepix[ct+1]) > int(xedgepix[ct]):
        
                    # selects if next bin edge is in the NEXT pixel
                    if int(xedgepix[ct+1]) == int(xedgepix[ct]) + 1:
                        # signal from the left pixel
                        fracLeft = 1-(xedgepix[ct]-int(xedgepix[ct]))
                        SLeft = spec[int(xedgepix[ct])]*fracLeft
                        # signal from the right pixel
                        fracRight = xedgepix[ct+1]-int(xedgepix[ct+1])
                        SRight = spec[int(xedgepix[ct + 1])]*fracRight
                        # add these together
                        S = SLeft + SRight
                        count.append(S)
                        ct = ct+1
        
                    # selects if next bin edge is NOT in the NEXT pixel
                    else:
                        qq = int(xedgepix[ct])
                        temp = 0
                        # signal from the left pixel
                        fracLeft = 1 - (xedgepix[ct]-int(xedgepix[ct]))
                        SLeft = spec[qq]*fracLeft
                        # add this to a cumulative count
                        temp += SLeft
                        # move to the next pixel
                        for i in range(1000):
                            qq = qq+1
                            S = spec[qq]
                            # add whole pixel count to cumulative
                            temp += S
                            # check if next pixel has the bin edge
                            if xedgepix[ct+1] < qq+2:
                                # add the right pixel fraction to the count
                                fracRight = (xedgepix[ct+1]-int(xedgepix[ct+1]))
                                SRight = spec[qq+1]*fracRight
                                # final count for bin
                                temp += SRight
        
                                count.append(temp)
                                ct = ct+1
                                break
        
                else:
                    # selects if next bin edge is in SAME pixel
                    # find fraction of pixel in the bin
                    frac = xedgepix[ct+1]-xedgepix[ct]
                    # add count
                    S = frac*spec[int(xedgepix[ct])]
                    count.append(S)
                    ct = ct+1
        
            wavcen_list0 = wavcen0[1:]
        
            if intg == 0:
                count_array = count
            else:
                count_array = np.vstack((count_array, count))
        
            # plt.figure('comp spec')
            # plt.plot(wavcen_list0, count, 'ro-')
            # plt.plot(wav, spec, 'bo-')
            # plt.grid()
        
            # plt.figure('comp R')
            # plt.plot(wav, wav/np.gradient(wav), 'bo-')
            # plt.plot(wav, [R]*len(wav), 'ro-')
            # plt.grid()
        
        binned_lc = count_array
        binned_wav = wavcen_list0
        binned_edges = wavedge
        if len(wavedge)  == len(binned_wav)+2:
            binned_edges = wavedge[1:] 
        elif  len(wavedge)  == len(binned_wav)+3:
            binned_edges = wavedge[2:] 
        
        idx =  xedgepix
   
        plt.figure('xpix vs wav')  
        plt.plot(x_pix, wav, '-')
        # plt.plot(xcen, binned_wav, '-')
        # plt.plot(xedgepix,  binned_edges, 'o-')

    
    elif bin_type == 'wavgrid':
     
        # print ('R min and max:', R.min(), R.max())
        if bin_type == 'R_min' or bin_type == 'R-min':
            print ('picking min R')
            R = np.int(R.min())
            print ('binning to R =', R)
  
        wav_range = [wav[0], wav[-1]]
        pixSize = 18
        spec_stack = spec
   
        print('binning to fixed wavelength grid...')
 
        wl = wav
        x_wav = wav
        x_pix = np.arange(len(wl))
        
        # remove zeros from wl solution ends
        for i in range(len(wl)):
            if wl[i] > 0:
                idx0 = i
                break
        for i in range(len(wl)-1, 0, -1):
            if wl[i] > 0:
                idx1 = i+1
                break
        
        wl0 = wl[idx0:idx1]
        x_pix = x_pix[idx0:idx1]
        
        # b) find w0, the starting wavelength
        if wl0[-1] < wl0[0]:
            w0 = wl0[-1]
        else:
            w0 = wl0[0]
            
        if wav_range[1]>wav_range[0]:
            diff = wav_range[1]-wav_range[0] 
        else:
            diff = wav_range[0]-wav_range[1]
        
         
        # c) calculate the size of each bin in microns of wavelength
        dw = w0/(R-0.5)
        bin_sizes = [dw]
        for i in range(1000):
            dw2 = (1+1/(R-0.5))*dw
            bin_sizes.append(dw2)
            dw = dw2
            # if np.sum(bin_sizes) > wav_range[1]-w0:
            if np.sum(bin_sizes) > diff:
     
                break

        bin_sizes = np.array(bin_sizes)
        
        # d) find the edges of each bin in wavelength space
        wavcen = w0+np.cumsum(bin_sizes)  # the central wavelength of each bin
         
        wavedge1 = wavcen-bin_sizes/2.  # front edges
        wavedge2 = wavcen+bin_sizes/2.  # back edges
        # obtain an average value for each edge
        wavedge = np.hstack(
            (wavedge1[0], ((wavedge1[1:]+wavedge2[:-1])/2.), wavedge1[-1]))

        # =============================================================================
        #         
        # =============================================================================
        wavcen = wavgrid
        edges = []
        for i in range(len(wavcen)-1):
            edges.append(wavcen[i] + (wavcen[i+1]- wavcen[i]) / 2)
        edges = np.array(edges)
        plt.figure('test1')
        plt.plot(edges, 'o-')
        
        #deal with edge points
        diff_ =np.diff(edges[:4])
        diff0 = diff_[0]-np.gradient(diff_).mean()
        edges0 = edges[0]-diff0
        edges = np.array([edges0] + edges.tolist())
        # plt.plot(edges, 'o-')
        
        diff_ =np.diff(edges[-4:])
        diff1 = diff_[-1]+np.gradient(diff_).mean()
        edges1 = edges[-1]+diff1
        edges = np.array(edges.tolist()+[edges1])
        # plt.plot(edges, 'o-')
        wavedge = edges
        
   
        # e)  find the bin edges in spatial units, in microns where 0 is at the left edge and centre of pixel is pixsize/2
        # e1) translate wl to x (microns)
        wl_osr = x_wav
        x_osr = np.arange(pixSize/2., (pixSize)*(len(x_wav)), pixSize)
        # e2) convert wavedge to xedge (bin edges in spatial units)
        xedge = interpolate.interp1d(
            wl_osr, x_osr, kind='linear', bounds_error=False)(wavedge)
        # e3) invert depending on wavelength solution
        if wl0[-1] < wl0[0]:
            xedge = xedge[::-1]
            wavcen = wavcen[::-1]
        
        xedge = xedge[:]  # make len(xedge) = len(wavcen)
        # e4) remove nans
        idx = np.argwhere(np.isnan(xedge))
        xedge0 = np.delete(xedge, idx)
        wavcen0 = np.delete(wavcen, idx)
        # So now we have A) edges of bins in wavelength units, B) edges of bins distance units : xedge0
        
        for intg in range(spec_stack.shape[0]):
    
            spec = spec_stack[intg]  # pick a 1 D spectrum
            ct = 0
            count = []
            xedge1 = xedge0
            wavcen1 = wavcen0
            xedgepix = xedge1/pixSize
        
            for j in range(len(wavcen1)):
        
                # selects if next bin edge is NOT in the same pixel
                if int(xedgepix[ct+1]) > int(xedgepix[ct]):
        
                    # selects if next bin edge is in the NEXT pixel
                    if int(xedgepix[ct+1]) == int(xedgepix[ct]) + 1:
                        # signal from the left pixel
                        fracLeft = 1-(xedgepix[ct]-int(xedgepix[ct]))
                        SLeft = spec[int(xedgepix[ct])]*fracLeft
                        # signal from the right pixel
                        fracRight = xedgepix[ct+1]-int(xedgepix[ct+1])
                        SRight = spec[int(xedgepix[ct + 1])]*fracRight
                        # add these together
                        S = SLeft + SRight
                        count.append(S)
                        if ct <= len(xedgepix):
                            ct = ct+1

        
                    # selects if next bin edge is NOT in the NEXT pixel
                    else:
                        qq = int(xedgepix[ct])
                        temp = 0
                        # signal from the left pixel
                        fracLeft = 1 - (xedgepix[ct]-int(xedgepix[ct]))
                        SLeft = spec[qq]*fracLeft
                        # add this to a cumulative count
                        temp += SLeft
                        # move to the next pixel
                        for i in range(1000):
                            qq = qq+1
                            S = spec[qq]
                            # add whole pixel count to cumulative
                            temp += S
                            # check if next pixel has the bin edge
                            if xedgepix[ct+1] < qq+2:
                                # add the right pixel fraction to the count
                                fracRight = (xedgepix[ct+1]-int(xedgepix[ct+1]))
                                SRight = spec[qq+1]*fracRight
                                # final count for bin
                                temp += SRight
        
                                count.append(temp)
                                ct = ct+1
                                break
        
                else:
                    # selects if next bin edge is in SAME pixel
                    # find fraction of pixel in the bin
                    frac = xedgepix[ct+1]-xedgepix[ct]
                    # add count
                    S = frac*spec[int(xedgepix[ct])]
                    count.append(S)
                    ct = ct+1
        
            wavcen_list0 = wavcen0[:]
        
            if intg == 0:
                count_array = count
            else:
                count_array = np.vstack((count_array, count))
        
            # plt.figure('comp spec')
            # plt.plot(wavcen_list0, count, 'ro-')
            # plt.plot(wav, spec, 'bo-')
            # plt.grid()
        
            # plt.figure('comp R')
            # plt.plot(wav, wav/np.gradient(wav), 'bo-')
            # plt.plot(wav, [R]*len(wav), 'ro-')
            # plt.grid()
           
        binned_lc = count_array
        binned_wav = wavcen_list0
        binned_edges = edges 
        
       
    elif bin_type =='col':
    # =============================================================================
    # bins per pix columns
    # =============================================================================

        print ('binning by pixel columns')
        print ('binning to %s pixel columns'%(bin_size))
    
        offs =0
        
        spec = np.add.reduceat(spec, np.arange(int(offs), spec.shape[1])[::int(bin_size)], axis = 1)
        wl = np.add.reduceat(wav, np.arange(offs,len(wav))[::int(bin_size)])  / bin_size     
        idx = np.arange(offs,len(wav))[::int(bin_size)]
        
        
        # print (idx-1)
        grad = np.gradient(wav)
        # print (grad)
        wl_pre_edge = wav[idx-1]
        wl_pre_edge[0] = wav[0]-grad[0]/2
        wl_post_edge = wav[idx]
        wl_edge =  (wl_pre_edge +  wl_post_edge) /2


        if wl[-1] < wl [-2]:
            wl = wl[0:-1]
            spec = spec[:,0:-1]
            idx = idx[:-1]
            
        # print(wl)
                        
        binned_lc = spec
        binned_wav = wl
        binned_edges = wl_edge

        plt.figure('wavelength bins and edges')
        x_range = np.arange(len(wl_edge))
        # plt.plot(x_range[:-1]+0.5, wl, 'ro')
        plt.plot(x_range,  wl_edge, 'bs')
        

    elif bin_type =='col_grid':    
        
        print ('binning by pixel columns to a given grid')
        offs =0        
        idx = np.hstack(([0], colgrid))
        diff = np.diff(idx)
    
        spec = np.add.reduceat(spec, idx, axis=1)[:,:-1] /diff
        wl = np.add.reduceat(wav, idx) [:-1]/ diff
    
        
        # print (idx-1)
        grad = np.gradient(wav)
        # print (grad)
        wl_pre_edge = wav[idx-1]
        wl_pre_edge[0] = wav[0]-grad[0]/2
        wl_post_edge = wav[idx]
        wl_edge =  (wl_pre_edge +  wl_post_edge) /2

        if wl[-1] < wl [-2]:
            wl = wl[0:-1]
            spec = spec[:,0:-1]
            idx = idx[:-1]
                   
        binned_lc = spec
        binned_wav = wl
        binned_edges = wl_edge
       
        plt.figure('wavelength bins and edges')
        x_range = np.arange(len(wl_edge))
        # plt.plot(x_range[:-1]+0.5, wl, 'ro')
        plt.plot(x_range,  wl_edge, 'bs')

    elif bin_type =='col_grid_R':    
 
        print ('binning to R =', R, 'in whole pixels')
        wav_range = [wav[0], wav[-1]]
        if wav[-1]<wav[0]:
            wav_range = wav_range[::-1]

        pixSize = 18 # any number will do , doesn't make a difference     
        spec_stack = spec
        wl = wav
        x_wav = wav
        x_pix = np.arange(len(wl))
        # remove zeros from wl solution ends
        for i in range(len(wl)):
            if wl[i] > 0:
                idx0 = i
                break
        for i in range(len(wl)-1, 0, -1):
            if wl[i] > 0:
                idx1 = i+1
                break
        wl0 = wl[idx0:idx1]
        x_pix = x_pix[idx0:idx1]
     
        
        # b) find w0, the starting wavelength
        if wl0[-1] < wl0[0]:
            w0 = wl0[-1]
        else:
            w0 = wl0[0]
    
        # c) calculate the size of each bin in microns of wavelength
        dw = w0/(R-0.5)
        bin_sizes = [dw]
        for i in range(1000):
            dw2 = (1+1/(R-0.5))*dw
            bin_sizes.append(dw2)
            dw = dw2
            if np.sum(bin_sizes) > wav_range[1]-w0:
                break
        bin_sizes = np.array(bin_sizes)
        
        # d) find the edges of each bin in wavelength space
        wavcen = w0+np.cumsum(bin_sizes)  # the central wavelength of each bin
        wavedge1 = wavcen-bin_sizes/2.  # front edges
        wavedge2 = wavcen+bin_sizes/2.  # back edges
        # obtain an average value for each edge
        wavedge = np.hstack(
            (wavedge1[0], ((wavedge1[1:]+wavedge2[:-1])/2.), wavedge1[-1]))
        
        # e)  find the bin edges in spatial units, in microns where 0 is at the left edge and centre of pixel is pixsize/2
        # e1) translate wl to x (microns)
        wl_osr = x_wav
        x_osr = np.arange(pixSize/2., (pixSize)*(len(x_wav)), pixSize)
        # e2) convert wavedge to xedge (bin edges in spatial units)
        
        xedge = interpolate.interp1d(
            wl_osr, x_osr, kind='linear', bounds_error=False)(wavedge)
        # e3) invert depending on wavelength solution
        if wl0[-1] < wl0[0]:
            xedge = xedge[::-1]
            wavcen = wavcen[::-1]
            wavedge = wavedge[::-1]
      
        # qq = np.round(xedge/pixSize)[2:].astype(int)
        
        qq = np.round(xedge/pixSize)[:].astype(int)

        diff = np.diff(qq)        
        wav0 = np.add.reduceat(x_wav, qq)[:-1]/diff
     
        # print (wav0/np.gradient(wav0))

        binned_lc = np.add.reduceat(spec, qq, axis = 1)[:,:-1]
        binned_wav = wav0
        binned_edges = np.interp(qq, x_osr/pixSize, x_wav)
        idx = qq
        
        # R_ =  abs(wav0/np.gradient(wav0))
        # idx = np.argwhere((R_> R*1.5) | (R_< R*0.4)).T[0]
        
        # binned_lc = np.delete(binned_lc, idx, axis=1)
        # binned_wav = np.delete(binned_wav, idx)

    slc = binned_lc
    wav = binned_wav
   
    edges = binned_edges

    return slc, wav, edges, idx
    

    
