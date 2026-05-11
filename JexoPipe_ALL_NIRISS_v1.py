#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 22 21:28:19 2023
@author: user1
jexopipe run code for NIRISS

"""

import os
#set environmental variables
os.environ['CRDS_PATH'] ='/Users/c24050258/crds_cache'
os.environ['CRDS_SERVER_URL'] ='https://jwst-crds.stsci.edu'
 
import numpy as np
import matplotlib.pyplot as plt
import glob, re

# jwst imports 
import jwst
print(jwst.__version__)

import JexoPipe_stage_1_NIRISS_v1, JexoPipe_stage_2_NIRISS_v1
# import JexoPipe_stage_1_NIRISS_v1
import logging
logging.basicConfig(level=logging.INFO)
logging.basicConfig(level=logging.DEBUG)


channel = 'niriss'


# seg_list = ['001', '002', '003'] #wasp 39 b
# seg_list = ['001', '002'] # TOI-1231 b
# seg_list = ['001', '002', '003', '004'] #k2-18 b
# seg_list = ['001', '002', '003', '004', '005', '006', '007'] #toi1231b
# seg_list = ['001'] #toi1231b f277w
# seg_list = ['001', '002', '003'] #wasp-96
# seg_list = ['001', '002', '003', '004'] #wasp-39

# seg_list = ['001', '002' ] #wasp-52
# seg_list = ['001', '002', '003', '004'] # TOI-1468 c
# seg_list = ['001'] # TOI-1468 c
# seg_list = ['001', '002', '003', '004', '005'] #wasp-17 b
# seg_list = ['001', '002', '003', '004', '005'] #hat-p-11b
# seg_list = ['001', '002', '003', '004', '005'] #lhs-1140 b
seg_list = ['001', '002', '003', '004', '005'] #gj 9827 d visit 1 and 2


# root_folder ='/Volumes/Crucial X9/JexoPipe_NIRISS_raw_files/MAST_2025-11-11T1521_K2_18/JWST/'  #k2-18b
# root_folder ='/Users/c1341133/Downloads/MAST_2025-04-26T1407_br2_nis/JWST/' #toi-1231b
# root_folder ='/Users/c1341133/Downloads/MAST_2025-05-11T1444/JWST/' #wasp-96
# root_folder ='/Users/c1341133/Downloads/MAST_2025-05-11T1516/JWST/' #wasp-39
# root_folder ='/Users/c1341133/Downloads/MAST_2025-05-31T1037_WASP52/JWST/' #wasp-52
# root_folder ='/Users/c1341133/Downloads/MAST_2025-07-11T1705_br3/JWST/'# TOI-1468 c
root_folder ='/Volumes/Crucial X9/JexoPipe_NIRISS_raw_files/MAST_2025-11-07T1055_WASP_17/JWST/' #wasp-17 b
# root_folder='/Users/c1341133/Downloads/MAST_2025-10-24T2315_HAT_P_11b/JWST/' #hat-p-11b
# root_folder = '/Volumes/Crucial X9/JexoPipe_NIRISS_raw_files/MAST_2026-01-12T1444_LHS_1140/JWST/' #lhs-1140 b
# root_folder = '/Volumes/Crucial X9/JexoPipe_NIRISS_raw_files/MAST_2026-01-27T1357_GJ_9827_V1/JWST/' #gj 9827 d visit 1
# root_folder = '/Volumes/Crucial X9/JexoPipe_NIRISS_raw_files/MAST_2026-01-27T1403_GJ_9827_V2/JWST/' #gj 9827 d visit 2
 


# obs_number = 'jw02722003001_04101_00001' #k2-18b
# obs_number = 'jw02722003001_04102_00001' #k2-18b f2
# obs_number = 'jw03557001001_04101_00001' #toi1231b
# obs_number = 'jw03557001001_04102_00001' #toi1231b f277w
# obs_number = 'jw02734002001_04101_00001' #wasp-96 
# obs_number = 'jw01366001001_04101_00001' #wasp-39
# obs_number = 'jw01201501001_04101_00001' #wasp-5s2

# obs_number = 'jw03557007001_04101_00001' # TOI-1468 c
obs_number = 'jw01353101001_04101_00001' #wasp-17 b
# obs_number = 'jw05924001001_04102_00001' #hat-p-11b
# obs_number = 'jw06543001001_04101_00001' #lhs-1140 b
# obs_number = 'jw04098007001_04101_00001' #gj 9827 d visit 1
# obs_number = 'jw04098008001_04101_00001' #gj 9827 d visit 2
    

jexo_dic={}
jexo_dic['f277w']=0
iter =2

if jexo_dic['f277w']!=0:
    seg_list = ['001']
    jexo_dic['f277w']=1
    iter =1
 
    # root_folder = '/Volumes/Crucial X9/JexoPipe_NIRISS_raw_files/MAST_2025-11-07T1055_WASP_17/JWST/'  #wasp17b
    # obs_number = 'jw01353101001_04102_00001'
    
    # root_folder = '/Users/c1341133/Downloads/MAST_2025-04-26T1407_br2_nis/JWST/'  #toi-1231b
    # obs_number = 'jw03557001001_04102_00001'

    
    # root_folder = '/Users/c1341133/Downloads/MAST_2025-07-11T1705_br3/JWST/'# TOI-1468 c
    # obs_number = 'jw03557007001_04102_00001' # TOI-1468 c
    
    # root_folder = '/Volumes/Crucial X9/JexoPipe_NIRISS_raw_files/MAST_2026-01-12T1444_LHS_1140/JWST/' #lhs-1140 b
    # obs_number = 'jw06543001001_04102_00001'


tag = 'WASP_17'

bkd_model = np.load('/Volumes/Crucial X9/NIRISS_Pipeline/Data/model_background256.npy')


for qqq in range(iter):
    
    print ('iteration========', qqq+1)
    stages = [1,2]
    # stages = [2]
    # stages = [1]

    mp = False

    jexo_dic['tag']=tag
    jexo_dic['obs_number']= obs_number
    jexo_dic['root_folder']=root_folder
    jexo_dic['seg_list']=seg_list
    jexo_dic['channel']=channel
    jexo_dic['bkd_model'] = bkd_model
    
    jexo_dic['mp']=mp

    jexo_dic['extra_flag'] =1
    if jexo_dic['f277w']==True:
        jexo_dic['extra_flag'] =0 # need this to stop zeroth order images from being cut
        
    jexo_dic['get_gp_medians_now']=0
    jexo_dic['subtract_zeroth'] =0
    jexo_dic['subtract_contam'] =0
    jexo_dic['sep_spectra'] = 1
    jexo_dic['ss_method_1f']=1
    jexo_dic['apply1f']=1
    jexo_dic['atoca']=0
    
    if jexo_dic['f277w'] ==1:
        jexo_dic['apply1f'] =0
        jexo_dic['ss_method_1f']=0
        jexo_dic['get_gp_medians_now']=0
    
    if __name__ == '__main__':
        if 1 in stages:
            stage_1_files = JexoPipe_stage_1_NIRISS_v1.run(jexo_dic)  

    if __name__ == '__main__':
        if 1 not in stages and 2 in stages:
            root_folder ='./fits_files/%s'%(channel)
            stage_1_files =[]
 
            ct = int(seg_list[0])-1
  
            for seg in seg_list:
                tag0 = 'seg%s_order%s_nis_rateints_%s'%(seg, ct, tag)
                file = '%s/%s-%s.fits'%(root_folder, obs_number, tag0)
                stage_1_files.append(file)
                ct+=1
            print (stage_1_files)
          
            
    tag2 = ''
    just_extract=0  #run both halves of stage 2
    # just_extract=1  #run 2nd half of stage 2 (from calints onwards)
    
    if __name__ == '__main__':
        if 2 in stages:
            wlc_list =  JexoPipe_stage_2_NIRISS_v1.run(jexo_dic, stage_1_files, channel, 0, just_extract=just_extract, tag=tag2)

            if jexo_dic['f277w'] ==1:
                pass
            else:
        
                wlc= wlc_list[0]
                wlc = wlc/np.mean(wlc[10:110])
                plt.figure('wlc0')
                plt.plot(wlc)
                np.save('./%s_%s_wlc.npy'%(obs_number, tag), wlc)
