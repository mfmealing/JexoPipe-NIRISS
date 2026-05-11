#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 22 21:28:19 2023

@author: user1

jexopipe stage 1 for NIRSS

"""
from multiprocessing import Process, Queue

import os
import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
import pastasoss
 
# jwst imports 
import jwst
print(jwst.__version__)


# Individual steps that make up calwebb_detector1
from jwst.dq_init import DQInitStep
from jwst.saturation import SaturationStep
from jwst.superbias import SuperBiasStep
from jwst.ipc import IPCStep                                                                                    
from jwst.refpix import RefPixStep                                                                
from jwst.linearity import LinearityStep
from jwst.persistence import PersistenceStep
from jwst.dark_current import DarkCurrentStep
from jwst.jump import JumpStep
from jwst.ramp_fitting import RampFitStep
from jwst import datamodels


from jwst.gain_scale import GainScaleStep
from jwst.group_scale import GroupScaleStep



def get_gp_medians_sci(queue, xx,  root_folder,obs_number, channel, tag, mp):
    
    print ('pre-pipeline run to obtain sci data upto ref pix stage')
    seg = xx[0:3]
     
    file = '%s/%s-seg%s_nis/%s-seg%s_nis_uncal.fits'%(root_folder,obs_number,seg,  obs_number,xx )

    result = file 
    step = GroupScaleStep()
    result = step.run(result)

    step = DQInitStep()
    result = step.run(result)
            
    step = SaturationStep()
    result = step.run(result)
     
    step = SuperBiasStep()
    result = step.run(result)
    
    step = RefPixStep()
    result = step.run(result)
    
    return result.data

 
def process_seg(queue, xx, jexo_dic):
    
    mp= jexo_dic['mp'] 
    tag = jexo_dic['tag'] 
    obs_number = jexo_dic['obs_number'] 
    root_folder = jexo_dic['root_folder'] 
    channel = jexo_dic['channel'] 
     
    seg = xx[0:3]
    file = '%s/%s-seg%s_nis/%s-seg%s_nis_uncal.fits'%(root_folder,obs_number,seg,  obs_number,xx )
    result = file 
    
# =============================================================================
#   steps
# =============================================================================
    
    step = GroupScaleStep()
    result = step.run(result)
    
    step = DQInitStep()
    result = step.run(result)
            
    step = SaturationStep()
    result = step.run(result)
     
    step = SuperBiasStep()
    result = step.run(result)
    
    step = RefPixStep()
    result = step.run(result)
        
# =============================================================================
# 1/f noise step
# =============================================================================
    if jexo_dic['apply1f'] ==1:
# =============================================================================
#  subtract off the background using the saved jexo_dic['scaled_bkg'] crated in the pre-amble
# =============================================================================
        plt.figure('pre bkg1 subtraction final gp')
        plt.imshow(result.data[0][-1], aspect='auto',vmin=0, vmax=200)
        
        #subtract off background model in each group
        result.data  = result.data - jexo_dic['scaled_bkg'][np.newaxis,:,:,:]
        
        plt.figure('post bkg1 subtraction final gp')
        plt.imshow(result.data[0][-1], aspect='auto',vmin=0, vmax=200)
    
# =============================================================================
#   obtain difference image with the g_stack created in the pre-amble
# =============================================================================
      
        print (jexo_dic['int_start'], jexo_dic['int_end'])
        
        g_stack0 = jexo_dic['g_stack'][jexo_dic['int_start']: jexo_dic['int_end']+1]
          
        diff = result.data- g_stack0
        plt.figure('1/f noise revealed final gp')
        plt.imshow(diff[0][-1], aspect='auto', vmin =0, vmax=200)
        
        plt.figure('1/f noise revealed first gp')
        plt.imshow(diff[0][0], aspect='auto', vmin =0, vmax=200)
        
      
        #mask off bad pixels and the trace
        
        temp_img = result.data[0][-1]
    
        mask = np.ones_like(temp_img)
        try:
            aa = np.load('./niriss_wav_sol_%s_%s.npy'%(jexo_dic['obs_number'], jexo_dic['tag'])) #order 1
        
        except:
            pwcpos = result.meta.instrument.pupil_position
            trace_order1 = pastasoss.get_soss_traces(pwcpos=pwcpos, order='1', interp=True) 
            x_order1, y_order1, wav_order1 = trace_order1.x, trace_order1.y, trace_order1.wavelength
            aa = np.vstack((x_order1, y_order1, wav_order1))
        
        xpos = aa[0]; ypos=aa[1];wl=aa[2]  
        ap_hw =18
        start =int(xpos[0])
        for i in range(0, len(ypos)): #needs to be different for o2
            mask[:,i+start][int(np.round(ypos[i]))-ap_hw: int(np.round(ypos[i]))+ap_hw+1] = np.nan
    
        try:
            aa = np.load('./niriss_wav_sol2_%s_%s.npy'%(jexo_dic['obs_number'],  jexo_dic['tag'])) #order 2
        
        except:
            trace_order2 = pastasoss.get_soss_traces(pwcpos=pwcpos, order='2', interp=True) 
            x_order2, y_order2, wav_order2 = trace_order2.x, trace_order2.y, trace_order2.wavelength
            aa = np.vstack((x_order2, y_order2, wav_order2))
            
        xpos = aa[0]; ypos=aa[1];wl=aa[2]  
        ap_hw =18
        start =int(xpos[0])
        for i in range(0, len(ypos)): #needs to be different for o2
                mask[:,i+start][int(np.round(ypos[i]))-ap_hw: int(np.round(ypos[i]))+ap_hw+1] = np.nan
        plt.figure('1/f mask')
        plt.imshow(mask, aspect = 'auto', vmin=0, vmax = 1)
        
        mask_dq = np.where(result.groupdq !=0, np.nan, 1)
        mask = mask_dq*mask
     
        plt.figure('1/f mask0')
        plt.imshow(mask[0][0], aspect = 'auto', vmin=0, vmax = 1)
        
        plt.figure('1/f mask2')
        plt.imshow(mask[0][-1], aspect = 'auto', vmin=0, vmax = 1)
        
        diff = diff*mask
        
        plt.figure('masked 1/f noise final gp')
        plt.imshow(diff[0][-1], aspect='auto', vmin =0, vmax=200)
        
        plt.figure('masked 1/f noise first gp')
        plt.imshow(diff[0][0], aspect='auto', vmin =0, vmax=200)
        
        #now subtract off 1/f from the bkg-subtracted data
        
        plt.figure('pre 1/f subtraction final gp')
        plt.imshow(result.data[0][-1], aspect='auto', vmin =-20, vmax=20)
     
        result.data = result.data - np.nanmedian(diff, axis = 2)[:,:,np.newaxis, : ]
        
        plt.figure('post 1/f subtraction final gp')
        plt.imshow(result.data[0][-1], aspect='auto', vmin =-20, vmax=20)
        
        #now add back background
        
        plt.figure('pre add back bkg final gp')
        plt.imshow(result.data[0][-1], aspect='auto', vmin =-20, vmax=200)
        
        result.data =  result.data +  jexo_dic['scaled_bkg'][np.newaxis,:,:,:]
        
        plt.figure('post add back bkg final gp')
        plt.imshow(result.data[0][-1], aspect='auto', vmin =-20, vmax=200)
        
         
  
# =============================================================================
#  
# =============================================================================
 
    step = LinearityStep()
    result = step.run(result)
          
    # step = DarkCurrentStep()
    # result = step.run(result)
     
    #omit if very few groups?
    step = JumpStep()
    step.rejection_threshold = 5
    print ("step.rejection_threshold", step.rejection_threshold)
    # for some reason there is a small difference due to division of a segment in the data values if jump is included 
    result = step.run(result)       
  
    step = RampFitStep()
    result = step.run(result)[1]

    step = GainScaleStep()
    result = step.run(result)
    
# =============================================================================
#    
# =============================================================================
    import re
    idx1 = [m.start() for m in re.finditer('/', file)][-1]+1
    idx2  =  file.find('.fits')
    file_name = file[idx1:idx2]
    file_name = file_name.replace('uncal', 'rateints')
    file_path = './fits_files/%s'%(channel)
    if not os.path.exists(file_path):
        os.makedirs(file_path)
           
    result.save('%s/%s_%s.fits'%(file_path,file_name, tag)) 
    output_file='%s/%s_%s.fits'%(file_path,file_name, tag)
    
    
    if mp==True:
        queue.put(output_file)
    else:
        return output_file


def run(jexo_dic):

    mp= jexo_dic['mp'] 
    tag = jexo_dic['tag'] 
    obs_number = jexo_dic['obs_number'] 
    root_folder = jexo_dic['root_folder'] 
    seg_list = jexo_dic['seg_list'] 
    channel = jexo_dic['channel'] 
    get_gp_medians_now = jexo_dic['get_gp_medians_now'] 
    
    file_list =[]
    
    for seg in seg_list:
        file = '%s/%s-seg%s_nis/%s-seg%s_nis_uncal.fits'%(root_folder,obs_number,seg,  obs_number,seg)
        file_list.append(file)
 
    
    ct = int(seg_list[0])-1
    div_list = []
    int_start_list = []
    int_end_list = []
    

    for x in seg_list:
        # file = '%s/%s-seg%s_mirimage/%s-seg%s_mirimage_uncal.fits'%(root_folder,obs_number,x,  obs_number,x)
        file = '%s/%s-seg%s_nis/%s-seg%s_nis_uncal.fits'%(      root_folder,obs_number,x,  obs_number,x )

        hdul = fits.open(file)
        
        
        header0 = hdul[0].header
        
        INTSTART0 = header0['INTSTART']
        INTEND0 = header0['INTEND']
        print ('=====')
        
        print ("INTSTART0, INTEND0", INTSTART0, INTEND0)
        
        int_start_list.append(INTSTART0-1)
        int_end_list.append(INTEND0-1)
    
        
        # outfile0  = '%s/%s-seg%s_mirimage/%s-seg%s_order%s_mirimage_uncal.fits'%(root_folder,obs_number,x,  obs_number,x, ct)
       
        order =   int( float(x)-1)
        # outfile0  = '%s/%s-seg%s_mirimage/%s-seg%s_order%s_mirimage_uncal.fits'%(root_folder,obs_number,x,  obs_number,x, order)
        outfile0  = '%s/%s-seg%s_nis/%s-seg%s_order%s_nis_uncal.fits'%(      root_folder,obs_number,x,  obs_number,x, order )

       
        # # Write the new HDU structure to outfile
        hdul.writeto(outfile0, overwrite=True)
        
        hdul.close()
        


        # file = '%s/%s-seg%s_mirimage/%s-seg%s_order%s_mirimage_uncal.fits'%(root_folder,obs_number,x,   obs_number,x, ct )
        div_list.append('%s_order%s'%(x, ct))
        # div_list.append(file) 
        ct+=1
        
 
# =============================================================================
     # get the first group-wise median stack and scale bkg model
     
     # 1) run the pipeline upto 1/f step and collect the sci from each segment and stack
    if get_gp_medians_now == 1:
        for xx in div_list:
            sci = get_gp_medians_sci(0, xx,  root_folder,obs_number, channel, tag, mp)
            # print (sci.sum())
            if xx == div_list[0]:
                sci_stack = sci
            else:
                sci_stack = np.vstack((sci_stack, sci))
            
      
         # 2) select 100 integrations for the OOT part and create a gp-wise median from this
      
        idx1=10;idx2=110
        sci_stack0 = sci_stack[idx1:idx2]
        
        for gp in range(sci_stack0.shape[1]):
             
             gp_med = np.nanmedian(sci_stack0[:,gp,:,:],axis=0)
             if gp == 0:
                 gp_med_stack = gp_med
             else:
                 gp_med_stack = np.dstack((gp_med_stack, gp_med))
   
        gp_med_stack= np.rollaxis(gp_med_stack, 2, 0)
      
        for i in range(sci_stack.shape[1]):
            plt.figure('group median %s'%(i))
            plt.imshow(gp_med_stack[i,:,:], aspect='auto')
        
        # np.save('/Users/c1341133/Desktop/k2-18b_niriss_gp_median_stack.npy', gp_med_stack)
        np.save('./sci_stack_%s_%s.npy'%(jexo_dic['obs_number'],jexo_dic['tag']), sci_stack)

        np.save('./gp_median_stack_1_%s_%s.npy'%(jexo_dic['obs_number'],jexo_dic['tag']), gp_med_stack)
        jexo_dic['gp_med_stack_1'] = gp_med_stack
        
        #3) save the gp-wise median and the full sci stack
   
# =============================================================================
#       
#     # do bkg subtraction and get second set of medians = bkg subtracted medians
#       
# =============================================================================
    if jexo_dic['apply1f']  ==1:   
        
    #4) we load the gp-wise median and the sci_stack
        
        sci_stack = np.load('./sci_stack_%s_%s.npy'%(jexo_dic['obs_number'],jexo_dic['tag']))        
        gp_med_stack = np.load('./gp_median_stack_1_%s_%s.npy'%(jexo_dic['obs_number'],jexo_dic['tag']))
        
        for i in range( gp_med_stack.shape[0]):
            plt.figure('group median %s 2'%(i))
            plt.imshow(gp_med_stack[i,:,:], aspect='auto')  
            
        # load the bkg model
        bkd_model = jexo_dic['bkd_model'] 
    
    # =============================================================================
    # 5 we assume the sky bkg does not change with integration so we must scale the bkg model for each gp, using the gp median
    # 5b : we produce scaled bkg of the same shape as the group stack
    
        # define the sampling region: check these do not fall over contaminants or spectra
        x1=250;x2=500;y1=210;y2=250 # standard
        # x1=250;x2=500;y1=225;y2=250 # gj 9827 d
        x1A=715;x2A=750;y1A=235;y2A=250
        
        if jexo_dic['obs_number'] == 'jw03557007001_04101_00001':  #toi-1468 c
            x1=250;x2=500;y1=190;y2=230
            x1A=715;x2A=760;y1A=210;y2A=220
          
        plt.figure('box check')
        plt.imshow(gp_med_stack[-1], aspect='auto', vmin=0, vmax=200)
        plt.plot([x1,x1], [y1,y2], 'r-')
        plt.plot([x2,x2], [y1,y2], 'r-')
        plt.plot([x1,x2], [y1,y1], 'r-')
        plt.plot([x1,x2], [y2,y2], 'r-')
        
        
        plt.plot([x1A,x1A], [y1A,y2A], 'r-')
        plt.plot([x2A,x2A], [y1A,y2A], 'r-')
        plt.plot([x1A,x2A], [y1A,y1A], 'r-')
        plt.plot([x1A,x2A], [y2A,y2A], 'r-')
             
              
    
    # =============================================================================
    #   supreme spoon method
    # =============================================================================
        print ('supspn method scale 1') 
        # x1=250;x2=500;y1=5;y2=45
      
        # x1=350;x2=550;y1=230;y2=250
        
        
        section = gp_med_stack[:,y1:y2,x1:x2]
        bkd_section = bkd_model[y1:y2,x1:x2]
        
        print (section.shape, bkd_section.shape)
        
        
           
        # plt.figure('scaled background box st1')
        # plt.imshow(sci_stack[9][-1], aspect = 'auto')       
        # plt.plot([x1,x1], [y1,y2], 'r-')
        # plt.plot([x2,x2], [y1,y2], 'r-')
        # plt.plot([x1,x2], [y1,y1], 'r-')
        # plt.plot([x1,x2], [y2,y2], 'r-')
        
        scale_factor1_list =[]
        
        plt.figure('gp bkg')
        shifts=np.zeros((gp_med_stack.shape[0]))
        
        b_list =[]
        for i in range( gp_med_stack.shape[0]):
            
            section_med =  section[i]
            scale_factor1 = -1000
        
            while scale_factor1 < 0:
                   
                    bkg_ratio = ((section_med + shifts[i]) / bkd_section)
                    # Instead of a straight median, use the median of the 2nd
                    # quartile to limit the effect of any remaining illuminated
                    # pixels.
                    q1 = np.nanpercentile(bkg_ratio, 25)
                    q2 = np.nanpercentile(bkg_ratio, 50)
                    ii = np.where((bkg_ratio > q1) & (bkg_ratio < q2))
                    scale_factor1 = np.nanmedian(bkg_ratio[ii])
                    print (i,shifts[i], scale_factor1)
                    if scale_factor1 < 0:
                        shifts[i] -= scale_factor1 * np.median(bkd_section)
         
            print (i,  scale_factor1)
            scale_factor1_list.append(scale_factor1)
            
        for i in range( gp_med_stack.shape[0]):
            print (i,  scale_factor1_list[i])
            
     
        plt.figure('bkg slope supspn')
    
        b_list =[]
        for i in range( gp_med_stack.shape[0]):    
            section_med =  section[i]
            plt.plot(i, np.nanmedian (section_med), 'kx')
            b_list.append(np.nanmedian (section_med))
    
       
        print ('supspn method scale 2') 
    
        section = gp_med_stack[:,y1A:y2A,x1A:x2A]
        bkd_section = bkd_model[y1A:y2A,x1A:x2A]
                     
        # plt.figure('scaled background box st1')
        # plt.imshow(sci_stack[9][-1], aspect = 'auto')       
    
        # plt.plot([x1A,x1A], [y1A,y2A], 'r-')
        # plt.plot([x2A,x2A], [y1A,y2A], 'r-')
        # plt.plot([x1A,x2A], [y1A,y1A], 'r-')
        # plt.plot([x1A,x2A], [y2A,y2A], 'r-')
           
        scale_factor2_list =[]
     
        b_list =[]
        plt.figure('gp bkg')
        for i in range( gp_med_stack.shape[0]):
             
            section_med =  section[i]
            bkg_ratio = ((section_med + shifts[i]) / bkd_section)
            # Instead of a straight median, use the median of the 2nd
            # quartile to limit the effect of any remaining illuminated
            # pixels.
            q1 = np.nanpercentile(bkg_ratio, 25)
            q2 = np.nanpercentile(bkg_ratio, 50)
            ii = np.where((bkg_ratio > q1) & (bkg_ratio < q2))
            scale_factor2 = np.nanmedian(bkg_ratio[ii])
            if scale_factor2 < 0:
                scale_factor2 =0
                   
            print (i, scale_factor2)
            scale_factor2_list.append(scale_factor2)
            
        plt.figure('bkg slope supspn')
    
        b_list =[]
        for i in range( gp_med_stack.shape[0]):    
            section_med =  section[i]
            plt.plot(i, np.nanmedian (section_med), 'kx')
            b_list.append(np.nanmedian (section_med))
     
        plt.figure('scale factor')
        plt.plot(scale_factor1_list, 'rx-')
        plt.plot(scale_factor2_list, 'bx-')
         
        
        
        grad_bkg = np.gradient(bkd_model, axis=1)
        step_pos = np.argmax(grad_bkg[:, 10:-10], axis=1) + 10 - 4
        
        print ('step pos and shifts')  
        print (step_pos)
        print (shifts)
        
      
        
        # Apply differential scaling to either side of step.
        scaled_bkg = np.zeros_like(gp_med_stack)
        
        for i in range( gp_med_stack.shape[0]):
            for j in range(256):
                    scaled_bkg[i,j, :step_pos[j]] = bkd_model[j, :step_pos[j]] * scale_factor1_list[i] - shifts[i]
                    scaled_bkg[i,j, step_pos[j]:] = bkd_model[j, step_pos[j]:] * scale_factor2_list[i] - shifts[i]
             
        plt.figure('bkg slope supspn')
        for i in range( gp_med_stack.shape[0]):    
       
            plt.plot(i, np.nanmedian(scaled_bkg[i][y1:y2,x1:x2]), 'r+')
            plt.plot(i, np.nanmedian(scaled_bkg[i][y1A:y2A,x1A:x2A]), 'r+')
     
                
    # =============================================================================
    # JexoPipe custom method    - use this       
    # =============================================================================
        print ('subi method scale 1') 
    
        section1 = gp_med_stack[:,y1:y2,x1:x2]
        bkd_section1 = bkd_model[y1:y2,x1:x2]
                     
           
        plt.figure('scaled background box st1')
        plt.imshow(sci_stack[9][-1], aspect = 'auto')       
        plt.plot([x1,x1], [y1,y2], 'r-')
        plt.plot([x2,x2], [y1,y2], 'r-')
        plt.plot([x1,x2], [y1,y1], 'r-')
        plt.plot([x1,x2], [y2,y2], 'r-')
        
        
        plt.figure('bkg slope subi', figsize=(10,10))
    
        b_list1 =[]
        for i in range( gp_med_stack.shape[0]):    
            section_med =  section1[i]
            plt.plot(i, np.nanmedian (section_med), 'kx')
            b_list1.append(np.nanmedian (section_med))
                      
      
        time1 = np.arange(0,len(b_list1))
        signal1 = np.array(b_list1)
        
        
        # Perform linear fit (y = mx + c)
        # np.polyfit returns [slope, intercept]
        slope, intercept = np.polyfit(time1, signal1, 1)
        
        
        # Create fitted line
        time1a = np.linspace(-1, time1[-1],100)
        
        fitted_signal1 = slope * time1a + intercept
        
        
        m1,c1= slope, intercept
    
        


        
        print ('subi method scale 2') 
        
        section2 = gp_med_stack[:,y1A:y2A,x1A:x2A]
        bkd_section2 = bkd_model[y1A:y2A,x1A:x2A]
                     
        # plt.figure('scaled background box st1')
        plt.figure('scaled background box st1')
        plt.plot([x1A,x1A], [y1A,y2A], 'r-')
        plt.plot([x2A,x2A], [y1A,y2A], 'r-')
        plt.plot([x1A,x2A], [y1A,y1A], 'r-')
        plt.plot([x1A,x2A], [y2A,y2A], 'r-')
        
     
        plt.figure('bkg slope subi')
        b_list2 =[]
        for i in range( gp_med_stack.shape[0]):   
            section_med =  section2[i]
            plt.plot(i, np.nanmedian (section_med), 'kx')
            b_list2.append(np.nanmedian (section_med))
                             
      
        time2 = np.arange(0,len(b_list2))
        signal2 = np.array(b_list2)
        
        # Perform linear fit (y = mx + c)
        # np.polyfit returns [slope, intercept]
        slope, intercept = np.polyfit(time2, signal2, 1)
        m2,c2= slope, intercept
        
        # Create fitted line
        time2a = np.linspace(-1, time2[-1],100)
        
        fitted_signal2 = slope * time2a + intercept
     
        
        plt.figure('bkg slope subi')
        
        
        # plt.scatter(time, signal,  color='blue', label='Data points')
        plt.plot(time1a, fitted_signal1, color='red', label='Linear fit: Pre 700')
        plt.plot(time2a, fitted_signal2, color='blue', label='Linear fit: Post 700')
    
        plt.xlabel('Time', fontsize=15, fontweight='bold')
        plt.ylabel('Signal', fontsize=15, fontweight='bold')
        # plt.title('Linear Fit of Time vs Signal')
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
        plt.legend(fontsize=15, fontweight='bold')
        plt.grid(True)
        plt.show()
    
 
            
        def find_intersection_y(m1, c1, m2, c2):
            if m1 == m2:
                return None  # Lines are parallel, no intersection
            y = (m1 * c2 - m2 * c1) / (m1 - m2)
            return y
    
        intersection = find_intersection_y(m1, c1, m2, c2)
        print ('intersection', intersection)
        
        pedestal1 = pedestal2 = intersection
        plt.axhline(y=intersection, color='black', linestyle='--', label='Intersection')
        
        xxx
        
        print (pedestal1,pedestal2)
        
        scale_factor1_list=[]
        for i in range( gp_med_stack.shape[0]):
            section_med =  section1[i]
            bkg_ratio = ((section_med -pedestal1) / bkd_section1)
            q1 = np.nanpercentile(bkg_ratio, 25)
            q2 = np.nanpercentile(bkg_ratio, 50)
            ii = np.where((bkg_ratio > q1) & (bkg_ratio < q2))
            # scale_factor1 = np.nanmedian(bkg_ratio[ii])
            scale_factor1 = np.nanmedian(bkg_ratio)
            print (i,  scale_factor1)
            scale_factor1_list.append(scale_factor1)
            
     
        scale_factor2_list=[]
        for i in range( gp_med_stack.shape[0]):
            section_med =  section2[i]
            bkg_ratio = ((section_med -pedestal2) / bkd_section2)
            q1 = np.nanpercentile(bkg_ratio, 25)
            q2 = np.nanpercentile(bkg_ratio, 50)
            ii = np.where((bkg_ratio > q1) & (bkg_ratio < q2))
            # scale_factor2 = np.nanmedian(bkg_ratio[ii])
            scale_factor2 = np.nanmedian(bkg_ratio)
    
            print (i,  scale_factor2)
            scale_factor2_list.append(scale_factor2)
     
    
        for i in range( gp_med_stack.shape[0]):
            for j in range(256):
                    scaled_bkg[i,j, :step_pos[j]] = bkd_model[j, :step_pos[j]] * scale_factor1_list[i]  + pedestal1
                    scaled_bkg[i,j, step_pos[j]:] = bkd_model[j, step_pos[j]:] * scale_factor2_list[i]  + pedestal2
       
        plt.figure('bkg slope subi')
        for i in range( gp_med_stack.shape[0]):    
       
            plt.plot(i, np.nanmedian(scaled_bkg[i][y1:y2,x1:x2]), 'r+')
            plt.plot(i, np.nanmedian(scaled_bkg[i][y1A:y2A,x1A:x2A]), 'r+')
        
        jexo_dic['scaled_bkg'] = scaled_bkg
        
        #test
        qqq = gp_med_stack - scaled_bkg
        
        for gp_ in range(gp_med_stack.shape[0]):
            fig, axes = plt.subplots(1, 2, figsize=(10, 5))  # 1 row, 2 columns
            fig.suptitle(f'GP {gp_}', fontsize=14)
        
            # --- Pre image ---
            ax = axes[0]
            im0 = ax.imshow(gp_med_stack[gp_], aspect='auto', vmin=0, vmax=100)
            ax.set_title('Pre')
            
            # Draw first red box
            ax.plot([x1, x1], [y1, y2], 'r-')
            ax.plot([x2, x2], [y1, y2], 'r-')
            ax.plot([x1, x2], [y1, y1], 'r-')
            ax.plot([x1, x2], [y2, y2], 'r-')
            
            # Draw second red box
            ax.plot([x1A, x1A], [y1A, y2A], 'r-')
            ax.plot([x2A, x2A], [y1A, y2A], 'r-')
            ax.plot([x1A, x2A], [y1A, y1A], 'r-')
            ax.plot([x1A, x2A], [y2A, y2A], 'r-')
        
            # --- Post image ---
            ax = axes[1]
            im1 = ax.imshow(qqq[gp_], aspect='auto', vmin=0, vmax=100)
            ax.set_title('Post')
        
            # Draw the same boxes on post image
            ax.plot([x1, x1], [y1, y2], 'r-')
            ax.plot([x2, x2], [y1, y2], 'r-')
            ax.plot([x1, x2], [y1, y1], 'r-')
            ax.plot([x1, x2], [y2, y2], 'r-')
        
            ax.plot([x1A, x1A], [y1A, y2A], 'r-')
            ax.plot([x2A, x2A], [y1A, y2A], 'r-')
            ax.plot([x1A, x2A], [y1A, y1A], 'r-')
            ax.plot([x1A, x2A], [y2A, y2A], 'r-')
        
            # Optional: add a colorbar for consistency
            # fig.colorbar(im1, ax=axes.ravel().tolist(), shrink=0.8)
        
            plt.tight_layout()
            plt.show()
         
           
            
    # =============================================================================
    #     6.   remove bkg (and offset), to produce a time series of background-subtracted gp-medians, modulated by the wlc
    #          --- these allow difference images to be produced for 1/f subtraction in pipeline
    # =============================================================================
       
        #there might be other ways to do this like direct subtraction from the gp median
        # here, we first subtract the background from the sci_stack, and then sample 100 integrations in the OOT section
        # this gives a bkg-subtracted median
        # this is then duplicated and modulated by the wlc 
        # the final time series allow us to obtain difference images for 1/f noise in the pipeline
        
        sci_stack = sci_stack - jexo_dic['scaled_bkg'][np.newaxis,:,:,:]
        
        idx1=10;idx2 =110
        sci_stack0 = sci_stack[idx1:idx2]
     
        for gp in range(sci_stack0.shape[1]):
            
            gp_med = np.nanmedian(sci_stack0[:,gp,:,:],axis=0)
            if gp == 0:
                gp_med_stack = gp_med
            else:
                gp_med_stack = np.dstack((gp_med_stack, gp_med))
     
        # A single bkg-subtracted gp median ready for duplication and modulation by the wlc
        gp_med_stack= np.rollaxis(gp_med_stack, 2, 0)
        
        try:
            wlc = np.load('./%s_%s_wlc.npy'%(jexo_dic['obs_number'],jexo_dic['tag']))
            plt.figure('wlc11')
            plt.plot(wlc)
            
            if len(wlc) < sci_stack.shape[0]:
                wlc = np.hstack((wlc[0], wlc))
            
            plt.plot(wlc)
            norm = wlc
     
        except:
            norm = np.ones((sci_stack.shape[0]))  
            print ('=========NO CURRENT WLC FOR SCALING!!!!!!! USING ONES ONLY')
             
    
        # array mimicing the sci_stack
        g_stack = np.zeros_like(sci_stack)
        
        plt.figure('gp median stack')
        
        #now modulate with the wlc
        for i in range(sci_stack.shape[0]):
            gp_med_stack0  =np.copy(gp_med_stack)* norm[i]
            
            # gp_med_stack0  =np.copy(gp_med_stack)
            # print (norm[i], np.sum(gp_med_stack0))
            plt.plot(i, np.sum(gp_med_stack0), 'o')
           
            g_stack[i] = gp_med_stack0 
            
        # g_stack is saved as the bkg-subtracted gp-median time-series modulated by the wlc; needed for difference images
        jexo_dic['g_stack'] =g_stack
     
        print (g_stack.shape)
    
    # so going into pipeline, we need the scaled background and the g_stack.

# =============================================================================
#  
# =============================================================================
            
    if mp== True: 
 
        queue = Queue()
        processes = [Process(target=process_seg, args=(queue, xx, jexo_dic)            
                                                              ) for xx in div_list] 
       
        for p in processes:
            p.start()
    
        for p in processes:
            p.join()
    
        results = [queue.get() for p in processes]
        
    else:
        results =[]

        for i in range(len(div_list)):
            
             jexo_dic['sequence_number'] = i
             jexo_dic['int_start'] = int_start_list[i]
             jexo_dic['int_end'] = int_end_list[i]
             
             # i = 3
             # file = process_seg(1, div_list[i],  root_folder,obs_number, channel, tag, mp)
             
             file = process_seg(1, div_list[i], jexo_dic)
             # xxx
             results.append(file)
            
        
    idx = []
    for i in range(len(results)):
        file = results[i]
        if 'order' in file:
           idx0 = file.find('order')
           for j in range(12):
               idx1 = idx0+j
               if file[idx1]=='_':
                   break
           idx.append(int(file[idx0+5:idx1]))
           
    idx = np.array(idx)
    
    
    sort = np.argsort(idx)
  
    
    results = np.array(results)[sort]
    results = results.tolist()
 
    
    return results
        