
"""
    Bit	Value	Name	Description
    0	1	DO_NOT_USE	Bad pixel. Do not use.
    1	2	SATURATED	Pixel saturated during exposure
    2	4	JUMP_DET	Jump detected during exposure
    3	8	DROPOUT	Data lost in transmission
    4	16	RESERVED	 
    5	32	RESERVED	 
    6	64	RESERVED	 
    7	128	RESERVED	 
    8	256	UNRELIABLE_ERROR	Uncertainty exceeds quoted error
    9	512	NON_SCIENCE	Pixel not on science portion of detector
    10	1024	DEAD	Dead pixel
    11	2048	HOT	Hot pixel
    12	4096	WARM	Warm pixel
    13	8192	LOW_QE	Low quantum efficiency
    14	16384	RC	RC pixel
    15	32768	TELEGRAPH	Telegraph pixel
    16	65536	NONLINEAR	Pixel highly nonlinear
    17	131072	BAD_REF_PIXEL	Reference pixel cannot be used
    18	262144	NO_FLAT_FIELD	Flat field cannot be measured
    19	524288	NO_GAIN_VALUE	Gain cannot be measured
    20	1048576	NO_LIN_CORR	Linearity correction not available
    21	2097152	NO_SAT_CHECK	Saturation check not available
    22	4194304	UNRELIABLE_BIAS	Bias variance large
    23	8388608	UNRELIABLE_DARK	Dark variance large
    24	16777216	 UNRELIABLE_SLOPE	Slope variance large (i.e., noisy pixel)
    25	33554432	 UNRELIABLE_FLAT	Flat variance large
    26	67108864 	OPEN	Open pixel (counts move to adjacent pixels)
    27	134217728	ADJ_OPEN	Adjacent to open pixel
    28	268435456	UNRELIABLE_RESET	Sensitive to reset anomaly
    29	536870912	MSA_FAILED_OPEN	Pixel sees light from failed-open shutter
    30	1073741824	OTHER_BAD_PIXEL	A catch-all flag   
    """
    

from multiprocessing import Process, Queue

from numba import jit, prange

import random

import os
os.environ['CRDS_PATH'] ='/Users/c24050258/crds_cache'
os.environ['CRDS_SERVER_URL'] ='https://jwst-crds.stsci.edu'
 
import asdf
import copy
import shutil
import numpy as np 
from astropy.io import fits
import matplotlib.pyplot as plt
import pastasoss
 

# jwst imports 
import jwst
print(jwst.__version__)

 
# The entire calwebb_spec2 pipeline
from jwst.pipeline.calwebb_spec2 import Spec2Pipeline

# Individual steps that make up calwebb_spec2 and datamodels
from jwst.assign_wcs.assign_wcs_step import AssignWcsStep
from jwst.flatfield.flat_field_step import FlatFieldStep
from jwst.extract_1d.extract_1d_step import Extract1dStep
from jwst import datamodels
 

import pipeline_lib
import  NIRISS_stage_2_lib

 
from jwst.stpipe import Step 


@jit(nopython=True, parallel=True)
def flag(sci, dq_):
    
    n_intg = sci.shape[0]
    
    for intg in prange(n_intg):
 
        img = sci[intg]
        dq = dq_[intg]

        # alpha =  3 #sigma clip level
        alpha =  10 #sigma clip level
        
        iter = 3 # number of iterations
 
        for ii in prange(iter):
            bbox = 5 # rolling median of 5 pixels centred on the column i
            local_median = np.zeros((img.shape[0], img.shape[1]))
            local_std = np.zeros((img.shape[0], img.shape[1]))  
            for i in prange(bbox,img.shape[1]-bbox):
                slice_ = img[:,i-bbox:i+bbox+1]
                for j in prange(img.shape[0]):        
            
                    local_std[j][i] = np.nanstd(slice_[j])
                    local_median[j][i] = np.nanmedian(slice_[j])

            for i in range(0,bbox):
                local_median[:,i] = local_median[:,bbox]
                local_std[:,i] = local_std[:,bbox]

            for i in range(img.shape[1]-bbox, img.shape[1]):
                local_median[:,i] = local_median[:,-bbox-1]
                local_std[:,i] = local_std[:,-bbox-1]
  
            line_pc16 = np.zeros(img.shape[0])
            line_pc84 = np.zeros(img.shape[0])
            line_sigma_ = np.zeros(img.shape[0])

            for j in prange(img.shape[0]):
                line_pc16[j] = np.nanpercentile(img[j],16) #  
                line_pc84[j] = np.nanpercentile(img[j],84) #  
                line_sigma_[j] =    (line_pc84[j]-line_pc16[j]) /2
            line_sigma =np.zeros((img.shape[0], img.shape[1]))
            for i in prange(img.shape[1]):
                line_sigma[:,i] =  line_sigma_
                
            # now clip outliers and replace with nans : based on line sigma 
                
            idx = np.argwhere(img > local_median +alpha*line_sigma).T
            
            for j in prange(len(idx[0])):
                img[idx[0][j]][idx[1][j]] = np.nan
                
            idx = np.argwhere(img < local_median -alpha*line_sigma).T

            for j in prange(len(idx[0])):
                img[idx[0][j]][idx[1][j]] = np.nan
                
            # clip again based on the local std
                
            idx = np.argwhere(img > local_median + alpha*local_std).T

            for j in prange(len(idx[0])):
                img[idx[0][j]][idx[1][j]] = np.nan
                
            idx = np.argwhere(img < local_median - alpha*local_std).T
            
            for j in prange(len(idx[0])):
                img[idx[0][j]][idx[1][j]] = np.nan
            
        sci[intg] = img
    
    return sci

@jit(nopython=True, parallel=True)
def get_median2(sci_stack, box):
      
    median_stack = np.zeros_like(sci_stack)
  
    # # seq = np.arange(sci_stack.shape[0])
    for i in prange(sci_stack.shape[0]):

        if i <box:
            box_l = i*1
            box_r = box + 1
        elif i + box+1 > sci_stack.shape[0]-1:
            box_l = box
            box_r = box
        else:
            box_l = box
            box_r = box
            
        for j in range(sci_stack.shape[1]):
            for k in range(sci_stack.shape[2]):
                
                median_stack[i][j][k] = np.median(sci_stack[i-box_l:i+box_r,j,k])

    return median_stack



@jit(nopython=True, parallel=True)
def get_median(sci_stack, var_stack, box, alpha):
      
    median_stack = np.zeros_like(sci_stack)
    median_var_stack = np.zeros_like(sci_stack)
    std_stack = np.zeros_like(sci_stack)
  
    # # seq = np.arange(sci_stack.shape[0])
    for i in prange(sci_stack.shape[0]):

        if i <box:
            box_l = i*1
            box_r = box + 1
        elif i + box+1 > sci_stack.shape[0]-1:
            box_l = box
            box_r = box
        else:
            box_l = box
            box_r = box
   
        for j in range(sci_stack.shape[1]):
            for k in range(sci_stack.shape[2]):
                
                median_stack[i][j][k] = np.median(sci_stack[i-box_l:i+box_r,j,k])
                median_var_stack[i][j][k] = np.median(var_stack[i-box_l:i+box_r,j,k])
                std_stack[i][j][k] = np.std(sci_stack[i-box_l:i+box_r,j,k])

    median_std_ = np.zeros((sci_stack.shape[1], sci_stack.shape[2]))
    for j in prange(sci_stack.shape[1]):
        for k in prange(sci_stack.shape[2]):
            median_std_[j][k] = np.median(std_stack[:,j,k])
    median_std = np.zeros_like(sci_stack)
    for i in prange(sci_stack.shape[0]):
        median_std[i] = median_std_ 
        
    alpha2 = alpha 
    
    sci_stack = np.where(sci_stack>median_stack+alpha*std_stack, median_stack, sci_stack)
    sci_stack = np.where(sci_stack<median_stack-alpha*std_stack, median_stack, sci_stack)
    
    sci_stack = np.where(sci_stack>median_stack+alpha2*median_std, median_stack, sci_stack)
    sci_stack = np.where(sci_stack<median_stack-alpha2*median_std, median_stack, sci_stack)
    
    var_stack = np.where(sci_stack>median_stack+alpha*std_stack, median_var_stack, var_stack)
    var_stack = np.where(sci_stack<median_stack-alpha*std_stack, median_var_stack, var_stack)
    
    var_stack = np.where(sci_stack>median_stack+alpha2*median_std, median_var_stack, var_stack)
    var_stack = np.where(sci_stack<median_stack-alpha2*median_std, median_var_stack, var_stack)

    return sci_stack, var_stack


 
def get_median_f277w(sci_stack, var_stack, box, alpha):
  
    median_stack = np.median(sci_stack, axis=0)
    median_var_stack = np.median(var_stack, axis=0)
    std_stack = np.std(sci_stack, axis=0)
    
    # alpha2 = alpha 
    
    sci_stack = np.where(sci_stack>median_stack+alpha*std_stack, median_stack, sci_stack)
    sci_stack = np.where(sci_stack<median_stack-alpha*std_stack, median_stack, sci_stack)
 
    var_stack = np.where(sci_stack>median_stack+alpha*std_stack, median_var_stack, var_stack)
    var_stack = np.where(sci_stack<median_stack-alpha*std_stack, median_var_stack, var_stack)
 
    return sci_stack, var_stack
    
    
class CustomBadPixelFlagging(Step):
     
    class_alias = "custom_bad_corr"

    spec = """ """
    # reference_file_types = ['superbias']

    def process(self, input, extra_flag):

        # Open the input data model
        with datamodels.CubeModel(input) as input_model:      
            result = self.custom_bad_pixel_correction(input_model,extra_flag)
            input_model.close()
            result.meta.cal_step.custom_bad_corr = 'COMPLETE'
        return result
    
    def custom_bad_pixel_correction(self, input_model, extra_flag):
        
        sci = input_model.data
        dq_ = input_model.dq
 
        # exclude these flags
        dq_ = np.where(dq_ & 2 > 0, dq_ -2, dq_)
        dq_ = np.where(dq_ & 4 > 0, dq_-4, dq_)
        # dq_ = np.where(dq_ & 262144 > 0, dq_-262144, dq_) # NO_FLAT_FIELD
        # dq_ = np.where(dq_ & 33554432 > 0, dq_-33554432, dq_)   #UNRELIABLE_FLAT
        # dq_ = np.where(dq_ & 1 > 0, dq_-1, dq_)  #DO_NOT_USE - EXPERIMENT?
        
        dq2 = dq_*1
        dq2 = np.where(dq2 & 1024 > 0, -1, dq2)
        dq2 = np.where(dq2 & 2048 > 0, -1, dq2)
        dq2 = np.where(dq2 & 4096 > 0, -1, dq2)
        
    
        # 0	1	DO_NOT_USE	Bad pixel. Do not use.
        # 1	2	SATURATED	Pixel saturated during exposure
        # 2	4	JUMP_DET	Jump detected during exposure
        # 3	8	DROPOUT	Data lost in transmission
        # 4	16	RESERVED	 
        # 5	32	RESERVED	 
        # 6	64	RESERVED	 
        # 7	128	RESERVED	 
        # 8	256	UNRELIABLE_ERROR	Uncertainty exceeds quoted error
        # 9	512	NON_SCIENCE	Pixel not on science portion of detector
        # 10	1024	DEAD	Dead pixel
        # 11	2048	HOT	Hot pixel
        # 12	4096	WARM	Warm pixel
        # 13	8192	LOW_QE	Low quantum efficiency
        # 14	16384	RC	RC pixel
        # 15	32768	TELEGRAPH	Telegraph pixel
        # 16	65536	NONLINEAR	Pixel highly nonlinear
        # 17	131072	BAD_REF_PIXEL	Reference pixel cannot be used
        # 18	262144	NO_FLAT_FIELD	Flat field cannot be measured
        # 19	524288	NO_GAIN_VALUE	Gain cannot be measured
        # 20	1048576	NO_LIN_CORR	Linearity correction not available
        # 21	2097152	NO_SAT_CHECK	Saturation check not available
        # 22	4194304	UNRELIABLE_BIAS	Bias variance large
        # 23	8388608	UNRELIABLE_DARK	Dark variance large
        # 24	16777216	 UNRELIABLE_SLOPE	Slope variance large (i.e., noisy pixel)
        # 25	33554432	 UNRELIABLE_FLAT	Flat variance large
        # 26	67108864 	OPEN	Open pixel (counts move to adjacent pixels)
        # 27	134217728	ADJ_OPEN	Adjacent to open pixel
        # 28	268435456	UNRELIABLE_RESET	Sensitive to reset anomaly
        # 29	536870912	MSA_FAILED_OPEN	Pixel sees light from failed-open shutter
        # 30	1073741824	OTHER_BAD_PIXEL	A catch-all flag   
              
        #new way
        bad_flag = np.where(dq_==0,1,np.nan)
        
        # bad_flag = np.where(dq2==-1, np.nan, 1)
 
        sci *= bad_flag  # bad flag applies nans to bad pixel locations
 
        print ("flagging bad pixels...")
        from tqdm import tqdm
        seq = np.arange(sci.shape[0])
        
        if extra_flag==1:
            
            sci  = flag(sci.astype(np.float32), dq_.astype(np.float32))
        else:
            pass
        
        input_model.data = sci   
        return input_model

 

# =============================================================================
#  combining segments here 
# =============================================================================

def run(jexo_dic, stage_1_file_list, channel, nrs,  just_extract=False, tag=''):  
    
      bkd_model = jexo_dic['bkd_model'] 
  
      for file in stage_1_file_list:
    
          rateints_file = file
       
          hdul = hdul = fits.open(rateints_file)
          sci = hdul[1].data
          err = hdul[2].data
          dq = hdul[3].data 
          int_times = hdul[4].data  
          varp = hdul[5].data   
          varr = hdul[6].data  
          
          if file == stage_1_file_list[0]:
              sci_stack = sci
              err_stack = err
              dq_stack = dq
              int_times_stack = int_times
              varp_stack = varp
              varr_stack = varr
       
          else:
              sci_stack = np.vstack((sci_stack, sci))
              err_stack = np.vstack((err_stack, err))
              dq_stack = np.vstack((dq_stack, dq))
              varp_stack = np.vstack((varp_stack, varp))
              varr_stack = np.vstack((varr_stack, varr))
              int_times_stack = np.hstack((int_times_stack, int_times))
         
          hdul.close()
    
      # pick first file as the template 
      hdul = hdul = fits.open(rateints_file)
    
      header = hdul[0].header
      header['EXSEGNUM'] = 1
      header['EXSEGTOT'] = 1
      header['INTSTART'] =  1
      header['INTEND'] = header['NINTS'] # assuming all the files have the same total nints in the header
      hdul[0].header = header
    
      hdul[1].data =  sci_stack
      hdul[2].data =  err_stack
      hdul[3].data =  dq_stack
      hdul[4].data =  int_times_stack
      hdul[5].data =  varp_stack
      hdul[6].data =  varr_stack
      
      bjd  = hdul[4].data['int_mid_BJD_TDB']
  
      outfile0  = rateints_file  
      idx = outfile0.find('.fits')
      aa = outfile0[idx:]
      outfile0  = outfile0.replace(aa, '%s.fits'%(tag))
      
      wlc = np.nansum(np.nansum(sci_stack,axis=2), axis=1)
 
      plt.figure('wlc prelim---')
      plt.plot(wlc)
           
      idx = outfile0.find('seg')
      aa = outfile0[idx:idx+6]
      outfile0  = outfile0.replace(aa, 'COMBINED')  
      # # Write the new HDU structure to outfile
      hdul.writeto(outfile0, overwrite=True)
    
      hdul.close()
    
      rateints_file = outfile0

      if just_extract==True:
          ct = 1
      else:
          ct= 0
          
      if ct==0:
    
          # =============================================================================
          # now run pipeline on combined file 
          # =============================================================================
          result = rateints_file
        
          output_dir = './fits_files/%s'%(channel)

          step = AssignWcsStep()
          step.output_dir = output_dir
          # step.save_results = True
          result = step.run(result)
          
          step = FlatFieldStep()
          result = step.run(result)
          
          idx  = np.isnan(result.data)
          print ('proportion of bad pixels', np.sum(idx)/result.data.size)
 
          step =  CustomBadPixelFlagging()  #flagging
          result = step.run(result, jexo_dic['extra_flag'])
          
          idx  = np.isnan(result.data)
          print ('proportion of bad pixels', np.sum(idx)/result.data.size)
          
          plt.figure('check1')
          plt.imshow(result.data[5], aspect='auto', vmin=0, vmax=20)
          
 
# =============================================================================
# stage 2 bkg subtraction
# ============================================================================= 
          if jexo_dic['f277w']==1:
               mask = np.ones_like(result.data[0])
               buffer0 =20; mask_buffer = np.zeros((buffer0, result.data.shape[2]))
               mask = np.vstack((mask_buffer, mask, mask_buffer))
                       
               try:
                   aa = np.load('./niriss_wav_sol_%s_%s.npy'%(jexo_dic['obs_number'], jexo_dic['tag'])) #order 1
               
               except:
                   pwcpos = result.meta.instrument.pupil_position
                   trace_order1 = pastasoss.get_soss_traces(pwcpos=pwcpos, order='1', interp=True) 
                   x_order1, y_order1, wav_order1 = trace_order1.x, trace_order1.y, trace_order1.wavelength
                   aa = np.vstack((x_order1, y_order1, wav_order1))
                
               xpos = aa[0]; ypos=aa[1];wl=aa[2]  
               ap_hw =41
               # ap_hw =31
               start =int(xpos[0])
               for i in range(0, len(ypos)): #needs to be different for o2
                      ypos0 = ypos[i]
                      ypos0 += buffer0
                      mask[:,i+start][int(np.round(ypos0))-ap_hw: int(np.round(ypos0))+ap_hw+1] = np.nan
    
               mask[:,500:]=1
                    
               plt.figure('mask')
               plt.imshow(mask, aspect = 'auto', vmin=0, vmax = 1)
               
               plt.figure('pre bkg sub')
               plt.imshow(result.data[2], aspect = 'auto', vmin=0, vmax = 20)
 
               for i in range(result.data.shape[0]):
                   img = result.data[i]
                   img_masked = img*mask[buffer0:-buffer0]
                   med = np.nanmedian(img_masked)
                   iqr =   0.5*(np.nanpercentile(img_masked, 84) - np.nanpercentile(img_masked, 16))
                   img_masked = np.where(img>(med+5*iqr), np.nan, img_masked)
                   img_masked = np.where(img_masked<(med-5*iqr), np.nan, img_masked)
 
                   colmed = np.nanmedian(img_masked, axis=0)
                   img = img -colmed
                   result.data[i] = img
                   
               plt.figure('post bkg sub')
               plt.imshow(result.data[2], aspect = 'auto', vmin=0, vmax = 20)    
 
          else:
              
              # x1=350;x2=550;y1=230;y2=250 # standard
              x1=250;x2=500;y1=225;y2=250 # gj 9827 d
              x1A=715;x2A=750;y1A=235;y2A=250
 
              if jexo_dic['obs_number'] == 'jw03557007001_04101_00001':  #toi-1468 c
                    x1=250;x2=500;y1=190;y2=230
                    x1A=715;x2A=760;y1A=210;y2A=220
 
              bkd_model = jexo_dic['bkd_model'] 
       
              section = result.data[:,y1:y2,x1:x2]
              bkd_section = bkd_model[y1:y2,x1:x2]
              
              plt.figure('scaled background box st2')
              plt.imshow(result.data[1], aspect = 'auto',  vmin=0, vmax=50)   
              plt.plot([x1,x1], [y1,y2], 'r-')
              plt.plot([x2,x2], [y1,y2], 'r-')
              plt.plot([x1,x2], [y1,y1], 'r-')
              plt.plot([x1,x2], [y2,y2], 'r-')
              
              section_med = np.nanmedian(section, axis=0)
              bkg_ratio = ((section_med) / bkd_section)
            # Instead of a straight median, use the median of the 2nd
            # quartile to limit the effect of any remaining illuminated
            # pixels.
              q1 = np.nanpercentile(bkg_ratio, 25)
              q2 = np.nanpercentile(bkg_ratio, 50)
              ii = np.where((bkg_ratio > q1) & (bkg_ratio < q2))
              scale_factor1 = np.nanmedian(bkg_ratio[ii])
    
 
              section = result.data[:,y1A:y2A,x1A:x2A]
              bkd_section = bkd_model[y1A:y2A,x1A:x2A]
                     
              plt.figure('scaled background box st2')
              plt.imshow(result.data[1], aspect = 'auto',  vmin=0, vmax=50)   
              plt.plot([x1A,x1A], [y1A,y2A], 'r-')
              plt.plot([x2A,x2A], [y1A,y2A], 'r-')
              plt.plot([x1A,x2A], [y1A,y1A], 'r-')
              plt.plot([x1A,x2A], [y2A,y2A], 'r-')
            
              section_med = np.nanmedian(section, axis=0)
              bkg_ratio = ((section_med) / bkd_section)
                 
              q1 = np.nanpercentile(bkg_ratio, 25)
              q2 = np.nanpercentile(bkg_ratio, 50)
              ii = np.where((bkg_ratio > q1) & (bkg_ratio < q2))
              scale_factor2 = np.nanmedian(bkg_ratio[ii])
   
              if scale_factor1 <0 or scale_factor2 <0:
                  
                  print (scale_factor1, scale_factor2)
                  print ('scale factor error')
                  xxxx
                      
              grad_bkg = np.gradient(bkd_model, axis=1)
              step_pos = np.argmax(grad_bkg[:, 10:-10], axis=1) + 10 - 4
              # Apply differential scaling to either side of step.
              scaled_bkg = np.zeros_like(bkd_model)
              for j in range(256):
                    scaled_bkg[j, :step_pos[j]] = bkd_model[j, :step_pos[j]] * scale_factor1
                    scaled_bkg[j, step_pos[j]:] = bkd_model[j, step_pos[j]:] * scale_factor2
            
              print (scale_factor1, scale_factor2)
 
              plt.figure('pre 1')
              plt.imshow(result.data[1], aspect='auto', vmin =0, vmax =20)
            
              plt.figure('pre 2')
              plt.imshow(result.data[10], aspect='auto', vmin =0, vmax =20)
              
              result.data -=scaled_bkg
 
              plt.figure('post 1')
              plt.imshow(result.data[1], aspect='auto', vmin =0, vmax =20)
            
              plt.figure('post 2')
              plt.imshow(result.data[10], aspect='auto', vmin =0, vmax =20)
              
 
# ============================================================================
#           fill in bad values
# =============================================================================  
          sci_stack = result.data
          var_stack = (result.err)**2
# =============================================================================
#           original method
# =============================================================================
           # fill in time (small numbers of nans in timeline can be filled in)
          x = np.arange(sci_stack.shape[0])
          for i in range(sci_stack.shape[2]):
               for j in range(sci_stack.shape[1]):
                   lc = sci_stack[:, j,i]
                   lc_var = var_stack[:, j,i]
                   # plt.figure('lc')
                   idx = np.argwhere(np.isnan(lc)).T[0]
                   if len(idx) >0 and len(idx) < 0.1*len(x):
                   # if len(idx) >0 and len(idx) < len(x):
                       x_new = np.delete(x,idx)
                       lc_new = np.delete(lc,idx)
                       lc_var_new = np.delete(lc_var,idx)
                       lc_filled = np.interp(x, x_new, lc_new)
                       lc_var_filled = np.interp(x, x_new, lc_var_new)
                       # plt.plot(lc_filled, 'ro-')
                       # plt.plot(lc, 'bo-')
                       sci_stack[:, j,i] = lc_filled
                       var_stack[:, j,i] = lc_var_filled
                   # if more than 10% missing then treat whole timeline as bad and make nan 
                   if len(idx) >0 and len(idx) >= 0.1*len(x) and len(idx)!= len(x):
                       sci_stack[:, j,i] = np.nan

# =============================================================================
#                 spatial filling 
# =============================================================================
          # now deal with remaining nans including all the way through: have to filled spatially.
          x = np.arange(sci_stack.shape[2])
          for k in range(sci_stack.shape[0]):
               for i in range(sci_stack.shape[1]):
                   # print (k,i)
   
                   row = sci_stack[k][i]
                   row0 = row*1
                   row_var = var_stack[k][i]
                   row0_var = row_var*1
         
                   idx = np.argwhere(np.isnan(row)).T[0]
                   
                   if len(idx) == len(row):
                       print ('intg %s row %s not filled full of nans'%(k,i))
                       pass
                   
                   elif len(idx)>0:
                       x_new = np.delete(x,idx)
                       row_new = np.delete(row, idx)
                       row_var_new =  np.delete(row_var, idx)
                       row_filled = np.interp(x, x_new, row_new)
                       row_var_filled = np.interp(x, x_new, row_var_new)
                      
                       sci_stack[k][i] = row_filled
                       var_stack[k][i] = row_var_filled
                      
                       # plt.plot(row_filled, 'ro-')
                       # plt.plot(row, 'bo-')
                      
                       if k==0 and i ==12:

                           plt.figure('row')
                           plt.plot(row_filled, 'ro-')
                           plt.plot(row0, 'bo')
                           plt.figure('row var')
                           plt.plot(row_var_filled, 'ro-')
                           plt.plot(row0_var, 'bo') 
          result.data = sci_stack
          result.err = var_stack**0.5
          

# =============================================================================
#  replace outliers with median values 
# =============================================================================
          box = 10
          print ("getting median")
          alpha= 5 # this gives the right amount of sensiticity for median step
          
          if jexo_dic['f277w']==0:
              sci_stack, var_stack = get_median(sci_stack.astype(np.float32), 
                                             var_stack.astype(np.float32),
                                             box, alpha)   
          else:
              sci_stack, var_stack = get_median_f277w(sci_stack.astype(np.float32), 
                                             var_stack.astype(np.float32),
                                             box, alpha)  
               
          result.data = sci_stack
          result.err = var_stack**0.5
          
          level =2000
          plt.figure('after interp')
          plt.imshow(result.data[1], aspect='auto', vmin=0, vmax =level)
          
# =============================================================================
#         save f277w
# =============================================================================

          if jexo_dic['f277w']==True:
              print ('making F277w image')
              level =20

              med_img = np.nanmedian(result.data, axis =0)
              plt.figure('f277w median after interp')
              plt.imshow(med_img, aspect='auto', vmin=0, vmax =level)

              for i in range(med_img.shape[1]):
                    col = med_img[:,i]
                    # col = med[:,1243]
                    # plt.plot(col, 'bo-') 
                    med =[]; idx=[]
                    box = 5
                    for j in range(box,len(col)-box):
                        rm = np.median(col[j-box:j+box+1])
                        med.append(rm)
                        idx.append(j)
                    med = np.array(med)
                    med = np.hstack((np.array([med[0]]*box), med, np.array([med[-1]]*box)))
                    # plt.plot(med, 'g-')
                    # idx = np.argwhere((col>(col2+col2*0.9)) & (col>0.5))
                    idx = np.argwhere((col>(med+med*5)) & (col>0.5))
                    # plt.plot(idx,col[idx], 'ro')  
                    col[idx] = med[idx]
                    # plt.plot(idx,col[idx], 'yo')
                    med_img[:,i]=col
              plt.figure('median after interp 2')
              plt.imshow(med_img, aspect='auto', vmin=0, vmax =level) 
              
              np.save('./f277w_%s_%s'%(jexo_dic['obs_number'], jexo_dic['tag']), med_img)

# =============================================================================
#       final stages
# =============================================================================
          pwcpos = result.meta.instrument.pupil_position
          trace_order1 = pastasoss.get_soss_traces(pwcpos=pwcpos, order='1', interp=True) 
          x_order1, y_order1, wav_order1 = trace_order1.x, trace_order1.y, trace_order1.wavelength
          # x_order1 = x_order1 - 157
          # y_order1 = y_order1 - 12
          
          aa = np.vstack((x_order1, y_order1, wav_order1))
          np.save('./niriss_wav_sol_%s_%s.npy'%(jexo_dic['obs_number'], jexo_dic['tag']), aa)
          
          trace_order2 = pastasoss.get_soss_traces(pwcpos=pwcpos, order='2', interp=True) 
          x_order2, y_order2, wav_order2 = trace_order2.x, trace_order2.y, trace_order2.wavelength
          # x_order2 = x_order2 - 157
          # y_order2 = y_order2 - 12
          
          aa = np.vstack((x_order2, y_order2, wav_order2))
          np.save('./niriss_wav_sol2_%s_%s.npy'%(jexo_dic['obs_number'],  jexo_dic['tag']), aa)

          rows = result.data.shape[1]
          wav_order1 = np.tile(wav_order1, (rows, 1))
          
          result.wavelength =  wav_order1
          
      
   
          file_name = rateints_file.replace('rateints', 'calints')
          result.save(file_name)  
           
         
          # # note the rateints file will not have a record of the custom background subtraction step
          
          
          
          aa = np.load('./niriss_wav_sol_%s_%s.npy'%(jexo_dic['obs_number'], jexo_dic['tag'])) #order 1
          xpos = aa[0]; ypos=aa[1];wl=aa[2]  
          # print (wl, len(wl))
          # level =20
          # ap_hw =12
          # # ap_hw =17
          # ap_hw =21
          # start =int(xpos[0])
          # mask = np.zeros((256,2048), dtype=bool)
          # for i in range(0, len(ypos)): #needs to be different for o2
          #     mask[:,i+start][int(np.round(ypos[i]))-ap_hw: int(np.round(ypos[i]))+ap_hw+1] = 1
          # plt.figure()
          # plt.imshow(mask, aspect='auto')

          img = result.data[1]
          plt.figure('img wav')
          plt.imshow(img, aspect='auto', vmin =0, vmax= 10)
          plt.plot(xpos,ypos, 'r--', lw=2)
          idx_ = np.arange(10,2000,100)
          for idx in idx_:
              plt.axvline(x=xpos[idx], color='w', linestyle='--')
              # plt.text(xpos[idx], ypos[idx], f'x={wl[idx]}', color='r', rotation=90, va='bottom')
              plt.text(xpos[idx], ypos[idx]-10, f'{np.round(wl[idx], 4 )}', color='w', rotation=90, va='bottom')
          aa = np.load('./niriss_wav_sol2_%s_%s.npy'%(jexo_dic['obs_number'],  jexo_dic['tag'])) #order 2
          xpos = aa[0]; ypos=aa[1];wl=aa[2]  
          plt.plot(xpos,ypos, 'r--', lw=2)
          idx_ = np.arange(10,1800,100)
          for idx in idx_:
              plt.axvline(x=xpos[idx], color='b', linestyle='--')
              # plt.text(xpos[idx], ypos[idx], f'x={wl[idx]}', color='r', rotation=90, va='bottom')
              plt.text(xpos[idx], ypos[idx]-10, f'{np.round(wl[idx], 4 )}', color='b', rotation=90, va='bottom')
          
          if jexo_dic['f277w']==1:
              return 0
          
          ct = 1
          
          
# =============================================================================
#      stage 2 b
# =============================================================================
        
      if ct ==1:
           
          file_name = rateints_file.replace('rateints', 'calints')

          hdul =  fits.open(file_name)
          sci = hdul[1].data
          dq =hdul[3].data
          err = hdul[2].data
          var_rn =  hdul[7].data
          wav = hdul[4].data
          int_times =  hdul[5].data
          
          aa= np.load('./niriss_wav_sol_%s_%s.npy'%(jexo_dic['obs_number'], jexo_dic['tag']))    
          aa2 = np.load('./niriss_wav_sol2_%s_%s.npy'%(jexo_dic['obs_number'],  jexo_dic['tag']))
                  
          # +157 to aa's for LHS 1140b
          xpos = aa[0]; ypos=aa[1];wl=aa[2]  
          xpos2 = aa2[0]; ypos2=aa2[1];wl2=aa2[2]  
          
          rows = sci.shape[1]
          wav1 = np.tile(wl, (rows, 1))
          wav2 = np.tile(wl2, (rows, 1))

          level =20
          
          plt.figure('wavelength soln')
          plt.imshow(sci[0], aspect='auto', vmin=0, vmax =level)  
          plt.plot(xpos,ypos, 'r--')
          plt.imshow(sci[0], aspect='auto', vmin=0, vmax =level)  
          plt.plot(xpos2,ypos2, 'r--')
      
          for i in range(0, len(wl), 150):
              # if np.round(wl[i] * 1) % 1 == 0:  # Check if wavelength is a multiple of 0.1
                  plt.annotate(f"{wl[i]:.3f}", (xpos[i], ypos[i]), textcoords="offset points", xytext=(0, 10), ha='center', color='r', rotation=90)
          
          for i in range(0, len(wl2), 150):
              # if np.round(wl[i] * 1) % 1 == 0:  # Check if wavelength is a multiple of 0.1
                  plt.annotate(f"{wl2[i]:.3f}", (xpos2[i], ypos2[i]), textcoords="offset points", xytext=(0, 10), ha='center', color='r', rotation=90)

          plt.xlabel('X')
          plt.ylabel('Y')
          
# =============================================================================
#           remove zeroth order contaminants
# =============================================================================          
          if jexo_dic['subtract_zeroth']==1:
              # find f277 calints file
              aa= file_name
              idx = aa.find('/jw')
              ss  = aa[idx+15:idx+20]
              aa = aa[:idx+19] + '2_' + aa[idx + 20+ 1:]
              idx = aa.find('order')
              idx2 = aa.find('nis')
              x = aa[idx+5:idx2]
              aa = aa.replace(x, '0_')
              
               
              plt.figure('before 0th sub')
              plt.imshow(sci[0], aspect='auto', vmin=0, vmax = 20)
               
              plt.figure('before 0th sub 2')
              plt.imshow(sci[100], aspect='auto', vmin=0, vmax = 20)
    
              sci = NIRISS_stage_2_lib.zeroth_sub(sci, jexo_dic, aa)
              
               
              plt.figure('after 0th sub')
              plt.imshow(sci[0], aspect='auto', vmin=0, vmax = 20)
               
              plt.figure('after 0th sub 2')
              plt.imshow(sci[100], aspect='auto', vmin=0, vmax = 20)
              
        
# =============================================================================
#          order separation
# =============================================================================          
          if jexo_dic['sep_spectra'] == 1:
                  
                   sci_o1,sci_o2 = NIRISS_stage_2_lib.sep_spectra(sci, jexo_dic)
                   
                   sci_o1 = np.load('./sci_o1_%s_%s.npy'%(jexo_dic['obs_number'], jexo_dic['tag']))
                   sci_o2 = np.load('./sci_o2_%s_%s.npy'%(jexo_dic['obs_number'], jexo_dic['tag']))
                   
                   plt.figure('sci_o1')
                   plt.imshow(sci_o1[10], aspect='auto', vmax=50)
                   
                   plt.figure('sci_o2')
                   plt.imshow(sci_o2[10], aspect='auto', vmax=50)

        
          if jexo_dic['atoca'] == 1:
              result = file_name
              step = Extract1dStep()
              result = step.run(result)
              result.save('./atoca_separation_%s_%s.fits'%(jexo_dic['obs_number'], jexo_dic['tag']))

              
# =============================================================================
#                apply a mask with given aperture
# =============================================================================          
          # ---- only for LHS 1140b ----
          # buffer = np.zeros((sci.shape[0],sci.shape[1],157))
          # sci = np.dstack((buffer, sci))
          # err = np.dstack((buffer, err))
          # dq = np.dstack((buffer, dq))
          # var_rn = np.dstack((buffer, var_rn))
          

          # ap_hw =20 #more standard - might be better for opt extract?
          ap_hw =18 #compromise (most people seem to do a box extraction)

          
          #make aperture mask for o1
              
          start = int(xpos[0])
          mask_y_max=[]
          mask_y_min = []
           
          mask = np.zeros_like(sci[0])
          for i in range(0, len(ypos)): #needs to be different for o2
            if i+start<0:
                mask[:,i+start][int(np.round(ypos[i]))-ap_hw: int(np.round(ypos[i]))+ap_hw+1] = 0
            else:
                mask[:,i+start][int(np.round(ypos[i]))-ap_hw: int(np.round(ypos[i]))+ap_hw+1] = 1    
            mask_y_min.append(int(np.round(ypos[i]))-ap_hw)
            mask_y_max.append(int(np.round(ypos[i]))+ap_hw+1)
            
          ap_mask = mask*1
          
          #make aperture mask for o2
          
          start2 = int(xpos2[0])
          mask_y_max2=[]
          mask_y_min2 = []
           
          mask2 = np.zeros_like(sci[0])
          for i in range(0, len(ypos2)): #needs to be different for o2
            if i+start2<0:
                mask2[:,i+start2][int(np.round(ypos2[i]))-ap_hw: int(np.round(ypos2[i]))+ap_hw+1] = 0
            else:
                mask2[:,i+start2][int(np.round(ypos2[i]))-ap_hw: int(np.round(ypos2[i]))+ap_hw+1] = 1    
            mask_y_min2.append(int(np.round(ypos2[i]))-ap_hw)
            mask_y_max2.append(int(np.round(ypos2[i]))+ap_hw+1)
            
          ap_mask2 = mask2*1
              
          plt.figure('aperture mask o1')
          plt.imshow(mask, aspect='auto', vmin=0, vmax =level)  
          plt.plot(xpos,ypos, 'r--')
          plt.plot(xpos,mask_y_min, 'w--')
          plt.plot(xpos,mask_y_max, 'w--')
          
          plt.figure('aperture mask o2')
          plt.imshow(mask2, aspect='auto', vmin=0, vmax =level)  
          plt.plot(xpos2,ypos2, 'r--')
          plt.plot(xpos2,mask_y_min2, 'w--')
          plt.plot(xpos2,mask_y_max2, 'w--')
          
          plt.figure('apertures')
          plt.imshow(sci[0], aspect='auto', vmin=0, vmax =level)         
          plt.plot(xpos,ypos, 'r--')
          plt.plot(xpos,mask_y_min, 'w--')
          plt.plot(xpos,mask_y_max, 'w--')
          plt.plot(xpos2,ypos2, 'r--')
          plt.plot(xpos2,mask_y_min2, 'w--')
          plt.plot(xpos2,mask_y_max2, 'w--')
          
          if jexo_dic['sep_spectra'] == 1:
              sci = sci_o1*1
              sci2 = sci_o2*1
          else:
              sci2= sci*1

          plt.figure('mask order 1')
          plt.imshow(mask, aspect='auto')
          
          level = 200
                    
          plt.figure('pre mask')
          plt.imshow(sci[0], aspect='auto', vmin=0, vmax =level)  
          
          dq_orig = np.copy(dq)
          err_orig = np.copy(err)
          var_rn_orig = np.copy(var_rn)
          
          sci = sci * mask[np.newaxis, :, :]
          err = err_orig * mask[np.newaxis, :, :]
          dq = dq_orig * mask[np.newaxis, :, :]
          var_rn = var_rn_orig * mask[np.newaxis, :, :]
  
          plt.figure('post mask')
          plt.imshow(sci[0], aspect='auto',vmin=0, vmax =level)  
          
                  
          plt.figure('pre mask2')
          plt.imshow(sci2[0], aspect='auto', vmin=0, vmax =level)  
        
          sci2 = sci2 * mask2[np.newaxis, :, :]
          err2 = err_orig * mask2[np.newaxis, :, :]
          dq2 = dq_orig * mask2[np.newaxis, :, :]
          var_rn2 = var_rn_orig * mask2[np.newaxis, :, :]

          plt.figure('post mask2')
          plt.imshow(sci2[0], aspect='auto',vmin=0, vmax =level)  
          
              
# =============================================================================
#          aperture application n/a 
          # wavelength solution covers only x pixels 4:2043 (0 = first) 
              
          sci = sci[:,:,4:2044]
          err = err[:,:,4:2044]
          dq = dq[:,:,4:2044]
          var_rn = var_rn[:,:,4:2044]

          sci2 = sci2[:,:,0:len(wl2)]
          err2 = err2[:,:,0:len(wl2)]
          dq2 = dq2[:,:,0:len(wl2)]
          var_rn2 = var_rn2[:,:,0:len(wl2)]
          
          plt.figure('o1')
          plt.imshow(sci[0], aspect='auto',vmin=0, vmax =level)  
          
          plt.figure('o2')
          plt.imshow(sci2[0], aspect='auto',vmin=0, vmax =level) 
          
# =============================================================================
#         apply aperture
# =============================================================================

            # if jexo_dic['order_2'] != 1:  #rectification fails below 0.59 microns for ap of 12 since reaches edge of array; will not bother (not use op extract)for o2
          ap_lim = ap_hw
          new_sci = np.zeros((sci.shape[0], ap_lim*2+1, sci.shape[2]))
          new_dq = np.zeros((sci.shape[0], ap_lim*2+1, sci.shape[2]))
          new_err  = np.zeros((sci.shape[0], ap_lim*2+1, sci.shape[2]))
          new_var_rn = np.zeros((sci.shape[0], ap_lim*2+1, sci.shape[2]))
          new_wav = np.zeros((ap_lim*2+1, sci.shape[2]))
      
          trace_y=[]
          for i in range(len(ypos)):
              trace_y.append(int(np.round(ypos[i])))
                    
          trace_y = np.array(trace_y)
          for i in range(sci.shape[2]):
                # print (i, trace_y[i], trace_y[i]-ap_lim, trace_y[i]+ap_lim+1, wl[i])
                # print (new_sci[:,:,i].shape, sci[:,trace_y[i]-ap_lim: trace_y[i]+ap_lim+1,i].shape)
                new_sci[:,:,i] = sci[:,trace_y[i]-ap_lim: trace_y[i]+ap_lim+1, i ]
                new_dq[:,:,i] = dq[:,trace_y[i]-ap_lim: trace_y[i]+ap_lim+1, i ]
                new_err[:,:,i] = err[:,trace_y[i]-ap_lim: trace_y[i]+ap_lim+1, i ]
                new_var_rn[:,:,i] = var_rn[:,trace_y[i]-ap_lim: trace_y[i]+ap_lim+1, i ]
                new_wav[:,i] = wav1[trace_y[i]-ap_lim: trace_y[i]+ap_lim+1, i ]

          plt.figure('rectified with aperture')
          plt.imshow(new_sci[0], aspect='auto')
            
          sci = new_sci 
          dq = new_dq
          err = new_err
          var_rn = new_var_rn
          wav = new_wav
          
# =============================================================================
#           now for o2
# =============================================================================
          #to allow for ap to fit at shorter wavelength will cut array by 130
          
          sci2 = sci2[:,:,:1700]

          ap_lim2 = ap_hw
          new_sci2 = np.zeros((sci2.shape[0], ap_lim2*2+1, sci2.shape[2]))
          new_dq2 = np.zeros((sci2.shape[0], ap_lim2*2+1, sci2.shape[2]))
          new_err2  = np.zeros((sci2.shape[0], ap_lim2*2+1, sci2.shape[2]))
          new_var_rn2 = np.zeros((sci2.shape[0], ap_lim2*2+1, sci2.shape[2]))
          new_wav2 = np.zeros((ap_lim2*2+1, sci2.shape[2]))
        
          trace_y2=[]
          for i in range(len(ypos2)):
            trace_y2.append(int(np.round(ypos2[i])))
                  
          trace_y2 = np.array(trace_y2)
          for i in range(sci2.shape[2]):
              # print (i, trace_y[i], trace_y[i]-ap_lim, trace_y[i]+ap_lim+1, wl[i])
              # print (new_sci[:,:,i].shape, sci[:,trace_y[i]-ap_lim: trace_y[i]+ap_lim+1,i].shape)
              new_sci2[:,:,i] = sci2[:,trace_y2[i]-ap_lim2: trace_y2[i]+ap_lim2+1, i ]
              new_dq2[:,:,i] = dq2[:,trace_y2[i]-ap_lim2: trace_y2[i]+ap_lim2+1, i ]
              new_err2[:,:,i] = err2[:,trace_y2[i]-ap_lim2: trace_y2[i]+ap_lim2+1, i ]
              new_var_rn2[:,:,i] = var_rn2[:,trace_y2[i]-ap_lim2: trace_y2[i]+ap_lim2+1, i ]
              new_wav2[:,i] = wav2[trace_y2[i]-ap_lim2: trace_y2[i]+ap_lim2+1, i ]
    
                  
          plt.figure('rectified with aperture o2')
          plt.imshow(new_sci2[0], aspect='auto')
          
          sci2 = new_sci2 
          dq2 = new_dq2
          err2 = new_err2
          var_rn2 = new_var_rn2
          wav2 = new_wav2
          
          
# =============================================================================
#        wavelengths after aperture application
# =============================================================================
          
          wl_array = np.nanmean(wav, axis=0) # opt extact assumes a constant wav per column
          wl_std = np.nanstd(wav, axis=0)
          
          plt.figure('wavelength1')
          plt.plot(wl_array, 'ro-')
          
          wl_array2 = np.nanmean(wav2, axis=0) # opt extact assumes a constant wav per column
          wl_std2 = np.nanstd(wav2, axis=0)
          
          plt.figure('wavelength2')
          plt.plot(wl_array2, 'ro-')
          
          
# =============================================================================
#        ATOCA
# =============================================================================        
          if jexo_dic['atoca'] == 1:
              hdul =  fits.open('./atoca_separation_%s_%s.fits'%(jexo_dic['obs_number'], jexo_dic['tag']))
              o1 = hdul[3].data
              o2 = hdul[4].data

              flux1 = []
              flux2 = []

              err1 = []
              err2 = []

              wav1 = o1[0][1][4:2044]
              wav2 = o2[0][1][0:len(wl2)]
              wav_all = [wav1, wav2]
              
              wlc_list =[]

              for i in range(len(o1)):
                  flux1.append(o1[i][2][4:2044])
                  flux2.append(o2[i][2][0:len(wl2)])
                  
                  err1.append(o1[i][3][4:2044])
                  err2.append(o2[i][3][0:len(wl2)])
              
              flux_all = [np.stack(flux1), np.stack(flux2)]
              flux_all[0][np.isnan(flux_all[0])] = flux_all[1][np.isnan(flux_all[1])] = 0
              err_all = [np.stack(err1), np.stack(err2)]
              err_all[0][np.isnan(err_all[0])] = err_all[1][np.isnan(err_all[1])] = 0

              for ii in range(2):
                  
                  wlc = np.nansum(flux_all[ii], axis=1)
                  wlc_list.append(wlc)
                  
                  n = np.arange(100.0)
                  hdu= fits.PrimaryHDU(n)
                  hdul = fits.HDUList([hdu])
                  table_hdu  = fits.BinTableHDU(data=int_times)
                  hdul.append(table_hdu)
                  hdul[1].header['EXTNAME']= 'INT_TIMES'
        
                  hdul.append(fits.ImageHDU(np.ones(10)))
                  hdul[2].header['EXTNAME']= 'SPEC'
                  hdul[2].data= flux_all[ii]
             
                  hdul.append(fits.ImageHDU(np.ones(10)))
                  hdul[3].header['EXTNAME']= 'WAV'
                  hdul[3].data= wav_all[ii]
             
                  hdul.append(fits.ImageHDU(np.ones(10)))
                  hdul[4].header['EXTNAME']= 'ERR'
                  hdul[4].data=  err_all[ii]**0.5
                      
                  if ii==0:
                      filename = rateints_file.replace('rateints','1Dspec_atoca_extract_o1')
                  else:
                      filename = rateints_file.replace('rateints','1Dspec_atoca_extract_o2')
                  
                  idx = filename.find('.fits')
                  filename = filename[:idx]+'%s.fits'%(tag)
               
                  hdul.writeto(filename, overwrite=True)
                  
              
          
# =============================================================================
#          optimal extraction - need to rectify images - in general do not use for NIRISS
# =============================================================================
          else:
              wlc_list =[]
              for ii in range(2):  # loop through o1 and o2
                  if ii == 0:
                      sci = sci*1
                      err = err*1
                      dq = dq*1
                      wl_array = wl_array*1
                      
                  else:
                      sci = sci2*1
                      err = err2*1
                      dq = dq2*1
                      wl_array = wl_array2*1
                      
                      
             
                  flux_array = np.zeros((sci.shape[0], sci.shape[2]))
                  flux_var_array = np.zeros((sci.shape[0], sci.shape[2]))
            
                  from tqdm import tqdm
                  seq = np.arange(sci.shape[0])
               
                  print ('extracting 1D spectra with optimal extraction')
               
                # median = np.median(sci, axis =0)  # use corrected wavecorr images > median as the probability profile
         
                  box = 6  
                  median = get_median2(sci.astype(np.float32), box)
                 
                  plt.figure('median')
                  plt.imshow(median[0], aspect = 'auto')
              
                  # if omit_badcorr == 1:
                  #   sci_ = sci_omit*1
                  #   err_ = err_omit*1
                  #   dq_ = dq_omit*1
                  # else:
                  #   sci_ = sci*1
                  #   err_ = err*1
                  #   dq_ = dq*1 
        
                  for intg in tqdm(seq):
        
                    img = sci[intg]
                    err0 = err[intg]
                    dq0 = dq[intg]
                  
                    spec, var_spec, chi_sq = pipeline_lib.opt_extract(img, err0, dq0, median[intg], intg)
                   
                    plt.figure('opt extract spec')
                    plt.plot(spec)
                 
                    flux_array[intg] = spec
                    flux_var_array[intg] = var_spec
                  
                  n = np.arange(100.0)
                  hdu= fits.PrimaryHDU(n)
                  hdul = fits.HDUList([hdu])
                  table_hdu  = fits.BinTableHDU(data=int_times)
                  hdul.append(table_hdu)
                  hdul[1].header['EXTNAME']= 'INT_TIMES'
        
                  hdul.append(fits.ImageHDU(np.ones(10)))
                  hdul[2].header['EXTNAME']= 'SPEC'
                  hdul[2].data= flux_array
             
                  hdul.append(fits.ImageHDU(np.ones(10)))
                  hdul[3].header['EXTNAME']= 'WAV'
                  hdul[3].data= wl_array
             
                  hdul.append(fits.ImageHDU(np.ones(10)))
                  hdul[4].header['EXTNAME']= 'ERR'
                  hdul[4].data=  flux_var_array**0.5
                      
                  if ii==0:
                      filename = rateints_file.replace('rateints','1Dspec_opt_extract_o1')
                  else:
                      filename = rateints_file.replace('rateints','1Dspec_opt_extract_o2')
                  
                  idx = filename.find('.fits')
                  filename = filename[:idx]+'%s.fits'%(tag)
               
                  hdul.writeto(filename, overwrite=True)
                  
                 
         
        # =============================================================================
        #         box extraction
        # =============================================================================
         
         
                  flux_array =  np.nansum(sci,axis=1)
                  flux_var_array =  np.nansum((err**2), axis=1)
                  wl_grid = wl_array
          
                  
                  wlc = np.nansum(flux_array, axis=1)
                  
                  wlc_list.append(wlc)
        
                  plt.figure('wlc %s'%(ii))
                  plt.plot(wlc, '.')
                  plt.show()
                  
         
           
                  n = np.arange(100.0)
                  hdu= fits.PrimaryHDU(n)
                  hdul = fits.HDUList([hdu])
                  table_hdu  = fits.BinTableHDU(data=int_times)
                  hdul.append(table_hdu)
                  hdul[1].header['EXTNAME']= 'INT_TIMES'
             
                  hdul.append(fits.ImageHDU(np.ones(10)))
                  hdul[2].header['EXTNAME']= 'SPEC'
                  hdul[2].data= flux_array
                
                  hdul.append(fits.ImageHDU(np.ones(10)))
                  hdul[3].header['EXTNAME']= 'WAV'
                  hdul[3].data= wl_grid
                  
                  hdul.append(fits.ImageHDU(np.ones(10)))
                  hdul[4].header['EXTNAME']= 'ERR'
                  hdul[4].data=  flux_var_array**0.5
           
                  if ii==0:
                      filename = rateints_file.replace('rateints','1Dspec_box_extract_o1')
                  else:
                      filename = rateints_file.replace('rateints','1Dspec_box_extract_o2')
                  
                    
                  idx = filename.find('.fits')
                  filename = filename[:idx]+'%s.fits'%(tag)
                
                  hdul.writeto(filename, overwrite=True)
                 
    
      return wlc_list     
           
    
       