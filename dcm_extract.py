#!/usr/bin/python

import sys
import dicom
import os
import numpy as np

path = "./example_dcm_sample"
print path

directories = ['auxiliary','contours','images']
for directory in directories:
  if not os.path.exists(directory):
    os.makedirs(directory)

# Auxiliary file tags
tags=[
['(0008.0008)', (0x0008,0x0008)], #Image Type
#['(0008.0012)', (0x0008,0x0012)], #Instance Creation Date
#['(0008.0013)', (0x0008,0x0013)], #Instance Creation Time
#['(0008.0016)', (0x0008,0x0016)], #SOP Class UID
#['(0008.0018)', (0x0008,0x0018)], #SOP Instance UID
#['(0008.0020)', (0x0008,0x0020)], #Study Date
#['(0008.0021)', (0x0008,0x0021)], #Series Date
#['(0008.0023)', (0x0008,0x0023)], #Content Date
#['(0008.0030)', (0x0008,0x0030)], #Study Time
#['(0008.0031)', (0x0008,0x0031)], #Series Time
#['(0008.0033)', (0x0008,0x0033)], #Content Time
#['(0008.0050)', (0x0008,0x0050)], #Accession Number
['(0008.0060)', (0x0008,0x0060)], #Modality
['(0008.0070)', (0x0008,0x0070)], #Manufacturer
#['(0008.0090)', (0x0008,0x0090)], #Referring Physician's Name
#['(0008.103e)', (0x0008,0x103e)], #Series Description
['(0008.1090)', (0x0008,0x1090)], #Manufacturer's Model Name
#['(0010.0010)', (0x0010,0x0010)], #Patient's Name             
#['(0010.0020)', (0x0010,0x0020)], #Patient ID            
#['(0010.0030)', (0x0010,0x0030)], #Patient's Birth Date        
#['(0010.0040)', (0x0010,0x0040)], #Patient's Sex 
#['(0012.0062)', (0x0012,0x0062)], #Patient Identity Removed
#['(0012.0063)', (0x0012,0x0063)], #De-identification Method
['(0018.0022)', (0x0018,0x0022)], #Scan Options
['(0018.0050)', (0x0018,0x0050)], #Slice Thickness 
['(0018.0060)', (0x0018,0x0060)], #KVP           
['(0018.0090)', (0x0018,0x0090)], #Data Collection Diameter
['(0018.1020)', (0x0018,0x1020)], #Software Version(s)
['(0018.1100)', (0x0018,0x1100)], #Reconstruction Diameter
['(0018.1110)', (0x0018,0x1110)], #Distance Source to Detector
['(0018.1111)', (0x0018,0x1111)], #Distance Source to Patient
['(0018.1120)', (0x0018,0x1120)], #Gantry/Detector Tilt
['(0018.1130)', (0x0018,0x1130)], #Table Height          
['(0018.1140)', (0x0018,0x1140)], #Rotation Direction       
['(0018.1150)', (0x0018,0x1150)], #Exposure Time       
['(0018.1151)', (0x0018,0x1151)], #X-Ray Tube Current        
['(0018.1152)', (0x0018,0x1152)], #Exposure    
['(0018.1160)', (0x0018,0x1160)], #Filter Type    
['(0018.1170)', (0x0018,0x1170)], #Generator Power         
['(0018.1190)', (0x0018,0x1190)], #Focal Spot(s)
['(0018.1210)', (0x0018,0x1210)], #Convolution Kernel
['(0018.5100)', (0x0018,0x5100)], #Patient Position
#['(0020.000d)', (0x0020,0x000d)], #Study Instance UID
#['(0020.000e)', (0x0020,0x000e)], #Series Instance UID
#['(0020.0010)', (0x0020,0x0010)], #Study ID
#['(0020.0011)', (0x0020,0x0011)], #Series Number     
#['(0020.0012)', (0x0020,0x0012)], #Acquisition Number
#['(0020.0013)', (0x0020,0x0013)], #Instance Number
['(0020.0032)', (0x0020,0x0032)], #Image Position (Patient)
['(0020.0037)', (0x0020,0x0037)], #Image Orientation (Patient)
#['(0020.0052)', (0x0020,0x0052)], #Frame of Reference UID
['(0020.1040)', (0x0020,0x1040)], #Position Reference Indicator
['(0020.1041)', (0x0020,0x1041)], #Slice Location
['(0028.0002)', (0x0028,0x0002)], #Samples per Pixel
['(0028.0004)', (0x0028,0x0004)], #Photometric Interpretation
['(0028.0010)', (0x0028,0x0010)], #Rows
['(0028.0011)', (0x0028,0x0011)], #Columns
['(0028.0030)', (0x0028,0x0030)], #Pixel Spacing
#['(0028.0100)', (0x0028,0x0100)], #Bits Allocated
#['(0028.0101)', (0x0028,0x0101)], #Bits Stored
#['(0028.0102)', (0x0028,0x0102)], #High Bit
#['(0028.0103)', (0x0028,0x0103)], #Pixel Representation
['(0028.1050)', (0x0028,0x1050)], #Window Center
['(0028.1051)', (0x0028,0x1051)], #Window Width
['(0028.1052)', (0x0028,0x1052)], #Rescale Intercept
['(0028.1053)', (0x0028,0x1053)], #Rescale Slope
['(0028.1054)', (0x0028,0x1054)], #Rescale Type
#['(7fe0.0010)', (0x7fe0,0x0010)]  #Pixel Data
]

# Gather all the dicom files;
# CT -- contains scan slice images
# RTst -- contains structures
ct_files = []
rtst_files = []
for dir_name, subdir_list, file_list in os.walk(path):
  for filename in file_list:
    if ".dcm" in filename.lower():
      print filename
      if "ct" in dir_name.lower():
        ct_files.append(os.path.join(dir_name,filename))
      if "rtst" in dir_name.lower():
        rtst_files.append(os.path.join(dir_name,filename))

if 1:
  print 'Extracting structure labels...',
  sys.stdout.flush()
  f = open('structure.dat','w')
  sys.stdout.flush()
  for file in rtst_files:
    dcm_data = dicom.read_file(file)
    patient_id = dcm_data[0x10, 0x20].value
    structure_set_roi_sequence = dcm_data[0x3006,0x20].value
    for i, structure_set_roi in enumerate(structure_set_roi_sequence):
      if structure_set_roi[0x3006, 0x0022].value != i+1:
        print 'Script cannot accomodate your dicom data. Exiting now.'
        exit()
    structure_line = patient_id+'|'+'|'.join( [ elem[0x3006,0x26].value for elem in structure_set_roi_sequence ])
    f.write(structure_line+'\n')
  print 'DONE'
  sys.stdout.flush()
  f.close()

if 1:
  print 'Extracting slice data...'
  sys.stdout.flush()
  for file in ct_files:
    dcm_data = dicom.read_file(file)
    patient_id = dcm_data[0x10, 0x20].value
    instance_number = dcm_data[0x20, 0x13].value
    if 1:
      pixel_array = dcm_data.pixel_array
      image_fname = patient_id+'.'+str(instance_number)+'.image'
      print 'Writing pixel array : '+image_fname,
      sys.stdout.flush()
      np.savetxt('images/'+image_fname,pixel_array,delimiter=',',fmt='%d')
      print 'DONE'
      sys.stdout.flush()
    if 1:
      aux_fname = patient_id+'.'+str(instance_number)+'.auxiliary'
      print 'Writing auxiliary information : '+aux_fname,
      f = open('auxiliary/'+aux_fname,'a')
      for elem in tags:
        label, major_minor= elem
        major, minor = major_minor
        try: # Some dcm tags are not present in all data
          val = dcm_data[major, minor].value
        except KeyError:
          val = ""
        if isinstance(val, list):
          f.write(label+','+','.join(str(v) for v in val)+'\n')
        else:
          f.write(label+','+str(val)+'\n')
      print 'DONE'
      sys.stdout.flush()
      f.close()

if 1:
  # In order to correlate structures with images, we need a dictionary
  # which relates image sop_instance_uid with instance_numbers
  print 'Building index...'
  sys.stdout.flush()
  sop_instance_uid_to_instance_number_dict = {}
  for file in ct_files:
    dcm_data = dicom.read_file(file)
    instance_number = dcm_data[0x20, 0x13].value
    sop_instance_uid = dcm_data[0x08, 0x18].value
    sop_instance_uid_to_instance_number_dict[sop_instance_uid] = str(instance_number)
  print 'DONE'
  sys.stdout.flush()
  
  print 'Extracting contour data...'
  sys.stdout.flush()
  for file in rtst_files:
    dcm_data = dicom.read_file(file)
    patient_id = dcm_data[0x10, 0x20].value 
    roi_contour_sequence = dcm_data[0x3006,0x39].value
    for roi_contour in roi_contour_sequence:
      contour_sequence = roi_contour[0x3006,0x40].value
      referenced_roi_number = roi_contour[0x3006, 0x84].value
      for contour in contour_sequence:
        contour_image_sequence = contour[0x3006,0x16].value
        if len(contour_image_sequence)!=1:
          print 'We have a problem'
          sys.stdout.flush()
          exit()
        referenced_sop_instance_uid = contour_image_sequence[0][0x08,0x1155].value
        contour_data = contour[0x3006,0x50].value
        contour_fname = patient_id
        contour_fname += '.'
        contour_fname += sop_instance_uid_to_instance_number_dict[referenced_sop_instance_uid]
        contour_fname += '.'+str(referenced_roi_number)+'.contour'
        f = open('contours/'+contour_fname,'a')
        print 'Writing contour : '+contour_fname,
        sys.stdout.flush()
        f.write(','.join( str(v) for v in contour_data)+'\n')
        f.close()
        print 'DONE'
        sys.stdout.flush()
  print 'DONE'
  sys.stdout.flush()
