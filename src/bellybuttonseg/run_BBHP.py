#######################################################################################
import tensorflow as tf
import numpy as np
import os,time,sys,copy,datetime
from BBHP_function import BBHP

sys.path.append(os.path.dirname(os.path.realpath(__file__))) # Add parent directory to sys.path (for script)
#sys.path.append(os.path.dirname(os.getcwd())) # Add parent directory to sys.path (for notebook)

# define useful variables
param = {} # dictionary that holds all parameters
BBfolder = os.path.abspath(os.getcwd()) 
BBfolder = BBfolder[:BBfolder.index('Bellybutton')+len('Bellybutton')+1]+'/' # Bellybutton folder path
####### END OF HEADER #################################################################

### USER INSTRUCTIONS:

#######################################################################################
# (1) INDICATE MAIN FILEPATH WHERE IMAGES ARE
#######################################################################################

# file path where all subfolders below are located
#filepath = BBfolder + 'Useable_Datasets/GFPGOWT1_mouse_stem_cells/'
#filepath = BBfolder + 'Useable_Datasets/3D_Printed_Disks_1/'
filepath = BBfolder + 'UseableDatasets/printed_foam/'


#######################################################################################
# (2) DEFINE ALL RELEVANT PARAMETERS FOR TRAINING
#######################################################################################

param['S_half'] = 12# int, defines size of input image to NN: size = (2[S_half]+1 square). 12 is reccommended, 9 is minimum

param['scales'] = 2 # number of scaled images (int, 1 is just regular without scaling)
param['scalefactor'] = 2 # scale multiplier (int)
# for example, with 3 scales of scalefactor 4, images scaled by 1, 4^1=4, and 4^2=16 are included as inputs

param['batch_size'] = 1024 # int, must be multiple of 8 to allow rotations + flips. Same size for testing.
param['rotations'] = True # rotate images by 90, 180, 270 degrees in training to multiply data by 4
param['flips'] = True # flip images in training to multiply data by 2

param['images_to_grayscale'] = True; #convert images to grayscale (if images already one channel, this has no effect.)

param['HP_train_epochs'] = 3; # how many epochs to train

# (total AOI area)*fraction = training samples 
# These are automatically adjusted such that half come from each class, meaning some may be multi-counted.
# This does not affect test set.
param['fraction'] = 1



param['HP_network_num'] = 7; # For a list of what networks these numbers create, see readme.



#######################################################################################
# Run Bellybutton!
#######################################################################################


BBHP(filepath,param)


