import tensorflow as tf
import numpy as np
import cv2
from skimage.segmentation import watershed
from DataGenerator import circ_region
from generate_network import generate_network
from scipy.ndimage import maximum_filter

#Dictionary key for text and variable translation for image_dict, used in SegmentPrediction, UpdateAccTable, SaveImage and SaveMatrix functions
possible_prediction_names = {'output_binarized':'outmask','output_classprob':'mask','output_dist':'dist','output_segmented':'labels','output_markers':'markers'}

def convert_param_to_outputs(param):
    possible_prediction_names = {'output_binarized':'outmask','output_classprob':'mask','output_dist':'dist','output_segmented':'labels','output_markers':'markers'}
    to_save = []
    #Iterate through param variables
    for key in param:
        #Find the variables related to output formatting
        if key in possible_prediction_names.keys():
            #Select only the desired formats for output
            if param[key]:
                #remove the 'output' tag from the string and add to final list
                to_save.append(key[6:])
    return to_save


#Given a prediction and error functions, segment and watershed the prediction while updating the error count
def SegmentPrediction(BBout,dist_max):
    '''
    inputs:
    BBout = single prediction from BBpredictions
    dist_max = parameter from the generator, max value network predicts for distance to outie
    outputs:
    image_dict = dictionary containing segmented matrix and any other desired outputs
    ASSUME BACKGROUND IS 0
    '''
    kernel = np.array(circ_region(dist_max),dtype=np.uint8)
    dist = BBout[:,:,1]
    mask = BBout[:,:,0]
    outmask = BBout[:,:,0]>0.5

    # find points that are highest in their neighborhood
    maxima = dist == maximum_filter(dist, footprint=kernel) 
    maxima = np.array(maxima*outmask, dtype=np.uint8)
    maxima[dist<dist_max/2] = 0
    
    # Old way of doing things. Obsolete.
    if False:
        maxima[dist>dist_max-1] = 1

        #connectedmask = np.ubyte(np.array(cv2.filter2D(dist,-1,kernel)>=(np.sum(kernel)*(dist_max-0.5)),dtype=float))
        (_, markers, _, _) = cv2.connectedComponentsWithStats(maxima)

        labels = watershed(-dist, markers = markers, mask=outmask)

    if True:

        # give each a unique ID
        #connectedmask = np.ubyte(np.array(cv2.filter2D(dist,-1,kernel)>=(np.sum(kernel)*(dist_max-0.5)),dtype=float))
        (_, markers, _, _) = cv2.connectedComponentsWithStats(maxima)

        # watershed using these maxima, only on classprob>0.5, and using distance as rate
        labels = watershed(-dist, markers = markers, mask=outmask)
        
        # watershedding creates too many segments, find those that should be combined
        S = np.shape(labels)
        combinemat = np.zeros((np.max(labels)+1,np.max(labels)+1)) # catalog which regions should be combined

        # for each pixel jump right or down, if sum of dist of both pixels is too high, these regions are the same
        for r in range(0,S[0]-2):
            for c in range(0,S[1]-2):
                if labels[r,c]!=0: 
                    if labels[r,c+1] !=0 and labels[r,c+1]!=labels[r,c]:
                        combinemat[labels[r,c],labels[r,c+1]] += (dist[r,c]+dist[r,c+1])>(1.5*(dist_max))
                    if labels[r+1,c] !=0 and labels[r+1,c]!=labels[r,c]:
                        combinemat[labels[r,c],labels[r+1,c]] += (dist[r,c]+dist[r+1,c])>(1.5*(dist_max))

        # don't know which was first
        combinemat += np.transpose(combinemat)

        # relabel lower numbered regions with higher number if combining is needed
        S2= np.shape(combinemat)
        for r1 in range(1,S2[0]):
            for r2 in range(r1+1,S2[1]):
                if combinemat[r1,r2]: # combine these regions
                    labels[labels==r1]=r2
                    combinemat[r2,:] += combinemat[r1,:] # new combined region inherits all connections of merged regions

        # relabel with continuous values from 1            
        _, labels = np.unique(labels, return_inverse=True) 
        labels = np.array(np.reshape(labels,S),dtype='int32')

    
    #Add manipulated matrices to dictionary
    image_dict = {}
    image_dict['_binarized'] = outmask
    image_dict['_classprob'] = mask
    image_dict['_dist'] = dist
    image_dict['_segmented'] = labels
    image_dict['_markers'] = markers
    
    return image_dict
    

#Given ground truth and segmented predictions, update accuracy tables
#def UpdateAccTable(image_dict,true_pred,true_pred_seg,mask,AOI):
#    '''
#    image_dict = dictionary of final images from SegmentPrediction
#    true_pred = accuracy table from binarized prediction
#    true_pred_seg = accuracy table from segmented prediction
#    mask = Ground truth mask for image
#    AOI = AOI for image
#    '''
#    outmask = image_dict['_binarized']
#    labels = image_dict['_segmented']
#    
#    for inout in [0,1]:
#        for inout2 in [0,1]:
#            true_pred[inout,inout2] += np.sum((outmask==inout2)*((mask>0)==inout)*(AOI>0))
#            
#            true_pred_seg[inout,inout2] += np.sum(((labels>0)==inout2)*((mask>0)==inout)*(AOI>0))

                
#Save segmented matrix as image 
#COME BACK AND CLEAN UP THIS KEY THING ITS STUPID
def SaveImage(BB_output_folder,name,image_dict,images_to_save,dist_max):
    pixel_scaling = {'_binarized': 254,'_classprob': 254,'_dist': 254.0/(dist_max*1.2),'_segmented': 97,'_markers': 97}
    for key in images_to_save:
        cv2.imwrite(BB_output_folder+'/'+name[:name.index('.')]+key+'.png',np.array(image_dict[key]>0)+np.array((pixel_scaling[key]*image_dict[key]) % 255,dtype=int))


#Save segmented matrix as binary .npy           
def SaveMatrix(BB_output_folder,name,image_dict,images_to_save):
    for key in images_to_save:
        np.save(BB_output_folder+'/'+name[:name.index('.')]+key,image_dict[key])
        
        
#Load model parameters from a filepath, institute a model, load weights from a checkpoint (if any) and perform any
#additional training required using the train_gen and test_gen. Save new information to new output path? Add number of 
#epochs already completed as a stored value in a filename?
def CreateModel(filepath,dt_string,num_epochs,param,outputfilename,train_gen=None,test_gen=None,chkpt_filepath = None):    
    
    #Create model
    S = param['S_half']*2+1
    img_shapes = [S,S,param['scales']*param['dim3']]
    modelBB = generate_network(img_shapes,model_num=param['HP_network_num']) 
    
    BB_weight_filename = outputfilename+'/networkweights.ckpt'
    
    
    if chkpt_filepath is None:
        #print("[DEBUG] No checkpoint filename passed, starting from scratch")
        pass
    
    else:
        #This isn't the right file to check for
        #if os.path.isfile(chkpt_filepath)==False:
        #    raise Exception('No checkpoint found at given path')
        print('[BB] Loading model weights from previous checkpoint')
        #Load model
        modelBB.load_weights(chkpt_filepath)
        
        
    #Train the model more if desired
    if num_epochs!=0:
        if train_gen is None:
            raise Exception('[DEBUG] THERE IS NO GENERATOR OF TRAINING DATA TO TRAIN ON')
       # if test_gen is None:
       #     print("[DEBUG] No test generator was passed, so we assume there is none")
            
        # Create a callback that saves the model's weights every epoch
        cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=BB_weight_filename, verbose=1, save_weights_only=True)
    
        modelBB.fit(train_gen, validation_data=test_gen, epochs=num_epochs,callbacks=[cp_callback]);
    return modelBB
        



#Calculate the SEG accuracy for a whole image vs a ground truth mask.
#Returns np array of Jacard values for each true region.
def segmented_accuracy(mask,prediction,bg_val=0):
    '''
    mask = true segmented values
    prediction = predicted segmented values
    bg_val = value of the region we call background
    '''    

    # shape unimportant now, just overlap.
    mask = mask.flatten()
    prediction = prediction.flatten()

    # find areas of each unique predicted region (store for Jaccard similarity index (J) calculation later)
    pred_regs, areas_pred_regions = np.unique(prediction, return_counts=True)
    
    # get unique values of true regions
    true_regs = list(set(mask))
    true_regs.remove(bg_val) # background not relevant

    SEG = np.zeros(len(true_regs)) # SEG score for each true region
    areas_true_regions = np.zeros(len(true_regs), dtype=int) # Area of each true region


    # Cycle through true regions, find overlaps, find largest overlap and keep Jaccard (if overlap > half)
    for reg_ind,reg in enumerate(true_regs):

        # isolate entire true region in predicted segmentation, store total area, find unique predicted regions within
        pred_regs_local = prediction[mask==reg] # isolate true region
        R = len(pred_regs_local) # calculate its area
        areas_true_regions[reg_ind] = R 
        unique_local_regs, local_areas = np.unique(pred_regs_local, return_counts=True) # list overlapping predicted regions

        max_overlap_idx = np.argmax(local_areas) # find predicted region with highest overlap

        # if this overlap is not background AND is more than half of the true region, calculate J
        if unique_local_regs[max_overlap_idx] != bg_val and local_areas[max_overlap_idx] > R*(0.5):
            I = local_areas[max_overlap_idx] # intersection of pred and real
            P = areas_pred_regions[np.searchsorted(pred_regs,unique_local_regs[max_overlap_idx])] # whole area of pred region
            # R = whole area of real region

            SEG[reg_ind] = I/(P+R-I)

        # if no overlap is high enough, SEG[reg_ind] = 0

    return SEG, areas_true_regions
  
# THIS WILL STAY, BUT IS NOT YET FINISHED.
# Predict function that takes in a generator and model, returns a prediction
def Predict(param, model, gen, BB_output_folder, names, true_pred = None, true_pred_seg = None, seg_masks = None):
    '''
    inputs:
    param: list of parameters 
        including save_to_png and save_to_npy which specify file type to save
        and outputs to save: '_segmented', '_binarized', '_dist', '_classprob', '_markers' (seeds for watershedding)
    model: trained model used for prediction
    gen: generator of images for prediction
    BB_output_folder: filepath for saving segmented predictions 
    names: list of original image names for saving predictions
    true_pred: 2x2 accuracy matrix to be updated on binarized prediction of test set, default None
    true_pred_seg: 2x2 accuracy matrix to be updated on segmented prediction of test set, default None
    seg_masks: ground truth segmented masks
    '''

    BBpredictions = gen.get_prediction(model)
    dist_max = gen.dist_max

    images_to_save = convert_param_to_outputs(param)
    #print('[BB] Exporting ',images_to_save)
    
    SEG = None;
    areas = None; # return none if no mask present
    
    for k, BBout in enumerate(BBpredictions):
        image_dict = SegmentPrediction(BBout,dist_max)
        if (true_pred is not None) and (true_pred_seg is not None):
            mask = gen.get_masks(k)
            AOI = gen.get_AOI(k)
            if len(np.unique(mask))>1:
                #UpdateAccTable(image_dict,true_pred,true_pred_seg,mask,AOI)
                '''
                image_dict = dictionary of final images from SegmentPrediction
                true_pred = accuracy table from binarized prediction
                true_pred_seg = accuracy table from segmented prediction
                mask = Ground truth mask for image
                AOI = AOI for image
                '''
                outmask = image_dict['_binarized']
                labels = image_dict['_segmented']

                for inout in [0,1]:
                    for inout2 in [0,1]:
                        true_pred[inout,inout2] += np.sum((outmask==inout2)*((mask>0)==inout)*(AOI>0))

                        true_pred_seg[inout,inout2] += np.sum(((labels>0)==inout2)*((mask>0)==inout)*(AOI>0))

                SEG,areas = segmented_accuracy(seg_masks[k]*AOI,labels*AOI,bg_val=0) # returns Jacard indicees and areas
               
        if param['save_to_png']:
            SaveImage(BB_output_folder,names[k],image_dict,images_to_save,param['dist_max'])
        if param['save_to_npy']:
            SaveMatrix(BB_output_folder,names[k],image_dict,images_to_save)
        
    if not SEG is None:
        SEG = list(SEG)
    if not areas is None:
        areas = list(areas)

    return SEG, areas # return Jacard indicees for each 


