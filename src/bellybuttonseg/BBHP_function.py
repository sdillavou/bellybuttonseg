#######################################################################################
import tensorflow as tf
import numpy as np
import os,sys,datetime
import gc
from pathlib import Path
from DataGenerator import circ_region, CreateGenerator, HPinputGenerator
from saved_model_functions import convert_param_to_outputs,  CreateModel, Predict
from generate_network import model_S_half_minimum
from load_save_helpers import load_parameters,save_parameters

#sys.path.append(os.path.dirname(os.path.realpath(__file__))) # Add parent directory to sys.path (for script)
sys.path.append(os.path.dirname(os.getcwd())) # Add parent directory to sys.path (for notebook)

# round results (e.g. accuracy) to this many decimal places
round_out = 5

# remove test set from fit function. Speeds up training, loses mid-training updates 
include_test_in_training = False



physical_devices = tf.config.list_physical_devices('GPU') 

if len(physical_devices):
    #print('LIMITING TO 1 GPU')
    tf.config.set_visible_devices(physical_devices[-1:], 'GPU')

if len(physical_devices):
   # print('SETTING MEMORY GROWTH ON GPU TO TRUE')
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

####### END OF HEADER #################################################################


param_types={'S_half':int, 'scales':int, 'scalefactor':int, 'HP_neighborhood_radius':float, 'HP_particle_border_mult':float, 'HP_two_particle_mult':float, 'HP_img_edge_mult':float, 'batch_size':int, 'rotations':int, 'flips':int, 'images_to_grayscale':int, 'HP_network_num':int, 'HP_train_epochs':int, 'fraction':float, 'output_segmented':int, 'output_binarized':int, 'output_dist':int, 'output_classprob':int, 'output_markers':int, 'save_to_png':int, 'save_to_npy':int,'track_outies':int,'dim3':int, 'dist_max':int}

norms_types = {'img_min':float,'img_max':float}

results = ['test_acc','train_acc','test_right','test_total','train_right','train_total','train_innie_f1_preseg','train_outie_f1_preseg','test_innie_f1_preseg','test_outie_f1_preseg','train_innie_f1','train_outie_f1','test_innie_f1','test_outie_f1','train_seg','test_seg','train_area_weighted_seg','test_area_weighted_seg']
res_types = dict(zip(results,[float for _ in results]))




def BBHP(filepath,param,train_img_count=-1,test_img_count=-1,want_train=None, want_test=None,dt_string=None,predict_path=None):
    
    
    #######################################################################################
    # INDICATE ALL FOLDERS WHERE IMAGES, MASKS, AND AOI ARE STORED
    #######################################################################################
    
    # subfolders:
    train_img_folder = filepath+'train_images'      # images for training (with masks in mask_folder)
    test_img_folder = filepath+'test_images'        # images for testing (with masks in mask_folder)
    predict_img_folder = filepath+'predict_images'  # images for prediction (no masks needed)
    mask_folder = filepath+'masks'                  # ID'd masks with 0 as outies. Must be same 2D shape as training and test images
    AOI_folder = filepath+'areas_of_interest'       # binary masks representing area of interest (only 1s are included in training, testing, and predicting.)
    # if AOI does not exist for a given image, it is assumed every pixel is in the area of interest.

    # NOTE: masks, images, and AOIs are matched by name. The extension can be different, 
    # but the filename prior to the extension should be identical.

    # The max and min pixel values of the entire training set will be measured, and then all images
    # will be scaled so that these values correspond to 0.5 and -0.5 respectively. 
    # These scalings are not saved in the image files, but the max/min is stored in the "normalization.txt" file.


    #### CHECK PARAMETER COMBINATIONS, SAVE TO TEXT FILE AND LOAD THEM INTO WORKSPACE VARIABLES
    #######################################################################################
    
    norms = None
    param_input = param.copy()
    if dt_string is not None:
        outputfilename = filepath+'outputs_'+dt_string
        print('Date string passed, retrieving parameters and normalizations from ' + outputfilename)
        parameter_filename = outputfilename+'/parameters'
        HP_weight_filename = outputfilename+'/networkweights.ckpt'
        normalization_filename = outputfilename+'/normalization'
        
        
        param = load_parameters(parameter_filename,param_types) # load old parameters
        old_epochs = param['HP_train_epochs']
        
        #Adjust loaded (old) parameters to match any input (new) parameters
        for key,value in param_input.items():
            param[key] = value
            
        num_epochs = param_input['HP_train_epochs'] # must have specified this value in new input

        norms = load_parameters(normalization_filename,norms_types)
        
        param['HP_train_epochs'] += old_epochs # store the TOTAL number of epochs trained
    
        if num_epochs>0:
            
            dt_string = datetime.datetime.now().strftime("%y_%m_%d_%H_%M_%S")
            outputfilename = outputfilename+'_'+dt_string
            index = outputfilename.rfind('/')
            
            print('[BB] More training requested, creating '+outputfilename[index+1:]+' to hold new outputs')
            os.mkdir(Path(outputfilename))
            
            parameter_filename = outputfilename+'/parameters'
            normalization_filename = outputfilename+'/normalization'
            #HP_weight_filename = outputfilename+'/networkweights.ckpt'
        

        else:
            print("[BB] No more training requested, outputs staying in original folder")
    
    

    else:
        # create current date/time string that will be used to catalog all outputs and parameters
        dt_string = datetime.datetime.now().strftime("%y_%m_%d_%H_%M_%S")
        outputfilename = filepath+'outputs_'+dt_string
        print('[BB] Outputs will be stored in outputs_'+dt_string)
        os.mkdir(Path(outputfilename))

        
        parameter_filename = outputfilename+'/parameters'
        HP_weight_filename = None#outputfilename+'/networkweights.ckpt'
        normalization_filename = outputfilename+'/normalization'


        # Training batch size must be correct multiple when using rotations/flips
        unique_in_batch = param['batch_size']/((1+3*param['rotations'])*(1+param['flips']))
        if not unique_in_batch == int(unique_in_batch):
            raise Exception('Batch number must be divisible by 4 if doing rotations, 2 if flipping, and 8 if both.')

        # input window size must be big enough for selected network
        if not param['S_half'] >= model_S_half_minimum(param['HP_network_num']):
            raise Exception('HP network model '+str(param['HP_network_num'])+ ' requires S_half >= '+str(model_S_half_minimum(param['HP_network_num']))+'.')
        
        num_epochs = param['HP_train_epochs']
        
        
    if num_epochs>0:
        #print('[DEBUG] Preparing generators for model training...')
        
        #### LOAD TRAIN/TEST IMAGES, MASKS, AOIs, CREATE DATAGENERATORS FOR TRAINING/TESTING
        #######################################################################################

        train_genHP,train_img_names,_ = CreateGenerator(filepath=filepath, subfolder='train_images',\
                                      param=param, want=want_train, \
                                      count=train_img_count, train_test_predict=0, chatter = True)

        if include_test_in_training:
            test_genHP,test_img_names,_ = CreateGenerator(filepath=filepath, subfolder='test_images',\
                                         param=param, want=want_test, \
                                         count=test_img_count, train_test_predict=1, chatter = True)
        else:
            test_genHP = None
    
        #Add number of channels in image to parameters
        param['dim3'] = train_genHP.dim3
        if norms is None:
            
            #### Generate Normalization values from training images
            #######################################################################################
            norms = {}
            norms['img_max'],norms['img_min'] = train_genHP.get_img_range()
        
        #Normalize the images in our generators   
        train_genHP.normalize_images(norms['img_max'],norms['img_min'])
        if include_test_in_training:
            test_genHP.normalize_images(norms['img_max'],norms['img_min'])
            
    elif num_epochs==0:
        print('Number of epochs = 0, entering predict only mode')
        train_genHP = None
        test_genHP = None
    elif num_epochs<0:
        raise Exception('Please change number of epochs to be >=0')
    
    # save parameters
    save_parameters(parameter_filename,list(param.keys()),list(param.values()))
    save_parameters(normalization_filename,list(norms.keys()),list(norms.values()))

    #### CREATE AND TRAIN NEURAL NETWORK FOR HP, SAVE FINAL WEIGHTS, GENERATE PREDICTIONS
    #######################################################################################

    
    #train_genHP.plot_items(5)
    
    print('[BB] Training Neural Network')
    if not include_test_in_training:
        print('[BB] -- Excluding test images from .fit function. Test set results available at end.')

    modelHP = CreateModel(filepath,dt_string,num_epochs,param,outputfilename,train_gen=train_genHP,test_gen=test_genHP,chkpt_filepath=HP_weight_filename)
    
    # garbage collect
    del train_genHP
    del test_genHP
    gc.collect()
    
    #If dt_string has changed during training, update receiving folders for output
    HP_output_folder = outputfilename+'/predictions'
    results_filename = outputfilename+'/results'
        
    if not (param['save_to_png'] or param['save_to_npy']):
        print('No save flags are activated, prediction images/data will not be saved')
        
    if num_epochs>0:
        os.mkdir(HP_output_folder)
        results_filename = outputfilename+'/results'

        # generator for training data
        #print('[BB] Calculating accuracy and exporting images of predictions...')
        train_true_pred = np.zeros((2,2));
        train_true_pred_seg = np.zeros((2,2));
        train_SEG = [];
        train_areas = [];
        test_SEG = [];
        test_areas = [];
        more_data = True
        index = 0
        
        images_to_save = convert_param_to_outputs(param)
        print('[BB] Exporting',', '.join([i[1:] for i in images_to_save]))
        
        while more_data:
            try:
                all_genHP,img_names,seg_masks = CreateGenerator(filepath=filepath, subfolder='train_images',\
                                      param=param, want=want_train, \
                                      count=train_img_count, train_test_predict=1,index = index, chatter = False)
                all_genHP.normalize_images(norms['img_max'],norms['img_min'])
                SEG0, areas0 = Predict(param, modelHP, all_genHP, HP_output_folder, img_names, true_pred = train_true_pred, true_pred_seg = train_true_pred_seg,seg_masks = seg_masks)
                
                train_SEG += SEG0
                train_areas += areas0
                
                print('[BB] -- Train Image '+str(index)+': '+img_names[0]+' -- SEG: '+str(np.round(np.mean(SEG0),round_out)))

                    
                index+=1
                del all_genHP
                gc.collect()
            except IndexError:
                more_data= False

        # generator for test data
        test_true_pred = np.zeros((2,2))
        test_true_pred_seg = np.zeros((2,2))
        more_data = True
        index = 0
        while more_data:
            try:
                all_genHP,img_names,seg_masks = CreateGenerator(filepath=filepath, subfolder='test_images',\
                                      param=param, want=want_train, \
                                      count=train_img_count, train_test_predict=1,index = index)
                all_genHP.normalize_images(norms['img_max'],norms['img_min'])
                SEG0, areas0 = Predict(param, modelHP, all_genHP, HP_output_folder, img_names, true_pred = test_true_pred, true_pred_seg = test_true_pred_seg, seg_masks = seg_masks)
                
                test_SEG += SEG0
                test_areas += areas0
               
                print('[BB] -- Test Image '+str(index)+': '+img_names[0]+' -- SEG: '+str(np.round(np.mean(SEG0),round_out)))

                index+=1

                del all_genHP
                gc.collect()
            except IndexError:
                more_data= False

        tf.keras.backend.clear_session()
        train_f1s = f1_score(train_true_pred)
        test_f1s = f1_score(test_true_pred)    

        train_f1s_seg = f1_score(train_true_pred_seg)
        test_f1s_seg = f1_score(test_true_pred_seg)

        #print(train_true_pred)
        #print(test_true_pred)

        results = {}
        results['test_acc'] = np.sum(np.diag(test_true_pred_seg))/np.sum(test_true_pred_seg)
        results['train_acc'] = np.sum(np.diag(train_true_pred_seg))/np.sum(train_true_pred_seg)
        results['test_right'] = np.sum(np.diag(test_true_pred_seg))
        results['test_total'] = np.sum(test_true_pred_seg)
        results['train_right'] = np.sum(np.diag(train_true_pred_seg))
        results['train_total'] = np.sum(train_true_pred_seg)



        results['train_innie_f1_preseg'] = train_f1s[1]
        results['train_outie_f1_preseg'] = train_f1s[0]
        results['test_innie_f1_preseg'] = test_f1s[1]
        results['test_outie_f1_preseg'] = test_f1s[0] 
        results['train_innie_f1'] = train_f1s_seg[1]
        results['train_outie_f1'] = train_f1s_seg[0]
        results['test_innie_f1'] = test_f1s_seg[1]
        results['test_outie_f1'] = test_f1s_seg[0] 
        
        

        results['train_SEG'] = np.mean(train_SEG)
        results['test_SEG'] = np.mean(test_SEG)
        results['train_area_weighted_SEG'] = np.mean(np.array(train_SEG)*np.array(train_areas))/np.mean(train_areas)
        results['test_area_weighted_SEG'] = np.mean(np.array(test_SEG)*np.array(test_areas))/np.mean(test_areas)

       
        for key in results:
            results[key] = round(results[key],round_out )
        
        #print(results)


        # save results
        save_parameters(results_filename,list(results.keys()),list(results.values()))

    if num_epochs==0 or not (predict_path is None):
        if os.path.isdir(HP_output_folder)==False:
            print('Original training never made predictions, creating prediction folder now...')
            os.mkdir(HP_output_folder)

        # generator for prediction data
        if predict_path is None or predict_path == 'predict_images':
            subfolder = 'predict_images'
            print('Predicting on images in predictions folder...')
            true_pred = None
            true_pred_seg = None
            train_test_predict = 2
            prestr = '';
        elif predict_path == 'test_images':
            subfolder = 'test_images'
            print('Predicting on images in the test folder...')
            true_pred = np.zeros((2,2))
            true_pred_seg = np.zeros((2,2))
            train_test_predict = 1
            prestr = 'test_'
        elif predict_path == 'train_images':
            subfolder = 'train_images'
            print('Predicting on images in the train folder...')
            true_pred = np.zeros((2,2))
            true_pred_seg = np.zeros((2,2))
            train_test_predict = 1
            prestr = 'train_'
        else:
            raise Exception("predict_path must be None (defaults to predict_images), 'train_images', 'test_images', or 'predict_images'")
        more_data = True
        index = 0
        prediction_results_filename = outputfilename+'/prediction_results'
        while more_data:
            try:
                all_genHP,img_names,seg_masks = CreateGenerator(filepath=filepath, subfolder=subfolder,\
                                      param=param, train_test_predict=train_test_predict,index = index, chatter = False)
                all_genHP.normalize_images(norms['img_max'],norms['img_min'])

                SEG0, areas0 = Predict(param, modelHP, all_genHP, HP_output_folder, img_names,true_pred = true_pred, true_pred_seg = true_pred_seg ,seg_masks = seg_masks)
                index+=1
                del all_genHP
                gc.collect()
            except IndexError:
                more_data= False
        if true_pred is not None:
            tf.keras.backend.clear_session()
            f1s = f1_score(true_pred)
            f1s_seg = f1_score(true_pred_seg)

            results = {}
            results[prestr+'acc'] = np.sum(np.diag(true_pred_seg))/np.sum(true_pred_seg)
            results[prestr+'right'] = np.sum(np.diag(true_pred_seg))
            results[prestr+'total'] = np.sum(true_pred_seg)



            results[prestr+'innie_f1_preseg'] = f1s[1]
            results[prestr+'outie_f1_preseg'] = f1s[0]
            results[prestr+'innie_f1'] = f1s_seg[1]
            results[prestr+'outie_f1'] = f1s_seg[0]

            for key in results:
                results[key] = round(results[key],round_out )
        
            #print(results)


            # save results
            save_parameters(prediction_results_filename,list(results.keys()),list(results.values()))

    return dt_string

# takes square prediction matrix and outputs f1 score for each class
def f1_score(true_pred):
    out = np.zeros(len(true_pred))
    for k in range(len(true_pred)):
        # precision = true_positives / predicted_positives
        precision = true_pred[k,k]/np.sum(true_pred[:,k])
        # recall = true_positives / possible_positives
        recall = true_pred[k,k]/np.sum(true_pred[k,:])
        out[k] = 2*((precision*recall)/(precision+recall+0.0000000001))
    
    return out