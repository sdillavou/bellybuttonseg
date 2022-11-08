import tkinter as tk
from tkinter import filedialog
import io,os,sys,shutil
sys.path.append(os.path.dirname(__file__))
from load_save_helpers import create_default_params, save_parameters, load_parameters
from BBHP_function import param_types, BBHP
from PIL import Image
import requests 

base_param_name = 'base_parameters'

# Shell function to run the training and/or prediction using Bellybutton
def runBB(train=True, predict=False):
    
    
    file_path = get_filepath()
    dt_string = None
    predict_path = None
   
    idx = file_path[:-1].rfind('/')

    # we are being asked to load a network, change file_path to base folder name and store dt_string
    if file_path[idx+1:idx+9] == 'outputs_':
        dt_string = file_path[idx+9:-1]
        file_path = file_path[:idx+1]

    param = load_parameters(file_path+base_param_name,param_types)

    if not (dt_string is None):
        del param['dim3']
        del param['HP_network_num']
        del param['S_half']
    
    if not train:
        param['HP_train_epochs'] = 0

    if predict:
        predict_path = 'predict_images'


    # trying to predict without training or selecting an existing network
    if dt_string is None and not train and predict:
        print('Selected folder ('+file_path+') does not contain network weights to use for prediction.')
        return


    BBHP(file_path,param,predict_path=predict_path,dt_string=dt_string)
    #, train_img_count = train_img_count)
    #,dt_string='22_06_06_20_44_28');
    #,train_img_count=train_img_count,test_img_count=test_img_count,dt_string=dt_string)

    print('[BB] Task Completed Successfully.')

# returns images of a list of filenames for a given folder
def createdir(example=0):
    
    # request filepath from user
    file_path = get_filepath()

    # if pre-loaded example, images, masks, etc will be added
    if example == 1:
        folder_name = 'PhotoelasticDisks'
        
    elif not example == 0:
        print('[BB] Bad input, no such example #'+example+'.')
        return
    
    else: # example == 0
        folder_name = input('[INPUT] Enter folder name for new project: ')
        
        
    full_filename = file_path+folder_name
    if os.path.isdir(full_filename):
        print('[BB] Desired new folder already exists! Nohting has been changed.')
        return
   
    
    
    # Create file structure
    os.mkdir(full_filename)
    os.mkdir(full_filename+'/train_images')
    os.mkdir(full_filename+'/test_images')
    os.mkdir(full_filename+'/predict_images')
    os.mkdir(full_filename+'/masks')
    os.mkdir(full_filename+'/areas_of_interest')

    param = create_default_params();



    
    # given example must moves images, masks, aois into structure
    if example ==1:
        PEurl = "https://raw.githubusercontent.com/sdillavou/BellybuttonExampleData/main/PhotoelasticDisks/"

        # Areas of interest
        download_image(PEurl+"areas_of_interest/test.png", full_filename+'/areas_of_interest/test_01.png')
        shutil.copyfile(full_filename+'/areas_of_interest/test_01.png', full_filename+'/areas_of_interest/test_02.png')
        shutil.copyfile(full_filename+'/areas_of_interest/test_01.png', full_filename+'/areas_of_interest/test_03.png')
        download_image(PEurl+"areas_of_interest/train.png", full_filename+'/areas_of_interest/train_01.png')
        shutil.copyfile(full_filename+'/areas_of_interest/train_01.png', full_filename+'/areas_of_interest/train_02.png')
        shutil.copyfile(full_filename+'/areas_of_interest/train_01.png', full_filename+'/areas_of_interest/train_03.png')

         # Masks
        download_image(PEurl+"masks/early.png", full_filename+'/masks/train_01.png')
        download_image(PEurl+"masks/mid.png", full_filename+'/masks/train_02.png')
        download_image(PEurl+"masks/late.png", full_filename+'/masks/train_03.png')
        shutil.copyfile(full_filename+'/masks/train_01.png', full_filename+'/masks/test_01.png')
        shutil.copyfile(full_filename+'/masks/train_02.png', full_filename+'/masks/test_02.png')
        shutil.copyfile(full_filename+'/masks/train_03.png', full_filename+'/masks/test_03.png')

         # Images
        download_image(PEurl+"images/early.tif", full_filename+'/train_images/train_01.tif')
        download_image(PEurl+"images/mid.tif", full_filename+'/train_images/train_02.tif')
        download_image(PEurl+"images/late.tif", full_filename+'/train_images/train_03.tif')
        shutil.copyfile(full_filename+'/train_images/train_01.tif', full_filename+'/test_images/test_01.tif')
        shutil.copyfile(full_filename+'/train_images/train_02.tif', full_filename+'/test_images/test_02.tif')
        shutil.copyfile(full_filename+'/train_images/train_03.tif', full_filename+'/test_images/test_03.tif')
        # Prediction images
        shutil.copyfile(full_filename+'/train_images/train_01.tif', full_filename+'/predict_images/whole_01.tif')
        shutil.copyfile(full_filename+'/train_images/train_02.tif', full_filename+'/predict_images/whole_02.tif')
        shutil.copyfile(full_filename+'/train_images/train_03.tif', full_filename+'/predict_images/whole_03.tif')
        



    save_parameters(full_filename+'/'+base_param_name,list(param.keys()),list(param.values()))
   
    print('Created and populated '+full_filename)

    
def get_filepath():
    # request filepath from user
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askdirectory()
    return file_path + '/'

# https://gist.github.com/mjdietzx/545fa2874b2688e9bcb71e2ee92cd5a0
def download_image(url, image_file_path):
    r = requests.get(url, timeout=4.0)
    if r.status_code != requests.codes.ok:
        assert False, 'Status code error: {}.'.format(r.status_code)

    with Image.open(io.BytesIO(r.content)) as im:
        im.save(image_file_path)