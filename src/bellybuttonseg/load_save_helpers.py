import configparser
import cv2
import numpy as np
import os
import copy
from skimage.segmentation import watershed


# Credit: https://stackoverflow.com/questions/2068372/fastest-way-to-list-all-primes-below-n-in-python/3035188#3035188
def primesto(n):
    """ Returns a array of primes, 3 <= p < n """
    sieve = np.ones(n//2, dtype=bool)
    for i in range(3,int(n**0.5)+1,2):
        if sieve[i//2]:
            sieve[i*i//2::i] = False
    return np.insert(2*np.nonzero(sieve)[0][1::]+1,0,2)

primelist = primesto(10**7) # this is 664579 primes. If you have more particles than this, good luck.
primelist = np.insert(primelist,0,0)


# create a circular region where pixels within distance r of the center are 1's, others are 0's.
def circ_region(r):
    
    if r <=1:
        return 1
    
    R = int(np.floor(r))
    xv,yv = np.meshgrid(range(-R,R+1),range(-R,R+1))
    
    return (xv**2 + yv**2) <= r**2


def save_parameters(filename,keys,vals):

    config = configparser.RawConfigParser()

    config.add_section('Parameters')
    
    for k,v in zip(keys,vals):
        config.set('Parameters',  k.lower(), v)
        
    with open(filename+'.txt', 'w') as configfile:
          config.write(configfile)
            
def load_parameters(filename,param_types):
    config = configparser.ConfigParser()
    config.read(filename+'.txt')
    
    keys = list(param_types.keys())
    values = list(param_types.values())
    
    out = [0 for _ in keys]
    keys2 = [k.lower() for k in keys]

    for key in config['Parameters']: 
        if key.lower() in keys2:
            out[keys2.index(key.lower())]=config['Parameters'][key.lower()]
    
    param = LoadConverter(out,values)
    param = dict(zip(keys,param))

    return param
    
    
    
#Convert array of parameters formatted as strings into array of parameters formatted according to param_var_types
def LoadConverter(param_string,param_var_types):
    for ind in range(len(param_string)):
        param_string[ind] = param_var_types[ind](param_string[ind])
    return param_string

    
def load_image(filepath, binarize = False, integerize = False, RGB_to_gray = False, segment = False):
    
    if filepath is None:
        return None
    
    if binarize and segment:
        raise Exception('Binarization and Segmentation are incompatible')
    
    
    if filepath[-4:] == '.txt': # load text
        img = np.array(np.loadtxt(filepath,skiprows=0))
    else: # load image file
        img = cv2.imread(filepath, cv2.IMREAD_ANYDEPTH)
        if img is None:
            print('Default image loading method failed, trying backup')
            img = cv2.imread(filepath)
        if img is None:
            raise Exception('Loaded image is empty')
        if len(np.unique(img)) == 1:
            print('Only one value for the entire image '+filepath)
    
    
    # average color channels
    if RGB_to_gray:
        #print('Converting '+ filepath + ' to grayscale before use.')
        if len(np.shape(img))==3 and np.shape(img)[2]==3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        elif len(np.shape(img))>=3:
            print('Error for: '+filepath)
            raise Exception('Attempted to convert to grayscale but image is wrong dimensions')
        
    # convert all non-zero values to 1's
    if binarize:
        if len(np.shape(img))>2 and np.shape(img)[2]>1:
            for k in range(1,np.shape(img)[2]):
                if not np.array_equal(img[:,:,0], img[:,:,k]):
                    raise Exception('Attempting to binarize image with multiple and different color channels.')
            img = img[:,:,0] # reduce to one channel
            
        img = 1*(img!=0)
    
    
    # take given IDs and break them up according to disconnected regions (works for binary masks, too).
    if segment:
        
        # distance inside ID'd regions
        dist = distance_maker(img)

        # give each disconnected region a unique ID
        (_, markers, _, _) = cv2.connectedComponentsWithStats((dist>2).astype('uint8'))

        # watershed using these IDs, fill out given mask, use distance as rate
        img = watershed(-dist, markers = markers, mask=(img>0)).astype(int)
        
        
    
    
    
    
    # convert unique values into unique integers. 0 is interpreted as background (outies)
    if integerize and not segment: # segmenting already does this
        S = np.shape(img)
        if len(S)>2 and S[2]>1:
            for k in range(1,S[2]):
                if not np.array_equal(img[:,:,0], img[:,:,k]):
                    raise Exception('Attempting to integerize image with multiple and different color channels.')
            img = np.sum(img,tuple(k for k in range(2,len(S)))) # reduce to one channel
            
         # take unique values and convert them to indicees
        U = np.unique(img)
        img2 = np.copy(img)
        for idx,val in enumerate(U):
            img[img2==val] = idx+1
            
        img[img2==0] = 0 # maintain background as background (it was assigned a different value above)
        img = np.array(img,dtype=int)
    return img


# find masks from filenames mask_names with matching names (ignore file extension) to the image filenames in img_names
# return a list of mask names that match, in the same order as img_names 
def find_matching_masks(img_names, mask_names, raise_issues=True):
     
    img_name_bases = [name[:name.index('.')] for name in img_names]
    mask_name_bases = [name[:name.index('.')] for name in mask_names]
    mask_idxs = [mask_name_bases.index(img_name) if img_name in mask_name_bases else None for img_name in img_name_bases]
    
    if raise_issues and None in mask_idxs:
        raise Exception('Masks are missing for the following images: '+', '.join([name for name, idx in zip(img_names,mask_idxs) if idx is None]))
    
    return [mask_names[k] if not k is None else None for k in mask_idxs] 
    

# returns a string listing input image names and their sizes
def list_imgs_and_sizes(img_names,imgs):
    # str call makes None type into 'None'
    out = [str(name) + ' ('+','.join(str(k) for k in np.shape(img))+')' for name,img in zip(img_names,imgs)]
    return ', '.join(out)  
    
# returns a list of filenames (assumption is they are all images or text)
# want is indicees of desired images (None -> all files), count is total number of names returned (always a subset of want/all)
def get_image_list(folder, want = None, count=-1):
    
    names = os.listdir(folder) 
    for name in names: # remove .DStore files and the like
        if name[0] == '.':
            names.remove(name)

    # want = set of images to draw from (None means all). count = number of images to end up with (-1 means all)
    if count ==-1: # take all images of want
        if want is None:
            want = np.linspace(0, len(names)-1,len(names))
    
    elif count >= 0: # take [count] images
        if want is None:
            want = np.linspace(0, len(names)-1, count)
        else:
            want = want[np.linspace(0, len(want)-1, count)]

    else: # count needs to be positive or -1
        raise Exception('Image counting parameter has an illegal value.')
  
    # narrow names down only to desired ones, then create array for images
    return [names[int(k)] for k in want] 



# returns images of a list of filenames for a given folder
def load_image_list(folder, names, binarize = False, integerize = False, RGB_to_gray = False, chatter = True, segment = False):
    
    if RGB_to_gray and (not integerize) and (not binarize) and chatter:
        index = folder.rfind('/') # include only this folder's name (not filepath)
        print('[BB] Converting images in '+ folder[index+1:] + ' to grayscale before use.')
        
    names2 = copy.copy(names)
    for i,name in enumerate(names2):
        if not name is None:
            names2[i] = folder + '/' + name
            
    return [load_image(name, binarize = binarize, \
                          integerize = integerize, RGB_to_gray = RGB_to_gray, segment = segment) \
                          for name in names2] 


def load_images_aois_masks(param,img_folder,AOI_folder,mask_folder=None):
    #Find image list
    img_names, imgs = get_image_list(img_folder, RGB_to_gray = param['images_to_grayscale'])
    
    #If there are AOI's, load them
    #if aoi_folder!=None:
    AOI_names = os.listdir(AOI_folder)
    AOI_names = find_matching_masks(img_names, AOI_names, raise_issues=False)
    _, AOIs = get_image_list(AOI_folder, AOI_names, integerize = True)
    AOIs = [elem if (not elem is None) else np.ones(np.shape(imgs[k])[:2]) for k,elem in enumerate(AOIs)]
    
    #If there are masks, load them
    if mask_folder!=None:
        mask_names = os.listdir(mask_folder)
        mask_names = find_matching_masks(img_names, mask_names)
        _, masks = get_image_list(mask_folder,mask_names, integerize = True)
    else:
        mask_names, masks = [],[]
    
    return img_names,imgs,AOI_names,AOIs,mask_names,masks




# creates a 2D distance map (dist to nearest border) from an ID'd mask.
def distance_maker(mask):

    eroder = 1.01
    img2 = np.sqrt(np.array(primelist[np.array(mask,dtype=int)],dtype=float)) # make segmented image using only primes, then take sqrt
    kernel = np.array(circ_region(eroder),dtype=np.uint8)

    # convolve image over kernel -- those that catch another sqrt(prime) will become non-integers when multiplied by sqrt(prime) img
    eroded_mask = np.multiply(cv2.filter2D(img2,-1,kernel),img2) 

    # detect non integers - these are innies next to other particles
    twoParticles = 1-(np.abs(eroded_mask-np.round(eroded_mask))>.000001)  #this number is somewhat arbitrary. Too small is a problem though.
    

    #eroded_mask = cv2.erode(np.array(img2>0,dtype=float),kernel) - twoParticles # erode and subtract particle-particle borders
    #eroded_mask[eroded_mask<0] = 0 # any subtractions below 0 from twoParticles are ignored.
    # mask now is binary and eroded by 1 pixel from ID'd mask

    # take distance transform (distance from zeros)
    dist = cv2.distanceTransform(np.array(mask>0,dtype='uint8'), distanceType=cv2.DIST_L2, maskSize=0)
    
    # take distance from two particle borders (and add one, since border itself is 2px)
    dist2P = 1+cv2.distanceTransform(np.array(twoParticles,dtype='uint8'), distanceType=cv2.DIST_L2, maskSize=0)
    #dist[dist>0]+=1
    #dist += 1*((mask>0) * (eroded_mask==0)) # add back in eroded away spots (with distance 1)
    
    # distance should be capped and also minimum of distance to 0 and distance to another particle
    dist = np.minimum(dist,dist2P)
    
    
    #dist -= cv2.distanceTransform(np.array(1-(mask>0),dtype='uint8'), distanceType=cv2.DIST_L2, maskSize=0)
   # dist[dist<-self.dist_max] = -self.dist_max
                    
    return dist


def create_default_params():

    param = {};

    param['S_half'] = 12# int, defines size of input image to NN: size = (2[S_half]+1 square). 12 is reccommended.

    param['scales'] = 4 # number of scaled images (int, 1 is just regular without scaling)
    param['scalefactor'] = 2 # scale multiplier (int)
    # for example, with 3 scales of scalefactor 4, images scaled by 1, 4^1=4, and 4^2=16 are included as inputs

    param['batch_size'] = 1024 # int, must be multiple of 8 to allow rotations + flips. Same size for testing.
    #True/False
    param['rotations'] = 1 # rotate images by 90, 180, 270 degrees in training to multiply data by 4
    param['flips'] = 1 # flip images in training to multiply data by 2

    #True/False
    param['images_to_grayscale'] = 1; #convert images to grayscale (if images already one channel, this has no effect.)

    param['HP_network_num'] = 7; # For a list of what networks these numbers create, see readme.
    param['HP_train_epochs'] = 3; # how many epochs to train

    # (total AOI area)*fraction = training samples 
    # These are automatically adjusted such that half come from each class, meaning some may be multi-counted.
    # This does not affect test set.
    param['fraction'] = 0.3

    # maximum distance to train into network (dist from nearest outie). 2*dist_max < min particle diameter
    param['dist_max'] = 10;  

    param['track_outies']  = 0 # invert inputs and track 'outies' instead (works for small, separated particles)

    # optional values for spatially weighting pixels near borders of particles/images
    param['HP_neighborhood_radius'] = 1 # neighborhood of relevance for weighting
    param['HP_particle_border_mult'] =  1# weight multiplier for pixels within HP_neighborhood of a particle border
    param['HP_two_particle_mult'] =1 # weight multiplier for pixels within HP_neighborhood of two particles
    param['HP_img_edge_mult'] = 1 # weight multiplier for pixels within HP_neighborhood of an image edge

    # Specify the type(s) of final images to be saved, True(1) or False(0)
    param['output_segmented']=1
    param['output_binarized']=0
    param['output_dist']=0
    param['output_classprob']=0
    param['output_markers']=0

    # Pick output file type (.png / .npy)
    param['save_to_png']=1
    param['save_to_npy']=0
    
    return param