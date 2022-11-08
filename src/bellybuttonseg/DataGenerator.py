import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from scipy.signal import convolve2d
import matplotlib.pyplot as plt
import copy
import cv2
import gc
from load_save_helpers import find_matching_masks, list_imgs_and_sizes, get_image_list, load_image_list, distance_maker, primelist, circ_region


def_neighborhood_radius = 1.01
def_particle_border_mult = 1, 
def_two_particle_mult = 1, 
def_img_edge_mult = 1,

def_dist_max = 10;




#################################################################################################################       
#################################################################################################################      
########################    CreateGenerator from param/filepath  ################################################ 
#################################################################################################################   
#################################################################################################################   

def CreateGenerator(filepath, subfolder, param, want=None, count=-1, train_test_predict=0, index = -1, chatter = False):
    
    # determine type of generator
    if train_test_predict == 0: # training generator: include manipulations of data and shuffle data
        flip,rotate,randomize = param['flips'],param['rotations'],True
    elif train_test_predict == 1 or train_test_predict == 2: # testing/prediction generator: leave data alone 
        flip,rotate,randomize = False,False,False
    else:
        raise Exception('train_test_predict parameter must be 0 (training), 1 (testing), or 2 (predicting).')

    # create generator object
    dataGenerator = HPinputGenerator(S_half = param['S_half'], \
                                      batch_size = param['batch_size'],\
                                      scales=param['scales'], \
                                      scalefactor=param['scalefactor'], \
                                      flip=flip, rotate=rotate, randomize=randomize,\
                                      dist_max = param['dist_max'])
    
    # load images specified in inputs
    img_names = get_image_list(filepath+subfolder, want=want, count=count)
    
    if train_test_predict == 0 and len(img_names)==0:
        raise Exception('Attempt to create training generator without any images!')
        
    if index!=-1:
        if index>=len(img_names):
            raise IndexError('Index outside of want list')
        else:
            img_names = [img_names[index]]
    imgs = load_image_list(filepath+subfolder, img_names, RGB_to_gray=param['images_to_grayscale'], chatter = chatter)
    if chatter:
        print('[BB] -- '+subfolder+' images: '+list_imgs_and_sizes(img_names,imgs))
        
    # if training or testing, we need masks, find names
    if train_test_predict < 2: 
        mask_folder = filepath+'masks'
        mask_names = find_matching_masks(img_names, get_image_list(mask_folder))
        masks = load_image_list(mask_folder,mask_names, integerize=True,RGB_to_gray = True, segment = True)
        if chatter:
            print('[BB] -- '+subfolder+' masks: '+list_imgs_and_sizes(mask_names,masks))
    else:
        masks = [None for k in imgs]
        
    # look for AOIs, load them (allow only 1's and 0's)
    AOI_folder = filepath+'areas_of_interest'
    AOI_names = find_matching_masks(img_names, get_image_list(AOI_folder), raise_issues=False)
    AOIs = load_image_list(AOI_folder, AOI_names, binarize=True)  
    
    # for all non-defined AOIs, assume entire image is fair game
    for k,img in enumerate(imgs): 
        if AOIs[k] is None:
            AOIs[k] = np.ones(np.shape(img))

    if chatter:
        print('[BB] -- '+subfolder+' AOIs: '+list_imgs_and_sizes(AOI_names,AOIs))
    
    # Enforce specified fraction of data if training set
    if train_test_predict == 0:
        AOIs = fractionalize_AOI(masks, AOIs, param['fraction'])

        
    # if there are masks and track_outies flag is true, alert user and invert masks
    if param['track_outies'] and np.sum([not m is None for m in masks])>0:
        if chatter:
            print('Inverting masks to track zero-valued pixels.')
        masks = [1*(m == 0) if (not m is None) else None for m in masks]
    
    # create HP inputs
    for k,img in enumerate(imgs):
        dataGenerator.add_ID_img(img, mask=masks[k], AOI=AOIs[k], neighborhood_radius=param['HP_neighborhood_radius'], particle_border_mult=param['HP_particle_border_mult'], two_particle_mult=param['HP_two_particle_mult'],  img_edge_mult = param['HP_img_edge_mult'])                   

    return dataGenerator, img_names, copy.deepcopy(masks)


#################################################################################################################       
#################################################################################################################      
########################    Generator Class   ################################################################### 
#################################################################################################################   
#################################################################################################################   




    
# generator class that generates img - label pairs from a given img, row/col arrays, snapshot half size, and labels
class BBinputGenerator(tf.keras.utils.Sequence):
  
    def __init__(self, S_half=20, batch_size=32, flip=True, rotate=True, scales=1, scalefactor=2, randomize = True, neighborhood_radius=def_neighborhood_radius,dist_max = def_dist_max):
        
       
        self.S2 = S_half
        self.S = S_half*2+1
        self.batch_size = batch_size
        self.flip = flip
        self.rotate = rotate
        self.mult = 1*(1+3*rotate)*(1+flip) # flipping doubles, rotating 4x's unique images
        self.randomize = randomize
        self.neighborhood_radius = neighborhood_radius # neighborhood for weighting and erosion if needed
        self.getcount = 0
        self.maxgetcount = 100 # calls garbage collection every maxgetcount calls to __getitem__
        self.outstack = None
        self.dist_max = dist_max
        self.scales = scales
        self.scalefactor = scalefactor
        self.padding = scalefactor**(scales-1)*S_half+1 # this is how far away the image edge must be from AOI
        self.input_shape = None # defined when first image is added
          
        # flag for changing mask to categorical, randomizing order, and adding weights.
        self.data_prepared = False 
        # this is performed on the first call of __getitem__, and undone whenever new data is added (add_img or add_img_vec)
               
        if not int(batch_size/self.mult) == batch_size/self.mult:
            raise Exception('When manipulation (flipping/rotating) is on, batch size must be multiple of these manipulations.')

        self.dim3 = None # no images added yet!

       
        # Placeholder fields to be filled with add_image() method
        self.imgs = [] # first index is image number
        self.img_paddings = []
        
        
        # vectors to create output data (must always keep same order as each other)
        self.img_num = []
        self.rows = []
        self.cols = []
        self.sample_weights = None # to be added by the first __getitem__ call
        self.relative_weights = [] # list of relative weights for innies and outies (e.g. edges are double insides)
        self.label_list = [] # innie vs outie
        self.labels = None # this is in "categorical" terms, to be added by the first __getitem__ call
        
        self.dist_list = [];
        
    
    # Add an image to the dataset (along with mask (innie/outie), AOI) 
    # Uses add_img_vec after processing 2D data.
    def add_img(self,img, mask, AOI=None, weights=None):
         
        img = np.copy(img)
                
        shp = np.shape(img)[:2]
        # reshape to 3D (with 3rd dimension 1), and record 2D shape
        if len(np.shape(img)) == 2:
            img = np.reshape(img,(shp[0],shp[1],1))
        
        # generate dummy mask and total AOI if needed
        if mask is None: # if not provided, we just need a space filler
            mask = np.ones(shp)  
        if AOI is None: # if not provided, everything is in the area of interest
            AOI = np.ones(np.shape(mask))   

        if (not shp == np.shape(mask)) or (not shp == np.shape(AOI)):
            raise Exception('img, mask, AOI, must have the same 2D shape')
                                                                     
        rowmat,colmat = np.mgrid[0:np.size(img,0),0:np.size(img,1)]
        
        # flatten AOI and check for multiple counts on a single pixel.
        # pixels will be repeated in the generator n = AOI[r,c] times
        AOI = np.ndarray.flatten(AOI)
        idx_used = []
        while np.max(AOI)>0:
            AOIbool = AOI>0
            AOI[AOIbool] -=1
            idx_used += list(np.argwhere(AOIbool))

        idx_used = np.ndarray.flatten(np.array(idx_used))
        
        # flatten all matrices, and use AOI to trim/duplicate indicees correctly
        rows = np.ndarray.flatten(rowmat)[idx_used]
        cols = np.ndarray.flatten(colmat)[idx_used]
        labels = np.ndarray.flatten(mask)[idx_used]
        
        if not weights is None:
            weights = np.ndarray.flatten(weights)[idx_used]
        
        # use method below to add the data to the generator!
        self.add_img_vec(img, rows, cols, labels, weights)     
        
        gc.collect()
    
    # Add an image to the dataset (along with rows, columns, and labels of points to generate data from)
    def add_img_vec(self,img, rows, cols, labels, weights=None):
        
        if (not len(rows)==len(labels)) or (not len(cols) == len(labels)):
            raise Exception('rows, cols, and labels must have the same length L and be Lx1')
        
        shp = np.shape(img)
        if len(shp) == 2:
            self.imgs.append(np.copy(img).reshape(shp[0],shp[1],1))
            shp = np.array([shp[0],shp[1],1])
        else:
            self.imgs.append(np.copy(img))
          
      
        if self.dim3 is None:
            self.dim3 = shp[2]
            self.input_shape = self.S,self.S,self.scales*self.dim3

        elif not self.dim3 == shp[2]:
            raise Exception('Added images must have same 3rd dimension: '+str(self.dim3))
            
        if weights is None:
            weights = np.ones(np.shape(rows)).astype(float)
          
        
        # determine how far away the AOI is from the edges of the image 
        # (for rows and columns separately) then add padding to the image
        toppad = max(self.padding-np.min(rows),0)
        bottompad = max(self.padding-(shp[0]-np.max(rows)),0)
        leftpad = max(self.padding-np.min(cols),0)
        rightpad = max(self.padding-(shp[1]-np.max(cols)),0)
        padinstruction = ((toppad, bottompad), (leftpad, rightpad),(0,0)) # 3D 
        
        self.imgs[-1] = np.pad(self.imgs[-1],padinstruction)
        self.img_paddings.append(padinstruction)
        
        rows = np.array(rows)+toppad # correct rows and columns  (labels are unchanged)
        cols = np.array(cols)+leftpad
        
            
        # add blurred version(s) of image to accomodate easy fetching of scaled up images    
        # note that padding happens before blurring. However because blurred images are always subsampled
        # at the same frequency as the blur, this order does not effect the eventual outputs.
        baseimg = np.copy(self.imgs[-1])
        for scale in range(1,self.scales): # for every scale except 1 (powers of scalefactor)
            
            n = self.scalefactor**scale
            newimg = np.zeros(np.shape(baseimg))
            
            # blur at many scales and stack (concatenate) the images along the color axis
            for dim in range(self.dim3):   
                newimg[:,:,dim] = convolve2d(baseimg[:,:,dim],np.ones((n,n))/(n**2), boundary='symm', mode='same')
            
            self.imgs[-1] = np.concatenate([self.imgs[-1],newimg],axis=2)
            
        
        self.rows = np.concatenate([self.rows,rows], axis=0).astype(int)
        self.cols = np.concatenate([self.cols,cols], axis=0).astype(int)
        self.img_num = np.concatenate([self.img_num,[len(self.imgs)-1 for _ in rows]], axis=0).astype(int)
        self.label_list = np.concatenate([self.label_list, labels], axis=0).astype(int)
        self.relative_weights = np.concatenate([self.relative_weights, weights], axis=0).astype(float)
        
        self.data_prepared = False # adding more images makes the entire data set no longer prepped!
    
    
     # Add a mask with automatically generated weights near borders, etc. Can add ID'd mask for higher weighting of regions at  particle contacts.
    def add_weighted_img(self,img, mask, AOI=None, neighborhood_radius=def_neighborhood_radius, particle_border_mult=def_particle_border_mult, two_particle_mult=def_two_particle_mult, img_edge_mult = def_img_edge_mult,weights=None):
        
        # create eroded mask and appropriate weights
        if weights is None:
            if particle_border_mult==1 and two_particle_mult == 1 and img_edge_mult == 1: 
                # no need to do anything fancy if just 1's
                weights = np.copy(AOI)
            else:
                weights, _ = weight_and_erosion_maker(mask, neighborhood_radius=neighborhood_radius, \
                                        particle_border_mult=particle_border_mult, two_particle_mult=two_particle_mult, \
                                        img_edge_mult = img_edge_mult, AOI=AOI)
            # if ID'd mask, make binary
        self.add_img(img, mask>0, AOI=AOI, weights=weights)
    
    # returns a 2D map of relative weights for image idx (with any added padding removed)
    def get_weights(self,idx=0): 
        out = np.zeros(np.shape(self.imgs[idx])[:2]) # same 2D shape as image, no channels.
        want = self.img_num == idx
        out[self.rows[want],self.cols[want]] = self.relative_weights[want]
        p = self.get_padding_array(idx)
        return out[p[0]:p[1],p[2]:p[3]]
    
    # returns a 2D map of innie vs outie mask (with any added padding removed)
    def get_masks(self,idx=0): 
        out = np.zeros(np.shape(self.imgs[idx])[:2]) # same 2D shape as image, no channels.
        want = self.img_num == idx
        out[self.rows[want],self.cols[want]] = self.label_list[want]
        p = self.get_padding_array(idx)
        return out[p[0]:p[1],p[2]:p[3]]
    
    def get_dists(self,idx=0): 
        out = np.zeros(np.shape(self.imgs[idx])[:2]) # same 2D shape as image, no channels.
        want = self.img_num == idx
        out[self.rows[want],self.cols[want]] = self.dist_list[want]
        p = self.get_padding_array(idx)
        return out[p[0]:p[1],p[2]:p[3]]
    
    # returns a 2D map of Area of Interest (AOI) for image idx (with any added padding removed)
    def get_AOI(self,idx=0): 
        out = np.zeros(np.shape(self.imgs[idx])[:2]) # same 2D shape as image, no channels.
        want = self.img_num == idx
        out[self.rows[want],self.cols[want]] = 1
        p = self.get_padding_array(idx)
        return out[p[0]:p[1],p[2]:p[3]]
    
    # returns an input image with any added padding removed
    def get_img(self,idx=0):      
        p = self.get_padding_array(idx)
        return self.imgs[idx][p[0]:p[1],p[2]:p[3],:self.dim3] # cut out scalings
    
    # returns array with indicees required to remove padding from an image
    def get_padding_array(self,idx):
        p = np.ndarray.flatten(np.array(self.img_paddings[idx]))
        p[1] = np.shape(self.imgs[idx])[0]-p[1]
        p[3] = np.shape(self.imgs[idx])[1]-p[3]
        return p
    
    # returns all images with any added padding removed
    def get_all_imgs(self):
        outimgs = []
        for idx in range(np.shape(self.imgs)[0]):
            outimgs.append(self.get_img(idx))
        
        return outimgs
    
    # returns max and minimum value of all images
    def get_img_range(self):
        maxval,minval = np.max(self.get_img(0)),np.min(self.get_img(0))
        for idx in range(1,np.shape(self.imgs)[0]):
            maxval = np.max([maxval,np.max(self.get_img(idx))])
            minval = np.min([minval,np.min(self.get_img(idx))])
        
        return maxval,minval
    
    # normalize all images in this generator -- sends maxval to 0.5 and minval to -0.5
    def normalize_images(self,maxval,minval): 
        for idx in range(np.shape(self.imgs)[0]):
            self.imgs[idx] = (self.imgs[idx]-minval)/(maxval-minval) - 0.5
        
        
        
    # returns an array of 2D predictions for each image in self.imgs, based on the model given.
    def get_prediction(self,model):
        
        #Models with these features will generate multiple predictions per pixel and will not match with the 
        # internal arrays like self.rows/self.cols. It will be massively inefficient and is thus prevented.
        if self.rotate or self.flip:
            raise Exception('This generator has flips or rotates and should not be used as a prediction generator.')
            
        #print('Predicting can be a time-consuming step.')
        
        # do the prediction!     
        outputs = model.predict(self)
        class_vec = outputs[0][:,1] # keep only one side of the class output (innie/outie)
        dist_vec = np.squeeze(outputs[1]) # distance to outie
        

        predictions = []
        for idx in range(np.shape(self.imgs)[0]): # for every image, find predictions
            #print('exporting image #'+str(idx))
            want = self.img_num == idx # identify which elements are for this image

            dummy_img = np.zeros(np.shape(self.imgs[idx])[:2]) 
            dummy_img[self.rows[want],self.cols[want]] = class_vec[want] # place predictions in 2D
            


            # remove padding
            p = np.ndarray.flatten(np.array(self.img_paddings[idx]))
            p[1] = np.shape(self.imgs[idx])[0]-p[1]
            p[3] = np.shape(self.imgs[idx])[1]-p[3]
            
            # add distances to output
            dummy_img2 = np.zeros(np.shape(self.imgs[idx])[:2]) 
            dummy_img2[self.rows[want],self.cols[want]] = dist_vec[want] # place predictions in 2D

            dummy_img = np.stack([dummy_img,dummy_img2],axis=2)
            
            # add to overall output array
            predictions.append(dummy_img[p[0]:p[1],p[2]:p[3],:])
           
        return predictions
        
        
    # randomly mix the order of data
    def shuffle(self):
        
        ordr = np.arange(len(self.rows))
        np.random.shuffle(ordr)

        self.label_list = self.label_list[ordr]
        self.dist_list = self.dist_list[ordr]
        self.rows = self.rows[ordr]
        self.cols = self.cols[ordr]
        self.sample_weights = self.sample_weights[ordr]
        self.relative_weights = self.relative_weights[ordr]
        self.img_num = self.img_num[ordr]
        
 
    # return the number of batches to pull for an epoch of training
    def __len__(self):
        return np.ceil(self.mult*len(self.label_list)/self.batch_size).astype(int)
  
    # return a batch of data. This may include rotations and flips, as well as multiple channels and/or scales
    def __getitem__(self, idx00):
        
        self.getcount += 1
        if self.getcount >= self.maxgetcount:
            gc.collect()
            self.getcount=0
        
        # Check to see if the data is prepped. If not... prep!
        if not self.data_prepared: # happens only on the first call.

            self.enforce_dist_max() # crops label list at self.dist_max
            self.dist_list = copy.deepcopy(self.label_list); # store distances
            
            self.label_list = self.label_list>0        # create binary categorization
            
            # generate weights. Equal total weight for both innies and outies (and this is built to allow more classes)
            self.sample_weights = np.zeros(np.shape(self.rows))
            U = np.unique(self.label_list)
            for k in U:
                want = self.label_list==k # identify elements of this class
                # sum of all elements of each class is equal
                self.sample_weights += np.multiply(self.relative_weights,want)/np.sum(self.relative_weights[want])

            self.sample_weights = self.sample_weights/np.mean(self.sample_weights) # make mean weight = 1

            if self.randomize: # if randomization is in the cards, deal.
                self.shuffle()
        
            self.labels = to_categorical(self.label_list) # make Keras-friendly labels
            
            self.data_prepared = True # data is now prepped.
        
        
        # identify what the unique index and number of unique elements in the batch is (repeats are flips/rots)
        idx0 = idx00/self.mult
        batch = self.batch_size/self.mult
        
        # Indices of the data to be pulled.
        # Repeat self.mult times and sort to give an ordered list [32,32,...,32,83,83...,83,120,...]
        vec = sorted(np.repeat(np.arange((idx0*batch),np.min([(idx0+1)*batch,len(self.labels)]),dtype='int'),self.mult))

        # Scales to pull. If self.scales=1, this will just be 1.
        scalevec = np.power(self.scalefactor,np.arange(self.scales)) # e.g. 1,2,4,8, etc
     
        # create stacked image output. 
        # Images may have multiple channels (e.g. colors) and scales: these are stacked in the same dimension (3 in this stack). 
        # There will also be self.mult repeats of every such color and scale stack (for later rotation/flipping).
        # Output stack dimensions are [self.batch_size,self.S,self.S,self.dim3*self.scales] with [batch] unique elements.

        self.outstack = np.stack([ \
            np.concatenate([ \
                self.imgs[self.img_num[idx]][self.rows[idx]-self.S2*s:self.rows[idx]+self.S2*s+1:s, \
                             self.cols[idx]-self.S2*s:self.cols[idx]+self.S2*s+1:s, \
                             self.dim3*snum:self.dim3*(snum+1)].reshape(self.S,self.S,self.dim3) \
                for s,snum in zip(scalevec,range(self.scales))], axis=2) for idx in vec] ,axis=0)
                     
        # If rotations/flips are used, go through stack and modify unaltered images to create these new data.
        if self.flip or self.rotate:
            multiscaledim3= self.scales*self.dim3

            # rotate and flip depending on idx (which indicates which repeat # of a given environment this is)
            for idx in range(len(vec)):
                # reshape image so that it can use numpy rotation/flip functions
                snap = self.outstack[idx,:,:,:].reshape(self.S,self.S,multiscaledim3)
                
                # rotate 90 degrees for every additional index (stack is sorted by index, so this achieves desired effect)
                if self.rotate:
                    snap = np.rot90(snap,idx%4)
                
                # flip half of the images if self.flip
                if self.flip and (idx%self.mult)>=self.mult/2:
                    snap = np.fliplr(snap)
                    
                # reshape to put back into stack
                self.outstack[idx,:,:,:] = snap.reshape([1,self.S,self.S,multiscaledim3])       
        
        return  np.copy(self.outstack), [np.array(self.labels[vec]),np.array(self.dist_list[vec])], np.array(self.sample_weights[vec])
       
    # Plot sample items created by this generator
    def plot_items(self,count=1,nums=None):
        
        if nums is None:
            nums = (self.__len__()*np.random.rand(count)/self.mult).astype(int)
        else:
            count = len(nums)
            
        c = self.scales+self.rotate+self.flip
        fig, axs = plt.subplots(count,c)

        for r0 in range(count):
            snap,inorout,_ = self.__getitem__(nums[r0]/self.mult)
            

            for s in range(self.scales):
                snap0 = snap[0,:,:,s*self.dim3].reshape(self.S,self.S)
                theplot = axs[r0][s]
                theplot.imshow(snap0,cmap=plt.get_cmap('gray'))
                
                if r0==0:
                    x = str(self.scalefactor**s)
                    theplot.set_title(x+'x')
                    
                if s==0:
                    val = inorout[0][s][1]
                    
                    if val == 1:
                        theplot.set_ylabel('Innie')
                    else:
                        theplot.set_ylabel('Outie')

                            
            if self.rotate: # add one rotation if that's happening
                snap0 = snap[1,:,:,0].reshape(self.S,self.S)
                theplot = axs[r0][c-self.flip-1]
                theplot.imshow(snap0,cmap=plt.get_cmap('gray'))
                if r0==0:
                    theplot.set_title('Rotate')
                        
            if self.flip: # add one flip if that's happening
                snap0 = snap[(1+3*self.rotate),:,:,0].reshape(self.S,self.S)
                theplot = axs[r0][c-1]
                theplot.imshow(snap0,cmap=plt.get_cmap('gray'))
                if r0==0:
                    theplot.set_title('Flip')
                
            for c0 in range(c):
                axs[r0][c0].set_xticks([])
                axs[r0][c0].set_yticks([])
                axs[r0][c0].plot(self.S2,self.S2,'r+')
            
            axs[r0][-1].yaxis.set_label_position("right")
            axs[r0][-1].set_ylabel('#'+str(nums[r0]*self.batch_size))

    
    # Plot sample items created by this generator
    def enforce_dist_max(self):
        self.label_list[self.label_list>self.dist_max] = self.dist_max

        


# creates a BBinputGenerator but with no flipping, rotation, or randomization
class predictionGenerator(BBinputGenerator):
    
    # Removed options that are no-go's in prediction
    def __init__(self, S_half=20, batch_size=1024, scales=1, scalefactor=2):
        super().__init__(S_half=S_half, batch_size=batch_size, flip=False, rotate=False, scales=scales, scalefactor=scalefactor, randomize = False)
        
    # allow for adding an image without a mask (just uses zeros as mask) 
    def add_maskless_img(self,img, AOI=None):      
        mask = np.zeros(np.shape(img)[:2])
        super().add_img(img, mask, AOI=AOI, weights=None)

        
        
# Identical to BBinputGenerator except with add_ID_img() method that erodes ID'd mask and weights appropriately
class HPinputGenerator(BBinputGenerator):
    
    # Removed options that are no-go's in prediction
    def __init__(self, S_half=20, batch_size=32, flip=True, rotate=True, scales=1, scalefactor=2, randomize = True,dist_max=def_dist_max):
       
        super().__init__(S_half=S_half, batch_size=batch_size, flip=flip, rotate=rotate, scales=scales, scalefactor=scalefactor, randomize = randomize,dist_max = dist_max)
        
    # add ID'd mask, which includes generating distances to outies
    def add_ID_img(self, img, mask=None, AOI=None, \
                   neighborhood_radius=def_neighborhood_radius, \
                   particle_border_mult=def_particle_border_mult, \
                   two_particle_mult=def_two_particle_mult, \
                   img_edge_mult = def_img_edge_mult):
        
        if mask is None:
            self.add_img(img, None, AOI=AOI, weights=None)
        
        else:
            # create eroded mask and appropriate weights
            weights, _ = weight_and_erosion_maker(mask, neighborhood_radius=neighborhood_radius, \
                        particle_border_mult=particle_border_mult, two_particle_mult=two_particle_mult, \
                        img_edge_mult = img_edge_mult, AOI=AOI)

            self.add_img(img, distance_maker(mask), AOI=AOI, weights=weights)



#################################################################################################################       
#################################################################################################################      
########################    Helper Functions   ##################################################################
#################################################################################################################   
#################################################################################################################   




# creates a 2D weight map (not balanced for innies vs outies yet) and eroded mask based on an ID'd mask. Mult variables determine additional weighting for regions near the borders of particles, near two particles, and near the image borders (neighborhood defined by r).
def weight_and_erosion_maker(mask, neighborhood_radius=def_neighborhood_radius, particle_border_mult=def_particle_border_mult, two_particle_mult=def_two_particle_mult, img_edge_mult = def_img_edge_mult, AOI=None):

    img2 = np.sqrt(np.array(primelist[np.array(mask,dtype=int)],dtype=float)) # make segmented image using only primes, then take sqrt
    kernel = np.array(circ_region(neighborhood_radius),dtype=np.uint8)

    # convolve image over kernel -- those that catch another sqrt(prime) will become non-integers when multiplied by sqrt(prime) img
    eroded_mask = np.multiply(cv2.filter2D(img2,-1,kernel),img2) 

    # detect non integers - these are innies next to other particles
    twoParticles = np.abs(eroded_mask-np.round(eroded_mask))>.000001  #this number is somewhat arbitrary. Too small is a problem though.

    eroded_mask = cv2.erode(np.array(img2>0,dtype=float),kernel) - twoParticles # erode and subtract particle-particle borders
    eroded_mask[eroded_mask<0] = 0 # any subtractions below 0 from twoParticles are ignored.

    dilated_mask = cv2.dilate(np.array(img2>0,dtype=float),kernel)

    if AOI is None:
        AOI = np.ones(mask.shape[:2])


    # implement all weightings
    weights = np.array(AOI >0,dtype=float) #weights nonzero only in AOI

    weights[(dilated_mask-eroded_mask)>0] *= particle_border_mult
    weights[twoParticles] *= two_particle_mult
    weights[:int(neighborhood_radius),:] *= img_edge_mult
    weights[-int(neighborhood_radius):,:] *= img_edge_mult
    weights[:,:int(neighborhood_radius)] *= img_edge_mult
    weights[:,-int(neighborhood_radius):] *= img_edge_mult
                          
    return weights, eroded_mask


# takes a binary (eroded) mask, identifies connected regions, then dilates them according to given radius. Outputs ID'd dilated mask. Note that the order of dilation matters in some cases.
def dilate_and_ID_mask(eroded_mask,neighborhood_radius=def_neighborhood_radius):
    
    # determine connected regions in eroded mask
    connectedmask = np.ubyte(np.array(eroded_mask,dtype=float))
    (numregions, finlabels, _, _) = cv2.connectedComponentsWithStats(connectedmask)
  
    # create neighborhood mask and integer length scale
    C = circ_region(neighborhood_radius)
    R = int(np.floor(neighborhood_radius))

    IDmask_padded = np.pad(np.zeros(np.shape(finlabels)),((R, R), (R, R)),'constant', constant_values=-1 )
    # pad eventual IDmask to avoid edge issues (for now)

    # cycle through entire image, painting neighborhoods
    for r in range(np.shape(finlabels)[0]):
        for c in range(np.shape(finlabels)[1]):
            if finlabels[r,c]>0: # if this is in a connected region, dilate!
                current_neighborhood = IDmask_padded[r:r+2*R+1,c:c+2*R+1]
                current_neighborhood[C]= finlabels[r,c]

                IDmask_padded[r:r+2*R+1,c:c+2*R+1] = current_neighborhood # label all elements in neighborhood with value
            
    
    return IDmask_padded[R:-R,R:-R]
    


def fractionalize_AOI(masks, AOIs, fraction):

    train_innie,train_total = 0,0
    
    for k,AOI_dummy in enumerate(AOIs):
        train_innie += np.sum((masks[k]>0)*AOI_dummy)
        train_total += np.sum(AOI_dummy)
        
    train_innie_frac = train_innie/train_total
    train_delta = [(fraction/2)/(1-train_innie_frac) - 1  ,  (fraction/2)/train_innie_frac - 1] # [0] is outie, [1] is innie

    print('[BB] Balance classes by over/under sampling: innies x',np.round(train_delta[1]+1,2),' outies x',np.round(train_delta[0]+1,2))
    
    for k in range(len(AOIs)):

        train_delta2 = copy.copy(train_delta)
        checkmask = masks[k] > 0
        AOImask = 0*AOIs[k] + 1

        for inout in [0,1]:

            while train_delta2[inout] >=1: # if small enough, one class may need to be added multiple times
                train_delta2[inout] -=1
                AOImask[checkmask==inout]+=1

            # adjust by remaining delta, so that innies are now fraction/2 of AOI (may be counted 2 or more times per px)
            dummy = np.random.choice([0, 1], size=AOImask.shape, p=(1-abs(train_delta2[inout]), abs(train_delta2[inout])))
            # add to AOI count if positive adjustment, subtract from it if negative (np.sign)
            
            AOImask[checkmask==inout] += (np.sign(train_delta2[inout])*dummy[checkmask==inout]).astype(int)
                                    
        AOIs[k]  = AOImask*AOIs[k]

                                    
    
    return AOIs


  