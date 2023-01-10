# Bellybutton Image Segmentation (bellybuttonseg)

Bellybutton is a machine-learning-based image segmentation package, designed for ease of use. It uses a convolutional neural network, trained on images and paired user-generated masks, to classify pixels as "innies" or "outies" (inside or outside of a segmented region), and uses watershedding to group these results into individually ID'd regions. Using the package requires no further coding: by editing a text file, users can edit hyper-parameters, modify data augmentation (e.g. rotate images), specify outputs and filetypes, and run training and prediction.

This readme includes:
(A) Quickstart guide with example datasets (run Bellybutton in a few commands).
(B) How to create and train on your own dataset
(C) Adjustable parameters for the algorithm

If you use our package, please cite it as "Bellybutton Image Segmentation Package" and link to this page.
Full author list for Bellybutton project: Sam Dillavou, Jesse Hanlan, Anthony Chieco, Hongyi Xiao, and Douglas J Durian.



## (A) Quickstart guide

Text in ``` this formatting ``` is meant to be entered line by line in the command line / terminal.


### Install Bellybutton

Create a new Python environment, then activate it. Here we named it 'bellybuttonenv' and used [Anaconda](shorturl.at/elRTZ).

```
conda create --name bellybuttonenv
conda activate bellybuttonenv
```

Download Bellybutton package (pip will install most required packages automatically). Install TensorFlow (or appropriate variant for your machine).

```
pip install bellybuttonseg
pip install tensorflow 
```

### Run Bellybutton on an example dataset


#### 1) Download example dataset.
Run the following line. You will be prompted to select a location for files to be saved. 
```
python3 -m bellybuttonseg.example1
```

Note that the directory you selected now has a new folder: PhotoelasticDisks. Inside there folders that hold images (train, test, predict), masks (user-defined segmentation for each training and test image), areas of interest (what regions of each image are relevant), and outputs. We have chosen a simple dataset of three images, each of which we have sub-divided into training and testing regions. Note that copies of each image exist both in the train_images and test_images folder because we are training and testing on different parts of the same images.

#### 2) Train Bellybutton on this dataset.
You will be prompted to select the location of the dataset to train on, in this case select your PhotoelasticDisks folder that you just created. 
```
python3 -m bellybuttonseg.train 
```

The algorithm should run and you can track its progress on the command line/terminal. Bellybutton does the following: loads the training images and their corresponding masks, ignoring regions in each image not specified by corresonding 'areas of interest'. It then trains a neural network to identify whether a pixel is inside or outside of any desired region, and also to predict that pixel's distance to the edge of the region it's in. After training, these predictions allow Bellybutton to create a binary map of 'innies' and 'outies,' and then subdivide the innies into distinct regions using the distance-to-edge prediction.

During this process, a new folder (whose name starts with "output" and is based on the date and time) will be created inside PhotoelasticDisks that will store results. Once training is completed, peruse the results inside this folder. Note that a new used_parameters.txt file has been created inside this outputs folder that stores the parameters used.

To run Bellybutton again using different parameters, edit the base_parameters.txt file inside the PhotoelasticDisks folder before executing the above code line again.

#### 3) Use trained Bellybutton to predict on new images.

You may also use your trained network to segment images in the "PhotoelasticDisks/predict_images" folder by running the following command line, and selecting the output folder that you just trained.
```
python3 -m bellybuttonseg.predict 
```

#### Alternative: 2) and 3) Together

You may instead train a new network and then segment the images in predict_images folder automatically by running:
```
python3 -m bellybuttonseg.trainandpredict 
```


#### A short list of pre-made example datasets:


###### PhotoelasticDisks
Connected disks of plastic that visualize internal forces in the material. Includes 3 images of varying stress, and uses regions within each image for training and testing. 
```python3 -m bellybuttonseg.example1 ```

###### Honeycomb
A hexagonal pattern of clear (also photoelastic) plastic slowly being fractured, changing its structure. In this example there is only one 'region,' the material, but its shape is the desired output. Therefore the selected outputs for this dataset are a binarized innie vs outie image and also distance from the edge of the material.
``` python3 -m bellybuttonseg.example2 ```

###### Spacefoam
[Bubbles in microgravity!](https://arxiv.org/abs/2209.03464) Lighting, focus, bubble size, and other factors all change in this dataset. The default training amount may take a while, but try it for smaller fraction and/or fewer epochs to get a sense of how fast Bellybutton converges. (You can add additional training epochs if training wasn't long enough, see section on "Adding Epochs" below). This was the dataset that inspired the Bellybutton package.
``` python3 -m bellybuttonseg.example3 ```


## (B) How to Create Your Own Dataset
To train (and test) Bellybutton, the user must provide images and segmented masks of example data, and (optionally) areas of interest for each image. Bellybutton will learn from these examples and be able to extrapolate to new data. 

##### Images (Data)
Images may be of varying size and format. Images placed in train_images/ will be used to train the network, images placed in test_images will be used as a benchmark for performance after training. Images in both these folders require corresponding mask files (see below) and optionally can have corresponding area of interest files (see below). Images placed in predict_images/ can be segmented by the algorithm after training. They do not use masks but can (optionally) have area of interest files.

##### Masks (Labels)
Masks must be image (or text) files with the same height and width as the training or testing images they correspond to. Masks for all three image folders are put into the single 'masks/' folder. They must have identical file names (excluding extension) to the images they correspond to (e.g. to pair with train_images/img_001.png, any of masks/img_001.png or masks/img_001.tif or masks/img_001.txt will work, but masks/img_001_mask.png would not). When loading these masks, Bellybutton will find contiguous regions of identical pixel values (e.g. an island of pixel value 17) and consider each such island an individual region. Any pixels with value 0 will be considered 'outies' -- not inside a desired segmentation (such as the background). Note that two separated blobs of the same value will be split into distinct regions -- Bellybutton only evaluates contiguous segments. Therefore, if none of your desired segmentations are in contact with one another, simply labeling all pixels inside a desired region as 1 and all outside as 0 will be sufficient, Bellybutton will treat them as individual regions for you. You do not need to use the same numbering convention from mask to mask, and as long as corresponding masks and images (and areas of interest, see below) are the same size as each other, your data may include images of varying size.

For making masks we suggest your favorite programming language, or directly decorating an image of data using image-editing tools like [Fiji](https://imagej.net/software/fiji/) (e.g. copy the image, paint over desired regions with unique values, zero the rest of the values).

##### Areas of Interest
AOI's are optional, and if no matching AOI is present, every pixel of an image will be considered in the area of interest. If included, AOI's must be image (or text) files with the same height and width as the training, testing, or prediction images they correspond to. When loaded, any pixels with non-zero value will be considered relevant (used for training, testing, and/or predicting), and all pixels with value 0 will not. Note that the AOI does not crop the image, but merely specifies which pixels should be evaluated as inside or outside a region. When training or predicting, Bellybutton will use a square patch around the pixel in question to determine its innie or outie status, and this patch may extend beyond the AOI. If it would extend beyond the boundary of the image, Bellybutton fills that extra space with black pixels.

##### Default AOI
If an image is included in the areas_of_interest/ folder with the base name 'default' (e.g. 'default.png' or 'default.tif'), it will be used for every image (training, testing, and predicting) that does not have its own matching AOI. Note that an error will result if this default AOI's dimensions do not match these images.


### Run Bellybutton on your own dataset

#### 1) Create a directory and bellybutton-required folders. 
```
python3 -m bellybuttonseg.newproject
```
When the code line above is run, you will be prompted to select a location and then prompted to enter a name for the base folder of this project (`[INPUT] Enter folder name for new project:`), which you can always rename later.

#### 2) Move your data into the newly created folders
To run bellybutton, the images, masks, and (optionally) areas of interest must be moved into the appropriate folders created in this new directory. Remember that images in train_images/ and test_images/ must have corresponding masks.

#### 3) Edit parameters (optional)
The configuration of the Bellybutton algorithm is specified using the text file 'parameters.txt' which was automatically generated with default values inside the base folder. A detailed list of parameters is below. These parameters control the input data augmentation, the number of datapoints (pixels) used, the output types, the number of training epochs, etc.

#### 4) Train Bellybutton on your data.
```
python3 -m bellybuttonseg.train
```
You will be prompted to select your project folder. After training, results from the training and test set will be stored inside a new output folder inside your project's base folder, along with statistics in the text file 'results.txt'. You may always edit the base_parameters.txt file in your base folder and run Bellybutton again with the new parameters using the above line.

#### 5) Predict using Bellybutton
After training, Bellybutton can segment images without providing a mask. Place images in the predict_images folder, then run
```
python3 -m bellybuttonseg.predict 
```
and select the output folder for the network you just trained. Voila!

#### Alternative: 4) and 5) together
```
python3 -m bellybuttonseg.trainandpredict 
```
Select the base folder, and Bellybutton will create a new output folder, train, and then predict on predict_images/ files.

#### Adding Epochs

After training, you can train longer (add more epochs) by running ``` python3 -m bellybuttonseg.train ``` (or ```.trainandpredict```) and selecting the output folder of the already trained network. This action will load the trained network and begin training from where it left off, training for the number of epochs specified in the base_parameters.txt file. Note that if a subset of pixels is used in training (see fraction and balance_classes parameters below), it will be a different random subset from the previous training. A new output file will be created with the old output file's name, plus a new date and time. You can repeat this as many times as you'd like, but the file names will get a little long :)

Note that certain parameters should not be changed, namely s_half, scales, scalefactor, images_to_grayscale, or track_outies, as these either change the structure of the network, or modify the data intake in a way that makes the previous training incorrect. Changing these will result in an error.

Enjoy!


## (C) Adjustable Parameters

There are a handful of hyper-parameters that adjust how Bellybutton runs. However, it has sufficient flexibility to be insensitive to many changes in these values. The values that are most likely to require modification are listed first, and explained as follows.

#### Data Augmentation (flips, rotations):
##### TL;DR: duplicates training data by flipping and rotating - useful to prevent overfitting.
During training by default, Bellybutton rotates and flips the input images to produce additional training data. If the system in question has these symmetries (i.e. if the objects being identified will rotate, or has a shape symmetric to flips like a circle or oval), then including these data augmentations may reduce overfitting. If the objects you are segmenting do not have these symmetries (e.g. they are always upright teardrop shapes), then these augmentations may make the problem harder to solve than is needed.

#### Input Image Patch Sizes (scales, scalefactor):
##### TL;DR: Number of scales and zoom factor for each that is fed into network - make big enough to see important features of data
When the data are fed into the neural network, a 25x25 pixel neighborhood centered at the pixel in question is used. In addition, zoomed out, down-sampled regions are included as an input, with the total number of slices (including the original 25x25) equal to the Scales parameter. The difference in zoom between each of these slices is given by the Scalefactor. A user should make sure that the largest object to be identified fits within the largest scale, that is, it is smaller than $25*\text{[Scalefactor]}^{\text{[Scales]}-1}$. Bellybutton automatically fills any window that extends beyond the end of the image with black pixels.

#### Particle Border Size (dist_max):
##### TL;DR: minimum radius of segmented particles or 10, whichever is smaller.
As part of Bellybutton's algorithm, it is asked to predict distance of each pixel inside a particle to the nearest pixel not in that particle. This helps eventually segment particles in contact. This distance is capped at [Max Distance]. This value can be thought of as a minimum radius for particles, although Bellybutton may capture some particles below this radius. Good practice is to make this value a few pixels smaller than the minimum radius of the smallest desired regions. We find a value any higher than 10 pixels is unnecessary, as the values closest to the edges are most important. Note that Bellybutton is not designed for regions smaller than a handful of pixels across, but may still perform well in some cases. Typically, if your desired regions are too small or thin, Bellybutton will subdivide what should be continuous regions. In this case, for your mask consider using these regions plus an added border to give them enough size.

#### Amount of Training (fraction, epochs):
##### TL;DR: the fraction of training pixels used, and number of times network is exposed to each training pixel - the product of the two is typically the important parameter 
Often, not all of the user-generated data is necessary to train the network. Systems with thousands of particles in a single image may not even require a full image for training. Users should select their data to include as many lighting, rotation, focus, shape, and size scenarios as possible, but this may result in a massive number of pixels in the training set. The Fraction parameter indicates what fraction (0 to 1) of pixels identified as training data should be used to train the network. That data is then run through the network the number of times specified by Epochs. We find that in many systems, the important parameter is [Fraction] * [Epochs], and that, provided the Fraction is not too low (typically above 5-10\%, to give the network exposure to most situations), the trade-off between these two numbers is minimal. Users should train their networks until the test set error reaches a plateau, which will likely happen in fewer Epochs when more training data is used. If Bellybutton is predicting badly with a fraction less than 1, consider raising this parameter.

#### Balance Training Data (balance_classes):
##### TL;DR: Balance innies and outies for training? 1 yes, 0 no. Use 1 unless an overwhelming majority of pixels are innies.
When training a network for classification, as we do here for determining whether a pixel is inside or outside an identified region, networks train better when exposed to equal quantities of training examples of each class. Selecting 1 for this parameter will ensure that innies and outies are evenly represented in the training set. This may result in using some examples of one class more times than once per epoch. Selecting 0 will ignore class when selecting training data, and classes will be represented in proportion to the rate they appear in the training images. Selecting a value in between (e.g. 0.5) interpolates between the two extremes (e.g. some but not total balancing of class quantities). Note that this parameter does not change the total number of training datapoints used. It is best to balance classes (use 1) unless the vast majority are innies, in which case the network will not get sufficient exposure to the most important pixels, as they will be very undersampled to balance the classes.

#### Outputs and Output File Types (output_segmented, output_binarized, output_dist, output_classprob, output_markers, save_to_png, save_to_npy):
##### TL;DR: save_to_ values designate what file types are saved - images (png), data (npy). output_ values designate what is saved (segmented regions, classification probability, etc)
All of these options are true (1) or false (0). Any combination may be selected, including all or none of them. Output types are as follows.
output_segmented: segmented regions (default output)
output_dist: distance to outside a particle (used for distinguishing touching regions)
output_classprob: probability that bellybutton thinks a pixel is inside vs outside a particle
output_binarized: a binarized version of that probability
output_markers: the starting points for the watershedding algorithm (primarily for debugging). 

Note that if output_segmented is 0, Bellybutton will not go through the segmentation process, and results.txt values will be calculated based on the binary output (inside-or-outside of a region).

#### Parameters that probably do not need adjusting:

##### Input Width (s_half):
Size of smallest window fed into network. S_half=12 is default, which means the window is $2*S_{half}+1 = 25$ pixels wide. S_half of 9 and above will run with no errors. Note that because this changes the size of the images fed into the network, it also changes the size of the network itself. Therefore, making this value larger may result in slower training or memory issues, and making it smaller may result in decreased prediction capacity.

##### batch_size:
Number of training examples fed to network at once. Default is 1024. Must be multiple of 8.

##### track_outies (default 0):
If 1, Bellybutton inverts the masks given and treats contiguous regions of 0's as if they were labeled as 1's.




