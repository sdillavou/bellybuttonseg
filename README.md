# Bellybutton Image Segmentation (bellybuttonseg)

Bellybutton is a machine-learning-based image segmentation package, designed for ease of use.
This package uses a convolutional neural network, trained on images and paired user-generated masks, and outputs individually ID'd regions.
Using the package requires no writing code; by editing a text file, users can edit hyper-parameters, modify data augmentation (e.g. rotate images), specify outputs and filetypes, and run training and prediction.


## Quickstart guide

### Install Bellybutton from command line / terminal

Create a new python environment, here we named it 'bellybuttonenv' ([e.g. using anaconda](shorturl.at/elRTZ))

``` conda create --name bellybuttonenv ```

Activate this environment (here using anaconda)

``` conda activate bellybuttonenv ```

Download Bellybutton package (pip will install most required packages automatically)

``` pip install bellybuttonseg ```

Install tensorflow 

``` pip install tensorflow ```


### Run Bellybutton on an example dataset


Create example dataset. You will be prompted to select a location for files to be saved. 
```
python3 -m bellybuttonseg.example1
```

Note that the directory you selected now has a new folder: PhotoelasticDisks. Inside there folders that hold images (train, test, predict), masks, areas of interest, and outputs. We have chosen a simple dataset of three images which we have divided into training and testing regions. Note that copies of each image exist both in the train_images and test_images folder because we are training and testing on different parts of the same images.

Run Bellybutton on this dataset. You will be prompted to select the location of the dataset to train on, in this case select your PhotoelasticDisks folder that you just created. 
```
python3 -m bellybuttonseg.train 
```

The algorithm should run and you can track its progress on the command line/terminal. A new folder (whose name starts with "output" and is based on the date and time) will be created inside PhotoelasticDisks that will store results. Once training and prediction are completed, peruse the results inside this folder. Note that a new used_parameters.txt file has been created inside this outputs folder that stores the parameters used.

To run Bellybutton again using different parameters, edit the base_parameters.txt file inside the PhotoelasticDisks folder before executing the above code line again.

You may also use your trained network to segment images in the "PhotoelasticDisks/predict_images" folder by running the following command line, and selecting the output folder that you just trained.
```
python3 -m bellybuttonseg.predict 
```

You may instead train a new network and then segment the images in predict_images folder automatically by running:
```
python3 -m bellybuttonseg.trainandpredict 
```


### Run Bellybutton on your own dataset

- Create a directory and bellybutton-required folders. 
```
python3 -m bellybuttonseg.newproject
```
When the code line above is run, you will be prompted to select a location and then prompted to enter a name for the base folder of this project (`[INPUT] Enter folder name for new project:`), which you can always rename later.

 - Move your images, masks, and (optionally) areas of interest into the appropriate folders created in this new directory. Note that masks and area of interest files must have identical base names to the images they correspond to (image image1.png and mask and image1.txt is allowed). 

 - Edit the parameters inside the parameters.txt file if desired (see above section for details).

 - Train Bellybutton on your data. You will be prompted to select your project folder after running the below command.

```
python3 -m bellybuttonseg.train
```

Results from the training and test set will be stored inside a new output folder inside your project's base folder. You may always edit the base_parameters.txt file in your base folder and run Bellybutton again with the new parameters using the above line.

To segment images without a mask, place them in the predict_images folder, and run
```
python3 -m bellybuttonseg.predict 
```
and select the output folder for the network you just trained.


Alternatively, do the entire training and prediction process in one step with
```
python3 -m bellybuttonseg.trainandpredict 
```
and select the base folder (new output folder will be created).


Enjoy!


### Adjustable Parameters

There are a handful of hyper-parameters that adjust how Bellybutton runs. However, it has sufficient flexibility to be insensitive to many changes in these values. The values that are most likely to require modification are listed first, and explained as follows.

#### Flips, Rotations:
##### TL;DR: duplicates training data by flipping and rotating - useful to prevent overfitting.
During training by default, Bellybutton rotates and flips the input images to produce additional training data. If the system in question has these symmetries (i.e. if the objects being identified will rotate, or has a shape symmetric to flips like a circle or oval), then including these data augmentations may reduce overfitting.

#### Scales, Scalefactor:
##### TL;DR: Number of scales and zoom factor for each that is fed into network - make big enough to see important features of data
When the data are fed into the neural network, a 25x25 pixel neighborhood centered at the pixel in question is used. In addition, zoomed out, down-sampled regions are included as an input, with the total number of slices (including the original 25x25) equal to the Scales parameter. The difference in zoom between each of these slices is given by the Scalefactor. A user should make sure that the largest object to be identified fits within the largest scale, that is, it is smaller than $25*\text{[Scalefactor]}^{\text{[Scales]}-1}$. Bellybutton automatically fills any window that extends beyond the end of the image with black pixels.

#### Max Distance:
##### TL;DR: minimum radius of segmented particles or 10, whichever is smaller.
As part of Bellybutton's algorithm, it is asked to predict distance of each pixel inside a particle to the nearest pixel not in that particle. This helps eventually segment particles in contact. This distance is capped at [Max Distance]. This value can be thought of as a minimum radius for particles, although Bellybutton may capture some particles below this radius. Good practice is to make this value a few pixels smaller than the minimum radius of the smallest desired regions. We find a value any higher than 10 pixels is unnecessary, as the values closest to the edges are most important, and values of 5 or higher will run without error.

#### Fraction, Epochs:
##### TL;DR: fraction of training pixels used, and number of times network is exposed to each - product of the two is typically the important parameter 
Often, not all of the user-generated data is necessary to train the network. Systems with thousands of particles in a single image may not even require a full image for training. Users should select their data to include as many lighting, rotation, focus, shape, and size scenarios as possible, but this may result in a massive number of pixels in the training set. The Fraction parameter indicates what fraction (0 to 1) of pixels identified as training data should be used to train the network. That data is then run through the network the number of times specified by Epochs. We find that in many systems, the important parameter is [Fraction] * [Epochs], and that provided Fraction is not too low (typically above 5-10\%, to give the network exposure to most situations), the trade-off between these two numbers is minimal. Users should train their networks until the test set error reaches a plateau, which will likely happen in fewer Epochs when more training data is used.


#### Output and Save_To:
##### TL;DR: Save_To is what file types are saved - images (png), data (npy). Outputs are what is saved (segmented regions, classification probability, etc)
All of these options are true (1) or false (0). Any combination may be selected, including all or none of them. Output types are segmented regions (default output), distance to outside a particle (used for distinguishing touching regions), probability that bellybutton thinks a pixel is inside vs outside a particle, a binarized version of that probability, and markers, which are the starting points for the watershedding algorithm.

#### Parameters that probably do not need adjusting:

##### Input Width (s_half):
Size of smallest window fed into network. S_half=12 is default, which means the window is $2*S_{half}+1 = 25$ pixels wide. S_half of 9 and above will run with no errors.

##### Batch Size:
Number of training examples fed to network at once. Default is 1024. Must be multiple of 8.

##### Track Outies (0 or 1):
If 1, Bellybutton inverts the masks given and tracks contiguous regions of 0's as if they are the particles.




