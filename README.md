# DL Tutorial
This gives you the basic guidiance for what to do in the project.

## Step 0 - Setup
Install Anaconda for Windows. Create a new environment with `conda create -n tensorflow pip python=3.6 `.

Install Tensorflow with
```
pip install --ignore-installed --upgrade tensorflow
```
See [Tensorflow](https://www.tensorflow.org/install/) for details.
You also may want to install tensorflow-gpu, if you have a GPU. 
In that case, make sure, that everything is running as it should.

Install all the requirements listet in `requirements.txt`. If you later on find some dependencies to be missing, feel free to install them.

## Step 1 - Download the data
Visit [Microsoft](https://www.microsoft.com/en-us/download/details.aspx?id=54765) and download the dataset. Unpack it to somwhere. You will have to move files from this directory in the next step.

## Step 2 - Training Data vs Test Data
The dataset does not come with separated training and test data. The separation of these two types of data is crucial, so do that now. Create a directory structure like this:
```
.
+-- data
    +-- train
        +-- Cat
        +-- Dog
    +-- test
        +-- Cat
        +-- Dog
```      
Move into the folders under data/train the images with indices 0-9999 and move the images with indices from 10000-12500 to the test images.
This corresponds to 80% train data and 20% test data.

## Step 3 - Read the data
In order to learn from the data, you first have to pass this data to tensorflow. 
To do so, create a function that you can use as input_fn for the estimator. 
It should return cats and dogs similarily often, so don't have first only cats and then only dogs. 
The images should be randomly selected from all available training images. 
Also make sure, that this method is reusable: You will have to be able to use it for both the training and the test set.

## Step 4 - Create the neural net
We provided you with a rough framework, you manly have to fill in the blanks. 
Take a look at reference implementations of Estimators for the MNIST dataset. 
You can learn a lot from there.
Also consider using a highlevel wrapper, such as Keras for the creation of the net.
But if you just want to start playing around, the pure tensorflow code will be enough.

## Step 5 - Learn and evaluate the net
Now actually learn the net. But beware: There are so many hyperparameters, you will have your fun playing with these. You also may want to change the network architecture, as ours is not actually intended to win this competition :)

# Problems you will encounter
## Designing the input pipeline
This step is crucial. If you mess up here, things will be slower and, depending on your setup, by magnitudes slower.
See [here](https://www.tensorflow.org/performance/performance_guide#input_pipeline_optimization) for some quick tips.

## Image input
If you use the Tnesorflow Dataset API, you will have to take special care of your input images. 
Some images may have a different format, than others, so you might want to do some preprocessing here. 
Maybe even external, to keep your code nice and clean.
There will also be very bad examples of images in the pipeline. 
Do you really want to include those? 
And more importantly: How do you find out, that an image is a bad example?
For example, look for images with the index 666. 
Here you may want to process images with another python script first and the either write them to the disk and use the input pipeline or use the images in RAM as input. 
This saves you from slow disk access.

# Things to try
Just a short list of things on my mind. Some relate to architecture others to preprocessing.
- Spatial Pyramid Pooling
- Global Max Pooling
- More Layers
- More Dense Layers
- More Filters
- Larger Filters
- Different Padding
- Image Augmentation
    - Rotation
    - Scaling
    - Noise
    - Cropping
    - Other distortions
- Grayscale images
- Only external preprocessing
- Cache images in RAM
