#A demo of Data Augmentation

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
import numpy as np
import argparse


#construct the argument parse and parse the argument
ap = argparse.ArgumentParser()
ap.add_argument('-i', '--image', required = True, help= 'Path to input image')
ap.add_argument('-o', '--output', required = True, help = 'Path to output directory')
ap.add_argument('-p', '--prefix', type = str, default = 'image', help = 'output filename prefix')
args = vars(ap.parse_args())


#load input image, conver to Numpy array and reshape to have extra dimension
print('[INFO] loading example image ...')
image = load_img(args['image'])
image = img_to_array(image)
image = np.expand_dims(image, axis = 0)


#construct the image generator for data augmentation then inirialize total num of images generated thus far
aug = ImageDataGenerator(rotation_range = 30, width_shift_range = 0.1, height_shift_range = 0.1,
	shear_range = 0.2, zoom_range = 0.2, horizontal_flip = True, fill_mode = 'nearest')
total = 0


#construct the Python generator
imageGen = aug.flow(image, batch_size = 1, save_to_dir = args['output'],
	save_prefix = args['prefix'], save_format = 'jpg')


#loop over examples 
for image in imageGen:
	#increment counter
	total +=1

	#if reach 10 example break from loop
	if total == 10:
		break


