from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from Aspectawrepre import AspectAwarePreprocessor
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.optimizers import SGD
from minivggnet import MiniVGGNet
from imagetoarraypreprocessor import ImageToArrayPreprocessor
from simpledatasetloader import SimpleDatasetLoader
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import cv2
import os


#construct the argument parse and parse the argument
ap = argparse.ArgumentParser()
ap.add_argument('-d', '--dataset', required = True, help = 'Path to input dataset')
args = vars(ap.parse_args())


#grab the list of images we will be describing and extract class labels
print('[INFO] loading images ...')
imagePaths = list(paths.list_images(args['dataset']))
classNames = [pt.split(os.path.sep)[-2] for pt in imagePaths]
classNames = [str(x) for x in np.unique(classNames)]


#initialize image preprocessors
aap = AspectAwarePreprocessor(64, 64)
iap = ImageToArrayPreprocessor()


#load dataset from disk and scale pixel intensities to range [0, 1]
sdl = SimpleDatasetLoader(preprocessors = [aap, iap])
(data, labels) = sdl.load(imagePaths, verbose = 500)
data = data.astype('float')/255.0


#splitting the dataset into training and testing set
(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size = 0.25, random_state = 42)


#convert labels from integers to vectors
trainY = LabelBinarizer().fit_transform(trainY)
testY = LabelBinarizer().fit_transform(testY)


#initializing optimizer and model
opt = SGD(lr = 0.05)
model = MiniVGGNet.build(width = 64, height = 64, depth = 3, classes = len(classNames))
model.compile(loss = 'categorical_crossentropy', optimizer = opt, metrics = ['accuracy'])


#train network
H = model.fit(trainX, trainY, validation_data = (testX, testY), batch_size = 32, epochs = 100, verbose = 1)


#evaluate the network
print('[INFO] evaluating the network ...')
predictions = model.predict(testX, batch_size=32)
print(classification_report(testY.argmax(axis = 1), predictions.argmax(axis = 1), target_names = classNames))


#plot training loss and accuracy
plt.style.use('ggplot')
plt.figure()
plt.plot(np.arange(0, 100), H.history['loss'], label = 'train_loss')
plt.plot(np.arange(0, 100), H.history['val_loss'], label = 'val_loss')
plt.plot(np.arange(0, 100), H.history['acc'], label = 'train_acc')
plt.plot(np.arange(0, 100), H.history['val_acc'], label = 'val_acc')
plt.title('Training Loss and Accuracy')
plt.xlabel('Epoch #')
plt.ylabel('Loss/Accuracy')
plt.legend()
plt.show()


