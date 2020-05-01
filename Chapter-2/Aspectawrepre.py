import imutils
import cv2

class AspectAwarePreprocessor:
	def __init__(self, width, height, inter = cv2.INTER_AREA):

		self.width = width
		self.height = height
		self.inter = inter

	def preprocess(self, image):

		(h, w) = image.shape[:2]
		dW = 0
		dH = 0
		#if width is smaller then height, resize along the width (i.e, smaller dimension)
		#and then update deltas to crop the height to desired dimeension

		if w < h:
			image = imutils.resize(image, width = self.width, inter = self.inter)
			dH = int((image.shape[0] - self.height)/2.0)
		#else if height is smaller then
		else:
			image = imutils.resize(image, height = self.height, inter = self.inter)
			dW = int((image.shape[1] - self.width)/2.0)

		#now that image is resized, we need to grab the width and height, and perform the crop
		(h, w) = image.shape[:2]
		image = image[dH:h - dH, dW:w - dW]

		#finally we resize the image to provided spatial dimensions to ensure output image is always fixed size
		return cv2.resize(image, (self.width, self.height), interpolation = self.inter)



