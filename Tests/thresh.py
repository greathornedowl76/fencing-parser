try:
    from PIL import Image
except ImportError:
    import Image
import pytesseract
import PIL
import subprocess as sp
from moviepy.editor import *
from moviepy.editor import VideoFileClip
import threading
import numpy as np
import cv2
import PIL.ImageOps
import imutils
import matplotlib
import matplotlib.pyplot as plt

from skimage.data import page
from skimage.filters import (threshold_otsu, threshold_niblack,
                             threshold_sauvola)


pytesseract.pytesseract.tesseract_cmd = r'C:\Users\st123\AppData\Local\Tesseract-OCR\tesseract.exe'

image = cv2.imread('image.png')

#Sharpening
blur = cv2.GaussianBlur(image, (51,51), 51)
unsharp_image = cv2.addWeighted(image, 1.5, blur, -.5, 0, image)
cv2.imwrite('sharpened1.jpg', unsharp_image)



#Threshold
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
invert = cv2.bitwise_not(gray)

for value in range(0, 255):
	blur = cv2.GaussianBlur(invert,(5,5),0)
	ret, thresh1 = cv2.threshold(blur, value, 255, cv2.THRESH_BINARY)
	thresh2 = cv2.adaptiveThreshold(blur, 255, \
		cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 31, 1)
	print(pytesseract.image_to_string(thresh1) + str(value))



#Contours
kernel = np.ones((3,3),np.uint8)
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

lower_gray = np.array([0, 0, 140])
upper_gray = np.array([0, 0, 255])
mask = cv2.inRange(hsv, lower_gray, upper_gray)


copy = image
'''
_, contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
cv2.drawContours(copy, contours, -1, (0, 0, 0), -1)
'''

#mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
#mask = cv2.dilate(mask, kernel, iterations = 1)
#mask = cv2.erode(mask, kernel, iterations = 1)

mask = cv2.bitwise_not(mask)

'''
cv2.imshow('ugahlirkj', invert)
cv2.waitKey(0)
print(pytesseract.image_to_string(invert))
'''



'''
cv2.imshow('reg', thresh1)
cv2.imshow('adapt', thresh2)
cv2.imshow('contour_mask', mask)
#cv2.imshow('contour_image', copy)
cv2.waitKey(0)
'''