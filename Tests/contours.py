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
import cv2
import PIL.ImageOps
import numpy as np
import imutils
from skimage.filters import (threshold_otsu, threshold_niblack,
                             threshold_sauvola)
from skimage.data import page
import matplotlib
import matplotlib.pyplot as plt

kernel = np.ones((3,3),np.uint8)

pytesseract.pytesseract.tesseract_cmd = r'C:\Users\st123\AppData\Local\Tesseract-OCR\tesseract.exe'

image = cv2.imread('wtf.jpg')
#cv2.imshow('fucking', image)
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)


lower_gray = np.array([0, 0, 140])
upper_gray = np.array([0, 0, 255])
mask = cv2.inRange(hsv, lower_gray, upper_gray)


'''_, contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

cv2.drawContours(image, contours, -1, (255, 255, 255), -1)'''


mask = cv2.dilate(mask, kernel, iterations = 1)
mask = cv2.bitwise_not(mask)


cv2.imshow("Frame", image)
cv2.imshow("Mask", mask)



print(pytesseract.image_to_string(mask))

cv2.waitKey(0) 	