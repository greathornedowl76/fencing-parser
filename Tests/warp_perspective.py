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

img = cv2.imread('Stills/up_0-0.jpg')
rows, cols, ch = img.shape

pts1 = np.float32([[679, 616], [1090, 616], [712, 653], [1123, 653]])
pts2 = np.float32([[0, 0], [378, 0], [0, 37], [378, 37]])

pts3 = np.float32([[-33, 0], [378, 0], [0, 37], [411, 37]])


M = cv2.getPerspectiveTransform(pts1,pts2)
dst = cv2.warpPerspective(img, M, (400, 37))


M = cv2.getPerspectiveTransform(pts2, pts3)
dst = cv2.warpPerspective(dst, M, (400, 37))


pytesseract.pytesseract.tesseract_cmd = r'C:\Users\st123\AppData\Local\Tesseract-OCR\tesseract.exe'

def threshhold(image):
	ret, thresh1 = cv2.threshold(image, 175, 255, cv2.THRESH_BINARY)
	image1 = Image.fromarray(thresh1)
	invert = PIL.ImageOps.invert(image1)
	return invert

dst = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)
cv2.imshow('whyu', np.array(threshhold(dst)))
cv2.waitKey(0)

print(pytesseract.image_to_string(threshhold(dst)))

 