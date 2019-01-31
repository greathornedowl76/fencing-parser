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
import numpy

pytesseract.pytesseract.tesseract_cmd = r'C:\Users\st123\AppData\Local\Tesseract-OCR\tesseract.exe'

print(pytesseract.image_to_string("joemarti2_right.png"))

array = cv2.imread("joemarti2_right.png", 0)
#Image.fromarray(array).show()

ret, thresh1 = cv2.threshold(array, 145, 255, cv2.THRESH_BINARY)
thresh2 = cv2.adaptiveThreshold(array, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 41, 2)
thresh3 = cv2.adaptiveThreshold(array, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 13, 2)



image1 = Image.fromarray(thresh1)
imageadapt = Image.fromarray(thresh2)
imagegaus = Image.fromarray(thresh3)



invert = PIL.ImageOps.invert(image1)
invert2 = PIL.ImageOps.invert(imageadapt)
invert3 = PIL.ImageOps.invert(imagegaus)
print(pytesseract.image_to_string(invert))
invert.show()