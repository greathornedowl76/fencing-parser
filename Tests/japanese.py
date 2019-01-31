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


pytesseract.pytesseract.tesseract_cmd = r'C:\Users\st123\AppData\Local\Tesseract-OCR\tesseract.exe'

print(pytesseract.image_to_string('haehyun.png', lang = "jpn"))