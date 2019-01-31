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

'''
template = cv2.imread('template2.jpg')
template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
template = cv2.Canny(template, 50, 200)
(tH, tW) = template.shape[:2]
cv2.imwrite("template_edged.png", template)
'''

template = cv2.imread('template_cropped.png')
template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
(tH, tW) = template.shape[:2]

image = cv2.imread('jp_v_kor_Moment.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
edged = cv2.Canny(gray, 50, 200)
result = cv2.matchTemplate(edged, template, cv2.TM_CCOEFF)
cv2.imwrite("Edged.png", edged)

#cv2.imshow("Image", image)
#cv2.waitKey(0)

print(cv2.minMaxLoc(result))

'''
image1 = Image.fromarray(template)
image1.show()
'''