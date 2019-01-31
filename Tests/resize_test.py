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


SKIP_SEC = 300
TEMPLATE_LEFT_NAME_COORDINATES = (247, 616, 529, 653)
TEMPLATE_LEFT_NAME_DIMENSIONS = (298, 37)
TMEPLATE_LEFT_GAP_COORDINATES = (529, 616, 572, 653)
TEMPLATE_LEFT_GAP_DIMENSIONS = (43, 37)
TEMPLATE_COORDINATES = (572, 616, 707, 653)
TEMPLATE_DIMENSIONS = (135, 37)
TEMPLATE_RIGHT_NAME_COORDINATES = (757, 616, 1039, 653)
TEMPLATE_RIGHT_NAME_DIMENSIONS = (298, 37)
TEMPLATE_RIGHT_GAP_COORDINATES = (707, 616, 757, 653)
TEMPLATE_RIGHT_GAP_DIMENSIONS = (43, 37)
TEMPLATE_RECTANGLE_COORDINATES = (247, 616, 1047, 653)
TEMPLATE_RECTANGLE_DIMENSIONS = (800, 37)
GAP_DIMENSIONS = (43, 37)
NAME_DIMENSIONS = (387, 37)
COUNTRY_GAP = (90, 37)
NAME_THRESHOLD_VALUE = 175
COUNTRY_THRESHOLD_VALUE = 125

pytesseract.pytesseract.tesseract_cmd = r'C:\Users\st123\AppData\Local\Tesseract-OCR\tesseract.exe'

def cannify(image, value):
	canny = cv2.Canny(image, 50, 200)
	image1 = Image.fromarray(canny)
	invert = PIL.ImageOps.invert(image1)
	return canny

def threshify(image, value):
	ret, thresh1 = cv2.threshold(image, value, 255, cv2.THRESH_BINARY)
	image1 = Image.fromarray(thresh1)
	invert = PIL.ImageOps.invert(image1)
	return invert

def truncate(string):
	if len(string.split()) > 1:
		return string.split()[len(string.split())-1]
	return string



template = cv2.imread('templates/trapezoid_with_no_time.png')
template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
(tH, tW) = template.shape[:2]

image = cv2.imread('Stills/epee_yellow.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
found = None

#cv2.imshow('gray', gray)

#gray = pygame.surfarray.make_surface(gray)


#resized = pygame.transform.scale(gray, (3000, 3000))
#resized = pygame.surfarray.array2d(resized)



#resized = resizeimage.resize_cover(gray, [5000, 6000], validate = False)
#resized =  cv2.resize(gray, (500, 30))

#resized = imutils.resize(gray, height = int(gray.shape[0] * 1.5), width = int(gray.shape[1] * 1.5))
#cv2.imshow('test', resized)
#cv2.waitKey(0)


# loop over the scales of the image
for yscale in np.linspace(1.0, 1.3, 20):
	for xscale in np.linspace(1.0, 1.3, 20):
		# resize the image according to the scale, and keep track
		# of the ratio of the resizing
		resized = cv2.resize(gray, (int(gray.shape[1] * xscale), int(gray.shape[0] * yscale)))
		#resized = imutils.resize(gray, height = int(gray.shape[0] * xscale), width = int(gray.shape[1] * yscale))\
		xr = gray.shape[1] / float(resized.shape[1])
		yr = gray.shape[0] / float(resized.shape[0])
		
		# if the resized image is smaller than the template, then break
		# from the loop
		#if resized.shape[0] < tH or resized.shape[1] < tW:
			#break
		# detect edges in the resized, grayscale image and apply template
		# matching to find the template in the image
		edged = cv2.Canny(resized, 50, 200)
		result = cv2.matchTemplate(edged, template, cv2.TM_CCOEFF)
		(_, maxVal, _, maxLoc) = cv2.minMaxLoc(result)

		
		# check to see if the iteration should be visualized
		
		# draw a bounding box around the detected region
		'''
		clone = np.dstack([edged, edged, edged])
		cv2.rectangle(clone, (maxLoc[0], maxLoc[1]),
			(maxLoc[0] + tW, maxLoc[1] + tH), (0, 0, 255), 2)
		cv2.imshow("Visualize", clone)
		cv2.waitKey(0)
		'''

		# if we have found a new maximum correlation value, then update
		# the bookkeeping variable
		if found is None or maxVal > found[0]:
			found = (maxVal, maxLoc, xr, yr)
			#found_image = clone

# unpack the bookkeeping variable and compute the (x, y) coordinates
# of the bounding box based on the resized ratio
(maxVal, maxLoc, xr, yr) = found
(startX, startY) = (int(maxLoc[0] * xr), int(maxLoc[1] * yr))
(endX, endY) = (int((maxLoc[0] + tW) * xr), int((maxLoc[1] + tH) * yr))

#print(maxVal, startX, startY, endX, endY)


right_pts1 = np.float32([[endX - GAP_DIMENSIONS[0] * xr + TEMPLATE_RIGHT_GAP_DIMENSIONS[0] * xr, startY], [endX - GAP_DIMENSIONS[0] * xr + TEMPLATE_RIGHT_GAP_DIMENSIONS[0] * xr + NAME_DIMENSIONS[0] * xr, startY], [endX + TEMPLATE_RIGHT_GAP_DIMENSIONS[0] * xr, endY], [endX + TEMPLATE_RIGHT_GAP_DIMENSIONS[0] * xr + NAME_DIMENSIONS[0] * xr, endY]])
left_pts1 = np.float32([[startX + GAP_DIMENSIONS[0] * xr - TEMPLATE_LEFT_GAP_DIMENSIONS[0] * xr - NAME_DIMENSIONS[0] * xr, startY], [startX + GAP_DIMENSIONS[0] * xr - TEMPLATE_LEFT_GAP_DIMENSIONS[0] * xr, startY], [startX - TEMPLATE_LEFT_GAP_DIMENSIONS[0] * xr - NAME_DIMENSIONS[0] * xr, endY], [startX - TEMPLATE_LEFT_GAP_DIMENSIONS[0] * xr, endY]])


pts2 = np.float32([[0, 0], [NAME_DIMENSIONS[0] * xr, 0], [0, endY - startY], [NAME_DIMENSIONS[0] * xr, endY - startY]])

right_pts3 = np.float32([[0 - GAP_DIMENSIONS[0] * xr, 0], [NAME_DIMENSIONS[0] * xr - GAP_DIMENSIONS[0] * xr, 0], [0, endY - startY], [NAME_DIMENSIONS[0] * xr, endY - startY]])
left_pts3 = np.float32([[GAP_DIMENSIONS[0] * xr, 0], [NAME_DIMENSIONS[0] * xr + GAP_DIMENSIONS[0] * xr, 0], [0, endY - startY], [NAME_DIMENSIONS[0] * xr, endY - startY]])



right_M = cv2.getPerspectiveTransform(right_pts1, pts2)
right_dst = cv2.warpPerspective(gray, right_M, (int(NAME_DIMENSIONS[0] * xr), endY - startY))
right_M = cv2.getPerspectiveTransform(pts2, right_pts3)
right_dst = cv2.warpPerspective(right_dst, right_M, (int(NAME_DIMENSIONS[0] * xr), endY - startY))


left_M = cv2.getPerspectiveTransform(left_pts1, pts2)
left_dst = cv2.warpPerspective(gray, left_M, (int(NAME_DIMENSIONS[0] * xr), endY - startY))
left_M = cv2.getPerspectiveTransform(pts2, left_pts3)
left_dst = cv2.warpPerspective(left_dst, left_M, (int(NAME_DIMENSIONS[0] * xr), endY - startY))

'''cv2.imwrite('left.jpg', left_dst)
cv2.imwrite('right.jpg', right_dst)'''


'''threshify(left_dst).show()
threshify(right_dst).show()'''


height, width = left_dst.shape[:2]
left_country = left_dst[0:height, 0:int(COUNTRY_GAP[0] * xr)]
right_country = right_dst[0:height, (width -int(COUNTRY_GAP[0] * xr)):width]

cv2.imwrite('omegagay1.jpg', left_country)
cv2.imwrite('omegagay2.jpg', right_country)


'''thresh_niblack = threshold_niblack(left_country, 25, k = .8)
thresh_sauvola = threshold_sauvola(left_country, 25)

binary_niblack = left_country > thresh_niblack
binary_sauvola = left_country > thresh_sauvola


plt.imshow(binary_niblack, cmap=plt.cm.gray)
plt.show()'''

'''
cv2.imshow('gay1', np.array(cannify(left_country, COUNTRY_THRESHOLD_VALUE)))
cv2.imshow('gay2', np.array(cannify(right_country, COUNTRY_THRESHOLD_VALUE)))
cv2.waitKey(0)



print(truncate(pytesseract.image_to_string(np.array(threshify(left_country, COUNTRY_THRESHOLD_VALUE)))))
print(truncate(pytesseract.image_to_string(np.array(threshify(right_country, COUNTRY_THRESHOLD_VALUE)))))
'''

'''
# draw a bounding box around the detected result and display the image
#Left Name
cv2.rectangle(image, (startX - int(TEMPLATE_LEFT_GAP_DIMENSIONS[0] * xr) - int(TEMPLATE_LEFT_NAME_DIMENSIONS[0] * xr), endY), (startX - int(TEMPLATE_LEFT_GAP_DIMENSIONS[0] * xr), startY), (0, 0, 255), 2)
#Right Name
cv2.rectangle(image, (endX + int(TEMPLATE_RIGHT_GAP_DIMENSIONS[0] * xr), endY), (endX + int(TEMPLATE_RIGHT_NAME_DIMENSIONS[0] * xr) + int(TEMPLATE_RIGHT_GAP_DIMENSIONS[0] * xr), startY), (0, 0, 255), 2)
#cv2.rectangle(image, (startX - int(TEMPLATE_LEFT_GAP_DIMENSIONS[0] * r) - int(TEMPLATE_LEFT_NAME_DIMENSIONS[0] * r), endY), (endX + int(TEMPLATE_RIGHT_DIMENSION[0]*r) + int(TEMPLATE_RIGHT_GAP_DIMENSION[0] * r), startY), (0, 0, 255), 2)

cv2.rectangle(image, (startX, endY), (endX, startY), (0, 0, 255), 2)
cv2.imshow("Image", image)
#cv2.imwrite("still.jpg", image)
#cv2.imwrite("still_test.jpg", image)
#cv2.imshow("copy", found_image)
cv2.waitKey(0)
'''