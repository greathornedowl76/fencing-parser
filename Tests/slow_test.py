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

SKIP_SEC = 300
TEMPLATE_LEFT_NAME_COORDINATES = (247, 616, 529, 653)
TEMPLATE_LEFT_NAME_DIMENSIONS = (298, 37)
TMEPLATE_LEFT_GAP_COORDINATES = (529, 616, 572, 653)
TEMPLATE_LEFT_GAP_DIMENSIONS = (43, 37)
TEMPLATE_COORDINATES = (572, 616, 707, 653)
TEMPLATE_DIMENSIONS = (35, 37)
TEMPLATE_RIGHT_NAME_COORDINATES = (757, 616, 1039, 653)
TEMPLATE_RIGHT_NAME_DIMENSION = (298, 37)
TEMPLATE_RIGHT_GAP_COORDINATES = (707, 616, 757, 653)
TEMPLATE_RIGHT_GAP_DIMENSION = (43, 37)
TEMPLATE_RECTANGLE_COORDINATES = (247, 616, 1047, 653)
TEMPLATE_RECTANGLE_DIMENSIONS = (800, 37)
anomolies = ['|', '©', '_', '~', '/', '<', '>', ':', ';', '?', '!', '-']

pytesseract.pytesseract.tesseract_cmd = r'C:\Users\st123\AppData\Local\Tesseract-OCR\tesseract.exe'

#the pattern used as the base of the fencing doccument requires a different threshold
def threshhold(image):
	ret, thresh1 = cv2.threshold(image, 175, 255, cv2.THRESH_BINARY)
	image1 = Image.fromarray(thresh1)
	invert = PIL.ImageOps.invert(image1)
	return invert

def check_anomolies(string):
	for character in anomolies:
		if character in string:
			string = string.replace(character, '')
	return string	
		
def array_to_image(numpy_array):
	return Image.fromarray(numpy_array)

def image_to_array(image):
	return np.array(image)

def frame_splitter(frame, leftstartX, leftstartY, leftendX, leftendY, rightstartX, rightstartY, rightendX, rightendY): 
	left_fencer = image_to_array(frame.crop((leftstartX, leftstartY, leftendX, leftendY))) #original: (247, 613, 532, 653)
	right_fencer = image_to_array(frame.crop((rightstartX, rightstartY, rightendX, rightendY))) #original: ((745, 613, 1046, 653))
	return [left_fencer, right_fencer]


clip = VideoFileClip('Videos/jp_v_kor.mp4', audio=False)
left_names = {}
right_names = {}
frame_count = 0
next_sec = 0
last_left = ''
last_right = ''
template = cv2.imread('templates/trapezoid_with_no_time.png')
template_found = False

for sec, numpy_array in clip.iter_frames(with_times=True, progress_bar=True):
	if sec < next_sec:
		continue 

	if template_found is False:
		template = cv2.imread('templates/trapezoid_with_no_time.png')
		template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
		(tH, tW) = template.shape[:2]

		image = cv2.cvtColor(numpy_array, cv2.COLOR_BGR2RGB)
		gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		found = None
		# loop over the scales of the image
		for yscale in np.linspace(1.0, 1.3, 20):
			for xscale in np.linspace(1.0, 1.3, 20):
				# resize the image according to the scale, and keep track of the ratio of the resizing
				resized = cv2.resize(gray, (int(gray.shape[1] * xscale), int(gray.shape[0] * yscale)))
				xr = gray.shape[1] / float(resized.shape[1])
				yr = gray.shape[0] / float(resized.shape[0])
				
				# detect edges in the resized, grayscale image and apply templatematching to find the template in the image
				edged = cv2.Canny(resized, 50, 200)
				result = cv2.matchTemplate(edged, template, cv2.TM_CCOEFF)
				(_, maxVal, _, maxLoc) = cv2.minMaxLoc(result)

				
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

		# draw a bounding box around the detected result and display the image
		#Left Name
		cv2.rectangle(image, (startX - int(TEMPLATE_LEFT_GAP_DIMENSIONS[0] * xr) - int(TEMPLATE_LEFT_NAME_DIMENSIONS[0] * xr), endY), (startX - int(TEMPLATE_LEFT_GAP_DIMENSIONS[0] * xr), startY), (0, 0, 255), 1)
		#Right Name
		cv2.rectangle(image, (endX + int(TEMPLATE_RIGHT_GAP_DIMENSION[0] * xr), endY), (endX + int(TEMPLATE_RIGHT_NAME_DIMENSION[0] * xr) + int(TEMPLATE_RIGHT_GAP_DIMENSION[0] * xr), startY), (0, 0, 255), 1)
		#cv2.rectangle(image, (startX - int(TEMPLATE_LEFT_GAP_DIMENSIONS[0] * r) - int(TEMPLATE_LEFT_NAME_DIMENSIONS[0] * r), endY), (endX + int(TEMPLATE_RIGHT_DIMENSION[0]*r) + int(TEMPLATE_RIGHT_GAP_DIMENSION[0] * r), startY), (0, 0, 255), 2)

		cv2.rectangle(image, (startX, endY), (endX, startY), (0, 0, 255), 2)
		cv2.imshow("Image", image)
		cv2.imwrite("video_test.jpg", image)
		cv2.waitKey(0)
		if maxVal > 11000000:
			template_found = True


	#frame = threshhold_left(numpy_array)
	frame_count += 1
	#frame_splitter(frame)[0].show()
	numpy_array = cv2.cvtColor(numpy_array, cv2.COLOR_BGR2GRAY)
	#print(endY, startY)
	cropped_image = frame_splitter(array_to_image(numpy_array), startX - int(TEMPLATE_LEFT_GAP_DIMENSIONS[0] * xr) - int(TEMPLATE_LEFT_NAME_DIMENSIONS[0] * xr), startY, startX - int(TEMPLATE_LEFT_GAP_DIMENSIONS[0] * xr), endY, endX + int(TEMPLATE_RIGHT_GAP_DIMENSION[0] * xr), startY, endX + int(TEMPLATE_RIGHT_NAME_DIMENSION[0] * xr) + int(TEMPLATE_RIGHT_GAP_DIMENSION[0] * xr), endY)
	#threshhold(cropped_image[0]).show(0)

	left_fencer = check_anomolies(pytesseract.image_to_string(threshhold(cropped_image[0])))
	right_fencer = check_anomolies(pytesseract.image_to_string(threshhold(cropped_image[1])))                                
	if left_fencer in left_names:
		left_names[left_fencer] = left_names[left_fencer] + 1
	else:
		left_names[left_fencer] = 0
	if right_fencer in right_names:
		right_names[right_fencer] = right_names[right_fencer] + 1
	else:
		right_names[right_fencer] = 0
	if frame_count >= 5:
		accurate_left = (max(left_names, key=left_names.get))
		accurate_right = (max(right_names, key=right_names.get))
		left_names = {}
		right_names = {}
		frame_count = 0
		if accurate_left == last_left or accurate_right == last_right or accurate_right == "" or accurate_left == "":	
			next_sec = sec + SKIP_SEC
		else:
			last_left, last_right = accurate_left, accurate_right
			print(accurate_left + " vs " + accurate_right)
			#threshhold(cropped_image[0]).show()
			#threshhold(cropped_image[1]).show()
		
			