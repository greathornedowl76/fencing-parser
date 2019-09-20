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
import subprocess
import ast

SKIP_SEC = 300
TEMPLATE_LEFT_NAME_COORDINATES = (247, 616, 529, 653)
TEMPLATE_LEFT_NAME_DIMENSIONS = (298, 37)
TMEPLATE_LEFT_GAP_COORDINATES = (529, 616, 572, 653)
TEMPLATE_LEFT_GAP_DIMENSIONS = (38, 37)
TEMPLATE_COORDINATES = (572, 616, 707, 653)
TEMPLATE_DIMENSIONS = (135, 37)
TEMPLATE_RIGHT_NAME_COORDINATES = (757, 616, 1039, 653)
TEMPLATE_RIGHT_NAME_DIMENSIONS = (298, 37)
TEMPLATE_RIGHT_GAP_COORDINATES = (707, 616, 757, 653)
TEMPLATE_RIGHT_GAP_DIMENSIONS = (38, 37)
TEMPLATE_RECTANGLE_COORDINATES = (247, 616, 1047, 653)
TEMPLATE_RECTANGLE_DIMENSIONS = (800, 37)
GAP_DIMENSIONS = (39, 37)
DIAGONAL_DIMENSIONS = (45, 37) #45
SCORE_DIMENSIONS = (25, 37) #29 20
NAME_DIMENSIONS = (355, 37) #345 354 364 360
COUNTRY_GAP = (90, 37)
anomolies = ['|', '©', '_', '~', '/', '<', '>', ':', ';', '?', '!', '-', '`', '=','+', '`' , "'", '"', '*']

#Debugging
visualize_all = False
visualize_box = True
test = False

pytesseract.pytesseract.tesseract_cmd = r'C:\Users\st123\AppData\Local\Tesseract-OCR\tesseract.exe'



#the pattern used as the base of the fencing doccument requires a different threshold
def threshify(image):
	_, thresh1 = cv2.threshold(image, 165, 255, cv2.THRESH_BINARY)
	thresh2 = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 17, -31)
	invert = cv2.bitwise_not(thresh1)
	return invert

def contourify(image):
	h, w = image.shape[:2]
	canvas = np.zeros((h, w, 3), np.uint8)
	cavnas = cv2.bitwise_not(canvas)
	kernel = np.ones((2,2),np.uint8)
	hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

	lower_gray = np.array([0, 0, 140])
	upper_gray = np.array([0, 0, 255])
	mask = cv2.inRange(hsv, lower_gray, upper_gray)

	_, contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

	for contour in contours:
		area = cv2.contourArea(contour)
		if area > 10:
			cv2.drawContours(canvas, contour, -1, (255, 255, 255), -1)
	'''
	_, contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
	cv2.drawContours(image, contours, -1, (255, 255, 255), -1)
	'''
	#mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

	#mask = cv2.dilate(mask, kernel, iterations = 1)
	#mask = cv2.erode(mask, kernel, iterations = 1)
	#invert = cv2.bitwise_not(mask)

	canvas = cv2.dilate(canvas, kernel, iterations = 1)
	invert = cv2.bitwise_not(canvas)
	'''
	cv2.imshow('ugahlirkj', invert)
	cv2.waitKey(0)
	print(pytesseract.image_to_string(invert))
	'''
	return invert

def check_anomolies(string):
	if not string:
		return "N/A"
	for character in anomolies:
		if character in string:
			string = string.replace(character, '')
	return string	

def truncate_left(string):
	return string.replace(string.split()[0], '')

def truncate_right(string):
	return string.replace(string.split()[::-1][0], '')

def fix_abbreviations(string):
	if string is not None:
		words = string.split()
		if words[0] in anomolies:
			string.replace(words[0], '')
		reverse = string[::-1]
		has_abbreviation = False
		for index in range(len(reverse)):
			if reverse[index] is '.':
				has_abbreviation = True
			if has_abbreviation:
				if reverse[index].isupper() is True:	
					if reverse[index+1] is not ' ':
						return (reverse[:index+1] + ' ' + reverse[index+1:])[::-1]
					return string
		return string
	return string

def return_left_name(string):
	words = string.split()
	if len(words) > 1:
		return words[len(words)-2] + words[len(words)-1]
	return string

def return_right_name(string):
	words = string.split()
	if len(words) > 1:
		return words[0] + words[1]
	return string

def array_to_image(numpy_array):
	return Image.fromarray(numpy_array)

def image_to_array(image):
	return np.array(image)

def locate(image, startX, startY, endX, endY, xr): 
	right_pts1 = np.float32([\
	[endX + SCORE_DIMENSIONS[0] * xr, startY], \
	[endX + SCORE_DIMENSIONS[0] * xr + NAME_DIMENSIONS[0] * xr, startY], \
	[endX + SCORE_DIMENSIONS[0] * xr + DIAGONAL_DIMENSIONS[0] * xr, endY], \
	[endX + SCORE_DIMENSIONS[0] * xr + DIAGONAL_DIMENSIONS[0] * xr + NAME_DIMENSIONS[0] * xr, endY]])

	'''
	right_pts1 = np.float32([\
	[endX - GAP_DIMENSIONS[0] * xr + GAP_DIMENSIONS[0] * xr + TEMPLATE_RIGHT_GAP_DIMENSIONS[0] * xr, startY], \
	[endX - GAP_DIMENSIONS[0] * xr + GAP_DIMENSIONS[0] * xr + TEMPLATE_RIGHT_GAP_DIMENSIONS[0] * xr + NAME_DIMENSIONS[0] * xr, startY], \
	[endX + GAP_DIMENSIONS[0] * xr + TEMPLATE_RIGHT_GAP_DIMENSIONS[0] * xr, endY], \
	[endX + GAP_DIMENSIONS[0] * xr + TEMPLATE_RIGHT_GAP_DIMENSIONS[0] * xr + NAME_DIMENSIONS[0] * xr, endY]])
	'''

	left_pts1 = np.float32([\
	[startX - SCORE_DIMENSIONS[0] * xr - NAME_DIMENSIONS[0] * xr, startY], \
	[startX - SCORE_DIMENSIONS[0] * xr, startY], \
	[startX - SCORE_DIMENSIONS[0] * xr - DIAGONAL_DIMENSIONS[0] * xr - NAME_DIMENSIONS[0] * xr, endY], \
	[startX - SCORE_DIMENSIONS[0] * xr - DIAGONAL_DIMENSIONS[0] * xr, endY]])
	'''
	left_pts1 = np.float32([\
	[startX + GAP_DIMENSIONS[0] * xr - GAP_DIMENSIONS[0] * xr - TEMPLATE_LEFT_GAP_DIMENSIONS[0] * xr - NAME_DIMENSIONS[0] * xr, startY], \
	[startX + GAP_DIMENSIONS[0] * xr - GAP_DIMENSIONS[0] * xr - TEMPLATE_LEFT_GAP_DIMENSIONS[0] * xr, startY], \
	[startX - GAP_DIMENSIONS[0] * xr - TEMPLATE_LEFT_GAP_DIMENSIONS[0] * xr - NAME_DIMENSIONS[0] * xr, endY], \
	[startX - GAP_DIMENSIONS[0] * xr - TEMPLATE_LEFT_GAP_DIMENSIONS[0] * xr, endY]])
	'''
	pts2 = np.float32([[0, 0], [NAME_DIMENSIONS[0] * xr, 0], [0, endY - startY], [NAME_DIMENSIONS[0] * xr, endY - startY]])

	right_pts3 = np.float32([\
	[0                          , 0            ], [NAME_DIMENSIONS[0] * xr                              , 0            ], \
	[DIAGONAL_DIMENSIONS[0] * xr, endY - startY], [NAME_DIMENSIONS[0] * xr + DIAGONAL_DIMENSIONS[0] * xr, endY - startY]])

	left_pts3 = np.float32([\
	[DIAGONAL_DIMENSIONS[0] * xr, 0            ], [NAME_DIMENSIONS[0] * xr + DIAGONAL_DIMENSIONS[0] * xr, 0            ], \
	[0                          , endY - startY], [NAME_DIMENSIONS[0] * xr                              , endY - startY]])


	right_M = cv2.getPerspectiveTransform(right_pts1, pts2)
	right_dst = cv2.warpPerspective(image, right_M, (int(NAME_DIMENSIONS[0] * xr) , endY - startY))
	right_M = cv2.getPerspectiveTransform(pts2, right_pts3)
	right_dst = cv2.warpPerspective(right_dst, right_M, (int(NAME_DIMENSIONS[0] * xr) + (int(DIAGONAL_DIMENSIONS[0] * xr)), endY - startY))

	left_M = cv2.getPerspectiveTransform(left_pts1, pts2)
	left_dst = cv2.warpPerspective(image, left_M, (int(NAME_DIMENSIONS[0] * xr) , endY - startY))
	left_M = cv2.getPerspectiveTransform(pts2, left_pts3)
	left_dst = cv2.warpPerspective(left_dst, left_M, (int(NAME_DIMENSIONS[0] * xr) + (int(DIAGONAL_DIMENSIONS[0] * xr)), endY - startY))

	height, width = left_dst.shape[:2]
	left_name = left_dst[0:height, int(COUNTRY_GAP[0] * xr):width]
	right_name = right_dst[0:height, 0:(width - int(COUNTRY_GAP[0] * xr))]
	left_country = left_dst[0:height, 0:int(COUNTRY_GAP[0] * xr)]
	right_country = right_dst[0:height, (width - int(COUNTRY_GAP[0] * xr)):width]
	return [left_name, right_name, left_country, right_country]


#clip = VideoFileClip('Videos/test_large.mp4', audio=False)
left_names = {}
right_names = {}
frame_count = 0
next_sec = 0


last_left = ''
last_right = ''
template = cv2.imread('templates/trapezoid_with_no_time.png')
template_found = False
path = 'C:\\Users\\st123\\Desktop\\Folders\\projects\\Fencing_ungo_bungo\\Videos'
frame_path = 'C:\\Users\\st123\\Desktop\\Folders\\projects\\Fencing_ungo_bungo\\frames'

for filename in os.listdir(path):
	next_sec = 0
	if (filename.endswith(".mp4")):
		noExtFilename = os.path.splitext(filename)[0]
		folderName = frame_path + "\\" + noExtFilename
		if not os.path.isdir(folderName):
			os.mkdir(folderName)
		#temp_sec = temp_sec + SKIP_SEC
		print(frame_path + "\\" + folderName)

		sts = subprocess.Popen("ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 " + path + "\\" + filename, stdout = subprocess.PIPE, stderr = subprocess.STDOUT)
		raw = sts.stdout.readlines()[0]
		time = float(ast.literal_eval(raw.decode("utf-8")))

		while next_sec < time:
			os.system("ffmpeg -ss {2} -i {0} -y -loglevel quiet -vframes 1 -q:v 2 {1}\\thumb.jpg ".format(path + "\\" + filename, frame_path + "\\" + noExtFilename, next_sec))
			numpy_array = cv2.imread(frame_path + "\\" + noExtFilename + "\\thumb.jpg")
			
			if template_found is False:
				template = cv2.imread('templates/trapezoid_with_no_time.png')
				template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
				(tH, tW) = template.shape[:2]


				image = numpy_array
				#image = cv2.cvtColor(numpy_array, cv2.COLOR_BGR2RGB)
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
						if visualize_all is True:
							clone = np.dstack([edged, edged, edged])
							cv2.rectangle(clone, (maxLoc[0], maxLoc[1]),
								(maxLoc[0] + tW, maxLoc[1] + tH), (0, 0, 255), 2)
							cv2.imshow("Visualize", clone)
							cv2.waitKey(0)

						# if we have found a new maximum correlation value, then update
						# the bookkeeping variable
						if found is None or maxVal > found[0]:
							found = (maxVal, maxLoc, xr, yr)

				# unpack the bookkeeping variable and compute the (x, y) coordinates
				# of the bounding box based on the resized ratio
				(maxVal, maxLoc, xr, yr) = found
				(startX, startY) = (int(maxLoc[0] * xr), int(maxLoc[1] * yr))
				(endX, endY) = (int((maxLoc[0] + tW) * xr), int((maxLoc[1] + tH) * yr))

				# draw a bounding box around the detected result and display the image
				if visualize_box is True:
					#Left Name
					cv2.rectangle(image, (startX - int(TEMPLATE_LEFT_GAP_DIMENSIONS[0] * xr) - int(TEMPLATE_LEFT_NAME_DIMENSIONS[0] * xr), endY), (startX - int(TEMPLATE_LEFT_GAP_DIMENSIONS[0] * xr), startY), (0, 0, 255), 1)
					#Right Name
					cv2.rectangle(image, (endX + int(TEMPLATE_RIGHT_GAP_DIMENSIONS[0] * xr), endY), (endX + int(TEMPLATE_RIGHT_NAME_DIMENSIONS[0] * xr) + int(TEMPLATE_RIGHT_GAP_DIMENSIONS[0] * xr), startY), (0, 0, 255), 1)
					#cv2.rectangle(image, (startX - int(TEMPLATE_LEFT_GAP_DIMENSIONS[0] * r) - int(TEMPLATE_LEFT_NAME_DIMENSIONS[0] * r), endY), (endX + int(TEMPLATE_RIGHT_DIMENSION[0]*r) + int(TEMPLATE_RIGHT_GAP_DIMENSIONS[0] * r), startY), (0, 0, 255), 2)
					cv2.rectangle(image, (startX, endY), (endX, startY), (0, 0, 255), 2)
					cv2.imshow("Image", image)
					cv2.imwrite("video_test.jpg", image)
					cv2.waitKey(0)

				if maxVal > 11000000:
					template_found = True

			frame_count += 1

			numpy_array = cv2.cvtColor(numpy_array, cv2.COLOR_RGB2BGR)
			gray = cv2.cvtColor(numpy_array, cv2.COLOR_BGR2GRAY)
			gray2 = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
			(gray_left, gray_right, gray_left_country, gray_right_country) = locate(gray, startX, startY, endX, endY, xr)
			(RGB_left, RGB_right, RGB_left_country, RGB_right_country) = locate(gray2, startX, startY, endX, endY, xr)

			RGB_left_country = cv2.resize(RGB_left_country, (0,0), fx = 1.0, fy = 1.0)
			RGB_right_country = cv2.resize(RGB_right_country, (0,0), fx = 1.0, fy = 1.0)

			left_fencer = check_anomolies(pytesseract.image_to_string(threshify(gray_left)))
			right_fencer = check_anomolies(pytesseract.image_to_string(threshify(gray_right)))
			'''
			cv2.imwrite('aiya1.jpg', gray_left)
			cv2.imwrite('aiya2.jpg', gray_right)
			cv2.imshow('aiya4', contourify(RGB_left_country))
			cv2.imwrite('aiya5.jpg', contourify(RGB_right_country))
			print(pytesseract.image_to_string('aiya5.jpg'))
			cv2.waitKey(0)
			'''
			left_country = check_anomolies(pytesseract.image_to_string(contourify(RGB_left_country)))
			right_country = check_anomolies(pytesseract.image_to_string(contourify(RGB_right_country)))
			
			'''

			'''

			cv2.waitKey(0)
			'''
			image = cv2.imread('sad.jpg')
			image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
			image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
			print(pytesseract.image_to_string(contourify(image)))'''
			#print(left_country)

			'''
			cv2.imshow('1', contourify(cropped_imageA[0]))
			cv2.imshow('2', contourify(cropped_imageA[1]))
			cv2.imshow('3', contourify(cropped_imageB[2]))
			cv2.imshow('4', contourify(cropped_imageB[3]))
			cv2.waitKey(0)
			'''

			if left_fencer in left_names:
				left_names[left_fencer] = left_names[left_fencer] + 1
			else:
				left_names[left_fencer] = 0
			if right_fencer in right_names:
				right_names[right_fencer] = right_names[right_fencer] + 1
			else:
				right_names[right_fencer] = 0
			if frame_count >= 5:
				accurate_left = fix_abbreviations(max(left_names, key=left_names.get))
				accurate_right = fix_abbreviations(max(right_names, key=right_names.get))
				left_names = {}
				right_names = {}
				frame_count = 0
				if accurate_left == last_left or accurate_right == last_right or accurate_right == "" or accurate_left == "":	
					next_sec = next_sec + SKIP_SEC
				else:
					last_left, last_right = accurate_left, accurate_right
					print(accurate_left + " from " + left_country + " vs " + accurate_right + " from " + right_country)
					#cv2.imshow('aiya1', threshify(gray_left))
					#cv2.imshow('aiya2', threshify(gray_right))
					#cv2.imshow('aiya4', contourify(RGB_left_country))
					#cv2.imshow('aiya5', contourify(RGB_right_country))
					cv2.waitKey(0)
					if test is True:
						exit()
			next_sec = next_sec + 1