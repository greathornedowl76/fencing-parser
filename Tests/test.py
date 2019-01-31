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

#original
'''
TEMPLATE_LEFT_NAME_COORDINATES = (247, 616, 529, 653)
TEMPLATE_LEFT_NAME_DIMENSIONS = (282, 37)
TMEPLATE_LEFT_GAP_COORDINATES = (529, 616, 572, 653)
TEMPLATE_LEFT_GAP_DIMENSIONS = (43, 37)
TEMPLATE_COORDINATES = (572, 616, 707, 653)
TEMPLATE_DIMENSIONS = (35, 37)
TEMPLATE_RIGHT_COORDINATES = (757, 616, 1039, 653)
TEMPLATE_RIGHT_DIMENSION = (295, 37)
TEMPLATE_RIGHT_GAP_COORDINATES = (707, 616, 757, 653)
TEMPLATE_RIGHT_GAP_DIMENSION = (50, 37)
TEMPLATE_RECTANGLE_COORDINATES = (247, 616, 1047, 653)
TEMPLATE_RECTANGLE_DIMENSIONS = (800, 37)
'''

TEMPLATE_LEFT_NAME_COORDINATES = (247, 616, 529, 653)
TEMPLATE_LEFT_NAME_DIMENSIONS = (305, 37)
TMEPLATE_LEFT_GAP_COORDINATES = (529, 616, 572, 653)
TEMPLATE_LEFT_GAP_DIMENSIONS = (43, 37)
TEMPLATE_COORDINATES = (572, 616, 707, 653)
TEMPLATE_DIMENSIONS = (35, 37)
TEMPLATE_RIGHT_NAME_COORDINATES = (757, 616, 1039, 653)
TEMPLATE_RIGHT_NAME_DIMENSION = (305, 37)
TEMPLATE_RIGHT_GAP_COORDINATES = (707, 616, 757, 653)
TEMPLATE_RIGHT_GAP_DIMENSION = (50, 37)
TEMPLATE_RECTANGLE_COORDINATES = (247, 616, 1047, 653)
TEMPLATE_RECTANGLE_DIMENSIONS = (800, 37)

pytesseract.pytesseract.tesseract_cmd = r'C:\Users\st123\AppData\Local\Tesseract-OCR\tesseract.exe'

#the pattern used as the base of the fencing doccument requires a different threshold
def threshhold(image):
	#print(image.dtype, image.shape)
	#image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	#image = cv2.imread(image, 0)
	ret, thresh1 = cv2.threshold(image, 175, 255, cv2.THRESH_BINARY)
	#thresh2 = cv2.adaptiveThreshold(array, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
	image1 = Image.fromarray(thresh1)
	#image1.show()
	#imageadapt = Image.fromarray(thresh2)
	invert = PIL.ImageOps.invert(image1)
	#invert2 = PIL.ImageOps.invert(imageadapt)
	#print(pytesseract.image_to_string(invert))
	return invert

'''
def checker(left_fencer, right_fencer):
	if left_fencer == right_fencer:
		return True
	return False
'''

def array_to_image(numpy_array):
	return Image.fromarray(numpy_array)

def image_to_array(image):
	return np.array(image)

def frame_splitter(frame, leftstartX, leftstartY, leftendX, leftendY, rightstartX, rightstartY, rightendX, rightendY): 
	left_fencer = image_to_array(frame.crop((leftstartX, leftstartY, leftendX, leftendY))) #original: (247, 613, 532, 653)
	right_fencer = image_to_array(frame.crop((rightstartX, rightstartY, rightendX, rightendY))) #original: ((745, 613, 1046, 653))
	#print(right_fencer.dtype)
	#print(left_fencer.dtype)
	return [left_fencer, right_fencer]
'''
def splitter(clip):
	left_fencer = clip.crop(x1=227, y1=610, x2=533, y2=660)
	right_fencer = clip.crop(x1=745, y1=617, x2=1046, y2=653)
	return left_fencer
'''


clip = VideoFileClip('Videos/up.mp4', audio=False)
#splitter(clip).preview(fps=60)
left_names = {}
right_names = {}
frame_count = 0
next_sec = 0
last_left = ''
last_right = ''
template = cv2.imread('trapezoid_with_time.png')
template_found = False


for sec, numpy_array in clip.iter_frames(with_times=True, progress_bar=True):
	if sec < next_sec:
		continue 

	if template_found is False:
		template = cv2.imread('trapezoid_with_time.png')
		template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
		(tH, tW) = template.shape[:2]

		image = numpy_array
		gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		found = None
		# loop over the scales of the image
		for scale in np.linspace(1.0, 1.5, 20):
		# resize the image according to the scale, and keep track
		# of the ratio of the resizing
			resized = imutils.resize(gray, width = int(gray.shape[1] * scale))
			r = gray.shape[1] / float(resized.shape[1])

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
				found = (maxVal, maxLoc, r)
				#found_image = clone

		# unpack the bookkeeping variable and compute the (x, y) coordinates
		# of the bounding box based on the resized ratio
		(maxVal, maxLoc, r) = found
		(startX, startY) = (int(maxLoc[0] * r), int(maxLoc[1] * r))
		(endX, endY) = (int((maxLoc[0] + tW) * r), int((maxLoc[1] + tH) * r))

		#print(maxVal, startX, startY, endX, endY)

		# draw a bounding box around the detected result and display the image

		cv2.rectangle(image, (startX, startY), (endX, endY), (0, 0, 255), 2)
		#cv2.imshow("Image", image)
		#cv2.imshow("copy", found_image)
		#cv2.waitKey(0)
		if maxVal > 10000000:
			template_found = True


	#frame = threshhold_left(numpy_array)
	frame_count += 1
	#frame_splitter(frame)[0].show()
	numpy_array = cv2.cvtColor(numpy_array, cv2.COLOR_BGR2GRAY)
	#print(endY, startY)
	cropped_image = frame_splitter(array_to_image(numpy_array), startX - int(TEMPLATE_LEFT_GAP_DIMENSIONS[0] * r) - int(TEMPLATE_LEFT_NAME_DIMENSIONS[0] * r), startY, startX - int(TEMPLATE_LEFT_GAP_DIMENSIONS[0] * r), endY, endX + int(TEMPLATE_RIGHT_GAP_DIMENSION[0] * r), startY, endX + int(TEMPLATE_RIGHT_NAME_DIMENSION[0]*r) + int(TEMPLATE_RIGHT_GAP_DIMENSION[0] * r), endY)
	#threshhold(cropped_image[0]).show(0)

	left_fencer = pytesseract.image_to_string(threshhold(cropped_image[0]))
	right_fencer = pytesseract.image_to_string(threshhold(cropped_image[1]))                                  
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
		
		
		




#works for one side
'''
clip = VideoFileClip('jp_v_kor.mp4', audio=False)
#splitter(clip).preview(fps=60)
names = {}
right_names = {}
frame_count = 0

for fencer in splitter(clip).iter_frames(progress_bar=True):
	#for frame in fencer.iter_frames(progress_bar=True):
	#print(pytesseract.image_to_string(frame))
	frame_count += 1
	name = pytesseract.image_to_string(fencer)
	if name in names:
		names[name] = names[name] + 1
	else:
		names[name] = 0
	if frame_count > 30:
		print(max(names, key = names.get))
		frame_count = 0
'''


#works for both sides, but not how i want it to
'''
for frame in split_clip[index].iter_frames(progress_bar=True):
	for index in range(2):
	#print(pytesseract.image_to_string(frame))
		frame_count += 1
		name = pytesseract.image_to_string(frame)
		if index == 0:
			if name in left_names:
				left_names[name] = left_names[name] + 1
			else:
				left_names[name] = 0
			if frame_count > 30:
				print(max(left_names, key=left_names.get))
		if index == 1:
			if name in right_names:
				right_names[name] = right_names[name] + 1
			else:
				right_names[name] = 0
			if frame_count > 30:
				print(max(right_names, key=right_names.get))
				frame_count = 0
'''