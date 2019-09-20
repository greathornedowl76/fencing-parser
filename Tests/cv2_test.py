import time
import os
import subprocess
import ast

path = 'C:\\Users\\st123\\Desktop\\Folders\\projects\\Fencing_ungo_bungo\\Videos'
frame_path = 'C:\\Users\\st123\\Desktop\\Folders\\projects\\Fencing_ungo_bungo\\frames'
SKIP_SEC = 60

for filename in os.listdir(path):
	temp_sec = 0
	if (filename.endswith(".mp4")):
		noExtFilename = os.path.splitext(filename)[0]
		folderName = frame_path + "\\" + noExtFilename
		if not os.path.isdir(folderName):
			os.mkdir(folderName)
		temp_sec = temp_sec + SKIP_SEC
		#print(frame_path + "\\" + folderName)

		#cmd = 
		sts = subprocess.Popen("ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 " + path + "\\" + filename, stdout = subprocess.PIPE, stderr = subprocess.STDOUT)
		raw = sts.stdout.readlines()[0]
		time = float(ast.literal_eval(raw.decode("utf-8")))
 
		#info = [x for x in result.stdout.readlines() if "Duration" in x.decode()]
		
		#print(result.stdout.readlines())
		count = 0
		while temp_sec < time:
			count += 1
			os.system("ffmpeg -ss {2} -i {0} -y -loglevel quiet -frames:v 1 -q:v 2 {1}\\thumb{3}.jpg ".format(path + "\\" + filename, frame_path + "\\" + noExtFilename, temp_sec, count))
			temp_sec = temp_sec + SKIP_SEC
		#os.system("ffmpeg -ss {2} -i {0} -vframes 1 -q:v 2 {1}\\thumb.jpg ".format(path + "\\" + filename, frame_path + "\\" + noExtFilename, temp_sec))
		
	else:
		continue

