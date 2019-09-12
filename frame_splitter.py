import cv2 
import os
from moviepy.editor import *
import time

SKIP_SEC = 1800

#@source: https://www.geeksforgeeks.org/python-program-extract-frames-using-opencv/
def FrameCapture(path):
    next_sec = 0
    count = 0
    clip = VideoFileClip('Videos/test_large.mp4', audio=False)

    for sec, numpy_array in clip.iter_frames(with_times=True, progress_bar=True):
        if sec < next_sec:
            continue 
        
        path = r'C:\Users\st123\Desktop\Folders\projects\Fencing_ungo_bungo\frames'
        image = cv2.cvtColor(numpy_array, cv2.COLOR_BGR2RGB)
        cv2.imwrite(os.path.join(path , "frame%d.jpg" % count), image) 
        count += 1

        next_sec = sec + SKIP_SEC
  
if __name__ == '__main__': 
    path = r'C:\Users\st123\Desktop\Folders\projects\Fencing_ungo_bungo\Videos\test_large.mp4'
    now = time.time()
    FrameCapture(path) 
    print(time.time() - now)