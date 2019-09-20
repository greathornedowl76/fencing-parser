# fencing-parser
Parses through fencing videos and outputs the fencers.
![Final Output](https://raw.githubusercontent.com/ravenseattuna/fencing-parser/master/Stills/output.png)

# Overview
Currently, the most common overlay used for livestreams of fencing tournaments is Fencing Vision. Due to the semi-consistency of this overlay, this program parses out relevant fencer information such as name and nationality by using OpenCV's template match to locate the overlay's timer.

![Bounding Boxes With Clock as Anchor](https://raw.githubusercontent.com/ravenseattuna/fencing-parser/master/Stills/video_test.jpg)

The extracted images are then read by Google's Tesseract OCR Engine, cleaned up, and outputted in the cmd prompt.

# Limitations
Only works for matches using recent versions of Fencing Vision. Older versions may result in extra noise, resulting in less accurate results

# Dependencies
* FFMPEG
* pytesseract
* OpenCV

# To-Do List
* Add youtube-dl support
* Compile hyperlinks to videos with timestaps and fencer info into a database
