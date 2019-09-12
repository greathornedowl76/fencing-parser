import pytesseract

pytesseract.pytesseract.tesseract_cmd = r'C:\Users\st123\AppData\Local\Tesseract-OCR\tesseract.exe'


print(pytesseract.image_to_string("Screen Shot 2018-09-02 at 6.48.10 PM.png"))