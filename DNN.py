from paddleocr import PaddleOCR, draw_ocr
from hezar.models import Model
import cv2
import numpy
import os

ocr = PaddleOCR(use_angle_cls=True, lang='en') # need to run only once to download and load model into memory

def get_text_from_image(img):
    # convert to numpy array
    img = numpy.array(img)

    # convert to BGR
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    # run ocr
    result = ocr.ocr(img)
    
    return result


model = Model.load("hezarai/crnn-fa-printed-96-long")

def get_text_from_image_arabic(img):


    os.makedirs("assets", exist_ok=True)
    cv2.imwrite("assets/license_plate_ocr_example.jpg", img)
    plate_text = model.predict("assets/license_plate_ocr_example.jpg")
    return plate_text