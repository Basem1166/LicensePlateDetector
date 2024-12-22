from typing import Counter
import cv2
import numpy as np
from preprocessing import *


def getGreenAreas(img):
    # Convert to HSV color space
    hsv_image = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2HSV)

    # Define the range for green color in HSV
    lower_green = np.array([0, 255, 40])  # Adjust these values as needed
    upper_green = np.array([80, 255, 255])

    # Create a mask for green color
    green_mask = cv2.inRange(hsv_image, lower_green, upper_green)

    # Find contours in the mask
    contours, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    return contours

def extractPlate(plate):

    img = plate.copy()


    # binary_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # _, binary_image = cv2.threshold(binary_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    canny = cv2.Canny(np.uint8(img), 130, 255, 1)

    # show_images([canny], ["Binary Image"])

    cnts = cv2.findContours(canny, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]

    for c in cnts:
        cv2.drawContours(img,[c], 0, (0,255,0), 1)

    show_images([img], ["Detected License Plate"])

    return img

def getFillingColor(area):

    filling_colors = []
    # Loop through horizontal lines in the image
    for row in range(area.shape[0]):
        inside_region = False  # Flag to check if we're inside the green-bordered region
        current_color = None
        last_pixel = None
        for col in range(area.shape[1]):
            pixel = tuple(area[row, col])  # Get the BGR color of the current pixel

            if last_pixel == pixel:
                continue

            # Check if the pixel is green
            if pixel == (0, 255, 0):
                # Toggle the inside_region flag when entering or exiting a region
                if inside_region:
                    inside_region = False  # Exiting the region
                else:
                    inside_region = True  # Entering the region
                    current_color = []  # Reset the current color list for the new region
            elif inside_region:
                # Add pixel to the current region's binary color list
                current_color.append(pixel)

            last_pixel = pixel
        
        # Add the region's colors to the filling_colors list
        if current_color:
            filling_colors.extend(current_color)

    # Find the most frequent color among all regions
    if filling_colors:
        most_common_color = Counter(filling_colors).most_common(1)
        if most_common_color:
            most_common_color = most_common_color[0][0]
    else:
        most_common_color = None

    return most_common_color



def getCharacters(plate):
    
    contour = getGreenAreas(plate)

    # binary_plate = cv2.cvtColor(plate.copy(), cv2.COLOR_RGB2GRAY)
    # _, binary_plate = cv2.threshold(binary_plate, 120, 255, cv2.THRESH_BINARY)
    # binary_plate = cv2.morphologyEx(255-binary_plate, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=1)

    # show_images([binary_plate], ["Binary Plate"])

    characters = []
    filling_colors = []
    for c in contour:
        x, y, w, h = cv2.boundingRect(c)
        if w*h > 100:
            area = plate[y:y+h, x:x+w]
            filling_color = getFillingColor(area)
            filling_colors.append(filling_color)

    # Find the most frequent filling color among all returned contours to distinguish between characters and noise
    if filling_colors:
        most_common_color = Counter(filling_colors).most_common(1)[0][0]
    else:
        most_common_color = None

    # Loop through the contours to extract the characters
    for c in contour:
        x, y, w, h = cv2.boundingRect(c)

        if w == plate.shape[1] and h == plate.shape[0]:
            continue

        if w*h > 100:
            area = plate[y:y+h, x:x+w]
            filling_color = getFillingColor(area)

            # Getting based on the actual color of the region nor the binary color
            diff = [0,0,0]
            for i in range(3):
                if filling_color[i] > most_common_color[i]:
                    diff[i] = (filling_color[i] - most_common_color[i])**2
                else:
                    diff[i]= (most_common_color[i] - filling_color[i])**2

            distance = math.sqrt(np.sum(diff))
            if distance <= 20:
                characters.append(area)
                #show_images([area], ["Character"])
        

    return characters
