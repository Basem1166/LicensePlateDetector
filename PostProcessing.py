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
    """
    Extract license plate area with rotation handling
    Args:
        plate: Input image containing license plate
    Returns:
        Extracted and aligned plate image
    """
    img = plate.copy()
    
    # Convert to grayscale if not already
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()
    
    # Apply adaptive thresholding
    binary = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 11, 2
    )
    
    # Noise removal
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    
    # Edge detection
    edges = cv2.Canny(binary, 50, 150, apertureSize=3)
    
    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter and sort contours by area
    possible_plates = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 1000:  # Filter small contours
            continue
            
        # Get rotated rectangle
        rect = cv2.minAreaRect(cnt)
        (x, y), (width, height), angle = rect
        
        # Filter by aspect ratio
        aspect_ratio = max(width, height) / min(width, height)
        if 2 <= aspect_ratio <= 5:  # License plate aspect ratio range
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            possible_plates.append((box, area, angle))
    
    if not possible_plates:
        return plate  # Return original if no plate found
    
    # Sort by area and get largest
    possible_plates.sort(key=lambda x: x[1], reverse=True)
    box, _, angle = possible_plates[0]
    
    # Get rotation matrix
    # Ensure angle is between -45 and 45 degrees
    if angle < -45:
        angle += 90
    
    # Get the center, size and angle of the plate
    center = ((box[0][0] + box[2][0]) // 2, (box[0][1] + box[2][1]) // 2)
    width = int(np.linalg.norm(box[1] - box[0]))
    height = int(np.linalg.norm(box[2] - box[1]))
    
    # Perform rotation
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))
    
    # Crop the plate
    x = center[0] - width // 2
    y = center[1] - height // 2
    plate = rotated[max(y, 0):min(y + height, rotated.shape[0]),
                    max(x, 0):min(x + width, rotated.shape[1])]
    
    return plate

# Match contours to license plate or character template
def find_contours(dimensions, img) :

    # Find all contours in the image
    cntrs, _ = cv2.findContours(img.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Retrieve potential dimensions
    lower_width = dimensions[0]
    upper_width = dimensions[1]
    lower_height = dimensions[2]
    upper_height = dimensions[3]
    
    # Check largest 5 or  15 contours for license plate or character respectively
    cntrs = sorted(cntrs, key=cv2.contourArea, reverse=True)[:15]
    
    ii = cv2.imread('contour.jpg')
    
    x_cntr_list = []
    img_res = []
    for cntr in cntrs :
        # detects contour in binary image and returns the coordinates of rectangle enclosing it
        intX, intY, intWidth, intHeight = cv2.boundingRect(cntr)
        
        # checking the dimensions of the contour to filter out the characters by contour's size
        if intWidth > lower_width and intWidth < upper_width and intHeight > lower_height and intHeight < upper_height :
            x_cntr_list.append(intX) #stores the x coordinate of the character's contour, to used later for indexing the contours

            char_copy = np.zeros((44,24))
            # extracting each character using the enclosing rectangle's coordinates.
            char = img[intY:intY+intHeight, intX:intX+intWidth]
            char = cv2.resize(char, (20, 40))
            
            cv2.rectangle(ii, (intX,intY), (intWidth+intX, intY+intHeight), (50,21,200), 2)
            plt.imshow(ii, cmap='gray')

            # Make result formatted for classification: invert colors
            char = cv2.subtract(255, char)

            # Resize the image to 24x44 with black border
            char_copy[2:42, 2:22] = char
            char_copy[0:2, :] = 0
            char_copy[:, 0:2] = 0
            char_copy[42:44, :] = 0
            char_copy[:, 22:24] = 0

            img_res.append(char_copy) # List that stores the character's binary image (unsorted)
            
    # Return characters on ascending order with respect to the x-coordinate (most-left character first)
            
    # plt.show()
    # arbitrary function that stores sorted list of character indeces
    indices = sorted(range(len(x_cntr_list)), key=lambda k: x_cntr_list[k])
    img_res_copy = []
    for idx in indices:
        img_res_copy.append(img_res[idx]//255)# stores character images according to their index
    img_res = np.array(img_res_copy)

    return img_res

def segment_characters(image) :

    # Preprocess cropped license plate image
    img_lp = cv2.resize(image, (333, 75))
    img_gray_lp = cv2.cvtColor(img_lp, cv2.COLOR_BGR2GRAY)
    _, img_binary_lp = cv2.threshold(img_gray_lp, 200, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    img_binary_lp = cv2.erode(img_binary_lp, (3,3))
    img_binary_lp = cv2.dilate(img_binary_lp, (3,3))

    LP_WIDTH = img_binary_lp.shape[0]
    LP_HEIGHT = img_binary_lp.shape[1]

    # Make borders white
    img_binary_lp[0:3,:] = 255
    img_binary_lp[:,0:3] = 255
    img_binary_lp[72:75,:] = 255
    img_binary_lp[:,330:333] = 255

    # Estimations of character contours sizes of cropped license plates
    dimensions = [LP_WIDTH/6,
                       LP_WIDTH/2,
                       LP_HEIGHT/10,
                       2*LP_HEIGHT/3]
    plt.imshow(img_binary_lp, cmap='gray')
    plt.show()
    cv2.imwrite('contour.jpg',img_binary_lp)

    # Get contours within cropped license plate
    char_list = find_contours(dimensions, img_binary_lp)

    return char_list

def getCharacters(plate):
    # Convert to grayscale if needed
    if len(plate.shape) == 3:
        gray = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)
    else:
        gray = plate.copy()
    
    # Adaptive thresholding
    binary = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 11, 2
    )
    
    # Noise removal and character enhancement
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    characters = []
    valid_contours = []
    
    # Get average character dimensions
    heights = []
    widths = []
    
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if h > 10:  # Minimum height threshold
            heights.append(h)
            widths.append(w)
    
    if not heights or not widths:
        return []
    
    avg_height = np.mean(heights)
    avg_width = np.mean(widths)
    
    # Filter contours by size and position
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        
        # Size filtering
        if (0.5 * avg_height <= h <= 1.5 * avg_height and 
            0.2 * avg_width <= w <= 2 * avg_width):
            valid_contours.append((x, y, w, h, cnt))
    
    # Sort contours left to right
    valid_contours.sort(key=lambda x: x[0])
    
    # Extract character images
    for x, y, w, h, cnt in valid_contours:
        # Add padding
        pad = 2
        x_start = max(0, x - pad)
        x_end = min(binary.shape[1], x + w + pad)
        y_start = max(0, y - pad)
        y_end = min(binary.shape[0], y + h + pad)
        
        char_img = binary[y_start:y_end, x_start:x_end]
        
        # Ensure minimum size
        if char_img.shape[0] > 8 and char_img.shape[1] > 4:
            characters.append(char_img)
    
    print("Number of characters found:", len(characters))
    return characters

class CharacterRecognizer:
    def __init__(self):
        # Define template patterns for characters and digits
        # Using a simplified 5x3 grid representation
        self.templates = {
            # Digits
            '0': [
                [1, 1, 1],
                [1, 0, 1],
                [1, 0, 1],
                [1, 0, 1],
                [1, 1, 1]
            ],
            '1': [
                [0, 1, 0],
                [1, 1, 0],
                [0, 1, 0],
                [0, 1, 0],
                [1, 1, 1]
            ],
            '2': [
                [1, 1, 1],
                [0, 0, 1],
                [1, 1, 1],
                [1, 0, 0],
                [1, 1, 1]
            ],
            '3': [
                [1, 1, 1],
                [0, 0, 1],
                [0, 1, 1],
                [0, 0, 1],
                [1, 1, 1]
            ],
            '4': [
                [1, 0, 1],
                [1, 0, 1],
                [1, 1, 1],
                [0, 0, 1],
                [0, 0, 1]
            ],
            '5': [
                [1, 1, 1],
                [1, 0, 0],
                [1, 1, 1],
                [0, 0, 1],
                [1, 1, 1]
            ],
            '6': [
                [1, 1, 1],
                [1, 0, 0],
                [1, 1, 1],
                [1, 0, 1],
                [1, 1, 1]
            ],
            '7': [
                [1, 1, 1],
                [0, 0, 1],
                [0, 1, 0],
                [0, 1, 0],
                [0, 1, 0]
            ],
            '8': [
                [1, 1, 1],
                [1, 0, 1],
                [1, 1, 1],
                [1, 0, 1],
                [1, 1, 1]
            ],
            '9': [
                [1, 1, 1],
                [1, 0, 1],
                [1, 1, 1],
                [0, 0, 1],
                [1, 1, 1]
            ],
            # Letters
            'A': [
                [1, 1, 1],
                [1, 0, 1],
                [1, 1, 1],
                [1, 0, 1],
                [1, 0, 1]
            ],
            'B': [
                [1, 1, 0],
                [1, 0, 1],
                [1, 1, 0],
                [1, 0, 1],
                [1, 1, 0]
            ],
            'C': [
                [1, 1, 1],
                [1, 0, 0],
                [1, 0, 0],
                [1, 0, 0],
                [1, 1, 1]
            ],
            'D': [
                [1, 1, 0],
                [1, 0, 1],
                [1, 0, 1],
                [1, 0, 1],
                [1, 1, 0]
            ],
            'E': [
                [1, 1, 1],
                [1, 0, 0],
                [1, 1, 0],
                [1, 0, 0],
                [1, 1, 1]
            ],
            'F': [
                [1, 1, 1],
                [1, 0, 0],
                [1, 1, 0],
                [1, 0, 0],
                [1, 0, 0]
            ],
            'G': [
                [1, 1, 1],
                [1, 0, 0],
                [1, 0, 1],
                [1, 0, 1],
                [1, 1, 1]
            ],
            'H': [
                [1, 0, 1],
                [1, 0, 1],
                [1, 1, 1],
                [1, 0, 1],
                [1, 0, 1]
            ],
            'I': [
                [1, 1, 1],
                [0, 1, 0],
                [0, 1, 0],
                [0, 1, 0],
                [1, 1, 1]
            ],
            'J': [
                [1, 1, 1],
                [0, 0, 1],
                [0, 0, 1],
                [1, 0, 1],
                [1, 1, 1]
            ],
            'K': [
                [1, 0, 1],
                [1, 0, 1],
                [1, 1, 0],
                [1, 0, 1],
                [1, 0, 1]
            ],
            'L': [
                [1, 0, 0],
                [1, 0, 0],
                [1, 0, 0],
                [1, 0, 0],
                [1, 1, 1]
            ],
            'M': [
                [1, 0, 1],
                [1, 1, 1],
                [1, 0, 1],
                [1, 0, 1],
                [1, 0, 1]
            ],
            'N': [
                [1, 0, 1],
                [1, 1, 1],
                [1, 1, 1],
                [1, 0, 1],
                [1, 0, 1]
            ],
            'O': [
                [1, 1, 1],
                [1, 0, 1],
                [1, 0, 1],
                [1, 0, 1],
                [1, 1, 1]
            ],
            'P': [
                [1, 1, 1],
                [1, 0, 1],
                [1, 1, 1],
                [1, 0, 0],
                [1, 0, 0]
            ],
            'Q': [
                [1, 1, 1],
                [1, 0, 1],
                [1, 0, 1],
                [1, 1, 1],
                [0, 0, 1]
            ],
            'R': [
                [1, 1, 1],
                [1, 0, 1],
                [1, 1, 0],
                [1, 0, 1],
                [1, 0, 1]
            ],
            'S': [
                [1, 1, 1],
                [1, 0, 0],
                [1, 1, 1],
                [0, 0, 1],
                [1, 1, 1]
            ],
            'T': [
                [1, 1, 1],
                [0, 1, 0],
                [0, 1, 0],
                [0, 1, 0],
                [0, 1, 0]
            ],
            'U': [
                [1, 0, 1],
                [1, 0, 1],
                [1, 0, 1],
                [1, 0, 1],
                [1, 1, 1]
            ],
            'V': [
                [1, 0, 1],
                [1, 0, 1],
                [1, 0, 1],
                [1, 0, 1],
                [0, 1, 0]
            ],
            'W': [
                [1, 0, 1],
                [1, 0, 1],
                [1, 0, 1],
                [1, 1, 1],
                [1, 0, 1]
            ],
            'X': [
                [1, 0, 1],
                [1, 0, 1],
                [0, 1, 0],
                [1, 0, 1],
                [1, 0, 1]
            ],
            'Y': [
                [1, 0, 1],
                [1, 0, 1],
                [0, 1, 0],
                [0, 1, 0],
                [0, 1, 0]
            ],
            'Z': [
                [1, 1, 1],
                [0, 0, 1],
                [0, 1, 0],
                [1, 0, 0],
                [1, 1, 1]
            ]
        }

    def preprocess_image(self, character_image, target_size=(5, 3)):
        """
        Preprocess the character image to match template size
        Args:
            character_image: Binary numpy array of the character
            target_size: Desired size for comparison
        Returns:
            Preprocessed binary image
        """
        # Convert to binary (0 or 1)
        threshold = 128
        binary = [[1 if pixel > threshold else 0 for pixel in row] 
                 for row in character_image]
        
        # Resize to match template size (simplified)
        # In practice, you'd want to use proper interpolation
        h_ratio = len(binary) / target_size[0]
        w_ratio = len(binary[0]) / target_size[1]
        
        resized = [[0 for _ in range(target_size[1])] 
                  for _ in range(target_size[0])]
        
        for i in range(target_size[0]):
            for j in range(target_size[1]):
                orig_i = int(i * h_ratio)
                orig_j = int(j * w_ratio)
                resized[i][j] = binary[orig_i][orig_j]
                
        return resized

    def calculate_similarity(self, image1, image2):
        """
        Calculate similarity between two binary patterns
        Args:
            image1, image2: Binary patterns to compare
        Returns:
            Similarity score between 0 and 1
        """
        if len(image1) != len(image2) or len(image1[0]) != len(image2[0]):
            raise ValueError("Patterns must have the same dimensions")
        
        matches = sum(1 for i in range(len(image1))
                     for j in range(len(image1[0]))
                     if image1[i][j] == image2[i][j])
        
        total_pixels = len(image1) * len(image1[0])
        return matches / total_pixels

    def recognize_character(self, char_image, threshold=0.8):
        """
        Recognize a character by comparing with templates
        Args:
            char_image: Input character image
            threshold: Minimum similarity threshold
        Returns:
            Recognized character and confidence score
        """
        preprocessed = self.preprocess_image(char_image)
        
        best_match = None
        best_score = 0
        
        for char, template in self.templates.items():
            score = self.calculate_similarity(preprocessed, template)
            
            if score > best_score:
                best_score = score
                best_match = char
        
        if best_score < threshold:
            return None, best_score
        
        return best_match, best_score

    def recognize_plate(self, plate_characters):
        """
        Recognize full license plate
        Args:
            plate_characters: List of character images from plate
        Returns:
            Recognized plate string and confidence scores
        """
        result = []
        scores = []
        
        for char_img in plate_characters:
            char, score = self.recognize_character(char_img)
            if char is None:
                continue
            result.append(char)
            scores.append(score)
        
        return ''.join(result), sum(scores) / len(scores) if scores else 0

def potentialCharacter(characters):
    
    plate_text = ""

    recognizer = CharacterRecognizer()

    for char in characters:
        char, score = recognizer.recognize_character(char)
        if char:
            plate_text += char
        else:
            plate_text += "?"
    
    return plate_text
            
