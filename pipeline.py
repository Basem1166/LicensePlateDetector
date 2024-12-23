from preprocessing import *
from processing import *
import cv2
import numpy as np
from PostProcessing import *
from DNN import *


def LicenesePlateDetector(img):
    gray_img = rgb2gray(img)
    #filtered_image = bilateral_filter(gray_img, 17, 17)
    # 8u and 32f images
    gray_img_img = gray_img.astype(np.float32)
    filtered_image = cv2.bilateralFilter(gray_img_img,17, 17, 17)
    #filtered_image = median(gray_img)
    show_images([gray_img, filtered_image], ["Original", "Filtered"])
    
    equalized_image = CLAHE(filtered_image)
    show_images([gray_img, equalized_image], ["Original", "Equalized"])

    vertical_edges = np.abs(sobel_v(equalized_image))
    show_images([equalized_image, vertical_edges], ["Equilized", "Vertical Edges"])

    # Binarize the vertical edges image
    binary_image = image_binarization(vertical_edges)
    show_images([binary_image], ["Binary Image"])

    filtered_roi_regions = initial_roi_region(binary_image, gray_img)

    # show before filtering ROI regions
    roi_img = np.zeros_like(gray_img)
    for region in filtered_roi_regions:
        roi_img[region[0]:region[1], :] = gray_img[region[0]:region[1], :]
    show_images([gray_img, roi_img], ["Original", "Initial ROI Regions"])
    inital_roi_image = roi_img

    # mask the image with the ROI regions
    # Filter and merge ROI regions

    final_roi_regions = findFinalRoiRegion(filtered_roi_regions, gray_img)
    # Mask the image with the final ROI regions
    filtered_roi_img = np.zeros_like(gray_img)
    for region in final_roi_regions:
        filtered_roi_img[region[0]:region[1], :] = gray_img[region[0]:region[1], :]
    show_images([gray_img, roi_img], ["Original", "Final ROI Regions"])

    

    power_edges = update_edge_power(vertical_edges, final_roi_regions)

    # Select the ROI candidate with the maximum edge power
    selected_roi = select_roi_candidate(power_edges, final_roi_regions)

    # Visualize the selected ROI
    roi_img = np.zeros_like(gray_img)
    if selected_roi:
        start_row, end_row = selected_roi
        roi_img[start_row:end_row, :] = gray_img[start_row:end_row, :]
    show_images([gray_img, roi_img], ["Original", "Selected ROI"])

    # make the image the size of the selected ROI
    if selected_roi:
        start_row, end_row = selected_roi
        roi_img = gray_img[start_row:end_row, :]
        show_images([roi_img], ["Selected ROI"])
    
    
    # Use the selected ROI to locate the license plate columns
    start_col, end_col = locate_license_plate_columns(binary_image[selected_roi[0]:selected_roi[1], 300:500])
    

    # Visualize the located license plate columns
    detected_plate = roi_img[:, start_col+300:end_col+300]
 

    show_images([roi_img, detected_plate], ["Selected ROI", "Detected License Plate"])
    
    
    
    # apply growing window filter
    final_cropped_plate, left_start_col, right_end_col = growing_window_filter(roi_img, start_col+300, end_col+300, binary_image=binary_image[selected_roi[0]:selected_roi[1], :])
    
    # if height of row < 25 pixels expand up and down till the height = 30
    if final_cropped_plate.shape[0] < 25:
        while final_cropped_plate.shape[0] < 30:
            start_row -= 1
            end_row += 1
            final_cropped_plate = gray_img[start_row:end_row, left_start_col:right_end_col]

    show_images([roi_img, final_cropped_plate], ["Selected ROI", "Final Cropped Plate"])

    extracted_plate = extractPlate(img[start_row:end_row, left_start_col:right_end_col])
    # characters = segment_characters(img[start_row:end_row, left_start_col:right_end_col])
    characters = getCharacters(extracted_plate)
    plate_text = potentialCharacter(characters)

    result = get_text_from_image(img[start_row:end_row, left_start_col:right_end_col])
    if  result[0] is not None:
        result = result[0][0][1][0]
    else:
        #result = get_text_from_image_arabic(img[start_row:end_row, left_start_col:right_end_col])
        result = None
    return gray_img,filtered_image, equalized_image, binary_image, inital_roi_image,filtered_roi_img, roi_img, detected_plate, final_cropped_plate , extracted_plate, characters, plate_text, result
    

    

    