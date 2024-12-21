from preprocessing import *

def initial_roi_region(weighted_edges, gray_img):
        
        # Threshold rows depending on edges variance
        row_var = np.var(weighted_edges, axis=1)
        thresh = max(row_var)/2.5
        roi_img = np.zeros(weighted_edges.shape)
        roi_img[row_var>thresh, :] = gray_img[row_var>thresh, :]
        # Get ROI regions and then filter them
        roi_sum = np.sum(roi_img, axis=1)
        roi_start = 0
        roi_end = 0
        roi_regions = []

        inRegion = False
        for i in range(len(roi_sum)):
            if roi_sum[i] != 0 and inRegion == False:
                if len(roi_regions) != 0 and i-roi_regions[-1][1] < 10:
                    roi_start,_ = roi_regions.pop()
                else:
                    roi_start = i
                inRegion = True
            if roi_sum[i] == 0 and inRegion == True:
                roi_end = i-1
                inRegion = False
                
                if roi_end - roi_start >15:
                    roi_regions.append([roi_start, roi_end])

        
        if len(roi_regions) == 0 or roi_regions[-1][0] != roi_start:

            roi_regions.append([roi_start,roi_end])

        filtered_regions = []
        for region in roi_regions:
            if region[1] - region[0] > 10 and region[1] - region[0] < gray_img.shape[0]/3 and region[0] > 200: 

                filtered_regions.append(region)
        return filtered_regions

def findFinalRoiRegion(filtered_roi_regions, gray_img):
    final_roi_regions = []
    for region in filtered_roi_regions:
        height = region[1] - region[0]
        if  height <= gray_img.shape[0] / 4.5:
            if final_roi_regions and region[0] - final_roi_regions[-1][1] < 15:
                final_roi_regions[-1][1] = region[1]
            else:
                final_roi_regions.append(region)
    return final_roi_regions

def update_edge_power(vertical_edges, roi_regions):
    power_edges = np.zeros_like(vertical_edges)
    print(roi_regions)
    cBig = 0
    cSmall = 0
    for region in roi_regions:

        start_row, end_row = region
        for y in range(start_row, end_row + 1):
            for x in range(vertical_edges.shape[1]):
                if vertical_edges[y, x] > 0:
                    # Rule 1: Edges not in the extreme left or right get higher power
                    if x > vertical_edges.shape[1] * 0.1 and x < vertical_edges.shape[1] * 0.9:
                        power_edges[y, x] += 2
                    else:
                        power_edges[y, x] += 0.5

                    # Rule 2: Edges with the same height and occurrence distance less than 25 get maximum power
                    for prev_x in range(max(0, x - 25), min(vertical_edges.shape[1], x + 25)):
                        if vertical_edges[y, prev_x] > 0 and abs(prev_x - x) < 25:
                            power_edges[y, x] += 3
                        
                            

                    # Rule 3: Increase edge power from top to bottom
                    power_edges[y, x] += (y - start_row) / (end_row - start_row + 1)
    print(cSmall, cBig)
    return power_edges

def select_roi_candidate(power_edges, roi_regions):
    max_score = 0
    selected_roi = None
    for region in roi_regions:
        start_row, end_row = region
        region_power = np.sum(power_edges[start_row:end_row + 1, :])
        region_variance = np.var(power_edges[start_row:end_row + 1, :])
        score = region_power * region_variance  # Combine power and variance to score the region
        if region_power > max_score:
            max_score = region_power
            selected_roi = region

    return selected_roi


def locate_license_plate_columns(roi_img, magic_number=4):
    # Calculate the threshold for row variance
    row_var = np.var(roi_img, axis=1)
    mvx = np.mean(row_var)
    thresh_vxroi = mvx + 2.5 * mvx

    # Initialize the signal wave
    signal_wave = np.zeros(roi_img.shape[1])

    # Check each row and form the signal wave
    for row in roi_img:
        signal_wave += (row > thresh_vxroi).astype(int)

    # Calculate the check length
    check_length = len(signal_wave) // 10

    # Find the maximum non-zero count and its location
    max_count = 0
    max_count_index = 0
    for i in range(len(signal_wave) - check_length):
        cnt = np.count_nonzero(signal_wave[i:i + check_length])
        if cnt > max_count:
            max_count = cnt
            max_count_index = i

    # Crop the number plate region based on the max count index
    cropped_plate = roi_img[:, max_count_index:max_count_index + check_length]

    return max_count_index, max_count_index + check_length

def growing_window_filter(roi_img, initial_start_col, initial_end_col, step=10, binary_image=None):
    rows, cols = roi_img.shape

    # Expand to the right
    right_end_col = initial_end_col
    while right_end_col + step < cols:
        right_end_col += step
        if np.sum(binary_image[:, right_end_col:right_end_col + step]) < 0.5 * rows:
            break

    # Expand to the left
    left_start_col = initial_start_col
    while left_start_col - step > 0:
        left_start_col -= step
        if np.sum(binary_image[:, left_start_col - step:left_start_col]) < 0.5 * rows:
            break

    # Crop the final number plate region
    final_cropped_plate = roi_img[:, left_start_col:right_end_col]

    return final_cropped_plate, left_start_col, right_end_col



