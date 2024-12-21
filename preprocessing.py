import math
import numpy as np
import skimage.io as io
from skimage.filters import  *
from skimage.feature import  *
import matplotlib.pyplot as plt
from scipy import ndimage


def show_images(images,titles=None):
    #This function is used to show image(s) with titles by sending an array of images and an array of associated titles.
    # images[0] will be drawn with the title titles[0] if exists
    # You aren't required to understand this function, use it as-is.
    n_ims = len(images)
    if titles is None: titles = ['(%d)' % i for i in range(1,n_ims + 1)]
    fig = plt.figure()
    n = 1
    for image,title in zip(images,titles):
        a = fig.add_subplot(1,n_ims,n)
        if image.ndim == 2:
            plt.gray()
        plt.imshow(image)
        a.set_title(title)
        n += 1
    fig.set_size_inches(np.array(fig.get_size_inches()) * n_ims)
    plt.show()

# This function is used to convert an RGB image to a grayscale image using the formula: Y = 0.299R + 0.587G + 0.114B
def rgb2gray(img):
    R = img[:,:,0] # Red channel
    G = img[:,:,1] # Green channel
    B = img[:,:,2] # Blue channel
    gray_img = np.zeros_like(img) # Create a blank image with the same size as the original image
    gray_img = 0.299 * R + 0.587 * G + 0.114 * B # Convert the image to grayscale
    return gray_img

def gaussian(x_square, sigma):
    return np.exp(-0.5 * x_square / sigma ** 2)

# Bilateral filter function
def bilateral_filter(image, sigma_space, sigma_intensity):
    image = image.astype(float)
    # kernel_size should be twice the sigma space to avoid calculating negligible values
    kernel_size = int(2 * sigma_space + 1)
    # Calculate half of the kernel size
    half_kernel_size = kernel_size // 2
    # Initialize the result image with zeros
    result = np.zeros_like(image)
    # Initialize the normalization factor
    W = 0

    # Iterating over the kernel
    for x in range(-half_kernel_size, half_kernel_size + 1):
        for y in range(-half_kernel_size, half_kernel_size + 1):
            # Calculate the spatial Gaussian component
            Gspace = gaussian(x ** 2 + y ** 2, sigma_space)
            # Shift the image by (x, y)
            shifted_image = np.roll(image, [x, y], [1, 0])
            # Calculate the intensity difference
            intensity_difference_image = image - shifted_image
            # Calculate the intensity Gaussian component
            Gintenisity = gaussian(intensity_difference_image ** 2, sigma_intensity)
            # Update the result image
            result += Gspace * Gintenisity * shifted_image
            # Update the normalization factor
            W += Gspace * Gintenisity

    # Normalize the result image
    return result / W

def CLAHE(image, clip_limit=0.005, grid_size=(8, 8)):
    image = image.astype(float)
    rows, cols = image.shape

    # Number of grid tiles in rows and cols
    grid_rows, grid_cols = grid_size
    tile_height = rows // grid_rows
    tile_width = cols // grid_cols

    # Initialize output image
    equalized_image = np.zeros_like(image)

    # Compute histograms and apply CLAHE for each tile
    histograms = []
    for i in range(grid_rows):
        row_histograms = []
        for j in range(grid_cols):
            # Tile boundaries
            start_row = i * tile_height
            end_row = start_row + tile_height if i < grid_rows - 1 else rows
            start_col = j * tile_width
            end_col = start_col + tile_width if j < grid_cols - 1 else cols

            # Extract tile
            tile = image[start_row:end_row, start_col:end_col]

            # Compute histogram and clip it
            hist, _ = np.histogram(tile.flatten(), bins=256, range=(0, 256))
            if clip_limit > 0:
                clip_value = clip_limit * tile.size
                excess = np.maximum(hist - clip_value, 0).sum()
                hist = np.minimum(hist, clip_value)
                # Redistribute excess pixels
                hist += excess // 256

            # Compute cumulative distribution function (CDF)
            cdf = hist.cumsum()
            cdf_normalized = np.clip((cdf / cdf[-1]) * 255, 0, 255)  # Normalize to [0, 255]

            # Save the CDF for this tile
            row_histograms.append(cdf_normalized)
        histograms.append(row_histograms)

    # Apply bilinear interpolation for smooth transitions between tiles
    for i in range(rows):
        for j in range(cols):
            # Determine tile indices
            tile_i = min(i // tile_height, grid_rows - 1)
            tile_j = min(j // tile_width, grid_cols - 1)

            # Get neighboring tiles
            i1 = max(tile_i, 0)
            j1 = max(tile_j, 0)
            i2 = min(tile_i + 1, grid_rows - 1)
            j2 = min(tile_j + 1, grid_cols - 1)

            # Compute relative position inside the tile
            row_ratio = (i - tile_i * tile_height) / tile_height
            col_ratio = (j - tile_j * tile_width) / tile_width

            # Retrieve CDF values
            cdf_11 = histograms[i1][j1]
            cdf_12 = histograms[i1][j2]
            cdf_21 = histograms[i2][j1]
            cdf_22 = histograms[i2][j2]

            # Perform bilinear interpolation
            value = image[i, j]
            interpolated_value = (
                (1 - row_ratio) * (1 - col_ratio) * cdf_11[int(value)]
                + (1 - row_ratio) * col_ratio * cdf_12[int(value)]
                + row_ratio * (1 - col_ratio) * cdf_21[int(value)]
                + row_ratio * col_ratio * cdf_22[int(value)]
            )
            equalized_image[i, j] = interpolated_value

    return equalized_image

def image_binarization(vertical_edges):
    # Calculate the threshold based on mean and standard deviation
    img_mean = np.mean(vertical_edges[vertical_edges != 0])
    thresh = img_mean + 3.5 * img_mean

    # Initialize variables
    binary_image = np.zeros_like(vertical_edges)
    prev_x, prev_y = None, None

    # Iterate through pixels
    for y in range(vertical_edges.shape[0]):
        for x in range(vertical_edges.shape[1]):
            if vertical_edges[y, x] > thresh:
                if prev_x is None and prev_y is None:  # First edge pixel
                    prev_x, prev_y = x, y
                    binary_image[y, x] = 1
                else:
                    # Compute distance to previous edge pixel
                    dist = math.sqrt((prev_x - x) ** 2 + (prev_y - y) ** 2)
                    if dist < 15:
                        binary_image[y, x] = 1
                    else:
                        binary_image[y, x] = 0.5

                    # Update previous coordinates
                    prev_x, prev_y = x, y
            else:
                binary_image[y, x] = 0

    return binary_image
