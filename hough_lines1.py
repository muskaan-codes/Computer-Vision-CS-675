# Hough Transform Demo for CS 675 at UMass Boston
# Finding the most significant straight contours in a given image

import cv2
import numpy as np
import math

def compute_edge_image(bgr_img):
    """ Compute the edge magnitude of an image using a pair of Sobel filters """

    sobel_v = np.array([[-1, -2, -1],   # Sobel filter for the vertical gradient. Note that the filter2D function computes a correlation
                        [ 0,  0, 0],    # instead of a convolution, so the filter is *not* rotated by 180 degrees.
                        [ 1,  2, 1]])
    sobel_h = sobel_v.T                 # The convolution filter for the horizontal gradient is simply the transpose of the previous one  

    gray_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)
    gradient_v = cv2.filter2D(gray_img, ddepth=cv2.CV_32F, kernel=sobel_v)
    gradient_h = cv2.filter2D(gray_img, ddepth=cv2.CV_32F, kernel=sobel_h)
    gradient_magni = np.sqrt(gradient_v**2 + gradient_h**2)
    near_max = np.percentile(gradient_magni, 99.5)      # Clip magnitude at percentile 99.5 to prevent outliers from determining the range of relevant magnitudes  
    edge_img = np.clip(gradient_magni * 255.0 / near_max, 0.0, 255.0).astype(np.uint8)  # Normalize and convert magnitudes into grayscale image 
    return edge_img

def hough_transform_lines(input_space, num_alpha_bins, num_d_bins):
    """" Perform Hough transform of an image to an alpha-d space represemnting straight lines """

    output_space = np.zeros((num_alpha_bins, num_d_bins), dtype=np.int)

    # Generate matrix of sine and cosine values for each alpha bin to speed up subsequent computation.
    d_max = math.sqrt(input_space.shape[0]**2 + input_space.shape[1]**2)
    alpha_bins = np.linspace(-0.5 * math.pi, math.pi, num_alpha_bins) # Value of angle alpha for each bin, i.e., row in the output matrix
    cos_sin_matrix = np.column_stack((np.cos(alpha_bins), np.sin(alpha_bins))) * num_d_bins / d_max     # cos and sin values of alpha for each bin

    edge_coords = np.row_stack(np.nonzero(input_space >= 64))   # Only consider edges exceeding a threshold; higher threshold speeds up computation but ignores more edges
    edge_magnitudes = input_space[edge_coords[0, :], edge_coords[1, :]]     # List of magnitudes of all considered edges 
    d_bins = np.matmul(cos_sin_matrix, edge_coords).astype(np.int)          # Matrix multiplication yields matrix of d-values of shape num_alpha_bins x number of considered edges
    
    for alpha_bin in range(num_alpha_bins):
        for edge_index in range(len(edge_magnitudes)):
            d_bin = d_bins[alpha_bin, edge_index]
            if d_bin >= 0.5:                                                    # For each considered edge and value of alpha, if d is positive,
                output_space[alpha_bin, d_bin] += edge_magnitudes[edge_index]   # increase the current (alpha, d) counter by the edge magnitude
    
    return output_space, alpha_bins, d_max

def find_hough_maxima(output_space, num_maxima, min_dist_alpha, min_dist_d):
    """ Find the given number of vote maxima in the output space with certain minimum distances in alpha and d between them """
    maxima = []
    output_copy = output_space.copy()
    height, width = output_copy.shape
    
    for i in range(num_maxima):
        row, col = np.unravel_index(np.argmax(output_copy), (height, width))    # Get coordinates (alpha, d) of global maximum
        maxima.append((row, col))
        output_copy[max(0, row - min_dist_alpha):min(height - 1, row + min_dist_alpha),     # Set all cells within the minimum distances to -1
                    max(0, col - min_dist_d):    min(width - 1,  col + min_dist_d)] = -1.0  # so that no further maxima will be selected from this area
    
    return maxima       # Return list of (alpha, d) pairs indicating locations of maxima 

def draw_hough_line(img, alpha, d):
    """ Add a straight line with parameters alpha (radians) and d (pixels) to a given image """

    h, w = img.shape[:2]
    i0, j0 = d * math.cos(alpha), d * math.sin(alpha)   # Determine (i0, j0) where normal intersects with line
    for s in range(-h - w, h + w):                      # Cover a wide range of distances to find all points of the line within the image...
        i, j = int(i0 - s * math.sin(alpha) + 0.5), int(j0 + s * math.cos(alpha) + 0.5)     # ... by going from (i0, j0) in perpendicular direction from the normal
        if i >= 0 and i < h and j >= 0 and j < w:
            if s % 20 < 10:                             # Draw black and white dashes for better visibility
                color = (0, 0, 0)
            else:
                color = (255, 255, 255)
            cv2.circle(img, (j, i), 1, color, 2)

NUM_ALPHA_BINS = 500        # Bins are the cells or "vote counters" for each dimension in the output space
NUM_D_BINS = 500            # The output space has shape NUM_ALPHA_BINS x NUM_D_BINS
NUM_MAXIMA = 9              # Number of most significant lines to be found in the input image
INPUT_IMAGE = 'desk.png'

orig_img = cv2.imread(INPUT_IMAGE)
cv2.imshow('Input Image', orig_img)
cv2.waitKey(1)

edge_img = compute_edge_image(orig_img)

cv2.imshow('Edge Image', edge_img)
cv2.waitKey(1)

output_space, alpha_bins, d_max = hough_transform_lines(edge_img, NUM_ALPHA_BINS, NUM_D_BINS)
output_img = (output_space * 255.0 / np.max(output_space)).astype(np.uint8)
output_max_img = cv2.cvtColor(output_img, cv2.COLOR_GRAY2BGR)       # Convert output space image to color so we can mark maxima with red circles
output_lines_img = orig_img.copy()
line_parameters = find_hough_maxima(output_space, NUM_MAXIMA, 40, 40)

for (alpha, d) in line_parameters:
    cv2.circle(output_max_img, (d, alpha), 10, (0, 0, 255), 2)
    draw_hough_line(output_lines_img, alpha_bins[alpha], d * d_max / NUM_D_BINS)

cv2.imshow('Output Space with Maxima', output_max_img)
cv2.imshow('Input Image with Lines', output_lines_img)
cv2.waitKey(0)
