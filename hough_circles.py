import cv2
import numpy as np
from collections import defaultdict

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

def hough_transform_circles(input_space, min_radius, max_radius, num_row_bins, num_col_bins, num_radius_bins):
    # Thetas is bins created from 0 to 360 degree with increment of the dtheta
    thetas = np.arange(0, 360, step=int(360 / 100))

    step = int((max_radius - min_radius) / num_radius_bins)

    # Radius ranges from r_min to r_max with step according to num_radius_bins
    rs = np.arange(min_radius, max_radius, step=step)

    cos_thetas = np.cos(np.deg2rad(thetas))
    sin_thetas = np.sin(np.deg2rad(thetas))

    # candidate circles dx and dy for different delta radius
    circle_candidates = []
    for r in rs:
        for t in range(num_thetas):
            circle_candidates.append((r, int(r * cos_thetas[t]), int(r * sin_thetas[t])))

    output_space = defaultdict(int)

    for y in range(num_row_bins):
        for x in range(num_col_bins):
            if input_space[y][x] != 0:  # white pixel
                # Edge pixel here: vote for circle from the candidate circles passing through this pixel.
                for r, rcos_t, rsin_t in circle_candidates:
                    x_center = x - rcos_t
                    y_center = y - rsin_t
                    output_space[(x_center, y_center, r)] += 1  # vote

    return output_space


def find_hough_maxima_circles(output_space, bin_threshold):
    maxima = []

    # Sort the candidates based on the votes
    for candidate_circle, votes in sorted(output_space.items(), key=lambda i: -i[1]):
        x, y, r = candidate_circle
        current_vote_percentage = votes / 100
        if current_vote_percentage >= bin_threshold:
            # Shortlist the circle for final result
            maxima.append((x, y, r, current_vote_percentage))
            print(x, y, r, current_vote_percentage)

    return maxima


def draw_hough_circles(output_img, maxima, pixel_threshold):
    output_img = output_img.copy()
    postprocess_circles = []
    for x, y, r, v in maxima:
        # Remove nearby duplicate circles based on pixel_threshold
        if all(abs(x - xc) > pixel_threshold or abs(y - yc) > pixel_threshold or abs(r - rc) > pixel_threshold for
               xc, yc, rc, v in postprocess_circles):
            postprocess_circles.append((x, y, r, v))
    maxima = postprocess_circles

    # Draw shortlisted circles on the output image
    for x, y, r, v in maxima:
        output_img = cv2.circle(output_img, (x, y), r, (0, 255, 0), 2)

    return output_img


num_row_bins = 300  # Bins are the cells or "vote counters" for each dimension in the output space
num_col_bins = 300  # The output space has shape num_row_bins x num_col_bins x ((max_radius- min_radius)/ num_radius_bins)
min_radius = 30
max_radius = 40
num_radius_bins = max_radius - min_radius
delta_r = 1
num_thetas = 100
bin_threshold = 0.9
min_edge_threshold = 100
max_edge_threshold = 200
pixel_threshold = 5

INPUT_IMAGE = 'circles.jpg'

orig_img = cv2.imread(INPUT_IMAGE)
cv2.imshow('Input Image', orig_img)
cv2.waitKey(1)
edge_img = compute_edge_image(orig_img)

cv2.imshow('Edge Image', edge_img)
cv2.waitKey(1)

print(edge_img.shape[:2])
# resize_edge_image = cv2.resize(edge_img, (100,100), interpolation = cv2.INTER_AREA)
# print(resize_edge_image.shape[:2])
# cv2.imshow('resize_edge_image Image', resize_edge_image)
cv2.waitKey(1)

output_space = hough_transform_circles(edge_img, min_radius, max_radius, num_row_bins, num_col_bins, num_radius_bins)

maxima = find_hough_maxima_circles(output_space, bin_threshold)

# resize_og_image = cv2.resize(orig_img, (100,100), interpolation = cv2.INTER_AREA)
# output_circles_img, circles = draw_hough_circles(resize_og_image, maxima, pixel_threshold)
output_circles_img = draw_hough_circles(orig_img, maxima, pixel_threshold)


cv2.imshow('Input Image with Lines', output_circles_img)
cv2.waitKey(0)