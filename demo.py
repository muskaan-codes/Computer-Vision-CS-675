import cv2
import numpy as np

from assignment2 import label_components_by_flooding


def label_components(img):
    h, w = img.shape
    label_img = np.zeros((h, w), dtype=np.int)
    num_labels = 1
    equiv_classes = []

    # # Use the following code whenever you find an object pixel with mismatching upper and left labels
    # need_new_class = True
    # for cl in equiv_classes:
    #     if cl.intersection({upper, left}):
    #         cl.update({upper, left})
    #         need_new_class = False
    #         break
    # if need_new_class:
    #     equiv_classes.append({upper, left})

    # Create a 1-D array for relabeling label_img after the first pass in a single, efficient step
    # First: Make sure that all pixels of each component have the same label
    relabel_map = np.array(range(num_labels))
    for eq in equiv_classes:
        relabel_map[list(eq)] = min(eq)

    # Second: Make sure that there are no gaps in the labeling
    # For example, a set of labels [0, 1, 2, 4, 5, 7] would turn into [0, 1, 2, 3, 4, 5]
    remaining_labels = np.unique(relabel_map)
    relabel_map = np.searchsorted(remaining_labels, relabel_map)

    # Finally, relabel label_img and return it
    final_label_img = relabel_map[label_img]
    return final_label_img

def size_filter(binary_img, min_size):
    label_matrix = label_components_by_flooding(binary_img)
    _, areas = np.unique(label_matrix, return_counts=True)

    # Again, we use a 1-D array for efficient image manipulation. This filter contains one number for each component.
    # If the component is smaller than the threshold (or it is the background component 0), its entry in the array is 255.
    # Otherwise, its entry is 0. This way, after the mapping, only the above-threshold components will be visible (black).
    filter = np.array(255 * (areas < min_size), dtype=np.uint8)
    filter[0] = 255
    return filter[label_matrix]

input_img = cv2.imread("shape_image.png")
binary_img = np.array(255 * (cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY) > 128), dtype=np.uint8)

cv2.imshow("Input Image", binary_img)
cv2.waitKey(1)

img_filtered = size_filter(binary_img, 4000)

cv2.imshow("Image after Size Filtering", img_filtered)
cv2.waitKey(0)

input_img = cv2.imread("text_image.png")
binary_img = np.array(255 * (cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY) > 128), dtype=np.uint8)

cv2.imshow("Input Image", binary_img)
cv2.waitKey(1)

img_filtered = size_filter(binary_img, 4000)

cv2.imshow("Image after Size Filtering", img_filtered)
cv2.waitKey(0)
# def posneg_size_filter(img, threshold):


def posneg_size_filter(img, threshold):
    cv2.imshow("Input Image", img)
    cv2.waitKey(1)

    imgFilter1 = size_filter(img, threshold)

    cv2.imshow("Filter 1 Image", img)
    cv2.waitKey(0)

    cv2.bitwise_not(imgFilter1, imgFilter2)

    cv2.imshow("Inv Image", imgFilter2)
    cv2.waitKey(2)

    op = size_filter(imgFilter2, threshold)

    cv2.imshow("final Image", op)
    cv2.waitKey(3)


