import cv2
import numpy as np

def label_components_by_flooding(img):

    h, w = img.shape
    label_img = -(img < 128).astype(np.int)     # Object pixels initialized with label -1, background pixels with 0
    current_label = 1                           # Label for first connected component

    for row in range(h):
        for col in range(w):
            if label_img[row, col] < 0:
                open_nodes = {(row, col)}       # Set of known unlabeled pixels (possibly with unlabeled neighbors) in the current component 
                while open_nodes:               # Pop a pixel from the set, label it, and add its neighbors to the set if they are unlabeled object pixels 
                    (r, c) = open_nodes.pop()
                    label_img[r, c] = current_label    
                    if r > 0 and label_img[r - 1, c] < 0: 
                        open_nodes.add((r - 1, c))
                    if r < h - 1 and label_img[r + 1, c] < 0: 
                        open_nodes.add((r + 1, c))
                    if c > 0 and label_img[r, c - 1] < 0: 
                        open_nodes.add((r, c - 1))
                    if c < w - 1 and label_img[r, c + 1] < 0: 
                        open_nodes.add((r, c + 1))
                current_label += 1              # No more unlabeled pixels -> move on to next component and increment the label it will get
                
    return label_img

def label_components(img):
    h, w = img.shape
    label_img = np.zeros((h, w), dtype=np.int)
    num_labels = 1
    equiv_classes = []
    
    # Traverse the image as required by the algorithm
    for row in range(h):
        for col in range(w):
            if img[row, col] == 0:  # For all object pixels, ...
                upper, left = 0, 0  # Determine the labels of their upper and left neighbors (= 0 if there is no neighbor)
                if row > 0:         # Now just consider all the different cases as explained on the slides
                    upper = label_img[row - 1, col]
                if col > 0:
                    left = label_img[row, col - 1]
                if upper == 0 and left == 0:
                    label_img[row, col] = num_labels
                    num_labels += 1
                elif upper == 0:
                    label_img[row, col] = left
                elif left == 0 or left == upper:
                    label_img[row, col] = upper
                else:
                    label_img[row, col] = upper     # Only if there are two different labels, we have to enter them into the equivalence table
                    upper_cl, left_cl = set({upper}), set({left})
            
                    for cl in equiv_classes:
                        if upper in cl:
                            upper_cl = cl
                        if left in cl:
                            left_cl = cl

                    if len(upper_cl) > 1:
                        equiv_classes.remove(upper_cl)
                    if len(left_cl) > 1 and left_cl in equiv_classes:
                        equiv_classes.remove(left_cl)
                    
                    equiv_classes.append(upper_cl.union(left_cl))
                     
    relabel_map = np.array(range(num_labels))
    for eq in equiv_classes:
        relabel_map[list(eq)] = min(eq)
    
    remaining_labels = np.unique(relabel_map)
    relabel_map = np.searchsorted(remaining_labels, relabel_map)

    final_label_img = relabel_map[label_img]
    return final_label_img

def size_filter(binary_img, min_size):
    label_matrix = label_components(binary_img)
    _, areas = np.unique(label_matrix, return_counts=True)

    # Again, we use a 1-D array for efficient image manipulation. This filter contains one number for each component. 
    # If the component is smaller than the threshold (or it is the background component 0), its entry in the array is 255.
    # Otherwise, its entry is 0. This way, after the mapping, only the above-threshold components will be visible (black).
    filter = np.array(255 * (areas < min_size), dtype=np.uint8)
    filter[0] = 255
    return filter[label_matrix]

def posneg_size_filter(img, threshold):
    img_pos_filtered = size_filter(img, threshold)          # Remove positve noise (black pixels) as before
    img_pos_filtered_inv = 255 - img_pos_filtered           # Now invert the image so that the negative noise becomes positive 
    img_posneg_filtered_inv = size_filter(img_pos_filtered_inv, threshold)   # This way the size filter will remove the (formerly) negative noise
    img_posneg_filtered = 255 - img_posneg_filtered_inv     # Finally, invert the image once more so that the objects are black again
    return img_posneg_filtered

def shrink(img):
    h, w = img.shape
    output_img = img.copy()

    for row in range(h):
        for col in range(w):
            if img[row, col] == 0:
                if (row > 0 and img[row - 1, col] == 255) or (row < h - 1 and img[row + 1, col] == 255) or \
                   (col > 0 and img[row, col - 1] == 255) or (col < w - 1 and img[row, col + 1] == 255):
                    output_img[row, col] = 255
    return output_img

def expand(img):
    h, w = img.shape
    output_img = img.copy()

    for row in range(h):
        for col in range(w):
            if img[row, col] == 255:
                if (row > 0 and img[row - 1, col] == 0) or (row < h - 1 and img[row + 1, col] == 0) or \
                   (col > 0 and img[row, col - 1] == 0) or (col < w - 1 and img[row, col + 1] == 0):
                    output_img[row, col] = 0
    return output_img

def expand_and_shrink(img):
    img_expand1 = expand(img)
    img_shrink1 = shrink(img_expand1)
    img_shrink2 = shrink(img_shrink1)
    img_expand2 = expand(img_shrink2)
    return img_expand2
    
     
input_img = cv2.imread("text_image.png")
binary_img = np.array(255 * (cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY) > 128), dtype=np.uint8)

cv2.imshow("Input Image", binary_img)
cv2.waitKey(1)

img_size_filtered = posneg_size_filter(binary_img, 20)

cv2.imshow("Image after Pos-Neg-Size Filtering", img_size_filtered)
cv2.waitKey(1)

img_expand_shrink_filtered = expand_and_shrink(binary_img)

cv2.imshow("Image after Expand-Shrink Noise Removal", img_expand_shrink_filtered)
cv2.waitKey(0)

