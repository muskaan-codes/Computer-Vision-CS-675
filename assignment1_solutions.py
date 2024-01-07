import cv2
import numpy as np

def grayscale_resize(img, new_size):
    old_height, old_width = img.shape
    new_height, new_width = new_size
    new_img = np.zeros(new_size, dtype=np.uint8)

    scale_height = old_height / new_height
    scale_width = old_width / new_width  
    
    for m in range(new_height):
        for n in range(new_width):
            new_img[m, n] = img[int((m + 0.5) * scale_height), int((n + 0.5) * scale_width)]
    
    return new_img

def grayscale_resize_bilinear(img, new_size):
    old_height, old_width = img.shape
    new_height, new_width = new_size

    img_float = img.astype(np.float32)
    new_img = np.zeros(new_size, dtype=np.uint8)

    scale_height = old_height / new_height
    scale_width = old_width / new_width  
    
    for m in range(new_height):
        i0 = (m + 0.5) * scale_height - 0.5
        i = int(i0)                 
        for n in range(new_width):
            j0 = (n + 0.5) * scale_width - 0.5
            j = int(j0)           
            
            if i >= 0 and i < old_height - 1 and j >= 0 and j < old_width - 1: 
                new_img[m, n] = np.uint8((i + 1 - i0) * (j + 1 - j0) * img_float[i,     j    ] + \
                                         (i + 1 - i0) * (j0 - j)     * img_float[i,     j + 1] + \
                                         (i0 - i)     * (j + 1 - j0) * img_float[i + 1, j    ] + \
                                         (i0 - i)     * (j0 - j)     * img_float[i + 1, j + 1] + 0.5)
            else:
                new_img[m, n] = img[int(i0 + 0.5), int(j0 + 0.5)]   # Just use nearest neighbor method if we are too close to the border 

    return new_img
    
def grayscale_dither(img, threshold):
    height, width = img.shape
    new_img = img.astype(np.float32)

    for i in range(height):
        for j in range(width):
            old_val = new_img[i, j]
            new_val = 255 * (old_val >= threshold)
            new_img[i, j] = new_val
            err = new_val - old_val

            if i < height - 1 and j < width - 1:
                new_img[i,     j + 1] -= 7 / 16 * err
                new_img[i + 1, j - 1] -= 3 / 16 * err
                new_img[i + 1, j    ] -= 5 / 16 * err
                new_img[i + 1, j + 1] -= 1 / 16 * err
                 
    return new_img.astype(np.uint8)

def grayscale_dither_multilevel(img, levels):
    height, width = img.shape
    new_img = img.astype(np.float32)
    levels_np = np.array(levels)

    for i in range(height):
        for j in range(width):
            old_val = new_img[i, j]
            new_val = levels_np[np.argmin(np.abs(levels_np - old_val))]     # Find the gray level that is closest to old_val
            new_img[i, j] = new_val
            err = new_val - old_val

            if i < height - 1 and j < width - 1:
                new_img[i,     j + 1] -= 7 / 16 * err
                new_img[i + 1, j - 1] -= 3 / 16 * err
                new_img[i + 1, j    ] -= 5 / 16 * err
                new_img[i + 1, j + 1] -= 1 / 16 * err
                 
    return new_img.astype(np.uint8)
    
def color_dither_multilevel(img, colors):
    height, width = img.shape[:2]
    new_img = img.astype(np.float32)
    colors_np = np.array(colors)

    for i in range(height):
        for j in range(width):
            old_val = new_img[i, j].copy()
            new_val = colors_np[np.argmin(np.linalg.norm(colors_np - old_val, axis=1))]     # Find the BGR vector in colors that is closest to the one in old_val
            new_img[i, j] = new_val                                                         # Distance between two colors (B1, G1, R1) and (B2, G2, R2) is computed as
            err = new_val - old_val                                                         # the L2-norm for vectors: sqrt((B1 - B2)**2 + (G1 - G2)**2 + (R1 - R2)**2)

            if i < height - 1 and j < width - 1:
                new_img[i,     j + 1] -= 7 / 16 * err
                new_img[i + 1, j - 1] -= 3 / 16 * err
                new_img[i + 1, j    ] -= 5 / 16 * err
                new_img[i + 1, j + 1] -= 1 / 16 * err
                 
    return new_img.astype(np.uint8)
