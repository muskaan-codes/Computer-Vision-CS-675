import cv2
import numpy as np

def dither_random(gray_img):
    height, width = gray_img.shape[:2]
    new_img = np.zeros_like(gray_img)

    for row in range(height):
        for col in range(width):
            if np.random.randint(257) >= gray_img[row, col]:
                new_img[row, col] = 0
            else:
                new_img[row, col] = 255
            
    return new_img

def threshold_image(gray_img, threshold):
    return np.array(255 * (gray_img >= threshold), dtype=np.uint8)


BGR_img = cv2.imread('baby_yoda.jpg')

scale_factor = 1.0

if scale_factor != 1.0:
    BGR_img = cv2.resize(BGR_img, (int(BGR_img.shape[1] * scale_factor), int(BGR_img.shape[0] * scale_factor)), interpolation=cv2.INTER_CUBIC)

gray_img = cv2.cvtColor(BGR_img, cv2.COLOR_BGR2GRAY)
thresh_img = threshold_image(gray_img, 128)
random_img = dither_random(gray_img)

cv2.imshow('Original Gray Image', gray_img)
cv2.imshow('Thresholded Image', thresh_img)
cv2.imshow('Randomly Dithered Image', random_img)

cv2.waitKey(0)
