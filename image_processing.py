import cv2
import numpy as np

ogImg = cv2.imread("Downloads/baby_yoda.jpg")

cv2.imshow("Original Image:", ogImg)

cv2.waitKey(0)


def minmax(v):
    if v > 255:
        v = 255
    if v < 0:
        v = 0
    return v

def grayscale_dither(inGrayImg, threshold):
    # image dimensions
    h, w = inGrayImg.shape[:2]
    
    # loop over the image
    for y in range(h-1):  # pixels in last row can be thresholded without any adjustment
        for x in range(1, w-1):  # pixels in first and last column can be thresholded without any adjustment
            # threshold the pixel
            oldPixel = inGrayImg[y, x]
            newPixel = np.round(threshold * oldPixel/255.0) * (255/threshold)
            inGrayImg[y, x] = newPixel
            
            quantErrorPixel = oldPixel - newPixel
            
            inGrayImg[y, x+1] = minmax(inGrayImg[y, x+1] + quantErrorPixel * 7 / 16.0)
            inGrayImg[y+1, x-1] = minmax(inGrayImg[y+1, x-1] + quantErrorPixel * 3 / 16.0)
            inGrayImg[y+1, x] = minmax(inGrayImg[y+1, x] + quantErrorPixel * 5 / 16.0)
            inGrayImg[y+1, x+1] = minmax(inGrayImg[y+1, x+1] + quantErrorPixel * 1 / 16.0)
            
    # return dithered image        
    return inGrayImg


grayMat = cv2.cvtColor(ogImg, cv2.COLOR_BGR2GRAY)
outMat_gray = grayscale_dither(grayMat, 1)
cv2.imshow('gray.jpg', outMat_gray)
cv2.waitKey(0)

def grayscale_dither_multilevel(inGrayImg, levels):
    
    def minmax_multilevel(v):
        levelCount = len(levels)
        levels.sort()
        for l in range(levelCount):
            if l == 0:
                if v < levels[0]:
                    v = levels[0]
                    break
            elif l == levelCount - 1:
                if v > levels[levelCount-1]:
                    v = levels[levelCount-1]
                    break
            else:
                if v > levels[l-1] and v < levels[l]:
                    v = levels[l]
                    break
        return v

    # image dimensions
    h, w = inGrayImg.shape[:2]
    threshold = 1
    
    # loop over the image
    for y in range(h-1):  # pixels in last row can be thresholded without any adjustment
        for x in range(1, w-1):  # pixels in first and last column can be thresholded without any adjustment
            # threshold the pixel
            oldPixel = inGrayImg[y, x]
            newPixel = np.round(threshold * oldPixel/255.0) * (255/threshold)
            inGrayImg[y, x] = newPixel
            
            quantErrorPixel = oldPixel - newPixel
            
            inGrayImg[y, x+1] = minmax_multilevel(inGrayImg[y, x+1] + quantErrorPixel * 7 / 16.0)
            inGrayImg[y+1, x-1] = minmax_multilevel(inGrayImg[y+1, x-1] + quantErrorPixel * 3 / 16.0)
            inGrayImg[y+1, x] = minmax_multilevel(inGrayImg[y+1, x] + quantErrorPixel * 5 / 16.0)
            inGrayImg[y+1, x+1] = minmax_multilevel(inGrayImg[y+1, x+1] + quantErrorPixel * 1 / 16.0)
            
    # return dithered image        
    return inGrayImg

outMat_gray_multilevel = grayscale_dither_multilevel(grayMat, [0, 85, 170, 255])
cv2.imshow('grayMultilevel.jpg', outMat_gray_multilevel)
cv2.waitKey(0)

outMat_gray_multilevel_less = grayscale_dither_multilevel(grayMat, [50, 85, 155])
cv2.imshow('grayMultilevelLess.jpg', outMat_gray_multilevel_less)
cv2.waitKey(0)

def color_dither_multilevel(colorImg, levels):
    """ input: levels: (list)(integer) B,G,R order"""
    
    def minmax_multilevel(v, colorIndex):
        levelCount = len(levels)
        levels.sort()
        for l in range(levelCount):
            if l == 0:
                if v < levels[0][colorIndex]:
                    v = levels[0][colorIndex]
                    break
            elif l == levelCount - 1:
                if v > levels[levelCount-1][colorIndex]:
                    v = levels[levelCount-1][colorIndex]
                    break
            else:
                if v > levels[l-1][colorIndex] and v < levels[l][colorIndex]:
                    v = levels[l][colorIndex]
                    break
        return v
    
    # similar to grayscale but doing for B,G,R separately 
    h, w = colorImg.shape[:2]
    threshold = 1
     
    # loop over the image
    for y in range(h-1):  # pixels in last row can be thresholded without any adjustment
        for x in range(1, w-1):  # pixels in first and last column can be thresholded without any adjustment
            # threshold the pixel
            oldB = colorImg[y, x, 0]
            oldG = colorImg[y, x, 1]
            oldR = colorImg[y, x, 2]
            
            newB = np.round(threshold * oldB/255.0) * (255/threshold)
            newG = np.round(threshold * oldG/255.0) * (255/threshold)
            newR = np.round(threshold * oldR/255.0) * (255/threshold)

            colorImg[y, x, 0] = newB
            colorImg[y, x, 1] = newG
            colorImg[y, x, 2] = newR

            quantErrorB = oldB - newB
            quantErrorG = oldG - newG
            quantErrorR = oldR - newR

            colorImg[y, x+1, 0] = minmax_multilevel(colorImg[y, x+1, 0] + quantErrorB * 7 / 16.0, 0)
            colorImg[y, x+1, 1] = minmax_multilevel(colorImg[y, x+1, 1] + quantErrorG * 7 / 16.0, 1)
            colorImg[y, x+1, 2] = minmax_multilevel(colorImg[y, x+1, 2] + quantErrorR * 7 / 16.0, 2)
            
            colorImg[y+1, x-1, 0] = minmax_multilevel(colorImg[y+1, x-1, 0] + quantErrorB * 3 / 16.0, 0)
            colorImg[y+1, x-1, 1] = minmax_multilevel(colorImg[y+1, x-1, 1] + quantErrorG * 3 / 16.0, 1)
            colorImg[y+1, x-1, 2] = minmax_multilevel(colorImg[y+1, x-1, 2] + quantErrorR * 3 / 16.0, 2)

            colorImg[y+1, x, 0] = minmax_multilevel(colorImg[y+1, x, 0] + quantErrorB * 5 / 16.0, 0)
            colorImg[y+1, x, 1] = minmax_multilevel(colorImg[y+1, x, 1] + quantErrorG * 5 / 16.0, 1)
            colorImg[y+1, x, 2] = minmax_multilevel(colorImg[y+1, x, 2] + quantErrorR * 5 / 16.0, 2)

            colorImg[y+1, x+1, 0] = minmax_multilevel(colorImg[y+1, x+1, 0] + quantErrorB * 1 / 16.0, 0)
            colorImg[y+1, x+1, 1] = minmax_multilevel(colorImg[y+1, x+1, 1] + quantErrorG * 1 / 16.0, 1)
            colorImg[y+1, x+1, 2] = minmax_multilevel(colorImg[y+1, x+1, 2] + quantErrorR * 1 / 16.0, 2)

    # return dithered image
    return colorImg

outMat_multilevel = color_dither_multilevel(ogImg, [[0,0,0], [85,85,85], [170,170,170], [255,255,255]])
cv2.imshow('colorMultilevel.jpg', outMat_multilevel)
cv2.waitKey(0)

# import math

# def grayscale_resize(img, new_h, new_w):
#     h, w = img.shape[:2]
#     temp = np.empty([new_h, new_w])
#     x_ratio = w/new_w
#     y_ratio = h/new_h
#     # traverse pixels in image
#     for i in range(new_h):
#         for j in range(new_w):
#             px = math.floor(j*x_ratio)
#             py = math.floor(i*y_ratio)
#             temp[i*new_w][j] = img[py*w][px]

#     return temp

def grayscale_resize(img, new_h, new_w):
    def per_axis(in_sz, out_sz):
        ratio = 0.5 * in_sz / out_sz
        return np.round(np.linspace(ratio - 0.5, in_sz - ratio - 0.5, num=out_sz)).astype(int)

    return img[per_axis(img.shape[0], new_h)[:, None],
               per_axis(img.shape[1], new_w)]

gray_resize = grayscale_resize(grayMat, 500, 800)
cv2.imshow('grayResize.jpg', gray_resize)
cv2.waitKey(0)

####Trial 
import math

def grayscale_resize_bilinear(img, new_h, new_w):
  
  h, w = img.shape[:2]

  resized = np.empty([new_h, new_w])

  x_ratio = float((w - 1) / (new_w - 1)) if new_w > 1 else 0
  y_ratio = float((h - 1) / (new_h - 1)) if new_h > 1 else 0

  for i in range(new_h):
    for j in range(new_w):

      x_l, y_l = math.floor(x_ratio * j), math.floor(y_ratio * i)
      x_h, y_h = math.ceil(x_ratio * j), math.ceil(y_ratio * i)

      x_weight = (x_ratio * j) - x_l
      y_weight = (y_ratio * i) - y_l

      a = img[y_l, x_l]   # 4 closest neighbours
      b = img[y_l, x_h]
      c = img[y_h, x_l]
      d = img[y_h, x_h]

      pixel = a * (1 - x_weight) * (1 - y_weight) + b * x_weight * (1 - y_weight) + c * y_weight * (1 - x_weight) + d * x_weight * y_weight

      resized[i][j] = pixel

  return resized

gray_bl_resize = grayscale_resize_bilinear(grayMat, 250, 400)
cv2.imshow('grayblResize.jpg', gray_bl_resize)
cv2.waitKey(0)


