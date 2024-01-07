# Test script for Assignment #1
# Point homework_file to the file containing your code (minus the '.py' extension), which must be in the same dircetory as this script, and
# choose an output_dir if you want to save the resulting images (SAVE_IMAGES = True)
# If you use VS Code, you may have to add the line 
# "code-runner.runInTerminal": true 
# to your settings.json file.
 
import cv2
import numpy as np
import importlib
import inspect
from colorama import Fore, Style, init

init()

output_dir = 'nambati/HW1'
homework_file = 'image_processing'
yoda_file, magnus_file, rects_file = 'baby_yoda.jpg', 'magnus.png', 'rectangles.png'

SHOW_IMAGES = False
SAVE_IMAGES = True

def func_available(funcs, func_name):
    if hasattr(funcs, func_name):
        print(Fore.GREEN + f'\nFunction {func_name} found!' + Style.RESET_ALL)
        return True
    print(Fore.RED + f'\nFunction {func_name} not found!' + Style.RESET_ALL) 
    return False
    
def apply_func(func, arguments, title):
    try:
        new_img = func(*arguments)
    except Exception as e:
        print(Fore.RED + 'Error while computing image "%s":\n%s'%(title, e) + Style.RESET_ALL)
        new_img = np.zeros((100, 200), dtype=np.uint8)
        cv2.putText(new_img, 'Error!', (70, 40), cv2.FONT_HERSHEY_SIMPLEX, 2, 255, 2)

    full_title = '%s - Size: %dx%d'%(title, new_img.shape[0], new_img.shape[1])
    print(full_title)
    
    if SHOW_IMAGES: 
        cv2.imshow(title, new_img)
        cv2.waitKey(1)
    if SAVE_IMAGES: 
        cv2.imwrite('%s/%s.png'%(output_dir, title), new_img)
    
    return new_img

def get_BGR_combinations(levels):
    colors = []
    for b in levels:
        for g in levels:
            for r in levels:
                colors.append([b, g, r])
    return colors
    
funcs = importlib.import_module(homework_file)
imported_functions = inspect.getmembers(funcs, inspect.isfunction)

print(Fore.BLUE + '--------------------------------------' + Style.RESET_ALL)
print('Module "%s" includes the following functions:\n'%(homework_file))

for f in imported_functions:
    print(f[0])

print(Fore.BLUE + '--------------------------------------\n' + Style.RESET_ALL)

BGR_yoda = cv2.imread(yoda_file)
gray_yoda = cv2.cvtColor(BGR_yoda, cv2.COLOR_BGR2GRAY)

BGR_magnus = cv2.imread(magnus_file)
gray_magnus = cv2.cvtColor(BGR_magnus, cv2.COLOR_BGR2GRAY)

BGR_rects = cv2.imread(rects_file)
gray_rects = cv2.cvtColor(BGR_rects, cv2.COLOR_BGR2GRAY)

if func_available(funcs, 'grayscale_resize'):
    small_gray_magnus_NN = apply_func(funcs.grayscale_resize, (gray_magnus, (90, 70)), 'Small Gray Magnus NN (90x70)')
    wide_gray_magnus_NN = apply_func(funcs.grayscale_resize, (gray_magnus, (600, 1500)), 'Wide Gray Magnus NN (600x1500)')
    tall_gray_magnus_NN = apply_func(funcs.grayscale_resize, (gray_magnus, (900, 200)), 'Tall Gray Magnus NN (900x200)')
    upscaled_gray_magnus_NN = apply_func(funcs.grayscale_resize, (small_gray_magnus_NN, (800, 600)), 'Upscaled Gray Magnus NN (800x600)')

if func_available(funcs, 'grayscale_resize_bilinear'):
    small_gray_magnus_BL = apply_func(funcs.grayscale_resize_bilinear, (gray_magnus, (90, 70)), 'Small Gray Magnus BL (90x70)')
    wide_gray_magnus_BL = apply_func(funcs.grayscale_resize_bilinear, (gray_magnus, (600, 1500)), 'Wide Gray Magnus BL (600x1500)')
    tall_gray_magnus_BL = apply_func(funcs.grayscale_resize_bilinear, (gray_magnus, (900, 200)), 'Tall Gray Magnus BL (900x200)')
    upscaled_gray_magnus_BL = apply_func(funcs.grayscale_resize_bilinear, (small_gray_magnus_BL, (800, 600)), 'Upscaled Gray Magnus BL (800x600)')
    
if func_available(funcs, 'grayscale_dither'):
    gray_dither_thresh_yoda   = apply_func(funcs.grayscale_dither, (gray_yoda, 128), 'Gray threshold-dithered Yoda')
    gray_dither_thresh_magnus = apply_func(funcs.grayscale_dither, (gray_magnus, 128), 'Gray threshold-dithered Magnus')
    gray_dither_thresh_rects = apply_func(funcs.grayscale_dither, (gray_rects, 128), 'Gray threshold-dithered Rectangles')
    
if func_available(funcs, 'grayscale_dither_multilevel'):
    gray_dither2_yoda   = apply_func(funcs.grayscale_dither_multilevel, (gray_yoda, [0, 255]), 'Gray 2-level dithered Yoda')
    gray_dither3_yoda   = apply_func(funcs.grayscale_dither_multilevel, (gray_yoda, [0, 127, 255]), 'Gray 3-level dithered Yoda')
    gray_dither4_yoda   = apply_func(funcs.grayscale_dither_multilevel, (gray_yoda, [0, 85, 170, 255]), 'Gray 4-level dithered Yoda')
    gray_dither8_yoda   = apply_func(funcs.grayscale_dither_multilevel, (gray_yoda, [0, 36, 73, 109, 146, 182, 219, 255]), 'Gray 8-level dithered Yoda')
    gray_dither2_magnus   = apply_func(funcs.grayscale_dither_multilevel, (gray_magnus, [0, 255]), 'Gray 2-level dithered Magnus')
    gray_dither3_magnus   = apply_func(funcs.grayscale_dither_multilevel, (gray_magnus, [0, 127, 255]), 'Gray 3-level dithered Magnus')
    gray_dither4_magnus   = apply_func(funcs.grayscale_dither_multilevel, (gray_magnus, [0, 85, 170, 255]), 'Gray 4-level dithered Magnus')
    gray_dither8_magnus   = apply_func(funcs.grayscale_dither_multilevel, (gray_magnus, [0, 36, 73, 109, 146, 182, 219, 255]), 'Gray 8-level dithered Magnus')
    
if func_available(funcs, 'color_dither_multilevel'):
    levels5 = [[255, 0, 0], [0, 255, 0], [0, 0, 255], [0, 0, 0], [255, 255, 255]]
    color_dither5_yoda = apply_func(funcs.color_dither_multilevel, (BGR_yoda, levels5), '5-color dithered Yoda')
    color_dither5_magnus = apply_func(funcs.color_dither_multilevel, (BGR_magnus, levels5), '5-color dithered Magnus')
    color_dither5_rectangles = apply_func(funcs.color_dither_multilevel, (BGR_rects, levels5), '5-color dithered Rectangles')

    levels8 = get_BGR_combinations([0, 255])    
    color_dither8_yoda = apply_func(funcs.color_dither_multilevel, (BGR_yoda, levels8), '8-color dithered Yoda')
    color_dither8_magnus = apply_func(funcs.color_dither_multilevel, (BGR_magnus, levels8), '8-color dithered Magnus')
    color_dither8_rectangles = apply_func(funcs.color_dither_multilevel, (BGR_rects, levels8), '8-color dithered Rectangles')
    
    levels27 = get_BGR_combinations([0, 127, 255])    
    color_dither27_yoda = apply_func(funcs.color_dither_multilevel, (BGR_yoda, levels27), '27-color dithered Yoda')
    color_dither27_magnus = apply_func(funcs.color_dither_multilevel, (BGR_magnus, levels27), '27-color dithered Magnus')
    color_dither27_rectangles = apply_func(funcs.color_dither_multilevel, (BGR_rects, levels27), '27-color dithered Rectangles')

    levels64 = get_BGR_combinations([0, 85, 170, 255])    
    color_dither64_yoda = apply_func(funcs.color_dither_multilevel, (BGR_yoda, levels64), '64-color dithered Yoda')
    color_dither64_magnus = apply_func(funcs.color_dither_multilevel, (BGR_magnus, levels64), '64-color dithered Magnus')
    color_dither64_rectangles = apply_func(funcs.color_dither_multilevel, (BGR_rects, levels64), '64-color dithered Rectangles')

if SHOW_IMAGES:
    cv2.waitKey(0)