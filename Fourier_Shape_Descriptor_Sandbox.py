import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog

gui = dict()    # Contains all global variables
font = cv2.FONT_HERSHEY_SIMPLEX

def load_image():
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename()
    img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    padded_img = cv2.copyMakeBorder(img, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=255) # Add a border of white pixels to avoid border checks 
    start_pixel = np.array(np.where(padded_img < 128))[:, 0]     # Starting point: Leftmost pixel in top row of object
    neighbors = np.array([[0, -1], [1, -1], [1, 0], [1, 1], [0, 1], [-1, 1], [-1, 0], [-1, -1]]) # relative 8-neighbor pixel coordinates in CCW order 
    current_pixel = start_pixel.copy()
    last_white_pixel = current_pixel - [0, 1]
    boundary = []
    
    while True:
        neighbor_index = np.argmin(np.linalg.norm(neighbors + current_pixel - last_white_pixel, axis=1))
        while padded_img[tuple(current_pixel + neighbors[neighbor_index])] > 127:
            last_white_pixel = current_pixel + neighbors[neighbor_index]
            neighbor_index = (neighbor_index + 1) % 8
        current_pixel += neighbors[neighbor_index]
        boundary.append(current_pixel.copy())
        if np.array_equal(current_pixel, start_pixel):
            break
    
    boundary = np.array(boundary, dtype=float)
    boundary -= np.mean(boundary, axis=0) 
    scale = np.max(np.abs(boundary))
    boundary *= 0.85 / scale

    input_steps = len(boundary)
    descriptor_length = input_steps // 2 + 1
    fourier_v = np.fft.rfft(boundary[:, 0]) / descriptor_length
    fourier_h = np.fft.rfft(boundary[:, 1]) / descriptor_length
    gui['orig_boundary'] = boundary

    reset_descriptors()
    max_freq = min(gui['max_num_freqs'], descriptor_length)    
    gui['ampli_v'] = np.abs(fourier_v[:max_freq])
    gui['phase_v'] = np.angle(fourier_v[:max_freq])
    gui['ampli_h'] = np.abs(fourier_h[:max_freq])
    gui['phase_h'] = np.angle(fourier_h[:max_freq])
    gui['num_freqs'] = gui['max_num_freqs']
    gui['path_pos'] = gui['path_length'] - 1
    compute_fourier_path()
    
def compute_fourier_path():
    input_v = np.zeros(gui['path_length'] // 2 + 1, dtype=complex)
    input_v[:gui['num_freqs']] = gui['ampli_v'][:gui['num_freqs']] * np.exp(gui['phase_v'][:gui['num_freqs']] * 1j) 
    path_v = np.fft.irfft(input_v) * len(input_v)
    if gui['show_1d']:
        path_h = np.linspace(-1.0, 1.0, len(path_v))
    else:
        input_h = np.zeros(gui['path_length'] // 2 + 1, dtype=complex)
        input_h[:gui['num_freqs']] = gui['ampli_h'][:gui['num_freqs']] * np.exp(gui['phase_h'][:gui['num_freqs']] * 1j) 
        path_h = np.fft.irfft(input_h) * len(input_h)
    gui['space_pixels_v'] = np.clip(np.array(gui['space_center'][0] + gui['space_size'] * path_v, dtype=int), 0, gui['window_size'][0] - 1)
    gui['space_pixels_h'] = np.clip(np.array(gui['space_center'][1] + gui['space_size'] * path_h, dtype=int), 0, gui['window_size'][1] - 1)
    chart_center = (gui['path_chart'][2] + gui['path_chart'][0]) / 2.0
    chart_height = (gui['path_chart'][2] - gui['path_chart'][0]) / 2.0
    gui['chart_pixels_v'] = np.clip(np.array(chart_center + chart_height * path_v, dtype=int), gui['path_chart'][0], gui['path_chart'][2] - 1)
    gui['chart_pixels_h'] = np.clip(np.array(chart_center + chart_height * path_h, dtype=int), gui['path_chart'][0], gui['path_chart'][2] - 1)
    gui['chart_pixels_x'] = np.linspace(gui['path_chart'][1], gui['path_chart'][3], len(path_v)).astype(int)
    
def draw_wheels(ampli_key, phase_key, center_key, path_phase, phase_shift, base_color):
    sz, freqs, bm = gui['space_size'], gui['num_freqs'], gui['bitmap']
    ampli, phase = gui[ampli_key][:freqs], gui[phase_key][:freqs] + phase_shift + path_phase * np.arange(freqs)
    center = np.array(gui[center_key], dtype=int)
    bm[center[0] - sz - 20:center[0] + sz + 20, center[1] - sz - 20:center[1] + sz + 20, :] = 0
    bm[center[0], center[1] - 10:center[1] + 11, :] = 128
    bm[center[0] - 10:center[0] + 11, center[1], :] = 128
    coords = np.zeros(2)

    for i in range(1, freqs):
        cv2.circle(bm, (int(center[1] + sz * coords[1]), int(center[0] + sz * coords[0])), int(sz * ampli[i]), base_color, 1)
        coords += ampli[i] * np.cos([phase[i], phase[i] - np.pi / 2.0]) 
        cv2.circle(bm, (int(center[1] + sz * coords[1]), int(center[0] + sz * coords[0])), 2, base_color, -1)

    return (center + sz * coords).astype(int)

def draw_wave_map(ampli_key, phase_key, center_key, path_phase, phase_shift, base_color, dim_index):
    sz, freqs, bm = gui['wave_map_radius'], gui['num_freqs'], gui['bitmap']
    ampli, phase = gui[ampli_key][:freqs], gui[phase_key][:freqs] + phase_shift + path_phase * np.arange(freqs)
    log_ampli = np.log(1000.0 * ampli + 1.0) / np.log(1001.0) 
    center = np.array(gui[center_key], dtype=int)
    bm[center[0] - sz - 20:center[0] + sz + 20, center[1] - sz - 20:center[1] + sz + 20, :] = 0
    bm[center[0], center[1] - 10:center[1] + 11, :] = 128
    bm[center[0] - 10:center[0] + 11, center[1], :] = 128
    
    for i in range(1, freqs):
        cv2.circle(bm, (center[1], center[0]), int(sz * log_ampli[i]), base_color, 1)
    for i in range(1, freqs):
        gui['wave_map_coords'][dim_index, i] = center + sz * log_ampli[i] * np.cos([phase[i], phase[i] - np.pi / 2.0])  
        y, x = gui['wave_map_coords'][dim_index, i].astype(int)
        cv2.circle(bm, (x, y), 10, base_color, -1)
        if i < 10:
            x_offset = -4
        else:
            x_offset = -8
        cv2.putText(bm, str(i), (x + x_offset, y + 5), font, 0.4, (255, 255, 255), 1)

def update_gui():
    if gui['update_all'] or gui['update_freq_bar']:
        gui['bitmap'][:, gui['freq_bar'][1] - 10:gui['freq_bar'][3] + 15, :] = 0
        bar_height = int((gui['freq_bar'][2] - gui['freq_bar'][0]) * gui['num_freqs'] / gui['max_num_freqs'])
        cv2.rectangle(gui['bitmap'], (gui['freq_bar'][1], gui['freq_bar'][2] - bar_height), (gui['freq_bar'][3], gui['freq_bar'][2]), [0, 255, 0], -1)
        cv2.rectangle(gui['bitmap'], (gui['freq_bar'][1], gui['freq_bar'][0]), (gui['freq_bar'][3], gui['freq_bar'][2]), [255, 255, 255], 1)
        cv2.putText(gui['bitmap'], str(gui['num_freqs'] - 1), (gui['freq_bar'][1], gui['freq_bar'][0] - 15), font, 0.7, (255, 255, 255), 1)
        gui['update_freq_bar'] = False
    
    if gui['update_all'] or gui['update_path_chart']:
        gui['bitmap'][gui['path_bar'][0]:gui['path_chart'][2], gui['path_bar'][1]:gui['path_bar'][3], :] = 0
        bar_width = int((gui['path_bar'][3] - gui['path_bar'][1]) * gui['path_pos'] / gui['path_length'])
        cv2.rectangle(gui['bitmap'], (gui['path_bar'][1], gui['path_bar'][0]), (gui['path_bar'][1] + bar_width, gui['path_bar'][2]), [0, 255, 0], -1)
        cv2.rectangle(gui['bitmap'], (gui['path_bar'][1], gui['path_bar'][0]), (gui['path_bar'][3], gui['path_bar'][2]), [255, 255, 255], 1)
        gui['bitmap'][gui['path_chart'][0]:gui['path_chart'][2], gui['path_chart'][1], :] = 128 
        gui['bitmap'][gui['path_chart'][0]:gui['path_chart'][2], gui['path_chart'][3], :] = 128
        gui['bitmap'][(gui['path_chart'][0] + gui['path_chart'][2]) // 2, gui['path_chart'][1]:gui['path_chart'][3], :] = 128
        gui['bitmap'][gui['path_chart'][2], gui['path_chart'][1]:gui['path_chart'][3], :] = 128
        gui['bitmap'][gui['path_chart'][0]:gui['path_chart'][2], gui['path_chart'][1] + bar_width, 1] = 255 
        gui['bitmap'][gui['chart_pixels_v'], gui['chart_pixels_x'], 0] = 255
        if not gui['show_1d']:
            gui['bitmap'][gui['chart_pixels_h'], gui['chart_pixels_x'], 2] = 255
        gui['update_path_chart'] = False
    
    if gui['update_all'] or gui['update_space']:
        top, center_v, bottom = gui['space_center'][0] - gui['space_size'], gui['space_center'][0], gui['space_center'][0] + gui['space_size'] 
        left, center_h, right = gui['space_center'][1] - gui['space_size'], gui['space_center'][1], gui['space_center'][1] + gui['space_size'] 
        gui['bitmap'][top - 80:bottom + 20, left - 80:right + 20, :] = 0
        gui['bitmap'][top:bottom, left, :] = 128
        gui['bitmap'][top:bottom, center_h, :] = 128
        gui['bitmap'][top:bottom, right, :] = 128
        gui['bitmap'][top, left:right, :] = 128
        gui['bitmap'][center_v, left:right, :] = 128
        gui['bitmap'][bottom, left:right, :] = 128

        visible_pixels = int(len(gui['space_pixels_v']) * gui['path_pos'] / gui['path_length'])
        if visible_pixels > 0:
            gui['bitmap'][gui['space_pixels_v'][:visible_pixels], gui['space_pixels_h'][:visible_pixels], 1] = 255
        gui['update_space'] = False

    if gui['update_all'] or gui['update_wheels']:
        path_phase = gui['path_pos'] / gui['path_length'] * 2.0 * np.pi
        endpoint_v = draw_wheels('ampli_v', 'phase_v', 'wheel_panel_v_center', path_phase, 0.0, [255, 0, 0])
        if gui['show_1d']:
            endpoint_h = [gui['space_center'][0] - gui['space_size'], gui['wheel_panel_h_center'][1] + int(2.0 * gui['space_size'] * (gui['path_pos'] / gui['path_length'] - 0.5))]
        else:
            endpoint_h = draw_wheels('ampli_h', 'phase_h', 'wheel_panel_h_center', path_phase, np.pi / 2.0, [0, 0, 255])
        cv2.line(gui['bitmap'], (endpoint_v[1], endpoint_v[0]), (endpoint_h[1], endpoint_v[0]), [255, 0, 0], 1)
        cv2.line(gui['bitmap'], (endpoint_h[1], endpoint_h[0]), (endpoint_h[1], endpoint_v[0]), [0, 0, 255], 1)
    
    if gui['update_all'] or gui['update_waves']:
        path_phase = gui['path_pos'] / gui['path_length'] * 2.0 * np.pi
        draw_wave_map('ampli_v', 'phase_v', 'wave_map_v_center', path_phase, 0.0, [255, 0, 0], 0)
        if not gui['show_1d']:
            draw_wave_map('ampli_h', 'phase_h', 'wave_map_h_center', path_phase, np.pi / 2.0, [0, 0, 255], 1)
  
    gui['update_all'] = False
    cv2.imshow(gui['prog_name'], gui['bitmap'])

def reset_descriptors():
    gui['ampli_v'] = np.zeros(gui['max_num_freqs'])
    gui['ampli_v'][1:4] = np.array([0.8, 0.0, 0.0])
    gui['phase_v'] = np.zeros(gui['max_num_freqs'])
    gui['phase_v'][1:4] = np.array([0.0, 0.0, 0.0]) * np.pi / 180.0
    gui['ampli_h'] = np.zeros(gui['max_num_freqs'])
    gui['ampli_h'][1:4] = np.array([0.8, 0.0, 0.0])
    gui['phase_h'] = np.zeros(gui['max_num_freqs'])
    gui['phase_h'][1:4] = np.array([-90.0, 0.0, 0.0]) * np.pi / 180.0
    
def get_button_name(x, y):
    for (name, x1, y1, x2, y2) in gui['button_list']:
        if x >= x1 and x < x2 and y >= y1 and y < y2:
            return name
    return ''

def set_freq_bar(mouse_y):
    y1, _, y2, _ = gui['freq_bar']
    gui['num_freqs'] = max(2, min(gui['max_num_freqs'], int(gui['max_num_freqs'] * (1.0 - (mouse_y - y1) / (y2 - y1)) + 0.5)))
    compute_fourier_path()
    gui['update_all'] = True
    update_gui()
    return

def set_path_bar(mouse_x):
    _, x1, _, x2 = gui['path_bar']
    gui['path_pos'] = gui['path_length'] * (mouse_x - x1) / (x2 - x1 - 1) 
    gui['update_all'] = True
    update_gui()
    return

def on_mouse_click(event, x, y, flags, param): 
    if event == cv2.EVENT_LBUTTONUP:
        gui['mouse_state'] = ''
    elif event == cv2.EVENT_MOUSEMOVE:
        if gui['mouse_state'] == 'freq_bar':
            y1, _, y2, _ = gui['freq_bar']
            if y >= y1 and y < y2:
                set_freq_bar(y)
        elif gui['mouse_state'] == 'path_bar':
            _, x1, _, x2 = gui['path_bar']
            if x >= x1 and x < x2:
                set_path_bar(x)
        
    if event == cv2.EVENT_LBUTTONDOWN:
        button = get_button_name(x, y)
        if button == 'Load Image':
            load_image()
            gui['update_all'] = True
            update_gui()
            return
        
        if button == '1D Mode':
            gui['show_1d'] = True
            center = gui['wave_map_h_center']
            radius = gui['wave_map_radius'] + 20
            gui['bitmap'][center[0] - radius:center[0] + radius, center[1] - radius:center[1] + radius, :] = 0
            center = gui['wheel_panel_h_center']
            radius = gui['space_size'] + 20
            gui['bitmap'][center[0] - radius:center[0] + radius, center[1] - radius:center[1] + radius, :] = 0
            compute_fourier_path()
            gui['update_all'] = True
            update_gui()
            return
        
        if button == '2D Mode':
            gui['show_1d'] = False
            compute_fourier_path()
            gui['update_all'] = True
            update_gui()
            return
        
        if button == 'Reset':
            reset_descriptors()
            compute_fourier_path()
            gui['path_pos'] = 0
            gui['update_all'] = True
            update_gui()
            return

        y1, x1, y2, x2 = gui['freq_bar']
        if x >= x1 and x < x2 and y >= y1 and y < y2:
            set_freq_bar(y)
            gui['mouse_state'] = 'freq_bar'
            return
        
        y1, x1, y2, x2 = gui['path_bar']
        if x >= x1 and x < x2 and y >= y1 and y < y2:
            set_path_bar(x)
            gui['mouse_state'] = 'path_bar'
            return

def add_button(name):
    row = 30 + 60 * len(gui['button_list'])
    cv2.rectangle(gui['bitmap'], (20, row), (150, row + 40), [50, 50, 50], -1)
    cv2.rectangle(gui['bitmap'], (20, row), (150, row + 40), [255, 255, 255], 1)
    cv2.putText(gui['bitmap'], name, (30, row + 28), font, 0.6, (255, 255, 255), 1)
    gui['button_list'].append((name, 20, row, 150, row + 40))

def init_gui():
    gui['prog_name'] = 'Fourier Shape Descriptor Sandbox V0.1'
    gui['window_size'] = 950, 1800
    gui['bitmap'] = np.zeros(gui['window_size'] + (3,), dtype=np.uint8)
    
    gui['num_freqs'] = 2
    gui['max_num_freqs'] = 51
    gui['freq_bar'] = 50, 190, gui['window_size'][0] - 10, 220 
    
    gui['wheel_panel_v_center'] = gui['window_size'][0] * 3 // 4, gui['window_size'][1] // 2 + 100 
    gui['wheel_panel_h_center'] = gui['window_size'][0] // 4, gui['window_size'][1] * 5 // 6
    
    gui['space_center'] = gui['wheel_panel_v_center'][0], gui['wheel_panel_h_center'][1]
    gui['space_size'] = gui['wheel_panel_h_center'][0] - 30
    
    gui['path_bar'] = 50, gui['window_size'][1] // 6 - 40, 80, gui['wheel_panel_v_center'][1] - gui['window_size'][1] // 8 - 30
    gui['path_chart'] = 80, gui['window_size'][1] // 6 - 40, gui['window_size'][0] // 2 - 40, gui['wheel_panel_v_center'][1] - gui['window_size'][1] // 8 - 30
    gui['path_length'] = 2000
    gui['path_pos'] = 0 

    gui['wave_map_v_center'] = gui['space_center'][0], (gui['path_bar'][1] + gui['path_bar'][3]) // 2
    gui['wave_map_h_center'] = gui['wheel_panel_h_center'][0], gui['wheel_panel_v_center'][1]
    gui['wave_map_radius'] = gui['space_size']
    gui['wave_map_coords'] = np.zeros((2, gui['max_num_freqs'], 2))

    gui['space_pixels_v'] = []
    gui['space_pixels_h'] = []
    gui['chart_pixels_v'] = []
    gui['chart_pixels_h'] = []
    gui['chart_pixels_x'] = []
    
    gui['show_1d'] = False
    gui['update_all'] = True
    gui['update_freq_bar'] = False
    gui['update_path_chart'] = False
    gui['update_wheels'] = False
    gui['mouse_state'] = ''
    gui['mouse_param'] = 0
    gui['button_list'] = []
    
    add_button('Load Image')
    add_button('1D Mode')
    add_button('2D Mode')
    add_button('Reset')
    
    reset_descriptors()
    compute_fourier_path()
    update_gui()
        
init_gui()
cv2.moveWindow(gui['prog_name'], 30, 30)
cv2.setMouseCallback(gui['prog_name'], on_mouse_click) 
cv2.waitKey(0) 
cv2.destroyAllWindows()