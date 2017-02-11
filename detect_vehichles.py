import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
import time
import pickle
import random

from scipy.ndimage.measurements import label
from train_classifier import extract_features

from moviepy.editor import VideoFileClip

class Car():
    def __init__(self, confidence_threshold):
        self.position = []
        self.boundingbox = []
        self.validated = False
        self.tracked_frames = 0
        self.confidence_threshold = confidence_threshold
        self.car_id = random.randint(1, 1e9)
        self.in_doubt = False

    def update(self):
        self.tracked_frames += 1
        if self.confidence_threshold > confidence_threshold:
            self.validated = True

class Tracing_algorithm():
    def __init__(self, queuelength):

        self.queuelength = queuelength

        self.current_heatmap = None

        self.recent_heatmaps = []

        self.avg_heatmap = None

        self.frame = 0

        self.buffer = 0

        self.recent_center_of_vehicles = []

        self.current_center_of_vehicles = []

        self.labels = None

        self.recent_vehicles_detected = [0]

        self.current_vehicles_detected = 0

        self.binary_map = None

    def add_data(self):

        if self.buffer==self.queuelength:
            self.recent_heatmaps.pop(0)

        self.recent_heatmaps.append(self.current_heatmap)
        self.buffer = len(self.recent_heatmaps)

        # print('Data added, buffer=', self.buffer)

    def set_vehicles_position(self):

        positions = []
        for car_number in range(1, self.current_vehicles_detected + 1):
            # Find pixels with each car_number label value
            nonzero = (self.binary_map == car_number).nonzero()

            # Identify x and y values of those pixels
            nonzeroy = np.array(nonzero[0])
            nonzerox = np.array(nonzero[1])

            x = int(np.mean([np.min(nonzerox), np.max(nonzerox)]))
            y = int(np.mean([np.min(nonzeroy), np.max(nonzeroy)]))
            positions.append([x, y])

        self.current_center_of_vehicles = positions

        # if self.buffer>self.queuelength:
        #     self.recent_center_of_vehicles.pop(0)
        self.recent_center_of_vehicles.append(self.current_center_of_vehicles)

    def check_data(self):
        plausible=False
        if self.current_vehicles_detected < self.recent_vehicles_detected[-2]:
            print('detected less vehicles than before')
            for center in self.current_center_of_vehicles:
                for last_center in self.recent_center_of_vehicles[-2]:
                    distance = np.sqrt(np.square(center[0]-last_center[0]) + np.square(center[1]-last_center[1]))
                    print('Distance', distance)
                    if distance<20:
                        plausible=True
                        print('Data is plausible')
        else:
            plausible = True
        return plausible

    def remove_data_from_buffer(self):
        print('Data was not plausible, dropping data')
        self.recent_center_of_vehicles.pop()
        self.recent_vehicles_detected.pop()

        self.current_vehicles_detected = self.recent_vehicles_detected[-1]
        self.current_center_of_vehicles = self.recent_center_of_vehicles[-1]

    def extract_vehicles(self):

        heatmap_threshed = apply_threshold(self.avg_heatmap, 2)

        labels = label(heatmap_threshed)
        self.binary_map = labels[0]
        self.current_vehicles_detected = labels[1]
        #
        # if self.buffer > self.queuelength:
        #     self.recent_vehicles_detected.pop(0)
        self.recent_vehicles_detected.append(self.current_vehicles_detected)

        self.set_vehicles_position()
        print('Extracting_vehicles()...')
        print('current_vehicles_detected', self.current_vehicles_detected)
        print('recent_vehicles_detected', self.recent_vehicles_detected)
        print('current_center_of_vehicles', self.current_center_of_vehicles)
        print('recent_center_of_vehicles', self.recent_center_of_vehicles)
    def set_average_heatmap(self):

        if self.buffer == 1:
            # prev_heatmap * heatmap_factor + heatmap_image * (1 - heatmap_factor) #
            self.avg_heatmap = self.recent_heatmaps[0]
        else:
            weightings = np.linspace(1, self.buffer, self.buffer)
            weightings_norm = weightings/sum(weightings)
            # print('Weightings:', weightings_norm)
            img_array = np.array(self.recent_heatmaps).T
            # print('img_array.shape', img_array.shape)
            self.avg_heatmap = img_array.dot(np.array(weightings)).T
            # print('self.avg_heatmap.shape', self.avg_heatmap.shape)

    def update(self, heatmp):
        print('###\nNew frame\n###')
        self.current_heatmap = heatmp

        self.add_data()

        self.set_average_heatmap()

        self.frame += 1

        self.extract_vehicles()

        if self.buffer>2:
            if self.check_data() is False:

                self.remove_data_from_buffer()
                #self.set_vehicles_position()

def generate_windows_list(img, x_start_stop=[None, None], y_start_stop=[None, None],
                          xy_window=(64, 64), xy_overlap=(0.5, 0.5)):
    """
    takes an image and generates list of windows to be searched in
    """
    if x_start_stop[0] is None:
        x_start_stop[0] = 0
    if x_start_stop[1] is None:
        x_start_stop[1] = img.shape[1]
    if y_start_stop[0] is None:
        y_start_stop[0] = 0
    if y_start_stop[1] is None:
        y_start_stop[1] = img.shape[0]

    # Compute the span of the region to be searched
    xspan = x_start_stop[1] - x_start_stop[0]
    yspan = y_start_stop[1] - y_start_stop[0]

    # Compute the number of pixels per step in x/y
    nx_pix_per_step = np.int(xy_window[0]*(1 - xy_overlap[0]))
    ny_pix_per_step = np.int(xy_window[1]*(1 - xy_overlap[1]))

    # Compute the number of windows in x/y
    nx_windows = np.int(xspan/nx_pix_per_step) - 1
    ny_windows = np.int(yspan/ny_pix_per_step) - 1

    window_list = []

    # Loop through finding x and y window positions
    for ys in range(ny_windows):
        for xs in range(nx_windows):

            startx = xs*nx_pix_per_step + x_start_stop[0]
            endx = startx + xy_window[0]
            starty = ys*ny_pix_per_step + y_start_stop[0]
            endy = starty + xy_window[1]

            window_list.append(((startx, starty), (endx, endy)))

    return window_list


def search_windows(img, windows_list, clf, scaler, color_space='RGB',
                   spatial_size=(32, 32), hist_bins=32,
                   hist_range=(0, 256), orient=9,
                   pix_per_cell=8, cell_per_block=2,
                   hog_channel=0, use_spatial_feat=True,
                   use_hist_feat=True, use_hog_feat=True):
    """
    For each window in windows_list, features are extracted and fed into
    classifier. If classified as a car, it is kept and finally returned.
    """
    windows_positive = []

    for window in windows_list:

        test_img_part = cv2.resize(img[window[0][1]:window[1][1], window[0][0]:window[1][0]], (64, 64))

        features = extract_features(test_img_part,
                                    color_space=color_space,
                                    spatial_size=spatial_size,
                                    hist_bins=hist_bins,
                                    orient=orient,
                                    pix_per_cell=pix_per_cell,
                                    cell_per_block=cell_per_block,
                                    hog_channel=hog_channel,
                                    use_spatial_feat=use_spatial_feat,
                                    use_hist_feat=use_hist_feat,
                                    use_hog_feat=use_hog_feat)

        test_features = scaler.transform(np.array(features).reshape(1, -1))

        prediction = clf.predict(test_features)

        if prediction == 1:
            windows_positive.append(window)

    return windows_positive


def draw_boxes_on_image(img, bboxes, color=(0, 0, 1), thick=6):
    """
    draw rectangles specified in bboxes (bounding boxes) to image
    """
    imcopy = np.copy(img)

    for bbox in bboxes:
        cv2.rectangle(imcopy, bbox[0], bbox[1], color, thick)

    return imcopy


def add_heat(heatmap, bbox_list):
    """
    Iterate through list of bboxes and add 1 for all pixels inside each bbox

    """
    for box in bbox_list:

        # Assuming each "box" takes the form ((x1, y1), (x2, y2))
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

    return heatmap


def apply_threshold(heatmap, threshold):
    """
    Zero out pixels below the threshold
    """
    heatmap[heatmap <= threshold] = 0

    return heatmap


def draw_boxes_cars(img, vehicles_instance):
    """
    Draw boxes around all detected cars
    """

    for car_number in range(1, vehicles_instance.current_vehicles_detected+1):
        # Find pixels with each car_number label value
        nonzero = (vehicles_instance.binary_map == car_number).nonzero()

        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))

        # Draw the box on the image
        cv2.rectangle(img, bbox[0], bbox[1], (0, 0, 255), 6)

    return img


def detect(img):

    global tracer

    annotation = '_frame' + str(tracer.frame)
    xy_overlap = (0.8, 0.8)
    windows64 = generate_windows_list(img, x_start_stop=[475, 1200], y_start_stop=[360, 480],
                                      xy_window=(64, 64), xy_overlap=xy_overlap)

    if DEBUG_MODE:
        windows_img64 = draw_boxes_on_image(np.copy(img.astype(np.float32)/255), [windows64[0]], color=(0, 0, 1), thick=4)
        windows_img64 = draw_boxes_on_image(windows_img64, windows64, color=(0, 0, 1), thick=1)
        mpimg.imsave('output_images/windows_img64' + annotation, windows_img64)


    windows128 = generate_windows_list(img, x_start_stop=[400, 1200], y_start_stop=[340, 530],
                                      xy_window=(128, 128), xy_overlap=xy_overlap)

    if DEBUG_MODE:
        windows_img128 = draw_boxes_on_image(np.copy(img.astype(np.float32)/255), [windows128[0]], color=(0, 0, 1), thick=4)
        windows_img128 = draw_boxes_on_image(windows_img128, windows128, color=(0, 0, 1), thick=1)
        mpimg.imsave('output_images/windows_img128' + annotation, windows_img128)


    # windows192 = generate_windows_list(img, x_start_stop=[300, None], y_start_stop=[275, 650],
    #                                   xy_window=(192, 192), xy_overlap=xy_overlap)
    #
    # windows_img192 = draw_boxes_on_image(np.copy(img), [windows192[0]], color=(0, 0, 255), thick=4)
    # windows_img192 = draw_boxes_on_image(windows_img192, windows192, color=(0, 0, 255), thick=1)
    # mpimg.imsave('output_images/windows_img192', windows_img192)

    windows256 = generate_windows_list(img, x_start_stop=[340, None], y_start_stop=[300, 550],
                                      xy_window=(256, 256), xy_overlap=xy_overlap)

    if DEBUG_MODE:
        windows_img256 = draw_boxes_on_image(np.copy(img.astype(np.float32)/255), [windows256[0]], color=(0, 0, 1), thick=4)
        windows_img256 = draw_boxes_on_image(windows_img256, windows256, color=(0, 0, 1), thick=1)
        mpimg.imsave('output_images/windows_img256' + annotation, windows_img256)

    windows = windows64 + windows128 + windows256

    hot_windows = search_windows(img.astype(np.float32)/255,
                                 windows,
                                 svc,
                                 X_scaler,
                                 color_space=color_space,
                                 spatial_size=spatial_size,
                                 hist_bins=histogram_bins,
                                 orient=orientation,
                                 pix_per_cell=num_pix_per_cell,
                                 cell_per_block=num_cell_per_block,
                                 hog_channel=hog_channel_select,
                                 use_spatial_feat=use_spatial_features,
                                 use_hist_feat=use_hist_features,
                                 use_hog_feat=use_hog_features)
    if DEBUG_MODE:
        pos_windows_img = draw_boxes_on_image(np.copy(img), hot_windows, color=(0, 0, 1), thick=6)
        mpimg.imsave('output_images/pos_windows_img' + annotation, pos_windows_img)

    heatmap = np.zeros_like(img[:, :, 0]).astype(np.float)
    heatmap = add_heat(heatmap, hot_windows)

    tracer.update(heatmap)


    if DEBUG_MODE:
        mpimg.imsave('output_images/heatmap' + annotation, heatmap, cmap='gray')
        mpimg.imsave('output_images/heatmap_current' + annotation, tracer.current_heatmap, cmap='gray')
        mpimg.imsave('output_images/heatmap_average' + annotation, tracer.avg_heatmap, cmap='gray')

        print(tracer.current_vehicles_detected, 'cars found')

        mpimg.imsave('output_images/labelling' + annotation, tracer.binary_map, cmap='gray')

    # detected_cars_img = draw_boxes_cars(np.copy(img), tracer)


    detected_cars_img = np.copy(img)
    for center in tracer.current_center_of_vehicles:
        print('Center', center)
        cv2.circle(detected_cars_img, tuple(center), 7,255,-1)

    font = cv2.FONT_HERSHEY_SIMPLEX

    str_coeff1 = '{}'.format(str(tracer.current_vehicles_detected) + ' cars found')
    cv2.putText(detected_cars_img, str_coeff1, (130,250), font, 1, (1,0,0), 2, cv2.LINE_AA)

    if DEBUG_MODE:
        mpimg.imsave('output_images/detected_cars_img' + annotation, detected_cars_img)

    return detected_cars_img


def process_image(img):

    global tracer

    img_processed = detect(img)

    return img_processed

if __name__ == "__main__":

    PIPELINE_VIDEO = True
    DEBUG_MODE = True#not(PIPELINE_VIDEO)

    data_file = 'ClassifierData.p'
    with open(data_file, mode='rb') as f:
        data = pickle.load(f)

    svc = data['svc']
    X_scaler = data['X_scaler']
    color_space = data['color_space']
    spatial_size = data['spatial_size']
    X_scaler = data['X_scaler']
    histogram_bins = data['histogram_bins']
    orientation = data['orientation']
    num_pix_per_cell = data['num_pix_per_cell']
    num_cell_per_block = data['num_cell_per_block']
    hog_channel_select = data['hog_channel_select']
    use_spatial_features = data['use_spatial_features']
    use_hist_features = data['use_hist_features']
    use_hog_features = data['use_hog_features']

    if PIPELINE_VIDEO:
        tracer = Tracing_algorithm(3)
        white_output = 'project_video_processed.mp4'
        clip1 = VideoFileClip("project_video.mp4").subclip(39,43)
        white_clip = clip1.fl_image(process_image)
        white_clip.write_videofile(white_output, audio=False)

    else:
        tracer = Tracing_algorithm(1)
        fname = 'test_images/test1.jpg'
        image = mpimg.imread(fname)
        # image = image.astype(np.float32)/255

        img_detected_vehicle = process_image(image)
