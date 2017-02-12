import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
import time
import pickle
import random

from scipy.ndimage.measurements import label
from train_classifier import change_color_space, get_bin_spatial, get_color_hist, extract_features, get_hog_features

from moviepy.editor import VideoFileClip

class Car():
    def __init__(self, queuelength, confidence_threshold, initial_position):
        self.recent_positions = [initial_position]
        self.position = initial_position
        self.boundingbox = []
        self.validated = False
        self.tracked_frames = [1]
        self.queuelength = queuelength
        self.confidence_threshold = confidence_threshold
        self.car_id = random.randint(1, 1e9)
        self.to_delete = False
        self.buffer = 0

    def found_again(self, new_position):
        if self.buffer == self.queuelength:
            self.tracked_frames.pop(0)
            self.recent_positions.pop(0)
        self.tracked_frames.append(1)
        self.recent_positions.append(new_position)
        self.position = np.mean(self.recent_positions, axis=0).astype(int)

        if sum(self.tracked_frames) > self.confidence_threshold:
            self.validated = True
        else:
            self.validated = False

        self.buffer = len(self.tracked_frames)

    def not_found(self):

        if self.buffer == self.queuelength:
            self.tracked_frames.pop(0)

        self.tracked_frames.append(0)

        if sum(self.tracked_frames) < 0.2:
            self.to_delete = True

        self.buffer = len(self.tracked_frames)

class Tracing_algorithm():
    def __init__(self, queuelength):

        self.queuelength = queuelength

        self.current_heatmap = None

        self.recent_heatmaps = []

        self.avg_heatmap = None

        self.frame = 0

        self.buffer = 0

        self.center_of_found_cars = []

        self.labels = None

        self.number_of_found_cars = 0

        self.binary_map = None

        self.list_of_cars = []

    def add_heatmap_to_buffer(self):

        if self.buffer == self.queuelength:
            self.recent_heatmaps.pop(0)

        self.recent_heatmaps.append(self.current_heatmap)
        self.buffer = len(self.recent_heatmaps)

    def set_vehicles_positions(self):

        positions = []
        for car_number in range(1, self.number_of_found_cars + 1):
            # Find pixels with each car_number label value
            nonzero = (self.binary_map == car_number).nonzero()

            # Identify x and y values of those pixels
            nonzeroy = np.array(nonzero[0])
            nonzerox = np.array(nonzero[1])

            x = int(np.mean([np.min(nonzerox), np.max(nonzerox)]))
            y = int(np.mean([np.min(nonzeroy), np.max(nonzeroy)]))
            positions.append([x, y])

        self.center_of_found_cars = positions

    def update_car_list(self):
        found_cars_mathed = [False]*self.number_of_found_cars
        new_car_list = []

        for car in self.list_of_cars:
            car_found = False
            for idx_center, center in enumerate(self.center_of_found_cars):
                distance = np.sqrt(np.square(center[0]-car.position[0]) + np.square(center[1]-car.position[1]))
                print('Distance', distance)
                if distance < 100:
                    print('You have a match')
                    found_cars_mathed[idx_center] = True
                    car_found = True

                    car.found_again(center)
                    new_car_list.append(car)
            if not car_found:
                car.not_found()
                if not car.to_delete:
                    new_car_list.append(car)

        print('found_cars_mathed', found_cars_mathed)
        print('self.center_of_found_cars', self.center_of_found_cars)

        self.list_of_cars = new_car_list

        for idx_center, center in enumerate(self.center_of_found_cars):
            if found_cars_mathed[idx_center] == False:
                print('Creating new car instance at', center)
                new_car = Car(5, 3, center)
                self.list_of_cars.append(new_car)

    def extract_vehicles(self):

        heatmap_threshed = apply_threshold(self.avg_heatmap, 6)

        labels = label(heatmap_threshed)
        self.binary_map = labels[0]
        self.number_of_found_cars = labels[1]

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

        self.frame += 1

        self.current_heatmap = heatmp

        self.add_heatmap_to_buffer()

        self.set_average_heatmap()

        self.extract_vehicles()

        self.set_vehicles_positions()

        print('Extracting_vehicles()...')
        print('number_of_found_cars', self.number_of_found_cars)
        print('center_of_found_cars', self.center_of_found_cars)

        self.update_car_list()

        for car in self.list_of_cars:
            print('\ncar id:', car.car_id)
            print('   position', car.position)
            print('   validated', car.validated)
            print('   tracked_frames', car.tracked_frames)


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


def find_cars(img, clf, scaler, color_space='RGB',
                   spatial_size=(32, 32), hist_bins=32, scale=1, cells_per_step=2, x_start_stop=[None, None], y_start_stop=[None, None], orient=9,
            pix_per_cell=8, cell_per_block=2):
    """
    cells_per_step ... instead of overlap, define how many cells to step
    """
    draw_img = np.copy(img)

    heatmap = np.zeros_like(img[:, :, 0])

    img = img.astype(np.float32)/255

    img_to_search = img[y_start_stop[0]:y_start_stop[1], x_start_stop[0]:x_start_stop[1], :]

    # color transformed image
    ctrans_to_search = change_color_space(img_to_search, colorspace=color_space)

    if scale != 1: # use this as window size
        imshape = ctrans_to_search.shape
        ctrans_to_search = cv2.resize(ctrans_to_search, (np.int(imshape[1]/scale), np.int(imshape[0]/scale)))

    ch1 = ctrans_to_search[:, :, 0]
    ch2 = ctrans_to_search[:, :, 1]
    ch3 = ctrans_to_search[:, :, 2]

    nxblocks = (ch1.shape[1] // pix_per_cell) - 1 # number of hog cells
    nyblocks = (ch1.shape[0] // pix_per_cell) - 1
    nfeat_per_block = orient*cell_per_block**2
    window = 64
    nblocks_per_window = (window // pix_per_cell) - 1

    nxsteps = (nxblocks - nblocks_per_window) // cells_per_step
    nysteps = (nyblocks - nblocks_per_window) // cells_per_step

    # compute individual channel HOG features for the intire image
    hog1 = get_hog_features(ch1, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog2 = get_hog_features(ch1, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog3 = get_hog_features(ch1, orient, pix_per_cell, cell_per_block, feature_vec=False)

    for xb in range(nxsteps):
        for yb in range(nysteps):
            ypos = yb*cells_per_step
            xpos = xb*cells_per_step
            # extract hog features for this patch
            hog_feat1 = hog1[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel()
            hog_feat2 = hog2[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel()
            hog_feat3 = hog3[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel()
            hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))

            xleft = xpos*pix_per_cell
            ytop = ypos*pix_per_cell

            # extract the image path
            subimg = cv2.resize(ctrans_to_search[ytop:ytop+window, xleft:xleft+window], (64,64))

            # get color features
            spatial_features = get_bin_spatial(subimg, size=spatial_size)
            hist_features = get_color_hist(subimg, nbins=hist_bins)

            # scale features and make prediction
            test_features = scaler.transform(np.hstack((spatial_features, hist_features, hog_features)))

            test_prediction = clf.predict(test_features)

            if test_prediction == 1:
                xbox_left = np.int(xleft*scale)
                ytop_draw = np.int(ytop*scale)
                win_draw = np.int(window*scale)
                cv2.rectangle(draw_img, (xbox_left+x_start_stop[0], ytop_draw+y_start_stop[0]), (xbox_left+win_draw+x_start_stop[0], ytop_draw+win_draw+y_start_stop[0]), (0,0,255), 6)
                # img_boxes.append(((xbox_left, ytop_draw+y_start_stop[0]), (xbox_left+win_draw, ytop_draw+win_draw+y_start_stop[0])))
                heatmap[ytop_draw+y_start_stop[0]:ytop_draw+win_draw+y_start_stop[0], xbox_left+x_start_stop[0]:xbox_left+win_draw+x_start_stop[0]] += 1

    return draw_img, heatmap


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

    for car_number in range(1, vehicles_instance.number_of_found_cars+1):
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
    # xy_overlap = (0.8, 0.8)
    # windows64 = generate_windows_list(img, x_start_stop=[475, 1200], y_start_stop=[360, 480],
    #                                   xy_window=(64, 64), xy_overlap=xy_overlap)
    #
    # if DEBUG_MODE:
    #     windows_img64 = draw_boxes_on_image(img.astype(np.float32)/255, [windows64[0]], color=(0, 0, 1), thick=4)
    #     windows_img64 = draw_boxes_on_image(windows_img64, windows64, color=(0, 0, 1), thick=1)
    #     mpimg.imsave('output_images/windows_img64' + annotation, windows_img64)
    #
    #
    # windows128 = generate_windows_list(img, x_start_stop=[400, 1200], y_start_stop=[340, 530],
    #                                   xy_window=(128, 128), xy_overlap=xy_overlap)
    #
    # if DEBUG_MODE:
    #     windows_img128 = draw_boxes_on_image(img.astype(np.float32)/255, [windows128[0]], color=(0, 0, 1), thick=4)
    #     windows_img128 = draw_boxes_on_image(windows_img128, windows128, color=(0, 0, 1), thick=1)
    #     mpimg.imsave('output_images/windows_img128' + annotation, windows_img128)
    #
    #
    # # windows192 = generate_windows_list(img, x_start_stop=[300, None], y_start_stop=[275, 650],
    # #                                   xy_window=(192, 192), xy_overlap=xy_overlap)
    # #
    # # windows_img192 = draw_boxes_on_image(np.copy(img), [windows192[0]], color=(0, 0, 255), thick=4)
    # # windows_img192 = draw_boxes_on_image(windows_img192, windows192, color=(0, 0, 255), thick=1)
    # # mpimg.imsave('output_images/windows_img192', windows_img192)
    #
    # windows256 = generate_windows_list(img, x_start_stop=[340, None], y_start_stop=[300, 550],
    #                                   xy_window=(256, 256), xy_overlap=xy_overlap)
    #
    # if DEBUG_MODE:
    #     windows_img256 = draw_boxes_on_image(img.astype(np.float32)/255, [windows256[0]], color=(0, 0, 1), thick=4)
    #     windows_img256 = draw_boxes_on_image(windows_img256, windows256, color=(0, 0, 1), thick=1)
    #     mpimg.imsave('output_images/windows_img256' + annotation, windows_img256)
    #
    # windows = windows64 + windows128 + windows256
    #
    # hot_windows = search_windows(img.astype(np.float32)/255,
    #                              windows,
    #                              svc,
    #                              X_scaler,
    #                              color_space=color_space,
    #                              spatial_size=spatial_size,
    #                              hist_bins=histogram_bins,
    #                              orient=orientation,
    #                              pix_per_cell=num_pix_per_cell,
    #                              cell_per_block=num_cell_per_block,
    #                              hog_channel=hog_channel_select,
    #                              use_spatial_feat=use_spatial_features,
    #                              use_hist_feat=use_hist_features,
    #                              use_hog_feat=use_hog_features)
    # if DEBUG_MODE:
    #     pos_windows_img = draw_boxes_on_image(np.copy(img), hot_windows, color=(0, 0, 1), thick=6)
    #     mpimg.imsave('output_images/pos_windows_img' + annotation, pos_windows_img)
    #
    # heatmap = np.zeros_like(img[:, :, 0]).astype(np.float)
    # heatmap = add_heat(heatmap, hot_windows)

    pos_windows_img64, heatmap64 = find_cars(img,
                                         svc,
                                         X_scaler,
                                         color_space=color_space,
                                         spatial_size=spatial_size,
                                         hist_bins=histogram_bins,
                                         scale=1,
                                         cells_per_step=2,
                                         x_start_stop=[475, None],
                                         y_start_stop=[400, 656],
                                         orient=orientation,
                                         pix_per_cell=num_pix_per_cell,
                                         cell_per_block=num_cell_per_block)

    pos_windows_img128, heatmap128 = find_cars(img,
                                         svc,
                                         X_scaler,
                                         color_space=color_space,
                                         spatial_size=spatial_size,
                                         hist_bins=histogram_bins,
                                         scale=1.5,
                                         cells_per_step=2,
                                         x_start_stop=[475, None],
                                         y_start_stop=[400, 656],
                                         orient=orientation,
                                         pix_per_cell=num_pix_per_cell,
                                         cell_per_block=num_cell_per_block)

    if DEBUG_MODE:
        mpimg.imsave('output_images/pos_windows_img64' + annotation, pos_windows_img64)
        mpimg.imsave('output_images/pos_windows_img128' + annotation, pos_windows_img128)

    heatmap = heatmap64 + heatmap128
    tracer.update(heatmap)


    if DEBUG_MODE:
        mpimg.imsave('output_images/heatmap' + annotation, heatmap, cmap='gray')
        mpimg.imsave('output_images/heatmap_current' + annotation, tracer.current_heatmap, cmap='gray')
        mpimg.imsave('output_images/heatmap_average' + annotation, tracer.avg_heatmap, cmap='gray')

        print(tracer.number_of_found_cars, 'cars found')

        mpimg.imsave('output_images/labelling' + annotation, tracer.binary_map, cmap='gray')

    # detected_cars_img = draw_boxes_cars(np.copy(img), tracer)


    detected_cars_img = np.copy(img)
    for car in tracer.list_of_cars:
        if car.validated:
            print('Position', car.position)
            cv2.circle(detected_cars_img, tuple(car.position), 7,255,-1)

    font = cv2.FONT_HERSHEY_SIMPLEX

    str_coeff1 = '{}'.format(str(tracer.number_of_found_cars) + ' cars found')
    cv2.putText(detected_cars_img, str_coeff1, (130,250), font, 1, (1,0,0), 2, cv2.LINE_AA)

    if DEBUG_MODE:
        mpimg.imsave('output_images/detected_cars_img' + annotation, detected_cars_img)

    return detected_cars_img


def process_image(img):

    global tracer

    img_processed = detect(img)

    return img_processed

if __name__ == "__main__":

    PIPELINE_VIDEO = False
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
        clip1 = VideoFileClip("test_video.mp4")
        # clip1 = VideoFileClip("project_video.mp4").subclip(0,1)
        white_clip = clip1.fl_image(process_image)
        white_clip.write_videofile(white_output, audio=False)

    else:
        tracer = Tracing_algorithm(1)
        fname = 'test_images/test2.jpg'
        image = mpimg.imread(fname)
        # image = image.astype(np.float32)/255

        img_detected_vehicle = process_image(image)
