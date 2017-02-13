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

DEBUG_MODE = True

class Car():
    def __init__(self, queuelength, confidence_threshold, initial_coordinates):

        self.position = [initial_coordinates[0], initial_coordinates[1]]
        self.recent_positions = [self.position]

        self.height = initial_coordinates[2]
        self.recent_heights = [self.height]

        self.width = initial_coordinates[3]
        self.recent_widths = [self.width]

        self.boundingbox = ((int(self.position[0]-self.height/2), int(self.position[1]-self.width/2)), (int(self.position[0]+self.height/2), int(self.position[1]+self.width/2)))
        self.validated = False
        self.tracked_frames = [1]
        self.queuelength = queuelength
        self.confidence_threshold = confidence_threshold
        self.car_id = random.randint(1, 1e9)
        self.to_delete = False
        self.buffer = 0

    def found_again(self, new_coordinates):
        if self.buffer == self.queuelength:
            self.tracked_frames.pop(0)
            self.recent_positions.pop(0)
            self.recent_heights.pop(0)
            self.recent_widths.pop(0)

        self.tracked_frames.append(1)

        self.recent_positions.append([new_coordinates[0], new_coordinates[1]])
        self.position = np.mean(self.recent_positions, axis=0).astype(int)

        self.recent_heights.append(new_coordinates[2])
        self.height = np.mean(self.recent_heights).astype(int)

        self.recent_widths.append(new_coordinates[3])
        self.width = np.mean(self.recent_widths).astype(int)

        self.boundingbox = ((int(self.position[0]-self.height/2), int(self.position[1]-self.width/2)), (int(self.position[0]+self.height/2), int(self.position[1]+self.width/2)))

        if sum(self.tracked_frames) >= self.confidence_threshold:
            self.validated = True
        else:
            self.validated = False

        self.buffer = len(self.tracked_frames)

    def not_found(self):

        if self.buffer == self.queuelength:
            self.tracked_frames.pop(0)

        self.tracked_frames.append(0)

        if np.mean(self.tracked_frames) < 0.3:
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

        self.coord_of_found_cars = []

        self.labels = None

        self.number_of_found_cars = 0

        self.binary_map = None

        self.list_of_cars = []

    def add_heatmap_to_buffer(self):
        """
        adds heatmap to recent_heatmaps list
        """
        if self.buffer == self.queuelength:
            self.recent_heatmaps.pop(0)

        self.recent_heatmaps.append(self.current_heatmap)
        self.buffer = len(self.recent_heatmaps)

    def set_vehicles_coordinates(self):
        """
        sets center, width and height from binary_map
        """
        coordinates = []

        for car_number in range(1, self.number_of_found_cars + 1):
            # Find pixels with each car_number label value
            nonzero = (self.binary_map == car_number).nonzero()

            # Identify x and y values of those pixels
            nonzeroy = np.array(nonzero[0])
            nonzerox = np.array(nonzero[1])

            x_min_max = [np.min(nonzerox), np.max(nonzerox)]
            y_min_max = [np.min(nonzeroy), np.max(nonzeroy)]

            coordinates.append([int(np.mean(x_min_max)), int(np.mean(y_min_max)), int(x_min_max[1]-x_min_max[0]), int(y_min_max[1]-y_min_max[0])])

        self.coord_of_found_cars = coordinates

    def get_closest_car(self, coord):
        dist = np.empty(shape=(len(self.list_of_cars), 2))
        for idx, car in enumerate(self.list_of_cars):
            dist[idx, 0] = car.car_id
            dist[idx, 1] = np.sqrt(np.square(coord[0]-car.position[0]) + np.square(coord[1]-car.position[1]))

        min_idx = dist[:, 1].argmin()
        closest_dist = dist[min_idx, 1]
        closest_car_id = dist[min_idx, 0]

        return closest_car_id, closest_dist

    def get_car(self, car_id):
        for car in self.list_of_cars:
            if car.car_id == car_id:
                return_car = car

        return return_car

    def check_for_overlap(self, coord):
        boundingboxes = [cars.boundingbox for cars in self.list_of_cars]
        is_overlapping = False

        for bbox in boundingboxes:
            within_x_coordinates = coord[0] > bbox[0][0] and coord[0] < bbox[1][0]
            within_y_coordinates = coord[1] > bbox[0][1] and coord[1] < bbox[1][1]
            if within_x_coordinates and within_y_coordinates:
                is_overlapping = True

        return is_overlapping

    def update_car_list(self):
        found_cars_matched = []
        coord_to_create_car_from = []

        if len(self.list_of_cars) > 0:
            # check for matches
            for idx, coordinates in enumerate(self.coord_of_found_cars):
                closest_car_id, closest_dist = self.get_closest_car(coordinates)

                print('Distance', closest_dist)

                if closest_dist < 50:
                    if closest_car_id not in found_cars_matched:

                        print('You have a match')

                        found_cars_matched.append(closest_car_id)
                        car = self.get_car(closest_car_id)
                        car.found_again(coordinates)
                else:
                    coord_to_create_car_from.append(coordinates)

            # mark car objects that were not found
            for car in self.list_of_cars:
                if car.car_id not in found_cars_matched:
                    car.not_found()

            # delete objects
            new_car_list = []
            for car in self.list_of_cars:
                if not car.to_delete:
                    new_car_list.append(car)

            self.list_of_cars = new_car_list
        else:
            coord_to_create_car_from = self.coord_of_found_cars

        # add new objects
        for coord in coord_to_create_car_from:
            ratio_height_width = coord[2]/coord[3]

            if ratio_height_width > 0.2 and ratio_height_width < 2:
                if not self.check_for_overlap(coord):
                    print('Creating new car instance at', coord)
                    new_car = Car(6, 6, coord)
                    self.list_of_cars.append(new_car)
                else:
                    print('Merged window with previous one')
            else:
                print('detected shape not plausible')


    def extract_vehicles(self):
        """
        Extracts boundingboxes and vehicle position from heatmap.
        """
        heatmap_threshed = apply_threshold(self.avg_heatmap, 7)

        labels = label(heatmap_threshed)
        self.binary_map = labels[0]
        self.number_of_found_cars = labels[1]

    def set_average_heatmap(self):
        """
        Averages heatmap. Uses higher weightings for more recent heatmaps
        """
        if self.buffer == 1:

            self.avg_heatmap = self.recent_heatmaps[0]
        else:
            weightings = np.linspace(1, self.buffer, self.buffer)
            weightings_norm = weightings/sum(weightings)

            heatmap_array = np.array(self.recent_heatmaps).T

            self.avg_heatmap = heatmap_array.dot(np.array(weightings)).T


    def update(self, heatmp):

        self.frame += 1

        self.current_heatmap = heatmp

        self.add_heatmap_to_buffer()

        self.set_average_heatmap()

        self.extract_vehicles()

        self.set_vehicles_coordinates()

        print('###\nNew frame\n###')
        print('Extracting_vehicles()...')
        print('number_of_found_cars', self.number_of_found_cars)
        print('coord_of_found_cars', self.coord_of_found_cars)

        self.update_car_list()

        for car in self.list_of_cars:
            print('\ncar id:', car.car_id)
            print('   position', car.position)
            print('   validated', car.validated)
            print('   tracked_frames', car.tracked_frames)


def generate_windows_list(img,
                          x_start_stop,
                          y_start_stop,
                          xy_window,
                          xy_overlap):
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


def find_cars(img,
              clf,
              scaler,
              color_space,
              spatial_size,
              hist_bins,
              scale,
              cells_per_step,
              x_start_stop,
              y_start_stop,
              orient,
              pix_per_cell,
              cell_per_block):
    """
    cells_per_step ... instead of overlap, define how many cells to step
    """
    draw_img = np.copy(img)

    heatmap = np.zeros_like(img[:, :, 0])

    img = img.astype(np.float32)/255

    img_to_search = img[y_start_stop[0]:y_start_stop[1], x_start_stop[0]:x_start_stop[1], :]

    # color transformed image
    ctrans_to_search = change_color_space(img_to_search, colorspace=color_space)

    if scale != 1:
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
                heatmap[ytop_draw+y_start_stop[0]:ytop_draw+win_draw+y_start_stop[0], xbox_left+x_start_stop[0]:xbox_left+win_draw+x_start_stop[0]] += 1

    return draw_img, heatmap


def search_windows(img,
                   windows_list,
                   clf,
                   scaler,
                   color_space='RGB',
                   spatial_size=(32, 32),
                   hist_bins=32,
                   orient=9,
                   pix_per_cell=8,
                   cell_per_block=2,
                   hog_channel=0,
                   use_spatial_feat=True,
                   use_hist_feat=True,
                   use_hog_feat=True):
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
                                         cells_per_step=3,
                                         x_start_stop=[475, None],
                                         y_start_stop=[400, 656],
                                         orient=orientation,
                                         pix_per_cell=num_pix_per_cell,
                                         cell_per_block=num_cell_per_block)

    if DEBUG_MODE:
        mpimg.imsave('output_images/pos_windowsALL_img64' + annotation, pos_windows_img64)
        mpimg.imsave('output_images/pos_windowsALL_img128' + annotation, pos_windows_img128)

    heatmap_combined = heatmap64 + heatmap128

    tracer.update(heatmap_combined)


    if DEBUG_MODE:
        mpimg.imsave('output_images/heatmap' + annotation, heatmap_combined, cmap='hot')
        mpimg.imsave('output_images/heatmap_current' + annotation, tracer.current_heatmap, cmap='hot')
        mpimg.imsave('output_images/heatmap_average' + annotation, tracer.avg_heatmap, cmap='hot')

        print(tracer.number_of_found_cars, 'cars found')

        mpimg.imsave('output_images/labelling' + annotation, tracer.binary_map, cmap='hot')

    # detected_cars_img = draw_boxes_cars(np.copy(img), tracer)


    detected_cars_img = np.copy(img)
    for car in tracer.list_of_cars:
        if car.validated:
            cv2.circle(detected_cars_img, tuple(car.position), 7, 255, -1)
            cv2.rectangle(detected_cars_img, car.boundingbox[0], car.boundingbox[1], (0, 0, 255), 6)

    font = cv2.FONT_HERSHEY_SIMPLEX

    str_coeff1 = '{}'.format(str(tracer.number_of_found_cars) + ' cars found')
    cv2.putText(detected_cars_img, str_coeff1, (130,250), font, 1, (1,0,0), 2, cv2.LINE_AA)

    if DEBUG_MODE:
        mpimg.imsave('output_images/detected_cars_img' + annotation, detected_cars_img)

    return detected_cars_img, heatmap_combined, pos_windows_img64, pos_windows_img128


def process_image(img):

    global tracer

    img_processed, __, __, __ = detect(img)

    return img_processed

if __name__ == "__main__":

    PIPELINE_VIDEO = True

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
        # clip1 = VideoFileClip("project_video.mp4")
        white_clip = clip1.fl_image(process_image)
        white_clip.write_videofile(white_output, audio=False)

    else:
        tracer = Tracing_algorithm(1)
        filelist = ['test_images/test1.jpg', 'test_images/test2.jpg', 'test_images/test3.jpg', 'test_images/test4.jpg', 'test_images/test5.jpg']
        plt.figure(figsize=(12,16))
        for i, fname in enumerate(filelist):
        # fname = 'test_images/test4.jpg'
            image = mpimg.imread(fname)

            img_detected_vehicle, heatmap, pos_windows64, pos_windows128 = detect(image)
            plt.subplot(len(filelist), 3, i*3+1)
            plt.imshow(image)
            plt.title('Image')
            plt.subplot(len(filelist), 3, i*3+2)
            plt.imshow(pos_windows64)
            plt.title('Positive Windows 64x64')
            plt.subplot(len(filelist), 3, i*3+3)
            plt.imshow(pos_windows128)
            plt.title('Positive Windows 128x128')
            plt.savefig('output_images/pipeline', dpi=150)
