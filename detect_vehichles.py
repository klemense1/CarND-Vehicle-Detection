import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
import time
import pickle

from scipy.ndimage.measurements import label
from train_classifier import extract_features

from moviepy.editor import VideoFileClip

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


def draw_boxes_on_image(img, bboxes, color=(0, 0, 255), thick=6):
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


def draw_boxes_cars(img, labels):
    """
    Draw boxes around all detected cars
    """

    for car_number in range(1, labels[1]+1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()

        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))

        # Draw the box on the image
        cv2.rectangle(img, bbox[0], bbox[1], (0, 0, 255), 6)

    return img


def process_image(img):

    xy_overlap = (0.75, 0.75)
    windows64 = generate_windows_list(img, x_start_stop=[450, 1200], y_start_stop=[350, 500],
                                      xy_window=(64, 64), xy_overlap=xy_overlap)

    if DEBUG_MODE:
        windows_img64 = draw_boxes_on_image(np.copy(img), [windows64[0]], color=(0, 0, 255), thick=4)
        windows_img64 = draw_boxes_on_image(windows_img64, windows64, color=(0, 0, 255), thick=1)
        mpimg.imsave('output_images/windows_img64', windows_img64)


    windows128 = generate_windows_list(img, x_start_stop=[350, 1200], y_start_stop=[325, 550],
                                      xy_window=(128, 128), xy_overlap=xy_overlap)

    if DEBUG_MODE:
        windows_img128 = draw_boxes_on_image(np.copy(img), [windows128[0]], color=(0, 0, 255), thick=4)
        windows_img128 = draw_boxes_on_image(windows_img128, windows128, color=(0, 0, 255), thick=1)
        mpimg.imsave('output_images/windows_img128', windows_img128)


    # windows192 = generate_windows_list(img, x_start_stop=[300, None], y_start_stop=[275, 650],
    #                                   xy_window=(192, 192), xy_overlap=xy_overlap)
    #
    # windows_img192 = draw_boxes_on_image(np.copy(img), [windows192[0]], color=(0, 0, 255), thick=4)
    # windows_img192 = draw_boxes_on_image(windows_img192, windows192, color=(0, 0, 255), thick=1)
    # mpimg.imsave('output_images/windows_img192', windows_img192)

    windows256 = generate_windows_list(img, x_start_stop=[200, None], y_start_stop=[300, 700],
                                      xy_window=(256, 256), xy_overlap=xy_overlap)

    if DEBUG_MODE:
        windows_img256 = draw_boxes_on_image(np.copy(img), [windows256[0]], color=(0, 0, 255), thick=4)
        windows_img256 = draw_boxes_on_image(windows_img256, windows256, color=(0, 0, 255), thick=1)
        mpimg.imsave('output_images/windows_img256', windows_img256)

    windows = windows64 + windows128 + windows256

    hot_windows = search_windows(img,
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

    pos_windows_img = draw_boxes_on_image(np.copy(img), hot_windows, color=(0, 0, 255), thick=6)
    mpimg.imsave('output_images/pos_windows_img', pos_windows_img)

    heatmap = np.zeros_like(img[:, :, 0]).astype(np.float)
    heatmap = add_heat(heatmap, hot_windows)

    heatmap_threshed = apply_threshold(heatmap, 2)
    labels = label(heatmap_threshed)

    if DEBUG_MODE:
        print(labels[1], 'cars found')

    if DEBUG_MODE:
        mpimg.imsave('output_images/labelling', labels[0], cmap='gray')

    detected_cars_img = draw_boxes_cars(np.copy(img), labels)

    if DEBUG_MODE:
        mpimg.imsave('output_images/detected_cars_img', detected_cars_img)

    return detected_cars_img


if __name__ == "__main__":

    PIPELINE_VIDEO = True
    DEBUG_MODE = not(PIPELINE_VIDEO)

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

        white_output = 'project_video_processed.mp4'
        clip1 = VideoFileClip("project_video.mp4")
        white_clip = clip1.fl_image(process_image)
        white_clip.write_videofile(white_output, audio=False)

    else:

        fname = 'test_images/test1.jpg'
        image = mpimg.imread(fname)
        image = image.astype(np.float32)/255

        img_detected_vehicle = process_image(image)
