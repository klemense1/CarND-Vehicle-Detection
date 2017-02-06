import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob
import os
import time
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog
from sklearn.model_selection import train_test_split
from scipy.ndimage.measurements import label


def get_hog_features(img, orient, pix_per_cell, cell_per_block, visualize_hog=False, feature_vec=True):
    """
    computes a histogram of oriented gradients (HOG), which is robust to variations in shape
    returns HOG features and visualization (optional)
    """

    pixels_per_cell = (pix_per_cell, pix_per_cell)
    cells_per_block = (cell_per_block, cell_per_block)

    if visualize_hog is True:
        hog_features, hog_image = hog(img,
                                      orientations=orient,
                                      pixels_per_cell=pixels_per_cell,
                                      cells_per_block=cells_per_block,
                                      transform_sqrt=True,
                                      visualise=visualize_hog,
                                      feature_vector=feature_vec)

        return features, hog_image

    else:
        features = hog(img,
                       orientations=orient,
                       pixels_per_cell=cells_per_block,
                       cells_per_block=cells_per_block,
                       transform_sqrt=True,
                       visualise=visualize_hog,
                       feature_vector=feature_vec)

        return features


def bin_spatial(img, size=(32, 32)):
    """
    Defines a function to compute binned color features
    """
    spatial_features = cv2.resize(img, size).ravel()

    return spatial_features


# NEED TO CHANGE bins_range if reading .png files with mpimg!
def color_hist(img, nbins=32, bins_range=(0, 1)):
    """
    compute color histogram features

    - Compute the histogram of the color channels separately
    - Concatenate the histograms into a single feature vector

    """
    channel1_hist = np.histogram(img[:, :, 0], bins=nbins, range=bins_range)
    channel2_hist = np.histogram(img[:, :, 1], bins=nbins, range=bins_range)
    channel3_hist = np.histogram(img[:, :, 2], bins=nbins, range=bins_range)

    hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))

    return hist_features


def extract_features(img, color_space='RGB', spatial_size=(32, 32),
                     hist_bins=32, orient=9,
                     pix_per_cell=8, cell_per_block=2, hog_channel=0,
                     spatial_feat=True, hist_feat=True, hog_feat=True):

    img_features = []

    if color_space != 'RGB':
        if color_space == 'HSV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
            feature_image[:, :, 1] = feature_image[:, :, 1]/feature_image[:, :, 1].max()
        elif color_space == 'LUV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
        elif color_space == 'HLS':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        elif color_space == 'YUV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
        elif color_space == 'YCrCb':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    else:
        feature_image = np.copy(img)

    if spatial_feat is True:
        spatial_features = bin_spatial(feature_image, size=spatial_size)
        img_features.append(spatial_features)

    if hist_feat is True:
        hist_features = color_hist(feature_image, nbins=hist_bins)
        img_features.append(hist_features)

    if hog_feat is True:
        if hog_channel == 'ALL':
            hog_features = []
            for channel in range(feature_image.shape[2]):
                hog_features.extend(get_hog_features(feature_image[:, :, channel],
                                                     orient,
                                                     pix_per_cell,
                                                     cell_per_block,
                                                     visualize_hog=False,
                                                     feature_vec=True))
        else:
            hog_features = get_hog_features(feature_image[:, :, hog_channel],
                                            orient,
                                            pix_per_cell,
                                            cell_per_block,
                                            visualize_hog=False,
                                            feature_vec=True)

        img_features.append(hog_features)

    return np.concatenate(img_features)


def extract_features_from_imglist(image_path_list,
                                  color_space='RGB',
                                  spatial_size=(32, 32),
                                  hist_bins=32,
                                  orient=9,
                                  pix_per_cell=8,
                                  cell_per_block=2,
                                  hog_channel=0,
                                  spatial_feat=True,
                                  hist_feat=True,
                                  hog_feat=True):
    """
    Define a function to extract features from a list of images

    """

    features = []

    for image_path in image_path_list:

        img = mpimg.imread(image_path)

        img_features = extract_features(img,
                                        color_space=color_space,
                                        spatial_size=spatial_size,
                                        hist_bins=hist_bins,
                                        orient=orient,
                                        pix_per_cell=pix_per_cell,
                                        cell_per_block=cell_per_block,
                                        hog_channel=hog_channel,
                                        spatial_feat=spatial_feat,
                                        hist_feat=hist_feat,
                                        hog_feat=hog_feat)

        features.append(img_features)

    return features


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


def slide_window(img, x_start_stop=[None, None], y_start_stop=[None, None],
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
                   hog_channel=0, spatial_feat=True,
                   hist_feat=True, hog_feat=True):
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
                                    spatial_feat=spatial_feat,
                                    hist_feat=hist_feat,
                                    hog_feat=hog_feat)

        test_features = scaler.transform(np.array(features).reshape(1, -1))

        prediction = clf.predict(test_features)

        if prediction == 1:
            windows_positive.append(window)

    return windows_positive


def draw_boxes_pos_windows(img, bboxes, color=(0, 0, 255), thick=6):
    """
    draw rectangles specified in bboxes (bounding boxes) to image
    """
    imcopy = np.copy(img)

    for bbox in bboxes:
        cv2.rectangle(imcopy, bbox[0], bbox[1], color, thick)

    return imcopy


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


if __name__ == "__main__":

    path_cars = '/Users/Klemens/Udacity_Nano_Car/P5_VehicleDetection/labeled_data/vehicles'
    path_notcars = '/Users/Klemens/Udacity_Nano_Car/P5_VehicleDetection/labeled_data/non-vehicles'
    cars = [os.path.join(dirpath, f) for dirpath, dirnames, files in os.walk(path_cars) for f in files if f.endswith('.png')]
    notcars = [os.path.join(dirpath, f) for dirpath, dirnames, files in os.walk(path_notcars) for f in files if f.endswith('.png')]


    # feature extraction
    ### TODO: Tweak these parameters and see how the results change.
    color_space = 'HSV' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
    orient = 9  # HOG orientations
    pix_per_cell = 8 # HOG pixels per cell
    cell_per_block = 2 # HOG cells per block
    hog_channel = "ALL" # Can be 0, 1, 2, or "ALL"
    spatial_size = (16, 16) # Spatial binning dimensions
    hist_bins = 10    # Number of histogram bins
    spatial_feat = True # Spatial features on or off
    hist_feat = True # Histogram features on or off
    hog_feat = True # HOG features on or off
    y_start_stop = [350, 600] # Min and max in y to search in slide_window()

    car_features = extract_features_from_imglist(cars,
                                                 color_space=color_space,
                                                 spatial_size=spatial_size,
                                                 hist_bins=hist_bins,
                                                 orient=orient,
                                                 pix_per_cell=pix_per_cell,
                                                 cell_per_block=cell_per_block,
                                                 hog_channel=hog_channel,
                                                 spatial_feat=spatial_feat,
                                                 hist_feat=hist_feat,
                                                 hog_feat=hog_feat)

    notcar_features = extract_features_from_imglist(notcars,
                                                    color_space=color_space,
                                                    spatial_size=spatial_size,
                                                    hist_bins=hist_bins,
                                                    orient=orient,
                                                    pix_per_cell=pix_per_cell,
                                                    cell_per_block=cell_per_block,
                                                    hog_channel=hog_channel,
                                                    spatial_feat=spatial_feat,
                                                    hist_feat=hist_feat,
                                                    hog_feat=hog_feat)

    X = np.vstack((car_features, notcar_features)).astype(np.float64)

    # Fit a per-column scaler
    X_scaler = StandardScaler().fit(X)

    # Apply the scaler to X
    scaled_X = X_scaler.transform(X)

    # Define the labels vector
    y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))

    # Split up data into randomized training and test sets
    rand_state = np.random.randint(0, 100)

    X_train_all, X_test, y_train_all, y_test = train_test_split(scaled_X,
                                                                y,
                                                                test_size=0.1,
                                                                random_state=rand_state)

    X_train, X_test, y_train, y_test = train_test_split(X_train_all,
                                                        y_train_all,
                                                        test_size=0.2,
                                                        random_state=rand_state)

    print('Using:', orient, 'orientations', pix_per_cell, 'pixels per cell and', cell_per_block, 'cells per block')
    print('Feature vector length:', len(X_train[0]))

    # Use a linear SVC classifier
    svc = LinearSVC()

    # Check the training time for the SVC
    t_start = time.time()
    svc.fit(X_train, y_train)

    print(round(time.time()-t_start, 2), 'Seconds to train SVC...')

    # Check the score of the SVC
    print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))

    image = mpimg.imread('test_images/test1.jpg')
    draw_image = np.copy(image)

    image = image.astype(np.float32)/255

    windows50 = slide_window(image, x_start_stop=[None, None], y_start_stop=y_start_stop,
                             xy_window=(50, 50), xy_overlap=(0.3, 0.3))

    windows96 = slide_window(image, x_start_stop=[None, None], y_start_stop=y_start_stop,
                             xy_window=(96, 96), xy_overlap=(0.5, 0.5))

    windows150 = slide_window(image, x_start_stop=[None, None], y_start_stop=y_start_stop,
                              xy_window=(150, 150), xy_overlap=(0.7, 0.7))

    windows = windows50 + windows96 + windows150

    # sliding window
    hot_windows = search_windows(image,
                                 windows,
                                 svc,
                                 X_scaler,
                                 color_space=color_space,
                                 spatial_size=spatial_size,
                                 hist_bins=hist_bins,
                                 orient=orient,
                                 pix_per_cell=pix_per_cell,
                                 cell_per_block=cell_per_block,
                                 hog_channel=hog_channel,
                                 spatial_feat=spatial_feat,
                                 hist_feat=hist_feat,
                                 hog_feat=hog_feat)

    window_img = draw_boxes_pos_windows(draw_image, hot_windows, color=(0, 0, 255), thick=6)

    plt.figure()
    plt.imshow(window_img)

    heatmap = np.zeros_like(image[:, :, 0]).astype(np.float)
    heatmap = add_heat(heatmap, hot_windows)

    heatmap_threshed = apply_threshold(heatmap, 2)
    labels = label(heatmap_threshed)

    print(labels[1], 'cars found')

    plt.figure()
    plt.imshow(labels[0], cmap='gray')

    draw_img = draw_boxes_cars(np.copy(image), labels)

    plt.figure()
    plt.imshow(draw_img)
#    plt.show()
