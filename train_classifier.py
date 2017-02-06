import cv2

import pickle
import os
import numpy as np

import matplotlib.image as mpimg

from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog
from sklearn.model_selection import train_test_split
from tqdm import *

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


def change_color_space(img, color_space):

    if color_space != 'RGB':
        if color_space == 'HSV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

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

    return img


def extract_features(img, color_space='RGB', spatial_size=(32, 32),
                     hist_bins=32, orient=9,
                     pix_per_cell=8, cell_per_block=2, hog_channel=0,
                     use_spatial_feat=True, use_hist_feat=True, use_hog_feat=True):

    img_features = []

    feature_image = change_color_space(img, color_space)

    if use_spatial_feat is True:
        spatial_features = bin_spatial(feature_image, size=spatial_size)
        img_features.append(spatial_features)

    if use_hist_feat is True:
        hist_features = color_hist(feature_image, nbins=hist_bins)
        img_features.append(hist_features)

    if use_hog_feat is True:
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
                                  use_spatial_feat=True,
                                  use_hist_feat=True,
                                  use_hog_feat=True):
    """
    Define a function to extract features from a list of images

    """

    features = []

    for image_path in tqdm(image_path_list):

        img = mpimg.imread(image_path)

        img_features = extract_features(img,
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

        features.append(img_features)

    return features

if __name__ == "__main__":

    path_cars = '/Users/Klemens/Udacity_Nano_Car/P5_VehicleDetection/labeled_data/vehicles'
    path_notcars = '/Users/Klemens/Udacity_Nano_Car/P5_VehicleDetection/labeled_data/non-vehicles'
    cars = [os.path.join(dirpath, f) for dirpath, dirnames, files in os.walk(path_cars) for f in files if f.endswith('.png')]
    notcars = [os.path.join(dirpath, f) for dirpath, dirnames, files in os.walk(path_notcars) for f in files if f.endswith('.png')]

    color_space = 'HSV'  # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
    orient = 9  # HOG orientations
    pix_per_cell = 8  # HOG pixels per cell
    cell_per_block = 2  # HOG cells per block
    hog_channel = 0 # Can be 0, 1, 2, or "ALL"
    spatial_size = (16, 16)  # Spatial binning dimensions
    hist_bins = 10    # Number of histogram bins
    use_spatial_feat = True
    use_hist_feat = True
    use_hog_feat = True

    print('Car Features')
    car_features = extract_features_from_imglist(cars,
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

    print('Notcar Features)')
    notcar_features = extract_features_from_imglist(notcars,
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

    pickle_file = 'ClassifierData.p'
    print('Saving data to pickle file...')

    try:
        with open(pickle_file, 'wb') as pfile:
            pickle.dump(
                        {'svc': svc,
                         'X_scaler': X_scaler,
                         'color_space': color_space,
                         'spatial_size': spatial_size,
                         'hist_bins': hist_bins,
                         'orient': orient,
                         'pix_per_cell': pix_per_cell,
                         'cell_per_block': cell_per_block,
                         'hog_channel': hog_channel,
                         'use_spatial_feat': use_spatial_feat,
                         'use_hist_feat': use_hist_feat,
                         'use_hog_feat': use_hog_feat
                         },
                        pfile, pickle.HIGHEST_PROTOCOL)

    except Exception as e:
        print('Unable to save data to', pickle_file, ':', e)
        raise
