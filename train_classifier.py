import os
import time
import pickle
import cv2
import numpy as np

import matplotlib.image as mpimg

from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from sklearn.utils import shuffle
import read_in_data

def get_hog_features(img, orient, pix_per_cell, cell_per_block, visualize_hog=False, feature_vec=True):
    """
    computes a histogram of oriented gradients (HOG), which is robust to variations in shape
    returns HOG features and visualization (optional)
    """

    pixels_per_cell = (pix_per_cell, pix_per_cell)
    cells_per_block = (cell_per_block, cell_per_block)

    if visualize_hog is True:
        hog_feat, hog_image = hog(img,
                                  orientations=orient,
                                  pixels_per_cell=pixels_per_cell,
                                  cells_per_block=cells_per_block,
                                  transform_sqrt=False,
                                  visualise=visualize_hog,
                                  feature_vector=feature_vec)

        return hog_feat, hog_image

    else:
        hog_feat = hog(img,
                       orientations=orient,
                       pixels_per_cell=pixels_per_cell,
                       cells_per_block=cells_per_block,
                       transform_sqrt=False,
                       visualise=visualize_hog,
                       feature_vector=feature_vec)

        return hog_feat


def get_bin_spatial(img, size=(32, 32)):
    """
    Defines a function to compute binned color features. Basically
    downsampling the image.
    """
    color1 = cv2.resize(img[:,:,0], size).ravel()
    color2 = cv2.resize(img[:,:,1], size).ravel()
    color3 = cv2.resize(img[:,:,2], size).ravel()

    return np.hstack((color1, color2, color3))


# NEED TO CHANGE bins_range if reading .png files with mpimg!
def get_color_hist(img, nbins=32):#, bins_range=(0, 1)):
    """
    compute color histogram features

    - Compute the histogram of the color channels separately
    - Concatenate the histograms into a single feature vector

    """
    channel1_hist = np.histogram(img[:, :, 0], bins=nbins)#, range=bins_range)
    channel2_hist = np.histogram(img[:, :, 1], bins=nbins)#, range=bins_range)
    channel3_hist = np.histogram(img[:, :, 2], bins=nbins)#, range=bins_range)

    hist_feat = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))

    return hist_feat


def change_color_space(img, colorspace):

    if colorspace != 'RGB':
        if colorspace == 'HSV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

        elif colorspace == 'LUV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2LUV)

        elif colorspace == 'HLS':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)

        elif colorspace == 'YUV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)

        elif colorspace == 'YCrCb':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)

    else:
        feature_image = np.copy(img)

    return feature_image


def extract_features(img, color_space='RGB', spatial_size=(32, 32),
                     hist_bins=32, orient=9,
                     pix_per_cell=8, cell_per_block=2, hog_channel=0,
                     use_spatial_feat=True, use_hist_feat=True, use_hog_feat=True):

    img_features = []

    feature_image = change_color_space(img, color_space)

    if use_spatial_feat is True:
        spatial_features = get_bin_spatial(feature_image, size=spatial_size)
        img_features.append(spatial_features)

    if use_hist_feat is True:
        hist_features = get_color_hist(feature_image, nbins=hist_bins)
        img_features.append(hist_features)

    if use_hog_feat is True:
        if type(hog_channel) is list:
            hog_features = []
            for channel in hog_channel:
                hog_features.append(get_hog_features(feature_image[:, :, channel],
                                                     orient,
                                                     pix_per_cell,
                                                     cell_per_block,
                                                     visualize_hog=False,
                                                     feature_vec=True))
            hog_features = np.ravel(hog_features)
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
        # print(image_path)
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

    path_cars = '/Users/Klemens/Udacity_Nano_Car/P5_labeled_data/vehicles'
    path_notcars = '/Users/Klemens/Udacity_Nano_Car/P5_labeled_data/non-vehicles'
    # cars = [os.path.join(dirpath, f) for dirpath, dirnames, files in os.walk(path_cars) for f in files if f.endswith('.png')]
    # notcars = [os.path.join(dirpath, f) for dirpath, dirnames, files in os.walk(path_notcars) for f in files if f.endswith('.png')]
    #
    # cars = cars#[0:10]
    # notcars = notcars#[0:10]

    cars_train, cars_test, no_train, no_test = read_in_data.read_in_seperately(path_cars, path_notcars)
    color_space = 'YCrCb'  # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
    orientation = 9  # HOG orientations
    num_pix_per_cell = 8  # HOG pixels per cell
    num_cell_per_block = 2  # HOG cells per block
    hog_channel_select = [0, 1, 2]  # Can be 0, 1, 2, or "ALL"
    spatial_size = (16, 16)  # Spatial binning dimensions
    histogram_bins = 32    # Number of histogram bins
    use_spatial_features = True
    use_hist_features = True
    use_hog_features = True

    print('Car Features')
    cars_train_features = extract_features_from_imglist(cars_train,
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


    cars_test_features = extract_features_from_imglist(cars_test,
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

    print('Notcar Features')
    no_train_features = extract_features_from_imglist(no_train,
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


    no_test_features = extract_features_from_imglist(no_test,
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


    X = np.vstack((cars_train_features, cars_test_features, no_train_features, no_test_features)).astype(np.float64)

    # Fit a per-column scaler
    X_scaler = StandardScaler().fit(X)

    # Apply the scaler to X
    scaled_X = X_scaler.transform(X)


    cars_ntrain = len(cars_train_features)
    cars_ntest = len(cars_test_features)
    ncars_ntrain = len(no_train_features)
    ncars_ntest = len(no_test_features)

    idx_feat_cars_train_end = cars_ntrain
    idx_feat_cars_test_end = idx_feat_cars_train_end + cars_ntest
    idx_feat_no_train_end = idx_feat_cars_test_end + ncars_ntrain

    cars_train_feat = scaled_X[:idx_feat_cars_train_end]
    cars_test_feat = scaled_X[idx_feat_cars_train_end:idx_feat_cars_test_end]

    notcars_train_feat = scaled_X[idx_feat_cars_test_end:idx_feat_no_train_end]
    notcars_test_feat = scaled_X[idx_feat_no_train_end:]

    y_train = np.hstack((np.ones(cars_ntrain), np.zeros(ncars_ntrain)))
    y_test = np.hstack((np.ones(cars_ntest), np.zeros(ncars_ntest)))

    X_train = np.vstack((cars_train_feat, notcars_train_feat))
    X_test = np.vstack((cars_test_feat, notcars_test_feat))

    rand_state = np.random.randint(0, 100)

    X_train, y_train = shuffle(X_train, y_train, random_state=rand_state)
    X_test, y_test = shuffle(X_test, y_test, random_state=rand_state)

    # # Define the labels vector
    # y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))

    # X_train_all, X_test, y_train_all, y_test = train_test_split(scaled_X,
    #                                                             y,
    #                                                             test_size=0.1,
    #                                                             random_state=rand_state)
    #
    # X_train, X_val, y_train, y_val = train_test_split(X_train_all,
    #                                                     y_train_all,
    #                                                     test_size=0.2,
    #                                                     random_state=rand_state)

    print('Using:', orientation, 'orientations', num_pix_per_cell, 'pixels per cell and', num_cell_per_block, 'cells per block')
    print('Feature vector length:', len(X_train[0]))

    # Use a linear SVC classifier
    svc = LinearSVC()

    # Check the training time for the SVC
    t_start = time.time()
    svc.fit(X_train, y_train)

    print(round(time.time()-t_start, 2), 'Seconds to train SVC...')

    # Check the score of the SVC
    print('Train Accuracy of SVC = ', round(svc.score(X_train, y_train), 4))
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
                         'histogram_bins': histogram_bins,
                         'orientation': orientation,
                         'num_pix_per_cell': num_pix_per_cell,
                         'num_cell_per_block': num_cell_per_block,
                         'hog_channel_select': hog_channel_select,
                         'use_spatial_features': use_spatial_features,
                         'use_hist_features': use_hist_features,
                         'use_hog_features': use_hog_features
                         },
                        pfile, pickle.HIGHEST_PROTOCOL)

    except Exception as e:
        print('Unable to save data to', pickle_file, ':', e)
        raise
