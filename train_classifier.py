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
import matplotlib.pyplot as plt

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


def get_bin_spatial(img, size=(32, 32), visualize=False):
    """
    Defines a function to compute binned color features. Basically
    downsampling the image.
    """
    color1 = cv2.resize(img[:,:,0], size)
    color2 = cv2.resize(img[:,:,1], size)
    color3 = cv2.resize(img[:,:,2], size)

    if visualize:
        images = [img, np.dstack((color1, color2, color3))]
        titles = ['car image', 'car spatial image']
        fig = plt.figure(figsize=(6,3))
        visualize_features(fig, 1, 2, images, titles, 'spatial_feature')

    return np.hstack((color1.ravel(), color2.ravel(), color3.ravel()))


# NEED TO CHANGE bins_range if reading .png files with mpimg!
def get_color_hist(img, nbins=32, visualize=False):#, bins_range=(0, 1)):
    """
    compute color histogram features

    - Compute the histogram of the color channels separately
    - Concatenate the histograms into a single feature vector

    """
    channel1_hist = np.histogram(img[:, :, 0], bins=nbins)#, range=bins_range)
    channel2_hist = np.histogram(img[:, :, 1], bins=nbins)#, range=bins_range)
    channel3_hist = np.histogram(img[:, :, 2], bins=nbins)#, range=bins_range)

    hist_feat = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))

    if visualize:
        plot_histfeatures(channel1_hist, channel2_hist, channel3_hist)
    return hist_feat


def change_color_space(img, colorspace):
    img_copy = np.copy(img)

    if colorspace != 'RGB':
        if colorspace == 'HSV':
            feature_image = cv2.cvtColor(img_copy, cv2.COLOR_RGB2HSV)

        elif colorspace == 'LUV':
            feature_image = cv2.cvtColor(img_copy, cv2.COLOR_RGB2LUV)

        elif colorspace == 'HLS':
            feature_image = cv2.cvtColor(img_copy, cv2.COLOR_RGB2HLS)

        elif colorspace == 'YUV':
            feature_image = cv2.cvtColor(img_copy, cv2.COLOR_RGB2YUV)

        elif colorspace == 'YCrCb':
            feature_image = cv2.cvtColor(img_copy, cv2.COLOR_RGB2YCrCb)

    else:
        feature_image = img_copy
    return feature_image


def extract_features(img, color_space='RGB', spatial_size=(32, 32),
                     hist_bins=32, orient=9,
                     pix_per_cell=8, cell_per_block=2, hog_channel=0,
                     use_spatial_feat=True, use_hist_feat=True, use_hog_feat=True, visualize=False):

    img_features = []
    feature_image = change_color_space(img, color_space)

    if use_spatial_feat is True:
        spatial_features = get_bin_spatial(feature_image, size=spatial_size, visualize=visualize)
        img_features.append(spatial_features)

    if use_hist_feat is True:
        hist_features = get_color_hist(feature_image, nbins=hist_bins, visualize=visualize)
        img_features.append(hist_features)

    if use_hog_feat is True:
        hog_features = []
        for channel in hog_channel:
            if visualize is True:
                hog_feature_chan, hog_image = get_hog_features(feature_image[:, :, channel],
                                                     orient,
                                                     pix_per_cell,
                                                     cell_per_block,
                                                     visualize_hog=visualize,
                                                     feature_vec=True)
                hog_features.append(hog_feature_chan)

                images = [img, hog_image]
                titles = ['car image', 'car hog image ' + str(channel)]
                fig = plt.figure(figsize=(6, 3))
                visualize_features(fig, 1, 2, images, titles, 'hog_feature_' + str(channel))
            else:

                hog_features.append(get_hog_features(feature_image[:, :, channel],
                                                     orient,
                                                     pix_per_cell,
                                                     cell_per_block,
                                                     visualize_hog=visualize,
                                                     feature_vec=True))
        hog_features = np.ravel(hog_features)

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
                                  use_hog_feat=True,
                                  visualize=False):
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
                                        use_hog_feat=use_hog_feat,
                                        visualize=visualize)

        features.append(img_features)

    return features

def plot_histfeatures(hist1, hist2, hist3):

        # Generating bin centers
    bin_edges = hist1[1]
    bin_centers = (bin_edges[1:] + bin_edges[0:len(bin_edges)-1])/2

    fig = plt.figure(figsize=(12, 3))
    plt.subplot(131)
    plt.bar(bin_centers, hist1[0])
    # plt.xlim(0, 256)
    plt.title('Histogram Channel 1')
    plt.subplot(132)
    plt.bar(bin_centers, hist2[0])
    # plt.xlim(0, 256)
    plt.title('Histogram Channel 2')
    plt.subplot(133)
    plt.bar(bin_centers, hist3[0])
    # plt.xlim(0, 256)
    plt.title('Histogram Channel 3')
    plt.savefig('output_images/hist_features', dpi=150)

    plt.close()

def visualize_features(fig, rows, cols, imgs, titles, savetitle):
    for i, img in enumerate(imgs):
        plt.subplot(rows, cols, i+1)
        plt.title(i+1)
        img_dims = len(img.shape)
        if img_dims < 3:
            plt.imshow(img, cmap='hot')
            plt.xlim((0, img.shape[1]))
            plt.ylim((0, img.shape[0]))
            plt.title(titles[i])
        else:
            plt.imshow(img)
            plt.title(titles[i])

        plt.savefig('output_images/' + savetitle, dpi=150)#    plt.savefig('output_images/binary_combo_example.jpg')
    plt.close()

if __name__ == "__main__":

    path_cars = '/Users/Klemens/Udacity_Nano_Car/P5_labeled_data/vehicles'
    path_notcars = '/Users/Klemens/Udacity_Nano_Car/P5_labeled_data/non-vehicles'
    cars = [os.path.join(dirpath, f) for dirpath, dirnames, files in os.walk(path_cars) for f in files if f.endswith('.png')]
    notcars = [os.path.join(dirpath, f) for dirpath, dirnames, files in os.walk(path_notcars) for f in files if f.endswith('.png')]
    VISUALIZE=True


    cars = cars[10:20]
    notcars = notcars[10:20]

    # cars_train, cars_test, no_train, no_test = read_in_data.read_in_seperately(path_cars, path_notcars)
    color_space = 'HLS'  # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
    orientation = 9  # HOG orientations
    num_pix_per_cell = 8  # HOG pixels per cell
    num_cell_per_block = 2  # HOG cells per block
    hog_channel_select = [0, 1, 2]  # Can be 0, 1, 2, or "ALL"
    spatial_size = (32, 32)  # Spatial binning dimensions
    histogram_bins = 32    # Number of histogram bins
    use_spatial_features = True
    use_hist_features = True
    use_hog_features = True


    if VISUALIZE:

        images = [mpimg.imread(cars[5]), mpimg.imread(notcars[5])]
        titles = ['car image', 'not-car image']
        fig = plt.figure(figsize=(6, 3))

        visualize_features(fig, 1, 2, images, titles, 'car-noncar')

    print('Car Features')
    cars_features = extract_features_from_imglist(cars,
                                                     color_space=color_space,
                                                     spatial_size=spatial_size,
                                                     hist_bins=histogram_bins,
                                                     orient=orientation,
                                                     pix_per_cell=num_pix_per_cell,
                                                     cell_per_block=num_cell_per_block,
                                                     hog_channel=hog_channel_select,
                                                     use_spatial_feat=use_spatial_features,
                                                     use_hist_feat=use_hist_features,
                                                     use_hog_feat=use_hog_features,
                                                     visualize=False)


    print('Notcar Features')
    not_car_features = extract_features_from_imglist(notcars,
                                                    color_space=color_space,
                                                    spatial_size=spatial_size,
                                                    hist_bins=histogram_bins,
                                                    orient=orientation,
                                                    pix_per_cell=num_pix_per_cell,
                                                    cell_per_block=num_cell_per_block,
                                                    hog_channel=hog_channel_select,
                                                    use_spatial_feat=use_spatial_features,
                                                    use_hist_feat=use_hist_features,
                                                    use_hog_feat=use_hog_features,
                                                    visualize=True)

    #

    X = np.vstack((cars_features, not_car_features)).astype(np.float64)

    # Fit a per-column scaler
    X_scaler = StandardScaler().fit(X)

    # Apply the scaler to X
    scaled_X = X_scaler.transform(X)

    # Define the labels vector
    y = np.hstack((np.ones(len(cars_features)), np.zeros(len(not_car_features))))

    rand_state = np.random.randint(0, 100)

    X_train, X_test, y_train, y_test = train_test_split(scaled_X,
                                                        y,
                                                        test_size=0.1,
                                                        random_state=rand_state)

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

    pickle_file = 'ClassifierData_HLS.p'
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
