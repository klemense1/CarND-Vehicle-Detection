#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  5 22:06:53 2017

@author: Klemens
"""

import os
import matplotlib.image as mpimg
import cv2
import numpy as np
import matplotlib.pyplot as plt

def saturation_threshold(img, thresh = (90, 255)):

    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    S = hls[:,:,2]

    sat_binary = np.zeros_like(S)
    sat_binary[(S > thresh[0]) & (S <= thresh[1])] = 1

    return sat_binary


if __name__ == "__main__":

    path_cars = '/Users/Klemens/Udacity_Nano_Car/P5_labeled_data//vehicles'
    path_notcars = '/Users/Klemens/Udacity_Nano_Car/P5_labeled_data//non-vehicles'
    cars = [os.path.join(dirpath, f) for dirpath, dirnames, files in os.walk(path_cars) for f in files if f.endswith('.png')]
    notcars = [os.path.join(dirpath, f) for dirpath, dirnames, files in os.walk(path_notcars) for f in files if f.endswith('.png')]

    file = cars[2]
    image = mpimg.imread(file)

    plt.figure(figsize = (16,4))
    (f, [ax1, ax2, ax3, ax4]) = plt.subplots(1,4, figsize = (20,5))
    ax1.imshow(image)
    ax1.set_title('RGB')

    R_Channel = image[:, :, 0]
    ax2.imshow(R_Channel, 'gray')
    minval = R_Channel.min().round(1)
    maxval = R_Channel.max().round(1)
    str_title = 'R ' + '[Min:' + str(minval) + ', Max:' + str(maxval) + ']'
    ax2.set_title(str_title)

    G_Channel = image[:, :, 1]
    ax3.imshow(G_Channel, 'gray')
    minval = G_Channel.min().round(1)
    maxval = G_Channel.max().round(1)
    str_title = 'G ' + '[Min:' + str(minval) + ', Max:' + str(maxval) + ']'
    ax3.set_title(str_title)

    B_Channel = image[:, :, 2]
    ax4.imshow(B_Channel, 'gray')
    minval = B_Channel.min().round(1)
    maxval = B_Channel.max().round(1)
    str_title = 'B ' + '[Min:' + str(minval) + ', Max:' + str(maxval) + ']'
    ax4.set_title(str_title)
    plt.savefig('output_images/rgb', dpi=150)#    plt.savefig('output_images/binary_combo_example.jpg')


    image_hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)

#    plt.figure()
    (f, [ax1, ax2, ax3, ax4]) = plt.subplots(1,4, figsize = (20,5))
    ax1.imshow(image_hls)
    ax1.set_title('HLS')

    H_Channel = image_hls[:, :, 0]
    ax2.imshow(H_Channel, 'gray')
    minval = H_Channel.min().round(1)
    maxval = H_Channel.max().round(1)
    str_title = 'H ' + '[Min:' + str(minval) + ', Max:' + str(maxval) + ']'
    ax2.set_title(str_title)

    L_Channel = image_hls[:, :, 1]
    ax3.imshow(L_Channel, 'gray')
    minval = L_Channel.min().round(1)
    maxval = L_Channel.max().round(1)
    str_title = 'L ' + '[Min:' + str(minval) + ', Max:' + str(maxval) + ']'
    ax3.set_title(str_title)

    S_Channel = image_hls[:, :, 2]
    ax4.imshow(S_Channel, 'gray')
    minval = S_Channel.min().round(1)
    maxval = S_Channel.max().round(1)
    str_title = 'S ' + '[Min:' + str(minval) + ', Max:' + str(maxval) + ']'
    ax4.set_title(str_title)

    plt.savefig('output_images/hls', dpi=150)#    plt.savefig('output_images/binary_combo_example.jpg')

    image_hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

    (f, [ax1, ax2, ax3, ax4]) = plt.subplots(1,4, figsize = (20,5))
    ax1.imshow(image_hsv)
    ax1.set_title('HSV')

    H_Channel = image_hsv[:, :, 0]
    ax2.imshow(H_Channel, 'gray')
    minval = H_Channel.min().round(1)
    maxval = H_Channel.max().round(1)
    str_title = 'H ' + '[Min:' + str(minval) + ', Max:' + str(maxval) + ']'
    ax2.set_title(str_title)

    S_Channel = image_hsv[:, :, 1]
    ax3.imshow(S_Channel, 'gray')
    minval = S_Channel.min().round(1)
    maxval = S_Channel.max().round(1)
    str_title = 'S ' + '[Min:' + str(minval) + ', Max:' + str(maxval) + ']'
    ax3.set_title(str_title)

    V_Channel = image_hsv[:, :, 2]
    ax4.imshow(V_Channel, 'gray')
    minval = V_Channel.min().round(1)
    maxval = V_Channel.max().round(1)
    str_title = 'V ' + '[Min:' + str(minval) + ', Max:' + str(maxval) + ']'
    ax4.set_title(str_title)

    plt.savefig('output_images/hsv', dpi=150)#    plt.savefig('output_images/binary_combo_example.jpg')
