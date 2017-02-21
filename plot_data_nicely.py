#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 21 15:37:31 2017

@author: Klemens
"""

import matplotlib.image as mpimg
import matplotlib.pyplot as plt

plt.close('all')

windows_list = [mpimg.imread("Writeup/Final_pipeline/pos_windowsALL_img64_frame9.png"), 
                mpimg.imread("Writeup/Final_pipeline/pos_windowsALL_img64_frame10.png"),
                mpimg.imread("Writeup/Final_pipeline/pos_windowsALL_img64_frame11.png"),
                mpimg.imread("Writeup/Final_pipeline/pos_windowsALL_img64_frame12.png"),
                mpimg.imread("Writeup/Final_pipeline/pos_windowsALL_img64_frame13.png"),
                mpimg.imread("Writeup/Final_pipeline/pos_windowsALL_img64_frame14.png")]

heatmap_list = [mpimg.imread("Writeup/Final_pipeline/heatmap_average_frame9.png"),
                mpimg.imread("Writeup/Final_pipeline/heatmap_average_frame10.png"),
                mpimg.imread("Writeup/Final_pipeline/heatmap_average_frame11.png"),
                mpimg.imread("Writeup/Final_pipeline/heatmap_average_frame12.png"),
                mpimg.imread("Writeup/Final_pipeline/heatmap_average_frame13.png"),
                mpimg.imread("Writeup/Final_pipeline/heatmap_average_frame14.png")]

for i in range(len(windows_list)):
    
    fig = plt.figure(figsize=(10,4))
    ax = fig.add_subplot(121)
    ax.imshow(windows_list[i])
    ax.set_title('Positive Windows with 64x64 window')
    ax = fig.add_subplot(122)
    ax.imshow(heatmap_list[i])
    ax.set_title('Averaged Heatmap')
    
    plt.tight_layout()
    
    fig.savefig('pipeline_frame' + str(i+9))