# -*- coding: utf-8 -*-
"""
Project Manager: Cioată Sebastian
Software Developer: Cioată Sebastian

Last date edited: 13.12.2020
"""

import numpy as np
import cv2
from matplotlib import pyplot as plt


# Set the path to the image
image_location = "/test_image5.jpg"

# Read the image 
image = cv2.imread(image_location)

# Display the image
cv2.imshow('Original Image', image) 

# Convert the original image to grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Display the grayscale image
cv2.imshow('Grayscale Image', gray_image)
cv2.waitKey()
cv2.destroyAllWindows()

# Compute the histogram of the grayscale image
histogram = plt.hist(gray_image.ravel(),256,[0,256]) 
plt.show() 

# Compute the number of pixels of the grayscale image
pixel_number = gray_image.shape[0] * gray_image.shape[1]
print('Number of pixels:', pixel_number)

# Compute the mean weight
mean_weight = 1.0/pixel_number
print('Mean weight:', mean_weight)

# Calculate the normalized histogram and the number of bins
his, bins = np.histogram(gray_image, np.array(range(0, 256)))

# Initialize the variables
final_thresh = -1
final_variance = -1

# Iterate through all possible thresholds
for t in bins[1:-1]: # This goes from 1 to 254 uint8 range 
    # Compute and update the probabilities
    Wb = np.sum(his[:t]) * mean_weight
    Wf = np.sum(his[t:]) * mean_weight

    # Compute and update the means
    mub = np.mean(his[:t])
    muf = np.mean(his[t:])
    
    # Compute the maximized inter-class variance
    variance = Wb * Wf * (mub - muf) ** 2

    # Compare current variance with the greatest variance until now
    if variance > final_variance:
        final_thresh = t
        final_variance = variance
        
final_img = gray_image.copy()        
print('Final threshold:', final_thresh)

# Makes all pixels greater than final threshold white
final_img[gray_image > final_thresh] = 255 

# Makes all pixels smaller than final threshold black
final_img[gray_image < final_thresh] = 0
 

cv2.imshow('Final Image', final_img) 
cv2.waitKey()
cv2.destroyAllWindows()