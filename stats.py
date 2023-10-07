"""
For internal usage to get statistics of images/datasets
"""

import cv2
import numpy as np
from matplotlib import pyplot as pt 
import os

# Function to draw a histogram for image
def draw_histogram(image):
    histogram = cv2.calcHist([image], [0], None, [256], [0, 256])
    pt.plot(histogram)
    

# Function to calculate optimal threshold for a list of intensity values
def calculate_threshold(intensity_values):
    intensity_array = np.array(intensity_values, dtype=np.uint8) # convert to numpy array
    histogram = cv2.calcHist([intensity_array], [0], None, [256], [0, 256])
    threshold, _ = cv2.threshold(intensity_array, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    
    return histogram, threshold


# Function to calculate mean intensity value for all images in a folder
def get_mean_intensity(folder_path):
    mean_intensity_values = []
    median_intensity_values = []

    for filename in os.listdir(folder_path):
        image_path = os.path.join(folder_path, filename)
        image = cv2.imread(image_path, cv2.COLOR_BGR2GRAY)

        if image is not None:
            mean_intensity = np.mean(image) # get mean intensity of a single image
            mean_intensity_values.append(mean_intensity)
            median_intensity = np.median(image)
            median_intensity_values.append(median_intensity)
     
    mean_histogram, mean_threshold = calculate_threshold(mean_intensity_values)
    # median_histogram, median_threshold = calculate_threshold(median_intensity_values)
    
    pt.plot(mean_histogram)
    # pt.plot(median_histogram)
    pt.title("Histogram of mean intensity value for " + folder_path)
    pt.xlabel("Intensity Value")
    pt.ylabel("Frequency")
    pt.show()
    
    print("The optimal mean threshold value is: " , str(mean_threshold))
    # print("The optimal median threshold value is: " , str(median_threshold))
    
# get_mean_intensity("./Dataset/10917/")

    
