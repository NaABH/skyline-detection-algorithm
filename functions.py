import numpy as np
import cv2

# Function to extract blue plane from image
# parameter: rgb image
def extract_blue_plane(image):
    blue_channel = image[:,:,0]
    blue_plane = cv2.merge([blue_channel, blue_channel, blue_channel])
    
    return blue_plane


# Function to pre process image
# parameter: rgb image
def preprocessing_image(image):    
    # convert image to greyscale image
    grayscale_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # smooth image using GaussianBlur
    smoothed_image = cv2.GaussianBlur(grayscale_img,(3,3),0)
    return smoothed_image


# Function to merge two images together (for visualising purpose)
# parameter: rgb image, greyscale image
def merge_img(original_image, output_image):
    output_with_3channel = cv2.merge([output_image, output_image, output_image])
    final_image = cv2.addWeighted(original_image, 0.7, output_with_3channel, 0.3, 0)
    
    return final_image


# Function to apply floodfill to an image with a given seed point
# parameter: mask, coordinate
def apply_floodFill(image, seed_point):
    height, width = image.shape[:2]
    temp_img = image.copy()
    
    cv2.floodFill(temp_img, None, seed_point, 255)
    
    # Invert floodfilled image
    inverted_img = cv2.bitwise_not(temp_img)
    
    # Combine the two images to get the foreground.
    output = cv2.bitwise_or(image, inverted_img)
    output = cv2.bitwise_not(output)
    
    return output

# Function to perform gamma correction to solve overexposure (for dataset 9730)
# retrieve from https://pyimagesearch.com/2015/04/06/zero-parameter-automatic-canny-edge-detection-with-python-and-opencv/
def adjust_gamma(image, gamma=1.5):
	invGamma = 1.0 / gamma
    
	table = np.array([((i / 255.0) ** invGamma) * 255
		for i in np.arange(0, 256)]).astype("uint8")
    
	# apply gamma correction using the lookup table
	return cv2.LUT(image, table)

# Function for day/night classification
def is_day(image):
    mean_intensity_value = np.mean(image)
    
    if mean_intensity_value >= 94:
        return True
    else:
        return False

# Function to draw skyline for a mask
# parameter: binary mask
def draw_skyline(binary_mask):
    kernel = np.ones((5, 5), np.uint8)
    dilated_mask = cv2.dilate(binary_mask, kernel, iterations=1)
    skyline_image = dilated_mask - binary_mask

    return skyline_image






