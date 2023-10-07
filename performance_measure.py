import os
import random
import matplotlib.pyplot as pt
import matplotlib.image as mpimg
from functions import is_day

# Function to calculate accuracy for a single image by comparing pixel by pixel with the given ground truth
# parameter: mask, given_groundtruth
def calculate_accuracy(image, groundtruth_image):
    true_positive = ((image == 255) & (groundtruth_image == 255)).sum()
    true_negative = ((image == 0) & (groundtruth_image == 0)).sum()
    false_positive = ((image == 0) & (groundtruth_image == 255)).sum()
    false_negative = ((image == 255) & (groundtruth_image == 0)).sum()
    accuracy = (true_positive + true_negative) / (true_positive + true_negative + false_positive + false_negative)
    return accuracy

# Function to display day/night classification result from 12 random image
def display_random_images(dataset_path, num_images=12, grid_shape=(3, 4)):
    image_files = os.listdir(dataset_path)
    random.shuffle(image_files)

    fig, axes = pt.subplots(*grid_shape, figsize=(12, 9))

    for i in range(num_images):
        image_file = image_files[i]
        image_path = os.path.join(dataset_path, image_file)

        img = mpimg.imread(image_path)

        # show image in subplot
        row = i // grid_shape[1]
        col = i % grid_shape[1]
        axes[row, col].imshow(img)
        axes[row, col].axis('off')
        
        if is_day(img):
            axes[row, col].set_title("Day")
        else:
            axes[row, col].set_title("Night")

    pt.tight_layout()
    pt.show()
