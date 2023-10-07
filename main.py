"""
@author: Na Hang Wei 19114479

Attention: This program take about 12 minutes to run all 4 datasets in author's environement setup
It is advised to run the dataset one by one as I face 'python not responding' problem when running all tgt
Uncomment the block to test the algorithm for the dataset.
"""

from algorithm import *
import time

starting_time = time.time()


# Dataset 623
dataset_623_path = "./Dataset/623/"
output_623_mask_path = "./GroundTruth/623/Mask"
output_623_skyline_path = "./GroundTruth/623/Skyline"
ground_truth_623 = "./GroundTruth/623.png"
skyline_detection(dataset_623_path, output_623_mask_path, output_623_skyline_path, ground_truth_623)

# # Dataset 684
# dataset_684_path = "./Dataset/684/"
# output_684_mask_path = "./GroundTruth/684/Mask"
# output_684_skyline_path = "./GroundTruth/684/Skyline"
# ground_truth_684 = "./GroundTruth/684.png"
# skyline_detection(dataset_684_path, output_684_mask_path, output_684_skyline_path, ground_truth_684)

# # Dataset 9730
# dataset_9730_path = "./Dataset/9730/"
# output_9730_mask_path = "./GroundTruth/9730/Mask"
# output_9730_skyline_path = "./GroundTruth/9730/Skyline"
# ground_truth_9730 = "./GroundTruth/9730.png"
# skyline_detection(dataset_9730_path, output_9730_mask_path, output_9730_skyline_path, ground_truth_9730)

# # Dataset 10917
# dataset_10917_path = "./Dataset/10917/"
# output_10917_mask_path = "./GroundTruth/10917/Mask"
# output_10917_skyline_path = "./GroundTruth/10917/Skyline"
# ground_truth_10917 = "./GroundTruth/10917.png"
# skyline_detection(dataset_10917_path, output_10917_mask_path, output_10917_skyline_path, ground_truth_10917)


ending_time = time.time()
processing_time = ending_time - starting_time
minutes, second = divmod(processing_time, 60)
print("Processing time used:", str(minutes), "minutes", str(second), "seconds")



