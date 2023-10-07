from performance_measure import *
from functions import *
import os


se = cv2.getStructuringElement(cv2.MORPH_CROSS, (9,9))
se = se.astype(np.uint8)
sE = np.ones((5,5),np.uint8)

def skyline_detection(input_path, output_path, output_skyline_path, gt_path):
    # List all image files in the folder
    image_files = [file for file in os.listdir(input_path) if file.endswith(".jpg")]
    
    # Initialize variables
    overall_accuracy = 0
    total_images = len(image_files)
    ground_truth = cv2.imread(gt_path, cv2.COLOR_BGR2GRAY)
    total_processed = 0

    # Loop through each image in the folder
    for image_file in image_files:
        image = cv2.imread(os.path.join(input_path, image_file))
        
        # Day image
        if is_day(image):
            mask, accuracy = optimal_mask(image, ground_truth)
            
            # Write mask into a folder
            output_image_file = image_file.replace(".jpg", "_ground_truth.jpg")
            output_skyline_file = image_file.replace(".jpg", "_skyline.jpg")
            
            # Write mask into folder
            output_gt_image_path = os.path.join(output_path, output_image_file)
            cv2.imwrite(output_gt_image_path, mask)
            
            # Write skyline into a folder
            output_sl_image_path = os.path.join(output_skyline_path, output_skyline_file)
            cv2.imwrite(output_sl_image_path, draw_skyline(mask))
            
            # Write skyline into a folder
            skyline = draw_skyline(mask)
            
    
            # Add accuracy to the overall accuracy
            overall_accuracy += accuracy
            print(f"Accuracy for {image_file}: {accuracy:.4f}")
            total_processed += 1

    # Compute the overall accuracy
    overall_accuracy /= total_processed
    print("*********************************\nRunning on Dataset: " + input_path)
    print(f"Overall Accuracy: {overall_accuracy:.4f}")
    print("Total Image processed: ", str(total_processed))
    print("All ground truth image generated is saved in " + output_path)
    print("All skyline image generated is saved in " + output_skyline_path)
    display_random_images(input_path)


def edge_based_canny(skyimg):
    # Phase1: Preprocessing step
    smooth_img = preprocessing_image(skyimg)
    
    # Phase2: Detect edges and enclose the gaps (Segmentation)
    skyimg_edge = cv2.Canny(smooth_img, 16, 186)
    enclosed_edge = cv2.morphologyEx(skyimg_edge, cv2.MORPH_CLOSE, se, iterations=2)
    
    # Phase3: Fill region at point (0,0)
    # starting_point = get80_percentile_value(image)
    mask = apply_floodFill(enclosed_edge, (0,0))
    
    return mask


def region_based_watershed(skyimg):
    # Phase1: Preprocessing step
    smooth_img = preprocessing_image(skyimg)
    
    # Phase 2: Apply otsu method
    _, thrlmg = cv2.threshold(smooth_img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    
    # Phase 3: Watershed segmentation
    
    # find sure sky region
    sureBg = cv2.dilate(thrlmg,sE,iterations=3)
    # use distance tranform
    distTransform = cv2.distanceTransform(thrlmg,cv2.DIST_L2,5)
    # find sure ground region
    _, sureFg = cv2.threshold(distTransform,0.05*distTransform.max(),255,0)
    sureFg = np.uint8(sureFg)
    # find unknown region
    unknownRegion = cv2.subtract(sureBg,sureFg)
    # label all the regions
    noRegion, markers = cv2.connectedComponents(sureFg)
    height, width = skyimg.shape[:2]
    
    if noRegion > 20:
        mask = np.zeros((height, width), np.uint8)
        return mask
    
    # print(noRegion)
    # background label as 1
    markers = markers+1
    # unknown label as 0
    markers[unknownRegion==255] = 0
    # apply watershed
    watershedMarkers = cv2.watershed(skyimg, markers.copy())
    # cv2.imshow("Mask", watershedMarkers)
    # Phase 4: Filter regions
    # Get the unique region labels and their counts
    unique_labels, label_counts = np.unique(watershedMarkers, return_counts=True)
    regions = {}
    
    # calculate area and center point of each region
    for x in range(len(unique_labels)):
        region_id = unique_labels[x]
        region_area = label_counts[x]
        
        if region_id == -1:  # skip watershed line
            continue

        region_mask = np.uint8(watershedMarkers == region_id)
        
        if region_area < 800:
            continue
        
        y_coords, x_coords = np.where(region_mask)
        center_x = int(np.mean(x_coords))
        center_y = int(np.mean(y_coords))
        
        if center_y < (int(skyimg.shape[0] * 0.8)):
            regions[region_id] = {"area": region_area, "center": (center_x, center_y)}

    # sort the regions based on area and position of center point
    # the higher the position, the most likely the region is sky
    sorted_regions = sorted(regions.items(), key=lambda x: (x[1]["area"], -x[1]["center"][1]), reverse=True)

    # choose the highest rank region that does not touch bottom of sky image as sky
    mask = np.zeros((height, width), np.uint8)
    for region_id, _ in sorted_regions:
        region_mask = np.uint8(watershedMarkers == region_id)
        y_coords, _ = np.where(region_mask)
        if max(y_coords) < (skyimg.shape[0] - 2):
            mask[watershedMarkers == region_id] = 255
        
    return mask


def otsu_contour(skyimg):
    
    # Phase 1: Preprocessing
    smooth_img = preprocessing_image(skyimg)
    
    # Phase 2: Apply otsu method
    _, thrlmg = cv2.threshold(smooth_img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    
    # Phase 3: Find contours
    contours,_ = cv2.findContours(thrlmg, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    
    # Phase 4: Find largest area contour
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Phase 5: Choose largest contours as sky
    # fill sky with white
    mask = thrlmg.copy()
    cv2.drawContours(mask, [largest_contour], -1, (255, 255, 255), cv2.FILLED)
    
    # fill other contour (ground) with black
    for contour in contours:
        if contour is not largest_contour:
            cv2.drawContours(mask, [contour], -1, (0, 0, 0), -1)

    return mask


def optimal_mask(image, ground_truth):
    canny_mask = edge_based_canny(image)
    watershed_mask = region_based_watershed(image)
    otsu_mask = otsu_contour(image)
    
    algo = ["Canny", "Watershed", "Otsu_Contour"]
    masks = [canny_mask, watershed_mask, otsu_mask]
    accuracies = [calculate_accuracy(mask, ground_truth) for mask in masks]
    
    best_idx = np.argmax(accuracies)
    best_mask = masks[best_idx]
    best_accuracy = accuracies[best_idx]
    # print(algo[best_idx])
    
    return best_mask, best_accuracy

    
    
    

    

# input_foldfer_path = "./Dataset/10917/"
# output_folder_path = "./GroundTruth/10917/"
# ans_gt = "./GroundTruth/10917.png"
# ans = cv2.imread(ans_gt,cv2.COLOR_BGR2GRAY)

# get_mean_intensity_dark(input_foldfer_path)
# algorithm2(input_foldfer_path, output_folder_path, ans_gt)

