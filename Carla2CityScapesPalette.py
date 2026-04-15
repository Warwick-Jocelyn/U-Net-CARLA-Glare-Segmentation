import os
import cv2
import numpy as np

# Input and output folders
input_folder = '/home/wang3_y_WMGDS.WMG.WARWICK.AC.UK/Code/U-Net-Cityscapes/try/input'
output_folder = '/home/wang3_y_WMGDS.WMG.WARWICK.AC.UK/Code/U-Net-Cityscapes/try/result'

# Create the output folder if it doesn't exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Tag to color mapping
# TrainId to color mapping
trainId_to_color = {
    0:  [128, 64,128],   # road
    1:  [244, 35,232],   # sidewalk
    2:  [70, 70, 70],   # building
    3:  [102,102,156],   # wall
    4:  [190,153,153],   # fence
    5:  [153,153,153],   # pole
    6:  [250,170, 30],   # traffic light
    7:  [220,220,  0],   # traffic sign
    8:  [107,142, 35],   # vegetation
    9:  [152,251,152],   # terrain
    10: [70,130,180],   # sky
    11: [220, 20, 60],   # person
    12: [255,  0,  0],   # rider
    13: [0,0,142],   # car
    14: [0,  0, 70],   # truck
    15: [0, 60,100],   # bus
    16: [0, 80,100],   # train
    17: [0,  0,230],   # motorcycle
    18: [119, 11, 32]    # bicycle
}


# Iterate through each file in the input folder
for filename in os.listdir(input_folder):
    # Make sure the file is an image file
    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
        # Load the image
        image_path = os.path.join(input_folder, filename)
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Load as grayscale (8-bit class ID map)

        # Create an empty image for the result with 3 channels
        result = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)

        # Iterate through each tag and apply the mapping
        for tag, color in tag_mapping.items():
            # Find pixels with the specified value (tag)
            pixels_with_tag = (image == tag)

            # Set the color for these pixels
            result[pixels_with_tag] = color

        # Save the result in the output folder
        output_path = os.path.join(output_folder, f'colorful_{filename}')
        cv2.imwrite(output_path, result)

print("Processing complete.")
