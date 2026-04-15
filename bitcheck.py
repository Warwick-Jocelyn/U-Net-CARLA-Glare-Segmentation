from PIL import Image
import numpy as np

def get_image_info(image_path):
  # Open the image
  with Image.open(image_path) as img:
    # Convert image to numpy array
    img_array = np.array(img)

    # Get mode of the image to determine bit depth
    mode = img.mode
    if mode in ['1']:  # 1 bit per pixel
      bit_depth = 1
    elif mode in ['L', 'P']:  # 8 bits per pixel
      bit_depth = 8
    elif mode in ['I;16', 'I;16B', 'I;16L']:  # 16 bits per pixel
      bit_depth = 16
    elif mode in ['RGB', 'YCbCr']:  # 24 bits per pixel (8 bits for each of 3 channels)
      bit_depth = 24
    elif mode in ['RGBA', 'CMYK']:  # 32 bits per pixel (8 bits for each of 4 channels)
      bit_depth = 32
    else:
      bit_depth = 'Unknown'

    # Determine channel number based on the image array shape
    if len(img_array.shape) == 2:  # Grayscale image
      channels = 1
    elif len(img_array.shape) == 3:  # Color image
      channels = img_array.shape[2]
    else:
      channels = 'Unknown'

    # Calculate max, min, and mean
    max_val = img_array.max()
    min_val = img_array.min()
    mean_val = img_array.mean()
    # 找到图像中出现的所有类ID
    msk = np.unique(img_array)
    car_num = np.sum(img_array == 26)
    # 获取图像的高度和宽度
    height, width = img_array.shape # 

    # 计算中间位置的索引
    mid_y = height // 2
    mid_x = width // 2

    # # 打印中间位置的数值
    # print(f"The value at the center of the image array is: {img_array[mid_y, mid_x]}")

    print(img_array[height // 2][width // 2])
    print(img_array[(height // 2)+2][(width // 2)+2])

    print(img_array[511][100])
    print(img_array[300][100])

    # # 打印最上方中间位置的数值
    # top_middle_value = img_array[0, width-1]
    # print(f"The value at the top middle of the image array is: {top_middle_value}")

    # # 打印最左方中间位置的数值
    # top_middle_value = img_array[mid_y, 0]
    # print(f"The value at the left middle of the image array is: {top_middle_value}")


    # # 打印最右方中间位置的数值
    # top_middle_value = img_array[mid_y, 0]
    # print(f"The value at the left middle of the image array is: {top_middle_value}")

    return bit_depth, channels, max_val, min_val, mean_val,msk,car_num 

# Example usage
# image_path = '/home/wang3_y_WMGDS.WMG.WARWICK.AC.UK/Boda/lyft-udacity-challenge/dataA/CameraSeg/0.png'
# bit_depth, channels, max_val, min_val, mean_val = get_image_info(image_path)
# print("Example dataset from lyft:")
# print(f"The bit depth of the image is: {bit_depth}")
# print(f"The number of channels in the image is: {channels}")
# print(f"Maximum value in the image: {max_val}")
# print(f"Minimum value in the image: {min_val}")
# print(f"Average value in the image: {mean_val}")

# image_path1 = '/home/wang3_y_WMGDS.WMG.WARWICK.AC.UK/Boda/Bayer_Glare/Glare_example/CameraRGB/002079.png'/home/wang3_y_WMGDS.WMG.WARWICK.AC.UK/Code/OneFormer_ACDC_eval_with_TTA/datasets/cityscapes_ori/gtFine/cityscapes_panoptic_val/lindau_000000_000019_gtFine_panoptic.png
image_path1 = '/home/wang3_y_WMGDS.WMG.WARWICK.AC.UK/Code/OneFormer_ACDC_eval_with_TTA/Jocelyn/demo/lindau_000000_000019_gtFine_labelIds.png'

bit_depth1, channels1, max_val1, min_val1, mean_val1,msk,car_num  = get_image_info(image_path1)

 
print(f"Detected class IDs: {msk.tolist()}")
print(f"Detected class Cars: {car_num}")

print("Example dataset from Boda Glare redonly data:")
print(f"The bit depth of the image is: {bit_depth1}")
print(f"The number of channels in the image is: {channels1}")
print(f"Maximum value in the image: {max_val1}")
print(f"Minimum value in the image: {min_val1}")
print(f"Average value in the image: {mean_val1}")