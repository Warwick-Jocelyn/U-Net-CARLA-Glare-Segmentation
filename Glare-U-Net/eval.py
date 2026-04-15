import numpy as np
from PIL import Image
from os.path import join
import re
import os

colors_palette = np.array([
    [0, 0, 0],         # Unlabeled
    [70, 70, 70],      # Building
    [100, 40, 40],     # Fence
    [55, 90, 80],      # Other
    [220, 20, 60],     # Pedestrian
    [153, 153, 153],   # Pole
    [157, 234, 50],    # Roadline
    [128, 64, 128],    # Road
    [244, 35, 232],    # Sidewalk
    [107, 142, 35],    # Vegetation
    [0, 0, 142],       # Car
    [102, 102, 156],   # Wall
    [220, 220, 0]      # Traffic sign
])

labels = {
    'Unlabeled': 0,
    'Building': 1,
    'Fence': 2,
    'Other': 3,
    'Pedestrian': 4,
    'Pole': 5,
    'Roadline': 6,
    'Road': 7,
    'Sidewalk': 8,
    'Vegetation': 9,
    'Car': 10,
    'Wall': 11,
    'Traffic sign': 12
}

def rgb_to_class(arr, colors_palette):
    class_map = np.zeros((arr.shape[0], arr.shape[1]), dtype=int)
    for i, color in enumerate(colors_palette):
        mask = np.all(arr == color, axis=-1)
        class_map[mask] = i
    return class_map

def fast_hist(a, b, n):
    k = (a >= 0) & (a < n)
    return np.bincount(n * a[k].astype(int) + b[k], minlength=n ** 2).reshape(n, n)

def per_class_iu(hist):
    return np.diag(hist) / np.maximum((hist.sum(1) + hist.sum(0) - np.diag(hist)), 1)

def per_class_PA(hist):
    return np.diag(hist) / np.maximum(hist.sum(1), 1)

def sort_key(filename):
    numbers = re.findall(r'\d+', filename)
    return int(numbers[0]) if numbers else -1

def compute_mIoU(gt_dir, pred_dir, num_classes, name_classes):
    print('Num classes', num_classes)

    hist = np.zeros((num_classes, num_classes))
    gt_list = os.listdir(gt_dir)
    gt_list.sort(key=sort_key)
    pred_list = os.listdir(pred_dir)
    pred_list.sort(key=sort_key)
    gt_imgs = [join(gt_dir, x) for x in gt_list]
    pred_imgs = [join(pred_dir, x) for x in pred_list[:154]]
    total_num = len(gt_imgs)

    for ind in range(total_num):
        img = Image.open(pred_imgs[ind])
        pred = rgb_to_class(np.array(img), colors_palette)

        img1 = Image.open(gt_imgs[ind])
        label = rgb_to_class(np.array(img1), colors_palette)

        if len(label.flatten()) != len(pred.flatten()):
            print(
                'Skipping: len(gt) = {:d}, len(pred) = {:d}, {:s}, {:s}'.format(
                    len(label.flatten()), len(pred.flatten()), gt_imgs[ind],
                    pred_imgs[ind]))
            continue

        hist += fast_hist(label.flatten(), pred.flatten(), num_classes)

        if ind > 0 and ind % 10 == 0:
            print('{:d} / {:d}: mIou-{:0.3f}; mPA-{:0.2f}'.format(ind, len(gt_imgs),
                                                                    100 * np.nanmean(per_class_iu(hist)),
                                                                    100 * np.nanmean(per_class_PA(hist))))
    mIoUs = per_class_iu(hist)
    mPA = per_class_PA(hist)
    print(">>>>>>>iIOU Done!")

    for ind_class in range(num_classes):
        print('===>' + name_classes[ind_class] + ':\tmIoU:' + str(round(mIoUs[ind_class] * 100, 2)) + '; mPA:' + str(
            round(mPA[ind_class] * 100, 2)))

    print('===> mIoU: ' + str(round(np.nanmean(mIoUs) * 100, 2)) + '; mPA: ' + str(round(np.nanmean(mPA) * 100, 2)))
    return mIoUs

# Directories for saving masks and predictions
masks_dir = '/home/wang3_y_WMGDS.WMG.WARWICK.AC.UK/Boda/Glare-U-Net/masks'
predictions_dir = '/home/wang3_y_WMGDS.WMG.WARWICK.AC.UK/Boda/Glare-U-Net/predictions'

compute_mIoU(masks_dir, predictions_dir , 13, ["Unlabeled", "Building", "Fence", "Other", "Pedestrian", "Pole", "Roadline", "Road", "Sidewalk", "Vegetation", "Car", "Wall", "Traffic sign"])
