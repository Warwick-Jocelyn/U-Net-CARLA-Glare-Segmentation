# imports cell
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler
from torch.cuda.amp import autocast
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchsummary import summary
import cv2
import numpy as np
import torch
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from copy import deepcopy
from tqdm import tqdm
import random
import warnings
import gc

# to create a tensor on the gpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

warnings.filterwarnings('ignore')
image_path = ["/home/wang3_y_WMGDS.WMG.WARWICK.AC.UK/Boda/Bayer_Glare/Glare_example/CameraRGB/" ]
mask_path = ["/home/wang3_y_WMGDS.WMG.WARWICK.AC.UK/Boda/Bayer_Glare/Glare_example/CameraSeg/"]

def list_image_paths(directory_paths):
    image_paths = []
    for i in range(len(directory_paths)):  # 5 folders
        image_filenames = os.listdir(directory_paths[i])

        for image_filename in image_filenames:
            image_paths.append(directory_paths[i] + image_filename)
    return image_paths

image_list = list_image_paths(image_path)
mask_list = list_image_paths(mask_path)

labels = {'Unlabeled': 0,
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
          'Traffic sign': 12}

colors_platte = np.array([[0, 0, 0],  # Unlabeled
                           [70, 70, 70],  # Building
                           [100, 40, 40],  # Fence
                           [55, 90, 80],  # Other -> Everything that does not belong to any other category.
                           [220, 20, 60],  # Pedestrian
                           [153, 153, 153],  # Pole
                           [157, 234, 50],  # Roadline
                           [128, 64, 128],  # Road
                           [244, 35, 232],  # Sidewalk
                           [107, 142, 35],  # Vegetation
                           [0, 0, 142],  # Car
                           [102, 102, 156],  # Wall
                           [220, 220, 0],  # Traffic sign
                           # not used in current model
                           [70, 130, 180],  # Sky
                           [81, 0, 81],  # Ground
                           [150, 100, 100],  # Bridge
                           [230, 150, 140],  # RailTrack
                           [180, 165, 180],  # GuardRail
                           [250, 170, 30],  # TrafficLight
                           [110, 190, 160],  # Static
                           [170, 120, 50],  # Dynamic
                           [45, 60, 150],  # Water
                           [145, 170, 100]] )# Terrain

IMG_HEIGHT = 600
IMG_WIDTH = 800
RESIZE_HEIGHT = 240
RESIZE_WIDTH = 320
BATCH_SIZE = 10

# 映射规则
label_mapping = {
    1: 7,
    2: 8,
    3: 1,
    4: 11,
    5: 2,
    6: 5,
    7: 12,
    9: 9,
    11: 0,
    12: 4,
    13: 6,
    14: 10,
    0: 3,
    8: 3,
    10: 3, 
    15: 3,
    16: 3, 
    17: 3,
    18: 3,
    19: 3,
    20: 3,
    21: 3, 
    22: 3,
    23: 3, 
    24: 3,
    25: 3,
}

images = []
for image in image_list:
    img = cv2.cvtColor(cv2.imread(image), cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (RESIZE_WIDTH, RESIZE_HEIGHT), interpolation=cv2.INTER_NEAREST)
    img = img.astype("float32") / 255.0
    images.append(img)

masks = []
for mask in mask_list:
    msk = cv2.cvtColor(cv2.imread(mask), cv2.COLOR_BGR2RGB)
    msk = cv2.resize(msk, (RESIZE_WIDTH, RESIZE_HEIGHT), interpolation=cv2.INTER_NEAREST)
    msk = np.max(msk, axis=-1)
    # print(msk)
    # 映射掩码值
    remapped_msk = np.zeros_like(msk)
    for old_label, new_label in label_mapping.items():
        remapped_msk[msk == old_label] = new_label
    # print(remapped_msk)
    masks.append(remapped_msk)

masks = np.array(masks)
images = np.array(images).transpose((0, 3, 1, 2))

print(f'mages.shape: {images.shape}')
print(f'masks.shape: {masks.shape}')

# 找到最小值和最大值
min_value = np.min(masks)
max_value = np.max(masks)
# 输出结果
print(f'Minimum value in the matrix: {min_value}')
print(f'Maximum value in the matrix: {max_value}')

x_train_val, x_test, y_train_val, y_test = train_test_split(images, masks, test_size=0.1,
                                                            random_state=42)
x_train, x_val, y_train, y_val = train_test_split(x_train_val, y_train_val, test_size=0.1, random_state=42)
print(x_train.shape, y_train.shape, x_val.shape, y_val.shape, x_test.shape, y_test.shape)

class SegmentationDataset(Dataset):
    def __init__(self, _images, _masks, transform=None):
        self.images = _images
        self.masks = _masks
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        _img = self.images[index]
        _mask = self.masks[index]
        if self.transform is not None:
            image_tensor = self.transform(_img).to(device)
        else:
            image_tensor = torch.tensor(_img, dtype=torch.float32, device=device)
        mask_tensor = torch.tensor(_mask, dtype=torch.long, device=device)
        return image_tensor, mask_tensor
    
valDataset = SegmentationDataset(x_val, y_val)
val_dataloader = DataLoader(valDataset, batch_size=BATCH_SIZE, shuffle=True)
testDataset = SegmentationDataset(x_test, y_test)
test_dataloader = DataLoader(testDataset, batch_size=BATCH_SIZE, shuffle=True)


class DownSamplingBlock(nn.Module):
    def __init__(self, inChannels, n_filters, dropout_prob=0.0, max_pooling=True):
        super().__init__()
        self.dropout_prob = dropout_prob
        self.max_pooling = max_pooling
        self.conv1 = nn.Conv2d(in_channels=inChannels, out_channels=n_filters, kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(n_filters)
        self.conv2 = nn.Conv2d(in_channels=n_filters, out_channels=n_filters, kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(n_filters)
        if dropout_prob > 0.0:
            self.drop = nn.Dropout(p=dropout_prob)
        if max_pooling:
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)

    def forward(self, x):
        conv = F.relu(self.bn1(self.conv1(x)))
        conv = F.relu(self.bn2(self.conv2(conv)))
        if self.dropout_prob > 0.0:
            conv = self.drop(conv)
        if self.max_pooling:
            next_layer = self.pool(conv)
        else:
            next_layer = conv
        skip_connection = conv
        return next_layer, skip_connection
    
# Decoder (Up-sampling Block)
# Takes the arguments expansive_input (which is the input tensor from the previous layer) and contractive_input (the input tensor from the previous skip layer)
class UpSamplingBlock(nn.Module):
    def __init__(self, inChannels, n_filters):
        super().__init__()
        # out = (in-1)*s -2p + d(k-1) + op + 1
        self.upSample = nn.ConvTranspose2d(in_channels=inChannels, out_channels=n_filters, kernel_size=2, stride=2)
        self.conv1 = nn.Conv2d(in_channels=inChannels, out_channels=n_filters, kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(n_filters)
        self.conv2 = nn.Conv2d(in_channels=n_filters, out_channels=n_filters, kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(n_filters)

    def forward(self, expansive_input, contractive_input):
        up = self.upSample(expansive_input)
        if up.shape != contractive_input.shape:
            # crop the contractive input to fit the Up sampled expansive input
            (B, C, H, W) = up.shape
            contractive_input = transforms.CenterCrop([H, W])(contractive_input)
        merge = torch.cat([up, contractive_input], dim=1)
        conv = F.relu(self.bn1(self.conv1(merge)))
        conv = F.relu(self.bn2(self.conv2(conv)))
        return conv

class UNet(nn.Module):
    def __init__(self, inChannels=3, n_filters=64, n_classes=13):
        super().__init__()
        encChannels = (inChannels, n_filters, 128, 256, 512, 1024)
        decChannels = (1024, 512, 256, 128, n_filters)
        self.encBlocks = nn.ModuleList([
            DownSamplingBlock(encChannels[i], encChannels[i + 1]) for i in range(len(encChannels) - 2)
        ])
        self.lastEncBlocks = DownSamplingBlock(encChannels[len(encChannels) - 2], encChannels[len(encChannels) - 1],
                                               dropout_prob=0.3, max_pooling=None)
        self.decBlocks = nn.ModuleList([
            UpSamplingBlock(decChannels[i], decChannels[i + 1]) for i in range(len(decChannels) - 1)
        ])
        self.conv = nn.Conv2d(in_channels=n_filters, out_channels=n_classes, kernel_size=1, stride=1)
    def forward(self, x):
        skipConnections = []
        # Contracting Path (encoding)
        next_layer = x
        for enc_block in self.encBlocks:
            next_layer, skip_connection = enc_block(next_layer)
            skipConnections.append(skip_connection)
        next_layer, _ = self.lastEncBlocks(next_layer)
        # Expanding Path (decoding)
        for dec_block in self.decBlocks:
            next_layer = dec_block(next_layer, skipConnections.pop())
        conv = self.conv(next_layer)
        return conv

def pixel_accuracy(output, _mask):
    with torch.no_grad():
        output = torch.argmax(F.log_softmax(output, dim=1), dim=1) # first output: [batch_size, num_classes, height, width], 
        # then 形状为 [batch_size, height, width] 的张量，其中的每个值代表了对应位置的预测类别
        correct = torch.eq(output, _mask).int()
        accuracy = float(correct.sum()) / float(correct.numel())
    return accuracy

def test_model(_model, dataloader, Ncrop=False):
    _model.eval()
    criterion = nn.CrossEntropyLoss()
    accuracy = 0.0
    total_loss, total_acc = 0.0, 0.0
    with torch.no_grad():
        for (data, label) in dataloader:
            if Ncrop:
                # fuse crops and batch size
                bs, ncrops, c, h, w = data.shape
                data = data.view(-1, c, h, w)
                # forward
                output = model(data)
                # combine results across the crops
                output = output.view(bs, ncrops, -1)
                output = torch.sum(output, dim=1) / ncrops
            else:
                output = _model(data)
            # compute loss
            _loss = criterion(output, label)
            total_loss += _loss.item()
            accuracy += pixel_accuracy(output, label)
    total_loss /= len(dataloader)
    total_acc = accuracy / len(dataloader) * 100.
    return total_acc, total_loss

# model_path = "./Unet_model"
# torch.cuda.empty_cache()
# gc.collect()

model = UNet().to(device)


print(' ')
print('=================================START==================================')
print(' ')
pretrained_dict = torch.load('/home/wang3_y_WMGDS.WMG.WARWICK.AC.UK/Boda/Bayer_Glare/Unet_epoch_100_checkpoint.pth')
model_dict = model.state_dict() 
pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict} 
model_dict.update(pretrained_dict)  
model.load_state_dict(model_dict)

test_accuracy, test_loss = test_model(model, test_dataloader)
print('Test loss: ', test_loss, '- Test Accuracy: ', test_accuracy, '%')

print(' ')
print('=================================TEST ENDS==================================')