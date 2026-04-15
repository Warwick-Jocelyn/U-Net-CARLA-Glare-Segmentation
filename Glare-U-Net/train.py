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

print(image_path)
print(mask_path)

def list_image_paths(directory_paths):
    image_paths = []
    for i in range(len(directory_paths)):  # 5 folders
        image_filenames = os.listdir(directory_paths[i])

        for image_filename in image_filenames:
            image_paths.append(directory_paths[i] + image_filename)
    return image_paths

image_list = list_image_paths(image_path)
mask_list = list_image_paths(mask_path)

print(image_list[0:10])
print(mask_list[0:10])

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
    

trainDataset = SegmentationDataset(x_train, y_train)
train_dataloader = DataLoader(trainDataset, batch_size=BATCH_SIZE, shuffle=True)
valDataset = SegmentationDataset(x_val, y_val)
val_dataloader = DataLoader(valDataset, batch_size=BATCH_SIZE, shuffle=True)
testDataset = SegmentationDataset(x_test, y_test)
test_dataloader = DataLoader(testDataset, batch_size=BATCH_SIZE, shuffle=True)

# Encoder (Down-sampling Block)
# The encoder is a stack of various conv blocks
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

def train_model(_model, dataloader, _optimizer, _scaler, Ncrop=False):
    _model.train()

    criterion = nn.CrossEntropyLoss()

    accuracy = 0.0
    total_loss, total_acc = 0.0, 0.0

    for (data, label) in dataloader:

        with autocast():

            if Ncrop:
                # fuse crops and batch size
                bs, ncrops, c, h, w = data.shape
                data = data.view(-1, c, h, w)

                # repeat labels ncrops times
                label = torch.repeat_interleave(label, repeats=ncrops, dim=0)

            # reset gradients (it will accumulate gradients otherwise)
            _optimizer.zero_grad()

            # forward pass
            output = _model(data)

            # compute loss
            _loss = criterion(output, label)

            # scale the loss then backward propagation dl/dw -> gradients， why???
            _scaler.scale(_loss).backward()

            # update weights
            _scaler.step(optimizer)
            _scaler.update()

            ######################################

            total_loss += _loss.item()

            accuracy += pixel_accuracy(output, label)

    total_loss /= len(dataloader)
    total_acc = accuracy / len(dataloader) * 100.

    return total_acc, total_loss

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

# 保存的是某些参数还是整体？而且这个怎么看？save之后怎么掉用
def save_model(_model, _optimizer, _training_data, _path):
    info_dict = {
        'net_state': _model.state_dict(),
        'optimizer_state': _optimizer.state_dict(),
        'training_data': _training_data
    }
    torch.save(info_dict, _path + ".pth")

model_path = "./Unet_model"
torch.cuda.empty_cache()
gc.collect()

model = UNet().to(device)
print(summary(model, (3, RESIZE_WIDTH, RESIZE_HEIGHT), batch_size=BATCH_SIZE))

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5, amsgrad=True)
scaler = GradScaler()
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=10, verbose=True)

training_data = {"train_loss": [], "train_accuracy": [], "val_loss": [], "val_acc": []}
model_loader = torch.load("/home/wang3_y_WMGDS.WMG.WARWICK.AC.UK/Boda/Unet_model.pth")


EPOCH = 100
best_acc = 0  # Initialize best accuracy to 0
model_path = "./model_"  # Base path for model saving
training_data = {
    "train_loss": [],
    "train_accuracy": [],
    "val_loss": [],
    "val_acc": []
}

def save_model(model, optimizer, training_data, path):
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'training_data': training_data
    }, path)

# Train and validate model
loop = tqdm(range(EPOCH), desc="Epoch")

for e in loop:
    train_accuracy, train_loss = train_model(model, train_dataloader, optimizer, scaler)
    val_accuracy, val_loss = test_model(model, val_dataloader)

    # Decay Learning Rate
    scheduler.step(val_loss)

    training_data["train_loss"].append(train_loss)
    training_data["train_accuracy"].append(train_accuracy)
    training_data["val_loss"].append(val_loss)
    training_data["val_acc"].append(val_accuracy)

    # save best model so far
    if val_accuracy > best_acc:
        best_acc = val_accuracy
        torch.save(model.state_dict(), 'final.pth')
        # save_model(model, optimizer, training_data, "final.pth")
    

    if (e + 1) % 2 == 0 :
        # Print training and validation results
        print(f"Epoch {e + 1}/{EPOCH} - Training_loss: {train_loss:.4f} - Training_Acc: {train_accuracy:.4f} % - Val_loss: {val_loss:.4f} - Val_Acc: {val_accuracy:.4f} % - Best_Val_Acc: {best_acc:.4f} %")

        # Update tqdm loop
        loop.set_postfix_str(
            "Training_loss: {:.4f} - Training_Acc: {:.4f} % - Val_loss: {:.4f} - Val_Acc: {:.4f} %".format(
                train_loss,
                train_accuracy,
                val_loss,
                val_accuracy))

        print('Learning rate:', optimizer.state_dict()['param_groups'][0]['lr'])

    # Checkpoint
    if (e + 1) % 2 == 0:
        path = f"./Unet_epoch_{e + 1}_checkpoint.pth"
        torch.save(model.state_dict(), path)
        # save_model(model, optimizer, training_data, path)

plt.figure(figsize=(12, 10))
plt.plot(training_data["train_accuracy"], '-o')
plt.plot(training_data["val_acc"], '-o')
plt.xlabel('epoch', size=14)
plt.ylabel('accuracy', size=14)
plt.legend(['Train', 'Valid'])
plt.title('Train vs Valid Accuracy', size=20)

# 保存图像到当前文件夹并命名为 accuracy.png
output_path = os.path.join(os.getcwd(), 'accuracy.png')
plt.savefig(output_path)

# 关闭图形以释放内存
plt.close()

print(f'The accuracy plot has been saved to {output_path}')

plt.figure(figsize=(12, 10))
plt.plot(training_data["train_loss"], '-o')
plt.plot(training_data["val_loss"], '-o')
plt.xlabel('epoch', size=14)
plt.ylabel('losses', size=14)
plt.legend(['Train', 'Valid'])
plt.title('Train vs Valid Losses', size=20)


# 保存图像到当前文件夹并命名为 losses.png
output_path1 = os.path.join(os.getcwd(), 'losses.png')
plt.savefig(output_path1)

# 关闭图形以释放内存
plt.close()

print(f'The losses plot has been saved to {output_path1}')