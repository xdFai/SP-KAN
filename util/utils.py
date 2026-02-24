import torch
import numpy as np
from PIL import Image
import random
import torch.nn as nn
import torch.nn.functional as F
import os
from torch.nn import init
from torchvision import transforms
from util.warmup_scheduler import GradualWarmupScheduler

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


def seed_pytorch(seed=50):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def weights_init_xavier(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1 and classname.find('SplAtConv2d') == -1:
        init.xavier_normal(m.weight.data)


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('BatchNorm') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


class Get_gradient_nopadding(nn.Module):
    def __init__(self):
        super(Get_gradient_nopadding, self).__init__()
        kernel_v = [[0, -1, 0],
                    [0, 0, 0],
                    [0, 1, 0]]
        kernel_h = [[0, 0, 0],
                    [-1, 0, 1],
                    [0, 0, 0]]
        kernel_h = torch.FloatTensor(kernel_h).unsqueeze(0).unsqueeze(0)
        kernel_v = torch.FloatTensor(kernel_v).unsqueeze(0).unsqueeze(0)
        self.weight_h = nn.Parameter(data=kernel_h, requires_grad=False).cuda()
        self.weight_v = nn.Parameter(data=kernel_v, requires_grad=False).cuda()

    def forward(self, x):
        x0 = x[:, 0]
        x0_v = F.conv2d(x0.unsqueeze(1), self.weight_v, padding=1)
        x0_h = F.conv2d(x0.unsqueeze(1), self.weight_h, padding=1)

        x0 = torch.sqrt(torch.pow(x0_v, 2) + torch.pow(x0_h, 2) + 1e-6)

        return x0


def random_crop(img, mask, patch_size, pos_prob=None):
    h, w = img.shape
    if min(h, w) < patch_size:
        img = np.pad(img, ((0, max(h, patch_size) - h), (0, max(w, patch_size) - w)),
                     mode='constant')  # 将不足 256的一边填充至256
        mask = np.pad(mask, ((0, max(h, patch_size) - h), (0, max(w, patch_size) - w)),
                      mode='constant')  # label 与image 进行相同的变换
        h, w = img.shape

    while 1:
        h_start = random.randint(0, h - patch_size)
        h_end = h_start + patch_size
        w_start = random.randint(0, w - patch_size)
        w_end = w_start + patch_size

        img_patch = img[h_start:h_end, w_start:w_end]
        mask_patch = mask[h_start:h_end, w_start:w_end]

        if pos_prob == None or random.random() > pos_prob:
            break
        elif mask_patch.sum() > 0:
            break

    return img_patch, mask_patch


def Normalized(img, img_norm_cfg):
    return (img - img_norm_cfg['mean']) / img_norm_cfg['std']


def Denormalization(img, img_norm_cfg):
    return img * img_norm_cfg['std'] + img_norm_cfg['mean']


def get_img_norm_cfg(dataset_name, dataset_dir):
    if dataset_name == 'NUAA-SIRST':
        img_norm_cfg = dict(mean=101.06385040283203, std=34.619606018066406)
    elif dataset_name == 'NUDT-SIRST':
        img_norm_cfg = dict(mean=107.80905151367188, std=33.02274703979492)
    elif dataset_name == 'IRSTD-1K':
        img_norm_cfg = dict(mean=87.4661865234375, std=39.71953201293945)
    elif dataset_name == 'SIRST3':
        img_norm_cfg = dict(mean=101.06385040283203, std=34.619606018066406)
    elif dataset_name == 'NUDT-SIRST-Sea':
        img_norm_cfg = dict(mean=43.62403869628906, std=18.91838264465332)
    elif dataset_name == 'SIRST4':
        img_norm_cfg = dict(mean=62.10432052612305, std=23.96998405456543)
    elif dataset_name == 'IRDST-real':
        img_norm_cfg = {'mean': 101.54053497314453, 'std': 56.49856185913086}
    else:
        with open(dataset_dir + '/' + dataset_name + '/img_idx/train_' + dataset_name + '.txt', 'r') as f:
            train_list = f.read().splitlines()
        with open(dataset_dir + '/' + dataset_name + '/img_idx/test_' + dataset_name + '.txt', 'r') as f:
            test_list = f.read().splitlines()
        img_list = train_list + test_list
        img_dir = dataset_dir + '/' + dataset_name + '/images/'
        mean_list = []
        std_list = []
        for img_pth in img_list:
            try:
                img = Image.open((img_dir + img_pth).replace('//', '/') + '.png').convert('I')
            except:
                try:
                    img = Image.open((img_dir + img_pth).replace('//', '/') + '.jpg').convert('I')
                except:
                    img = Image.open((img_dir + img_pth).replace('//', '/') + '.bmp').convert('I')
            img = np.array(img, dtype=np.float32)
            mean_list.append(img.mean())
            std_list.append(img.std())
        img_norm_cfg = dict(mean=float(np.array(mean_list).mean()), std=float(np.array(std_list).mean()))
    return img_norm_cfg


def get_optimizer(net, optimizer_name, scheduler_name, optimizer_settings, scheduler_settings):
    if optimizer_name == 'Adam':
        optimizer = torch.optim.Adam(net.parameters(), lr=optimizer_settings['lr'])
    elif optimizer_name == 'Adagrad':
        optimizer = torch.optim.Adagrad(net.parameters(), lr=optimizer_settings['lr'])
    elif optimizer_name == 'SGD':
        optimizer = torch.optim.SGD(net.parameters(), lr=optimizer_settings['lr'],
                                    momentum=scheduler_settings['momentum'],
                                    weight_decay=scheduler_settings['weight_decay'])
    elif optimizer_name == 'AdamW':
        optimizer = torch.optim.AdamW(net.parameters(), lr=optimizer_settings['lr'], betas=optimizer_settings['betas'],
                                      eps=optimizer_settings['eps'], weight_decay=optimizer_settings['weight_decay'],
                                      amsgrad=optimizer_settings['amsgrad'])

    if scheduler_name == 'MultiStepLR':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=scheduler_settings['step'],
                                                         gamma=scheduler_settings['gamma'])
    elif scheduler_name == 'DNACosineAnnealingLR':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=scheduler_settings['epochs'],
                                                               eta_min=scheduler_settings['min_lr'])
    elif scheduler_name == 'CosineAnnealingLR':
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=scheduler_settings['T_max'],
        #                                                        eta_min=scheduler_settings['eta_min'],
        #                                                        last_epoch=scheduler_settings['last_epoch'])
        warmup_epochs = 10
        scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=scheduler_settings[
                                                                                           'epochs'] - warmup_epochs,
                                                                      eta_min=scheduler_settings['eta_min'])
        scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=warmup_epochs,
                                           after_scheduler=scheduler_cosine)

    return optimizer, scheduler


def PadImg(img, times=64):
    h, w = img.shape
    if not h % times == 0:
        img = np.pad(img, ((0, (h // times + 1) * times - h), (0, 0)), mode='constant')
    if not w % times == 0:
        img = np.pad(img, ((0, 0), (0, (w // times + 1) * times - w)), mode='constant')
    return img

def PadImg_len(oldh, oldw,long_side_length):
    scale = long_side_length * 1.0 / max(oldh, oldw)
    newh, neww = oldh * scale, oldw * scale
    neww = int(neww + 0.5)
    newh = int(newh + 0.5)
    return (newh, neww)


def preprocess(eval_size, x):
    # Pad
    h, w = x.shape[-2:]
    padh = eval_size - h
    padw = eval_size - w
    x = F.pad(x, (0, padw, 0, padh))  # 填充成 1024* 1024
    return x

def rot_patch_32_numpy(image_array, coordinate, k):
    """
    对三维numpy数组形式的单张图像进行裁剪、翻转和旋转操作。

    Args:
        image_array (numpy.ndarray): 输入的图像数组，形状为 (channels, height, width)
        coordinate (tuple): 中心点坐标 (x, y)
        k (int): 旋转的次数 (90度的倍数)

    Returns:
        numpy.ndarray: 处理后的图像数组，形状与输入相同
    """
    channels, height, width = image_array.shape

    # 复制输入数组，用于存储修改后的图像
    output_array = np.copy(image_array)

    # 根据坐标计算裁剪区域
    if (coordinate[0] >= 32 and coordinate[0] <= 224) and (coordinate[1] >= 32 and coordinate[1] <= 224):
        a = (coordinate[0] - 16, coordinate[1] - 16, coordinate[0] + 16, coordinate[1] + 16)
    elif coordinate[0] < 32 and (coordinate[1] >= 32 and coordinate[1] <= 224):
        a = (0, coordinate[1] - 16, 32, coordinate[1] + 16)
    elif coordinate[0] > 224 and (coordinate[1] >= 32 and coordinate[1] <= 224):
        a = (224, coordinate[1] - 16, 256, coordinate[1] + 16)
    elif (coordinate[0] >= 32 and coordinate[0] <= 224) and coordinate[1] < 32:
        a = (coordinate[0] - 16, 0, coordinate[0] + 16, 32)
    elif (coordinate[0] >= 32 and coordinate[0] <= 224) and coordinate[1] > 224:
        a = (coordinate[0] - 16, 224, coordinate[0] + 16, 256)
    elif coordinate[0] < 32 and coordinate[1] < 32:
        a = (0, 0, 32, 32)
    elif coordinate[0] > 224 and coordinate[1] > 224:
        a = (224, 224, 256, 256)
    elif coordinate[0] < 32 and coordinate[1] > 224:
        a = (0, 224, 32, 256)
    elif coordinate[1] < 32 and coordinate[0] > 224:
        a = (224, 0, 256, 32)

    # 裁剪图像（注意数组的裁剪是按 [C, H, W] 排列）
    patch = image_array[:, a[1]:a[3], a[0]:a[2]]  # 裁剪图像，顺序是 [C, H, W]

    # 进行水平翻转和旋转
    # patch = np.flip(patch, axis=2)  # flip(2) 在宽度维度上做水平翻转
    patch = np.rot90(patch, k, axes=(1, 2))  # rot90(k) 对 patch 进行 90° * k 顺时针旋转

    # 将处理后的 patch 放回原图像
    output_array[:, a[1]:a[3], a[0]:a[2]] = patch  # 将翻转和旋转后的 patch 放回原图像

    return output_array


def rot_patch_3_numpy(image_array, coordinate, k):
    output_array = np.copy(image_array)

    # Create patch
    if (coordinate[0] >= 1 and coordinate[0] <= 255) and (coordinate[1] >= 1 and coordinate[1] <= 255):
        a = (coordinate[0] - 1, coordinate[1] - 1, coordinate[0] + 1, coordinate[1] + 1)
    elif coordinate[0] < 1 and (coordinate[1] >= 1 and coordinate[1] <= 255):
        a = (0, coordinate[1] - 1, 2, coordinate[1] + 1)
    elif coordinate[0] > 255 and (coordinate[1] >= 1 and coordinate[1] <= 255):
        a = (254, coordinate[1] - 1, 256, coordinate[1] + 1)
    elif (coordinate[0] >= 1 and coordinate[0] <= 255) and coordinate[1] < 1:
        a = (coordinate[0] - 1, 0, coordinate[0] + 1, 2)
    elif (coordinate[0] >= 1 and coordinate[0] <= 255) and coordinate[1] > 255:
        a = (coordinate[0] - 1, 254, coordinate[0] + 1, 256)
    elif coordinate[0] < 1 and coordinate[1] < 1:
        a = (0, 0, 2, 2)
    elif coordinate[0] > 255 and coordinate[1] > 255:
        a = (254, 254, 256, 256)
    elif coordinate[0] < 1 and coordinate[1] > 255:
        a = (0, 254, 2, 256)
    elif coordinate[1] < 1 and coordinate[0] > 255:
        a = (254, 0, 256, 2)

    # 裁剪图像（注意数组的裁剪是按 [C, H, W] 排列）
    patch = image_array[:, a[1]:a[3], a[0]:a[2]]  # 裁剪图像，顺序是 [C, H, W]

    # 进行水平翻转和旋转
    # patch = np.flip(patch, axis=2)  # flip(2) 在宽度维度上做水平翻转
    patch = np.rot90(patch, k, axes=(1, 2))  # rot90(k) 对 patch 进行 90° * k 顺时针旋转

    # 将处理后的 patch 放回原图像
    output_array[:, a[1]:a[3], a[0]:a[2]] = patch  # 将翻转和旋转后的 patch 放回原图像

    return output_array


def rot_patch_2_numpy(image_array, coordinate, k, q):
    """
    对图像 image_array 中以 coordinate 为中心、边长为 2q+1 的区域进行 k 次 90° 顺时针旋转，并填回原图。

    Parameters:
        image_array: 2D numpy array (H, W)
        coordinate: tuple (x, y) 表示旋转中心坐标（注意：0-based）
        k: int, 表示旋转次数（每次90度）
        q: int, 表示旋转区域向外扩张的像素数

    Returns:
        output_array: 增强后的图像副本
    """

    h, w = image_array.shape
    x, y = coordinate
    output_array = np.copy(image_array)

    x1 = max(0, x - q)
    y1 = max(0, y - q)
    x2 = min(h, x + q + 1)
    y2 = min(w, y + q + 1)

    if x1 == 0:
        x2 = x1 + q + 2
    if y1 == 0:
        y2 = y1 + q + 2
    if x2 == h:
        x1 = h - q - 2
    if y2 == w:
        y1 = w - q - 2
    patch = image_array[x1:x2, y1:y2]

    # 进行 k 次 90 度顺时针旋转（np.rot90 默认是逆时针，需要 k=4-k 次）
    k_mod = k % 4
    if k_mod != 0:
        patch_rot = np.rot90(patch, k=4 - k_mod)
    else:
        patch_rot = patch
    # 将旋转后的 patch 放回原图
    output_array[x1:x2, y1:y2] = patch_rot

    return output_array
