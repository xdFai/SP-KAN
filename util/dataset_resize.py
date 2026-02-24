from util.utils import *
from torchvision.transforms.functional import InterpolationMode
import os
from torchvision.transforms.functional import resize, to_pil_image  # type: ignore
from torch.utils.data import Dataset
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'



class TrainSetLoader_Re_Pad(Dataset):
    def __init__(self, dataset_dir, dataset_name, patch_size, img_norm_cfg=None):
        super(TrainSetLoader_Re_Pad).__init__()
        self.dataset_name = dataset_name
        self.dataset_dir = dataset_dir + '/' + dataset_name
        self.patch_size = patch_size
        with open(self.dataset_dir + '/img_idx/train_' + dataset_name + '.txt', 'r') as f:
            self.train_list = f.read().splitlines()
        if img_norm_cfg == None:
            self.img_norm_cfg = get_img_norm_cfg(dataset_name, dataset_dir)
        else:
            self.img_norm_cfg = img_norm_cfg
        self.tranform = augumentation()

    def __getitem__(self, idx):
        try:
            img = Image.open(
                (self.dataset_dir + '/images/' + self.train_list[idx] + '.png').replace('//', '/')).convert(
                'I')  # read image base on version ”I“
            # img = Image.open((self.dataset_dir + '/images/' + self.train_list[idx] + '.png').replace('//','/'))
            mask = Image.open((self.dataset_dir + '/masks/' + self.train_list[idx] + '.png').replace('//', '/'))
        except:
            img = Image.open(
                (self.dataset_dir + '/images/' + self.train_list[idx] + '.bmp').replace('//', '/')).convert('I')
            mask = Image.open((self.dataset_dir + '/masks/' + self.train_list[idx] + '.bmp').replace('//', '/'))
        img = Normalized(np.array(img, dtype=np.float32), self.img_norm_cfg)  # convert PIL to numpy  and  normalize
        mask = np.array(mask, dtype=np.float32) / 255.0
        if len(mask.shape) > 2:
            mask = mask[:, :, 0]
        # print(self.train_list[idx])
        # img_patch, mask_patch = random_crop(img, mask, self.patch_size,
        #                                     pos_prob=0.5)  # 把短的一边先pad至256 把长的一边 随机裁出256  输出 256 256
        h,w = img.shape
        # resize
        target_size = PadImg_len(h, w, self.patch_size)
        input_image = np.array(resize(to_pil_image(img), target_size, interpolation=InterpolationMode.BILINEAR))
        # input_mask = np.array(resize(to_pil_image(mask), target_size, interpolation=InterpolationMode.BILINEAR))
        input_mask = np.array(resize(to_pil_image(mask), target_size, interpolation=InterpolationMode.NEAREST))

        img_patch, mask_patch = self.tranform(input_image, input_mask)  # 数据翻转增强
        img_patch, mask_patch = img_patch[np.newaxis, :], mask_patch[np.newaxis, :]  # 升维
        img_patch = torch.from_numpy(np.ascontiguousarray(img_patch))  # numpy 转tensor
        mask_patch = torch.from_numpy(np.ascontiguousarray(mask_patch))  # numpy 转tensor
        # pad
        img_patch = preprocess(self.patch_size, img_patch)
        mask_patch = preprocess(self.patch_size, mask_patch)

        return img_patch, mask_patch

    def __len__(self):
        return len(self.train_list)




class TestSetLoader_Re_Pad(Dataset):
    def __init__(self, dataset_dir, train_dataset_name, test_dataset_name, patch_size_eva, img_norm_cfg=None):
        super(TestSetLoader_Re_Pad).__init__()
        self.dataset_dir = dataset_dir + '/' + test_dataset_name
        self.patch_size_eva = patch_size_eva
        with open(self.dataset_dir + '/img_idx/test_' + test_dataset_name + '.txt', 'r') as f:
            self.test_list = f.read().splitlines()
        if img_norm_cfg == None:
            self.img_norm_cfg = get_img_norm_cfg(train_dataset_name, dataset_dir)
        else:
            self.img_norm_cfg = img_norm_cfg

    def __getitem__(self, idx):
        try:
            img = Image.open((self.dataset_dir + '/images/' + self.test_list[idx] + '.png').replace('//', '/')).convert(
                'I')
            mask = Image.open((self.dataset_dir + '/masks/' + self.test_list[idx] + '.png').replace('//', '/'))
        except:
            img = Image.open((self.dataset_dir + '/images/' + self.test_list[idx] + '.bmp').replace('//', '/')).convert(
                'I')
            mask = Image.open((self.dataset_dir + '/masks/' + self.test_list[idx] + '.bmp').replace('//', '/'))

        img = Normalized(np.array(img, dtype=np.float32), self.img_norm_cfg)
        mask = np.array(mask, dtype=np.float32) / 255.0
        if len(mask.shape) > 2:
            mask = mask[:, :, 0]

        if img.shape!=mask.shape:
            print(self.test_list[idx])

        h, w = img.shape
        # resize
        target_size = PadImg_len(h, w, self.patch_size_eva)
        input_image = np.array(resize(to_pil_image(img), target_size))


        img, mask = input_image[np.newaxis, :], mask[np.newaxis, :]
        img = torch.from_numpy(np.ascontiguousarray(img))
        mask = torch.from_numpy(np.ascontiguousarray(mask))
        # pad
        img = preprocess(self.patch_size_eva, img)

        # input_mask = np.array(resize(to_pil_image(mask), target_size))
        # mask = preprocess(self.patch_size_eva, mask)

        return img, mask, target_size, [h, w], self.test_list[idx]

    def __len__(self):
        return len(self.test_list)




class augumentation(object):
    def __call__(self, input, target):
        if random.random() < 0.5:  # 水平反转
            input = input[::-1, :]
            target = target[::-1, :]
        if random.random() < 0.5:  # 垂直反转
            input = input[:, ::-1]
            target = target[:, ::-1]
        if random.random() < 0.5:  # 转置反转
            input = input.transpose(1, 0)
            target = target.transpose(1, 0)
        return input, target
