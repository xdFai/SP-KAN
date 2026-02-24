import numpy as np
import torch
from skimage import measure

def cal_tp_pos_fp_neg(output, target, score_thresh):
    predict = (output > score_thresh).float()
    if len(target.shape) == 3:
        print('？？？？')  # 加一个维度 使得target与 output的size一致
        target = target.unsqueeze(dim=0)
        # target = np.expand_dims(target.float(), axis=1)
        target.to('cuda', torch.float)

    elif len(target.shape) == 4:
        target = target.float()
    else:
        raise ValueError("Unknown target dimension")
    # 现在predict中高于阈值的部分为全1矩阵   target是GT

    intersection = predict * ((predict == target).float())

    tp = intersection.sum()  # 对的预测为对的
    fp = (predict * ((predict != target).float())).sum()  # 错的预测为对的 虚警像素数
    tn = ((1 - predict) * ((predict == target).float())).sum()  # 错的预测为错的
    fn = (((predict != target).float()) * (1 - predict)).sum()  # 对的预测为错的
    pos = tp + fn  # 标签中 阳性的个数
    neg = fp + tn  # 标签中 阴性的个数
    class_pos = tp + fp  # 检测出的个数

    return tp, pos, fp, neg, class_pos

class F1():
    """Computes pixAcc and mIoU metric scores
    """

    def __init__(self):
        # bin的意义实际上是确定ROC曲线上的threshold取多少个离散值
        # nclass :有几个类别 红外弱小目标检测只有一个类别
        super(F1, self).__init__()

        self.tp_arr = np.zeros(1)
        self.pos_arr = np.zeros(1)
        self.fp_arr = np.zeros(1)
        self.neg_arr = np.zeros(1)
        self.class_pos = np.zeros(1)
        self.reset()

    # 网络输入的结果和标签 计算两者之前的东西
    def update(self, preds, labels):
        self.tp_arr, self.pos_arr, self.fp_arr, self.neg_arr, self.class_pos = cal_tp_pos_fp_neg(preds, labels, 0.5)

    def get(self):
        # tp_rates = self.tp_arr / (self.pos_arr + 0.001)  # tp_rates = recall = TP/(TP+FN)
        # fp_rates = self.fp_arr / (self.neg_arr + 0.001)  # fp_rates =  FP/(FP+TN)
        # FP = self.fp_arr / (self.neg_arr + self.pos_arr)
        recall = self.tp_arr / (self.pos_arr + 0.001)  # recall = TP/(TP+FN)
        precision = self.tp_arr / (self.class_pos + 0.001)  # precision = TP/(TP+FP)
        f1_score = (2.0 * recall * precision) / (recall + precision + 0.00001)

        return float(f1_score.cpu().detach().numpy())

    def reset(self):
        self.tp_arr = np.zeros(1)
        self.pos_arr = np.zeros(1)
        self.fp_arr = np.zeros(1)
        self.neg_arr = np.zeros(1)
        self.class_pos = np.zeros(1)


class mIoU():
    
    def __init__(self):
        super(mIoU, self).__init__()
        self.reset()

    def update(self, preds, labels):
        correct, labeled = batch_pix_accuracy(preds, labels)     #  labeled GT中目标的像素数目   correct预测正确的像素数
        inter, union = batch_intersection_union(preds, labels)
        self.total_correct += correct
        self.total_label += labeled
        self.total_inter += inter
        self.total_union += union

    def get(self):
        pixAcc = 1.0 * self.total_correct / (np.spacing(1) + self.total_label)
        IoU = 1.0 * self.total_inter / (np.spacing(1) + self.total_union)
        mIoU = IoU.mean()
        return float(pixAcc), mIoU

    def reset(self):
        self.total_inter = 0
        self.total_union = 0
        self.total_correct = 0
        self.total_label = 0

class PD_FA():
    def __init__(self,):
        super(PD_FA, self).__init__()
        self.image_area_total = []
        self.image_area_match = []
        self.dismatch_pixel = 0
        self.all_pixel = 0
        self.PD = 0
        self.target= 0
    def update(self, preds, labels, size):
        predits  = np.array((preds).cpu()).astype('int64')
        labelss = np.array((labels).cpu()).astype('int64') 

        image = measure.label(predits, connectivity=2)
        coord_image = measure.regionprops(image)
        label = measure.label(labelss , connectivity=2)
        coord_label = measure.regionprops(label)

        self.target    += len(coord_label)
        self.image_area_total = []
        self.image_area_match = []
        self.distance_match   = []
        self.dismatch         = []

        for K in range(len(coord_image)):
            area_image = np.array(coord_image[K].area)
            self.image_area_total.append(area_image)

        for i in range(len(coord_label)):
            centroid_label = np.array(list(coord_label[i].centroid))
            for m in range(len(coord_image)):
                centroid_image = np.array(list(coord_image[m].centroid))
                distance = np.linalg.norm(centroid_image - centroid_label)
                area_image = np.array(coord_image[m].area)
                if distance < 3:
                    self.distance_match.append(distance)
                    self.image_area_match.append(area_image)

                    del coord_image[m]
                    break

        self.dismatch = [x for x in self.image_area_total if x not in self.image_area_match]
        self.dismatch_pixel +=np.sum(self.dismatch)
        self.all_pixel +=size[0]*size[1]
        self.PD +=len(self.distance_match)

    def get(self):
        Final_FA =  self.dismatch_pixel / self.all_pixel
        Final_PD =  self.PD /self.target
        return Final_PD, float(Final_FA.cpu().detach().numpy())

    def reset(self):
        self.FA  = np.zeros([self.bins+1])
        self.PD  = np.zeros([self.bins+1])

def batch_pix_accuracy(output, target):   
    if len(target.shape) == 3:
        target = np.expand_dims(target.float(), axis=1)
    elif len(target.shape) == 4:
        target = target.float()
    else:
        raise ValueError("Unknown target dimension")

    assert output.shape == target.shape, "Predict and Label Shape Don't Match"
    predict = (output > 0).float()
    pixel_labeled = (target > 0).float().sum()
    pixel_correct = (((predict == target).float())*((target > 0)).float()).sum()
    assert pixel_correct <= pixel_labeled, "Correct area should be smaller than Labeled"
    return pixel_correct, pixel_labeled

def batch_intersection_union(output, target):
    mini = 1
    maxi = 1
    nbins = 1
    predict = (output > 0).float()
    if len(target.shape) == 3:
        target = np.expand_dims(target.float(), axis=1)
    elif len(target.shape) == 4:
        target = target.float()
    else:
        raise ValueError("Unknown target dimension")
    intersection = predict * ((predict == target).float())

    area_inter, _  = np.histogram(intersection.cpu(), bins=nbins, range=(mini, maxi))
    area_pred,  _  = np.histogram(predict.cpu(), bins=nbins, range=(mini, maxi))
    area_lab,   _  = np.histogram(target.cpu(), bins=nbins, range=(mini, maxi))
    area_union     = area_pred + area_lab - area_inter

    assert (area_inter <= area_union).all(), \
        "Error: Intersection area should be smaller than Union area"
    return area_inter, area_union
