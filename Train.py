import argparse
import time

from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset
from util.dataset_resize import *
from util.metrics import *
from util.utils import *
from torch.utils.tensorboard import SummaryWriter
from model.SP_KAN import SP_KAN as SP_KAN
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

parser = argparse.ArgumentParser(description="PyTorch BasicIRSTD train")
parser.add_argument("--model_names", default=['SP_KAN'], type=list, help="'SP_KAN', 'ALCNet'")
parser.add_argument("--dataset_names", default=['SIRST3'],
                    type=list)  # SIRST3： NUAA NUDT-SIRST IRSTD-1K  SIRST3   SIRST3_Enhance
parser.add_argument("--optimizer_name", default='Adam', type=str, help="optimizer name: AdamW, Adam, Adagrad, SGD")
parser.add_argument("--epochs", default=1000, type=int, help="numbers of epoch")

parser.add_argument("--begin_test", default=100, type=int)
parser.add_argument("--every_test", default=2, type=int)
parser.add_argument("--every_print", default=10, type=int)

# parser.add_argument("--begin_test", default=1, type=int)
# parser.add_argument("--every_test", default=1, type=int)
# parser.add_argument("--every_print", default=1, type=int)

parser.add_argument("--dataset_dir", default=r'./datasets')
parser.add_argument("--batchSize", type=int, default=8, help="Training batch sizse")
# ******************* Others   *******************
parser.add_argument("--patchSize", type=int, default=512, help="Training patch size")
parser.add_argument("--patchSize_eva", type=int, default=512, help="Evaluation patch size")
parser.add_argument("--save", default=r'./log', type=str, help="Save path of checkpoints")
parser.add_argument("--log_dir", type=str, default="./otherlogs/SP_KAN", help='path of log files')
parser.add_argument("--img_norm_cfg", default=None, type=dict)
parser.add_argument("--threads", type=int, default=0, help="Number of threads for data loader to use")
parser.add_argument("--threshold", type=float, default=0.5, help="Threshold for test")
parser.add_argument("--seed", type=int, default=42, help="Threshold for test")
parser.add_argument("--resume", default=False, type=list, help="Resume from exisiting checkpoints (default: None)")

global opt
opt = parser.parse_args()
seed_pytorch(opt.seed)
print("---------------------------------------------------------------")
print('batchSize: {0} -- begin_test: {1} -- every_print: {2} -- every_test: {3}'.format(opt.batchSize, opt.begin_test,
                                                                                        opt.every_print,
                                                                                        opt.every_test))


def train():
    # *******************************************************************************************************
    #                                             Train
    # *******************************************************************************************************
    train_set = TrainSetLoader_Re_Pad(dataset_dir=opt.dataset_dir, dataset_name=opt.dataset_name, patch_size=opt.patchSize,
                               img_norm_cfg=opt.img_norm_cfg)
    train_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batchSize, shuffle=True)

    test_set = TestSetLoader_Re_Pad(opt.dataset_dir, opt.dataset_name, opt.dataset_name, opt.patchSize_eva,img_norm_cfg=opt.img_norm_cfg)
    test_loader = DataLoader(dataset=test_set, num_workers=opt.threads, batch_size=1, shuffle=False)

    net = Net(model_name=opt.model_name, mode='train').cuda()
    net.apply(weights_init_kaiming)
    net.train()
    epoch_state = 0
    total_loss_list = []
    total_loss_epoch = []

    if not os.path.exists(opt.log_dir):
        os.makedirs(opt.log_dir)
    writer = SummaryWriter(opt.log_dir)

    ### Default settings of SP_KAN
    if opt.optimizer_name == 'Adam':
        opt.optimizer_settings = {'lr': 0.001}
        opt.scheduler_name = 'CosineAnnealingLR'
        opt.scheduler_settings = {'epochs': opt.epochs, 'eta_min': 1e-5, 'last_epoch': -1}


    opt.nEpochs = opt.scheduler_settings['epochs']
    optimizer, scheduler = get_optimizer(net, opt.optimizer_name, opt.scheduler_name, opt.optimizer_settings,
                                         opt.scheduler_settings)

    for idx_epoch in range(epoch_state, opt.nEpochs):
        net.train()
        results1 = [0, 0]
        results2 = [0, 1]
        for idx_iter, (img, gt_mask) in enumerate(train_loader):
            img, gt_mask = Variable(img).cuda(), Variable(gt_mask).cuda()
            if img.shape[0] == 1:
                continue
            # print(img.shape)
            pred = net.forward(img)
            # gt_mask = gt_mask.float()
            loss = net.loss(pred, gt_mask)
            total_loss_epoch.append(loss.detach().cpu())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()

        if (idx_epoch + 1) % opt.every_print == 0:  # tensorboard : write train loss
            total_loss_list.append(float(np.array(total_loss_epoch).mean()))
            print(time.ctime()[4:-5] + ' Epoch---%d, total_loss---%f, lr---%f,'
                  % (idx_epoch + 1, total_loss_list[-1], scheduler.get_last_lr()[0]))
            opt.f.write(time.ctime()[4:-5] + ' Epoch---%d, total_loss---%f,\n'
                        % (idx_epoch + 1, total_loss_list[-1]))
            total_loss_epoch = []
            # Log the scalar values
            writer.add_scalar('loss', total_loss_list[-1], idx_epoch + 1)
            writer.add_scalar('lr', scheduler.get_last_lr()[0], idx_epoch + 1)

        if idx_epoch == 0:
            best_mIOU = results1
            best_Pd_Fa = results2

        if (idx_epoch + 1) >= opt.begin_test and (
                idx_epoch + 1) % opt.every_test == 0:  # tensorboard: write test evaluate
            # *******************************************************************************************************
            #                                             Test
            # *******************************************************************************************************
            net.eval()
            with torch.no_grad():
                eval_mIoU = mIoU()
                eval_PD_FA = PD_FA()
                Metric = F1()
                # test_loss = []
                for idx_iter, (img, gt_mask, target_size, org_size, _) in enumerate(test_loader):
                    img = Variable(img).cuda()
                    pred = net.forward(img)
                    if isinstance(pred, tuple):
                        pred = pred[-1]
                    elif isinstance(pred, list):
                        pred = pred[-1]
                    else:
                        pred = pred

                    pred = postprocess_masks(pred, target_size, org_size)

                    eval_mIoU.update((pred > opt.threshold).cpu(), gt_mask.cpu())
                    eval_PD_FA.update((pred[0, 0, :, :] > opt.threshold).cpu(), gt_mask[0, 0, :, :], org_size)
                    Metric.update(labels=gt_mask.cpu(), preds=pred.cpu())

                results1 = eval_mIoU.get()
                results2 = eval_PD_FA.get()
                f1_score = Metric.get()
                writer.add_scalar('mIOU', results1[-1], idx_epoch + 1)
                writer.add_scalar('F1', f1_score, idx_epoch + 1)
                writer.add_scalar('Pd', results2[0], idx_epoch + 1)
                writer.add_scalar('Fa', results2[1], idx_epoch + 1)

            # IOU
            if results1[1] > best_mIOU[1]:
                best_mIOU = results1
                print('------save the best model epoch', opt.model_name, '_%d ------' % (idx_epoch + 1))
                opt.f.write("the best model epoch \t" + str(idx_epoch + 1) + '\n')
                print("mIoU, F1:\t" + str(results1[1]) + ", " + str(f1_score))
                # print("testloss:\t" + str(test_loss[-1]))
                print("PD, FA :\t" + str(results2))
                opt.f.write("mIoU: " + str(results1[1]) + '\n')
                opt.f.write("PD, FA :\t" + str(results2) + '\n')
                best_IOU = format(results1[1], '.4f')
                best_Pd = format(results2[0], '.4f')
                best_Fa = format(results2[1], '.6f')

                save_pth = opt.save + '/' + opt.dataset_name + '/' + opt.model_name + '_' + str(
                    idx_epoch + 1) + '_' + str(best_IOU) + "_" + str(best_Pd) + "_" + str(best_Fa) + "_" + '.pth.tar'
                save_checkpoint({
                    'epoch': idx_epoch + 1,
                    'state_dict': net.state_dict(),
                    'total_loss': total_loss_list,
                }, save_pth)

            elif results2[0] > 0.978 and results2[1] < 1e-5:
                # best_Pd = results2
                print('------save the best model epoch', opt.model_name, '_%d ------PdPd' % (idx_epoch + 1))
                opt.f.write("the best model epoch \t" + str(idx_epoch + 1) + '\n')
                print("mIoU, F1:\t" + str(results1[1]) + ", " + str(f1_score))
                print("PD, FA:\t" + str(results2))
                opt.f.write("mIoU: " + str(results1[1]) + '\n')
                opt.f.write("PD, FA :\t" + str(results2) + '\n')
                best_IOU = format(results1[1], '.4f')
                best_Pd = format(results2[0], '.4f')
                best_Fa = format(results2[1], '.6f')
                save_pth = opt.save + '/' + opt.dataset_name + '/' + opt.model_name + 'PdPd' + '_Epoch' + str(
                    idx_epoch + 1) + '_' + str(best_IOU) + '_' + str(best_Pd) + "_" + str(best_Fa) + "_" + '.pth.tar'
                save_checkpoint({
                    'epoch': idx_epoch + 1,
                    'state_dict': net.state_dict(),
                    'total_loss': total_loss_list,
                }, save_pth)


def save_checkpoint(state, save_path):
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))
    torch.save(state, save_path)
    return save_path


def postprocess_masks(pred, input_size, original_size):
    preds = pred[..., : input_size[0], : input_size[1]]
    preds = F.interpolate(preds, original_size, mode="bilinear", align_corners=False)
    return preds


class Net(nn.Module):
    def __init__(self, model_name, mode):
        super(Net, self).__init__()
        self.model_name = model_name
        # *******************************************************************************************************
        #                                             Loss
        # *******************************************************************************************************
        self.cal_loss = nn.BCELoss(size_average=True)

        if model_name == 'SP_KAN':
            # *******************************************************************************************************
            #                                             Input channels
            # *******************************************************************************************************
            input_channels = 1
            if mode == 'train':
                self.model = SP_KAN(1, 1, mode='train', deepsuper=True)
            else:
                self.model = SP_KAN(1, 1, mode='test', deepsuper=True)
            print('input channels: {}'.format(input_channels))
            print("---------------------------------------------------------------")

    def forward(self, img):
        return self.model(img)

    def loss(self, preds, gt_masks):
        if isinstance(preds, list):
            loss_total = 0
            for i in range(len(preds)):
                pred = preds[i]
                gt_mask = gt_masks[i]
                loss = self.cal_loss(pred, gt_mask)
                loss_total = loss_total + loss
            return loss_total / len(preds)

        elif isinstance(preds, tuple):
            a = []
            for i in range(len(preds)):
                pred = preds[i]
                loss = self.cal_loss(pred, gt_masks)
                a.append(loss)
            # loss_total = mean(a)
            loss_total = a[0] + a[1] + a[2] + a[3] + a[4] + a[5]
            # loss_total = sum(a)
            return loss_total

        else:
            loss = self.cal_loss(preds, gt_masks)
            return loss


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('BatchNorm') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


if __name__ == '__main__':
    for dataset_name in opt.dataset_names:
        opt.dataset_name = dataset_name
        for model_name in opt.model_names:
            opt.model_name = model_name
            if not os.path.exists(opt.save):
                os.makedirs(opt.save)
            opt.f = open(opt.save + '/' + opt.dataset_name + '_' + opt.model_name + '_' + (time.ctime()).replace(' ',
                                                                                                                 '_').replace(
                ':', '_') + '.txt', 'w')
            print(opt.dataset_name + '\t' + opt.model_name)
            train()
            print('\n')
            opt.f.close()
