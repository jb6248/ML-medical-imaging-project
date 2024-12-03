# -*- coding: utf-8 -*-
from prettytable import PrettyTable
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2

import sklearn.metrics as metrics
from sklearn.metrics import roc_auc_score

import os
import argparse
import time
from core.utils import calculate_Accuracy, get_model, get_data, get_img_list
from pylab import *
import warnings

warnings.filterwarnings("ignore")
torch.set_warn_always(False)
plt.switch_backend("agg")

# --------------------------------------------------------------------------------

models_list = ["DG_Net", "M_Net", "UNet512"]
dataset_list = ['DRIVE', "STARE", "CHASEDB1"]

# --------------------------------------------------------------------------------
parser = argparse.ArgumentParser(description='PyTorch ASOCT_Demo')
# ---------------------------
# params do not need to change
# ---------------------------
parser.add_argument('--epochs', type=int, default=250,
                    help='the epochs of this run')
parser.add_argument('--n_class', type=int, default=2,
                    help='the channel of out img, decide the num of class, ASOCT_eyes is 2/4 class')
parser.add_argument('--lr', type=float, default=0.0015,
                    help='initial learning rate')
parser.add_argument('--GroupNorm', type=bool, default=True,
                    help='decide to use the GroupNorm')
parser.add_argument('--BatchNorm', type=bool, default=False,
                    help='decide to use the BatchNorm')
# ---------------------------
# model
# ---------------------------
parser.add_argument('--datasetID', type=int, default=2,
                    help='dir of the all img')
parser.add_argument('--SubImageID', type=int, default=20,
                    help='Only for Stare Dataset')
parser.add_argument('--best_model', type=str, default='./data/32.pth',
                    help='the pretrain model')
parser.add_argument('--model_id', type=int, default=0,
                    help='the id of choice_model in models_list')
parser.add_argument('--batch_size', type=int, default=1,
                    help='the num of img in a batch')
parser.add_argument('--img_size', type=int, default=512,
                    help='the train img size')
parser.add_argument('--my_description', type=str, default='CHASEDB1',
                    help='some description define your train')
# ---------------------------
# GPU
# ---------------------------
parser.add_argument('--use_gpu', type=bool, default=False,
                    help='dir of the all ori img')
parser.add_argument('--gpu_avaiable', type=str, default='0',
                    help='the gpu used')


def fast_test(model, args, img_list, model_name, logger):
    softmax_2d = torch.nn.Softmax2d()
    EPS = 1e-12

    Background_IOU = []
    Vessel_IOU = []
    ACC = []
    SE = []
    SP = []
    AUC = []

    ResultDir = './results'
    FullResultDir = os.path.join(ResultDir, model_name)
    if not os.path.exists(FullResultDir):
        os.makedirs(FullResultDir)

    accuracies = []
    sensitivity_epoch = []
    specificity_epoch = []
    losses = []
    background_iou = []
    vessel_iou = []

    # Lists to store metrics for the single epoch
    epoch_accuracies = []
    epoch_losses = []
    epoch_se = []
    epoch_sp = []
    epoch_background_iou = []
    epoch_vessel_iou = []

    # Process the test images
    for i, path in enumerate(img_list):

        start = time.time()
        img, imageGreys, gt, tmp_gt, img_shape, label_ori = get_data(dataset_list[args.datasetID], [path],
                                                                     img_size=args.img_size, gpu=args.use_gpu,
                                                                     flag='test')
        end = time.time()

        model.eval()
        path = path.rstrip("\n")
        FullFIleName = os.path.join(FullResultDir, path)

        out_avg, side_5, side_6, side_7, side_8 = model(img, imageGreys)

        out = torch.log(softmax_2d(side_8) + EPS)
        out = F.interpolate(out, size=(img_shape[0][0], img_shape[0][1]), mode='bilinear', align_corners=False)

        out = out.cpu().data.numpy()
        y_pred = out[:, 1, :, :]
        y_pred = y_pred.reshape([-1])
        ppi = np.argmax(out, 1)

        tmp_out = ppi.reshape([-1])
        tmp_gt = label_ori.reshape([-1])
        tmp_gt[tmp_gt == 255] = 1
        my_confusion = metrics.confusion_matrix(tmp_out, tmp_gt).astype(np.float32)
        meanIU, Acc, Se, Sp, IU = calculate_Accuracy(my_confusion)
        Auc = roc_auc_score(tmp_gt, y_pred)
        AUC.append(Auc)

        Background_IOU.append(IU[0])
        Vessel_IOU.append(IU[1])
        ACC.append(Acc)
        SE.append(Se)
        SP.append(Sp)

        accuracies.append(Acc)
        sensitivity_epoch.append(Se)
        specificity_epoch.append(Sp)
        background_iou.append(IU[0])
        vessel_iou.append(IU[1])

        logger.info(f"Image: {i+1}/{len(img_list)}: | Acc: {Acc:.3f} | Se: {Se:.3f} | Sp: {Sp:.3f} | Auc: {Auc:.3f} "
                    f"| Background_IOU: {IU[0]:.3f}, vessel_IOU: {IU[1]:.3f}  |  time: {end-start:.2f}s")

    # Calculate average values across the test set
    avg_acc = np.mean(ACC)
    avg_se = np.mean(SE)
    avg_sp = np.mean(SP)
    avg_auc = np.mean(AUC)
    avg_background_iou = np.mean(Background_IOU)
    avg_vessel_iou = np.mean(Vessel_IOU)

    logger.info(f'Average Test Fast Values: Acc: {avg_acc:.3f} | Se: {avg_se:.3f} | '
                f'Sp: {avg_sp:.3f} | Auc: {avg_auc:.3f} | Background_IOU: {avg_background_iou:.3f} | '
                f'vessel_IOU: {avg_vessel_iou:.3f}')

    # Create a PrettyTable to display the metrics for the single epoch
    t = PrettyTable(['Epoch', 'Average Accuracy', 'Loss', 'Sensitivity', 'Specificity', 'Background IOU', 'Vessel IOU'])

    # Add the single epoch metrics to the table
    t.add_row([
        1,  # Single epoch
        f"{avg_acc:.3f}",
        f"{np.mean(losses):.3f}",  # Average loss
        f"{avg_se:.3f}",
        f"{avg_sp:.3f}",
        f"{avg_background_iou:.3f}",
        f"{avg_vessel_iou:.3f}"
    ])

    # Print the table
    logger.info("\nEpoch-wise Metrics:")
    logger.info(str(t))

    return np.mean(np.stack(ACC))


if __name__ == '__main__':
    # os.environ['CUDA_DEVICE_ORDER'] = "PCI_BUS_ID"
    # os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_avaiable
    
    args = parser.parse_args()
    
    RootDir = os.getcwd()
    model_name = models_list[args.model_id]
    
    
    log_file_path = r"%s/logs/%s_%s.log" % (RootDir, model_name, args.my_description)
    os.makedirs(os.path.dirname(log_file_path), exist_ok=True)  # Ensure the log directory exists

    # Create a logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # File handler to log messages to a file
    file_handler = logging.FileHandler(log_file_path, mode='w')
    file_handler.setLevel(logging.INFO)

    # Console handler to logger.info messages to the console
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # Formatter for consistent logging output
    formatter = logging.Formatter('')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # Add handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    logger.info(args)

    model = get_model(model_name)
    model = model(n_classes=args.n_class, bn=args.GroupNorm, BatchNorm=args.BatchNorm)

    if args.use_gpu:
        model.cuda()
    #if False: 
    if True:
        #model_path = "/code/models/48.pth"
        model_path = args.best_model
        # model.load_state_dict(torch.load(model_path))
        model.load_state_dict(torch.load(model_path, weights_only=True))

        logger.info('success load models: %s' % (args.best_model))

    logger.info ('This model is %s_%s_%s' % (model_name, args.n_class, args.img_size))
    Dataset = dataset_list[args.datasetID]
    SubID = args.SubImageID
    
    test_img_list =  get_img_list(Dataset, SubID, flag='test')
    
    logger.info(f"Model Name: {model_name}")
    logger.info(f"Dataset: {Dataset}")
    logger.info("")  # Blank line for separation
    logger.info(f"SubImage ID: {SubID}")
    logger.info(f"Model: {model}")

    logger.info(
        "This model is %s_%s_%s_%s"
        % (model_name, args.n_class, args.img_size, args.my_description)
    )

    logger.info("")
    total_start_time = time.time()
    fast_test(model, args, test_img_list, model_name + args.my_description, logger)
    total_testing_time = time.time() - total_start_time  # Total training time for all epochs
    logger.info("Total testing time: %.1f seconds" % total_testing_time)

