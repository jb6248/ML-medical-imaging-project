from sklearn.metrics import roc_curve, auc
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F

import sklearn.metrics as metrics
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
import os
import argparse
import time
from core.debug import output_debug_image
from core.utils import calculate_Accuracy, get_img_list, get_model, get_data
from pylab import *
import random
import warnings

from experiment_code.exp_logger import ExpLogger

warnings.filterwarnings("ignore")
torch.set_warn_always(False)
plt.switch_backend("agg")

# --------------------------------------------------------------------------------
models_list = ["DG_Net", "M_Net", "UNet512"]
dataset_list = ["DRIVE", "STARE", "CHASEDB1"]

# --------------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="PyTorch ASOCT_Demo")

parser.add_argument("--epochs", type=int, default=500, help="the epochs of this run")
parser.add_argument(
    "--n_class",
    type=int,
    default=2,
    help="the channel of out img, decide the num of class, ASOCT_eyes is 2/4 class",
)
parser.add_argument("--lr", type=float, default=0.0001, help="initial learning rate")
parser.add_argument(
    "--GroupNorm", type=bool, default=True, help="decide to use the GroupNorm"
)
parser.add_argument(
    "--BatchNorm", type=bool, default=False, help="decide to use the BatchNorm"
)

# ---------------------------
# model
# ---------------------------
parser.add_argument("--datasetID", type=int, default=2, help="dir of the all img")
parser.add_argument("--SubImageID", type=int, default=1, help="Only for Stare Dataset")
parser.add_argument(
    "--model_id", type=int, default=0, help="the id of choice_model in models_list"
)
parser.add_argument(
    "--batch_size", type=int, default=1, help="the num of img in a batch"
)
parser.add_argument("--img_size", type=int, default=512, help="the train img size")
parser.add_argument(
    "--my_description",
    type=str,
    default="CHASEDB1_500DGNet",
    help="some description define your train",
)

# ---------------------------
# GPU
# ---------------------------
parser.add_argument("--use_gpu", action="store_true", help="dir of the all ori img")
parser.add_argument("--gpu_available", type=str, default="3", help="the gpu used")

args = parser.parse_args()
print(args)


# --------------------------------------------------------------------------------
# os.environ['CUDA_DEVICE_ORDER'] = "PCI_BUS_ID"
# os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_available


def fast_test(model, args, img_list, model_name, logger):
    softmax_2d = nn.Softmax2d()
    EPS = 1e-12

    Background_IOU = []
    Vessel_IOU = []
    ACC = []
    SE = []
    SP = []
    AUC = []

    # For ROC AUC plotting
    all_true_labels = []
    all_pred_scores = []

    for i, path in enumerate(img_list):
        start = time.time()
        img, imageGreys, gt, tmp_gt, img_shape, label_ori = get_data(
            dataset_list[args.datasetID],
            [path],
            img_size=args.img_size,
            gpu=args.use_gpu,
            flag="test",
        )
        model.eval()

        logger.info("fast_test path %s", path)

        out, side_5, side_6, side_7, side_8 = model(img, imageGreys)
        out = torch.log(softmax_2d(side_8) + EPS)

        out = F.upsample(out, size=(img_shape[0][0], img_shape[0][1]), mode="bilinear")
        out = out.cpu().data.numpy()
        y_pred = out[:, 1, :, :]
        y_pred = y_pred.reshape([-1])
        ppi = np.argmax(out, 1)

        print(f'ppi: {ppi.shape}, tmp_gt: {label_ori.shape}')
        print(f'ppi min: {np.min(ppi)}, max: {np.max(ppi)}')
        print(f'label_ori min: {np.min(label_ori)}, max: {np.max(label_ori)}')

        tmp_out = ppi.reshape([-1])
        tmp_gt = label_ori.reshape([-1])
        logger.info(f"tmp_gt shape: {tmp_gt.shape}, values: {tmp_gt}")

        my_confusion = metrics.confusion_matrix(tmp_out, tmp_gt).astype(np.float32)
        meanIU, Acc, Se, Sp, IU = calculate_Accuracy(my_confusion, debug=True)
        Auc = roc_auc_score(tmp_gt, y_pred)
        AUC.append(Auc)

        Background_IOU.append(IU[0])
        Vessel_IOU.append(IU[1])
        ACC.append(Acc)
        SE.append(Se)
        SP.append(Sp)

        # Collect true labels and predicted scores for ROC AUC plot
        all_true_labels.extend(tmp_gt)
        all_pred_scores.extend(y_pred)

    dry = 'Acc: %s  |  Se: %s |  Sp: %s |  Auc: %s |  Background_IOU: %s |  vessel_IOU: %s '%(str(np.mean(np.stack(ACC))),str(np.mean(np.stack(SE))), str(np.mean(np.stack(SP))),str(np.mean(np.stack(AUC))),str(np.mean(np.stack(Background_IOU))),str(np.mean(np.stack(Vessel_IOU))))
    print(dry)

    # store test information
    with open(r'./logs/%s_%s.txt' % (model_name, args.my_description), 'a+') as f:
        f.write(dry)
        f.write('\n\n')

        logger.info(
            str(i + 1)
            + r"/"
            + str(len(img_list))
            + ": "
            + "| Acc: {:.3f} | Se: {:.3f} | Sp: {:.3f} | Auc: {:.3f} |  Background_IOU: {:f}, vessel_IOU: {:f}".format(
                Acc, Se, Sp, Auc, IU[0], IU[1]
            )
            + "  |  time:%s" % (end - start)
        )

    logger.info(
        "Averages for fast_test Acc: %s  |  Se: %s |  Sp: %s |  Auc: %s |  Background_IOU: %s |  vessel_IOU: %s "
        % (
            str(np.mean(np.stack(ACC))),
            str(np.mean(np.stack(SE))),
            str(np.mean(np.stack(SP))),
            str(np.mean(np.stack(AUC))),
            str(np.mean(np.stack(Background_IOU))),
            str(np.mean(np.stack(Vessel_IOU))),
        )
    )
    logger.info("\n\n")

    # Generate ROC curve for the entire run
    # fpr, tpr, _ = roc_curve(all_true_labels, all_pred_scores, pos_label=1)  # Correct pos_label
    # roc_auc = auc(fpr, tpr)

    # Plot ROC curve
    # plt.figure()
    # plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    # plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
    # plt.xlim([0.0, 1.0])
    # plt.ylim([0.0, 1.05])
    # plt.xlabel('False Positive Rate')
    # plt.ylabel('True Positive Rate')
    # plt.title(f'Receiver Operating Characteristic - {model_name}')
    # plt.legend(loc='lower right')

    # Save the ROC curve plot to a file
    # Make sure RootDir is correctly defined or adjust the path
    # roc_plot_path = f"{os.getcwd()}/logs/{model_name}_{args.my_description}_roc_curve.png"
    # plt.savefig(roc_plot_path)
    # plt.close()

    # logger.info(f"ROC curve saved to {roc_plot_path}")

    # # Generate ROC curve for the entire run
    # fpr, tpr, _ = roc_curve(all_true_labels, all_pred_scores, pos_label=255)
    # roc_auc = auc(fpr, tpr)

    # Plotting AUC scores for all images processed
    # plt.figure()
    # plt.plot(AUC, color='blue', lw=2, label=f'Average AUC = {np.mean(AUC):.2f}')
    # plt.xlabel('Image Index')
    # plt.ylabel('AUC')
    # plt.title(f'ROC AUC Scores - {model_name}')
    # plt.legend(loc='lower right')

    # # Save the plot
    # roc_plot_path = f"{os.getcwd()}/logs/{model_name}_{args.my_description}_auc_scores.png"
    # plt.savefig(roc_plot_path)
    # plt.close()

    # logger.info(f"AUC plot saved to {roc_plot_path}")

    # # Plot ROC curve
    # plt.figure()
    # plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    # plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
    # plt.xlim([0.0, 1.0])
    # plt.ylim([0.0, 1.05])
    # plt.xlabel('False Positive Rate')
    # plt.ylabel('True Positive Rate')
    # plt.title(f'Receiver Operating Characteristic - {model_name}')
    # plt.legend(loc='lower right')

    # # Save the ROC curve plot to a file
    # roc_plot_path = f"{RootDir}/logs/{model_name}_{args.my_description}_roc_curve.png"
    # plt.savefig(roc_plot_path)
    # plt.close()

    # logger.info(f"ROC curve saved to {roc_plot_path}")

    # return np.mean(np.stack(Vessel_IOU))
    return np.mean(np.stack(ACC))


RootDir = os.getcwd()
# RootDir = "."
model_name = models_list[args.model_id]
Dataset = dataset_list[args.datasetID]
SubID = args.SubImageID
model = get_model(model_name)

argument_directory_base_path = os.path.join(
    RootDir,
    f"model_name{model_name}_dataset{Dataset}_sub_id{SubID}_description{args.my_description}",
)

os.makedirs(os.path.dirname(argument_directory_base_path), exist_ok=True)

log_file_path = r"%s/logs/logger.log" % (argument_directory_base_path,)
logger = ExpLogger(log_file_path)

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

model = model(n_classes=args.n_class, bn=args.GroupNorm, BatchNorm=args.BatchNorm)

if torch.cuda.device_count() > 1:
    logger.info("Let's use", torch.cuda.device_count(), "GPUs!")
    model = nn.DataParallel(model)

if args.use_gpu:
    model.cuda()
    logger.info("GPUs used: (%s)" % args.gpu_available)
    logger.info("------- success use GPU --------")

EPS = 1e-12
# define path
img_list = get_img_list(Dataset, SubID, flag="train")
test_img_list = get_img_list(Dataset, SubID, flag="test")

logger.info(f"model parameters : {str(model.parameters())}")


optimizer = torch.optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr
)
criterion = nn.NLLLoss2d()
# criterion = nn.NLLLoss()
# m = nn.LogSoftmax(dim=1)
softmax_2d = nn.Softmax2d()
IOU_best = 0


# if not os.path.exists(r"%s/models/%s_%s" % (RootDir, model_name, args.my_description)):
#     os.makedirs(r"%s/models/%s_%s" % (RootDir, model_name, args.my_description), exist_ok=True)


logger.info("This model is %s_%s: " % (model_name, args.my_description) + "\n")
logger.info("args: " + str(args) + "\n")
logger.info(
    "train lens: " + str(len(img_list)) + " | test #lens: " + str(len(test_img_list))
)
logger.info("\n\n----------------------------------------#-----\n\n")

total_start_time = time.time()  # Record the start time of the entire training process


from prettytable import PrettyTable
import time
import random
import torch
import numpy as np

BestAccuracy = 0
best_epoch = 0
epoch_accuracies = []  # List to store epoch number and average accuracy
epoch_losses = []
epoch_se = []
epoch_sp = []
epoch_background_iou = []
epoch_vessel_iou = []

for epoch in range(args.epochs):
    logger.info("#" * 20 + f" EPOCH #{epoch + 1} " + "#" * 20)

    model.train()
    begin_time = time.time()  # Record the start time for the current epoch

    logger.info(
        "This model is %s_%s_%s_%s"
        % (model_name, args.n_class, args.img_size, args.my_description)
    )
    random.shuffle(img_list)

    if (epoch % 100 == 0) and epoch != 0 and epoch < 500:
        args.lr /= 10
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    for i, (start, end) in enumerate(zip(range(0, len(img_list), args.batch_size),
                                         range(args.batch_size, len(img_list) + args.batch_size,
                                               args.batch_size))):
        
        debug = i == 0 # and epoch % 10 == 0

        path = img_list[start:end]
        img, imageGreys, gt, tmp_gt, img_shape,label_ori = get_data(Dataset, path, img_size=args.img_size, gpu=args.use_gpu, debug=debug)
        optimizer.zero_grad()
        out, side_5, side_6, side_7, side_8 = model(img, imageGreys)
        
        if debug:
            #output = criterion(m(out), gt)
            outimg = softmax_2d(out).cpu().detach().numpy()
            # print(f'out: {out.shape}, out: {torch.min(out)}, {torch.max(out)}')
            outimg = outimg[0] * 255
            empty_layer = np.zeros((1, outimg.shape[1], outimg.shape[2]))
            outimg = np.concatenate((outimg, empty_layer), axis=0)
            print(f'outimg: {outimg.shape}, outimg: {np.min(outimg)}, {np.max(outimg)}')
            outimg = np.array(outimg, dtype=np.uint8)
            output_debug_image(outimg, os.path.join(f'output_{epoch}_{i}.png'))

        out = torch.log(softmax_2d(out) + EPS)
        loss = criterion(out, gt)
        loss += criterion(torch.log(softmax_2d(side_5) + EPS), gt)
        loss += criterion(torch.log(softmax_2d(side_6) + EPS), gt)
        loss += criterion(torch.log(softmax_2d(side_7) + EPS), gt)
        loss += criterion(torch.log(softmax_2d(side_8) + EPS), gt)
        out = torch.log(softmax_2d(side_8) + EPS)

        loss.backward()
        optimizer.step()
        ppi = np.argmax(out.cpu().data.numpy(), 1)

        tmp_out = ppi.reshape([-1])
        tmp_gt = tmp_gt.reshape([-1])

        my_confusion = metrics.confusion_matrix(tmp_out, tmp_gt).astype(np.float32)

        meanIU, Acc, Se, Sp, IU = calculate_Accuracy(my_confusion)

        print(str('model: {:s}_{:s} | epoch_batch: {:d}_{:d} | loss: {:f}  | Acc: {:.3f} | Se: {:.3f} | Sp: {:.3f}'
                  '| Background_IOU: {:f}, vessel_IOU: {:f}').format(model_name, args.my_description,epoch, i, loss.item(), Acc,Se,Sp,
                                                                                  IU[0], IU[1]))

    print('training finish, time: %.1f s' % (time.time() - begin_time))

    if epoch % 2 == 0 and epoch != 0:
        Accuracy = fast_test(model, args, test_img_list, model_name)
        print('BestAccuracy:',BestAccuracy)
        if Accuracy > BestAccuracy:
            BestAccuracy = Accuracy
            directory = '%s/models/%s_%s'%(RootDir, model_name, args.my_description)
            if not os.path.exists(directory):
                os.makedirs(directory)
            torch.save(model.state_dict(),
                       os.path.join(directory, '%s.pth' % str(epoch)))
            # For evaluation
            print('success save Nucleus_best model')
