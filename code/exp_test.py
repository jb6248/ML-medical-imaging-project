import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

import sklearn.metrics as metrics
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
import time
from core.utils import calculate_Accuracy, get_data
from pylab import *
import warnings

warnings.filterwarnings("ignore")
torch.set_warn_always(False)
plt.switch_backend("agg")

dataset_list = ["DRIVE", "STARE", "CHASEDB1"]


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

        tmp_out = ppi.reshape([-1])
        tmp_gt = label_ori.reshape([-1])
        logger.info(f"tmp_gt shape: {tmp_gt.shape}, values: {tmp_gt}")

        my_confusion = metrics.confusion_matrix(tmp_out, tmp_gt).astype(np.float32)
        meanIU, Acc, Se, Sp, IU = calculate_Accuracy(my_confusion)
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

        end = time.time()

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
