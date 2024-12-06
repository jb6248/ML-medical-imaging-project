from prettytable import PrettyTable
from experiment_code.exp_logger import ExpLogger
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
from core.utils import calculate_Accuracy, get_img_list, get_model, get_data
from pylab import *
import random
import warnings

warnings.filterwarnings("ignore")
torch.set_warn_always(False)
plt.switch_backend("agg")

# -----   GLOBALS -------------------------------------------------------------
models_list = ["DG_Net", "M_Net", "UNet512"]
dataset_list = ["DRIVE", "STARE", "CHASEDB1"]

RootDir = os.getcwd()


# --------------------------------------------------------------------------------


def parse_args_int(args):
    parser = argparse.ArgumentParser(description="PyTorch ASOCT_Demo")

    parser.add_argument(
        "--epochs", type=int, default=500, help="the epochs of this run"
    )
    parser.add_argument(
        "--n_class",
        type=int,
        default=2,
        help="the channel of out img, decide the num of class, ASOCT_eyes is 2/4 class",
    )
    parser.add_argument(
        "--lr", type=float, default=0.0001, help="initial learning rate"
    )
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
    parser.add_argument(
        "--SubImageID", type=int, default=1, help="Only for Stare Dataset"
    )
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


def train(
    logger, args, model, model_name, img_list, criterion, Dataset, softmax_2d, EPS
):
    total_start_time = time.time()

    best_epoch_accuracy = 0
    best_epoch_number = 0
    epoch_accuracies = []
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

        accuracies = []
        sensitivity_epoch = []
        specificity_epoch = []
        losses = []
        background_iou = []
        vessel_iou = []

        for i, (start, end) in enumerate(
            zip(
                range(0, len(img_list), args.batch_size),
                range(
                    args.batch_size, len(img_list) + args.batch_size, args.batch_size
                ),
            )
        ):
            path = img_list[start:end]
            img, imageGreys, gt, tmp_gt, img_shape, label_ori = get_data(
                Dataset, path, img_size=args.img_size, gpu=args.use_gpu
            )
            optimizer.zero_grad()
            out, side_5, side_6, side_7, side_8 = model(img, imageGreys)

            # Loss calculation
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
            accuracies.append(Acc)
            sensitivity_epoch.append(Se)
            specificity_epoch.append(Sp)
            background_iou.append(IU[0])
            vessel_iou.append(IU[1])
            losses.append(loss.item())

            logger.info(
                str(
                    "epoch_batch: {:d}_{:d} | loss: {:f}  | Acc: {:.3f} | Se: {:.3f} | Sp: {:.3f}"
                    "| Background_IOU: {:f}, vessel_IOU: {:f}"
                ).format(
                    epoch,
                    i,
                    loss.item(),
                    Acc,
                    Se,
                    Sp,
                    IU[0],
                    IU[1],
                )
            )

        epoch_time = (
            time.time() - begin_time
        )  # Calculate the time for the current epoch
        logger.info("Epoch finished, time: %.1f s" % epoch_time)
        mean_accuracy = np.mean(accuracies)
        logger.info("Mean accuracy %s" % str(mean_accuracy))

        # Store the values for this epoch
        epoch_accuracies.append(mean_accuracy)
        epoch_losses.append(np.mean(losses))
        epoch_se.append(np.mean(sensitivity_epoch))
        epoch_sp.append(np.mean(specificity_epoch))
        epoch_background_iou.append(np.mean(background_iou))
        epoch_vessel_iou.append(np.mean(vessel_iou))

        if mean_accuracy > best_epoch_accuracy:
            best_epoch_number = epoch
            best_epoch_accuracy = mean_accuracy
            torch.save(
                model.state_dict(),
                "%s/models/%s_%s_.pth" % (RootDir, model_name, args.my_description),
            )
            logger.info("Successfully saved the best model")

        logger.info("*" * 60 + "\n")

    # After all epochs, create a PrettyTable to display the table
    t = PrettyTable(
        [
            "Epoch",
            "Average Accuracy",
            "Loss",
            "Sensitivity",
            "Specificity",
            "Background IOU",
            "Vessel IOU",
        ]
    )

    # Add rows to the table for each epoch
    for epoch in range(args.epochs):
        t.add_row(
            [
                epoch + 1,
                f"{epoch_accuracies[epoch]:.3f}",
                f"{epoch_losses[epoch]:.3f}",
                f"{epoch_se[epoch]:.3f}",
                f"{epoch_sp[epoch]:.3f}",
                f"{epoch_background_iou[epoch]:.3f}",
                f"{epoch_vessel_iou[epoch]:.3f}",
            ]
        )

    # Print the table
    logger.info("\nEpoch-wise Metrics:")
    logger.info(str(t))

    logger.info("\n\n")
    total_training_time = (
        time.time() - total_start_time
    )  # Total training time for all epochs
    logger.info("Model from Epoch %s was saved" % str(best_epoch_number))
    logger.info("Total training time: %.2f seconds" % total_training_time)


# RUN THE CODE AT THE ROOT DIRECTORY PATH
# {RIT_USERNAME}@{SERVER{}:~/ML-medical-imaging-project$
def main():
    parser = argparse.ArgumentParser(description="PyTorch ASOCT_Demo")
    args = parser.parse_args()

    RootDir = "."
    model_name = models_list[args.model_id]
    Dataset = dataset_list[args.datasetID]
    SubID = args.SubImageID
    model = get_model(model_name)

    argument_directory_base_path = (
        RootDir + "_" + model_name + "_" + args.my_description
    )

    log_file_path = r"%s/logs/%s_%s.log" % (RootDir, model_name, args.my_description)
    logger = ExpLogger(log_file_path)

    logger.info(f"argument_directory_base_path: {argument_directory_base_path}")

    logger.info(f"Model Name: {model_name}")
    logger.info(f"Dataset: {Dataset}")
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
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr
    )
    criterion = nn.NLLLoss2d()
    # criterion = nn.NLLLoss()
    # m = nn.LogSoftmax(dim=1)
    softmax_2d = nn.Softmax2d()
    IOU_best = 0

    train(
        logger, args, model, model_name, img_list, criterion, Dataset, softmax_2d, EPS
    )


if __name__ == "__main__":
    main()
