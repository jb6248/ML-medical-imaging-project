import os
import sys
from core.utils import get_img_list, get_model
from exp_logger import ExpLogger
import torch
import torch.nn as nn
import torch.nn.functional as F
from exp_train import train_experiement
from exp_test import fast_test
import matplotlib.pyplot as plt

import argparse

from pylab import *
import warnings

warnings.filterwarnings("ignore")
torch.set_warn_always(False)
plt.switch_backend("agg")

# -----   GLOBALS -------------------------------------------------------------
models_list = ["DG_Net", "M_Net", "UNet512"]
dataset_list = ["DRIVE", "STARE", "CHASEDB1"]
# --------------------------------------------------------------------------------


def parse_args():
    parser = argparse.ArgumentParser(description="PyTorch ASOCT_Demo")

    parser.add_argument(
        "--epochs", type=int, default=150, help="the epochs of this run"
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
        "--SubImageID", type=int, default=1, help="Only for Stare dataset"
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

    args = parser.parse_args()

    return args


# RUN THE CODE AT THE ROOT DIRECTORY PATH
# {RIT_USERNAME}@{SERVER{}:~/ML-medical-imaging-project$
def main():
    args = parse_args()

    root_dir = "./experiment_runs/"
    model_name = models_list[args.model_id]
    dataset = dataset_list[args.datasetID]
    sub_id = args.SubImageID
    model = get_model(
        model_name
    )  # Ensure this returns an instantiated model, not a class.

    argument_directory_base_path = os.path.join(
        root_dir,
        f"model_name{model_name}_dataset{dataset}_sub_id{sub_id}_description{args.my_description}",
    )

    # argument_directory_base_path_train = os.path.join(
    #     argument_directory_base_path, "train"
    # )
    # os.makedirs(argument_directory_base_path_train, exist_ok=True)

    log_file_path = os.path.join(argument_directory_base_path, "logger_train.log")
    logger = ExpLogger(log_file_path)

    logger.info(
        f"argument_directory_base_path_train: {argument_directory_base_path}"
    )
    logger.info(f"log_file_path: {log_file_path}")
    logger.info(f"model name: {model_name}")
    logger.info(f"dataset: {dataset}")
    logger.info(f"sub image ID: {sub_id}")
    logger.info(f"model: {model}")

    logger.info(
        "This model is %s_%s_%s_%s"
        % (model_name, args.n_class, args.img_size, args.my_description)
    )

    logger.info(str(args))

    EPS_BASELINE = 1e-8
    R_BASELINE = 2

    model = model(
        n_classes=args.n_class,
        bn=args.GroupNorm,
        BatchNorm=args.BatchNorm,
        r=R_BASELINE,
        eps=EPS_BASELINE,
    )

    logger.info("")
    if torch.cuda.device_count() > 1:
        logger.info("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)

    if args.use_gpu:
        model.cuda()
        logger.info("GPUs used: (%s)" % args.gpu_available)
        logger.info("------- success use GPU --------")

    train_img_list = get_img_list(dataset, sub_id, flag="train")
    test_img_list = get_img_list(dataset, sub_id, flag="test")

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr
    )
    criterion = nn.NLLLoss2d()
    softmax_2d = nn.Softmax2d()

    # RUN THE BASELINE
    test_model_path = train_experiement(
        logger,
        argument_directory_base_path,
        args,
        model,
        model_name,
        train_img_list,
        criterion,
        dataset,
        softmax_2d,
        EPS_BASELINE,
        R_BASELINE,
        optimizer,
    )

    test_model = get_model(model_name)
    test_model = test_model(
        n_classes=args.n_class, bn=args.GroupNorm, BatchNorm=args.BatchNorm
    )

    if args.use_gpu:
        model.cuda()
    # if False:
    if True:
        model.load_state_dict(torch.load(test_model_path, weights_only=True))
        logger.info("success load models: %s" % (test_model_path))

    argument_directory_base_path_test = os.path.join(
        argument_directory_base_path, "test"
    )
    os.makedirs(argument_directory_base_path_test, exist_ok=True)

    test_log_file_path = os.path.join(
        argument_directory_base_path_test, "logger_test.log"
    )
    test_logger = ExpLogger(test_log_file_path)

    test_logger.info(f"test_model_path: {test_model_path}")

    fast_test(model, args, test_img_list, model_name + args.my_description, logger)

    sys.exit(0)

    EPS_WINDOW = [1e-10, 1e-9, 1e-8, 1e-7, 1e-6]
    R_WINDOW = [0, 1, 2, 3, 4]

    for eps in EPS_WINDOW:
        for r in R_WINDOW:
            model = model(
                n_classes=args.n_class,
                bn=args.GroupNorm,
                BatchNorm=args.BatchNorm,
                r=r,
                eps=eps,
            )

            train(
                logger,
                argument_directory_base_path,
                args,
                model,
                model_name,
                train_img_list,
                criterion,
                dataset,
                softmax_2d,
                eps,
                r,
            )


if __name__ == "__main__":
    main()
