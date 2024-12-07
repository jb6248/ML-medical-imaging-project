import os
import sys
from core.utils import get_img_list, get_model
from exp_logger import ExpLogger
import torch
import torch.nn as nn
import torch.nn.functional as F
from exp_train import train_experiement
from exp_test import test_experiment
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


# Function to clear out files in a directory
def clear_directory(directory_path):
    if os.path.exists(directory_path):
        for filename in os.listdir(directory_path):
            file_path = os.path.join(directory_path, filename)
            if os.path.isfile(file_path):
                os.remove(file_path)  # Remove file

# Reusable function to initialize the model
def initialize_model(args, model_name, n_classes, eps, r, group_norm, batch_norm, use_gpu, logger):
    model = get_model(model_name)
    logger.info(f"Model: {model}")
    model = model(
        n_classes=n_classes,
        bn=group_norm,
        BatchNorm=batch_norm,
        r=r,
        eps=eps
    )
    
    if torch.cuda.device_count() > 1:
        logger.info(f"Let's use {torch.cuda.device_count()} GPUs!")
        model = nn.DataParallel(model)

    if use_gpu:
        model.cuda()
        logger.info(f"GPUs used: ({args.gpu_available})")
        logger.info("------- success use GPU --------")
    
    return model

# Reusable function to create logger
def create_logger(log_path):
    logger = ExpLogger(log_path)
    return logger


def log_run_variables(logger, model_name, dataset, sub_id, args, model):
    logger.info(f"Model name: {model_name}")
    logger.info(f"Dataset: {dataset}")
    logger.info(f"Sub image ID: {sub_id}")
    logger.info(f"This model is {model_name}_{args.n_class}_{args.img_size}_{args.my_description}")
    logger.info(str(args))

# Function to run a single window (train and test)
def run_window(args, base_path, eps, r, model_name, dataset, sub_id, train_img_list, test_img_list):
    window_path = os.path.join(base_path, f"window_eps_{eps}_r_{r}")
    clear_directory(window_path)
    os.makedirs(window_path, exist_ok=True)
    
    log_file_path = os.path.join(window_path, "logger_train.log")
    train_logger = create_logger(log_file_path)
    
    model = initialize_model(args,model_name, args.n_class, eps, r, args.GroupNorm, args.BatchNorm, args.use_gpu, train_logger)
    log_run_variables(train_logger, model_name, dataset, sub_id, args, model)
    train_logger.info(str(model.parameters()))
    
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    criterion = nn.NLLLoss2d()
    softmax_2d = nn.Softmax2d()

    # Training experiment
    train_experiement(train_logger, window_path, args, model, model_name, train_img_list, criterion, dataset, softmax_2d, optimizer)
    train_logger.close()

    # Testing
    test_log_file_path = os.path.join(window_path, "logger_test.log")
    test_logger = create_logger(test_log_file_path)
    test_model = initialize_model(args,model_name, args.n_class, eps, r, args.GroupNorm, args.BatchNorm, args.use_gpu, test_logger)
    
    log_run_variables(test_logger, model_name, dataset, sub_id, args, test_model)
    
    test_model_path = os.path.join(window_path, "best_train_model.pth")
    test_model.load_state_dict(torch.load(test_model_path, weights_only=True))
    test_logger.info(f"Success loading test model: {test_model_path}")

    test_accuracy_mean = test_experiment(test_model, args, test_img_list, f"{model_name}{args.my_description}", test_logger)
    test_logger.info(f"Test accuracy: {test_accuracy_mean}")
    test_logger.close()

# Main function to run experiments
def main():
    args = parse_args()

    root_dir = "./experiment_runs/"
    model_name = models_list[args.model_id]
    dataset = dataset_list[args.datasetID]
    sub_id = args.SubImageID

    argument_directory_base_path = os.path.join(
        root_dir,
        f"model_name{model_name}_dataset{dataset}_sub_id{sub_id}_description{args.my_description}_epochs{args.epochs}",
    )
    clear_directory(argument_directory_base_path)

    base_line_path = os.path.join(argument_directory_base_path, f"base_line_eps_1e-8_r_2")
    os.makedirs(base_line_path, exist_ok=True)

    train_img_list = get_img_list(dataset, sub_id, flag="train")
    test_img_list = get_img_list(dataset, sub_id, flag="test")

    # Run baseline experiment
    run_window(args, argument_directory_base_path, 1e-8, 2, model_name, dataset, sub_id, train_img_list, test_img_list)

    # Run experiments with different EPS and R values
    EPS_WINDOW = [1e-10, 1e-9, 1e-8, 1e-7, 1e-6]
    R_WINDOW = [-1, 1, 2, 3, 4]

    for eps in EPS_WINDOW:
        for r in R_WINDOW:
            run_window(args, argument_directory_base_path, eps, r, model_name, dataset, sub_id, train_img_list, test_img_list)




if __name__ == "__main__":
    main()
