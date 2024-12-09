import os
import sys
from prettytable import PrettyTable
import torch
import sklearn.metrics as metrics
import matplotlib.pyplot as plt

import time
from core.utils import calculate_Accuracy, get_data, output_debug_image
from pylab import *
import random
import warnings

warnings.filterwarnings("ignore")
torch.set_warn_always(False)
plt.switch_backend("agg")


PAPER_EPS = 1e-12


def train_experiement(
    logger,
    argument_directory_base_path,
    args,
    model,
    model_name,
    img_list,
    criterion,
    dataset,
    softmax_2d,
    optimizer,
    debugimages_path
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
            debug = i == 0 and (epoch == (args.epochs - 1) or epoch % 10 == 0)
            path = img_list[start:end]
            img, imageGreys, gt, tmp_gt, img_shape, label_ori = get_data(
                debugimages_path, logger, dataset, path, img_size=args.img_size, gpu=args.use_gpu,
                debug=debug and (epoch % 20 == 0 or epoch == (args.epochs - 1))
            )
            optimizer.zero_grad()
            out, side_5, side_6, side_7, side_8 = model(img, imageGreys, debug=debug)

            if debug:
                #output = criterion(m(out), gt)
                prediction = softmax_2d(out).cpu().detach().numpy()
                # print(f'out: {out.shape}, out: {torch.min(out)}, {torch.max(out)}')
                outimg = prediction[0] * 255
                empty_layer = np.zeros((1, outimg.shape[1], outimg.shape[2]))
                outimg = np.concatenate((outimg, empty_layer), axis=0)
                logger.info(f'outimg: {outimg.shape}, outimg: {np.min(outimg)}, {np.max(outimg)}')
                outimg = np.array(outimg, dtype=np.uint8)
                output_debug_image(outimg, logger, os.path.join(debugimages_path, f'output_{epoch}_{i}.png'), dated=True)

                # show the differences using a color scheme:
                # TP: WHITE
                # FP: RED
                # FN: BLUE
                # TN: BLACK

                logger.info(f'prediction: {prediction.shape}, gt: {gt.shape}')
                prediction = np.argmax(prediction, axis=1)[0]
                debug_gt = gt.cpu().detach().numpy()
                debug_img = np.zeros((prediction.shape[0], prediction.shape[1], 3), dtype=np.uint8)
                debug_img[:, :, 0] = prediction * 255
                debug_img[:, :, 1] = (debug_gt & prediction) * 255
                debug_img[:, :, 2] = debug_gt * 255
                output_debug_image(debug_img, logger, os.path.join(debugimages_path, f'output_diff_{epoch}_{i}_diff.png'), dated=True)

            # Loss calculation
            out = torch.log(softmax_2d(out) + PAPER_EPS)
            loss = criterion(out, gt)
            loss += criterion(torch.log(softmax_2d(side_5) + PAPER_EPS), gt)
            loss += criterion(torch.log(softmax_2d(side_6) + PAPER_EPS), gt)
            loss += criterion(torch.log(softmax_2d(side_7) + PAPER_EPS), gt)
            loss += criterion(torch.log(softmax_2d(side_8) + PAPER_EPS), gt)
            out = torch.log(softmax_2d(side_8) + PAPER_EPS)

            loss.backward()
            optimizer.step()
            ppi = np.argmax(out.cpu().data.numpy(), 1)

            tmp_out = ppi.reshape([-1])
            tmp_gt = tmp_gt.reshape([-1])

            my_confusion = metrics.confusion_matrix(tmp_out, tmp_gt).astype(np.float32)

            meanIU, Acc, Se, Sp, IU = calculate_Accuracy(my_confusion, logger, debug=False)
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
            model_file_path = os.path.join(
                argument_directory_base_path, "best_train_model.pth"
            )
            os.makedirs(os.path.dirname(model_file_path), exist_ok=True)
            torch.save(model.state_dict(), model_file_path)
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
