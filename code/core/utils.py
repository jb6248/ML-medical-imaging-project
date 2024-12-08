from __future__ import division
import numpy as np
import os
import cv2
import torch
from core.models import *
import pickle as pkl
from torch.autograd import Variable
import imageio
from imgaug import augmenters as iaa
import imgaug as ia
from imgaug.augmentables.segmaps import SegmentationMapsOnImage
from matplotlib import pyplot as plt
import matplotlib
from datetime import datetime
#matplotlib.use('TkAgg')

DRIVEDataSetPath = './data/DRIVE'
STAREDataSetPath = './data/Stare/'
CHASEDB1DataSetPath = './data/CHASEDB1'

def output_debug_image(img, logger, name, dated=True):
    '''
    params
    img: np.array
    name: str (should contain extension)
    dated: bool (whether to prefix the filename with the date and time)
    '''
    try:
        final_img = np.array(img, dtype=np.uint8)
        if final_img.shape[0] < 5: # this is the color dimension
            final_img = np.transpose(final_img, [1, 2, 0])
        # maybe add a date to the name to keep it from being overwritten between images
        if dated:
            datestring = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
            name = f'{datestring}_{name}'
        plt.imsave(name, final_img)
        logger.info(f'------------- print image ---------------')
        logger.info(f'shape: {img.shape}')
        logger.info(f'range: {np.min(final_img)} to {np.max(final_img)}')
        logger.info(f'save as: {name}')
    except Exception as e:
        logger.info(f'ERROR: unable to save image {name} with shape {img.shape}')
        logger.info(e)

def get_data(debugimages_path, logger, dataset, img_name, img_size=256, gpu=True, flag='train', debug=False):

    def get_label(label):
        tmp_gt = label.copy()
        label = label.astype(np.int64)
        label = Variable(torch.from_numpy(label)).long()
        if gpu:
            label = label.cuda()
        return label, tmp_gt

    images = []
    imageGreys = []
    labels = []
    tmp_gts = []

    img_shape =[]
    label_ori = []
    Mask_ori =[]
    batch_size = len(img_name)

    for i in range(batch_size):
        if dataset == "DRIVE":
            img_path = os.path.join(DRIVEDataSetPath, 'images', img_name[i].rstrip('\n'))
            if flag == 'train':
                label_name = img_name[i].rstrip('\n')[:-12] + 'manual1.gif'
            else:
                label_name = img_name[i].rstrip('\n')[:-8] + 'manual1.gif'
            label_path = os.path.join(DRIVEDataSetPath, 'label', label_name)
            img = cv2.imread(img_path)
            label = imageio.mimread(label_path)
            if label is not None:
                label = np.array(label)
                label = label[0]

        if dataset == "STARE":
            img_path = os.path.join(STAREDataSetPath, 'images', img_name[i].rstrip('\n'))
            label_name = img_name[i].rstrip('\n')[:-4] + '.vk.ppm'
            label_path = os.path.join(STAREDataSetPath, 'labels', label_name)
            img = cv2.imread(img_path)
            label = cv2.imread(label_path)
            if label is not None:
                label = label[:,:,0]

        if dataset == "CHASEDB1":
            img_path = os.path.join(CHASEDB1DataSetPath, 'images', img_name[i].rstrip('\n'))
            label_name = img_name[i].rstrip('\n')[:-4] + '_1stHO.png'
            label_path = os.path.join(CHASEDB1DataSetPath, 'label', label_name)
            img = cv2.imread(img_path)
            label = cv2.imread(label_path)
            if debug:
                output_debug_image(img, logger,str.join(debugimages_path, f'debug_src_{i}.png')) # + os.path.basename(img_path))
                output_debug_image(label,  logger,str.join(debugimages_path, f'debug_src_label_{i}.png')) # + os.path.basename(label_path))
            if label is not None:
                label = label[:,:,0]
                
        if dataset == "ORIGA":
            img_path = os.path.join(ORIGADATASETPATH, img_name[i].rstrip('\n'))
            label_name = img_name[i].rstrip('\n')[:-4] + '.bmp'
            label_path = os.path.join(ORIGAMASKPATH, label_name)
            img = cv2.imread(img_path)
            label = cv2.imread(label_path)

            if label is not None:
                label = np.array(label)
                label = label[:,:,0]

        if dataset == "REFUGE":
            img_path = os.path.join(ORIGADATASETPATH, img_name[i].rstrip('\n'))
            label_name = img_name[i].rstrip('\n')[:-4] + '.bmp'
            label_path = os.path.join(ORIGAMASKPATH, label_name)
            img = cv2.imread(img_path)
            label = cv2.imread(label_path)

            if label is not None:
                label = np.array(label)
                label = label[:,:,0]

        img_shape.append(img.shape)

        label_ori_temp = label.copy()
        label_ori_temp[label_ori_temp < 1] = 0
        label_ori_temp[label_ori_temp >= 1] = 1
        label_ori.append(label_ori_temp)
        # 0 : Optic Cup     ----> Class 2
        # 128 : Optic Disc  ----> Class 1
        # 255 : BackGround
        #LabelTemp = np.zeros_like(label)
        #LabelTemp[label == 128] = 1
        #LabelTemp[label == 0] = 2

        img = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_AREA)
        label = cv2.resize(label, (img_size, img_size), interpolation=cv2.INTER_AREA)

        segmap = SegmentationMapsOnImage(label, shape=label.shape)
        seq = iaa.Sequential([
            #iaa.Dropout([0.05, 0.2]),  # drop 5% or 20% of all pixels
            #iaa.Sharpen((0.0, 1.0)),  # sharpen the image
            iaa.Affine(rotate=(0, 20)),  # rotate by -45 to 45 degrees (affects segmaps)
            iaa.Fliplr(0.5),  # horizontally flip 50% of all images
            iaa.Flipud(0.2),
            #iaa.ElasticTransformation(alpha =50, sigma=5)  # apply water effect (affects segmaps)
            iaa.GammaContrast((0.5, 2.0))
            #iaa.Sometimes(0.5,iaa.GaussianBlur(sigma=(0, 0.5))),
        ], random_order=True)

        #image_aug = rotate.augment_image(img)
        if flag == 'train':
            img, label = seq(image=img, segmentation_maps=segmap)
            label = np.squeeze(label.arr)

        #print("Augmented:")
        #ia.imshow(image_aug)
        #cv2.imshow('img', img)
        #cv2.imwrite('img.png', label*128)
        #.imwrite('img2.png', segmaps_aug_i.arr*255)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()

        imgGrey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        imgGrey = imgGrey[np.newaxis,:,:]

        img = np.transpose(img, [2, 0, 1])
        img = img.copy()
        img = Variable(torch.from_numpy(img)).float()
        imgGrey = Variable(torch.from_numpy(imgGrey)).double()

        if gpu:
            img = img.cuda()
            imgGrey = imgGrey.cuda()

        label, tmp_gt = get_label(label)
        label[label < 1] = 0
        label[label >= 1] = 1
        
        if debug:
            output_debug_image(img.cpu(), logger,str.join(debugimages_path, f'debug_preprocessed_img_{i}.png'))
            output_debug_image(label.cpu(), logger,str.join(debugimages_path,f'debug_preprocessed_label_{i}.png'))
        
        images.append(img)
        labels.append(label)
        tmp_gts.append(tmp_gt)
        imageGreys.append(imgGrey)

    images = torch.stack(images)
    imageGreys = torch.stack(imageGreys)

    labels = torch.stack(labels)
    tmp_gts = np.stack(tmp_gts)

    if flag:
        label_ori = np.stack(label_ori)

    return images, imageGreys, labels, tmp_gts, img_shape, label_ori

def calculate_Accuracy(confusion, logger,debug=False):
    confusion=np.asarray(confusion)
    pos = np.sum(confusion, 1).astype(np.float32) # 1 for row
    res = np.sum(confusion, 0).astype(np.float32) # 0 for coloum
    tp = np.diag(confusion).astype(np.float32)
    IU = tp / (pos + res - tp)
    meanIU = np.mean(IU)
    
    # Accuracy: sum of true positives / total sum
    Acc = np.sum(tp) / np.sum(confusion)
    Se = confusion[1][1] / (confusion[1][1]+confusion[0][1])
    Sp = confusion[0][0] / (confusion[0][0]+confusion[1][0])
    if debug:
        # print out everything
        logger.info(f'confusion: {confusion}')
        logger.info(f'pos: {pos}')
        logger.info(f'res: {res}')
        logger.info(f'tp: {tp}')
        logger.info(f'IU: {IU}')
        logger.info(f'meanIU: {meanIU}')
        logger.info(f'Acc: {Acc}')
        logger.info(f'Se: {Se}')
        logger.info(f'Sp: {Sp}')
    return  meanIU,Acc,Se,Sp,IU

def get_model(model_name):
    if model_name=='M_Net':
        return M_Net
    if model_name=='DG_Net':
        return DG_Net
    if model_name=='UNet512':
        return UNet512

#dataset_list = ['DRIVE', "STARE", "CHASEDB1", "ORIGA", "REFUGE"]
def get_img_list(dataset, SubID, flag='train'):

    if dataset == "DRIVE":
        if flag=='train':
            with open(os.path.join(DRIVEDataSetPath, "DRIVEtraining.txt"),'r') as f:
                img_list = f.readlines()
        else:
            with open(os.path.join(DRIVEDataSetPath, "DRIVEtesting.txt"),'r') as f:
                img_list = f.readlines()

    if dataset == "ORIGA":
        if flag=='train':
            with open(os.path.join(ODOCDATA, "TrainOriga.txt"),'r') as f:
                img_list = f.readlines()
        else:
            with open(os.path.join(ODOCDATA, "TestOriga.txt"),'r') as f:
                img_list = f.readlines()

    if dataset == "REFUGE":
        if flag=='train':
            with open(os.path.join(ODOCDATA, "TrainRefuge.txt"), 'r') as f:
                img_list = f.readlines()
        else:
            with open(os.path.join(ODOCDATA, "ValRefuge.txt"), 'r') as f:
                img_list = f.readlines()

    if dataset == "STARE":
        TrainFileName = "Staretraining" + str(SubID) + '.txt'
        TestFileName = "Staretesting" + str(SubID) + '.txt'

        if flag=='train':
            with open(os.path.join(STAREDataSetPath, TrainFileName), 'r') as f:
                img_list = f.readlines()
        else:
            with open(os.path.join(STAREDataSetPath, TestFileName), 'r') as f:
                img_list = f.readlines()

    if dataset == "CHASEDB1":
        if flag=='train':
            with open(os.path.join(CHASEDB1DataSetPath, "CHASEDB1training.txt"),'r') as f:
                img_list = f.readlines()
        else:
            with open(os.path.join(CHASEDB1DataSetPath, "CHASEDB1testing.txt"),'r') as f:
                img_list = f.readlines()

    return img_list
