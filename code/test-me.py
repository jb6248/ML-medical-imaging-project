import torch
import torch.nn as nn
import torch.nn.functional as F

import sklearn.metrics as metrics
from sklearn.metrics import roc_auc_score

import os
import argparse
import time
from core.utils import calculate_Accuracy, get_img_list, get_model, get_data
from pylab import *
import random
import warnings
warnings.filterwarnings('ignore')
torch.set_warn_always(False)
plt.switch_backend('agg')


# save a numpy array as an image
img_test = np.array([[[255, 0, 0]] * 100] * 100, dtype='uint8')
print(img_test.shape)
plt.imsave('name.png', img_test)
