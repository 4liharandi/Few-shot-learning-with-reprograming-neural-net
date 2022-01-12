
!pip install easyfsl
import torch
import time
import glob
import requests
import cv2
import time
import PIL.Image
import urllib
from PIL import Image as im
from torchvision import models
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import skimage.io
from tqdm import tqdm
from torch import nn, optim
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torchvision import transforms
from torchvision import transforms
from skimage.transform import resize
from torch.utils.data import DataLoader
from torchvision.datasets import Omniglot, CIFAR10
from torchvision.models import resnet18
from easyfsl.data_tools import TaskSampler
from torchvision.transforms import Normalize
from sklearn.metrics import roc_auc_score
from easyfsl.utils import plot_images, sliding_average