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
import numpy as np
import skimage.io
from tqdm import tqdm
from torch import nn, optim
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torchvision import transforms
from torchvision.transforms import Normalize
from sklearn.metrics import roc_auc_score
from easyfsl.utils import plot_images, sliding_average
import seaborn as sns
from sklearn.manifold import TSNE
import pandas as pd 
from Adversarial_Reprogramming import Program, Advprogram
from Load_model_weight import *


FLAGS, unparsed = flags()

num_epochs = FLAGS.num_epochs
lr = FLAGS.lr
num_ensemble = FLAGS.num_ensemble

N_WAY = FLAGS.way # Number of classes in a task
N_SHOT = FLAGS.shot # Number of images per class in the support set
N_QUERY = FLAGS.query # Number of images per class in the query set
N_EVALUATION_TASKS = FLAGS.num_tasks # Number of episodes

test_set.labels = [
    instance[1] for instance in test_set._flat_character_images
]

test_sampler = TaskSampler(
    test_set, 
    n_way=N_WAY, 
    n_shot=N_SHOT, 
    n_query=N_QUERY, 
    n_tasks=N_EVALUATION_TASKS,
)

test_loader = DataLoader(
    test_set,
    batch_sampler=test_sampler,
    num_workers=2,
    pin_memory=True,
    collate_fn=test_sampler.episodic_collate_fn,
)


wd=0.00
lmd= 0.01
decay_step=2
lr_decay=0.96

for param in model.parameters():    
    param.requires_grad = False

device = torch.device('cuda')
model = model.to(device)

    
def run_epoch(mode,glob_blur_sigma, train_images, train_labels, test_images, test_labels, device = device, num_classes =15, optimizer=None, epoch=None, loss_criterion=None):
       
    
    if mode == 'train':

        adv_program.train()
    else:
        adv_program.eval()

    train_images = train_images.to(device)
    train_labels = train_labels.to(device)
    test_images = test_images.to(device)
    test_labels = test_labels.to(device)
    
    steps_per_epoch = 1
    reg_loss = 0
    for param in adv_program.parameters():
        reg_loss += torch.norm(param)
    lam =  0.01
    
    if mode == 'train':
        
        optimizer.zero_grad()

        x = adv_program(train_images)
        logits = model(x)
        logits = logits[:,:num_classes]
        batch_loss = loss_criterion(logits, train_labels) + lam * reg_loss
        batch_loss.backward()
        
        with torch.no_grad():
            blur(adv_program.program.weight.grad, sigma=glob_blur_sigma, size=5, padding=2)
        
        optimizer.step()
        
        with torch.no_grad():
            blur(adv_program.program.weight, sigma=glob_blur_sigma, size=5, padding=2)

        labels_pred = torch.argmax(torch.softmax(logits,dim=1), dim=1)
        
        dis = torch.softmax(logits,dim=1)
        accuracy = torch.sum(train_labels==labels_pred).item()/(train_labels.shape[0])
        
 
        
    else:
        
        with torch.no_grad():
                
            x= adv_program(test_images)
            logits = model(x)
            logits = logits[:,:num_classes]
            batch_loss = loss_criterion(logits,test_labels)
            dis = torch.softmax(logits,dim=1)
            labels_pred = torch.argmax(torch.softmax(logits,dim=1), dim=1)            
            accuracy = torch.sum(test_labels==labels_pred).item()/(test_labels.shape[0])
            if epoch ==99:
                plot_images(test_images.cpu(), "query images", images_per_row=N_QUERY)
                for i in range(len(test_labels)):
                    if test_labels[i]!=labels_pred[i]:
                        print('data {} wrong from {} --> {}'.format(i,test_labels[i],labels_pred[i]))
                print(test_labels)
                print(dis)


    loss = batch_loss.item()
    
    return {'loss': loss, 'accuracy': accuracy, 'label': labels_pred, 'dis' : dis, 'logit': logits}


N_EVALUATION_TASKS = 100

sum_avg_dis = 0
sum_weighted_avg_acc=0
sum_max_acc = 0
sum_best_acc = 0

for i in range(N_EVALUATION_TASKS):

    best_acc = 0
    av = 0
    
    num_ens =10
    lab = torch.zeros((50,num_ens))

    avgd=0
    avgdw=0
        
    (
    example_support_images,
    example_support_labels,
    example_query_images,
    example_query_labels,
    example_class_ids,
    ) = next(iter(test_loader))

    

    for k in range(num_ens):
        
        adv_program = AdvProgram(img_size, pimg_size, mask_size, device=device)
        adv_program = adv_program.to(device)
        optimizer = optim.Adam(adv_program.parameters(), lr=lr, weight_decay=wd)
        lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=lr_decay)
        loss_criterion = nn.CrossEntropyLoss()
        loss_criterion = loss_criterion.to(device)

        acc_train = []
        acc_test = []
        loss_train = []
        loss_test = [] 
            
        epoch = 0
        
        test_images = example_query_images
        test_labels = example_query_labels
        train_images = example_support_images
        train_labels = example_support_labels
        
        glob_blur_sigma = 0.3
        glob_blur_sigma_factor = 0.9

        while epoch < num_epochs:

            train_metrics = run_epoch('train',glob_blur_sigma, train_images, train_labels, test_images, test_labels, device,  5,optimizer, epoch=epoch, loss_criterion=loss_criterion)
            glob_blur_sigma *= glob_blur_sigma_factor
            test_metrics = run_epoch('valid',glob_blur_sigma, train_images, train_labels, test_images, test_labels, device, 5, epoch=epoch, loss_criterion=loss_criterion)
            loss_train.append(train_metrics['loss'])
            acc_train.append(train_metrics['accuracy'])
            loss_test.append(test_metrics['loss'])
            acc_test.append(test_metrics['accuracy'])
            #print('acc train is : {} and acc test is : {}'.format(train_metrics['accuracy'],test_metrics['accuracy']))

            epoch += 1
            lr_scheduler.step()
        
        lab[:,k] = test_metrics['label']
        
        avgd += test_metrics['dis']
        avgdw += test_metrics['accuracy'] * test_metrics['dis']

        av += test_metrics['accuracy']
        
        if test_metrics['accuracy'] > a:
                
            best_acc = test_metrics['accuracy']
                
    train_labels = train_labels.to(device)
    test_labels = test_labels.to(device)
    test_images = test_images.to(device)
    
    print('*************************')
    print('result episode {}'.format(i+1))
    print('**********')
    print('best weight is: {}'.format(a))

    max_lab = torch.mode(lab,dim = -1)[0]
    
    max_lab = max_lab.to(device)
    max_acc = torch.sum(test_labels==max_lab).item()/(test_labels.shape[0])
    
    avgd = avgd/num_ens
    
    avgdw = avgdw/num_ens
    
    labe = torch.argmax(avgd, dim=1)
    labei = torch.argmax(avgdw, dim=1)
    
    labe = labe.to(device)
    labei = labei.to(device)

    
    acc_avgd = torch.sum(test_labels==labe).item()/(test_labels.shape[0])
    acc_avgdw = torch.sum(test_labels==labei).item()/(test_labels.shape[0])
    print('max rep is: {}'.format(max_acc))
    print('avg diss is: {}'.format(acc_avgd))
    print('weighted diss is: {}'.format(acc_avgdw))
    sum_max_acc+=max_acc
    sum_weighted_avg_acc+=acc_avgdw
    sum_best_acc += best_acc
    sum_avg_dis +=acc_avgd
    
print('*************************')
print('final result ')
print('**********')
print('avg best acc {} '.format(avg_best_acc / N_EVALUATION_TASKS))
print('avg max rep acc {} '.format(sum_max_acc / N_EVALUATION_TASKS))
print('avg dis acc {} '.format(sum_avg_dis / N_EVALUATION_TASKS))
print('avg weighted dis rep acc {} '.format(sum_weighted_avg_acc / N_EVALUATION_TASKS))
