from __future__ import print_function
from p3d_model import *
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import sampler
import pandas as pd

import torchvision.datasets as dset
import torchvision.transforms as T
import torch.nn.functional as F

import skvideo.io
import skvideo.datasets
import numpy as np

# import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import sampler
import pandas as pd

import torchvision.datasets as dset
import torchvision.transforms as T
import torch.nn.functional as F

import skvideo.io
import skvideo.datasets
import numpy as np

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

# LOADER FOR VIDEOS
class VideoBasketballDataset(Dataset):

    def __init__(self, csv_file, train=False, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        # randomize
        self.dataframe = pd.read_csv(csv_file)        
        self.transform = transform
        self.train = train

    def __len__(self):
        # return number of rows in csv file
        return self.dataframe.shape[0]

    def __getitem__(self, idx):
        vid_name = self.dataframe.iloc[idx, 0]
        action = self.dataframe.iloc[idx, 1]

        video = skvideo.io.vread(vid_name, num_frames=18)
        video = np.moveaxis(video, 3, 0)

        # should be MEAN, VAR
        if self.transform:
            MEAN, VAR = transform
            
            # normalize
            video -= MEAN
            video /= (VAR ** 0.5)

        return (video, action)


train_dataset = VideoBasketballDataset(csv_file='video-train.csv', train=True)
val_dataset = VideoBasketballDataset(csv_file='video-val.csv', train=True)
test_dataset = VideoBasketballDataset(csv_file='video-test.csv', train=False)

# couldn't normalize due to different image sizes
NUM_TRAIN = len(train_dataset)
NUM_VAL = len(val_dataset)
NUM_TEST = len(test_dataset)

print (NUM_TRAIN, NUM_VAL, NUM_TEST)

BATCH_SIZE = 8

loader_train = DataLoader(train_dataset, batch_size=BATCH_SIZE, 
                          sampler=sampler.SubsetRandomSampler(range(NUM_TRAIN)))
loader_val = DataLoader(val_dataset, batch_size=BATCH_SIZE, 
                        sampler=sampler.SubsetRandomSampler(range(NUM_VAL)))
loader_test = DataLoader(test_dataset, batch_size=BATCH_SIZE)

def get_pretrained_C3D():
    model = P3D199(pretrained=True, num_classes=400)
        
    # freeze layers
    for param in model.parameters():
        param.requires_grad = False

    # 11 classes instead of 400 classes
    model.fc = torch.nn.Linear(2048,11)
    
    if torch.cuda.is_available():
        print ('Using CUDA!\n')
        model = model.cuda()    
    
    return model

def custom_train(model, optimizer, epochs=1, verbose=False):
    """
    Train a model on CIFAR-10 using the PyTorch Module API.
    
    Inputs:
    - model: A PyTorch Module giving the model to train.
    - optimizer: An Optimizer object we will use to train the model
    - epochs: (Optional) A Python integer giving the number of epochs to train for
    
    Returns: Nothing, but prints model accuracies during training.
    """
    model = model.to(device=device)  # move the model parameters to CPU/GPU
    for e in range(epochs):
        for t, (x, y) in enumerate(loader_train):
            model.train()  # put model to training mode
            x = x.to(device=device, dtype=dtype)  # move to device, e.g. GPU
            y = y.to(device=device, dtype=torch.long)

            scores = model(x)
            loss = F.cross_entropy(scores, y)

            # Zero out all of the gradients for the variables which the optimizer
            # will update.
            optimizer.zero_grad()

            # This is the backwards pass: compute the gradient of the loss with
            # respect to each  parameter of the model.
            loss.backward()

            # Actually update the parameters of the model using the gradients
            # computed by the backwards pass.
            optimizer.step()

            if t % print_every == 0:
                print('Iteration %d, loss = %.4f' % (t, loss.item()))
                check_accuracy_part34(loader_train, model)
                print()

def check_accuracy_part34(loader, model, verbose=False):
    if loader.dataset.train:
        print('Checking accuracy on validation set')
    else:
        print('Checking accuracy on test set')   
    num_correct = 0
    num_samples = 0
    model.eval()  # set model to evaluation mode
    
    class_matrix = np.zeros((11, 11))
    
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device, dtype=dtype)  # move to device, e.g. GPU
            y = y.to(device=device, dtype=torch.long)
            scores = model(x)
            _, preds = scores.max(1)
            
            num_correct += (preds == y).sum()
            num_samples += preds.size(0)
            
            for i in range(len(y)):
                actual = y[i]
                pred = preds[i]
                
                class_matrix[pred][actual] += 1

        acc = float(num_correct) / num_samples
        print('Got %d / %d correct (%.2f)' % (num_correct, num_samples, 100 * acc))
        
        if verbose:
            print('------')
            for i in range(11):
                label = reverse_class_map[i]

                # recall, precision for each class
                # precision: correctly declared i out of all instances where the algorithm declared i
                # recall: correctly declared i out of all of the cases where the true state of the world is i.
                correct = class_matrix[i][i]
                num_predicted = np.sum(class_matrix[i,:]) 
                num_actual = np.sum(class_matrix[:, i])

                print ('%30s:    Recall = %3d / %3d = (%.2f)    Precision = %3d / %3d = (%.2f)' % (
                    label, correct, num_actual, correct / num_actual * 100, correct, num_predicted,
                    correct / num_predicted * 100))
        
    return class_matrix
    
    
USE_GPU = True

dtype = torch.float32 # we will be using float throughout this tutorial

if USE_GPU and torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

# Constant to control how frequently we print train loss
print_every = 20

print('using device:', device)

    
model = get_pretrained_C3D()

# fine tune last 2 layers
for param in model.layer4.parameters():
    param.requires_grad = True
    
learning_rate = 5e-5
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate, weight_decay=1e-08)

custom_train(model, optimizer, epochs=10)
