#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#%% Prepare data
from __future__ import unicode_literals, print_function, division
from io import open
import glob
import os
import numpy as np
from sklearn.model_selection import train_test_split

os.chdir("path")

def findFiles(path): return glob.glob(path)


all_letters = "abcdefghijklmnopqrstuvwxyzæøå0123456789éüäö "        # Alle letters små og store
n_letters = len(all_letters)                        # længden af "alle bogstaver"

import re
def remove_emojis(data):
    emoj = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
        u"\U00002500-\U00002BEF"  # chinese char
        u"\U00002702-\U000027B0"
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        u"\U0001f926-\U0001f937"
        u"\U00010000-\U0010ffff"
        u"\u2640-\u2642" 
        u"\u2600-\u2B55"
        u"\u200d"
        u"\u23cf"
        u"\u23e9"
        u"\u231a"
        u"\ufe0f"  # dingbats
        u"\u3030"
                      "]+", re.UNICODE)
    return re.sub(emoj, '', data)



# Open files
labels = open('path', 'r')
adult = open('path', 'r')

line_lable = labels.readline()
line_adult = adult.readline()

X=[]
y=[]

# read files
while line_lable != "" and line_adult != "":
    tekst=remove_emojis(line_adult[:-1]).lower()
    tekst= re.sub(r'[^\w\s]',' ',tekst)
    tekst=" ".join(tekst.split())
    #tekst= re.sub(' +',' ',tekst)
    tekst= re.sub('_','',tekst)
    if tekst!='':
        if tekst[0]==" ":
            tekst=tekst[1:] 
        X.append(tekst)   
        y.append(line_lable[:-1])
            
    line_lable = labels.readline()
    line_adult = adult.readline()
labels.close()
adult.close()

# Divide into train and validation
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0,shuffle=True)
training_dict = {}  
test_dict = {}  
all_categories = [] 

# create test and train dictionarys
for i in range(len(X_train)):

    if y_train[i] not in training_dict:
        training_dict[y_train[i]]=[X_train[i]]
        all_categories.append(y_train[i])
    else:
        training_dict[y_train[i]].append(X_train[i])
        
for i in range(len(X_test)):

    if y_test[i] not in test_dict:
        test_dict[y_test[i]]=[X_test[i]]
    else:
        test_dict[y_test[i]].append(X_test[i])
        
# Number of diffrent categories
n_categories = len(all_categories)                

### Turning lines into tensors

import torch

# Find letter index from all_letters, e.g. "a" = 0
def letterToIndex(letter):
    if letter not in all_letters:
        raise ValueError("--"+letter+"--")
    return all_letters.find(letter)

# Just for demonstration, turn a letter into a <1 x n_letters> Tensor
def letterToTensor(letter):
    tensor = torch.zeros(1, n_letters)
    tensor[0][letterToIndex(letter)] = 1
    return tensor

# Turn a line into a <line_length x 1 x n_letters>,
# or an array of one-hot letter vectors
def lineToTensor(line):
    tensor = torch.zeros(len(line), 1, n_letters)
    for li, letter in enumerate(line):
        tensor[li][0][letterToIndex(letter)] = 1
    return tensor



    
#%% Creating the Network 1 hidden layer

import torch.nn as nn
import torch.nn.functional as F

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()

        self.hidden_size = hidden_size

        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        
        self.i2o = nn.Linear(input_size + hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        combined = torch.cat((input, hidden), 1)
        hidden = self.i2h(combined)
        output = self.i2o(combined)
        output = self.softmax(output)
        return output, hidden

    def initHidden(self):                           # initializing the hidden units
        return torch.zeros(1, self.hidden_size)


n_hidden = 150
# make network
#RNN(#input,#hidden,#categories)
rnn = RNN(n_letters,n_hidden,n_categories)


#%% network with 2 hidden layers


class RNN2(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN2, self).__init__()

        self.hidden_size = hidden_size

        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.h2h = nn.Linear(hidden_size + hidden_size, hidden_size)
        
        self.h2o = nn.Linear(hidden_size + hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden1, hidden2):
        
        # first layer
        combined = torch.cat((input, hidden1), 1)
        hidden1 = self.i2h(combined)
        hidden1_relu = F.relu(hidden1)
        
        # second layer
        combined2 = torch.cat((hidden1_relu, hidden2), 1)
        hidden2 = self.h2h(combined2)
    
        # outputs
        output = self.h2o(combined2)
        output = self.softmax(output)
        return output, hidden1, hidden2

    def initHidden(self):
        return torch.zeros(1, self.hidden_size)

# make network

rnn2 = RNN2(n_letters,n_hidden,n_categories)

#%% network with 3 hidden layers

class RNN3(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN3, self).__init__()

        self.hidden_size = hidden_size

        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.h2h = nn.Linear(hidden_size + hidden_size, hidden_size)
        self.h2h2 = nn.Linear(hidden_size + hidden_size, hidden_size)
        
        self.h2o = nn.Linear(hidden_size + hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden1, hidden2, hidden3):
        
        # first layer
        combined = torch.cat((input, hidden1), 1)
        hidden1 = self.i2h(combined)
        hidden1_relu = F.relu(hidden1)
        
        # second layer
        combined2 = torch.cat((hidden1_relu, hidden2), 1)
        hidden2 = self.h2h(combined2)
        hidden2_relu = F.relu(hidden2)
        
        # third layer
        combined3 = torch.cat((hidden2_relu, hidden3), 1)
        hidden3 = self.h2h2(combined3)
    
        # outputs
        output = self.h2o(combined3)
        output = self.softmax(output)
        return output, hidden1, hidden2, hidden3

    def initHidden(self):
        return torch.zeros(1, self.hidden_size)



# make network
rnn3 = RNN3(n_letters,n_hidden,n_categories)
#%% Training
# Preparing for Training
# Output = category , index of category
def categoryFromOutput(output):                     
    top_n, top_i = output.topk(1)                   # Get higest output value and its index
    category_i = top_i[0].item()                    # index of higest number
    return all_categories[category_i], category_i


import random
random.seed(9001)
#
def randomChoice(l):
    return l[random.randint(0, len(l) - 1)]

#
def randomTrainingExample():
    category = randomChoice(all_categories)         # Chose random from list
    line = randomChoice(training_dict[category])   # Choose random name from the random chosen country list (from dictionary)
    category_tensor = torch.tensor([all_categories.index(category)], dtype=torch.long) # Get index of tensor
    
    line_tensor = lineToTensor(line)                # Line to tensor
    return category, line, category_tensor, line_tensor

def randomTestExample():
    category = randomChoice(all_categories)         # Chose random from list
    line = randomChoice(test_dict[category])   # Choose random name from the random chosen country list (from dictionary)
    category_tensor = torch.tensor([all_categories.index(category)], dtype=torch.long) # Get index of tensor
    
    line_tensor = lineToTensor(line)                # Line to tensor
    return category, line, category_tensor, line_tensor



#%%
from torch.optim import Adam
criterion = nn.NLLLoss()
learning_rate = 0.001

optim = Adam(rnn.parameters(), lr=learning_rate, weight_decay=0)

def train(category_tensor, line_tensor, batchsize):
    output=[]                                                       # get all outputs to calculate missclassification rate
    Batch_loss=0
    optim.zero_grad()                                               # set gradient to zero, otherwise it will accumulate
    for b in range(batchsize):
        hidden = rnn.initHidden()                                   # initialize the hidden units
        for i in range(line_tensor[b].size()[0]):                   # loop over number of letters in name
            output_new, hidden = rnn(line_tensor[b][i], hidden) # get output for each letter and update hidden

        loss = criterion(output_new, category_tensor[b])
        Batch_loss+=loss
        output.append(output_new)
    Batch_loss.backward()   

    # clip gradient if norm is bigger than 50
    _ = torch.nn.utils.clip_grad_norm_(rnn.parameters(), 50)
    optim.step()

    return output, (Batch_loss.item())/batchsize

#%%
from torch.optim import Adam
criterion = nn.NLLLoss()
learning_rate = 0.001
optim2 = Adam(rnn2.parameters(), lr=learning_rate, weight_decay=0)

def train2(category_tensor, line_tensor, batchsize):
    output=[]                                                       # get all outputs to calculate missclassification rate
    Batch_loss=0
    optim2.zero_grad()                                               # set gradient to zero, otherwise it will accumulate
    for b in range(batchsize):
        hidden1 = rnn2.initHidden()
        hidden2 = rnn2.initHidden()                                    # initialize the hidden units
        for i in range(line_tensor[b].size()[0]):                   # loop over number of letters in name
            output_new, hidden1, hidden2 = rnn2(line_tensor[b][i], hidden1, hidden2) # get output for each letter and update hidden

        loss = criterion(output_new, category_tensor[b])
        Batch_loss+=loss
        output.append(output_new)
    Batch_loss.backward()   

    # clip gradient if norm is bigger than 50
    _ = torch.nn.utils.clip_grad_norm_(rnn2.parameters(), 50)
    optim2.step()

    return output, (Batch_loss.item())/batchsize

#%%

optim3 = Adam(rnn3.parameters(), lr=learning_rate, weight_decay=0)

def train3(category_tensor, line_tensor, batchsize):
    output=[]                                                       # get all outputs to calculate missclassification rate
    Batch_loss=0
    optim3.zero_grad()                                               # set gradient to zero, otherwise it will accumulate
    for b in range(batchsize):
        hidden1 = rnn3.initHidden()  
        hidden2 = rnn3.initHidden()
        hidden3 = rnn3.initHidden()                                 # initialize the hidden units
        for i in range(line_tensor[b].size()[0]):                   # loop over number of letters in name
            output_new, hidden1, hidden2, hidden3 = rnn3(line_tensor[b][i], hidden1, hidden2, hidden3 ) # get output for each letter and update hidden

        loss = criterion(output_new, category_tensor[b])
        Batch_loss+=loss
        output.append(output_new)
    Batch_loss.backward()   

    # clip gradient if norm is bigger than 50
    _ = torch.nn.utils.clip_grad_norm_(rnn3.parameters(), 50)
    optim3.step()

    return output, (Batch_loss.item())/batchsize




#%%
def evaluate(line_tensor):
    hidden = rnn.initHidden()

    for i in range(line_tensor.size()[0]):
        output, hidden = rnn(line_tensor[i], hidden)

    return output  


def evaluate2(line_tensor):
    hidden1 = rnn2.initHidden()
    hidden2 = rnn2.initHidden()

    for i in range(line_tensor.size()[0]):
        output, hidden1, hidden2 = rnn2(line_tensor[i], hidden1, hidden2)

    return output  

def evaluate3(line_tensor):
    hidden1 = rnn2.initHidden()
    hidden2 = rnn2.initHidden()
    hidden3 = rnn2.initHidden()

    for i in range(line_tensor.size()[0]):
        output, hidden1, hidden2, hidden3 = rnn3(line_tensor[i], hidden1, hidden2, hidden3)

    return output 

#%%
import time
import math


n_iters = 100000
print_every = 500
plot_every = 1000
n_test = 750
batchsize = 5
# Keep track of losses for plotting

# get time
def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

# start time for training
start = time.time()

current_loss = [0,0,0]
all_losses = [[],[],[]]
all_test_losses=[[],[],[]]
acc_train=[[],[],[]]
acc_test=[[],[],[]]
ClassEcount_train=[0,0,0]

for iter in range(1, n_iters + 1):                                              # loop throug n iterations
    categorylist, linelist, category_tensorlist, line_tensorlist=[],[],[],[]    # Get random training example
    for i in range(batchsize):
        category, line, category_tensor, line_tensor = randomTrainingExample()  # Get random training example
        categorylist.append(category)
        linelist.append(line)
        category_tensorlist.append(category_tensor)
        line_tensorlist.append(line_tensor)
    
    
    output, loss = train(category_tensorlist, line_tensorlist, batchsize) # Train
    current_loss[0] += loss
    
    output2, loss2 = train2(category_tensorlist, line_tensorlist, batchsize) # Train
    current_loss[1] += loss2
    
    output3, loss3 = train3(category_tensorlist, line_tensorlist, batchsize) # Train
    current_loss[2] += loss3
    
    # calculate missclassificationrate
    for batch in range(batchsize):
        guess, guess_i = categoryFromOutput(output[batch])
        ClassEcount_train[0] += 1 if guess != categorylist[batch] else 0
        guess2, guess_i2 = categoryFromOutput(output2[batch])
        ClassEcount_train[1] += 1 if guess2 != categorylist[batch] else 0
        guess3, guess_i3 = categoryFromOutput(output3[batch])
        ClassEcount_train[2] += 1 if guess3 != categorylist[batch] else 0
        

    # Print iter number, loss, name and guess
    if iter % print_every == 0:
        guess2, guess_i2 = categoryFromOutput(output2[1])
        correct = '✓' if guess2 == category else '✗ (%s)' % category
        print('%d %d%% (%s) %.4f %s / %s %s' % (iter, iter / n_iters * 100, timeSince(start), loss2, line, guess2, correct))

    # Add current loss avg to list of losses
    if iter % plot_every == 0:
        all_losses[0].append(current_loss[0] / plot_every)                                    # loss
        acc_train[0].append(ClassEcount_train[0]/(iter*batchsize))                            # missclassification rate
    
        
        all_losses[1].append(current_loss[1] / plot_every)                                    # loss
        acc_train[1].append(ClassEcount_train[1]/(iter*batchsize))                            # missclassification rate
        
        all_losses[2].append(current_loss[2] / plot_every)                                    # loss
        acc_train[2].append(ClassEcount_train[2]/(iter*batchsize))                            # missclassification rate

        
        current_loss_test=[0,0,0]
        ClassEcount_test=[0,0,0]
        current_loss = [0,0,0]
        
        # model validation
        for val in range(n_test):
            category_test, line_test, category_tensor_test, line_tensor_test = randomTestExample()
            
            output_test = evaluate(line_tensor_test)  
            guess_test, guess_i_test = categoryFromOutput(output_test)
            ClassEcount_test[0] += 1 if guess_test != category_test else 0                 # missclassification rate
            loss_test = criterion(output_test, category_tensor_test)                    # loss
            current_loss_test[0]+=loss_test.item()
            
            output_test2 = evaluate2(line_tensor_test)  
            guess_test2, guess_i_test2 = categoryFromOutput(output_test2)
            ClassEcount_test[1] += 1 if guess_test2 != category_test else 0                 # missclassification rate
            loss_test2 = criterion(output_test2, category_tensor_test)                    # loss
            current_loss_test[1]+=loss_test2.item()
            
            output_test3 = evaluate3(line_tensor_test)  
            guess_test3, guess_i_test3 = categoryFromOutput(output_test3)
            ClassEcount_test[2] += 1 if guess_test3 != category_test else 0                 # missclassification rate
            loss_test3 = criterion(output_test3, category_tensor_test)                    # loss
            current_loss_test[2]+=loss_test3.item()
            
        all_test_losses[0].append(current_loss_test[0] / n_test)    
        acc_test[0].append(ClassEcount_test[0]/n_test)
        
        all_test_losses[1].append(current_loss_test[1] / n_test)    
        acc_test[1].append(ClassEcount_test[1]/n_test)
        
        all_test_losses[2].append(current_loss_test[2] / n_test)    
        acc_test[2].append(ClassEcount_test[2]/n_test)
        

#%% Plot loses
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
iterationlist = [*range(1,n_iters+1,plot_every)]
plt.figure()

plt.xlabel('iterations')
plt.ylabel('loss')
plt.plot(iterationlist,all_losses[0],'r', label='1HL train')
plt.plot(iterationlist,all_test_losses[0],'r-.',label='1HL validation')
plt.plot(iterationlist,all_losses[1],'b', label='2HL train')
plt.plot(iterationlist,all_test_losses[1],'b-.',label='2HL validation')
plt.plot(iterationlist,all_losses[2],'g', label='3 HL train')
plt.plot(iterationlist,all_test_losses[2],'g-.',label='3HL validation')
plt.legend(loc='upper left', prop={'size': 8})
#%%
plt.figure()

plt.xlabel('iterations')
plt.ylabel('missclassification rate')
plt.plot(iterationlist,acc_train[0],'r', label='1HL train')
plt.plot(iterationlist,acc_test[0],'r-.',label='1HL validation')
plt.plot(iterationlist,acc_train[1],'b', label='2HL train')
plt.plot(iterationlist,acc_test[1],'b-.',label='2HL validation')
plt.plot(iterationlist,acc_train[2],'g', label='3HL train')
plt.plot(iterationlist,acc_test[2],'g-.',label='3HL validation')
plt.legend(loc='lower left', prop={'size': 8})


#%% 

min_loss_test=[min(all_test_losses[0]),min(all_test_losses[1]),min(all_test_losses[2])]
index_min_loss=[all_test_losses[0].index(min(all_test_losses[0])),all_test_losses[1].index(min(all_test_losses[1])),all_test_losses[2].index(min(all_test_losses[2]))]
min_loss_train=[all_losses[0][index_min_loss[0]],all_losses[1][index_min_loss[1]],all_losses[2][index_min_loss[2]]]

index_los_iter=[iterationlist[all_test_losses[0].index(min(all_test_losses[0]))],iterationlist[all_test_losses[1].index(min(all_test_losses[1]))],iterationlist[all_test_losses[2].index(min(all_test_losses[2]))]]
#%%
min_misclass_test=[min(acc_test[0]),min(acc_test[1]),min(acc_test[2])]
index_min_misclass=[acc_test[0].index(min(acc_test[0])),acc_test[1].index(min(acc_test[1])),acc_test[2].index(min(acc_test[2]))]
min_misclass_train=[acc_train[0][index_min_misclass[0]],acc_train[1][index_min_misclass[1]],acc_train[2][index_min_misclass[2]]]

index_misclass_iter=[iterationlist[acc_test[0].index(min(acc_test[0]))],iterationlist[acc_test[1].index(min(acc_test[1]))],iterationlist[acc_test[2].index(min(acc_test[2]))]]



