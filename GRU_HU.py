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


#%% load and preprocess data
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



#Open files
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


#%% divide into validation and train
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

#%% Turning lines into tensors

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


    
#%% Creating the Network

import torch.nn as nn

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()

        self.hidden_size = hidden_size
        self.gru = nn.GRU(input_size, hidden_size,1)
        
        self.out = nn.Linear(hidden_size, output_size) 
        
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        
        output, hidden = self.gru(input, hidden)
        output = self.out(output[0])
        output = self.softmax(output)
        return output, hidden

    def initHidden(self):                           # initializing the hidden units
        return torch.zeros(1,1, self.hidden_size)


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


#%% train
from torch.optim import Adam
criterion = nn.NLLLoss()
learning_rate = 0.001


def train(category_tensor, line_tensor, batchsize):
    output=[]                                                       # get all outputs to calculate missclassification rate
    Batch_loss=0
    optim.zero_grad()                                               # set gradient to zero, otherwise it will accumulate
    for b in range(batchsize):
        hidden = rnn.initHidden()                                   # initialize the hidden units
        for i in range(line_tensor[b].size()[0]):                   # loop over number of letters in name
            output_new, hidden = rnn(line_tensor[b][i:i+1], hidden) # get output for each letter and update hidden

        loss = criterion(output_new, category_tensor[b])
        Batch_loss+=loss
        output.append(output_new)
    Batch_loss.backward()   

    # clip gradient if norm is bigger than 50
    _ = torch.nn.utils.clip_grad_norm_(rnn.parameters(), 50)
    optim.step()

    return output, (Batch_loss.item())/batchsize



#%%
def evaluate(line_tensor):
    hidden = rnn.initHidden()

    for i in range(line_tensor.size()[0]):
        output, hidden = rnn(line_tensor[i:i+1], hidden)

    return output  
#%%
import time
import math


n_iters = 50000
print_every = 5000
plot_every = 1000
n_test = 750
batchsize = 5
# Keep track of losses for plotting

n_hidden = [*range(50,401,50)]

# get time
def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

# start time for training
start = time.time()

Loss_test_hidden=[]
Loss_train_hidden=[]
miss_test_hidden=[]
miss_train_hidden=[]

for num_hidden in n_hidden:
    turn_Loss_test_hidden=[]
    turn_Loss_train_hidden=[]
    turn_miss_test_hidden=[]
    turn_miss_train_hidden=[]
    
    for turn in range(3):
        current_loss = 0
        all_losses = []
        rnn = RNN(n_letters,num_hidden,n_categories)
        optim = Adam(rnn.parameters(), lr=learning_rate, weight_decay=0)
        all_test_losses=[]
        acc_train=[]
        acc_test=[]
        ClassEcount_train=0
    
        
        for iter in range(1, n_iters + 1):                                              # loop throug n iterations
            categorylist, linelist, category_tensorlist, line_tensorlist=[],[],[],[]    # Get random training example
            for i in range(batchsize):
                category, line, category_tensor, line_tensor = randomTrainingExample()  # Get random training example
                categorylist.append(category)
                linelist.append(line)
                category_tensorlist.append(category_tensor)
                line_tensorlist.append(line_tensor)
            
            output, loss = train(category_tensorlist, line_tensorlist, batchsize) # Train
            current_loss += loss
            # calculate missclassificationrate
            for batch in range(batchsize):
                guess, guess_i = categoryFromOutput(output[batch])
                ClassEcount_train += 1 if guess != categorylist[batch] else 0
                
        
            # Print iter number, loss, name and guess
            if iter % print_every == 0:
                guess, guess_i = categoryFromOutput(output[1])
                correct = '✓' if guess == category else '✗ (%s)' % category
                print('%d %d%% (%s) %.4f %s / %s %s' % (iter, iter / n_iters * 100, timeSince(start), loss, line, guess, correct))
        
            # Add current loss avg to list of losses
            if iter % plot_every == 0:
                all_losses.append(current_loss / plot_every)                                    # loss
                current_loss = 0
                current_loss_test=0
                acc_train.append(ClassEcount_train/(iter*batchsize))                            # missclassification rate
                ClassEcount_test=0
                # model validation
                for val in range(n_test):
                    category_test, line_test, category_tensor_test, line_tensor_test = randomTestExample()
                    output_test = evaluate(line_tensor_test)  
                    guess_test, guess_i_test = categoryFromOutput(output_test)
                    ClassEcount_test += 1 if guess_test != category_test else 0                 # missclassification rate
                    loss_test = criterion(output_test, category_tensor_test)                    # loss
                    current_loss_test+=loss_test.item()
                all_test_losses.append(current_loss_test / n_test)    
                acc_test.append(ClassEcount_test/n_test)
    
        index_loss=all_test_losses.index(min(all_test_losses))
        index_miss=acc_test.index(min(acc_test))
        turn_Loss_test_hidden.append(all_test_losses[index_loss])
        turn_Loss_train_hidden.append(all_losses[index_loss])
        turn_miss_test_hidden.append(acc_test[index_miss])
        turn_miss_train_hidden.append(acc_train[index_miss])
        
    Loss_test_hidden.append(np.mean(turn_Loss_test_hidden))
    Loss_train_hidden.append(np.mean(turn_Loss_train_hidden))
    miss_test_hidden.append(np.mean(turn_miss_test_hidden))
    miss_train_hidden.append(np.mean(turn_miss_train_hidden))
    
#%% Plot loses
import matplotlib.pyplot as plt

plt.figure()

plt.xlabel('n_hidden')
plt.ylabel('loss')
plt.plot(n_hidden,Loss_train_hidden, label='train')
plt.plot(n_hidden,Loss_test_hidden,label='validation')
plt.legend()

plt.figure()

plt.xlabel('n_hidden')
plt.ylabel('missclassification rate')
plt.plot(n_hidden,miss_train_hidden, label='train')
plt.plot(n_hidden,miss_test_hidden,label='validation')
plt.legend()


