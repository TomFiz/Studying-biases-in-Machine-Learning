import pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
import math
import copy
import io


import torch
import transformers
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from transformers import DistilBertModel, DistilBertTokenizer
from sklearn import metrics
import re
from torch import cuda

device = 'cuda' if cuda.is_available() else 'cpu'

#device ='cpu'  #FOR DEBUGGING

print('Device is',device)

#+ + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + +
#1) get saved data
#+ + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + +
infile = open('./TreatedData_medical_test_train.pk','rb')
SavedData = pickle.load(infile)
infile.close()

#X_train=SavedData["X"][:250000,:]
#X_test=SavedData["X"][250000:,:]
#Masks_train=SavedData["Masks"][:250000,:]
#Masks_test=SavedData["Masks"][250000:,:]
#S_train=SavedData["g"][:250000]
#S_test=SavedData["g"][250000:]
#y_train=SavedData["y"][:250000,:]
#y_test=SavedData["y"][250000:,:]


X_test=SavedData["X_test"]
Masks_test=SavedData["Masks_test"]
y_test=SavedData["y_test"]
S_test=SavedData["g_test"]
X_train=SavedData["X_train"]
Masks_train=SavedData["Masks_train"]
y_train=SavedData["y_train"]
S_train=SavedData["g_train"]

del SavedData


#+ + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + +
#2) define the model
#+ + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + +

list_jobs_used=['chiropractor','dentist','nurse','physician','surgeon']
class DistillBERTClass(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.distill_bert = DistilBertModel.from_pretrained("distilbert-base-uncased")
        self.drop = torch.nn.Dropout(0.3)
        self.pre_out = torch.nn.Linear(768, 100)
        self.out = torch.nn.Linear(100, len(list_jobs_used))

    def forward(self, ids, mask):
        distilbert_output = self.distill_bert(ids, mask)
        hidden_state = distilbert_output[0]  # (bs, seq_len, dim)
        pooled_output = hidden_state[:, 0]  # (bs, dim)
        output_1 = self.drop(pooled_output)
        output_2 = torch.relu(self.pre_out(output_1))
        output = torch.softmax(self.out(output_2),dim=1)
        return output



model = DistillBERTClass()
model.to(device)

#... test it
#
#torch.cuda.empty_cache()
#mb_X_d = X[[0,1,2,3],:].to(device)
#mb_Masks_d = Masks[[0,1,2,3],:].to(device)
#mb_y = y[[0,1,2,3],0].view(-1,1)
#outputs_d = model(ids=mb_X_d, mask=mb_Masks_d)
#outputs_h=outputs_d.to('cpu')
#print(outputs_h,mb_y)

#+ + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + +
#3) fit function
#+ + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + +


import sys
#sys.path.append('/home/laurent/Projects/2022_W2reg_package/')
sys.path.append('/projets/xnlp/2022_W2reg_package/')
from fit_NLP_model import *
from W2reg_core import *

#+ + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + +
#4) train
#+ + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + +


#tst_data={}
#tst_data['known']=True
#tst_data['X_test']=X_test
#tst_data['y_test']=y_test
#tst_data['S_test']=S_test
#
tst_data={}
tst_data['known']=False


#lambdavar=0.
#lambdavar=0.0000003
#lambdavar=0.0001
#lambdavar=0.0001
#lambdavar=0.001

EPOCHS_NB=3
#EPOCHS_NB=5


#lambdavar=0.001
#lambdavar=0.00102
#lambdavar=0.00103
#lambdavar=0.00107
lambdavar=0.00108


#with W2 regularisation:
#Lists_Results=W2R_fit_NLP(model,X_train[:,:],Masks_train[:,:],y_train, S_train.numpy(), lambdavar , f_loss_attach=nn.BCELoss() , EPOCHS = EPOCHS_NB, BATCH_SIZE = 8,obs_for_histo=16,DEVICE=device,ID_TreatedVars=[[8,4.],[5,8.],[1,1.]],optim_lr=0.0000005,DistBetween='Predictions',test_data=tst_data)


#without any regularisation:
lambdavar=0.
#Lists_Results=W2R_fit_NLP(model,X_train[:,:],Masks_train[:,:],y_train, S_train.numpy(), lambdavar , f_loss_attach=nn.BCELoss() , EPOCHS = EPOCHS_NB, BATCH_SIZE = 8,obs_for_histo=16,DEVICE=device,ID_TreatedVars=[[8,4.],[5,8.],[1,1.]],optim_lr=0.0000005,DistBetween='Predictions',test_data=tst_data)
Lists_Results=fit_NLP_model(model,X_train[:,:],Masks_train[:,:],y_train, EPOCHS = EPOCHS_NB, BATCH_SIZE = 6,DEVICE=device,optim_lr=0.00001)



plt.plot(Lists_Results['Loss'])
plt.savefig('l_convergence_Loss.pdf')  #show()
plt.clf()
#plt.plot(Lists_Results['W2'])
#plt.savefig('l'+str(lambdavar)+'_convergence_W2.pdf')  #show()
#plt.clf()





