import pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
#import seaborn as sns
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
#device = 'cuda' if cuda.is_available() else 'cpu'

device='cpu'

print('Device is',device)

#+ + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + +
#1) get saved data
#+ + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + +

#...test data
infile = open('./TreatedData_all3.pk','rb')
SavedData = pickle.load(infile)
infile.close()

X_test=SavedData["X"][250000:,:]
Masks_test=SavedData["Masks"][250000:,:]
y_test=SavedData["y"][250000:,:]

del SavedData

#... get occupations

data = pickle.load(open('BIOS.pkl','rb'))

Titles = ['' for i in range(len(data))]

for i in range(len(data)):
    Titles[i] = data[i]["title"]

del data



def CreateConversions_jobs_jobids(Titles):

  job_2_jobid={}
  jobid_2_job={}

  Titles_set=list(set(Titles))
  for i in range(len(Titles_set)):
    job_2_jobid[Titles_set[i]]=i
    jobid_2_job[i]=Titles_set[i]

  return [job_2_jobid,jobid_2_job]


[job_2_jobid,jobid_2_job]=CreateConversions_jobs_jobids(Titles)


column_surgeon=job_2_jobid["surgeon"]
column_nurse=job_2_jobid["nurse"]

raws_surgeon=torch.where(y_test[:,column_surgeon])[0]
raws_nurse=torch.where(y_test[:,column_nurse])[0]


#+ + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + +
#2) define and load the model
#+ + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + +

"""
class DistillBERTClass(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.distill_bert = DistilBertModel.from_pretrained("distilbert-base-uncased")
        self.drop = torch.nn.Dropout(0.3)
        self.out = torch.nn.Linear(768, 28)

    def forward(self, ids, mask):
        distilbert_output = self.distill_bert(ids, mask)
        hidden_state = distilbert_output[0]  # (bs, seq_len, dim)
        pooled_output = hidden_state[:, 0]  # (bs, dim)
        output_1 = self.drop(pooled_output)
        output = torch.softmax(self.out(output_1),dim=1)
        return output
"""

class DistillBERTClass(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.distill_bert = DistilBertModel.from_pretrained("distilbert-base-uncased")
        self.drop = torch.nn.Dropout(0.3)
        self.pre_out = torch.nn.Linear(768, 100)
        self.out = torch.nn.Linear(100, 28)

    def forward(self, ids, mask):
        distilbert_output = self.distill_bert(ids, mask)
        hidden_state = distilbert_output[0]  # (bs, seq_len, dim)
        pooled_output = hidden_state[:, 0]  # (bs, dim)
        output_1 = self.drop(pooled_output)
        output_2 = torch.relu(self.pre_out(output_1))
        output = torch.softmax(self.out(output_2),dim=1)
        return output


#    saved_models = { "normal": model.to('cpu') }
#    pickle.dump( saved_models, open( "saved_models.p", "wb" ) )
saved_models = pickle.load( open( "l0.0_MC_saved_model.p", "rb" ) )
model_cpu=saved_models["model"]
model=model_cpu.to(device)




"""
saved_models = { "normal": model_cpu }
pickle.dump( saved_models, open( "saved_models.p", "wb" ) )

# -> saved_models = pickle.load( open( "saved_models.p", "rb" ) )
# -> model_cpu=saved_models["normal"]
# -> model=model_cpu.to(device)
"""

#+ + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + +
#3) test
#+ + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + +

Curr_obsIDs=torch.cat([raws_nurse,raws_surgeon],0)


#Curr_obsIDs=raws_surgeon   # CHANGES THE OBSERVATIONS OF INTEREST



var_X_batch = X_test[Curr_obsIDs,:].to(device)
Masks_batch = Masks_test[Curr_obsIDs,:].to(device)
var_y_batch = y_test[Curr_obsIDs,:].float()  #added in the 1d case

with torch.no_grad():
  output = model(ids=var_X_batch[:,:150], mask=Masks_batch[:,:150])


y_pred=output.to('cpu')
y_true=var_y_batch

y_pred_labels=torch.argmax(y_pred,dim=1)
y_true_labels=torch.argmax(y_true,dim=1)

print('mean error:',torch.mean(torch.abs(y_pred-y_true)))

print('mean pred accuracy:',torch.mean(1.*(y_pred_labels==y_true_labels)))


#visual inspection of the main confusions

lst_jobs=list(jobid_2_job.values())

#... prediction probablilties for nurses and surgeons 

ax=plt.axes()

plt.imshow(y_pred.transpose(0,1))

ax.set_yticks(list(np.arange(len(lst_jobs))))
ax.set_yticklabels(lst_jobs)

plt.show()


#Pearson correlations of the predicted probabilities

PearsonCorr=np.corrcoef(y_pred.transpose(0,1).numpy())


ax=plt.axes()

#plt.imshow(PearsonCorr)
plt.imshow(np.abs(PearsonCorr))
plt.xticks(rotation=90)
ax.set_xticks(list(np.arange(len(lst_jobs))))
ax.set_xticklabels(lst_jobs)

ax.set_yticks(list(np.arange(len(lst_jobs))))
ax.set_yticklabels(lst_jobs)
plt.colorbar()
plt.show()


