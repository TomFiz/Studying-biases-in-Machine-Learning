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
device = 'cuda' if cuda.is_available() else 'cpu'


#device='cpu'

print('Device is',device)

#+ + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + +
#1) get saved data
#+ + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + +

#...test data
#infile = open('./TreatedData_all5.pk','rb')
infile = open('./TreatedData_medical_test_train.pk','rb')
SavedData = pickle.load(infile)
infile.close()


X_test=SavedData["X_test"]
Masks_test=SavedData["Masks_test"]
y_test=SavedData["y_test"]
S_test=SavedData["g_test"]
bio_test=SavedData["bio_test"]



job_2_jobid=SavedData['job_2_jobid']
jobid_2_job=SavedData['jobid_2_job']
 
list_jobs_used=['chiropractor','dentist','nurse','physician','surgeon']
del SavedData



#+ + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + +
#2) define and load the model
#+ + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + +


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



#No Reg:
#"./Res18_Multi_TP1/l0.0001_saved_model.p" *
#"./Res22_Multi_TP_1_5_8/l1e-07_saved_model.p" *
#"./Res22_Multi_TP_1_5_8/l2e-07_saved_model.p" *
#"./Res22_Multi_TP_1_5_8/l3e-07_saved_model.p" * 
#"./Res22_Multi_TP_1_5_8/l1.1e-06_saved_model.p" *

#Reg 1:
#"./Res17_Multi_TP1/l1e-09_saved_model.p" *
#"./Res19_Multi_TP1/l0.001_saved_model.p" *
#"./Res20_Multi_TP1/l0.001_saved_model.p" *
#"./Res20_Multi_TP1/l0.00112_saved_model.p" *
#"./Res20_Multi_TP1/l0.00113_saved_model.p" *

#Reg 1 5 8:
#"./Res21_Multi_TP_1_5_8/l0.001_saved_model.p" *
#"./Res21_Multi_TP_1_5_8/l0.00101_saved_model.p" *
#"./Res21_Multi_TP_1_5_8/l0.00102_saved_model.p" * 
#"./Res21_Multi_TP_1_5_8/l0.00103_saved_model.p" *
#"./Res21_Multi_TP_1_5_8/l0.0011_saved_model.p" *

#Reg 1 5w 8w:
#"./Res23_Multi_TP_1_5w_8w/l0.00104_saved_model.p" *
#"./Res23_Multi_TP_1_5w_8w/l0.00105_saved_model.p" * 
#"./Res23_Multi_TP_1_5w_8w/l0.00106_saved_model.p" *
#"./Res23_Multi_TP_1_5w_8w/l0.00107_saved_model.p" *
#"./Res23_Multi_TP_1_5w_8w/l0.00108_saved_model.p" *
#"./Res23_Multi_TP_1_5w_8w/l0.00109_saved_model.p" *

#Reg neutral gender
#"./Res24_NoGen/l1e-07_saved_model.p" *
#"./Res24_NoGen/l2e-07_saved_model.p" *
#"./Res24_NoGen/l3e-07_saved_model.p" *
#"./Res24_NoGen/l4e-07_saved_model.p" 
#"./Res24_NoGen/l5e-07_saved_model.p" 


SavedModelFile="saved_model.p"

saved_models = pickle.load( open(SavedModelFile, "rb" ) )



model_cpu=saved_models["model"]
model=model_cpu.to(device)



#+ + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + +
#3) test
#+ + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + +

n_test=y_test.shape[0]  #  -> 138862 in total
Curr_obsIDs=np.arange(n_test)
np.random.shuffle(Curr_obsIDs)
#Curr_obsIDs=Curr_obsIDs[:10000]





var_X_select = X_test[Curr_obsIDs,:]
Masks_select = Masks_test[Curr_obsIDs,:]
var_y_select = y_test[Curr_obsIDs,:].float()  #added in the 1d case
S_select = S_test[Curr_obsIDs]

import sys
sys.path.append('/home/laurent/Projects/2022_W2reg_package/')
from W2reg_misc import *



output=LargeDatasetPred_nlp(model,var_X_select[:,:],Masks_select[:,:],64,DEVICE=device)

#with torch.no_grad():
#  output = model(ids=var_X_select[:,:150], mask=Masks_select[:,:150])


y_pred=output.to('cpu')
y_true=var_y_select

y_pred_labels=torch.argmax(y_pred,dim=1)
y_true_labels=torch.argmax(y_true,dim=1)
print(sum(y_pred_labels>6))
print(sum(y_true_labels>6))
#Weird error here: decomment later
#print('mean error:',torch.mean(torch.abs(y_pred-y_true)))
#print('mean pred accuracy:',torch.mean(1.*(y_pred_labels==y_true_labels)))




#confusion matrix

import sklearn

#lst_jobs=list(jobid_2_job.values())
lst_jobs=list_jobs_used

y_pred_labels=torch.argmax(y_pred,dim=1)
y_true_labels=torch.argmax(y_true,dim=1)

confusionMatrix=sklearn.metrics.confusion_matrix(y_true_labels,y_pred_labels,normalize='true')

ax=plt.axes()
plt.imshow(confusionMatrix,cmap='nipy_spectral',vmin=0.,vmax=1.0)
plt.xticks(rotation=90)
ax.set_xticks(list(np.arange(len(lst_jobs))))
ax.set_xticklabels(lst_jobs)
ax.set_yticks(list(np.arange(len(lst_jobs))))
ax.set_yticklabels(lst_jobs)
plt.colorbar()
plt.title('confusion matrix')
plt.savefig('ConfMat.pdf')
#plt.show()
plt.clf()

np.savetxt("ConfMat.csv", confusionMatrix, delimiter=",",fmt='%5.4f')

#confusion matrix - gender gap


y_pred_labels=torch.argmax(y_pred,dim=1)
y_true_labels=torch.argmax(y_true,dim=1)

Lst0=np.where(S_select<0.5)[0]
Lst1=np.where(S_select>0.5)[0]

confusionMatrix0=sklearn.metrics.confusion_matrix(y_true_labels[Lst0],y_pred_labels[Lst0],normalize='true')
confusionMatrix1=sklearn.metrics.confusion_matrix(y_true_labels[Lst1],y_pred_labels[Lst1],normalize='true')



ax=plt.axes()
plt.imshow(confusionMatrix0,cmap='nipy_spectral',vmin=0.,vmax=1.0)
plt.xticks(rotation=90)
ax.set_xticks(list(np.arange(len(lst_jobs))))
ax.set_xticklabels(lst_jobs)
ax.set_yticks(list(np.arange(len(lst_jobs))))
ax.set_yticklabels(lst_jobs)
plt.colorbar()
plt.title('confusion matrix in group 0')
plt.savefig('ConfMat0.pdf')
#plt.show()
plt.clf()

np.savetxt("ConfMat0.csv", confusionMatrix0, delimiter=",",fmt='%5.4f')


ax=plt.axes()
plt.imshow(confusionMatrix1,cmap='nipy_spectral',vmin=0.,vmax=1.0)
plt.xticks(rotation=90)
ax.set_xticks(list(np.arange(len(lst_jobs))))
ax.set_xticklabels(lst_jobs)
ax.set_yticks(list(np.arange(len(lst_jobs))))
ax.set_yticklabels(lst_jobs)
plt.colorbar()
plt.title('confusion matrix in group 1')
plt.savefig('ConfMat1.pdf')
#plt.show()
plt.clf()

np.savetxt("ConfMat1.csv", confusionMatrix1, delimiter=",",fmt='%5.4f')


ax=plt.axes()
plt.imshow(np.abs(confusionMatrix1-confusionMatrix0),cmap='nipy_spectral',vmin=0.,vmax=0.5)  #absolute value of the difference
plt.xticks(rotation=90)
ax.set_xticks(list(np.arange(len(lst_jobs))))
ax.set_xticklabels(lst_jobs)

ax.set_yticks(list(np.arange(len(lst_jobs))))
ax.set_yticklabels(lst_jobs)
plt.title('absolute difference between the confusion matrices of groups 0 and 1')
plt.colorbar()
plt.savefig('ConfMatDiff01.pdf')
#plt.show()
plt.clf()


np.savetxt("ConfMatDiff01.csv", np.abs(confusionMatrix1-confusionMatrix0), delimiter=",",fmt='%5.4f')


for i in range(len(lst_jobs)):
  print(lst_jobs[i],':',np.round(confusionMatrix0[i,i],3),' ',np.round(confusionMatrix1[i,i],3),' ',np.round(np.abs(confusionMatrix1[i,i]-confusionMatrix0[i,i]),3))



#amount of M/F in each class


for jobID in range(len(lst_jobs)):
  MalesInClasses=len(torch.where(y_true_labels[Lst1]==jobID)[0])
  FemalesInClasses=len(torch.where(y_true_labels[Lst0]==jobID)[0])
  print(jobid_2_job[jobID]+': '+str(MalesInClasses+FemalesInClasses)+' '+str(np.round(100.*FemalesInClasses/(MalesInClasses+FemalesInClasses),1)))
  









#other stuffs
  

modelsID_male=torch.where((y_true_labels[Lst1]==9))[0]
modelsID_female=torch.where((y_true_labels[Lst0]==9))[0]


preds_model1=y_pred_labels[Lst1][modelsID_male]
preds_model0=y_pred_labels[Lst0][modelsID_female]

print(torch.mean(1.*(preds_model1==9)))
print(torch.mean(1.*(preds_model0==9)))
