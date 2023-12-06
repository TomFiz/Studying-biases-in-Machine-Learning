import pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

#+ + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + +
#PART 1 : same as usual on the bios dataset
#+ + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + +

#1.1) load data
data = pickle.load(open('BIOS.pkl','rb'))


print(len(data))

#1.2) show data

X_raw = ['' for i in range(len(data))]
Titles = ['' for i in range(len(data))]
Gen = ['' for i in range(len(data))]

for i in range(len(data)):
    X_raw[i] = data[i]["raw"][data[i]["start_pos"]:] 
    Titles[i] = data[i]["title"]
    Gen[i] = data[i]["gender"]


#obs_of_interest=10

#print(X_raw[obs_of_interest])
#print(Titles[obs_of_interest])
#print(Gen[obs_of_interest])

#+ + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + +
#PART 2 : adapt wat was made in
#https://www.kaggle.com/code/goutham794/distill-bert-fine-tuning-huggingface-and-pytorch/notebook
#+ + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + +

#2.1) init


import torch
import torch.nn as nn
import transformers
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertModel, DistilBertTokenizer
import re
from torch import cuda
#device = 'cuda' if cuda.is_available() else 'cpu'

device='cpu'

print('Device is',device)

#2.2) pre-processing...

#... my stuffs

def remove_URL(text):
    url = re.compile(r'https?://\S+|www\.\S+')
    return url.sub(r'',text)

def remove_numbers(text):
    text = ''.join([i for i in text if not i.isdigit()])
    return text

def remove_html(text):
    html=re.compile(r'<.*?>')
    return html.sub(r'',text)

def remove_username(text):
    url = re.compile(r'@[A-Za-z0-9_]+')
    return url.sub(r'',text)



def pre_process_text(text):
    text = remove_URL(text)
    text = remove_numbers(text)
    text = remove_html(text)
    text = remove_username(text)
    
    text=text.replace('she ','he ')
    text=text.replace('She ','He ')
    text=text.replace('her ','his ')
    text=text.replace('Her ','His ')
    text=text.replace('hers ','him ')
    text=text.replace('hers,','him,')
    text=text.replace('hers.','him.')
    text=text.replace('herself ','himself ')
    text=text.replace('herself,','himself,')
    text=text.replace('herself.','himself.')
    text=text.replace('ms ','mr ')
    text=text.replace('Ms ','Mr ')
    text=text.replace('ms.','mr.')
    
    return " ".join(text.split())

#txt_example=X_raw[1]
#
#print(txt_example)
#print('---')
#print(pre_process_text(txt_example))

#... Fanny stuffs

import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize

list_ponctuation = [".", "!", "?", "...", ".."]

#couper le texte après . ! ? ... ..
#ne pas couper lorsque c'est un mail ou un url: pas de soucis puisque tout les mails et les urls de notre dataset ont été remplacés par les mots "mail" et "url"
#couper le texte avant d'avoir n mots mais en coupant à la fin d'une phrase. 

def untokenize(s_tok, list_car = ["’", ",", ".", "...", "..", "....", ":", "!", "?", "'s", "s"]):
  len_s = len(s_tok)
  sentence = s_tok[0]
  for i in range(1,len_s):
    if s_tok[i] in list_car:
      sentence = sentence + s_tok[i]
    else :
      sentence = sentence + ' ' + s_tok[i]
  return(sentence)

def truncation_sentence(sentence, n, list_ponctuation = [".", "..", "...", "!", "!!", "!!!", "?", "??", "???"]):
  s_tok = word_tokenize(sentence)
  m = n

  if (len(s_tok) - 1) < m:
    return(sentence)

  else:
    
    while m > 1 and not(s_tok[m] in list_ponctuation):
      m = m - 1
  
    if m == 1:
      return("N/A")
  
    else:
      return(untokenize(s_tok[:(m+1)]))  

def truncation_X(X_raw, n):
  X_raw_n = []
  for i in range(len(X_raw)):
    X_raw_n = X_raw_n + [truncation_sentence(X_raw[i], n)]
    if i%5000 == 0:
      print(i)
  return(X_raw_n)

#... pre-processing





X_raw_sn=[]
Titles_sn=[]
Gen_sn=[]
list_jobs_used=['surgeon']
for i in range(len(X_raw)):
    if Titles[i] in list_jobs_used:
      truncated_sentence=truncation_sentence(pre_process_text(X_raw[i]),255)  #there will have a maximum of 255 words to guess the job
      X_raw_sn.append(truncated_sentence)
      Titles_sn.append(Titles[i])
      Gen_sn.append(Gen[i])
      
    
print(len(X_raw),' -> ',len(X_raw_sn))
     


#2.3) create the tokenizer and the functions to get the input and output observations at the torch format

tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased", do_lower_case=True)

def get_input(input_id,list_input_texts,tokenizer,Max_len):
  inputs = tokenizer.encode_plus(
            list_input_texts[input_id],
            None,
            add_special_tokens=True,
            max_length=Max_len,
            #padding=True,
            return_token_type_ids=True
        )
  ids = torch.tensor(inputs['input_ids'], dtype=torch.long)
  mask = torch.tensor(inputs['attention_mask'], dtype=torch.long)

  return [ids,mask]



def CreateConversions_jobs_jobids(Titles):

  job_2_jobid={}
  jobid_2_job={}

  #Titles_set=list(set(Titles))
  Titles_set=list_jobs_used
  for i in range(len(Titles_set)):
    job_2_jobid[Titles_set[i]]=i
    jobid_2_job[i]=Titles_set[i]

  return [job_2_jobid,jobid_2_job]


[job_2_jobid,jobid_2_job]=CreateConversions_jobs_jobids(Titles)

def get_output_id(input_id,lst_outputs,job_2_jobid):
    return job_2_jobid[lst_outputs[input_id]]


#... test it
#print(get_input(0,X_raw,tokenizer,Max_len))
#print(get_output(0,Titles))


#2.3.bis) transform the tokenized data as classic training X and y

#... X
list_all_ids=[]
list_all_masks=[]
n=len(X_raw_sn)
largest_tokens_nb=0

for i in range(n):
  print(i,'/',n,'    -    ',largest_tokens_nb)

  [ids,mask]=get_input(i,X_raw_sn,tokenizer,255)
  list_all_ids.append(ids)
  list_all_masks.append(mask)

  if largest_tokens_nb<len(ids):
    largest_tokens_nb=len(ids)

p=largest_tokens_nb


X=torch.zeros([n,p],dtype=torch.long)
Masks=torch.zeros([n,p],dtype=torch.long)

for i in range(n):
  print(i)
  p_loc=len(list_all_ids[i])
  X[i,:p_loc]=list_all_ids[i]*1
  Masks[i,:p_loc]=list_all_masks[i]*1

#... y

y=torch.zeros([n,len(set(Titles_sn))],dtype=torch.float32)

for i in range(n):
    #print(i)
    y[i,get_output_id(i,Titles_sn,job_2_jobid)]=1

#... gender

g=torch.zeros([n],dtype=torch.uint8)

for i in range(n):
  if Gen_sn[i]=='M':
    #print(i)
    g[i]=1


data2save ={"X":X,"Masks":Masks,"y":y,'g':g,'bio':X_raw_sn,"job_2_jobid":job_2_jobid,"jobid_2_job":jobid_2_job}
pickle.dump(data2save, open( "TreatedData_all4.pk", "wb" ) )




#make sure that the amount of test data in each group is well balanced

for i in range(len(list_jobs_used)):
  NbFemales=( 1 * ((y[:,i]==1)*(g[:]==0)) ).sum().item()
  NbMales=( 1 * ((y[:,i]==1)*(g[:]==1)) ).sum().item()
  if NbFemales>1000 and NbMales>1000:
    print(jobid_2_job[i] , ' ' , NbFemales , NbMales,'+++')
    newTestID_f=np.where((y[:,i]==1)*(g[:]==0))[0][:500]
    newTestID_m=np.where((y[:,i]==1)*(g[:]==1))[0][:500]
    newTrainID_f=np.where((y[:,i]==1)*(g[:]==0))[0][500:]
    newTrainID_m=np.where((y[:,i]==1)*(g[:]==1))[0][500:]
  else:
    print(jobid_2_job[i] , ' ' , NbFemales , NbMales,'---')
    newTestID_f=np.where((y[:,i]==1)*(g[:]==0))[0][:100]
    newTestID_m=np.where((y[:,i]==1)*(g[:]==1))[0][:100]
    newTrainID_f=np.where((y[:,i]==1)*(g[:]==0))[0][100:]
    newTrainID_m=np.where((y[:,i]==1)*(g[:]==1))[0][100:]
  
  if i==0:
    TestID=np.concatenate([newTestID_f,newTestID_m])
    TrainID=np.concatenate([newTrainID_f,newTrainID_m])
  else:
    TestID=np.concatenate([TestID,newTestID_f])
    TrainID=np.concatenate([TrainID,newTrainID_f])
    TestID=np.concatenate([TestID,newTestID_m])
    TrainID=np.concatenate([TrainID,newTrainID_m])

np.random.shuffle(TestID)
np.random.shuffle(TrainID)


X_test=X[TestID,:]
Masks_test=Masks[TestID,:]
y_test=y[TestID,:]
g_test=g[TestID]
bio_test=[]
for i in range(len(TestID)):
  bio_test.append(X_raw_sn[TestID[i]])
  
X_train=X[TrainID,:]
Masks_train=Masks[TrainID,:]
y_train=y[TrainID,:]
g_train=g[TrainID]
bio_train=[]
for i in range(len(TrainID)):
  bio_train.append(X_raw_sn[TrainID[i]])


data2save ={"X_test":X_test,"Masks_test":Masks_test,"y_test":y_test,'g_test':g_test,'bio_test':bio_test , "X_train":X_train,"Masks_train":Masks_train,"y_train":y_train,'g_train':g_train,'bio_train':bio_train , "job_2_jobid":job_2_jobid,"jobid_2_job":jobid_2_job}
pickle.dump(data2save, open( "TreatedData_all5_neutral.pk", "wb" ) )

    


"""
infile = open('./TreatedData_all5.pk','rb')
SavedData = pickle.load(infile)

X_test=SavedData["X_test"]
Masks_test=SavedData["Masks_test"]
y_test=SavedData["y_test"]
g_test=SavedData["g_test"]
bio_test=SavedData["bio_test"]

X_train=SavedData["X_train"]
Masks_train=SavedData["Masks_train"]
y_train=SavedData["y_train"]
g_train=SavedData["g_train"]
bio_train=SavedData["bio_train"]

infile.close()
"""

"""
photographer   8443 15197 +++
comedian   583 2171 ---
personal_trainer   631 750 ---
model   6135 1259 +++
pastor   587 1880 ---
dj   203 1217 ---
painter   3446 4093 +++
professor   52152 63397 +++
surgeon   2002 11471 +++
nurse   16992 1721 +++
architect   2358 7511 +++
journalist   9584 9761 +++
accountant   2006 3440 +++
physician   20198 20621 +++
dietitian   3615 281 ---
dentist   5218 9502 +++
filmmaker   2265 4606 +++
interior_designer   1150 271 ---
yoga_teacher   1352 254 ---
chiropractor   695 1942 ---
psychologist   11232 6874 +++
poet   3386 3521 +++
attorney   12129 19609 +++
rapper   134 1243 ---
paralegal   1465 265 ---
teacher   9532 6273 +++
composer   903 4586 ---
software_engineer   1083 5667 +++
"""
