
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torchvision.models as models
from collections import OrderedDict
from torch.optim.lr_scheduler import ReduceLROnPlateau
import time
import pickle
import pandas
import numpy as np # to handle matrix and data operation
import matplotlib.pyplot as plt   #image visualisation
import scipy.stats as st

def fit_NLP_model(model,X_train,Masks_train,y_train, f_loss_attach=nn.MSELoss() , EPOCHS = 5, BATCH_SIZE = 32,DEVICE='cpu',optim_lr=0.0001):
    """
    -> X_train: input pytorch tensor are supposed to have a shape structured as [NbObs,:], [NbObs,:,:], or [NbObs,:,:,:].
    -> Masks_train: same size as X_train; masks the values of X_train to indicate whether a word is present or not in the text.
    -> y_train: true output pytorch tensor. Supposed to be 2D (for one-hot encoding or others), or 1D (eg for binary classification)
    -> f_loss_attach: Data attachment term in the loss. Can be eg  nn.MSELoss(), nn.BCELoss(), ... . Must be coherent with the model outputs and the true outputs.
    """

    outputdatadim=len(y_train.shape)-1  #dimension of the output data (-1 takes into account the fact that the first dimension corresponds to the observations)

    optimizer = torch.optim.Adam(model.parameters(), lr=optim_lr) #,lr=0.001, betas=(0.9,0.999))
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.9, patience=250, verbose=True)

    model.train()

    n=X_train.shape[0]

    Lists_Results={}
    Lists_Results['Loss']=[]          #loss of the data attachement term

    epoch=0
    while epoch<EPOCHS:
        obsIDs=np.arange(n)
        np.random.shuffle(obsIDs)

        batch_start=0
        batchNb=0

        while batch_start+BATCH_SIZE < n:

            #1) mini-batch predictions

            #1.1) get the observation IDs
            Curr_obsIDs=obsIDs[batch_start:batch_start+BATCH_SIZE]

            X_batch = X_train[Curr_obsIDs,:].long().to(DEVICE)
            Masks_batch = Masks_train[Curr_obsIDs,:].to(DEVICE)

            if outputdatadim==0:
              y_batch = y_train[Curr_obsIDs].view(-1,1).float()  #as the outputs are in a 1d vector
            elif outputdatadim==1:
              y_batch = y_train[Curr_obsIDs,:].float()

            #2.3) set the NN gradient to zero
            optimizer.zero_grad()

            #2.4) mini-batch prediction
            output = model(ids=X_batch, mask=Masks_batch)

            #3) compute the attachement term loss
            loss=f_loss_attach(output, y_batch.to(DEVICE))
            # propose other losses more  focused on the classification task
            #loss2 = nn.CrossEntropyLoss()(output, y_batch.to(DEVICE).long().view(-1))
            #loss3 = nn.BCELoss()(output, y_batch.to(DEVICE).float().view(-1,1))
            #loss4 = nn.BCEWithLogitsLoss()(output, y_batch.to(DEVICE).float().view(-1,1))

            loss.backward()
            optimizer.step()

            #6) update the first observation of the batch
            batch_start+=BATCH_SIZE
            batchNb+=1

            #7) save pertinent information to check the convergence
            locLoss=loss.item()
            Lists_Results['Loss'].append(locLoss)

            if batchNb%10==0:
              print("epoch "+str(epoch)+" -- batchNb "+str(batchNb)+" / "+str(n/BATCH_SIZE)+": Loss="+str(Lists_Results['Loss'][-1]))
              current_lr = optimizer.param_groups[0]['lr']
              print(f"Epoch {epoch}, Batch {batchNb}: Learning Rate = {current_lr}")

            scheduler.step(Lists_Results['Loss'][-1]) # Update learning rate
            

        #update the epoch number
        epoch+=1

    model_cpu=model.to('cpu')
    saved_models = { "model": model_cpu }
    pickle.dump( saved_models, open( 'saved_model.p', "wb" ) )
    # -> saved_models = pickle.load( open( "saved_model.p", "rb" ) )
    # -> model_cpu=saved_models["model"]
    # -> model=model_cpu.to(DEVICE)

    return Lists_Results
