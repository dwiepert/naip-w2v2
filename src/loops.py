"""
Model loops for finetuning and extracting embeddings from W2V2 models

Last modified: 05/2023
Author: Daniela Wiepert
Email: wiepert.daniela@mayo.edu
File: loops.py
"""
#IMPORTS
#built-in
import json
import os

#third party
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, roc_curve

#local
from utilities import *

def finetune(model, dataloader_train, dataloader_val = None, 
             optim='adamw', learning_rate=0.001, weight_decay=0.0001,
             loss_fn='BCE',sched='onecycle', max_lr=0.01,
             epochs=10, exp_dir='', cloud=False, cloud_dir='', bucket=None):
    """
    Training loop for finetuning W2V2
    :param model: W2V2 model
    :param dataloader_train: dataloader object with training data
    :param dataloader_val: dataloader object with validation data
    :param optim: type of optimizer to initialize
    :param weight_decay: weight decay value for adamw optimizer
    :param learning_rate: optimizer learning rate
    :param loss_fn: type of loss function to initialize
    :param sched: type of scheduler to initialize
    :param max_lr: max learning rate for onecycle scheduler
    :param epochs: number of epochs to run pretraining
    :param exp_dir: output directory on local machine
    :param cloud: boolean indicating whether uploading to cloud
    :param cloud_dir: output directory in google cloud storage bucket
    :param bucket: initialized GCS bucket object
    :return model: finetuned W2V2 model
    """
    print('Training start')
    #send to gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    #loss
    if loss_fn == 'MSE':
        criterion = torch.nn.MSELoss()
    elif loss_fn == 'BCE':
        criterion = torch.nn.BCEWithLogitsLoss()
    else:
        raise ValueError('MSE must be given for loss parameter')
    #optimizer
    if optim == 'adam':
        optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad],lr=learning_rate)
    elif optim == 'adamw':
         optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=learning_rate, weight_decay=weight_decay)
    else:
        raise ValueError('adam must be given for optimizer parameter')
    
    if sched == 'onecycle':
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=max_lr, steps_per_epoch=len(dataloader_train), epochs=epochs)
    else:
        scheduler = None
    
    #train
    for e in range(epochs):
        training_loss = list()
        #t0 = time.time()
        model.train()
        for batch in tqdm(dataloader_train):
            x = torch.squeeze(batch['waveform'], dim=1)
            targets = batch['targets']
            x, targets = x.to(device), targets.to(device)
            optimizer.zero_grad()
            o = model(x)
            loss = criterion(o, targets)
            loss.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            loss_item = loss.item()
            training_loss.append(loss_item)

        if e % 10 == 0 or e == epochs-1:
            #SET UP LOGS
            if scheduler is not None:
                lr = scheduler.get_last_lr()
            else:
                lr = learning_rate
            logs = {'epoch': e, 'optim':optim, 'loss_fn': loss_fn, 'lr': lr, 'scheduler':sched}
    
            logs['training_loss_list'] = training_loss
            training_loss = np.array(training_loss)
            logs['running_loss'] = np.sum(training_loss)
            logs['training_loss'] = np.mean(training_loss)

            print('RUNNING LOSS', e, np.sum(training_loss) )
            print(f'Training loss: {np.mean(training_loss)}')

            if dataloader_val is not None:
                print("Validation start")
                validation_loss = validation(model, criterion, dataloader_val)

                logs['val_loss_list'] = validation_loss
                validation_loss = np.array(validation_loss)
                logs['val_running_loss'] = np.sum(validation_loss)
                logs['val_loss'] = np.mean(validation_loss)
                
                print('RUNNING VALIDATION LOSS',e, np.sum(validation_loss) )
                print(f'Validation loss: {np.mean(validation_loss)}')
            
            #SAVE LOGS
            print(f'Saving epoch {e}')
            json_string = json.dumps(logs)
            logs_path = os.path.join(exp_dir, 'logs_ft_epoch{}.json'.format(e))
            with open(logs_path, 'w') as outfile:
                json.dump(json_string, outfile)
            
            #SAVE CURRENT MODEL
            
            mdl_path = os.path.join(exp_dir, 'w2v2_ft_mdl_epoch{}.pt'.format(e))
            torch.save(model.state_dict(), mdl_path)
            
            optim_path = os.path.join(exp_dir, 'w2v2_ft_optim_epoch{}.pt'.format(e))
            torch.save(optimizer.state_dict(), optim_path)

            if cloud:
                upload(cloud_dir, logs_path, bucket)
                #upload_from_memory(model.state_dict(), args.cloud_dir, mdl_path, args.bucket)
                upload(cloud_dir, mdl_path, bucket)
                upload(cloud_dir, optim_path, bucket)
    return model

def validation(model, criterion, dataloader_val):
    '''
    Validation loop for finetuning the w2v2 classification head. 
    :param model: W2V2 model
    :param criterion: loss function
    :param dataloader_val: dataloader object with validation data
    :return validation_loss: list with validation loss for each batch
    '''
    validation_loss = list()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    with torch.no_grad():
        model.eval()
        for batch in tqdm(dataloader_val):
            x = torch.squeeze(batch['waveform'], dim=1)
            targets = batch['targets']
            x, targets = x.to(device), targets.to(device)
            o = model(x)
            val_loss = criterion(o, targets)
            validation_loss.append(val_loss.item())

    return validation_loss

def evaluation(model, dataloader_eval):
    """
    Start model evaluation
    :param model: W2V2 model
    :param dataloader_eval: dataloader object with evaluation data
    :return preds: model predictions
    :return targets: model targets (actual values)
    """
    print('Evaluation start')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    outputs = []
    t = []
    model = model.to(device)
    with torch.no_grad():
        model.eval()
        for batch in tqdm(dataloader_eval):
            x = torch.squeeze(batch['waveform'], dim=1)
            x = x.to(device)
            targets = batch['targets']
            targets = targets.to(device)
            o = model(x)
            outputs.append(o)
            t.append(targets)

    outputs = torch.cat(outputs).cpu().detach()
    t = torch.cat(t).cpu().detach()
    # SAVE PREDICTIONS AND TARGETS 

    print('Evaluation finished')
    return outputs, t

def embedding_extraction(model, dataloader,embedding_type='ft',layer=-1, pooling_mode='mean'):
    """
    Run a specific subtype of evaluation for getting embeddings.
    :param model: W2V2 model
    :param dataloader_eval: dataloader object with data to get embeddings for
    :param embedding_type: string specifying whether embeddings should be extracted from classification head (ft) or base pretrained model (pt)
    :return embeddings: an np array containing the embeddings
    :param layer: hidden layer to take out and do results for - must be between 0-12
    """
    print('Getting embeddings')
    embeddings = np.array([])

    # send to gpu
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    with torch.no_grad():
        model.eval()
        for batch in tqdm(dataloader):
            x = torch.squeeze(batch['waveform'], dim=1)
            x = x.to(device)
            e = model.extract_embedding(x, embedding_type,layer=layer, pooling_mode=pooling_mode)
            e = e.cpu().numpy()
            if embeddings.size == 0:
                embeddings = e
            else:
                embeddings = np.append(embeddings, e, axis=0)
        
    return embeddings

def calc_auc(preds, targets, target_labels,
         exp_dir, cloud, cloud_dir, bucket):
    """
    Get AUC scores, doesn't return, just saves the metrics to a csv
    :param args: dict with all the argument values
    :param preds: model predictions
    :param targets: model targets (actual values)
    """
    #get AUC score and all data for ROC curve
    preds = preds[targets.isnan().sum(1)==0]
    targets[targets.isnan().sum(1)==0]
    pred_mat=torch.sigmoid(preds).numpy()
    target_mat=targets.numpy()
    aucs=roc_auc_score(target_mat, pred_mat, average = None) #TODO: this doesn't work when there is an array with all labels as 0???
    print(aucs)
    data = pd.DataFrame({'Label':target_labels, 'AUC':aucs})
    data.to_csv(os.path.join(exp_dir, 'aucs.csv'), index=False)
    if cloud:
        upload(cloud_dir, os.path.join(exp_dir, 'aucs.csv'), bucket)

    return data
