'''
W2V2 score predictions for hyperparameter tuning

Last modified: 07/2023
Author: Daniela Wiepert
Email: wiepert.daniela@mayo.edu
File: score.py
'''

#IMPORTS
#built-in
import argparse
import itertools
import os
import glob
import re

#third-party
import torch
import pandas as pd
import numpy as np

from google.cloud import storage

from sklearn.metrics import roc_auc_score, roc_curve
#local
#from utilities import *
#from loops import *


def calc_auc(preds, targets):
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
    return aucs

def get_predictions(dirs):
    aucs = {}
    no_data = []
    for d in dirs:
        try:
            pred = torch.load(glob.glob(os.path.join(d,'*predictions.pt'))[0])
            target = torch.load(glob.glob(os.path.join(d,'*targets.pt'))[0])
            score = calc_auc(pred, target)
            aucs[d] = score
        except:
            no_data.append(d)
    return aucs

def save(data, outname, target_labels):
    
    aucs = get_predictions(data)
    aucs = pd.DataFrame.from_dict(aucs)
    aucs['labels'] = target_labels
    aucs = aucs.set_index('labels')
    aucs.to_csv(outname)

def download_predictions(prefix, save_dir, bucket):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    blobs = bucket.list_blobs(prefix=prefix)
    i = 0
    for blob in blobs:
        destination_uri = '{}/{}/{}'.format(save_dir, os.path.basename(os.path.dirname(blob.name)),os.path.basename(blob.name))
        
        if 'predictions' in destination_uri or 'targets' in destination_uri:
            if not os.path.exists(os.path.dirname(destination_uri)):
                os.makedirs(os.path.dirname(destination_uri))
            if not os.path.exists(destination_uri):
                blob.download_to_filename(destination_uri)

            i+= 1
            if i%50==0:
                print(f'{i} blobs downloaded')

def split_name(dir_list):
    params = {}
    for s in dir_list:
        dir = os.path.basename(s)
        if dir != '' and dir != 'configs':
            temp = {}
            p = dir.split("_")
            temp['weight_decay'] = re.sub("[A-Za-z]","",p[6])
            temp['epoch'] = re.sub("[A-Za-z]","",p[7])
            temp['clf'] = re.sub("[A-Za-z]","",p[9])
            temp['lr'] = re.sub("[A-Za-z]","",p[10])
            temp['bs'] = re.sub("[A-Za-z]","",p[11])
            temp['layer'] = re.sub("[A-Za-z]","",p[12])

            if 'onecycle' in dir:
                temp['scheduler'] = True
                if 'ws' not in dir:
                    temp['weighted'] = False
                    if 'sd' in dir:
                        temp['shared_dense'] = True
                        temp['sd_bottleneck'] = re.sub("[A-Za-z]","",p[13])
                        temp['dropout'] = re.sub("[A-Za-z]","",p[16])
                        temp['clf_bottleneck'] = re.sub("[A-Za-z]","",p[17])
                    else:
                        temp['shared_dense'] = False
                        temp['sd_bottleneck'] = 'None'
                        temp['dropout'] = re.sub("[A-Za-z]","",p[15])
                        temp['clf_bottleneck'] = re.sub("[A-Za-z]","",p[16])
                else: 
                    temp['weighted'] = True
                    if 'sd' in dir:
                        temp['shared_dense'] = True
                        temp['sd_bottleneck'] = re.sub("[A-Za-z]","",p[13])
                        temp['dropout'] = re.sub("[A-Za-z]","",p[17])
                        temp['clf_bottleneck'] = re.sub("[A-Za-z]","",p[18])
                    else:
                        temp['shared_dense'] = False
                        temp['sd_bottleneck'] = 'None'
                        temp['dropout'] = re.sub("[A-Za-z]","",p[16])
                        temp['clf_bottleneck'] = re.sub("[A-Za-z]","",p[17])

            else:
                temp['scheduler'] = False
                if 'ws' not in dir:
                    temp['weighted'] = False
                    if 'sd' in dir:
                        temp['shared_dense'] = True
                        temp['sd_bottleneck'] = re.sub("[A-Za-z]","",p[13])
                        temp['dropout'] = re.sub("[A-Za-z]","",p[14])
                        temp['clf_bottleneck'] = re.sub("[A-Za-z]","",p[15])
                    else:
                        temp['shared_dense'] = False
                        temp['sd_bottleneck'] = 'None'
                        temp['dropout'] = re.sub("[A-Za-z]","",p[13])
                        temp['clf_bottleneck'] = re.sub("[A-Za-z]","",p[14])
                else:
                    temp['weighted'] = True
                    if 'sd' in dir:
                        temp['shared_dense'] = True
                        temp['sd_bottleneck'] = re.sub("[A-Za-z]","",p[13])
                        temp['dropout'] = re.sub("[A-Za-z]","",p[15])
                        temp['clf_bottleneck'] = re.sub("[A-Za-z]","",p[16])
                    else:
                        temp['shared_dense'] = False
                        temp['sd_bottleneck'] = 'None'
                        temp['dropout'] = re.sub("[A-Za-z]","",p[14])
                        temp['clf_bottleneck'] = re.sub("[A-Za-z]","",p[15])
            params[s] = temp
    return params

def params_to_df(params):
    data = []
    for i,key in enumerate(params.keys() ):
        try:            
            data.append((key
                        ,params[key]['weight_decay']
                        ,params[key]['epoch']
                        ,params[key]['clf']
                        ,params[key]['lr'],
                        params[key]['bs'],
                        params[key]['layer'],
                        params[key]['scheduler'],
                        params[key]['weighted'],
                        params[key]['shared_dense'],
                        params[key]['sd_bottleneck'],
                        params[key]['dropout'],
                        params[key]['clf_bottleneck']))
        # if no entry, skip
        except:
            pass 
    df=pd.DataFrame(data=data,columns=['dir','weight_decay','epoch','clf','lr','bs', 'layer','scheduler','weighted','shared_dense','sd_bottleneck','dropout','clf_bottleneck'])

    return df

def auc_comp_metrics(df, target_labels):
    avg = []
    max = []
    for t in target_labels:
        avg.append(np.mean(df[t].values))
        max.append(np.max(df[t].values))

    dist_avg = None
    dist_max = None
    combined = None
    for i in range(len(target_labels)):
        t1 = df.apply(lambda x: np.absolute(df[target_labels[i]] - avg[i]), axis=0)[target_labels[i]].to_list()
        if dist_avg is None:
            dist_avg = t1
        else:
            dist_avg = [sum(x) for x in zip(dist_avg, t1)]

        t2 = df.apply(lambda x: np.absolute(df[target_labels[i]] - max[i]), axis=0)[target_labels[i]].to_list()

        if dist_max is None:
            dist_max = t2
        else:
            dist_max = [sum(x) for x in zip(dist_max, t2)]

        t3 = df[target_labels[i]].to_list()
        if combined is None:
            combined = t3
        else:
            combined = [sum(x) for x in zip(combined, t3)]
        
    df['dist_avg'] = dist_avg
    df['dist_max'] = dist_max
    df['combined'] = combined

    return df
    

def main():
    parser = argparse.ArgumentParser()
    #Inputs
    parser.add_argument('-i','--input_dir',default='', help='Input directory or location in google cloud storage bucket containing files to load')
    parser.add_argument('-s','--local_save_dir',default='', help='Input directory or location in google cloud storage bucket containing files to load')
    parser.add_argument('-l','--label_txt', default='') #default=None #default='./labels.txt'
    #GCS
    parser.add_argument('-b','--bucket_name', default=None, help="google cloud storage bucket name")
    parser.add_argument('-p','--project_name', default=None, help='google cloud platform project name')
    
    args = parser.parse_args()


    # Set up GCS
    if args.bucket_name is not None:
        storage_client = storage.Client(project=args.project_name)
        bucket = storage_client.bucket(args.bucket_name)
    else:
        bucket = None

    # get target labels
    if args.label_txt[:5] =='gs://':
        label_txt = args.label_txt[5:].replace(args.bucket_name,'')[1:]
        bn = os.path.basename(label_txt)
        blob = bucket.blob(label_txt)
        blob.download_to_filename(bn)
        label_txt = bn
    else:
        label_txt = args.label_txt
        
    with open(label_txt) as f:
        target_labels = f.readlines()
    target_labels = [l.strip().split(sep=",") for l in target_labels]
    target_labels = list(itertools.chain.from_iterable(target_labels))
    target_labels = [s.replace(" ","_") for s in target_labels]


    if args.input_dir[:5] =='gs://':
        args.input_dir = args.input_dir[5:].replace(args.bucket_name,'')[1:]
        download_predictions(args.input_dir, args.local_save_dir, bucket)
        args.input_dir = args.local_save_dir
        

    dirs = [x for x, _, _ in os.walk(args.input_dir) if os.path.isdir(x)]
    params = split_name(dirs)
    df = params_to_df(params)

    scores = get_predictions(df['dir'].to_list()) 
    score_df = pd.DataFrame.from_dict(scores)
    score_df['labels'] = target_labels
    score_df = score_df.set_index('labels')
    score_df = score_df.T
    score_df['dir'] = score_df.index
    score_df = score_df.reset_index()
    df = pd.merge(df, score_df, on='dir')

    df = auc_comp_metrics(df,target_labels)

    df.to_csv(os.path.join(args.input_dir,'hp_results.csv'),index=False)
    
if __name__ == "__main__":
    main()



