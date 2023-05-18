'''
Upload/Download helper functions

Last modified: 05/2023
Author: Daniela Wiepert
Email: wiepert.daniela@mayo.edu
File: load_utils.py
'''

#IMPORTS
import os
import pickle

import pandas as pd

from google.cloud import storage

#GCS helper functions
def download_dir(gcs_dir, bucket):
    '''
    Download a directory from google cloud storage bucket.
    Inputs:
    :param gcs_dir: directory path in the bucket (no gs://project-name in the path)
    :param bucket: initialized GCS bucket object
    Outputs:
    :return folder: a string path to the local directory with the downloaded files
    '''
    print('Downloading directory')
    folder = os.path.basename(gcs_dir)
    if not os.path.exists(folder):
        os.makedirs(folder)

    blobs = bucket.list_blobs(prefix=gcs_dir)
    for blob in blobs:
        destination_uri = '{}/{}'.format(folder, os.path.basename(blob.name))
        if not os.path.exists(destination_uri):
            blob.download_to_filename(destination_uri)
    return folder

def download_model(gcs_path,outpath, bucket):
    '''
    Download a model from google cloud storage and the args.pkl file located in the same folder(if it exists)

    Inputs:
    :param gcs_path: full file path in the bucket to a pytorch model(no gs://project-name in the path)
    :param outpath: string path to directory where you want the model to be stored
    :param bucket: initialized GCS bucket object
    Outputs:
    :return mdl_path: a string path to the local version of the finetuned model (args.pkl will be in the same folder as this model)
    '''
    if not os.path.exists(outpath):
        os.makedirs(outpath)

    dir_path = os.path.dirname(gcs_path)
    bn = os.path.basename(gcs_path)
    blobs = bucket.list_blobs(prefix=dir_path)
    mdl_path = ''
    for blob in blobs:
        blob_bn = os.path.basename(blob.name)
        if blob_bn == bn:
            destination_uri = '{}/{}'.format(outpath, blob_bn) #download model 
            mdl_path = destination_uri
        elif blob_bn == 'args.pkl':
            destination_uri = '{}/model_args.pkl'.format(outpath) #download args.pkl as model_args.pkl
        else:
            continue #skip any other files
        if not os.path.exists(destination_uri):
            blob.download_to_filename(destination_uri)
   
    return mdl_path

def upload(gcs_prefix, path, bucket):
    '''
    Upload a file to a google cloud storage bucket
    Inputs:
    :param gcs_prefix: path in the bucket to save file to (no gs://project-name in the path)
    :param path: local string path of the file to upload
    :param bucket: initialized GCS bucket object
    '''
    assert bucket is not None, 'no bucket given for uploading'
    if gcs_prefix is None:
        gcs_prefix = os.path.dirname(path)
    blob = bucket.blob(os.path.join(gcs_prefix, os.path.basename(path)))
    blob.upload_from_filename(path)

#Load functions
def load_args(args):
    '''
    Load in an .pkl file of args
    :param args: dict with all the argument values
    :return model_args: dict with all the argument values from the finetuned model
    '''
    # assumes that the model is saved in the same folder as an args.pkl file 
    folder = os.path.dirname(args.finetuned_mdl_path)

    if os.path.exists(os.path.join(folder, 'model_args.pkl')): #if downloaded from gcs into the exp dir, it should be saved under mdl_args.pkl to make sure it doesn't overwrite the args.pkl
        with open(os.path.join(folder, 'model_args.pkl'), 'rb') as f:
            model_args = pickle.load(f)
    elif os.path.exists(os.path.join(folder, 'args.pkl')): #if not downloaded and instead stored in a local place, it will be saved as args.pkl
        with open(os.path.join(folder, 'args.pkl'), 'rb') as f:
            model_args = pickle.load(f)
    else: #if there are no saved args
        print('No args.pkl or model_args.pkl stored with the finetuned model. Using the current args for initializing the finetuned model instead.')
        model_args = args
    
    return model_args

def setup_mdl_args(args):
    '''
    Get model args used during finetuning of the specified model
    :param args: dict with all the argument values
    :return model_args: dict with all the argument values from the finetuned model
    :return finetuned_mdl_path: updated finetuned_mdl_path (in case it needed to be downloaded from gcs)
    '''
    #if running a pretrained model only, use the args from this run
    if args.finetuned_mdl_path is None:
        model_args = args
    else:
    #if running a finetuned model
        #(1): check if saved on cloud and load the model and args.pkl
        if args.finetuned_mdl_path[:5] =='gs://':
                mdl_path = args.finetuned_mdl_path[5:].replace(args.bucket_name,'')[1:]
                args.finetuned_mdl_path = download_model(mdl_path, args.exp_dir, args.bucket)
        
        #(2): load the args used for finetuning
        model_args = load_args(args)

        #(3): check if the checkpoint for the finetuned model is downloaded
        if model_args.checkpoint[:5] =='gs://': #if checkpoint on cloud
            checkpoint = model_args.checkpoint[5:].replace(model_args.bucket_name,'')[1:]
            if model_args.bucket_name != args.bucket_name: #if the bucket is not the same as the current bucket, initialize the bucket for downloading
                if args.bucket_name is not None:
                    storage_client = storage.Client(project=model_args.project_name)
                    model_args.bucket = storage_client.bucket(model_args.bucket_name)
                else:
                    model_args.bucket = None

                checkpoint = download_dir(checkpoint, model_args.bucket) #download with the new bucket
            else:
                checkpoint = download_dir(checkpoint, args.bucket) #download with the current bucket
            model_args.checkpoint = checkpoint #reset the checkpoint path
        else: #load in from local machine, just need to check that the path exists
            assert os.path.exists(model_args.checkpoint), f'Current checkpoint does not exist on local machine: {model_args.checkpoint}'

    return model_args, args.finetuned_mdl_path

def load_data(data_split_root, exp_dir, cloud, cloud_dir, bucket):
    """
    Load the train and test data from a directory. Assumes the train and test data will exist in this directory under train.csv and test.csv
    :param data_split_root: specify str path where datasplit csvs are located
    :param exp_dir: specify LOCAL output directory as str
    :param cloud: boolean to specify whether to save everything to google cloud storage
    :param cloud_dir: if saving to the cloud, you can specify a specific place to save to in the CLOUD bucket
    :param bucket: google cloud storage bucket object
    :return train_df, val_df, test_df: loaded dataframes with annotations
    """
    train_path = f'{data_split_root}/train.csv'
    test_path = f'{data_split_root}/test.csv'
    #get data
    train_df = pd.read_csv(train_path, index_col = 'uid')
    test_df = pd.read_csv(test_path, index_col = 'uid')

    #randomly sample to get validation set 
    val_df = train_df.sample(50)
    train_df = train_df.drop(val_df.index)

    #save validation set
    val_path = os.path.join(exp_dir, 'validation.csv')
    val_df.to_csv(val_path, index=True)

    if cloud:
        upload(cloud_dir, val_path, bucket)

    #alter data columns
    train_df["distortions"]=((train_df["distorted Cs"]+train_df["distorted V"])>0).astype(int)
    val_df["distortions"]=((val_df["distorted Cs"]+val_df["distorted V"])>0).astype(int)
    test_df["distortions"]=((test_df["distorted Cs"]+test_df["distorted V"])>0).astype(int)

    return train_df, val_df, test_df