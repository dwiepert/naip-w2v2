'''
Dataset split
Generate train/test splits from a speech lake

Last modified: 07/2023
Author: Daniela Wiepert
Email: wiepert.daniela@mayo.edu
File: data_splits.py
'''

#IMPORTS
#built-in
import argparse
import ast
import itertools
import os
import pickle
import glob
import json
from random import sample

#third-party
import pandas as pd

from google.cloud import storage

#local
from utilities import *

def load_metadata_gcs(prefix, bucket):
    blobs = bucket.list_blobs(prefix=prefix)
    metadata_pth = [blob.name for blob in blobs if 'metadata' in blob.name]

    all_data = None
    i=0
    for md in metadata_pth:
        if i % 100 == 0:
            print(i)
        md_blob = bucket.blob(md)
        metadata = json.loads(md_blob.download_as_string())
        feats = pd.DataFrame.from_dict(metadata['features'])
        feats = feats[['name','score']]

        if feats.duplicated(subset=['name']).any():
            feats = feats.sort_values(by=['score'],ascending=False)
            feats = feats.drop_duplicates(subset='name', keep='first')

        if 'diag' in metadata: 
            diag = pd.DataFrame.from_dict(metadata['diag'])
            if diag.duplicated(subset=['name']).any():
                diag = diag.sort_values(by=['score'],ascending=False)
                diag = diag.drop_duplicates(subset='name', keep='first')
            
            data = pd.concat([diag, feats])
        else:
            data = feats  
        
        data = data.set_index('name').transpose()
        data= data.reset_index(drop=True)
        data['uid'] = metadata['uid']
        data['task'] = metadata['task']
        data['subject'] = metadata['subject']
        if all_data is None:
            all_data = data
        else:
            all_data = pd.concat([all_data,data], ignore_index=True)
            #all_data = all_data.reset_index(drop=True)
        i += 1
    all_data = all_data.set_index('uid')
    return all_data

def edit_features(data):
    data = data.replace([-1],np.nan)
    data = data.replace([6],np.nan)
    data = data.replace([1],0)
    data = data.replace([2,3,4,5],1)
    return data

def select_split(data, train_tasks,test_tasks,split_size, spk_split=None):
    if spk_split is not None:
        test = data.loc[data['task'].isin(test_tasks)]
        test_spks = test['subject']
        test_spks = test_spks.drop_duplicates().to_list()
        test_spks = [s for s in test_spks if s in spk_split] 
    
    else: 
        test = data.loc[data['task'].isin(test_tasks)]
        test_spks = test['subject']
        test_spks = test_spks.drop_duplicates().to_list()
    
    if split_size[1] != len(test_spks):
        test_spks = sample(test_spks,split_size[1])

    
    train = data.loc[data['task'].isin(train_tasks)]
    train_spks = train['subject']
    train_spks = train_spks.drop_duplicates().to_list()
    train_spks = [s for s in train_spks if s not in test_spks]

    if split_size[0] != len(train_spks):
        train_spks = sample(train_spks,split_size[0])

    train = train.loc[train['subject'].isin(train_spks)]
    test = test.loc[test['subject'].isin(test_spks)]

    if len(split_size) == 3:
        val_spks = sample(train_spks,split_size[2])
        train_noval_spks = [s for s in train_spks if s not in val_spks]
        val = train.loc[train['subject'].isin(val_spks)]
        train_noval = train.loc[train['subject'].isin(train_noval_spks)]
        return train, train_noval, val, test

    else:
        return train, test


def main():
    parser = argparse.ArgumentParser()
    #Inputs
    parser.add_argument('-i','--prefix',default='', help='Input directory or location in google cloud storage bucket containing files to load')
    parser.add_argument("-s", "--study", default='', help="specify study name")
    parser.add_argument("--split_size", default=[538,96,50], nargs="+", type=int)
    parser.add_argument("--train_task", default=["Alternating Motion Rate"], nargs="+")
    parser.add_argument("--test_task", default=["Alternating Motion Rate"], nargs="+")
    parser.add_argument("--csv_path", default=None, help="specify if there is an already existing csv")
    parser.add_argument("--existing_spk_split", default=None)
    parser.add_argument("--edit_feats",default=False, type=ast.literal_eval)
    #GCS
    parser.add_argument('-b','--bucket_name', default=None, help="google cloud storage bucket name")
    parser.add_argument('-p','--project_name', default=None, help='google cloud platform project name')
    parser.add_argument('--cloud', default=False, type=ast.literal_eval, help="Specify whether to save everything to cloud")
    parser.add_argument('--cloud_dir', default='', type=str, help="if saving to the cloud, you can specify a specific place to save to in the CLOUD bucket")
    #output
    parser.add_argument("-o", "--output_dir", default=".", help='specify LOCAL output directory')
    #OTHER
    parser.add_argument("--debug", default=True, type=ast.literal_eval)
    args = parser.parse_args()

    # (1) Set up GCS
    if args.bucket_name is not None:
        storage_client = storage.Client(project=args.project_name)
        bucket = storage_client.bucket(args.bucket_name)
    else:
        bucket = None

    # (2), check if given study or if the prefix is the full prefix.
    if args.study is not None:
        args.prefix = os.path.join(args.prefix, args.study)

    # (3) check output dir exists
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    #test_df = pd.read_csv(test_path, index_col = 'uid')
    if args.csv_path is None:
        all_data = load_metadata_gcs(args.prefix, bucket)
        path = args.prefix.replace("/","_")
        path = os.path.join(args.output_dir,path + '.csv' )
        all_data.to_csv(path)

        if args.cloud:
            upload(args.cloud_dir, path, bucket)
    else:
        all_data = pd.read_csv(args.csv_path, index_col='uid')

    if args.edit_feats:
        all_data = edit_features(all_data)
        path = args.prefix.replace("/","_")
        path = os.path.join(args.output_dir,path + '.csv' )
        all_data.to_csv(path)
        if args.cloud:
            upload(args.cloud_dir, path, bucket)

    if args.existing_spk_split is not None:
        spk_split = pd.read_csv(args.existing_spk_split)
        spk_split = spk_split['speakerID'].to_list()
    else:
        spk_split = None
    

    if len(args.split_size) == 3:
        train, train_noval, val, test = select_split(all_data, args.train_task, args.test_task, args.split_size, spk_split)
        train.to_csv(os.path.join(args.output_dir, 'train.csv'))
        train_noval.to_csv(os.path.join(args.output_dir, 'train_noval.csv'))
        val.to_csv(os.path.join(args.output_dir, 'val.csv'))
        test.to_csv(os.path.join(args.output_dir, 'test.csv'))

        if args.cloud:
            upload(args.cloud_dir, os.path.join(args.output_dir, 'train.csv'), bucket)
            upload(args.cloud_dir, os.path.join(args.output_dir, 'train_noval.csv'), bucket)
            upload(args.cloud_dir, os.path.join(args.output_dir, 'val.csv'), bucket)
            upload(args.cloud_dir, os.path.join(args.output_dir, 'test.csv'), bucket)
    else:
        train, test = select_split(all_data, args.train_task, args.test_task, args.split_size, spk_split)
        train.to_csv(os.path.join(args.output_dir, 'train.csv'))
        test.to_csv(os.path.join(args.output_dir, 'test.csv'))

        if args.cloud:
            upload(args.cloud_dir, os.path.join(args.output_dir, 'train.csv'), bucket)
            upload(args.cloud_dir, os.path.join(args.output_dir, 'test.csv'), bucket)

if __name__ == "__main__":
    main()