
'''
W2V2 run function
Performs fine-tuning of a classification head, evaluation, or embedding extraction. 

Last modified: 05/2023
Author: Daniela Wiepert
Email: wiepert.daniela@mayo.edu
File: run.py
'''

#IMPORTS
#built-in
import argparse
import json
import numpy as np
import os
import pandas as pd
import pickle
import pyarrow

#third-party
import torch
import torchvision
from tqdm import tqdm

from google.cloud import storage, bigquery
from sklearn.metrics import roc_auc_score, roc_curve
from torch.utils.data import  DataLoader

#local
from utilities.dataloader_utils import *
from models.w2v2_models import *

def download_checkpoint(checkpoint, bucket):
    print('Downloading checkpoint')
    folder = os.path.basename(checkpoint)
    if not os.path.exists(folder):
        os.makedirs(folder)

    blobs = bucket.list_blobs(prefix=checkpoint)
    for blob in blobs:
        destination_uri = '{}/{}'.format(folder, os.path.basename(blob.name))
        if not os.path.exists(destination_uri):
            blob.download_to_filename(destination_uri)
    return folder

def download(gcs_path,outpath, bucket):
    file = os.path.basename(gcs_path)
    blob = bucket.blob(gcs_path)
    destination_uri = '{}/{}'.format(outpath, file)
    if not os.path.exists(destination_uri):
        blob.download_to_filename(destination_uri)
    return destination_uri

def upload(gcs_prefix, path, bucket):
    assert bucket is not None, 'no bucket given for uploading'
    if gcs_prefix is None:
        gcs_prefix = os.path.dirname(path)
    blob = bucket.blob(os.path.join(gcs_prefix, os.path.basename(path)))
    blob.upload_from_filename(path)


def load_traintest(args):
    """
    Load the train and test data from a directory. Assumes the train and test data will exist in this directory under train.csv and test.csv
    :param args: dict with all the argument values
    :return diag_train, diag_test: dataframes with target labels selected
    """
    train_path = f'{args.data_split_root}/train.csv'
    test_path = f'{args.data_split_root}/test.csv'
    #get data
    train_df = pd.read_csv(train_path, index_col = 'uid')
    test_df = pd.read_csv(test_path, index_col = 'uid')

    #randomly sample to get validation set 
    val_df = train_df.sample(50)
    train_df = train_df.drop(val_df.index)

    #save validation set
    val_path = os.path.join(args.exp_dir, 'validation.csv')
    val_df.to_csv(val_path, index=True)

    if args.cloud:
        upload(args.cloud_dir, val_path, args.bucket)

    #get min number of columns containing all the target label columns
    diag_train = train_df[args.target_labels]
    diag_test = test_df[args.target_labels]
    diag_val = val_df[args.target_labels]
    return diag_train, diag_val, diag_test


def get_transform(args):
    """
    Set up pre-processing transform for raw samples 
    Loads data, reduces to 1 channel, downsamples, trims silence, truncate(?) and run feature extraction
    :param args: dict with all the argument values
    return: transform: transforms object 
    """
    waveform_loader = UidToWaveform(prefix = args.prefix, bucket=args.bucket, lib=args.lib)
    transform_list = [waveform_loader]
    if args.reduce:
        channel_sum = lambda w: torch.sum(w, axis = 0).unsqueeze(0)
        mono_tfm = ToMonophonic(reduce_fn = channel_sum)
        transform_list.append(mono_tfm)
    if args.resample_rate != 0: #16000
        downsample_tfm = Resample(args.resample_rate)
        transform_list.append(downsample_tfm)
    if args.trim:
        trim_tfm = TrimSilence()
        transform_list.append(trim_tfm)
    if args.clip_length != 0: #160000
        truncate_tfm = Truncate(length = args.clip_length)
        transform_list.append(truncate_tfm)

    tensor_tfm = ToTensor()
    transform_list.append(tensor_tfm)
    feature_tfm = Wav2VecFeatureExtractor(args.checkpoint)
    transform_list.append(feature_tfm)
    transform = torchvision.transforms.Compose(transform_list)
    return transform

def train_loop(args, model, dataloader_train, dataloader_val=None):
    """
    Training loop for finetuning the w2v2 classification head. 
    :param args: dict with all the argument values
    :param model: W2V2 model
    :param dataloader_train: dataloader object with training data
    :param dataloader_val: dataloader object with validation data
    :return model: fine-tuned w2v2 model
    """
    print('Training start')
    #send to gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    #loss
    if args.loss == 'MSE':
        criterion = torch.nn.MSELoss()
    elif args.loss == 'BCE':
        criterion = torch.nn.BCEWithLogitsLoss()
    else:
        raise ValueError('MSE must be given for loss parameter')
    #optimizer
    if args.optim == 'adam':
        optim = torch.optim.Adam([p for p in model.parameters() if p.requires_grad],lr=args.learning_rate)
    else:
        raise ValueError('adam must be given for optimizer parameter')
    
    if args.scheduler == 'onecycle':
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optim, max_lr=args.max_lr, steps_per_epoch=len(dataloader_train), epochs=args.epochs)
    else:
        scheduler = None
    
    #train
    for e in range(args.epochs):
        training_loss = list()
        #t0 = time.time()
        model.train()
        for batch in tqdm(dataloader_train):
            x = torch.squeeze(batch['waveform'], dim=1)
            targets = batch['targets']
            x, targets = x.to(device), targets.to(device)
            optim.zero_grad()
            o = model(x)
            loss = criterion(o, targets)
            loss.backward()
            optim.step()
            if scheduler is not None:
                scheduler.step()
            loss_item = loss.item()
            training_loss.append(loss_item)

        if e % 10 == 0:
            #SET UP LOGS
            if scheduler is not None:
                lr = scheduler.get_last_lr()
            else:
                lr = args.learning_rate
            logs = {'epoch': e, 'optim':args.optim, 'loss_fn': args.loss, 'lr': lr}
    
            logs['training_loss_list'] = training_loss
            training_loss = np.array(training_loss)
            logs['running_loss'] = np.sum(training_loss)
            logs['training_loss'] = np.mean(training_loss)

            print('RUNNING LOSS', e, np.sum(training_loss) )
            print(f'Training loss: {np.mean(training_loss)}')

            if dataloader_val is not None:
                print("Validation start")
                validation_loss = val_loop(model, criterion, dataloader_val)

                logs['val_loss_list'] = validation_loss
                validation_loss = np.array(validation_loss)
                logs['val_running_loss'] = np.sum(validation_loss)
                logs['val_loss'] = np.mean(validation_loss)
                
                print('RUNNING VALIDATION LOSS',e, np.sum(validation_loss) )
                print(f'Validation loss: {np.mean(validation_loss)}')
            
            #SAVE LOGS
            json_string = json.dumps(logs)
            logs_path = os.path.join(args.exp_dir, 'logs_epoch{}.json'.format(e))
            with open(logs_path, 'w') as outfile:
                json.dump(json_string, outfile)
            
            #SAVE CURRENT MODEL
            mdl_path = os.path.join(args.exp_dir, 'w2v2_mdl_epoch{}.pt'.format(e))
            torch.save(model.state_dict(), mdl_path)
            
            if args.cloud:
                upload(args.cloud_dir, logs_path, args.bucket)
                #upload_from_memory(model.state_dict(), args.cloud_dir, mdl_path, args.bucket)
                upload(args.cloud_dir, mdl_path, args.bucket)

    print('Training finished')
    mdl_path = os.path.join(args.exp_dir, '{}_epoch{}_w2v2_mdl.pt'.format(args.dataset,args.epochs))
    torch.save(model.state_dict(), mdl_path)

    if args.cloud:
        upload(args.cloud_dir, mdl_path, args.bucket)

    return model

def val_loop(model, criterion, dataloader_val):
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
    

def eval_loop(args, model, dataloader_eval):
    """
    Start model evaluation
    :param args: dict with all the argument values
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
    pred_path = os.path.join(args.exp_dir, 'eval_predictions.pt')
    target_path = os.path.join(args.exp_dir, 'eval_targets.pt')
    torch.save(outputs, pred_path)
    torch.save(t, target_path)

    if args.cloud:
        upload(args.cloud_dir, pred_path, args.bucket)
        upload(args.cloud_dir, target_path, args.bucket)

    return outputs, t

def embedding_loop(model, dataloader):
    """
    Run a specific subtype of evaluation for getting embeddings.
    :param args: dict with all the argument values
    :param model: W2V2 model
    :param dataloader_eval: dataloader object with data to get embeddings for
    :return embeddings: an np array containing the embeddings
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
            e = model(x)
            e = e.cpu().numpy()
            if embeddings.size == 0:
                embeddings = e
            else:
                embeddings = np.append(embeddings, e, axis=0)
        
    return embeddings

def metrics(args, preds, targets):
    """
    Get AUC scores, doesn't return, just saves the metrics to a csv
    :param args: dict with all the argument values
    :param preds: model predictions
    :param targets: model targets (actual values)
    """
    #get AUC score and all data for ROC curve
    metrics = {}
    pred_mat=torch.sigmoid(preds).numpy()
    target_mat=targets.numpy()
    aucs=roc_auc_score(target_mat, pred_mat, average = None)
    print(aucs)
    data = pd.DataFrame({'Label':args.target_labels, 'AUC':aucs})
    data.to_csv(os.path.join(args.exp_dir, 'aucs.csv'), index=False)
    if args.cloud:
        upload(args.cloud_dir, os.path.join(args.exp_dir, 'aucs.csv'), args.bucket)

def get_embeddings(args):
    """
    Run embedding extraction from start to finish
    :param args: dict with all the argument values
    :param bucket: google storage bucket object where data is saved
    """
    print('Running Embedding Extraction: ')

    # (1) load data to get embeddings for
    assert '.csv' in args.data_split_root, f'A csv file is necessary for embedding extraction. Please make sure this is a full file path: {args.data_split_root}'
    df = pd.read_csv(args.data_split_root, index_col = 'uid')
    annotations_df = df[args.target_labels]

    if args.debug:
        annotations_df = annotations_df.iloc[0:8,:]

    # (2) get transforms
    transform = get_transform(args)
    
    # (3) set up dataloaders
    waveform_dataset = WaveformDataset(annotations_df = annotations_df, target_labels = args.target_labels, transform = transform)
    dataloader = DataLoader(waveform_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    
    # (4) set up embedding model
    if args.mdl_path[:5] =='gs://':
        mdl_path = args.mdl_path[5:].replace(args.bucket_name,'')[1:]
        args.mdl_path = download(mdl_path, args.exp_dir, args.bucket)
    model = Wav2Vec2ForEmbeddingExtraction(args.checkpoint, args.pooling_mode, args.mdl_path)
    
    # (5) get embeddings
    embeddings = embedding_loop(model, dataloader)
        
    df_embed = pd.DataFrame([[r] for r in embeddings], columns = ['embedding'], index=annotations_df.index)

    try:
        pqt_path = '{}/{}_w2v2_embeddings.pqt'.format(args.exp_dir, args.dataset)
        df_embed.to_parquet(path=pqt_path, index=True, engine='pyarrow') #TODO: fix

        if args.cloud:
            upload(args.cloud_dir, pqt_path, args.bucket)
    except:
        print('Unable to save as pqt, saving instead as csv')
        csv_path = '{}/{}_w2v2_embeddings.csv'.format(args.exp_dir, args.dataset)
        df_embed.to_csv(csv_path, index=True)

        if args.cloud:
            upload(args.cloud_dir, csv_path, args.bucket)

    return df_embed

def finetuning(args):
    """
    Run finetuning from start to finish
    :param args: dict with all the argument values
    """
    print('Running finetuning: ')
    # (1) load data
    assert '.csv' not in args.data_split_root, f'May have given a full file path, please confirm this is a directory: {args.data_split_root}'
    diag_train, diag_val, diag_test = load_traintest(args)

    if args.debug:
        diag_train = diag_train.iloc[0:8,:]
        diag_val = diag_val.iloc[0:8,:]
        diag_test = diag_test.iloc[0:8,:]

    # (2) get data transforms    
    transform = get_transform(args)

    # (3) set up datasets and dataloaders
    dataset_train = WaveformDataset(diag_train, target_labels = args.target_labels, transform = transform)
    dataset_val = WaveformDataset(diag_val, target_labels = args.target_labels, transform = transform)
    dataset_test = WaveformDataset(diag_test, target_labels = args.target_labels, transform = transform)

    dataloader_train = DataLoader(dataset_train, batch_size = args.batch_size, shuffle = True, num_workers = args.num_workers)
    dataloader_val= DataLoader(dataset_val, batch_size = 1, shuffle = False, num_workers = args.num_workers)
    dataloader_test= DataLoader(dataset_test, batch_size = args.batch_size, shuffle = False, num_workers = args.num_workers)
    #dataloader_test = DataLoader(dataset_test, batch_size = len(diag_test), shuffle = False, num_workers = args.num_workers)

    # (4) initialize model
    model = Wav2Vec2ForSpeechClassification(args.checkpoint, args.pooling_mode, args.n_class)
    
    # (5) start fine-tuning classification
    model = train_loop(args, model, dataloader_train, dataloader_val)

    # (6) start evaluating
    preds, targets = eval_loop(args, model, dataloader_test)

    # (7) performance metrics
    metrics(args, preds, targets)

def eval_only(args):
    """
    Run only evaluation of a pre-existing model
    :param args: dict with all the argument values
    """
   # (1) load data
    if '.csv' in args.data_split_root: 
        df = pd.read_csv(args.data_split_root, index_col = 'uid')
        diag_eval = df[args.target_labels]
    else:
        diag_train, diag_val, diag_eval = load_traintest(args)
    
    if args.debug:
        diag_eval = diag_eval.iloc[0:8,:]

    # (2) get data transforms    
    transform = get_transform(args)

    # (3) set up datasets and dataloaders
    dataset_eval = WaveformDataset(diag_eval, target_labels = args.target_labels, transform = transform)
    dataloader_eval= DataLoader(dataset_eval, batch_size = args.batch_size, shuffle = False, num_workers = args.num_workers)
    #dataloader_test = DataLoader(dataset_test, batch_size = len(diag_test), shuffle = False, num_workers = args.num_workers)

    # (4) initialize model
    model = Wav2Vec2ForSpeechClassification(args.checkpoint, args.pooling_mode, args.n_class)

    # (5) load fine-tuned model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.mdl_path[:5] =='gs://':
        mdl_path = args.mdl_path[5:].replace(args.bucket_name,'')[1:]
        args.mdl_path = download(mdl_path, args.exp_dir, args.bucket)
    sd = torch.load(args.mdl_path, map_location=device)
    model.load_state_dict(sd, strict=False)

    # (6) start evaluating
    preds, targets = eval_loop(args, model, dataloader_eval)

    # (7) performance metrics
    metrics(args, preds, targets)

def main():
    parser = argparse.ArgumentParser()
    #Inputs
    parser.add_argument('-i','--prefix',default='speech_ai/speech_lake/', help='Input directory or location in google cloud storage bucket containing files to load')
    parser.add_argument("-s", "--study", choices = ['r01_prelim','speech_poc_freeze_1', None], default='speech_poc_freeze_1', help="specify study name")
    parser.add_argument("-d", "--data_split_root", default='gs://ml-e107-phi-shared-aif-us-p/speech_ai/share/data_splits/amr_subject_dedup_594_train_100_test_binarized_v20220620', help="specify file path where datasplit is located. If you give a full file path to classification, an error will be thrown. On the other hand, evaluation and embedding expects a single .csv file.")
    parser.add_argument('-l','--label_txt', default='/Users/m144443/Documents/GitHub/mayo-w2v2/labels.txt')
    parser.add_argument('--lib', default=False, type=bool, help="Specify whether to load using librosa as compared to torch audio")
    #GCS
    parser.add_argument('-b','--bucket_name', default='ml-e107-phi-shared-aif-us-p', help="google cloud storage bucket name")
    parser.add_argument('-p','--project_name', default='ml-mps-aif-afdgpet01-p-6827', help='google cloud platform project name')
    parser.add_argument('--cloud', default=True, type=bool, help="Specify whether to save everything to cloud")
    #output
    parser.add_argument("--dataset", default=None,type=str, help="When saving, the dataset arg is used to set file names. If you do not specify, it will assume the lowest directory from data_split_root")
    parser.add_argument("-o", "--exp_dir", default="./experiments", help='specify LOCAL output directory')
    parser.add_argument('--cloud_dir', default='m144443/temp_out', type=str, help="if saving to the cloud, you can specify a specific place to save to in the CLOUD bucket")
    #Mode specific
    parser.add_argument("-m", "--mode", choices=['finetune','eval-only','extraction'], default='finetune')
    parser.add_argument("-mp", "--mdl_path", default='gs://ml-e107-phi-shared-aif-us-p/m144443/temp_out/amr_subject_dedup_594_train_100_test_binarized_v20220620_epoch1_w2v2_mdl.pt', help='If running eval-only or extraction, you have the option to load a fine-tuned model by specifying the save path here. If passed a gs:// file, will download to local machine.')
    #Audio transforms
    parser.add_argument("--resample_rate", default=16000,type=int, help='resample rate for audio files')
    parser.add_argument("--reduce", default=True, type=bool, help="Specify whether to reduce to monochannel")
    parser.add_argument("--clip_length", default=160000, type=int, help="If truncating audio, specify clip length in # of frames. 0 = no truncation")
    parser.add_argument("--trim", default=True, type=int, help="trim silence")
    #Model parameters
    parser.add_argument("-c", "--checkpoint", default="gs://ml-e107-phi-shared-aif-us-p/m144443/checkpoints/wav2vec2-base-960h", help="specify path to pre-trained model weight checkpoint")
    parser.add_argument("-pm", "--pooling_mode", default="mean", help="specify method of pooling last hidden layer", choices=['mean','sum','max'])
    parser.add_argument("-bs", "--batch_size", type=int, default=8, help="specify batch size")
    parser.add_argument("-nw", "--num_workers", type=int, default=0, help="specify number of parallel jobs to run for data loader")
    parser.add_argument("-lr", "--learning_rate", type=float, default=0.0003, help="specify learning rate")
    parser.add_argument("-e", "--epochs", type=int, default=1, help="specify number of training epochs")
    parser.add_argument("--optim", type=str, default="adam", help="training optimizer", choices=["adam"])
    parser.add_argument("--loss", type=str, default="BCE", help="the loss function for finetuning, depend on the task", choices=["MSE", "BCE"])
    parser.add_argument("--scheduler", type=str, default="onecycle", help="specify lr scheduler", choices=["onecycle", None])
    parser.add_argument("--max_lr", type=float, default=0.01, help="specify max lr for lr scheduler")
    #OTHER
    parser.add_argument("--debug", default=True, type=bool)
    args = parser.parse_args()
    
    print('Torch version: ',torch.__version__)
    print('Cuda availability: ', torch.cuda.is_available())
    print('Cuda version: ', torch.version.cuda)
    
    #variables
    # (1) Set up GCS
    if args.bucket_name is not None:
        storage_client = storage.Client(project=args.project_name)
        bucket = storage_client.bucket(args.bucket_name)
    else:
        bucket = None

    # (2), check if given study or if the prefix is the full prefix.
    if args.study is not None:
        args.prefix = os.path.join(args.prefix, args.study)
    
    # (3) get dataset name
    if args.dataset is None:
        if '.csv' in args.data_split_root:
            args.dataset = '{}_{}'.format(os.path.basename(os.path.dirname(args.data_split_root)), os.path.basename(args.data_split_root[:-4]))
        else:
            args.dataset = os.path.basename(args.data_split_root)
    
    # (4) get target labels
     #get list of target labels
    with open(args.label_txt) as f:
        target_labels = f.readlines()
    target_labels = [l.strip() for l in target_labels]
    args.target_labels = target_labels

    args.n_class = len(target_labels)

    # (5) check if output directory exists, SHOULD NOT BE A GS:// path
    if not os.path.exists(args.exp_dir):
        os.makedirs(args.exp_dir)

    # (6) check that clip length has been set
    if args.clip_length == 0:
        try: 
            assert args.batch_size == 1, 'Not currently compatible with different length wav files unless batch size has been set to 1'
        except:
            args.batch_size = 1

    # (7) check if checkpoint is stored in gcs bucket
    if args.checkpoint[:5] =='gs://':
        checkpoint = args.checkpoint[5:].replace(args.bucket_name,'')[1:]
        checkpoint = download_checkpoint(checkpoint, bucket)
        args.checkpoint = checkpoint

    # (8) dump arguments
    args_path = "%s/args.pkl" % args.exp_dir
    with open(args_path, "wb") as f:
        pickle.dump(args, f)
    #in case of error, everything is immediately uploaded to the bucket
    if args.cloud:
        upload(args.cloud_dir, args_path, bucket)

    #(9) add bucket to args
    args.bucket = bucket


    # (10) run model
    print(args.mode)
    if args.mode == "finetune":
        finetuning(args)

    elif args.mode == 'eval-only':
        eval_only(args)
              
    elif args.mode == "extraction":
        df_embed = get_embeddings(args)
    
if __name__ == "__main__":
    main()