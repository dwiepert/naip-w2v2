
'''
W2V2 run function
Performs fine-tuning of a classification head, evaluation, or embedding extraction. 

Last modified: 05/2023
Author: Daniela Wiepert
Email: wiepert.daniela@mayo.edu
File: wav2vec2.py
'''

#IMPORTS
#built-in
import argparse
import numpy as np
import os
import pandas as pd

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

def load_traintest(args, bucket):
    """
    Load the train and test data from a directory. Assumes the train and test data will exist in this directory under train.csv and test.csv
    :param args: dict with all the argument values
    :param bucket: google storage bucket object where data is saved
    :return diag_train, diag_test: dataframes with target labels selected
    """
    train_path = f'{args.data_split_root}/train.csv'
    test_path = f'{args.data_split_root}/test.csv'
    #get data
    train_df = pd.read_csv(train_path, index_col = 'uid')
    test_df = pd.read_csv(test_path, index_col = 'uid')

    #get min number of columns containing all the target label columns
    diag_train = train_df[args.target_labels]
    diag_test = test_df[args.target_labels]
    print(len(diag_test))
    return diag_train, diag_test

def load_csv(args, bucket):
    df = pd.read_csv(args.data_split_root, index_col = 'uid')
    diag_df = df[args.target_labels]
    return diag_df

def get_transform(args, bucket):
    """
    Set up pre-processing transform for raw samples 
    Loads data, reduces to 1 channel, downsamples, trims silence, truncate(?) and run feature extraction
    :param args: dict with all the argument values
    :param bucket: google storage bucket object where data is saved
    return: transform: transforms object 
    """
    waveform_loader = UidToWaveform(prefix = args.prefix, bucket=bucket, lib=args.lib)
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

def train_loop(args, model, dataloader_train):
    """
    Training loop for finetuning the w2v2 classification head. 
    :param args: dict with all the argument values
    :param model: W2V2 model
    :param dataloader_train: dataloader object with training data
    :return model: fine-tuned w2v2 model
    """
    print('Training start')
    #send to gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.train()
    #loss
    if args.loss == 'MSE':
        criterion = torch.nn.MSELoss()
    else:
        raise ValueError('MSE must be given for loss parameter')
    #optimizer
    if args.optim == 'adam':
        optim = torch.optim.Adam([p for p in model.parameters() if p.requires_grad],lr=args.learning_rate)
    else:
        raise ValueError('adam must be given for optimizer parameter')

    #train
    for e in range(args.epochs):
        running_loss = 0
        #t0 = time.time()
        for batch in tqdm(dataloader_train):
            x = torch.squeeze(batch['waveform'])
            targets = batch['targets']
            x, targets = x.to(device), targets.to(device)
            optim.zero_grad()
            o = model(x)
            loss = criterion(o, targets)
            loss.backward()
            optim.step()
            loss_item = loss.item()
            running_loss += loss_item

        print('RUNNING LOSS', e, running_loss)

    outname = "_".join(['w2v2_mdl', args.dataset, str(args.n_class), args.optim, str(args.epochs)+'epoch'])+'.pt'
    outpath = os.path.join(args.exp_dir,outname)
    torch.save(model.state_dict(), outpath)

    torch.cuda.empty_cache()
    return model

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
            x = torch.squeeze(batch['waveform'])
            x = x.to(device)
            targets = batch['targets']
            targets = targets.to(device)
            o = model(x)
            outputs.append(o)
            t.append(targets)
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
            x = torch.squeeze(batch['waveform'])
            x = x.to(device)
            e = model(x)
            e = e.cpu().numpy()
            if embeddings.size == 0:
                embeddings = e
            else:
                embeddings = np.append(embeddings, e, axis=0)
        
    print(embeddings.shape)
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
    pred_mat=torch.sigmoid(torch.cat(preds)).cpu().detach().numpy()
    target_mat=torch.cat(targets).cpu().detach().numpy()
    aucs=roc_auc_score(target_mat, pred_mat, average = None)
    print(aucs)
    data = pd.DataFrame({'Label':args.target_labels, 'AUC':aucs})
    data.to_csv(os.path.join(args.exp_dir, 'metrics.csv'), index=False)

def get_embeddings(args, bucket):
    """
    Run embedding extraction from start to finish
    :param args: dict with all the argument values
    :param bucket: google storage bucket object where data is saved
    """
    print('Running Embedding Extraction: ')

    # (1) load data to get embeddings for
    assert '.csv' in args.data_split_root, f'A csv file is necessary for embedding extraction. Please make sure this is a full file path: {args.data_split_root}'
    annotations_df = load_csv(args, bucket)

    # (2) get transforms
    transform = get_transform(args, bucket)
    
    # (3) set up dataloaders
    waveform_dataset = WaveformDataset(annotations_df = annotations_df, target_labels = args.target_labels, transform = transform)
    dataloader = DataLoader(waveform_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    
    # (4) set up embedding model
    model = Wav2Vec2ForEmbeddingExtraction(args.checkpoint, args.pooling_mode, args.mdl_path)
    
    # (5) get embeddings
    embeddings = embedding_loop(model, dataloader)
        
    df_embed = pd.DataFrame([[r] for r in embeddings], columns = ['embedding'], index=annotations_df.index)

    #csv outpath
    outname = "_".join([args.dataset, 'w2v2_embeddings'])+'.csv'
    csv_outpath = os.path.join(args.exp_dir, outname)
    df_embed.to_csv(csv_outpath)

    outname = "_".join([args.dataset, 'w2v2_embeddings'])+'.pqt'
    outpath = os.path.join(args.exp_dir,outname)
    df_embed.to_parquet(path=outpath, index=True) #TODO: fix
    
    return df_embed

def finetuning(args, bucket):
    """
    Run finetuning from start to finish
    :param args: dict with all the argument values
    :param bucket: google storage bucket object where data is saved
    """
    print('Running finetuning: ')
    # (1) load data
    assert '.csv' not in args.data_split_root, f'May have given a full file path, please confirm this is a directory: {args.data_split_root}'
    diag_train, diag_test = load_traintest(args, bucket)

    # (2) get data transforms    
    transform = get_transform(args, bucket)

    # (3) set up datasets and dataloaders
    dataset_train = WaveformDataset(diag_train, target_labels = args.target_labels, transform = transform)
    dataset_test = WaveformDataset(diag_test, target_labels = args.target_labels, transform = transform)

    dataloader_train = DataLoader(dataset_train, batch_size = args.batch_size, shuffle = True, num_workers = args.num_workers)
    dataloader_test= DataLoader(dataset_test, batch_size = args.batch_size, shuffle = False, num_workers = args.num_workers)
    #dataloader_test = DataLoader(dataset_test, batch_size = len(diag_test), shuffle = False, num_workers = args.num_workers)

    # (4) initialize model
    model = Wav2Vec2ForSpeechClassification(args.checkpoint, args.pooling_mode, args.n_class)
    
    # (5) start fine-tuning classification
    model = train_loop(args, model, dataloader_train)

    # (6) start evaluating
    preds, targets = eval_loop(args, model, dataloader_test)

    # (7) performance metrics
    metrics(args, preds, targets)

def eval_only(args, bucket):
    """
    Run only evaluation of a pre-existing model
    :param args: dict with all the argument values
    :param bucket: google storage bucket object where data is saved
    """
   # (1) load data
    diag_eval = load_csv(args, bucket)

    # (2) get data transforms    
    transform = get_transform(args, bucket)

    # (3) set up datasets and dataloaders
    dataset_eval = WaveformDataset(diag_eval, target_labels = args.target_labels, transform = transform)
    dataloader_eval= DataLoader(dataset_eval, batch_size = args.batch_size, shuffle = False, num_workers = args.num_workers)
    #dataloader_test = DataLoader(dataset_test, batch_size = len(diag_test), shuffle = False, num_workers = args.num_workers)

    # (4) initialize model
    model = Wav2Vec2ForSpeechClassification(args.checkpoint, args.pooling_mode, args.n_class)

    # (5) load fine-tuned model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sd = torch.load(args.mdl_path, map_location=device)
    model.load_state_dict(sd, strict=False)

    # (6) start evaluating
    preds, targets = eval_loop(args, model, dataloader_eval)

    # (7) performance metrics
    metrics(args, preds, targets)
    

def main():
    parser = argparse.ArgumentParser()
    #google cloud storage
    parser.add_argument('-i','--prefix',default='speech_ai/speech_lake/', help='Input directory or location in google cloud storage bucket containing files to load')
    parser.add_argument("-s", "--study", choices = ['r01_prelim','speech_poc_freeze_1', None], default='speech_poc_freeze_1', help="specify study name")
    parser.add_argument("-d", "--data_split_root", default='gs://ml-e107-phi-shared-aif-us-p/speech_ai/share/data_splits/amr_subject_dedup_594_train_100_test_binarized_v20220620/test.csv', help="specify file path where datasplit is located. If you give a full file path to classification, an error will be thrown. On the other hand, evaluation and embedding expects a single .csv file.")
    parser.add_argument('-l','--label_txt', default='/Users/m144443/Documents/GitHub/mayo-w2v2/labels.txt')
    #GCS
    parser.add_argument('-b','--bucket_name', default='ml-e107-phi-shared-aif-us-p', help="google cloud storage bucket name")
    parser.add_argument('-p','--project_name', default='ml-mps-aif-afdgpet01-p-6827', help='google cloud platform project name')
    #librosa vs torchaudio
    parser.add_argument('--lib', default=True, type=bool, help="Specify whether to load using librosa as compared to torch audio")
    #output
    parser.add_argument("--dataset", default='amr_subject_dedup_594_test_binarized_v2022062',type=str, help="the dataset used for training")
    parser.add_argument("-o", "--exp_dir", default="/Users/m144443/Documents/GitHub/mayo-w2v2/experiments")
    #Audio transforms
    parser.add_argument("--resample_rate", default=16000,type=int, help='resample rate for audio files')
    parser.add_argument("--reduce", default=True, type=bool, help="Specify whether to reduce to monochannel")
    parser.add_argument("--clip_length", default=160000, type=int, help="If truncating audio, specify clip length in # of frames. 0 = no truncation")
    parser.add_argument("--trim", default=True, type=int, help="trim silence")
    #Mode specific
    parser.add_argument("-m", "--mode", choices=['finetune','eval-only','extraction'], default='extraction')
    parser.add_argument("-mp", "--mdl_path", default='/Users/m144443/Documents/GitHub/mayo-w2v2/experiments/w2v2_mdl_amr_subject_dedup_594_train_test_binarized_v2022062_6_adam_1epoch.pt', help='If running eval-only or extraction, you have the option to load a fine-tuned model by specifying the save path here.')
    #Model parameters
    parser.add_argument("-c", "--checkpoint", default="facebook/wav2vec2-base-960h", help="specify path to pre-trained model weight checkpoint")
    parser.add_argument("-n", "--num_labels", type=int, default=6, help="specify number of features to classify")
    parser.add_argument("-pm", "--pooling_mode", default="mean", help="specify method of pooling last hidden layer", choices=['mean','sum','max'])
    parser.add_argument("-si","--start", type=int, default=16, help="specify starting column index where target label data can be located in data table")
    parser.add_argument("-ei", "--end", type=int, default=27, help="specify ending column index where target label data can be located in data table, starting index - ending index should encompass all target_labels columns")
    parser.add_argument("-bs", "--batch_size", type=int, default=8, help="specify batch size")
    parser.add_argument("-nw", "--num_workers", type=int, default=0, help="specify number of parallel jobs to run for data loader")
    parser.add_argument("-lr", "--learning_rate", type=float, default=0.0003, help="specify learning rate")
    parser.add_argument("-e", "--epochs", type=int, default=1, help="specify number of training epochs")
    parser.add_argument("--optim", type=str, default="adam", help="training optimizer", choices=["adam"])
    parser.add_argument("--loss", type=str, default="MSE", help="the loss function for finetuning, depend on the task", choices=["MSE"])
    args = parser.parse_args()
    
    print('Torch version: ',torch.__version__)
    print('Cuda availability: ', torch.cuda.is_available())
    print('Cuda version: ', torch.version.cuda)
    
    #variables
    # (1) Set up GCS
    if args.bucket_name is not None:
        storage_client = storage.Client(project=args.project_name)
        bq_client = bigquery.Client(project=args.project_name)
        bucket = storage_client.bucket(args.bucket_name)
    else:
        bucket = None

    # (2), check if given study or if the prefix is the full prefix.
    if args.study is not None:
        args.prefix = os.path.join(args.prefix, args.study)
    
    
    # (3) get target labels
     #get list of target labels
    with open(args.label_txt) as f:
        target_labels = f.readlines()
    target_labels = [l.strip() for l in target_labels]
    args.target_labels = target_labels

    args.n_class = len(target_labels)

    # (4) check if output directory exists
    if not os.path.exists(args.exp_dir) and 'gs://' not in args.exp_dir:
        os.mkdir(args.exp_dir)

    # (5) check that clip length has been set
    if args.clip_length == 0:
        assert args.batch_size == 1, 'Not currently compatible with different length wave files unless batch size has been set to 1'

    # (6) run model
    print(args.mode)
    if args.mode == "finetune":
        finetuning(args, bucket)

    elif args.mode == 'eval-only':
        eval_only(args, bucket)
              
              
    elif args.mode == "extraction":
        df_embed = get_embeddings(args, bucket)
    
if __name__ == "__main__":
    main()