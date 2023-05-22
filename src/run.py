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
import os
import pickle

#third-party
import torch
import pandas as pd
import pyarrow

from google.cloud import storage
from torch.utils.data import  DataLoader

#local
from utilities import *
from models import *
from loops import *
from dataloader import W2V2Dataset

def get_embeddings(args):
    """
    Run embedding extraction from start to finish
    :param args: dict with all the argument values
    """
    print('Running Embedding Extraction: ')
    # Get original 
    if args.finetuned_mdl_path is not None:
        model_args, args.finetuned_mdl_path = setup_mdl_args(args, args.finetuned_mdl_path)
    else:
        model_args = args #we do not pretrain from scratch so the checkpoint will never include an args.pkl file


    # (1) load data to get embeddings for
    assert '.csv' in args.data_split_root, f'A csv file is necessary for embedding extraction. Please make sure this is a full file path: {args.data_split_root}'
    annotations_df = pd.read_csv(args.data_split_root, index_col = 'uid') #data_split_root should use the CURRENT arguments regardless of the finetuned model

    if args.debug:
        annotations_df = annotations_df.iloc[0:8,:]

    # (2) set up audio configuration for transforms
    audio_conf = {'checkpoint': args.checkpoint, 'resample_rate':args.resample_rate, 'reduce': args.reduce,
                  'trim': args.trim, 'clip_length': args.clip_length}
    
    # (3) set up dataloaders
    waveform_dataset = W2V2Dataset(annotations_df = annotations_df, target_labels = args.target_labels,
                                   audio_conf = audio_conf, prefix=args.prefix, bucket=args.bucket, librosa=args.lib) #not super important for embeddings, but the dataset should be selecting targets based on the FINETUNED model
    
    dataloader = DataLoader(waveform_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    
    # (4) set up embedding model
    model = Wav2Vec2ForSpeechClassification(checkpoint=model_args.checkpoint, label_dim = model_args.n_class, pooling_mode = model_args.pooling_mode, 
                                            freeze=model_args.freeze, activation=model_args.activation, final_dropout=model_args.final_dropout, 
                                            layernorm=model_args.layernorm, weighted=model_args.weighted, layer=model_args.layer)   #should look like the finetuned model (so using model_args). If pretrained model, will resort to current args
    
    if args.finetuned_mdl_path is not None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        sd = torch.load(args.finetuned_mdl_path, map_location=device)
        model.load_state_dict(sd, strict=False)
    else:
        print(f'Extracting embeddings from only a pretrained model: {args.pretrained_mdl_path}. Extraction method changed to pt.')
        args.embedding_type = 'pt' #manually change the type to 'pt' if not given a finetuned mdl path.

    # (5) get embeddings
    embeddings = embedding_extraction(model, dataloader, args.embedding_type, args.layer, args.pooling_mode)
        
    df_embed = pd.DataFrame([[r] for r in embeddings], columns = ['embedding'], index=annotations_df.index)

    if args.embedding_type == 'ft':
        args.layer='NA'
        args.pooling_mode='NA'
    elif args.embedding_type == 'wt':
        args.layer='NA'
    elif args.layer==-1:
        args.layer='Final'

    try:
        pqt_path = '{}/{}_layer{}_{}_w2v2_{}_embeddings.pqt'.format(args.exp_dir, args.dataset, args.layer, args.pooling_mode,args.embedding_type)
        
        df_embed.to_parquet(path=pqt_path, index=True, engine='pyarrow') #TODO: fix

        if args.cloud:
            upload(args.cloud_dir, pqt_path, args.bucket)
    except:
        print('Unable to save as pqt, saving instead as csv')
        csv_path = '{}/{}_layer{}_{}_w2v2_{}_embeddings.csv'.format(args.exp_dir, args.dataset, args.layer, args.pooling_mode,args.embedding_type)
        
        df_embed.to_csv(csv_path, index=True)

        if args.cloud:
            upload(args.cloud_dir, csv_path, args.bucket)

    print('Embedding extraction finished')
    return df_embed

def finetune_w2v2(args):
    """
    Run finetuning from start to finish
    :param args: dict with all the argument values
    """
    print('Running finetuning: ')
    # (1) load data
    assert '.csv' not in args.data_split_root, f'May have given a full file path, please confirm this is a directory: {args.data_split_root}'
    train_df, val_df, test_df = load_data(args.data_split_root, args.target_labels, args.exp_dir, args.cloud, args.cloud_dir, args.bucket)

    if args.debug:
        train_df = train_df.iloc[0:8,:]
        val_df = val_df.iloc[0:8,:]
        test_df = test_df.iloc[0:8,:]

    # (2) set up audio configuration for transforms
    audio_conf = {'checkpoint': args.checkpoint, 'resample_rate':args.resample_rate, 'reduce': args.reduce,
                  'trim': args.trim, 'clip_length': args.clip_length}
    

    # (3) set up datasets and dataloaders
    dataset_train = W2V2Dataset(train_df, target_labels = args.target_labels,
                                audio_conf = audio_conf, prefix=args.prefix, bucket=args.bucket, librosa=args.lib)
    dataset_val = W2V2Dataset(val_df, target_labels = args.target_labels,
                              audio_conf = audio_conf, prefix=args.prefix, bucket=args.bucket, librosa=args.lib)
    dataset_test = W2V2Dataset(test_df, target_labels = args.target_labels,
                               audio_conf = audio_conf, prefix=args.prefix, bucket=args.bucket, librosa=args.lib)

    dataloader_train = DataLoader(dataset_train, batch_size = args.batch_size, shuffle = True, num_workers = args.num_workers)
    dataloader_val= DataLoader(dataset_val, batch_size = 1, shuffle = False, num_workers = args.num_workers)
    dataloader_test= DataLoader(dataset_test, batch_size = args.batch_size, shuffle = False, num_workers = args.num_workers)
    #dataloader_test = DataLoader(dataset_test, batch_size = len(diag_test), shuffle = False, num_workers = args.num_workers)

    # (4) initialize model
    model = Wav2Vec2ForSpeechClassification(checkpoint=args.checkpoint, label_dim = args.n_class, pooling_mode = args.pooling_mode, 
                                            freeze=args.freeze, activation=args.activation, final_dropout=args.final_dropout, 
                                            layernorm=args.layernorm, weighted=args.weighted, layer=args.layer)    
    
    # (5) start fine-tuning classification
    model = finetune(model, dataloader_train, dataloader_val,
                     args.optim, args.learning_rate, args.weight_decay,
                     args.loss, args.scheduler, args.max_lr, args.epochs,
                     args.exp_dir, args.cloud, args.cloud_dir, args.bucket)

    print('Saving final epoch')

    if model.weighted:
        mdl_path = os.path.join(args.exp_dir, '{}_{}_{}_epoch{}_{}_mdl_weighted.pt'.format(args.dataset, args.n_class, args.optim, args.epochs, os.path.basename(args.checkpoint)))
    else:
        if args.layer==-1:
            args.layer='Final'
        mdl_path = os.path.join(args.exp_dir, '{}_{}_{}_layer{}_epoch{}_{}_mdl.pt'.format(args.dataset, args.n_class, args.optim, args.layer, args.epochs, os.path.basename(args.checkpoint)))
    torch.save(model.state_dict(), mdl_path)

    if args.cloud:
        upload(args.cloud_dir, mdl_path, args.bucket)

    # (6) start evaluating
    preds, targets = evaluation(model, dataloader_test, args.exp_dir, args.cloud, args.cloud_dir, args.bucket)

    print('Finetuning finished')


def eval_only(args):
    """
    Run only evaluation of a pre-existing model
    :param args: dict with all the argument values
    """
    assert args.finetuned_mdl_path is not None, 'Evaluation must be run on a finetuned model, otherwise classification head is completely untrained.'
    # get original model args (or if no finetuned model, uses your original args)
    model_args, args.finetuned_mdl_path = setup_mdl_args(args, args.finetuned_mdl_path)
    
   # (1) load data
    if '.csv' in args.data_split_root: 
        eval_df = pd.read_csv(args.data_split_root, index_col = 'uid')
        if 'distortions' in args.target_labels and 'distortions' not in eval_df.columns:
            eval_df["distortions"]=((eval_df["distorted Cs"]+eval_df["distorted V"])>0).astype(int)
        eval_df = eval_df.dropna(subset=args.target_labels)
    else:
        train_df, val_df, eval_df = load_data(args.data_split_root, args.target_labels, args.exp_dir, args.cloud, args.cloud_dir, args.bucket)
    
    if args.debug:
        eval_df = eval_df.iloc[0:8,:]

    # (2) set up audio configuration for transforms
    audio_conf = {'checkpoint': args.checkpoint, 'resample_rate':args.resample_rate, 'reduce': args.reduce,
                  'trim': args.trim, 'clip_length': args.clip_length}
    
    # (3) set up datasets and dataloaders
    dataset_eval = W2V2Dataset(eval_df, target_labels = model_args.target_labels,
                               audio_conf = audio_conf, prefix=args.prefix, bucket=args.bucket, librosa=args.lib)
    #the dataset should be selecting targets based on the FINETUNED model, so if there is a mismatch, it defaults to the arguments used for finetuning
    dataloader_eval= DataLoader(dataset_eval, batch_size = args.batch_size, shuffle = False, num_workers = args.num_workers)
    #dataloader_test = DataLoader(dataset_test, batch_size = len(diag_test), shuffle = False, num_workers = args.num_workers)

    # (4) initialize model
    model = Wav2Vec2ForSpeechClassification(checkpoint=model_args.checkpoint, label_dim = model_args.n_class, pooling_mode = model_args.pooling_mode, 
                                            freeze=model_args.freeze, activation=model_args.activation, final_dropout=model_args.final_dropout, 
                                            layernorm=model_args.layernorm, weighted=model_args.weighted, layer=model_args.layer)    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sd = torch.load(args.finetuned_mdl_path, map_location=device)
    model.load_state_dict(sd, strict=False)
    
    # (6) start evaluating
    preds, targets = evaluation(model, dataloader_eval, args.exp_dir, args.cloud, args.cloud_dir, args.bucket)

    print('Evaluation finished')


def main():
    parser = argparse.ArgumentParser()
    #Inputs
    parser.add_argument('-i','--prefix',default='speech_ai/speech_lake/', help='Input directory or location in google cloud storage bucket containing files to load')
    parser.add_argument("-s", "--study", choices = ['r01_prelim','speech_poc_freeze_1', None], default='speech_poc_freeze_1', help="specify study name")
    parser.add_argument("-d", "--data_split_root", default='gs://ml-e107-phi-shared-aif-us-p/speech_ai/share/data_splits/amr_subject_dedup_594_train_100_test_binarized_v20220620/test.csv', help="specify file path where datasplit is located. If you give a full file path to classification, an error will be thrown. On the other hand, evaluation and embedding expects a single .csv file.")
    parser.add_argument('-l','--label_txt', default=None) #default=None #default='./labels.txt'
    parser.add_argument('--lib', default=False, type=bool, help="Specify whether to load using librosa as compared to torch audio")
    parser.add_argument("-c", "--checkpoint", default="gs://ml-e107-phi-shared-aif-us-p/m144443/checkpoints/wav2vec2-base-960h", help="specify path to pre-trained model weight checkpoint")
    parser.add_argument("-mp", "--finetuned_mdl_path", default='/Users/m144443/Documents/GitHub/mayo-w2v2/experiments/amr_subject_dedup_594_train_100_test_binarized_v20220620_5_adam_layerFinal_epoch1_wav2vec2-base-960h_mdl.pt', help='If running eval-only or extraction, you have the option to load a fine-tuned model by specifying the save path here. If passed a gs:// file, will download to local machine.')
    #GCS
    parser.add_argument('-b','--bucket_name', default='ml-e107-phi-shared-aif-us-p', help="google cloud storage bucket name")
    parser.add_argument('-p','--project_name', default='ml-mps-aif-afdgpet01-p-6827', help='google cloud platform project name')
    parser.add_argument('--cloud', default=False, type=bool, help="Specify whether to save everything to cloud")
    #output
    parser.add_argument("--dataset", default=None,type=str, help="When saving, the dataset arg is used to set file names. If you do not specify, it will assume the lowest directory from data_split_root")
    parser.add_argument("-o", "--exp_dir", default="./experiments/embeddings", help='specify LOCAL output directory')
    parser.add_argument('--cloud_dir', default='m144443/temp_out/w2v2_ft_weighted', type=str, help="if saving to the cloud, you can specify a specific place to save to in the CLOUD bucket")
    #Mode specific
    parser.add_argument("-m", "--mode", choices=['finetune','eval','extraction'], default='extraction')
    parser.add_argument("--weighted", type=bool, default=False, help="specify whether to learn a weighted sum of layers for classification")
    parser.add_argument("--layer", default=-1, type=int, help="specify which hidden state is being used. It can be between -1 and 12")
    parser.add_argument("--freeze", type=bool, default=True, help='specify whether to freeze the base model')
    parser.add_argument('--embedding_type', type=str, default='ft', help='specify whether embeddings should be extracted from classification head (ft) or base pretrained model (pt)', choices=['ft','pt'])
    #Audio transforms
    parser.add_argument("--resample_rate", default=16000,type=int, help='resample rate for audio files')
    parser.add_argument("--reduce", default=True, type=bool, help="Specify whether to reduce to monochannel")
    parser.add_argument("--clip_length", default=10.0, type=int, help="If truncating audio, specify clip length in seconds. 0 = no truncation")
    parser.add_argument("--trim", default=False, type=int, help="trim silence")
    #Model parameters
    parser.add_argument("-pm", "--pooling_mode", default="mean", help="specify method of pooling last hidden layer", choices=['mean','sum','max'])
    parser.add_argument("-bs", "--batch_size", type=int, default=8, help="specify batch size")
    parser.add_argument("-nw", "--num_workers", type=int, default=0, help="specify number of parallel jobs to run for data loader")
    parser.add_argument("-lr", "--learning_rate", type=float, default=0.0003, help="specify learning rate")
    parser.add_argument("-e", "--epochs", type=int, default=1, help="specify number of training epochs")
    parser.add_argument("--optim", type=str, default="adam", help="training optimizer", choices=["adam", "adamw"])
    parser.add_argument("--weight_decay", type=float, default=.0001, help='specify weight decay for adamw')
    parser.add_argument("--loss", type=str, default="BCE", help="the loss function for finetuning, depend on the task", choices=["MSE", "BCE"])
    parser.add_argument("--scheduler", type=str, default=None, help="specify lr scheduler", choices=["onecycle", None])
    parser.add_argument("--max_lr", type=float, default=0.01, help="specify max lr for lr scheduler")
    #classification head parameters
    parser.add_argument("--activation", type=str, default='relu', help="specify activation function to use for classification head")
    parser.add_argument("--final_dropout", type=float, default=0.3, help="specify dropout probability for final dropout layer in classification head")
    parser.add_argument("--layernorm", type=bool, default=False, help="specify whether to include the LayerNorm in classification head")
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
    if args.label_txt is None:
        assert args.mode == 'extraction', 'Must give a txt with target labels for training or evaluating.'
        args.target_labels = None
        args.n_class = 0
    else:
        with open(args.label_txt) as f:
            target_labels = f.readlines()
        target_labels = [l.strip() for l in target_labels]
        args.target_labels = target_labels

        args.n_class = len(target_labels)

        if args.n_class == 0:
            assert args.mode == 'extraction', 'Target labels must be given for training or evaluating. Txt file was empty.'

    # (5) check if output directory exists, SHOULD NOT BE A GS:// path
    if not os.path.exists(args.exp_dir):
        os.makedirs(args.exp_dir)

    # (6) check that clip length has been set
    if args.clip_length != 0:
        args.truncation = False #w2v2 feature extractor does not PAD, so you need to deal with the length and such here. 
    else:
        if args.clip_length == 0:
            try: 
                assert args.batch_size == 1, 'Not currently compatible with different length wav files unless batch size has been set to 1'
            except:
                args.batch_size = 1
    
    # (7) dump arguments
    args_path = "%s/args.pkl" % args.exp_dir
    with open(args_path, "wb") as f:
        pickle.dump(args, f)

    # (8) check if checkpoint is stored in gcs bucket or confirm it exists on local machine
    assert args.checkpoint is not None, 'Must give a model checkpoint for W2V2'
    args.checkpoint = gcs_model_exists(args.checkpoint, args.bucket_name, args.exp_dir, bucket, True)
    
    #(9) add bucket to args
    args.bucket = bucket

    # (10) run model
    print(args.mode)
    if args.mode == "finetune":
        finetune_w2v2(args)

    elif args.mode == 'eval':
        eval_only(args)
              
    elif args.mode == "extraction":
        df_embed = get_embeddings(args)
    
if __name__ == "__main__":
    main()