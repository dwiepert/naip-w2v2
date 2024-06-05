# -*- coding: utf-8 -*-
'''
Dataset functions for W2V2

Last modified: 06/2023
Author: Daniela Wiepert
Email: wiepert.daniela@mayo.edu
File: dataloader.py
'''

#IMPORTS
#third party
import numpy as np
import pandas as pd
import torch
import torchvision
import torch.nn.functional
from torch.utils.data import Dataset

#local
from utilities import *
from models import *

        
class W2V2Dataset(Dataset):
    def __init__(self, annotations_df, target_labels, 
                 audio_conf, prefix, bucket=None, librosa=False):
        '''
        Dataset that manages audio recordings. 

        :param annotations_df: dataframe containing uid of audio file along with labels
        :type annotations_df: pd.DataFrame
        :param target_labels: list containing the specific target labels to select, can give None (assuming there are no target labels in the df) or an empty list or np array.
        :type target_labels: List[Str]
        :param audio_conf: dictionary containing all information for transforms (audio configuration dict)
        :type audio_conf: dict
        :param prefix: location of files to download - can be either prefix in gcs bucket or input directory
        :type prefix: Str
        :param bucket: Google cloud storage bucket, default=None
        :type bucket: GCS bucket
        :param librosa: boolean indicating loading from librosa as compared to torchaudio
        :type librosa: boolean
        '''

        #set instance variables
        self.annotations_df = annotations_df
        self.target_labels = target_labels
        if self.target_labels is None:
            self.target_labels = np.array([])
        self.prefix = prefix
        self.bucket = bucket
        self.lib = librosa #set up using librosa vs. torchaudio for loading

        #AUDIO CONFIGURATION PARAMETERS
        self.audio_conf = audio_conf
        self.checkpoint = self.audio_conf.get('checkpoint')
        ### AUDIO TRANSFORMATIONS
        self.resample_rate = self.audio_conf.get('resample_rate') #resample if resample rate != 0 and if resample rate != sample rate
        self.reduce = self.audio_conf.get('reduce') #reduce to monochannel if True
        self.clip_length = self.audio_conf.get('clip_length') #truncate clip to specified length if != 0
        #mixup
        self.mixup = self.audio_conf.get('mixup') #mixup if mixup != 0
  
        self.label_num = len(self.target_labels)
        
        self.audio_transform, self.feature_tfm = self._getaudiotransform() #get audio transforms
        
    
    def _getaudiotransform(self):
        '''
        Use audio configuration parameters to initialize classes for audio transformation. 
        Outputs two tranform variables, one for regular audio transformation and one for 
        augmentations using albumentations

        These transformations will always load the audio. 
        :outparam audio_transform: standard transforms
        '''
        waveform_loader = UidToWaveform(prefix = self.prefix, bucket=self.bucket, lib=self.lib)
        transform_list = [waveform_loader]
        if self.reduce:
            channel_sum = lambda w: torch.sum(w, axis = 0).unsqueeze(0)
            mono_tfm = ToMonophonic(reduce_fn = channel_sum)
            transform_list.append(mono_tfm)
        if self.resample_rate != 0: #16000
            downsample_tfm = Resample(self.resample_rate)
            transform_list.append(downsample_tfm)
        if self.clip_length != 0: #160000
            truncate_tfm = Truncate(length = self.clip_length)
            transform_list.append(truncate_tfm)

        tensor_tfm = ToTensor()
        transform_list.append(tensor_tfm)
        transform = torchvision.transforms.Compose(transform_list)
        #transform_list.append(feature_tfm)

        feature_tfm = Wav2VecFeatureExtractor(self.checkpoint, self.clip_length)

        return transform, feature_tfm

    def __getitem__(self, idx):
        '''
        Given an index, load and run transformations then return the sample dictionary

        Will run transformations in this order:
        Standard audio transformations (load audio -> reduce channels -> resample -> clip -> subtract mean) - also convert labels to tensor
        Albumentation transformations (Time shift -> speed tune -> add gauss noise -> pitch shift -> alter gain -> stretch audio)
        Spectrogram transformations (convert to spectrogram -> frequency mask -> time mask -> normalize -> add noise)

        The resulting sample dictionary contains the following info
        'uid': audio identifier
        'waveform': audio (n_channels, n_frames)
        'fbank': spectrogram (target_length, frequency_bins)
        'sample_rate': current sample rate
        'targets': labels for current file as tensor

        '''
    
        #If not doing mix-up
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        uid = self.annotations_df.index[idx] #get uid to load

        targets = self.annotations_df[self.target_labels].iloc[idx].values #get target labels for given uid
        
        sample = {
            'uid' : uid,
            'targets' : targets
        }
        
        sample = self.audio_transform(sample) #load and perform standard transformation
        
        #TODO: initialize mixup
        mix = Mixup()
        if self.mixup == 0:
            sample= mix(sample, None)

        elif random.random() < self.mixup: 
            mix_sample_idx = random.randint(0, len(self.annotations_df)-1)
            mix_uid = self.annotations_df.index[mix_sample_idx]
            mix_targets = self.annotations_df[self.target_labels].iloc[mix_sample_idx].values
        
            sample2 = {
                'uid': mix_uid,
                'targets': mix_targets
            }
            sample2 = self.audio_transform(sample2) #load and perform standard transformation

            sample = mix(sample, sample2)
        
        else:
            sample = mix(sample, None)

        sample = self.feature_tfm(sample)
        
        return sample
    

    def __len__(self):
        return len(self.annotations_df)
    