'''
W2V2 model classes

Base w2v2 models from HuggingFace.

Last modified: 05/2023
Author: Daniela Wiepert
Email: wiepert.daniela@mayo.edu
File: w2v2_models.py
'''

#built-in
import numpy as np

#third-party
import torch
import torch.nn as nn

from transformers import AutoFeatureExtractor
from transformers import AutoModelForAudioClassification, AutoConfig

#local
from utilities import *


class Wav2VecFeatureExtractor(object):
    '''
        Wav2Vec Feature Extractor Class for transforms
        Initialize with a feature extractor from transformers, optionally set max_length and truncation
        Call takes raw waveform and runs it through the feature extractor 
    '''
    
    def __init__(self, checkpoint, max_length=10.0, truncation=True):
        """
        :param checkpoint: checkpoint to w2v2 model from huggingFace
        :param max_length: specify max length of a sample in s (float, default = 10.0 s)
        :param truncation: specify whether to truncate based on max length (boolean, default = True)
        """
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(checkpoint)
        self.max_length = max_length
        self.truncation = truncation
        
        
    def __call__(self, sample):
        """
        Take a sample and run through feature extractor
        :param sample: dictionary object storing input value, associated label, attention mask, and other information about a given sample
        :return updated sample
        """
        x=sample['waveform'].numpy()
        x=np.squeeze(x)
        sr = sample['sample_rate']
        ml = int(self.max_length*sr)
        
        feature = self.feature_extractor(x, sampling_rate=sr, max_length=ml, truncation=self.truncation, return_tensors='pt', return_attention_mask=True)
        
        sample['waveform'] = feature['input_values']
        sample['attention_mask'] = feature['attention_mask']
        
        return sample 


class Wav2Vec2ForSpeechClassification(nn.Module):
    """ 
        Create Wav2Vec 2.0 model for speech classification 
        Initialize with a HuggingFace checkpoint (string path), pooling mode (mean, sum, max), and number of classes (num_labels)
    Source: https://colab.research.google.com/github/m3hrdadfi/soxan/blob/main/notebooks/Eating_Sound_Collection_using_Wav2Vec2.ipynb#scrollTo=Fv62ShDsH5DZ
    """
    def __init__(self, checkpoint, pooling_mode, num_labels, freeze=True, activation='relu', dropout=0.25, layernorm=False):
        """
        :param checkpoint: path to where model checkpoint is saved (str)
        :param pooling_mode: specify which method of pooling from ['mean', 'sum', 'max'] (str)
        :param num_labels: specify number of categories to classify
        """
        super(Wav2Vec2ForSpeechClassification, self).__init__()
        self.num_labels = num_labels
        self.pooling_mode = pooling_mode

        self.model = AutoModelForAudioClassification.from_pretrained(checkpoint,config=AutoConfig.from_pretrained(checkpoint, output_attentions=True,output_hidden_states=True))
       
        #freeze self.model here so none of the original pre-training is overwritten - only training classification head
        if freeze:
            for param in self.model.parameters():
                param.requires_grad = False

        self.classifier = ClassificationHead(768,self.num_labels,activation=activation, final_dropout=dropout,layernorm=layernorm)
        

    def merged_strategy(
            self,
            hidden_states,
            mode="mean"
    ):
        """
        Set up pooling method to reduce hidden state dimension
        :param hidden_states: output from last hidden states layer
        :param mode: pooling method (str, default="mean")
        :return outputs: hidden_state pooled to reduce dimension
        """
        if mode == "mean":
            outputs = torch.mean(hidden_states, dim=1)
        elif mode == "sum":
            outputs = torch.sum(hidden_states, dim=1)
        elif mode == "max":
            outputs = torch.max(hidden_states, dim=1)[0]
        else:
            raise Exception(
                "The pooling method hasn't been defined! Your pooling mode must be one of these ['mean', 'sum', 'max']")

        return outputs

    def extract_embeddings(self, x, 
                              embedding_type='ft', 
                              attention_mask=None):
        """
        Run model
        :param input_values: input values to the model (batch_size, input_size)
        :param attention_mask: give attention mask to model (default=None)
        :return logits: classifier output (batch_size, num_labels)
        """
        activation = {}
        def _get_activation(name):
            def _hook(model, input, output):
                activation[name] = output.detach()
            return _hook
        
        if embedding_type == 'ft':
            self.classifier.head.dense.register_forward_hook(_get_activation('embeddings'))
            
            logits = self.forward(x)
            e = activation['embeddings']

        elif embedding_type == 'pt':
            outputs = self.model(x, attention_mask=attention_mask)
            hidden_states = outputs['hidden_states'][-1]
            e = self.merged_strategy(hidden_states, mode=self.pooling_mode)

        else:
            raise ValueError('Embedding type must be finetune (ft) or pretrain (pt)')
        
        return e
    
    
    def forward(
            self,
            input_values,
            attention_mask=None,
    ):
        """
        Run model
        :param input_values: input values to the model (batch_size, input_size)
        :param attention_mask: give attention mask to model (default=None)
        :return logits: classifier output (batch_size, num_labels)
        """
        outputs = self.model(
            input_values,
            attention_mask=attention_mask
        )
        hidden_states = outputs['hidden_states'][-1]
        hidden_states = self.merged_strategy(hidden_states, mode=self.pooling_mode)
        
        logits = self.classifier(hidden_states)   
        return logits