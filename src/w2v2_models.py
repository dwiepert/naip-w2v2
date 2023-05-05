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
from utilities.dataloader_utils import *


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

class Wav2Vec2ClassificationHead(nn.Module):
    """
        Head for wav2vec classification task. 
        Initialize classification head with input sizes
        Source: https://colab.research.google.com/github/m3hrdadfi/soxan/blob/main/notebooks/Eating_Sound_Collection_using_Wav2Vec2.ipynb#scrollTo=Fv62ShDsH5DZ
    """

    def __init__(self, hidden_size, final_dropout, num_labels):
        """
        Create a classification head with a dense layer, relu activation, a dropout layer, and final classification layer
        :param hiden_size: size of input to classification head - equivalent to the last hidden layer size in the Wav2Vec 2.0 model (int)
        :param final_dropout: specify amount of dropout
        :param num_labels: specify number of categories to classify
        """
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size) 
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(final_dropout)
        self.out_proj = nn.Linear(hidden_size, num_labels)

    def forward(self, features, **kwargs):
        """
        Run input (features) through the classifier
        :param features: input 
        :return x: classifier output
        """
        x = features
        x = self.dense(x)
        x = self.relu(x)
        x = self.out_proj(x)
        return x

class Wav2Vec2ForEmbeddingExtraction(nn.Module): 
    """ 
        Create Wav2Vec 2.0 model for speech classification 
        Initialize with a HuggingFace checkpoint (string path), pooling mode (mean, sum, max), and number of classes (num_labels)
    Source: https://colab.research.google.com/github/m3hrdadfi/soxan/blob/main/notebooks/Eating_Sound_Collection_using_Wav2Vec2.ipynb#scrollTo=Fv62ShDsH5DZ
    """
    def __init__(self, checkpoint, pooling_mode, mdl_path=None):
        """
        :param checkpoint: path to where original trained model checkpoint is saved (str)
        :param pooling_mode: specify which method of pooling from ['mean', 'sum', 'max'] (str)
        :param mdl_path: TODO
        """
        super(Wav2Vec2ForEmbeddingExtraction, self).__init__()
        self.pooling_mode = pooling_mode
        
        self.model = AutoModelForAudioClassification.from_pretrained(checkpoint,config=AutoConfig.from_pretrained(checkpoint, output_attentions=True,output_hidden_states=True))

        if mdl_path is not None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            sd = torch.load(mdl_path, map_location=device)
            self.model.load_state_dict(sd, strict=False)
        
    
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
        self.model.eval()
        outputs = self.model(
            input_values,
            attention_mask=attention_mask
        )
        hidden_states = outputs['hidden_states'][-1]
        hidden_states = self.merged_strategy(hidden_states, mode=self.pooling_mode)
        
        return hidden_states

    
class Wav2Vec2ForSpeechClassification(nn.Module):
    """ 
        Create Wav2Vec 2.0 model for speech classification 
        Initialize with a HuggingFace checkpoint (string path), pooling mode (mean, sum, max), and number of classes (num_labels)
    Source: https://colab.research.google.com/github/m3hrdadfi/soxan/blob/main/notebooks/Eating_Sound_Collection_using_Wav2Vec2.ipynb#scrollTo=Fv62ShDsH5DZ
    """
    def __init__(self, checkpoint, pooling_mode, num_labels):
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
        for param in self.model.parameters():
            param.requires_grad = False
        self.classifier = Wav2Vec2ClassificationHead(768,0.25,self.num_labels)
        

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