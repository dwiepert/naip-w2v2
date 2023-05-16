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
    def __init__(self, checkpoint, pooling_mode, num_labels, freeze=True, activation='relu', dropout=0.25, layernorm=False, rand_weights=False):
        """
        :param checkpoint: path to where model checkpoint is saved (str)
        :param pooling_mode: specify which method of pooling from ['mean', 'sum', 'max'] (str)
        :param num_labels: specify number of categories to classify
        :param freeze: specify whether to freeze the pretrained model parameters
        :param activation: activation function for classification head
        :param final_dropout: amount of dropout to use in classification head
        :param layernorm: include layer normalization in classification head
        :param rand_weight: specify whether to randomize weights
        """
        super(Wav2Vec2ForSpeechClassification, self).__init__()
        self.num_labels = num_labels
        self.pooling_mode = pooling_mode

        self.model = AutoModelForAudioClassification.from_pretrained(checkpoint,config=AutoConfig.from_pretrained(checkpoint, output_attentions=True,output_hidden_states=True))
       
        #freeze self.model here so none of the original pre-training is overwritten - only training classification head
        if freeze:
            for param in self.model.parameters():
                param.requires_grad = False

        #if weighting each layer for classification, set up a random vector of size # hidden states (13)
        if rand_weights:
            self.weightsum=nn.Parameter(torch.rand(13))
        else:
            self.weightsum=nn.Parameter(torch.ones(13)/13)

        self.classifier = ClassificationHead(768,self.num_labels,activation=activation, final_dropout=dropout,layernorm=layernorm)


        

    def merged_strategy(
            self,
            hidden_states,
            mode="mean",
            reduce_dim = 1
    ):
        """
        Set up pooling method to reduce hidden state dimension
        :param hidden_states: output from last hidden states layer
        :param mode: pooling method (str, default="mean")
        :param reduce_dim: dimension to merge on. It will be 1 in almost all cases except when calling mat_mul_weights
        :return outputs: hidden_state pooled to reduce dimension
        """
        if mode == "mean":
            outputs = torch.mean(hidden_states, dim=reduce_dim)
        elif mode == "sum":
            outputs = torch.sum(hidden_states, dim=reduce_dim)
        elif mode == "max":
            outputs = torch.max(hidden_states, dim=reduce_dim)[0]
        else:
            raise Exception(
                "The pooling method hasn't been defined! Your pooling mode must be one of these ['mean', 'sum', 'max']")

        return outputs

    def extract_embedding(self, x, 
                              embedding_type='ft',
                              layer=-1, 
                              attention_mask=None):
        """
        Run model
        :param input_values: input values to the model (batch_size, input_size)
        :param embedding_type: 'ft' or 'pt' to indicate whether to extract from classification head or last hidden state
        :param layer: hidden layer to take out and do results for - must be between 0-12
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
            hidden_states = outputs['hidden_states'][layer]
            e = self.merged_strategy(hidden_states, mode=self.pooling_mode)

        else:
            raise ValueError('Embedding type must be finetune (ft) or pretrain (pt)')
        
        return e
    
    def mat_mul_weights(self, weightsum, hidden_states):
        """
        Pool each hiddenstate and set up weighted sum
        :param weightsum:
        :param hidden_states: hidden w2v2 states of size (batch, hidden_state_dim, 768)
        """
        #one layer hidden state size = batch size x 499 x 768
        hidden = self.merged_strategy(torch.stack(hidden_states), mode=self.pooling_mode, reduce_dim=2)
        #after pooling, shape is = # hidden layers x batch size x 768
        hidden = torch.permute(hidden, (1,0,2)) #permute the shape so it is batch size x # hidden layers x 768
        w_sum = torch.reshape(weightsum,(1,1,13)) #reshape weight sum to have the right number of dims
        w_sum=w_sum.repeat(hidden.shape[0],1,1) #repeat for each sample in batch to allow for batch matrix product, results in size #batch size x 1 x 13
        weighted_sum=torch.bmm(w_sum, hidden) #output: batch size x 1 x 768

        return torch.squeeze(weighted_sum, dim=1) #need to squeeze at output going into classifier should be batch size x 768 


    def weighted_forward(self,
                     input_values,
                     attention_mask=None):
        """
        Run model with weighted layers
        :param input_values: input values to the model (batch_size, input_size)
        :param attention_mask: give attention mask to model (default=None)
        :return logits: classifier output (batch_size, num_labels)
        """
        outputs = self.model(
            input_values,
            attention_mask=attention_mask
        )
        #one layer hidden state size = batch size x 499 x 768
        hidden_states = self.mat_mul_weights(self.weightsum, outputs['hidden_states'])
        #comes out of mat_mul_weights as batch_size x 768
        logits = self.classifier(hidden_states)

        return logits

        
    def forward(
            self,
            input_values,
            layer=-1,
            attention_mask=None,
    ):
        """
        Run model
        :param input_values: input values to the model (batch_size, input_size)
        :param layer: hidden layer to take out and do results for - must be between 0-12
        :param attention_mask: give attention mask to model (default=None)
        :return logits: classifier output (batch_size, num_labels)
        """
        assert layer >= -1 and layer < 13, 'layer must be 0-12 or -1'
        outputs = self.model(
            input_values,
            attention_mask=attention_mask
        )
        hidden_states = outputs['hidden_states'][layer]
        hidden_states = self.merged_strategy(hidden_states, mode=self.pooling_mode)
        
        logits = self.classifier(hidden_states)   
        return logits