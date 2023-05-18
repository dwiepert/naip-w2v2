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
    def __init__(self, checkpoint, num_labels=5, pooling_mode='mean', 
                 freeze=True, activation='relu', final_dropout=0.25, layernorm=False, 
                 weighted=False, layer=-1):
        """
        :param checkpoint: path to where model checkpoint is saved (str)
        :param num_labels: specify number of categories to classify
        :param pooling_mode: specify which method of pooling from ['mean', 'sum', 'max'] (str)
        :param freeze: specify whether to freeze the pretrained model parameters
        :param activation: activation function for classification head
        :param final_dropout: amount of dropout to use in classification head
        :param layernorm: include layer normalization in classification head
        :param weighted: specify which mode to run as the forward function (False: forward for a single hidden layer, True: weighted layer sum)
        :param layer: layer for single hidden layer extraction
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
        self.n_states, self.embedding_dim = self._get_shape()
    
        assert layer >= -1 and layer < self.n_states, f'invalid layer" {layer}. Layer must either be -1 for final layer, or a number between 0 and {self.n_states}'
        self.layer = layer

        self.weighted=weighted
        if self.weighted:
            self.weightsum=nn.Parameter(torch.ones(self.n_states)/self.n_states)
        else:
            self.weightsum=torch.ones(self.n_states)/self.n_states

        self.classifier = ClassificationHead(self.embedding_dim,self.num_labels,activation=activation, final_dropout=final_dropout,layernorm=layernorm)
        
    def _get_shape(self):
        """
        Get the number of hidden states and the original embedding dim from the checkpoint model
        """
        test_input = torch.randn(1,160000)
        outputs = self.model(test_input)
        h = outputs['hidden_states']
        n_states = len(h)
        embedding_dim = h[-1].shape[2]
        return n_states, embedding_dim


    def _merged_strategy(
            self,
            hidden_states,
            mode="mean",
            reduce_dim=1
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
    
    def _mat_mul_weights(self, 
                         hidden_states):
        """
        Matrix multiplication of pooled hidden states (# states, batch size, embedding dim) and weighted sum parameter
        :param hidden_states: hidden w2v2 states, pooled across hidden size, now of size (#states,batch size, embedding dim)
        :return: mat mul of weighted sum, squeezed to be size (batch size, embedding dim)
        """
        
        #after pooling, shape is = # hidden layers x batch size x 768
        hidden = torch.permute(hidden_states, (1,0,2)) #permute the shape so it is batch size x # hidden layers x 768
        w_sum = torch.reshape(self.weightsum,(1,1,13)) #reshape weight sum to have the right number of dims
        w_sum=w_sum.repeat(hidden.shape[0],1,1) #repeat for each sample in batch to allow for batch matrix product, results in size #batch size x 1 x 13
        weighted_sum=torch.bmm(w_sum, hidden) #output: batch size x 1 x 768

        return torch.squeeze(weighted_sum, dim=1) #need to squeeze at output going into classifier should be batch size x 768 

    def _pool(self, 
            hidden_states,
            pooling_mode='mean',
            weighted=False,
            layer=-1):
        """
        Pool hidden states across the hidden dimension. 
        :param hidden_states: list of hidden w2v2 states, with each state of size (batch, hidden_state_dim, embedding dim)
        :param pooling_mode: pooling method
        :param weighted: specify which mode to run as the forward function (False: forward for a single hidden layer, True: weighted layer sum)
        :param layer: layer for single hidden layer extraction
        :param output: pooled hidden state, now of size (batch, embedding dim)
        """
        if weighted:
            hidden_states = torch.stack(hidden_states)
            hidden_states = self._merged_strategy(hidden_states, mode=pooling_mode, reduce_dim=2)
            output = self._mat_mul_weights(hidden_states)
        
        else:
            hidden_states = hidden_states[layer]
            output = self._merged_strategy(hidden_states, mode=pooling_mode, reduce_dim=1)
        return output

    def extract_embedding(self, 
                          x, 
                          embedding_type='ft',
                          layer = None,
                          pooling_mode=None,
                          attention_mask=None):
        """
        Extract an embedding from various parts of the model
        :param x: waveform input (batch size, input size)
        :param embedding_type: 'ft', 'pt', or 'wt', to indicate whether to extract from classification head (ft), hidden state (pt), or weighted sum mat mul (wt)
        :param layer: int indicating which hidden state layer to use.
        :param pooling_mode: pooling method
        :param attention_mask: give attention mask to model (default=None)
        :return e: embeddings for a batch (batch_size, embedding dim)
        """
        ## EMBEDDING 'ft': extract from finetuned classification head
        if embedding_type == 'ft':

            #register a forward hook to grab the output of the first classification layer (called 'dense')
            activation = {}
            def _get_activation(name):
                def _hook(model, input, output):
                    activation[name] = output.detach()
                return _hook
            
            self.classifier.head.dense.register_forward_hook(_get_activation('embeddings'))
            
            logits = self.forward(x) #run the forward function using model parameters for the task (so that it's inline with the finetuning)
            e = activation['embeddings'] #get embedding
        
        ## EMBEDDING 'pt': extract from a hidden state, 'wt': extract after matmul with layer weights
        elif embedding_type == 'pt' or embedding_type == 'wt':
            outputs = self.model(x, attention_mask=attention_mask)

            if embedding_type == 'pt': # if 'pt', self.weighted must always be false
                weighted=False
            else: #else the model must have been trained for weighted sum, so original self.weighted must be True
                try:
                    weighted=self.weighted
                    assert weighted
                except:
                    raise ValueError('The model must be trained for weightsum')

            if layer is None:
                layer = self.layer

            if pooling_mode is None:
                pooling_mode = self.pooling_mode

            e = self._pool(outputs['hidden_states'], pooling_mode, weighted, layer) #pool across the specified layer

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

        hidden_states = outputs['hidden_states']
        x = self._pool(hidden_states, self.pooling_mode, self.weighted, self.layer)

        logits = self.classifier(x)   
        return logits