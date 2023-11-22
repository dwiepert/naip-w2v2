'''
Sample audio Dataset format (WaveformDataset) + transformation classes.
Classification head

Transforms from SSAST added (https://github.com/YuanGongND/ssast/tree/main/src/run.py)

Author(s): Neurology AI Program (NAIP) at Mayo Clinic
Last modified: 11/2023
'''
#IMPORTS
#built-in
import io
import json
import random

from collections import OrderedDict

#third party
import albumentations
import cv2
import librosa
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchaudio
import torch.nn.functional
from torch.utils.data import Dataset

from albumentations.core.transforms_interface import DualTransform, BasicTransform
from google.cloud import storage
from torch.utils.data import Dataset


#local
from albumentations.core.transforms_interface import DualTransform, BasicTransform

class ClassificationHead(nn.Module):
    """
        Head for classification task. 
        Initialize classification head with input sizes
        Source: https://colab.research.google.com/github/m3hrdadfi/soxan/blob/main/notebooks/Eating_Sound_Collection_using_Wav2Vec2.ipynb#scrollTo=Fv62ShDsH5DZ
    """

    def __init__(self, input_size=768, bottleneck=150, output_size=2, activation='relu',final_dropout=0.2, layernorm=False ):
        """
        Create a classification head with a dense layer, relu activation, a dropout layer, and final classification layer
        :param input_size: size of input to classification head 
        :param bottleneck: size to reduce to in intial dense layer (if you don't want to reduce size, set bottleneck=input size)
        :param output_size: number of categories for classification output
        :param num_labels: specify number of categories to classify
        :param activation: activation function for classification head
        :param final_dropout: amount of dropout to use in classification head
        :param layernorm: include layer normalization in classification head
        """
        super().__init__()
        self.input_size = input_size
        self.bottleneck= bottleneck
        self.output_size = output_size
        self.activation = activation
        self.layernorm = layernorm
        self.final_dropout = final_dropout

        classifier = []
        key = []
        classifier.append(nn.Linear(self.input_size, self.bottleneck))
        key.append('dense')
        if self.layernorm:
            classifier.append(nn.LayerNorm(self.bottleneck))
            key.append('norm')
        if self.activation == 'relu':
            classifier.append(nn.ReLU())
            key.append('relu')
        classifier.append(nn.Dropout(self.final_dropout))
        key.append('dropout')
        classifier.append(nn.Linear(self.bottleneck, self.output_size))
        key.append('outproj')

        self.classifier=classifier
        self.key=key

        seq = []
        for i in range(len(classifier)):
            seq.append((key[i],classifier[i]))
        
        self.head = nn.Sequential(OrderedDict(seq))

    def forward(self, x, **kwargs):
        """
        Run input (features) through the classifier
        :param features: input 
        :return x: classifier output
        """
        return self.head(x)

#overrid collate function to stack images differently
def collate_fn(batch):
    '''
    This collate function is meant for use when initializing a dataloader - pass for the collate_fn argument.
    Only use this version if you are wanting to maintain the waveform information of a batch that has different length tensors rather than
    padding the waveform. Otherwise, use the default collate_fn.
    This function also only accounts for information maintained with the transformations laid out in this script. If more information is added
    to the samples, it needs to be adjusted.
    '''
    if 'waveform' in batch[0]:
        waveform = [item['waveform'] for item in batch]
    else:
        waveform = None
    
    if 'fbank' in batch[0]:
        fbank = torch.stack([item['fbank'] for item in batch])
    else:
        fbank = None
    uid = [item['uid'] for item in batch]
    sr = [item['sample_rate'] for item in batch]
    targets = torch.stack([item['targets'] for item in batch])
    return {'uid':uid, 'waveform':waveform, 'fbank':fbank, 'targets':targets, 'sample_rate':sr}


#simple dataset
class WaveformDataset(Dataset):
    '''
    Simple audio dataset
    '''
    
    def __init__(self, annotations_df, target_labels, transform):
        '''
        Initialize dataset with dataframe, target labels, and list of transforms

        '''
        
        self.annotations_df = annotations_df
        self.transform = transform
        self.target_labels = target_labels
        
    def __len__(self):
        
        return len(self.annotations_df)
    
    def __getitem__(self, idx):
        '''
        Run transformation
        '''
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        uid = self.annotations_df.index[idx]
        targets = self.annotations_df[self.target_labels].iloc[idx].values
        
        sample = {
            'uid' : uid,
            'targets' : targets
        }
        
        return self.transform(sample)
    

### AUDIO TRANSFORMATIONS
#helper functions
def load_waveform_from_gcs(bucket, gcs_prefix, uid, extension = None, lib=False):
    '''
    load audio from google cloud storage
    bucket: gcs bucket object
    gcs_prefix: prefix leading to object in gcs bucket
    uid: audio identifier
    extension: audio type (default, None)
    lib: boolean indicating to load with librosa rather than torchaudio

    Returns loaded audio waveform as tensor + metadata
    '''
    gcs_metadata_path = f'{gcs_prefix}/{uid}/metadata.json'
    
    metadata_blob = bucket.blob(gcs_metadata_path)
    
    metadata = json.loads(metadata_blob.download_as_string())
    
    if extension is None:
        if metadata['encoding'] == 'MP3':
            extension = 'mp3'
        else:
            extension = 'wav'
        
    gcs_waveform_path = f'{gcs_prefix}/{uid}/waveform.{extension}'
    
    blob = bucket.blob(gcs_waveform_path)
    wave_string = blob.download_as_string()
    wave_bytes = io.BytesIO(wave_string)
    if not lib:
        waveform, _ = torchaudio.load(wave_bytes, format = extension)
    else:
        waveform, _ = librosa.load(wave_bytes, mono=False, sr=None)
        waveform = torch.from_numpy(waveform)
        if len(waveform.shape) == 1:
           waveform = waveform.unsqueeze(0)
    
    return waveform, metadata

def load_waveform_local(input_dir, uid, extension = None, lib=False):
    '''
    input_directory: directory where data is stored locally
    uid: audio identifier
    extension: audio type (default, None)
    lib: boolean indicating to load with librosa rather than torchaudio

    Returns loaded audio waveform as tensor + metadata
    '''
    
    metadata_path = f'{input_dir}/{uid}/metadata.json'
    with open(metadata_path) as f:
        metadata = json.load(f)
    
    if extension is None:
        if metadata['encoding'] == 'MP3':
            extension = 'mp3'
        else:
            extension = 'wav'
        
    waveform_path = f'{input_dir}/{uid}/waveform.{extension}'
    
    if not lib:
        waveform, _ = torchaudio.load(waveform_path, format = extension)
    else:
        waveform, _ = librosa.load(waveform_path, mono=False, sr=None)
        waveform = torch.from_numpy(waveform)
        if len(waveform.shape) == 1:
           waveform = waveform.unsqueeze(0)
    
    return waveform, metadata

def vectorize_scores(records, targets, map_fn):
    
    vals = {t : 0 for t in targets}
    for r in records:
        if r['name'] in vals:
            vals[r['name']] = map_fn(float(r['score']))
            
    return pd.Series(vals)

#classes
class UidToWaveform(object):
    '''
    Take a UID, find & load the data and metadata, add waveform and sample rate to sample
    '''
    
    def __init__(self, prefix, bucket=None, extension=None, lib=False):
        
        self.bucket = bucket
        self.prefix = prefix #either gcs_prefix or input_dir prefix
        self.cache = {}
        self.extension = None
        self.lib = lib
        
    def __call__(self, sample):
        
        uid, targets = sample['uid'], sample['targets']
        
        if uid not in self.cache:
            if self.bucket is not None:
                #load from google cloud storage
                self.cache[uid] = load_waveform_from_gcs(self.bucket, self.prefix, uid, self.extension, self.lib)
            else:
                 #load local
                 self.cache[uid] = load_waveform_local(self.prefix, uid, self.extension, self.lib)

            
        waveform, metadata = self.cache[uid]
        
        sample['waveform'] = waveform
        sample['sample_rate'] = int(metadata['sample_rate_hz'])
         
        return sample
    

class Truncate(object):
    '''
    Cut audio to specified length with optional opset
    '''
    def __init__(self, length, offset = 0):
        
        self.length = length
        self.offset = offset
        
    def __call__(self, sample):
        
        waveform = sample['waveform']
        sr = sample['sample_rate']
        frames = int(self.length*sr)

        waveform_offset = waveform[:, self.offset:]
        n_samples_remaining = waveform_offset.shape[1]
        
        if n_samples_remaining >= frames:
            waveform_trunc = waveform_offset[:, :frames]
        else:
            n_channels = waveform_offset.shape[0]
            n_pad = frames - n_samples_remaining
            channel_means = waveform_offset.mean(axis = 1).unsqueeze(1)
            waveform_trunc = torch.cat([waveform_offset, torch.ones([n_channels, n_pad])*channel_means], dim = 1)
            
        sample['waveform'] = waveform_trunc
        
        return sample
    
    
class ToMonophonic(object):
    '''
    Convert to monochannel with a reduce function (can alter based on how waveform is loaded)
    '''
    def __init__(self, reduce_fn):
        
        self.reduce_fn = reduce_fn
        
    def __call__(self, sample):
        
        waveform = sample['waveform']
        #print(waveform.shape)
        waveform_mono = self.reduce_fn(waveform)
        #print(waveform_mono)
        #print(waveform_mono.shape)
        
        if waveform_mono.shape != torch.Size([1, waveform.shape[1]]):
            raise ValueError(f'Result of reduce_fn wrong shape, expected [1, {waveform.shape[1]}], got [{waveform_mono.shape[0], waveform_mono.shape[1]}]')
            
        sample['waveform'] = waveform_mono
            
        return sample

    
class ToTensor(object):
    '''
    Convert labels to a tensor rather than ndarray
    '''
    def __call__(self, sample):
        
        targets = sample['targets']
        sample['targets'] = torch.from_numpy(sample['targets']).type(torch.float32)
        
        return sample
    
    
class Resample(object):
    '''
    Resample a waveform
    '''
    def __init__(self, resample_rate):
        
        self.resample_rate = resample_rate
        
    def __call__(self, sample):
        
        """
        Resample pytorch audio waveform

        :param waveform: torch audio
        :type waveform: tensor
        :param sample_rate: current audio sample rate
        :type sample_rate: int
        :param resample_rate: resampling rate
        :type resample_rate: int
        :return: resampled waveform
        """
        
        waveform, sample_rate = sample['waveform'], sample['sample_rate']
        if sample_rate != self.resample_rate:
            transformed = torchaudio.transforms.Resample(sample_rate, self.resample_rate)(waveform)
            sample['waveform'] = transformed
            sample['sample_rate'] = self.resample_rate
        
        return sample
    
    
    
class MelSpectrogram(object):
    '''
    Spectrogram conversion V1
    '''
    
    def __init__(self, n_fft=16000, n_mels=128):
        
        self.n_fft = n_fft
        self.n_mels = n_mels
        
    def __call__(self, sample):
        
        """
        Convert pytorch audio waveform to mel spectrogram

        :param waveform: torch audio
        :type waveform: tensor object
        :param n_fft: number of (output) frequency bins
        :type n_fft: int
        :param n_mels: number of mels
        :type n_fft: int
        :return: mel spectrogram
        """
        
        waveform, sample_rate = sample['waveform'], sample['sample_rate']
        
        
        mel_spectrogram = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=self.n_fft,
            norm='slaney',
            n_mels=self.n_mels,
        )

        sample['mel_spectrogram'] = mel_spectrogram(waveform)
        
        return sample


class TrimSilence(object):
    '''
    Trim silence in an audio file
    '''
    
    def __init__(self, trigger_level=4.0, trigger_time=0.25, search_time=1.0, allowed_gap=0.25, boot_time=0.35, noise_up_time=0.1, noise_down_time=0.01, noise_reduction_amount=1.35):
        """
        There are a few other parameters in the silence detection algorithm that are not included. Can see all options         at: https://curso-r.github.io/torchaudio/reference/functional_vad.html
        
        official torch source: https://pytorch.org/audio/stable/functional.html 
        
        :param trigger_level: level used to trigger activity detection. Change depending on the noise level, signal level, and other characteristics of the input audio. Smaller will catch softer voices, so if too much is getting removed, decrease. (Default: 4.0)
        :param trigger_time: time constant (in seconds) used to help ignore short bursts of sound. (Default: 0.25)
        :param search_time: amount of audio (in seconds) to search for quieter/shorter bursts of audio to include prior to the detected trigger point. (Default: 1.0)
        :param allowed_gap: allowed gap (in seconds) between quiteter/shorter bursts of audio to include prior to the detected trigger point. (Default: 0.25)
        :param boot_time: algorithm (internally) uses adaptive noise estimation/reduction in order to detect the start of the wanted audio. This option sets the time for the initial noise estimate. (Default: 0.35)
        :param noise_up_time: Time constant used by the adaptive noise estimator for when the noise level is increasing. (Default: 0.1)
        :param noise_down_time: Time constant used by the adaptive noise estimator for when the noise level is decreasing. (Default: 0.01)
        :param noise_reduction_amount: Amount of noise reduction to use in the detection algorithm (e.g. 0, 0.5, ...). (Default: 1.35)
        
        There are a few other parameters in the silence detection algorithm that I did not include 
        """

        self.trigger_level = trigger_level #level used to trigger activity detection. Change depending on the noise level, signal level, and other characteristics of the input audio. Smaller will catch softer voices, so if too much is getting removed, decrease
        self.trigger_time = trigger_time #0.25
        self.search_time = search_time #1.0
        self.allowed_gap = allowed_gap #0.25
        self.boot_time = boot_time #0.35
        self.noise_up_time = noise_up_time #0.1
        self.noise_down_time = noise_down_time #0.01
        self.noise_reduction_amount = noise_reduction_amount #1.35
    
    def __call__(self, sample):
        
        waveform, sample_rate = sample['waveform'], sample['sample_rate']
        #print(waveform.shape)
        
        if isinstance(waveform, np.ndarray):
            waveform = torch.from_numpy(waveform)
        
        transformed = torchaudio.functional.vad(waveform, sample_rate, self.trigger_level, self.trigger_time, self.search_time, self.allowed_gap,0.0,self.boot_time, self.noise_up_time, self.noise_down_time, self.noise_reduction_amount)

        #if self.trim_both:
        reversed1 = torch.flip(transformed, [1])
        transformed_reversed = torchaudio.functional.vad(reversed1, sample_rate, self.trigger_level, self.trigger_time, self.search_time, self.allowed_gap,self.boot_time, self.noise_up_time, self.noise_down_time, self.noise_reduction_amount)
        transformed = torch.flip(transformed_reversed, [1])
            
        sample['waveform'] = transformed
        
        return sample

class WaveMean(object):
    '''
    Subtract the mean from the waveform
    '''
    def __call__(self, sample):
        
        waveform = sample['waveform']
        waveform = waveform - waveform.mean()
        sample['waveform'] = waveform
        
        return sample

class Mixup(object):
    '''
    Implement mixup of two files
    '''

    def __call__(self, sample, sample2=None):
        if sample2 is None:
            waveform = sample['waveform']
            waveform = waveform - waveform.mean()
        else:
            waveform1 = sample['waveform']
            waveform2 = sample2['waveform']
            waveform1 = waveform1 - waveform1.mean()
            waveform2 = waveform2 - waveform2.mean()

            if waveform1.shape[1] != waveform2.shape[1]:
                if waveform1.shape[1] > waveform2.shape[1]:
                    temp_wav = torch.zeros(1, waveform1.shape[1])
                    temp_wav[0,0:waveform2.shape[1]] = waveform2
                    waveform2 = temp_wav
                else:
                    waveform2 = waveform2[0,0:waveform1.shape[1]]

            #sample lambda from beta distribution
            mix_lambda = np.random.beta(10,10)

            mix_waveform = mix_lambda*waveform1 + (1 - mix_lambda) * waveform2
            waveform = mix_waveform - mix_waveform.mean()   

            targets1 = sample['targets']
            targets2 = sample2['targets']
            targets = mix_lambda*targets1 + (1-mix_lambda)*targets2
            sample['targets'] = targets
            #TODO: what is happening here

        sample['waveform'] = waveform

        return sample

### FROM DATALOADER_GCS
class AudioTransform(BasicTransform):
    """ Transform for audio task. This is the main class where we override the targets and update params function for our need"""

    @property
    def targets(self):
        return {"data": self.apply}
    
    def update_params(self, params, **kwargs):
        if hasattr(self, "interpolation"):
            params["interpolation"] = self.interpolation
        if hasattr(self, "fill_value"):
            params["fill_value"] = self.fill_value
        return params

class TimeShifting(AudioTransform):
    """ Do time shifting of audio """
    def __init__(self, always_apply=False, p=0.5):
        super(TimeShifting, self).__init__(always_apply, p)
        
    def apply(self,sample,**params):
        '''
        data : ndarray of audio timeseries
        '''  
        data = sample['waveform']      
        start_ = int(np.random.uniform(-80000,80000))
        if start_ >= 0:
            audio_time_shift = np.r_[data[start_:], np.random.uniform(-0.001,0.001, start_)]
        else:
            audio_time_shift = np.r_[np.random.uniform(-0.001,0.001, -start_), data[:start_]]
        
        sample['waveform'] = torch.from_numpy(audio_time_shift)
        return sample
    
class SpeedTuning(AudioTransform):
    """ Do speed Tuning of audio """
    def __init__(self, always_apply=False, p=0.5,speed_rate = None):
        '''
        Give Rate between (0.5,1.5) for best results
        '''
        super(SpeedTuning, self).__init__(always_apply, p)
        
        if speed_rate:
            self.speed_rate = speed_rate
        else:
            self.speed_rate = np.random.uniform(0.6,1.3)
        
    def apply(self,sample,**params):
        '''
        data : ndarray of audio timeseries
        '''        
        data = sample['waveform']
        audio_speed_tune = cv2.resize(data, (1, int(len(data) * self.speed_rate))).squeeze()
        if len(audio_speed_tune) < len(data) :
            pad_len = len(data) - len(audio_speed_tune)
            audio_speed_tune = np.r_[np.random.uniform(-0.001,0.001,int(pad_len/2)),
                                   audio_speed_tune,
                                   np.random.uniform(-0.001,0.001,int(np.ceil(pad_len/2)))]
        else: 
            cut_len = len(audio_speed_tune) - len(data)
            audio_speed_tune = audio_speed_tune[int(cut_len/2):int(cut_len/2)+len(data)]
        sample['waveform'] = torch.from_numpy(audio_speed_tune)
        return sample
    
class StretchAudio(AudioTransform):
    """ Do stretching of audio file"""
    def __init__(self, always_apply=False, p=0.5 , rate = None):
        super(StretchAudio, self).__init__(always_apply, p)
        
        if rate:
            self.rate = rate
        else:
            self.rate = np.random.uniform(0.5,1.5)
        
    def apply(self,sample,**params):
        '''
        data : ndarray of audio timeseries
        '''      
        data = sample['waveform']  
        input_length = len(data)
        
        data = librosa.effects.time_stretch(data,self.rate)
        
        if len(data)>input_length:
            data = data[:input_length]
        else:
            data = np.pad(data, (0, max(0, input_length - len(data))), "constant")

        sample['waveform'] = torch.from_numpy(data)
        return sample
    
class PitchShift(AudioTransform):
    """ Do time shifting of audio """
    def __init__(self, always_apply=False, p=0.5 , n_steps=None):
        super(PitchShift, self).__init__(always_apply, p)
        '''
        nsteps here is equal to number of semitones
        '''
        
        self.n_steps = n_steps
        
    def apply(self,sample,**params):
        '''
        data : ndarray of audio timeseries
        '''     
        data = sample['waveform']  
        data =  librosa.effects.pitch_shift(data,sr=22050,n_steps=self.n_steps)
        sample['waveform'] = torch.from_numpy(data)
        return sample
    
    
class AddGaussianNoise(AudioTransform):
    """ Do time shifting of audio """
    def __init__(self, always_apply=False, p=0.5):
        super(AddGaussianNoise, self).__init__(always_apply, p)
        
        
    def apply(self,sample,**params):
        '''
        data : ndarray of audio timeseries
        ''' 
        data = sample['waveform']
        noise = np.random.randn(len(data))
        data_wn = data + 0.005*noise
        sample['waveform'] = torch.from_numpy(data_wn)
        return sample
    
class Gain(AudioTransform):
    """
    Multiply the audio by a random amplitude factor to reduce or increase the volume. This
    technique can help a model become somewhat invariant to the overall gain of the input audio.
    """

    def __init__(self, min_gain_in_db=-12, max_gain_in_db=12, always_apply=False,p=0.5):
        super(Gain,self).__init__(always_apply,p)
        assert min_gain_in_db <= max_gain_in_db
        self.min_gain_in_db = min_gain_in_db
        self.max_gain_in_db = max_gain_in_db


    def apply(self, sample, **args):
        data = sample['waveform']
        amplitude_ratio = 10**(random.uniform(self.min_gain_in_db, self.max_gain_in_db)/20)
        data = data * amplitude_ratio
        sample['waveform'] = torch.from_numpy(data)
        return sample
    
class CutOut(AudioTransform):
    def __init__(self, always_apply=False, p=0.5 ):
        super(CutOut, self).__init__(always_apply, p)
        
    def apply(self,sample,**params):
        '''
        data : ndarray of audio timeseries
        '''
        data = sample['waveform']
        start_ = np.random.randint(0,len(data))
        end_ = np.random.randint(start_,len(data))
        
        data[start_:end_] = 0
        sample['waveform'] = torch.from_numpy(data)
        
        return sample
    
### SPECTROGRAM TRANSFORMATIONS
#TODO: from where?
class FreqMask(object):
    '''
    Frequency masking
    '''
    def __init__(self, freqm):
        self.freqm = torchaudio.transforms.FrequencyMasking(freqm)
    
    def __call__(self, sample):
        fbank = sample['fbank']
        fbank = torch.transpose(fbank, 0, 1)
        # this is just to satisfy new torchaudio version.
        fbank = fbank.unsqueeze(0)

        fbank = self.freqm(fbank)
        
        fbank = fbank.squeeze(0)
        fbank = torch.transpose(fbank, 0, 1)
        sample['fbank'] = fbank

        return sample


class TimeMask(object):
    '''
    Time masking
    '''
    def __init__(self, timem):
        self.timem = torchaudio.transforms.TimeMasking(timem)
    
    def __call__(self, sample):
        fbank = sample['fbank']
        fbank = torch.transpose(fbank, 0, 1)
        # this is just to satisfy new torchaudio version.
        fbank = fbank.unsqueeze(0)

        fbank = self.timem(fbank)
        
        fbank = fbank.squeeze(0)
        fbank = torch.transpose(fbank, 0, 1)
        sample['fbank'] = fbank

        return sample

class Normalize(object):
    '''Normalize spectrogram using dataset mean and std'''
    def __init__(self, norm_mean, norm_std):
        self.norm_mean = norm_mean
        self.norm_std = norm_std
    
    def __call__(self, sample):
        fbank = sample['fbank']
        fbank = (fbank - self.norm_mean) / (self.norm_std * 2)
        sample['fbank'] = fbank
        return sample

class Noise(object):
    '''
    Add random noise to spectrogram
    '''
    def __call__(self, sample):
        fbank = sample['fbank']
        fbank = fbank + torch.rand(fbank.shape[0], fbank.shape[1]) * np.random.rand() / 10
        fbank = torch.roll(fbank, np.random.randint(-10, 10), 0)
        sample['fbank'] = fbank
        return sample
        
class Wav2Fbank(object):
    '''
    Spectrogram conversion V2
    '''
    def __init__(self, target_length, melbins, tf_co, tf_shift, override_wave=False):
        self.target_length = target_length
        self.melbins = melbins
        self.tf_co = tf_co
        self.tf_shift = tf_shift
        self.override_wave = override_wave

    def __call__(self, sample):
        waveform = sample['waveform']
        fbank = torchaudio.compliance.kaldi.fbank(
            waveform, htk_compat=True, sample_frequency=sample['sample_rate'], use_energy=False,
            window_type='hanning', num_mel_bins=self.melbins, dither=0.0, frame_shift=10)
        
        n_frames = fbank.shape[0]

        p = self.target_length - n_frames

        # cut and pad
        if p > 0:
            m = torch.nn.ZeroPad2d((0, 0, 0, p))
            fbank = m(fbank)
        elif p < 0:
            fbank = fbank[0:self.target_length, :]
            
        if self.tf_co is not None:
            fbank=torch.FloatTensor((self.tf_co(image=fbank.numpy()))['image'])
        
        if self.tf_shift is not None:
            fbank=torch.FloatTensor((self.tf_shift(image=fbank.numpy()))['image'])

        sample['fbank'] = fbank

        if self.override_wave:
            del sample['waveform']
        return sample