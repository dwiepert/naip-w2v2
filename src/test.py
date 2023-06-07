#test importing from local files

try:
    from utilities import *
except:
    print('Utilities not imported')

try:
    from models import *
except:
    print('Models not imported')

try:
    from loops import *

except:
    print('Loops not imported')


#check other required imports
try:
    import albumentations
except:
    print('Albumentations not installed')

try:
    import pyarrow
except:
    print('Pyarrow not installed')

try:
    from tqdm import tqdm
except:
    print('TQDM not installed')

try:
    import transformers
except:
    print('Transformers not installed')

try:
    import librosa
except:
    print('Librosa not installed')

try:
    import torchaudio
except:
    print('Torchaudio not installed')

try:
    import torchvision
except:
    print('Torchvision not installed')

try: 
    import matplotlib.pyplot as plt
except:
    print('Matplot lib not installed')


#try basic imports

try:
    import numpy as np
    import pandas as pd
    import torch
    import torch.nn as nn 
    from torch.utils.data import Dataset
    from sklearn.metrics import roc_auc_score, roc_curve
    from google.cloud import storage
except:
    print('Basic import statement failing')

print('All imports behaving properly :)')