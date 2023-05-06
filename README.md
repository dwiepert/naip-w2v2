# W2V2 for Embedding Extraction and Classification

## Running Requirements
In order to run this code, you must have access to a pretrained model. First, try a just a path to a huggingFace model like `facebook/wav2vec2-base-960h`. Other options can be found on [HuggingFace](https://huggingface.co/models). If this doesn't work, you can pass it a full file path to a locally saved or GCS saved checkpoint that was downloaded from [HuggingFace](https://huggingface.co/models). Checkpoints are already available in the GSC bucket at `gs://ml-e107-phi-shared-aif-us-p/m144443/checkpoints` if this option is necessary. 
These models can just be called from their path in our GCS bucket `gs://ml-e107-phi-shared-aif-us-p/m144443/checkpoints`, or you can download one like [wav2vec2-base-960h](https://huggingface.co/facebook/wav2vec2-base-960h). The names of other options are available in the GCS checkpoints folder.

The environment must include the following packages, all of which can be dowloaded with pip or conda:
* albumentations (has not yet been tested in GCP environment)
* librosa
* torch, torchvision, torchaudio
* tqdm (this is essentially enumerate(dataloader) except it prints out a nice progress bar for you)

If running on your local machine and not in a GCP environement, you will also need to install:
* google-cloud
* google-cloud-storage
* google-cloud-bigquery 

If working locally and the data is stored in GCS, additionally run

```gcloud auth application-default login```

```gcloud auth application-defaul set-quota-project PROJECT_NAME```

## dataloader_utils.py
All helper classes and functions for creating a waveform dataset and initializing transforms. 

## Running the Model
All data is loaded using a WaveformDataset class, where you pass it a dataframe of the file names (UIDs) and columns with label data, a list of the target labels, and a transforms objects of all the transforms to do to the sample. Results in a dictionary of samples. Can access the waveform with sample['waveform'] and the target labels with sample['targets]. See [dataloader_utils.py](https://github.com/dwiepert/mayo-w2v2/blob/main/src/utilities/dataloader_utils.py) for more information.

When initializing transforms, you can alter the  `bucket` variable and `lib` variable. As a default, `bucket` is set to None, which will force loading from the local machine. If using GCS, pass a fully initialized bucket. Setting the `lib` value to 'True' will cause the audio to be loaded using librosa rather than torchaudio. 

The command line usable, start-to-finish implementation of w2v2 is available with [run_w2v2_mayo.py](https://github.com/dwiepert/mayo-w2v2/blob/main/src/run_w2v2_mayo.py). We also have a notebook verision at [run_w2v2_mayo.ipynb](https://github.com/dwiepert/mayo-w2v2/blob/main/src/run_w2v2_mayo.ipynb). It contains options for fine-tuning, evaluation only, or getting embeddings
There are many possible arguments to set, including all the parameters associated with audio configuration. The main run function describes most of these, and you can alter defaults as required. We will list some of the most important.

* `-i`: sets the `prefix` or input directory. Compatible with both local and GCS bucket directories containing audio files, though do not include 'gs://'
* `-s`: optionally set the study. You can either include a full path to the study in the `prefix` arg or specify some parent directory in the `prefix` arg containing more than one study and further specify which study to select here.
* `-d`: sets the `data_split_root` directory or a full path to a single csv file. For classification, it must be  a directory containing a train.csv and test.csv of file names. If only evaluating a model or doing an embedding extraction, it should be a csv file. This path should include 'gs://' if it is located in a bucket. 
* `-l`: sets the `label_txt` path. This is a full file path to a .txt file contain a list of the target labels for selection (see [labels.txt](https://github.com/dwiepert/mayo-ssast/blob/main/src/labels.txt))
* `-b`: sets the `bucket_name` for GCS loading. Required if loading from cloud.
* `-p`: sets the `project_name` for GCS loading. Required if loading from cloud. 
* `--lib`: specifies whether to load using librosa (True) or torchaudio (False)
* `-o`: sets the `exp_dir`, the directory to save all outputs to. 
* `--dataset`: specify the name of the dataset you are using - will be used for naming outputs
* `--resample_rate`: an integer value for resampling. Default to 16000
* `--reduce`: a boolean indicating whether to reduce audio to monochannel. Default to True.
* `--clip_length`: integer specifying how many frames the audio should be. Default to 160000
* `--trim`: boolean indicating whether to trim silence. Default to False.
* `--mode`: Specify the mode you are running, i.e., whether to run fine-tuning for classification ('finetune'), evaluation only ('eval-only'), or embedding extraction ('extraction'). Default is 'finetune'.
* `--mdl_path`: if running eval-only or extraction, you can specify a fine-tuned model to load in.
* `--batch_size`: set the batch size (default 8)
* `--num_workers`: set number of workers for dataloader (default 0)
* `--epochs`: set number of training epochs (default 1)
* `-c`: specify a pretrained model checkpoint - this is a base model from w2v2, as mentioned earlier. Default is 'facebook/wav2vec2-base-960h' which is a base model trained on 960h of Librispeech. This is required regardless of whether you include a fine-tuned model path. 
* `-pm`: specify method of pooling the last hidden layer for embedding extraction. Options are 'mean', 'sum', 'max'.
* `-lr`: you can manually change the learning rate (default 0.0003)

Please note that there are a few different model classes depending on task. Please see [w2v2_models.py](https://github.com/dwiepert/mayo-w2v2/blob/main/src/models/w2v2_models.py) for more information.

### Finetuning
If running finetuning, we use the following architecture and parameters:
* MSE Loss
* Adam Optimizer
* Classficiation head with a dense layer, ReLU, dropout, and a final linear layer

### Evaluation
Evaluation metrics will be AUCs

### Embedding extraction
Take the last hidden layer of the w2v2 model and pool it to get one embedding vector that is hidden_state_dim length. A specific model version is used so that the model output is the embedding rather than predictions. 







