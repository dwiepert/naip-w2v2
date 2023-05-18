# W2V2 for Embedding Extraction and Classification
The command line usable, start-to-finish implementation of Wav2vec 2.0 is available with [run.py](https://github.com/dwiepert/mayo-w2v2/blob/main/src/run.py). A notebook tutorial version is also available at [run.ipynb](https://github.com/dwiepert/mayo-w2v2/blob/main/src/run.ipynb). 

This implementation contains options for finetuning a pretrained w2v2 model, evaluating a saved model, or extracting embeddings.

All data is loaded using the simple `WaveformDataset` class implemented in [dataloader_utils.py](https://github.com/dwiepert/mayo-w2v2/blob/main/src/utilities/dataloader_utils.py). It is initialized with a dataframe of labels, indexed by a unique identifier, along with a list of the target labels (df column names), and a transforms object with all of the transforms to perform on an audio file. When loading a sample with this class, it will output a dictionary containing the loaded waveform as a tensor (`sample['waveform']`), the target labels as a tensor (`sample['targets']`), and the unique identifier for the sample (`sample['uid']`).

The transform classes are also implemented in [speech_utils.py](https://github.com/dwiepert/mayo-w2v2/blob/main/src/utilities/speech_utils.py). See the `get_transforms(...)` function in [run.py](https://github.com/dwiepert/mayo-w2v2/blob/main/src/run.py) to see the transforms used for w2v2, or explore [speech_utils.py](https://github.com/dwiepert/mayo-w2v2/blob/main/src/utilities/speech_utils.py) for other transform options.Note that when initializing transforms, you can alter the  `bucket` variable and `lib` variable. As a default, `bucket` is set to None, which will force loading from the local machine. If using GCS, pass a fully initialized bucket. Setting the `lib` value to 'True' will cause the audio to be loaded using librosa rather than torchaudio. 

This implementation uses wrapper classes over the [wav2vec2 models](https://huggingface.co/docs/transformers/model_doc/wav2vec2) available from HuggingFace. The `Wav2Vec2ForSpeechClassification` is the wrapped model, which includes an added classification head with a Dense layer, ReLU activation, dropout, and a final linear projection layer (this class is defined as `ClassificationHead` in [speech_utils.py](https://github.com/dwiepert/mayo-w2v2/blob/main/src/utilities/speech_utils.py)) as well as a function for embedding extraction. See [w2v2_models.py](https://github.com/dwiepert/mayo-w2v2/blob/main/src/models/w2v2_models.py) for information on intialization arguments. Note: you can specify a specific hidden state to use for classification and embedding extraction using the `--layer` argument. 

## Running Environment
The environment must include the following packages, all of which can be dowloaded with pip or conda:
* albumentations
* librosa
* torch, torchvision, torchaudio
* tqdm (this is essentially enumerate(dataloader) except it prints out a nice progress bar for you)
* transformers (must be downloaded with pip)
* pyarrow

If running on your local machine and not in a GCP environment, you will also need to install:
* google-cloud-storage

The [requirements.txt](https://github.com/dwiepert/mayo-w2v2/blob/main/requirements.txt) can be used to set up this environment. 

To access data stored in GCS on your local machine, you will need to additionally run

```gcloud auth application-default login```

```gcloud auth application-defaul set-quota-project PROJECT_NAME```

## Model checkpoints
In order to initialize a wav2vec 2.0 model, you must have access to a pretrained model checkpoint. There are a few different checkpoint options which can be found at [HuggingFace](https://huggingface.co/models). The default model used is [facebook/wav2vec2-base-960h](https://huggingface.co/facebook/wav2vec2-base-960h). These model checkpoints can be loaded in a couple different ways.

1. Use the path directly from HuggingFace. For example in the following url, `https://huggingface.co/facebook/wav2vec2-base-960h`, the checkpoint path would be `facebook/wav2vec2-base-960h`. This method only works on local machines with `transformers` installed in the environment. 

2. Use a path to a local directory where the model checkpoint is saved. All the models on hugging face can be downloaded. To properly load a model, you will need all available model files from a [HuggingFace link](https://huggingface.co/facebook/wav2vec2-base-960h)  under the `files and version` tab. 

3. Use a model checkpoint saved in a GCS bucket. This option can be specified by giving a full file path starting with `gs://BUCKET_NAME/...`. The code will then download this checkpoint locally and reset the checkpoint path to the path it is saved locally. 

## Data structure
This code will only function with the following data structure.

SPEECH DATA DIR

    |

    -- UID 

        |

        -- waveform.EXT (extension can be any audio file extension)

        -- metadata.json (containing the key 'encoding' (with the extension in capital letters, i.e. mp3 as MP3), also containing the key 'sample_rate_hz' with the full sample rate)

and for the data splits

DATA SPLIT DIR

    |

    -- train.csv

    -- test.csv

## Audio Configuration
Data is loaded using an `W2V2Dataset` class, where you pass a dataframe of the file names (UIDs) along with columns containing label data, a list of the target labels (columns to select from the df), specify audio configuration, method of loading, and initialize transforms on the raw waveform and spectrogram (see [dataloader.py](https://github.com/dwiepert/mayo-w2v2/blob/main/src/dataloader.py)). 

To specify audio loading method, you can alter the `bucket` variable and `librosa` variable. As a default, `bucket` is set to None, which will force loading from the local machine. If using GCS, pass a fully initialized bucket. Setting the `librosa` value to 'True' will cause the audio to be loaded using librosa rather than torchaudio. 

The audio configuration parameters should be given as a dictionary (which can be seen in [run.py](https://github.com/dwiepert/mayo-w2v2/blob/main/src/runpy) and [run.ipynb](https://github.com/dwiepert/mayo-w2v2/blob/main/src/run.ipynb). Most configuration values are for initializing audio and spectrogram transforms. The transform will only be initialized if the value is not 0. If you have a further desire to add transforms, see [speech_utils.py](https://github.com/dwiepert/mayo-w2v2/blob/main/src/utilities/speech_utils.py)) and alter [dataloader.py](https://github.com/dwiepert/mayo-w2v2/blob/main/src/dataloader.py) accordingly. 

The following parameters are accepted (`--` indicates the command line argument to alter to set it):

*Audio Transform Information*
* `resample_rate`: an integer value for resampling. Set with `--resample_rate`
* `reduce`: a boolean indicating whether to reduce audio to monochannel. Set with `--reduce`
* `clip_length`: integer specifying how many frames the audio should be. Set with `--clip_length`
* `trim`: boolean specifying whether to trim beginning and end silence. Set with `--trim`


## Arguments
There are many possible arguments to set, including all the parameters associated with audio configuration. The main run function describes most of these, and you can alter defaults as required. 

### Loading data
* `-i, --prefix`: sets the `prefix` or input directory. Compatible with both local and GCS bucket directories containing audio files, though do not include 'gs://'
* `-s, --study`: optionally set the study. You can either include a full path to the study in the `prefix` arg or specify some parent directory in the `prefix` arg containing more than one study and further specify which study to select here.
* `-d, --data_split_root`: sets the `data_split_root` directory or a full path to a single csv file. For classification, it must be  a directory containing a train.csv and test.csv of file names. If runnning embedding extraction, it should be a csv file. Running evaluation only can accept either a directory or a csv file. This path should include 'gs://' if it is located in a bucket. 
* `-l, --label_txt`: sets the `label_txt` path. This is a full file path to a .txt file contain a list of the target labels for selection (see [labels.txt](https://github.com/dwiepert/mayo-ssast/blob/main/labels.txt))
* `--lib`: : specifies whether to load using librosa (True) or torchaudio (False), default=False
* `-c, --checkpoint`: specify a pretrained model checkpoint - this is a base model from w2v2, as mentioned earlier. Default is 'facebook/wav2vec2-base-960h' which is a base model trained on 960h of Librispeech. This is required regardless of whether you include a fine-tuned model path. 
* `-mp, --finetuned_mdl_path`: if running eval-only or extraction, you can specify a fine-tuned model to load in. This can either be a local path of a 'gs://' path, that latter of which will trigger the code to download the specified model path to the local machine. 

### Google cloud storage
* `-b, --bucket_name`: sets the `bucket_name` for GCS loading. Required if loading from cloud.
* `-p, --project_name`: sets the `project_name` for GCS loading. Required if loading from cloud. 
* `--cloud`: this specifies whether to save everything to GCS bucket. It is set as True as default.

### Saving data
* `--dataset`: Specify the name of the dataset you are using. When saving, the dataset arg is used to set file names. If you do not specify, it will assume the lowest directory from data_split_root. Default is None. 
* `-o, --exp_dir`: sets the `exp_dir`, the LOCAL directory to save all outputs to. 
* `--cloud_dir`: if saving to the cloud, you can specify a specific place to save to in the CLOUD bucket. Do not include the bucket_name or 'gs://' in this path.

### Run mode
* `-m, --mode`: Specify the mode you are running, i.e., whether to run fine-tuning for classification ('finetune'), evaluation only ('eval-only'), or embedding extraction ('extraction'). Default is 'finetune'.
* `--freeze`: boolean to specify whether to freeze the base model
* `--weighted`: boolean to trigger learning weights for hidden states
* `--layer`: Specify which model layer (hidden state) output to use. Default is -1 which is the final layer. 
* `--embedding_type`: specify whether embeddings should be extracted from classification head (ft) or base pretrained model (pt)

### Audio transforms
see the audio configurations section for which arguments to set

### Model parameters
* `-pm, --pooling_mode`: specify method of pooling the last hidden layer for embedding extraction. Options are 'mean', 'sum', 'max'.
* `-bs, --batch_size`: set the batch size (default 8)
* `-nw, --num_workers`: set number of workers for dataloader (default 0)
* `-lr, --learning_rate`: you can manually change the learning rate (default 0.0003)
* `-e, --epochs`: set number of training epochs (default 1)
* `--optim`: specify the training optimizer. Default is `adam`.
* `--loss`: specify the loss function. Can be 'BCE' or 'MSE'. Default is 'BCE'.
* `--scheduler`: specify a lr scheduler. If None, no lr scheduler will be use. The only scheduler option is 'onecycle', which initializes `torch.optim.lr_scheduler.OneCycleLR`
* `--max_lr`: specify the max learning rate for an lr scheduler. Default is 0.01.

### Classification Head parameters
* `--activation`: specify activation function to use for classification head
* `--final_dropout`: specify dropout probability for final dropout layer in classification head
* `--layernorm`: specify whether to include the LayerNorm in classification head

For more information on arguments, you can also run `python run.py -h`. 

## Functionality
This implementation contains many functionality options as listed below:

### 1. Finetuning
You can finetune W2V2 for classifying speech features using the `W2V2ForSpeechClassification` class in [w2v2_models.py]((https://github.com/dwiepert/mayo-w2v2/blob/main/src/models/w2v2_models.py) and the `finetune(...)` function in [loops.py](https://github.com/dwiepert/mayo-w2v2/blob/main/src/loops.py). 

This mode is triggered by setting `-m, --mode` to 'finetune' and also specifying which pooling method to use to pool the hidden dim with `--pm, --pooling_mode`. The options are 'mean', 'sum', and 'max'.

There are a few different parameters to consider. Firstly, the classification head can be altered to use a different amount of dropout and to include/exclude layernorm. See `ClassificationHead` class in [speech_utils.py](https://github.com/dwiepert/mayo-w2v2/blob/main/src/utilities/speech_utils.py) for more information. 

Default run mode will also freeze the base W2V2 model and only finetune the classification head. This can be altered with `--freeze`. 

We also include the option to use a different hidden state output as the input to the classification head. This can be specified with `--layer` and must be an integer between 0 and `model.n_states` (or -1 to get the final layer). This works in the `W2V2ForSpeechClassification` class by getting a list of hidden states and indexing using the `layer` parameter. 

Finally, we added functionality to train an additional parameter to learn weights for the contribution of each hidden state to classification. The weights can be accessed with `model.weightsum`. This mode is triggered by setting `--weighted` to True. If initializing a model outside of the run function, it is still triggered with an argument called `weighted`. 

### 2. Evaluation only
If you have a finetuned model and want to evaluate it on a new data set, you can do so by setting `-m, --mode` to 'eval'. You must then also specify a `-mp, --finetuned_mdl_path` to load in. 

It is expected that there is an `args.pkl` file in the same directory as the finetuned model to indicate which arguments were used to initialize the finetuned model. This implementation will load the arguments and initialize/load the finetuned model with these arguments. If no such file exists, it will use the arguments from the current run, which could be incompatible if you are not careful. 

### 3. Embedding extraction.
We implemented multiple embedding extraction methods for use with the SSAST model. The implementation is a function within `W2V2ForSpeechClassification` called `extract_embedding(x, embedding_type, layer, pooling_mode, ...)`, which is called on batches instead of the forward function. 

Embedding extraction is triggered by setting `-m, --mode` to 'extraction'. 

You must also consider where you want the embeddings to be extracted from. The options are as follows:
1. From the output of a hidden state? Set `embedding_type` to 'pt'. Can further set an exact hidden state with the `layer` argument. By default, it will use the layer specified at the time of model initialization. The model default is to give the last hidden state run through a normalization layer, so the embedding is this output merged to be of size (batch size, embedding_dim). It will also automatically use the merging strategy defined by the mode set at the time of model initialization, but this can be changed at the time of embedding extraction by redefining `pooling_mode`.
2. After weighting the hidden states? Set `embedding_type` to 'wt'. This version requires that the model was initially finetuned with  `weighted` set to True.
3. From a layer in the classification head that has been finetuned? Set `embedding_type` to 'ft'. This version requires no further specification and will always return the output from the first dense layer in the classification head, prior to any activation function or normalization. 

## Visualize Attention
Not yet implemented. 



