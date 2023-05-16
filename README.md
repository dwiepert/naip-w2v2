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
* `--weighted_layer`: boolean to trigger weighted_layers mode
* `--rand_weight`: boolean to specify whether to randomize weights for initializing hidden state weighted sum
* `--embedding_type`: specify whether embeddings should be extracted from classification head (ft) or base pretrained model (pt)

### Audio transforms
* `--resample_rate`: an integer value for resampling. Default to 16000
* `--reduce`: a boolean indicating whether to reduce audio to monochannel. Default to True.
* `--clip_length`: integer specifying how many frames the audio should be. Default to 160000.
* `--trim`: boolean indicating whether to trim silence. Default to False.

### Model parameters
* `--layer`:  specify which hidden state is being used. It can be between -1 and 12.
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

## Embedding Extraction
Embedding extraction is now a function within the wrapped Wav2Vec2 model `Wav2Vec2ForSpeechClassification` model (see `extract_embeddings(...)` in [w2v2_models.py](https://github.com/dwiepert/mayo-w2v2/blob/main/src/models/w2v2_models.py)). Notably, this function contains options to extract either the output of the Dense layer from the classification head or the final hidden layer of the base W2V2 model by specifying `embedding_type` as either `ft` for 'finetuned' embedding (extracting from classification head) or `pt` for 'pretrained' embedding (extracting from w2v2 hidden states), on the basis that we generally freeze the base w2v2 model before finetuning. 

We also have an option to extract embeddings from different hidden states by altering the `layer` argument, which takes an int between -1 and 12. 

## Weighted layers mode
For a specific form of training, we created an additional parameter for learning weights for all the hidden model layers. This mode will calculate the weighted sum of hidden states before feeding it to the classifier. 

To trigger this mode, set `--weighted_layers` to True. This mode has not been implemented in the notebook. 
'
Can access the weights after training with `model_name.weightsum`


