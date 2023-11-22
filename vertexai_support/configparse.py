'''
Create and submit configurations for Vertex AI jobs

Last modified: 08/2023
Author: Daniela Wiepert
Email: wiepert.daniela@mayo.edu
File: configparse.py
'''


import os
import json 
import argparse
import ast
import itertools
import subprocess
import glob


#### potential parameters - this can be heavily edited for other use cases/parameters
lr = {
    "parameterId": "learning_rate",
    "discreteValueSpec": {
    "values": [
        0.0001,
        0.001
    ]
    }
}

mlr = {
    "parameterId": "max_lr",
    "discreteValueSpec": {
    "values": [
        0.001,
        0.01,
        0.1
    ]
    }
}

wd = {
    "parameterId": "weight_decay",
    "discreteValueSpec": {
    "values": [
        0.0001,
        0.001,
        0.01
    ]
    }
}

cb = {
    "parameterId": "clf_bottleneck",
    "discreteValueSpec": {
    "values": [
        300,
        700,
        768
    ]
    }
}

cb2 = {
    "parameterId": "clf_bottleneck",
    "discreteValueSpec": {
    "values": [
        300,
        700
    ]
    }
}

cb1 = {
    "parameterId": "clf_bottleneck",
    "discreteValueSpec": {
    "values": [
        300
    ]
    }
}

sb = {
    "parameterId": "sd_bottleneck",
    "discreteValueSpec": {
    "values": [
        300,
        700,
        768
    ]
    }
}

fd = {
    "parameterId": "final_dropout",
    "discreteValueSpec": {
        "values": [
        0.2,
        0.3
        ]
    }
    }

l2 = {
    "parameterId": "layer",
    "discreteValueSpec": {
        "values": [
        0,
        6,
        12
        ]
    }
    }

l =  {
        "parameterId": "layer",
    "integerValueSpec": {
        "minValue": "0",
        "maxValue": "12"
    },
    "scaleType": "UNIT_LINEAR_SCALE"
        }

### TODO: figure out conditional parameters?
#https://cloud.google.com/vertex-ai/docs/reference/rest/v1/StudySpec#conditionalparameterspec
### TODO: figure out str parameters? CategoricalValueSpec
#https://cloud.google.com/vertex-ai/docs/training/hyperparameter-tuning-overview


####### setup parts of dictionary

def setup_args(arg_dict, alter_dict, tune_params):
    """
    Set up arguments for the vertex ai job 
    
    :param arg_dict: dictionary with all the default arguments loaded from a .txt in --arg_name form as keys and the value of the arguments as the value
    :param alter_dict: dictionary with all of the information for altering arguments with keys in --arg_name form and the value of the arguments as the value
    :param tune_params: list of argument names that are being tuned. Can be an empty list if this is a single job and not an hp tuning job
    :return arg_list: updated argument list
    """
    for k in alter_dict:
        v = alter_dict[k]
        if not isinstance(v, str):
            v = str(v)
        arg_dict[k] = v

    for p in tune_params:
        del arg_dict[p]
    
    arg_list = []
    for k in arg_dict:
        v = arg_dict[k]
        a = "=".join([k,v])
        arg_list.append(a)
    
    return arg_list

def setup_params(tune_params, sd_bottleneck, all_layers, weighted):
    """
    Set up parameter list for vertex ai hyperparameter tuning job. THIS CAN BE EDITED - currently not flexible.

    :param tune_params: list of argument names that are being tuned. Must not be an empty list.
    :param sd_bottleneck: value of sd_bottleneck for the given hptuning job. clf_bottleneck must be <= sd_bottleneck, so clf_bottleneck list is adjusted accordingly
    :param all_layers: boolean indicating whether to run all layers (0-12) or a smaller subset of layers
    :param weighted: boolean indicating weighted sum version of the model, in which case layer does not need to be specified
    :return param_list: list of all parameter dictionaries

    """

    param_list = []
    for p in tune_params:
        if 'learning_rate' in p:
            param_list.append(lr)
        elif 'max_lr' in p:
            param_list.append(mlr)
        elif 'weight_decay' in p:
            param_list.append(wd)
        elif 'clf_bottleneck' in p:
            if sd_bottleneck == 768:
                param_list.append(cb)
            elif sd_bottleneck == 700:
                param_list.append(cb2)
            elif sd_bottleneck == 300:
                param_list.append(cb1)
        elif 'sd_bottleneck' in p:
            param_list.append(sb)
        elif 'final_dropout' in p:
            param_list.append(fd)
        elif 'layer' in p and not weighted:
            if not all_layers:
                param_list.append(l2)
            else:
                param_list.append(l)

    return param_list

def setup_singlejob(arg_dict, image_uri):
    """
    :param arg_dict: dictionary with all the default arguments loaded from a .txt in --arg_name form as keys and the value of the arguments as the value
    :paramm image_uri: path to image in google cloud bucket
    :return tjobspec: config dictionary required for a single job
    """
    
    arg_list = setup_args(arg_dict, {}, [])

    tjobspec = {
        "workerPoolSpecs": [
          {
            "machineSpec": {
              "machineType": "n1-highmem-2",
              "acceleratorType": "NVIDIA_TESLA_T4",
              "acceleratorCount": 1
            },
            "replicaCount": "1",
            "diskSpec": {
              "bootDiskType": "pd-ssd",
              "bootDiskSizeGb": 100
            },
            "containerSpec": {
              "imageUri":image_uri,
              "args": arg_list
            }
          }
        ]
      }
    return tjobspec

def setup_hptuningjob(arg_dict, alter_dict, tune_params, image_uri, sd_bottleneck, all_layers, weighted):
    """
    :param arg_dict: dictionary with all the default arguments loaded from a .txt in --arg_name form as keys and the value of the arguments as the value
    :param alter_dict: dictionary with all of the information for altering arguments with keys in --arg_name form and the value of the arguments as the value
    :param tune_params: list of argument names that are being tuned. Can be an empty list if this is a single job and not an hp tuning job
    :param image_uri: path to image in google cloud bucket
    :param sd_bottleneck: value of sd_bottleneck for the given hptuning job. clf_bottleneck must be <= sd_bottleneck, so clf_bottleneck list is adjusted accordingly
    :param all_layers: boolean indicating whether to run all layers (0-12) or a smaller subset of layers
    :param weighted: boolean indicating weighted sum version of the model, in which case layer does not need to be specified
    :return config: config dictionary for hptuning job
    :return max_trials: max trial count based on parameters in the config file
    """

    arg_list = setup_args(arg_dict, alter_dict, tune_params)

    tjobspec = {
        "workerPoolSpecs": [
          {
            "machineSpec": {
              "machineType": "n1-highmem-2",
              "acceleratorType": "NVIDIA_TESLA_T4",
              "acceleratorCount": 1
            },
            "replicaCount": "1",
            "diskSpec": {
              "bootDiskType": "pd-ssd",
              "bootDiskSizeGb": 100
            },
            "containerSpec": {
              "imageUri":image_uri,
              "args": arg_list
            }
          }
        ]
      }
    
    param_list = setup_params(tune_params, sd_bottleneck, all_layers, weighted)
    sspec = { "algorithm": "GRID_SEARCH",
      "metrics": [
        {
          "metricId": "AUC",
          "goal": "MAXIMIZE"
        }],
        "parameters":param_list}
    
    config = {"studySpec":sspec, "trialJobSpec":tjobspec}
    
    
    return config


def config_combinations(labels, shared_dense, weighted, sd_bottlenecks):
    """"""
    #1clf vs 5clf
    clf = list(range(0,len(labels)))
    if shared_dense:
        sd = [False, True]
    else:
        sd = [False]

    if weighted:
        ws = [False, True]
    else:
        ws = [False]

    list1 = [clf, sd, sd_bottlenecks, ws]
    combos = [p for p in itertools.product(*list1)] #make list of all combinations

    #if shared dense is False, sd_bottleneck should only be one version
    new_combos = []
    for c in combos:
        if not c[1]:
            if c[2] == max(sd_bottlenecks):
                new_combos.append(c)
        else:
            new_combos.append(c)
    return new_combos

def make_path_1(config_dir, c, all_layers):
    """
    make file path for hp tuning
    """
    if c[0] == 0:
        clf='1clf'
    else:
        clf='5clf'
    
    if c[1]:
        sd = 'sd'
    else:
        sd = 'nosd'

    if c[3]:
        w = 'ws'
    else:
        w = 'nows'

    if all_layers:
        l = 'alllayers'
    else:
        l = 'sublayers'
    
    config_path = os.path.join(config_dir, f'config_{clf}_{sd}{c[2]}_{w}_{l}.json')

    return config_path


def generate_argdict(arg_list):
    arg_dict = {}
    for a in arg_list:
        a = a.split("=")
        arg_dict[a[0]]=a[1]
    return arg_dict

def generate_alterdict(label, shared_dense, sd_bottleneck, weighted, cloud_dir):
    """
    Generate a dictionary containing the values you'd like to alter. Can edit to include more options. 
    Assume that the input arg_list has the correct values for a single job (e.g., learning rate, layer, epochs, etc.), so this is for hptuning
    """
    alter_dict = {}
    alter_dict["--label_txt"] = label
    alter_dict["--shared_dense"] = str(shared_dense)
    alter_dict["--weighted"] = str(weighted)
    alter_dict["--sd_bottleneck"] = str(sd_bottleneck)
    alter_dict["--cloud_dir"] = cloud_dir

    return alter_dict

def main():
    parser = argparse.ArgumentParser()
    #Inputs
    parser.add_argument("--arg_txt", default="", help="txt file (separated by new lines) that contains all of the arguments for run.py" )
    parser.add_argument("--image", default="", help="specify image for the vertex AI job")
    parser.add_argument("--tune_params", nargs="+", default=["--layer"], help="list of parameters to tune, must include --") #108 combinations
    parser.add_argument("--sd_bottlenecks", nargs="+", default=[768], type=int)
    parser.add_argument("--shared_dense", type=ast.literal_eval, default=False, help="specify whether using shared_dense. If single job, uses this parameter. If hptuning job + True, makes configs for both False/True")
    parser.add_argument("--weighted", type=ast.literal_eval, default=False, help="specify whether using weighted sum. If single job, uses this parameter. If hptuning job + True, makes configs for both False/True")
    parser.add_argument("--all_layers", type=ast.literal_eval, default=True, help="specify whether using all layers or just a subset of layers for a hp_tuning job with layers as a parameter")
    parser.add_argument("-l","--labels", default=[''], nargs="+")
    parser.add_argument("--cloud_dir", default="")
    parser.add_argument("--config_dir", default='vertexai_support/configs')
    parser.add_argument("--hp_tuning", default=False, type=ast.literal_eval, help="specify if setting up a hyperparameter tuning job or running a single job")
    parser.add_argument("--max_trial_count", default=108, type=int)
    parser.add_argument("--parallel_trial_count", default=3, type=int)
    parser.add_argument("--make_configs", default=False, type=ast.literal_eval)
    parser.add_argument("--run_configs", default=True, type=ast.literal_eval)
    args = parser.parse_args()

    #check if the directory to store configuration files exists
    if not os.path.exists(args.config_dir):
        os.makedirs(args.config_dir)

    with open(args.arg_txt) as f:
        arg_list = f.readlines()
    arg_list = [a.strip() for a in arg_list]

    if not args.hp_tuning:
        assert len(args.labels) == 1, 'If running a single job, only one label can be given'

        config = setup_singlejob(generate_argdict(arg_list), args.image) #note: assumes that the arg list has the correct values for your single job
        config_path = "" #TODO
        cmd = ["gcloud", "ai", "custom-jobs", "create",
               "--region=us-central1",
               f"--config={c}",
               f"--display-name=w2v2layers_{c}"]
        subprocess.run(cmd)

    else:
        if args.make_configs:
            combos = config_combinations(args.labels, args.shared_dense, args.weighted, args.sd_bottlenecks) #TODO:SCHEDULER COULD LATER BE ADDED HERE AS A COMBINATION, CURRENTLY NOT AN OPTION.
            
            config_list = []
            for c in combos:
                arg_dict = generate_argdict(arg_list)
                alter_dict = generate_alterdict(args.labels[c[0]],c[1],c[2],c[3],args.cloud_dir)

                config = setup_hptuningjob(arg_dict, alter_dict, args.tune_params, args.image, c[2], args.all_layers, c[3])

                config_path = make_path_1(args.config_dir, c, args.all_layers)
                config_list.append(config_path)

                if not os.path.exists(config_path):
                    obj= json.dumps(config)
                    with open(config_path, "w") as outfile:
                        outfile.write(obj)
        else:
            config_list = glob.glob(os.path.join(args.config_dir,"*.json"))

        if args.run_configs:
            for c in config_list:
                cmd = ["gcloud", "ai", "hp-tuning-jobs", "create", 
                "--region=us-central1", 
                f"--display-name=w2v2layers_{os.path.basename(c)[7:-5]}", 
                f"--config={c}",
                f"--max-trial-count={args.max_trial_count}",
                f"--parallel-trial-count={args.parallel_trial_count}"]
                subprocess.run(cmd)
                #note, if this fails, run gcloud auth login andgcloud config set project PROJECT NAME


if __name__ == "__main__":
    main()
