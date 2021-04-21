import torch
import torch.nn as nn
import ipdb
import torch.multiprocessing as _mp
import torch.nn.functional as F
import sys
sys.path.append("./model_src/")
from preactresnet import *
from wideresnet import *
from cnn import *
from resnet import *
from resnet_8x import *

def get_model(params):
    if params.model_type == "preactresnet":
            model = PreActResNet18().to(params.device)  
    elif params.model_type == "cnn":
        model = CNN().to(params.device)
    elif params.model_type == "resnet34":
        model = resnet34().to(params.device)
    else:
        splits = params.model_type.split("-")
        depth = int(splits[1])
        widen_factor = int(splits[2])
        model = WideResNet(depth = depth, widen_factor = widen_factor).to(params.device)  
    
    return model

def load_model(params, model):
    model_name = params.path 
    try:
        model.load_state_dict(torch.load(f"{model_name}.pt", map_location = params.device))
    except:
        if params.model_type == "cnn":
            req_key = ["conv1.weight", "conv1.bias", "conv2.weight", "conv2.bias", "fc1.weight", "fc1.bias", "fc2.weight", "fc2.bias"]
            found_key = ["0.weight", "0.bias", "3.weight", "3.bias", "7.weight", "7.bias", "9.weight", "9.bias"] 
            dictionary = torch.load(f"{model_name}.pt", map_location = params.device)
            new_dict = {}
            for key in dictionary.keys():
                new_key = req_key[found_key.index(key)]
                new_dict[new_key] = dictionary[key]
            model.load_state_dict(new_dict)
        #todo: make sure resent34 can be correcrly loaded.
        else:
            dictionary = torch.load(f"{model_name}.pt", map_location = params.device)['state_dict']
            new_dict = {}
            for key in dictionary.keys():
                new_key = key[7:]
                if new_key.split(".")[0] == "sub_block1":
                    continue
                new_dict[new_key] = dictionary[key]
            model.load_state_dict(new_dict)
    return model
