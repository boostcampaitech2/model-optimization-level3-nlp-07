import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch.optim as optim
import yaml

import optuna
import argparse
import os
from datetime import datetime
from typing import Any, Dict, Tuple, Union
from src.dataloader import create_dataloader
from src.loss import CustomCriterion
from src.model import Model
from src.modules.__init__ import __all__
from src.trainer_mod import TorchTrainer
from src.utils.common import get_label_counts, read_yaml
from src.utils.torch_utils import check_runtime, model_info
from train_mod import train

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
DIR = "configs/data/taco.yaml"
BATCHSIZE = 128
N_TRAIN_EXAMPLES = BATCHSIZE * 30
N_VALID_EXAMPLES = BATCHSIZE * 10
MODULES=[
    "Bottleneck", # out_channels,shortcut(bool),groups(dw),expansion:float,activation:str
    "Conv", #out_channels,kernel_size,stride,null,groups,activation
    "DWConv", #out_channels,kernel_size,stride,activation
    "InvertedResidualv2", #out_channels,expansion,stride
    "InvertedResidualv3", #kernel_size,expansion,out_channels,se,hardswish,stride
    "FusedMBConv", #expand_ratio,out_channels,stride,kernel
    "MBConv", #expand_ratio,out_channels,stride,kernel
    "Fire", #squeeze_channels,expand1x1_channels,expand3x3_channels
    "Dropout", #p
]

def suggest_hp(module_name,in_features,trial,s,i):
    if module_name!="Fire":
        out_channels = int(in_features*trial.suggest_float("n_units_l{}".format(i), 1, 2)+4)//8*8
    stride=1
    if module_name=='Bottleneck':
        return [out_channels],0,out_channels
    elif module_name=='Conv':
        if s<5:
            stride=trial.suggest_int("stride", 1, 2)
        return [out_channels,3,stride,None,1,trial.suggest_categorical("activation", ["ReLU", "HardSwish"])],stride==2,out_channels
    elif module_name=='DWConv':
        if s<5:
            stride=trial.suggest_int("stride", 1, 2)
        return [out_channels,3,stride,None,trial.suggest_categorical("activation", ["ReLU", "HardSwish"])],stride==2,out_channels
    elif module_name=='InvertedResidualv2':
        if s<5:
            stride=trial.suggest_int("stride", 1, 2)
        return [out_channels,trial.suggest_int("expansion", 1, 4),stride],stride==2,out_channels
    elif module_name=='InvertedResidualv3':
        if s<5:
            stride=trial.suggest_int("stride", 1, 2)
        return [3,trial.suggest_int("expansion", 1, 4),out_channels,trial.suggest_categorical("se", [True, False]),trial.suggest_categorical("hardswish", [True, False]),stride],stride==2,out_channels
    elif module_name=='FusedMBConv':
        if s<5:
            stride=trial.suggest_int("stride", 1, 2)
        return [trial.suggest_int("expand_ratio", 1, 4),out_channels,stride,3],stride==2,out_channels
    elif module_name=='MBConv':
        if s<5:
            stride=trial.suggest_int("stride", 1, 2)
        return [trial.suggest_int("expand_ratio", 1, 4),out_channels,stride,3],stride==2,out_channels
    elif module_name=='Dropout':
        return [trial.suggest_uniform("p", 0, 0.7)],0,out_channels
    elif module_name=='Fire':
        fargs=[in_features//trial.suggest_int("squeeze_channels", 2, 4),trial.suggest_int("expand1x1_channels", 1, 32),trial.suggest_int("expand3x3_channels", 1, 32)]
        return fargs,0,int(fargs[1]+fargs[2])

def define_model(trial):
    n_layers = 6
    layers = []
    s=0
    layers.append([2,'Conv',[32,3,1,None,1,'HardSwish']])
    in_features = 32
    for i in range(n_layers):
        repeats=trial.suggest_int("repeat_" + str(i), 1, 2)
        module=trial.suggest_int("module_" + str(i), 0, len(MODULES)-1)
        arg,sp,out_features=suggest_hp(MODULES[module],in_features,trial,s,i)
        s+=sp
        layers.append([repeats,MODULES[module],arg])
        in_features = out_features
    layers.append([1, 'GlobalAvgPool', []])
    layers.append([1, 'Flatten', []])
    layers.append([1, 'Linear', [6]])
    cfg={"input_channel":3,"depth_multiple":1,'width_multiple':1,'backbone':layers}
    return cfg

def objective(trial):
    torch.cuda.empty_cache()
    data_config =read_yaml(DIR)
    data_config['EPOCHS']=10
    model_config = define_model(trial)
    log_dir = os.environ.get("SM_MODEL_DIR", os.path.join("search", 'latest'))
    try:
        if os.path.exists(log_dir): 
            modified = datetime.fromtimestamp(os.path.getmtime(log_dir + '/best.pt'))
            new_log_dir = os.path.dirname(log_dir) + '/' + modified.strftime("%Y-%m-%d_%H-%M-%S")
            os.rename(log_dir, new_log_dir)
    except:
        pass

    os.makedirs(log_dir, exist_ok=True)

    test_loss, test_f1,test_acc,flops=train(
        model_config=model_config,
        data_config=data_config,
        log_dir=log_dir,
        fp16=data_config["FP16"],
        device=DEVICE,
    )
    torch.cuda.empty_cache()
    return flops, test_f1

if __name__=="__main__":
    

    study = optuna.create_study(directions=["minimize","maximize"])
    study.optimize(objective, n_trials=20)
    #print(study.best_params)
    fig = optuna.visualization.plot_pareto_front(study)
    fig.show()