"""Baseline train
- Author: Junghoon Kim
- Contact: placidus36@gmail.com
"""
import gc
import argparse
import os
from datetime import datetime
from typing import Any, Dict, Tuple, Union

import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from yaml import loader
# from src.daliloader import create_dali_dl_train,create_dali_dl_test,create_dali_dl_valid
from src.daliref import DALIDataloader
from src.dataloader import create_dataloader
from src.dalipipe import dali_mixed_pipeline
from src.loss import CustomCriterion
from src.model import Model
from src.dali_trainer import DaliTrainer
from src.torch_trainer import TorchTrainer
from src.utils.common import get_label_counts, read_yaml
from src.utils.torch_utils import check_runtime, model_info




def train(
    model_config: Dict[str, Any],
    data_config: Dict[str, Any],
    loader_mode:str,
    log_dir: str,
    fp16: bool,
    device: torch.device,
    
) -> Tuple[float, float, float]:
    """Train."""
    # save model_config, data_config
    with open(os.path.join(log_dir, "data.yml"), "w") as f:
        yaml.dump(data_config, f, default_flow_style=False)
    with open(os.path.join(log_dir, "model.yml"), "w") as f:
        yaml.dump(model_config, f, default_flow_style=False)

    model_instance = Model(model_config, verbose=True)
    model_path = os.path.join(log_dir, "best.pt")
    print(f"Model save path: {model_path}")
    if os.path.isfile(model_path):
        model_instance.model.load_state_dict(
            torch.load(model_path, map_location=device)
        )
    model_instance.model.to(device)
    if loader_mode == "PT":
    # Create PyTorch dataloader
        train_dl, val_dl, test_dl = create_dataloader(data_config)
        steps = len(train_dl)
        
    elif loader_mode == "DALI":
    #Create dali dataloader #Added by KBS
        pip_train = dali_mixed_pipeline("train",num_threads=4,batch_size=64,device_id=0)
        pip_valid = dali_mixed_pipeline("val",num_threads=4,batch_size=64,device_id=0)
        pip_train.build()
        pip_valid.build()
        
        dali_train_dl = DALIDataloader(pipeline=pip_train,
                                size=pip_train.epoch_size("Reader"),
                                batch_size=64, 
                                onehot_label=True,
                                )
        
        
        dali_val_dl = DALIDataloader(pipeline=pip_valid,
                                size=pip_valid.epoch_size("Reader"),
                                batch_size=64, 
                                onehot_label=True,
                                )
        steps = len(dali_train_dl)
    #==========================================================

    

    # Create optimizer, scheduler, criterion
    optimizer = torch.optim.SGD(
        model_instance.model.parameters(), lr=data_config["INIT_LR"], momentum=0.9
    )
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer=optimizer,
        max_lr=data_config["INIT_LR"],
        steps_per_epoch=steps,
        epochs=data_config["EPOCHS"],
        pct_start=0.05,
    )
    criterion = CustomCriterion(
        samples_per_cls=get_label_counts(data_config["DATA_PATH"])
        if data_config["DATASET"] == "TACO"
        else None,
        device=device,
    )
    # Amp loss scaler
    scaler = (
        torch.cuda.amp.GradScaler() if fp16 and device != torch.device("cpu") else None
    )
    if loader_mode == "PT":
    # Create trainer
        trainer = TorchTrainer(
            model=model_instance.model,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            device=device,
            model_path=model_path,
            verbose=1,
    )
    
        best_acc, best_f1 = trainer.train(
            train_dataloader=train_dl,
            n_epoch=data_config["EPOCHS"],
            val_dataloader=val_dl ,
        )

    # evaluate model with test set
        model_instance.model.load_state_dict(torch.load(model_path))
        test_loss, test_f1, test_acc = trainer.test(
            model=model_instance.model, test_dataloader=val_dl 
        )
    elif loader_mode == "DALI":
        # Create trainer
        trainer = DaliTrainer(
            model=model_instance.model,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            device=device,
            model_path=model_path,
            verbose=1,
    )
    
        best_acc, best_f1 = trainer.train(
            train_dataloader=dali_train_dl,
            n_epoch=data_config["EPOCHS"],
            val_dataloader=dali_val_dl ,
        )

    # evaluate model with test set
        model_instance.model.load_state_dict(torch.load(model_path))
        test_loss, test_f1, test_acc = trainer.test(
            model=model_instance.model, test_dataloader=dali_val_dl 
        )
    return test_loss, test_f1, test_acc


if __name__ == "__main__":
    
    torch.cuda.empty_cache()
    gc.collect()
    parser = argparse.ArgumentParser(description="Train model.")
    parser.add_argument(
        "--model",
        default="configs/model/mobilenetv3.yaml",
        type=str,
        help="model config",
    )
    parser.add_argument(
        "--data", default="configs/data/taco.yaml", type=str, help="data config"
    )
    
    parser.add_argument( "--loader", default ="PT" )
    
    args = parser.parse_args()
    loader_mode= args.loader
    
    model_config = read_yaml(cfg=args.model)
    data_config = read_yaml(cfg=args.data)

    data_config["DATA_PATH"] = os.environ.get("SM_CHANNEL_TRAIN", data_config["DATA_PATH"])

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    log_dir = os.environ.get("SM_MODEL_DIR", os.path.join("exp", 'latest'))

    if os.path.isfile(log_dir+'/best.pt'): 
        modified = datetime.fromtimestamp(os.path.getmtime(log_dir + '/best.pt'))
        new_log_dir = os.path.dirname(log_dir) + '/' + modified.strftime("%Y-%m-%d_%H-%M-%S")
        os.rename(log_dir, new_log_dir)

    os.makedirs(log_dir, exist_ok=True)
    
    
    test_loss, test_f1, test_acc = train(
        model_config=model_config,
        data_config=data_config,
        loader_mode = loader_mode,
        log_dir=log_dir,
        fp16=data_config["FP16"],
        device=device,
    )

