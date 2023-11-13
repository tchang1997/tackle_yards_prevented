from argparse import ArgumentParser
import os

import lightning.pytorch as pl
import lightning.pytorch.callbacks
from lightning.pytorch.loggers import TensorBoardLogger
from torch.utils.data import DataLoader
import yaml

from datasets import PlayByPlayDataset, COLLATE_FN_DICT
import dragonnet.models
from dragonnet.models import DragonNet, TransformerRepresentor
from trainer import BaseCounterfactualTackleTrainer
from utils import get_default_text_spinner_context

def get_callbacks(cfg):
    return [
        getattr(lightning.pytorch.callbacks, callback_dict["name"])(**callback_dict["params"])
        for callback_dict in cfg["callbacks"]
    ]

def get_config(cfg_path, default_config_path="./configs/defaults.yaml"):
    with open(default_config_path, "r") as f:
        cfg = yaml.safe_load(f)
    if cfg_path is not None:
        with open(cfg_path, "r") as f:
            run_cfg = yaml.safe_load(f)
        cfg.update(run_cfg)
        print("Using configuration file", cfg_path, f"(overriding {default_config_path})")
    print(yaml.dump(cfg, allow_unicode=True, default_flow_style=False))
    return cfg

def get_torch_model(model_settings):
    with get_default_text_spinner_context("Initializing model...") as spinner:
        dragonnet_model = DragonNet(
            input_dims=model_settings["input_size"],
            backbone_class=getattr(dragonnet.models, model_settings["representation_class"]),
            **model_settings.get("model_kwargs", {})
        )
        spinner.ok("✅ ")
    return dragonnet_model

def get_trainer_and_module(cfg):
    with get_default_text_spinner_context("Setting up trainer...") as spinner:
        logger = TensorBoardLogger(**cfg["logger"])
        trainer = pl.Trainer(
            logger=logger,
            callbacks=get_callbacks(cfg),
            **cfg["trainer_args"]
        )
        wrapped_module = BaseCounterfactualTackleTrainer(
            dragonnet_model,
            cfg["loss_hparams"],
            cfg["optimizer_settings"],
            scheduler_settings=cfg.get("scheduler_settings", None)
        )
        spinner.ok("✅ ")
    return logger, trainer, wrapped_module

def get_data_split(split, path, collate_fn, dataloader_settings):
    with get_default_text_spinner_context(f"Loading {split} split from {path}...") as spinner:
        dataset = PlayByPlayDataset(path)
        dataloader = DataLoader(
            dataset,
            collate_fn=collate_fn,
            **dataloader_settings,
        )
    return dataloader

def get_dataloaders(data_cfg, collate_fn, dataloader_settings):
    dataloaders = {}
    for split, path in data_cfg.items():
        dataloaders[split] = get_data_split(split, path, collate_fn, dataloader_settings)
        spinner.ok(f"✅ ({len(dataloaders.dataset)} examples)")
    return dataloaders

if __name__ == '__main__':
    psr = ArgumentParser()
    psr.add_argument("--config", type=str, help="Config YAML file with experimental settings.")
    args = psr.parse_args()

    cfg = get_config(args.config)
    dragonnet_model = get_torch_model(cfg["model_settings"])

    logger, trainer, wrapped_module = get_trainer_and_module(cfg)
    dataloaders = get_dataloaders(cfg["data"], COLLATE_FN_DICT[cfg["model_settings"]["representation_class"]], cfg["dataloader_settings"])

    try:
        trainer.fit(
            model=wrapped_module,
            train_dataloaders=dataloaders["train"],
            val_dataloaders=dataloaders["val"]
        )
    except Exception as e:
        import traceback

        print("Raised exception", e, "in `trainer.fit()`:")
        traceback.print_tb(e.__traceback__)
    finally:
        with get_default_text_spinner_context("Saving config...") as spinner:
            final_config_path = os.path.join(logger.log_dir, "config.yml")
            with open(final_config_path, "w") as f:
                yaml.dump(cfg, f, default_flow_style=False)
            spinner.ok(f"✅ (saved at {final_config_path})")
